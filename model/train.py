#!/usr/bin/env python3
"""DeepSpeed ZeRO Stage 2 training script for ALS-LM.

Trains a GPT-2 style decoder-only transformer on the tokenized ALS corpus
using DeepSpeed ZeRO Stage 2 with CPU offloading. Launched via the DeepSpeed
launcher::

    deepspeed model/train.py --deepspeed --deepspeed_config config/ds_zero2.json

The script follows the nanoGPT pattern for data loading (memory-mapped numpy
binary files with random batch sampling) while using DeepSpeed for optimizer
state management, mixed-precision training (fp16), gradient accumulation,
and checkpointing.

Key features:

- **Startup validation:** Checks train.bin, val.bin, meta.pkl exist and
  validates vocab_size matches the canonical tokenizer before training.
- **Epoch tracking:** EpochTracker class converts step counts into epoch
  numbers, epoch progress, and detects epoch boundary crossings.
- **Best checkpoint:** Saves best checkpoint (lowest val loss) to best/
  directory with both DeepSpeed directory and .pt export for downstream use.
- **Regular checkpoints:** Saves DeepSpeed directory checkpoints (for resume)
  at each checkpoint interval. No .pt export for regular checkpoints.
- **Checkpoint retention:** Keeps last 3 regular checkpoints + best (lowest
  val loss), deleting expired DeepSpeed directories.
- **Console + JSON logging:** Colored one-liner console output every 10
  steps with ETA, JSON lines to a fixed log file at logs/training_log.jsonl.
- **Enriched JSONL metrics:** epoch, perplexity, loss_scale, grad_norm,
  generalization gap, and gap ratio logged for post-training analysis.
- **Anomaly detection:** Loss spikes and high gradient norms trigger
  yellow console warnings for real-time monitoring.
- **Validation + sample generation:** Every 500 steps, estimates train/val
  loss and generates sample text from a fixed ALS prompt.
- **Loss sanity checks:** Stops immediately on NaN/Inf; warns if loss hasn't
  decreased after 100 steps.
- **Resume support:** Auto-detects existing checkpoints or accepts explicit
  ``--resume`` path. Appends to existing JSONL log with a resume marker.
- **Dry-run mode:** ``--dry-run`` validates config and prints training plan
  without GPU time.

Usage::

    # Tiny model (pipeline validation)
    deepspeed model/train.py --deepspeed --deepspeed_config config/ds_zero2.json --config tiny

    # Full training with epoch count
    deepspeed model/train.py --deepspeed --deepspeed_config config/ds_zero2.json --config 500M --max-epochs 3

    # Dry run (no training)
    deepspeed model/train.py --deepspeed --deepspeed_config config/ds_zero2.json --config tiny --dry-run

    # Resume from checkpoint
    deepspeed model/train.py --deepspeed --deepspeed_config config/ds_zero2.json --config tiny --resume checkpoints/tiny_20260224_143022/
"""

import argparse
import json
import math
import os
import pickle
import shutil
import sys
import time
from collections import deque
from datetime import datetime, timezone

# Ensure the project root is on sys.path so that `from model.model import ...`
# resolves correctly when DeepSpeed launches this script via
# `python model/train.py --local_rank=0 ...`.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

# DeepSpeed imports
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

# TensorBoard for real-time metrics visualization
from torch.utils.tensorboard import SummaryWriter

# Tokenizer for sample generation
from tokenizers import Tokenizer

# Model class
from model.model import GPT, MODEL_CONFIGS

# GPU memory monitoring
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default hyperparameters per model configuration
# ---------------------------------------------------------------------------
CONFIG_DEFAULTS = {
    "tiny": {"lr": 6e-4, "batch_size": 16, "grad_accum": 2, "warmup": 100, "max_steps": 2000},
    "medium": {"lr": 3e-4, "batch_size": 8, "grad_accum": 4, "warmup": 200, "max_steps": 10000},
    "500M": {"lr": 3e-4, "batch_size": 4, "grad_accum": 8, "warmup": 500, "max_steps": 50000},
}

# Fixed prompts for sample text generation at validation intervals
# Covers 5 domains: clinical-declarative, clinical-question, molecular,
# treatment, and epidemiological (per CONTEXT.md diversity requirement)
SAMPLE_PROMPTS = [
    "Amyotrophic lateral sclerosis is",                          # Clinical - declarative
    "What are the early symptoms of ALS?",                       # Clinical - question
    "The SOD1 gene mutation in ALS leads to",                    # Molecular - declarative
    "Riluzole works by",                                         # Treatment - declarative
    "The incidence of ALS worldwide is approximately",           # Epidemiological - declarative
]

SAMPLE_TEMPERATURES = [0.0, 0.7]  # Greedy + sampled decoding


# ---------------------------------------------------------------------------
# Epoch tracking
# ---------------------------------------------------------------------------
class EpochTracker:
    """Tracks epoch boundaries and per-epoch metrics during training."""

    def __init__(self, train_tokens: int, tokens_per_step: int):
        self.train_tokens = train_tokens
        self.tokens_per_step = tokens_per_step
        self.steps_per_epoch = train_tokens // tokens_per_step
        self.current_epoch = 0
        self.epoch_losses = []
        self._total_epochs = 0
        self._completed_epoch_avg_loss = 0.0

    def update(self, step: int, loss: float) -> bool:
        """Update tracker. Returns True if epoch boundary crossed."""
        new_epoch = step // self.steps_per_epoch if self.steps_per_epoch > 0 else 0
        self.epoch_losses.append(loss)
        if new_epoch > self.current_epoch:
            # Save the completed epoch's average before clearing
            self._completed_epoch_avg_loss = self.avg_epoch_loss
            self.current_epoch = new_epoch
            self.epoch_losses = []
            return True
        return False

    @property
    def epoch(self) -> int:
        return self.current_epoch

    def epoch_progress(self, step: int) -> float:
        if self.steps_per_epoch == 0:
            return 0.0
        return (step % self.steps_per_epoch) / self.steps_per_epoch

    @property
    def avg_epoch_loss(self) -> float:
        if not self.epoch_losses:
            return 0.0
        return sum(self.epoch_losses) / len(self.epoch_losses)

    @property
    def completed_epoch_avg_loss(self) -> float:
        """Average loss of the most recently completed epoch.

        Only valid immediately after update() returns True. Captures the
        epoch average before epoch_losses is cleared for the new epoch.
        """
        return self._completed_epoch_avg_loss

    @property
    def total_epochs(self) -> int:
        """Total epochs based on max_steps (set after init)."""
        return self._total_epochs

    @total_epochs.setter
    def total_epochs(self, value: int):
        self._total_epochs = value


# ---------------------------------------------------------------------------
# ANSI color codes for console output
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="Train ALS-LM with DeepSpeed ZeRO Stage 2"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model config name (default: tiny)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override (default: per-config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Micro batch size per GPU (default: per-config)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: per-config)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="LR warmup steps (default: per-config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Total training steps (default: per-config)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides --max-steps when set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Steps between validation (default: 500)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Steps between checkpoint saves (default: 1000)",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=200,
        help="Batches to average for validation loss (default: 200)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between console/JSON log entries (default: 10)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tokenized",
        help="Path to tokenized data directory (default: data/tokenized)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer/als_tokenizer.json",
        help="Path to tokenizer JSON for sample generation (default: tokenizer/als_tokenizer.json)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Explicit checkpoint path for resume (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print training plan without training",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce GPU memory (trades compute for memory)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by DeepSpeed launcher)",
    )

    # Add DeepSpeed config arguments (--deepspeed, --deepspeed_config)
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # Apply per-config defaults for unset arguments
    defaults = CONFIG_DEFAULTS.get(args.config, CONFIG_DEFAULTS["tiny"])
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.grad_accum is None:
        args.grad_accum = defaults["grad_accum"]
    if args.warmup_steps is None:
        args.warmup_steps = defaults["warmup"]
    if args.max_steps is None:
        args.max_steps = defaults["max_steps"]

    return args


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def validate_data_files(data_dir):
    """Check that train.bin, val.bin, and meta.pkl exist. Fail fast if missing."""
    required = ["train.bin", "val.bin", "meta.pkl"]
    missing = []
    for fname in required:
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            missing.append(fname)

    if missing:
        print(f"\nFATAL: Missing data files in {data_dir}:")
        for m in missing:
            print(f"  - {m}")
        print(
            f"\nPlease prepare the tokenized data first:\n"
            f"  python scripts/prepare_data.py --encode\n"
        )
        sys.exit(1)


def load_and_validate_vocab(data_dir, tokenizer_path):
    """Load vocab_size from meta.pkl and validate against the canonical tokenizer.

    Returns the vocab_size from meta.pkl if validation passes.
    """
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]

    # Load the canonical tokenizer and compare
    if not os.path.isfile(tokenizer_path):
        print(f"\nWARNING: Tokenizer not found at {tokenizer_path}")
        print("  Cannot validate vocab_size. Proceeding with meta.pkl value.")
        return meta_vocab_size

    tok = Tokenizer.from_file(tokenizer_path)
    tok_vocab_size = tok.get_vocab_size()

    if meta_vocab_size != tok_vocab_size:
        print(f"\nFATAL: Vocab size mismatch!")
        print(f"  meta.pkl vocab_size:  {meta_vocab_size:,}")
        print(f"  tokenizer vocab_size: {tok_vocab_size:,}")
        print(
            f"\nThe tokenized data does not match the canonical tokenizer."
            f"\nRe-run: python scripts/prepare_data.py --encode "
            f"--tokenizer {tokenizer_path}\n"
        )
        sys.exit(1)

    return meta_vocab_size


# ---------------------------------------------------------------------------
# Data loading (nanoGPT memmap pattern)
# ---------------------------------------------------------------------------
def get_batch(split, batch_size, block_size, device, data_dir):
    """Load a random batch from memory-mapped binary token files.

    Recreates the numpy memmap on every call to prevent memory leaks from
    numpy reference counting. This is the correct nanoGPT pattern and costs
    negligible overhead since memmap is just a file descriptor.
    """
    fname = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
        device, non_blocking=True
    )
    return x, y


# ---------------------------------------------------------------------------
# Validation loss estimation
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model_engine, eval_iters, batch_size, block_size, device, data_dir):
    """Estimate train and validation loss over multiple batches.

    Uses model_engine.module (unwrapped model) in eval mode to avoid
    DeepSpeed's training-mode forward pass with gradient tracking.
    """
    model_engine.module.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, batch_size, block_size, device, data_dir)
            logits, loss = model_engine.module(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model_engine.module.train()
    return out


# ---------------------------------------------------------------------------
# Sample text generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_sample(model, tokenizer, prompt, max_new_tokens=128, temperature=0.8, device="cuda"):
    """Generate sample text from a fixed prompt for qualitative evaluation.

    Uses the unwrapped model (model_engine.module) in eval mode to avoid
    DeepSpeed training overhead during generation. Supports both greedy
    (temperature=0) and sampled decoding.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature < 1e-8:
            # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Sampled decoding
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, idx_next], dim=1)
    model.train()
    return tokenizer.decode(idx[0].tolist())


# ---------------------------------------------------------------------------
# GPU memory monitoring
# ---------------------------------------------------------------------------
def init_gpu_monitor(local_rank):
    """Initialize pynvml for GPU memory monitoring.

    Returns a GPU device handle, or None if pynvml is not available.
    """
    if not PYNVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        device_index = max(0, local_rank)
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return handle
    except Exception:
        return None


def get_gpu_memory_mb(handle):
    """Return current GPU memory usage in MB, or 0 if unavailable."""
    if handle is None:
        return 0
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used // (1024 * 1024)
    except Exception:
        return 0


def get_gpu_name(handle):
    """Return the GPU name string, or 'Unknown' if unavailable."""
    if handle is None:
        return "Unknown"
    try:
        return pynvml.nvmlDeviceGetName(handle)
    except Exception:
        return "Unknown"


def get_gpu_total_memory_mb(handle):
    """Return total GPU memory in MB, or 0 if unavailable."""
    if handle is None:
        return 0
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.total // (1024 * 1024)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Checkpoint saving (atomic pattern: temp-dir-then-rename)
# ---------------------------------------------------------------------------
def save_checkpoint_atomic(model_engine, run_dir, step, client_state, checkpoint_meta, config_name):
    """Save a regular DeepSpeed checkpoint atomically (no .pt export).

    Writes to a temporary directory, then atomically renames to the final
    path. If interrupted mid-write, only the temp directory is affected;
    existing checkpoints remain intact.

    Returns the save duration in seconds.
    """
    tag = f"step_{step}"
    final_dir = os.path.join(run_dir, tag)
    temp_run_dir = os.path.join(run_dir, f".tmp_run_step_{step}")

    # Clean up any leftover temp dir from a previous failed save
    if os.path.exists(temp_run_dir):
        shutil.rmtree(temp_run_dir)

    t0 = time.time()
    try:
        os.makedirs(temp_run_dir, exist_ok=True)

        # 1. DeepSpeed saves to temp run directory (creates temp_run_dir/tag/)
        model_engine.save_checkpoint(temp_run_dir, tag=tag, client_state=client_state)

        # 2. Write enriched checkpoint_meta.json into the DeepSpeed tag subdir
        meta_path = os.path.join(temp_run_dir, tag, "checkpoint_meta.json")
        with open(meta_path, "w") as f:
            json.dump(checkpoint_meta, f, indent=2)

        # 3. Rename into place. Note: the rmtree+rename pair is not truly
        # atomic -- a kill between the two loses both old and new. This is
        # acceptable because retention keeps 3 checkpoints as a buffer.
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(os.path.join(temp_run_dir, tag), final_dir)

        # 4. Update run_dir/latest file to point to the new tag
        latest_path = os.path.join(run_dir, "latest")
        with open(latest_path, "w") as f:
            f.write(tag)

        # 5. Clean up the temp run directory shell
        shutil.rmtree(temp_run_dir, ignore_errors=True)

        save_duration = time.time() - t0
        return save_duration

    except Exception:
        # Clean up temp on failure; existing checkpoints untouched
        if os.path.exists(temp_run_dir):
            shutil.rmtree(temp_run_dir, ignore_errors=True)
        raise


def save_best_checkpoint_atomic(model_engine, run_dir, step, val_loss,
                                model_config_dict, config_name, epoch,
                                wall_clock_elapsed):
    """Save the best checkpoint to best/ with atomic rename and .pt export.

    The best checkpoint includes both the DeepSpeed directory (for resume)
    and a raw .pt file (for export). Only the best checkpoint gets a .pt
    export to save ~2 GB per regular checkpoint.

    Returns the save duration in seconds.
    """
    best_dir = os.path.join(run_dir, "best")
    temp_best_run = os.path.join(run_dir, ".tmp_best")
    tag = f"step_{step}"

    # Clean up any leftover temp dir
    if os.path.exists(temp_best_run):
        shutil.rmtree(temp_best_run)

    t0 = time.time()
    try:
        os.makedirs(temp_best_run, exist_ok=True)

        # 1. DeepSpeed saves to temp directory
        client_state = {"step": step, "val_loss": val_loss,
                        "best_val_loss": val_loss, "best_val_step": step}
        model_engine.save_checkpoint(temp_best_run, tag=tag, client_state=client_state)

        # 2. Extract fp32 state dict and save as best.pt (only for best)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(temp_best_run, tag=tag)
        pt_checkpoint = {
            "model": state_dict,
            "config": model_config_dict,
            "step": step,
            "val_loss": val_loss,
        }
        torch.save(pt_checkpoint, os.path.join(temp_best_run, tag, "best.pt"))

        # 3. Write enriched checkpoint_meta.json with is_best=True
        meta = {
            "step": step,
            "val_loss": val_loss,
            "is_best": True,
            "epoch": epoch,
            "wall_clock_elapsed": wall_clock_elapsed,
            "config_name": config_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = os.path.join(temp_best_run, tag, "checkpoint_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 4. Replace best/. Note: rmtree+rename is not truly atomic -- a
        # kill between the two loses the old best. The regular checkpoint
        # rotation provides recovery if this narrow window is hit.
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.rename(os.path.join(temp_best_run, tag), best_dir)

        # 5. Clean up temp shell
        shutil.rmtree(temp_best_run, ignore_errors=True)

        save_duration = time.time() - t0
        return save_duration

    except Exception:
        # Clean up temp on failure
        if os.path.exists(temp_best_run):
            shutil.rmtree(temp_best_run, ignore_errors=True)
        raise


def cleanup_checkpoints(run_dir, keep_last=3, log_file=None):
    """Retain last N regular checkpoints + best/ + emergency, delete the rest.

    Only deletes regular step_XXXX/ DeepSpeed directories. Skips best/,
    emergency_*, and .tmp_* directories. Regular checkpoints no longer
    have .pt files (only best/ gets a .pt export).

    If log_file is provided, logs individual checkpoint_delete events to JSONL.
    """
    # Find all regular checkpoint step directories (skip best, emergency, tmp)
    steps = []
    for entry in os.listdir(run_dir):
        entry_path = os.path.join(run_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry.startswith("best") or entry.startswith("emergency_") or entry.startswith(".tmp"):
            continue
        if entry.startswith("step_"):
            try:
                step_num = int(entry.split("_")[1])
                steps.append(step_num)
            except (ValueError, IndexError):
                continue
    steps.sort()

    if len(steps) <= keep_last:
        return  # Nothing to clean up

    # Keep last N regular checkpoints (best/ is separate and always kept)
    keep_set = set(steps[-keep_last:])

    # Delete expired regular checkpoints (DeepSpeed directories only)
    for step_num in steps:
        if step_num not in keep_set:
            ds_dir = os.path.join(run_dir, f"step_{step_num}")
            if os.path.isdir(ds_dir):
                shutil.rmtree(ds_dir)
                print(f"  Deleted checkpoint: {ds_dir}")
                if log_file is not None:
                    log_checkpoint_event(log_file, "checkpoint_delete", step_num, {
                        "path": f"step_{step_num}",
                        "reason": "retention_policy",
                    })


# ---------------------------------------------------------------------------
# Emergency shutdown (NaN/Inf loss detection)
# ---------------------------------------------------------------------------
def emergency_shutdown(
    model_engine, step, loss_value, run_dir, log_file, tb_writer,
    loss_history, grad_norm_history, loss_scale_history,
    epoch_tracker, training_start_time, args, gpu_handle, optimizer,
):
    """Save emergency checkpoint, write diagnostics, print banner, and exit.

    Called when NaN or Inf is detected in the loss BEFORE backward pass,
    preserving clean optimizer state. Exits with code 2 (distinct from
    general errors which use code 1).
    """
    # Step A: Save emergency DeepSpeed checkpoint (non-atomic, diagnostics only)
    emergency_dir = os.path.join(run_dir, f"emergency_step_{step}")
    try:
        model_engine.save_checkpoint(
            run_dir, tag=f"emergency_step_{step}",
            client_state={"step": step, "loss_value": str(loss_value)},
        )
        print(f"  Emergency checkpoint saved: {emergency_dir}")
    except Exception as e:
        print(f"  {YELLOW}WARNING: Emergency checkpoint save failed: {e}{RESET}")
        print(f"  (Model state may contain NaN -- continuing with diagnostics)")

    # Step B: Write emergency_diagnostics.json alongside the emergency checkpoint
    diagnostics = {
        "type": "emergency",
        "step": step,
        "epoch": epoch_tracker.epoch,
        "epoch_progress": epoch_tracker.epoch_progress(step),
        "loss_value": str(loss_value),
        "wall_clock_sec": time.time() - training_start_time,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "loss_history": [float(v) for v in loss_history],
        "grad_norm_history": [float(v) for v in grad_norm_history],
        "loss_scale_history": [float(v) for v in loss_scale_history],
        "current_lr": optimizer.param_groups[0]["lr"],
        "gpu_memory_mb": get_gpu_memory_mb(gpu_handle),
        "gpu_total_mb": get_gpu_total_memory_mb(gpu_handle),
        "training_config": {
            "config_name": args.config,
            "max_steps": args.max_steps if hasattr(args, "max_steps") else None,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "seed": args.seed,
            "eval_interval": args.eval_interval,
            "checkpoint_interval": args.checkpoint_interval,
        },
    }
    diagnostics_path = os.path.join(run_dir, f"emergency_diagnostics_step_{step}.json")
    try:
        with open(diagnostics_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
    except Exception as e:
        print(f"  {YELLOW}WARNING: Failed to write diagnostics: {e}{RESET}")
        diagnostics_path = "(write failed)"

    # Step C: Log emergency event to JSONL
    try:
        log_checkpoint_event(log_file, "emergency", step, {
            "epoch": epoch_tracker.epoch,
            "loss_value": str(loss_value),
        })
    except Exception:
        pass  # Best effort -- file may already be in bad state

    # Step D: Find last good regular checkpoint
    last_good = None
    last_good_path = None
    try:
        for dir_entry in sorted(os.listdir(run_dir), reverse=True):
            if (
                dir_entry.startswith("step_")
                and not dir_entry.startswith("emergency_")
                and os.path.isdir(os.path.join(run_dir, dir_entry))
            ):
                last_good = dir_entry
                last_good_path = os.path.join(run_dir, dir_entry)
                break
    except OSError:
        pass

    # Step E: Print bold multi-line emergency banner
    print(f"\n{RED}{BOLD}{'=' * 60}")
    print(f"  EMERGENCY SHUTDOWN: NaN/Inf loss detected")
    print(f"{'=' * 60}{RESET}")
    print(f"  {RED}Step:              {step}{RESET}")
    print(f"  {RED}Loss value:        {loss_value}{RESET}")
    print(f"  {RED}Emergency ckpt:    {emergency_dir}{RESET}")
    print(f"  {RED}Last good ckpt:    {last_good_path if last_good else 'None'}{RESET}")
    print(f"  {RED}Diagnostics:       {diagnostics_path}{RESET}")
    resume_path = last_good_path if last_good else run_dir
    print(f"\n  {BOLD}To resume from last good checkpoint:{RESET}")
    print(f"  deepspeed model/train.py --deepspeed --deepspeed_config config/ds_zero2.json --config {args.config} --resume {resume_path}")
    print(f"\n{RED}{BOLD}{'=' * 60}{RESET}\n")

    # Step F: Close resources and exit
    log_file.close()
    tb_writer.close()
    sys.exit(2)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def try_resume(model_engine, args, run_dir):
    """Attempt to resume training from an existing checkpoint.

    If --resume is provided, uses that path. Otherwise, auto-detects the
    latest checkpoint in the run directory.

    Returns a dict with start_step, best_val_loss, and best_val_step
    (defaults for fresh start if no checkpoint found).
    """
    fresh = {"start_step": 0, "best_val_loss": float("inf"), "best_val_step": 0}
    ckpt_dir = args.resume if args.resume else run_dir

    if not os.path.isdir(ckpt_dir):
        return fresh

    # Check if there are any checkpoint subdirectories
    has_checkpoints = any(
        entry.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, entry))
        for entry in os.listdir(ckpt_dir)
    )
    if not has_checkpoints:
        return fresh

    try:
        load_path, client_state = model_engine.load_checkpoint(
            ckpt_dir,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        if load_path is not None and client_state is not None:
            start_step = client_state.get("step", 0) + 1
            last_loss = client_state.get("train_loss", "N/A")
            print(f"\n  Resumed from: {load_path}")
            print(f"  Starting at step: {start_step}")
            print(f"  Last loss: {last_loss}")
            return {
                "start_step": start_step,
                "best_val_loss": client_state.get("best_val_loss", float("inf")),
                "best_val_step": client_state.get("best_val_step", 0),
            }
    except Exception as e:
        print(f"\n  WARNING: Failed to resume from {ckpt_dir}: {e}")
        print("  Starting training from scratch.")

    return fresh


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(is_resuming=False):
    """Open the fixed JSONL log file, appending if it exists on resume."""
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/training_log.jsonl"
    if is_resuming and os.path.exists(log_path):
        log_file = open(log_path, "a")
    else:
        log_file = open(log_path, "w")
    return log_path, log_file


def write_config_header(log_file, args, model_config_dict, ds_config, vocab_size, train_tokens, val_tokens):
    """Write full config dump as the first JSON line in the log file."""
    header = {
        "type": "config",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_config": model_config_dict,
        "deepspeed_config": ds_config,
        "training_args": {
            "config": args.config,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "warmup_steps": args.warmup_steps,
            "max_steps": args.max_steps,
            "max_epochs": getattr(args, "max_epochs", None),
            "seed": args.seed,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "log_interval": args.log_interval,
            "data_dir": args.data_dir,
            "tokenizer_path": args.tokenizer_path,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "data_info": {
            "vocab_size": vocab_size,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
        },
    }
    log_file.write(json.dumps(header) + "\n")
    log_file.flush()


def log_step(log_file, step, loss, lr, tokens_per_sec, gpu_mem_mb, dt_sec,
             epoch, epoch_progress, total_tokens, loss_scale, grad_norm):
    """Write a training step entry to the JSONL log file."""
    try:
        perplexity = min(math.exp(loss), 1e5)
    except OverflowError:
        perplexity = 1e5
    entry = {
        "type": "step",
        "step": step,
        "loss": round(loss, 6),
        "perplexity": round(perplexity, 2),
        "lr": lr,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "gpu_mem_mb": gpu_mem_mb,
        "dt_sec": round(dt_sec, 3),
        "epoch": epoch,
        "epoch_progress": round(epoch_progress, 4),
        "total_tokens": total_tokens,
        "loss_scale": loss_scale,
        "grad_norm": round(grad_norm, 4) if grad_norm is not None else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


def log_validation(log_file, step, train_loss, val_loss, samples,
                   epoch, epoch_progress):
    """Write a validation entry to the JSONL log file."""
    try:
        train_ppl = min(math.exp(train_loss), 1e5)
    except OverflowError:
        train_ppl = 1e5
    try:
        val_ppl = min(math.exp(val_loss), 1e5)
    except OverflowError:
        val_ppl = 1e5
    gap = val_loss - train_loss
    gap_ratio = (val_loss - train_loss) / train_loss if train_loss > 0 else 0.0

    entry = {
        "type": "validation",
        "step": step,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
        "train_perplexity": round(train_ppl, 2),
        "val_perplexity": round(val_ppl, 2),
        "generalization_gap": round(gap, 6),
        "gap_ratio": round(gap_ratio, 4),
        "epoch": epoch,
        "epoch_progress": round(epoch_progress, 4),
        "samples": samples,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


def log_epoch(log_file, epoch, steps_in_epoch, avg_train_loss, total_tokens, elapsed_sec):
    """Write an epoch boundary entry to the JSONL log file."""
    try:
        avg_ppl = min(math.exp(avg_train_loss), 1e5)
    except OverflowError:
        avg_ppl = 1e5
    entry = {
        "type": "epoch",
        "epoch": epoch,
        "steps": steps_in_epoch,
        "avg_train_loss": round(avg_train_loss, 6),
        "avg_perplexity": round(avg_ppl, 2),
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed_sec, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


def log_resume(log_file, step):
    """Write a resume marker entry to the JSONL log file."""
    entry = {
        "type": "resume",
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


def log_checkpoint_event(log_file, event_type, step, details):
    """Log checkpoint-related events to JSONL.

    event_type: 'checkpoint_save', 'checkpoint_delete', 'best_update', 'emergency'
    """
    entry = {
        "type": event_type,
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


# ---------------------------------------------------------------------------
# DeepSpeed config resolution
# ---------------------------------------------------------------------------
def resolve_ds_config(ds_config_path, args):
    """Load the DeepSpeed JSON config and override 'auto' values programmatically.

    Returns the resolved config dict ready for deepspeed.initialize().
    """
    with open(ds_config_path) as f:
        ds_config = json.load(f)

    # Override auto values
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_accumulation_steps"] = args.grad_accum
    ds_config["train_batch_size"] = args.batch_size * args.grad_accum

    # Optimizer LR
    ds_config["optimizer"]["params"]["lr"] = args.lr

    # Scheduler steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = args.warmup_steps
    ds_config["scheduler"]["params"]["total_num_steps"] = args.max_steps

    return ds_config


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------
def print_dry_run(args, model_config, vocab_size, train_tokens, val_tokens, ds_config, param_count):
    """Print the training plan without training."""
    block_size = model_config.block_size
    effective_batch = args.batch_size * args.grad_accum
    tokens_per_step = effective_batch * block_size
    total_tokens = tokens_per_step * args.max_steps
    epochs_approx = total_tokens / train_tokens if train_tokens > 0 else 0

    print("\n" + "=" * 60)
    print("  DRY RUN - Training Plan")
    print("=" * 60)

    print(f"\n  Model Configuration ({args.config})")
    print(f"    Layers:      {model_config.n_layer}")
    print(f"    Heads:       {model_config.n_head}")
    print(f"    Embed dim:   {model_config.n_embd}")
    print(f"    Block size:  {model_config.block_size}")
    print(f"    Vocab size:  {vocab_size:,}")
    print(f"    Parameters:  {param_count:,}")

    print(f"\n  DeepSpeed Configuration")
    print(f"    ZeRO stage:  {ds_config['zero_optimization']['stage']}")
    print(f"    FP16:        {ds_config['fp16']['enabled']}")
    print(f"    CPU offload: {ds_config['zero_optimization']['offload_optimizer']['device']}")
    print(f"    Optimizer:   {ds_config['optimizer']['type']}")
    print(f"    Scheduler:   {ds_config['scheduler']['type']}")

    print(f"\n  Training Plan")
    print(f"    Total steps:     {args.max_steps:,}")
    print(f"    Micro batch:     {args.batch_size}")
    print(f"    Grad accum:      {args.grad_accum}")
    print(f"    Effective batch: {effective_batch}")
    print(f"    Learning rate:   {args.lr:.1e}")
    print(f"    Warmup steps:    {args.warmup_steps}")
    print(f"    Eval interval:   {args.eval_interval}")
    print(f"    Checkpoint:      every {args.checkpoint_interval} steps")
    print(f"    Seed:            {args.seed}")
    print(f"    Gradient checkpointing: {args.gradient_checkpointing}")

    print(f"\n  Data Info")
    print(f"    Data directory:  {args.data_dir}")
    print(f"    Train tokens:    {train_tokens:,}")
    print(f"    Val tokens:      {val_tokens:,}")
    print(f"    Tokens/step:     {tokens_per_step:,}")
    print(f"    Total tokens:    {total_tokens:,}")
    print(f"    Est. epochs:     {epochs_approx:.1f}")

    tb_cfg = ds_config.get("tensorboard", {})
    print(f"\n  Logging")
    print(f"    JSONL log:       logs/training_log.jsonl")
    print(f"    TensorBoard:     {tb_cfg.get('output_path', 'N/A')} (enabled={tb_cfg.get('enabled', False)})")
    print(f"    Sample prompts:  {len(SAMPLE_PROMPTS)} x {len(SAMPLE_TEMPERATURES)} temperatures")

    print("\n" + "=" * 60)
    print("  Dry run complete. Remove --dry-run to start training.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------
def format_eta(remaining_seconds):
    """Format remaining seconds as a human-readable ETA string."""
    if remaining_seconds <= 0:
        return "done"
    minutes, seconds = divmod(int(remaining_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    elif minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    else:
        return f"{seconds}s"


def format_time(elapsed_seconds):
    """Format elapsed seconds as a human-readable time string."""
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{minutes}m {seconds}s"


def print_config_summary(args, model_config, param_count, vocab_size,
                         train_tokens, val_tokens, ds_config,
                         tokens_per_step, epoch_tracker, gpu_handle):
    """Print a full config summary at training startup."""
    effective_batch = args.batch_size * args.grad_accum

    print(f"\n{BOLD}{'=' * 64}{RESET}")
    print(f"{BOLD}  ALS-LM Training Configuration{RESET}")
    print(f"{BOLD}{'=' * 64}{RESET}")

    print(f"\n  {BOLD}Model{RESET}")
    print(f"    Config:      {args.config}")
    print(f"    Parameters:  {param_count:,}")
    print(f"    Layers:      {model_config.n_layer}")
    print(f"    Heads:       {model_config.n_head}")
    print(f"    Embed dim:   {model_config.n_embd}")
    print(f"    Block size:  {model_config.block_size}")
    print(f"    Vocab size:  {vocab_size:,}")
    print(f"    Dropout:     {model_config.dropout}")

    print(f"\n  {BOLD}Data{RESET}")
    print(f"    Train tokens:  {train_tokens:,}")
    print(f"    Val tokens:    {val_tokens:,}")
    print(f"    Data dir:      {args.data_dir}")

    total_epochs = epoch_tracker.total_epochs if epoch_tracker else 0
    steps_per_epoch = epoch_tracker.steps_per_epoch if epoch_tracker else 0
    print(f"\n  {BOLD}Training{RESET}")
    print(f"    Total steps:     {args.max_steps:,}")
    print(f"    Epochs:          {total_epochs}")
    print(f"    Steps/epoch:     {steps_per_epoch:,}")
    print(f"    Effective batch: {effective_batch} (micro={args.batch_size} x accum={args.grad_accum})")
    print(f"    Tokens/step:     {tokens_per_step:,}")
    print(f"    Learning rate:   {args.lr:.1e}")
    print(f"    Warmup steps:    {args.warmup_steps}")
    print(f"    Eval interval:   {args.eval_interval}")
    print(f"    Checkpoint:      every {args.checkpoint_interval} steps")
    print(f"    Log interval:    {args.log_interval}")
    print(f"    Seed:            {args.seed}")

    zero_stage = ds_config.get("zero_optimization", {}).get("stage", "N/A")
    fp16_enabled = ds_config.get("fp16", {}).get("enabled", False)
    offload_device = ds_config.get("zero_optimization", {}).get("offload_optimizer", {}).get("device", "none")
    grad_clip = ds_config.get("gradient_clipping", "N/A")
    print(f"\n  {BOLD}DeepSpeed{RESET}")
    print(f"    ZeRO stage:   {zero_stage}")
    print(f"    FP16:         {fp16_enabled}")
    print(f"    CPU offload:  {offload_device}")
    print(f"    Grad clip:    {grad_clip}")

    gpu_name = get_gpu_name(gpu_handle)
    gpu_mem = get_gpu_total_memory_mb(gpu_handle)
    print(f"\n  {BOLD}Device{RESET}")
    print(f"    GPU:          {gpu_name}")
    print(f"    VRAM:         {gpu_mem:,} MB")

    tb_enabled = ds_config.get("tensorboard", {}).get("enabled", False)
    print(f"\n  {BOLD}Logging{RESET}")
    print(f"    JSONL:        logs/training_log.jsonl")
    print(f"    TensorBoard:  logs/tensorboard/ (enabled={tb_enabled})")
    print(f"    Samples:      {len(SAMPLE_PROMPTS)} prompts x {len(SAMPLE_TEMPERATURES)} temperatures per validation")

    print(f"\n{BOLD}{'=' * 64}{RESET}\n")


def print_training_summary(
    config_name,
    total_steps,
    elapsed_time,
    final_train_loss,
    final_val_loss,
    best_val_loss,
    best_val_step,
    total_tokens_processed,
    run_dir,
    epoch_tracker=None,
    total_checkpoints_saved=0,
    total_best_updates=0,
    total_bytes_written=0,
):
    """Print the end-of-run training summary with checkpoint statistics."""
    time_str = format_time(elapsed_time)
    avg_throughput = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0

    try:
        final_ppl = min(math.exp(final_train_loss), 1e5)
    except OverflowError:
        final_ppl = 1e5

    epochs_completed = epoch_tracker.epoch if epoch_tracker else 0

    # Best checkpoint is always in the fixed best/ directory
    best_ckpt_path = os.path.join(run_dir, "best")

    best_epoch = best_val_step // epoch_tracker.steps_per_epoch if epoch_tracker and epoch_tracker.steps_per_epoch > 0 else 0

    print(f"\n{BOLD}{'=' * 64}{RESET}")
    print(f"{BOLD}  Training Complete{RESET}")
    print(f"{BOLD}{'=' * 64}{RESET}")
    print(f"  Config:             {config_name}")
    print(f"  Total steps:        {total_steps:,}")
    print(f"  Total time:         {time_str}")
    print(f"  Epochs completed:   {epochs_completed}")
    print(f"  Final train loss:   {final_train_loss:.4f}")
    print(f"  Final perplexity:   {final_ppl:.2f}")
    print(f"  Final val loss:     {final_val_loss:.4f}")
    print(f"  Best val loss:      {best_val_loss:.4f} (step {best_val_step}, epoch {best_epoch})")
    print(f"  Total tokens:       {total_tokens_processed:,}")
    print(f"  Avg throughput:     {avg_throughput:,.0f} tokens/sec")
    print(f"  Best checkpoint:    {best_ckpt_path}")
    print(f"  Checkpoint dir:     {run_dir}")
    print(f"\n  {BOLD}Checkpoint Statistics:{RESET}")
    print(f"    Total checkpoints saved:  {total_checkpoints_saved}")
    print(f"    Best checkpoint updates:  {total_best_updates}")
    print(f"    Total disk written:       {total_bytes_written / 1024 / 1024 / 1024:.2f} GB")
    print(f"    Best checkpoint:          step {best_val_step} (val loss: {best_val_loss:.4f})")
    print(f"{BOLD}{'=' * 64}{RESET}\n")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ---- Startup validation ------------------------------------------------
    print("\n=== ALS-LM Training Script ===\n")

    validate_data_files(args.data_dir)
    vocab_size = load_and_validate_vocab(args.data_dir, args.tokenizer_path)
    print(f"  Vocab size: {vocab_size:,} (validated against tokenizer)")

    # Get data file sizes (token counts)
    train_data = np.memmap(
        os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r"
    )
    train_tokens = len(train_data)
    val_tokens = len(val_data)
    print(f"  Train tokens: {train_tokens:,}")
    print(f"  Val tokens:   {val_tokens:,}")

    # ---- Compute tokens_per_step and handle --max-epochs -------------------
    effective_batch = args.batch_size * args.grad_accum
    block_size = MODEL_CONFIGS[args.config].block_size
    tokens_per_step = effective_batch * block_size

    if args.max_epochs is not None:
        steps_per_epoch = train_tokens // tokens_per_step
        args.max_steps = args.max_epochs * steps_per_epoch
        print(f"\n  --max-epochs {args.max_epochs}:")
        print(f"    steps_per_epoch = {train_tokens:,} // {tokens_per_step:,} = {steps_per_epoch:,}")
        print(f"    max_steps = {args.max_epochs} * {steps_per_epoch:,} = {args.max_steps:,}")

    # Instantiate EpochTracker
    epoch_tracker = EpochTracker(train_tokens, tokens_per_step)
    epoch_tracker.total_epochs = args.max_steps // epoch_tracker.steps_per_epoch if epoch_tracker.steps_per_epoch > 0 else 0

    # ---- Resolve DeepSpeed config ------------------------------------------
    ds_config_path = "config/ds_zero2.json"
    if not os.path.isfile(ds_config_path):
        print(f"\nFATAL: DeepSpeed config not found at {ds_config_path}")
        sys.exit(1)

    ds_config = resolve_ds_config(ds_config_path, args)

    # ---- Build model -------------------------------------------------------
    model = GPT.from_config(args.config, vocab_size=vocab_size)
    model_config = model.config
    param_count = model.get_num_params()
    print(f"  Model: {args.config} ({param_count:,} parameters)")

    # Enable gradient checkpointing if requested (must happen before DeepSpeed wraps the model)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print(f"  Gradient checkpointing enabled on {len(list(model.transformer.h))} blocks")

    # Serialize model config for checkpoint metadata
    model_config_dict = {
        "block_size": model_config.block_size,
        "vocab_size": model_config.vocab_size,
        "n_layer": model_config.n_layer,
        "n_head": model_config.n_head,
        "n_embd": model_config.n_embd,
        "dropout": model_config.dropout,
        "bias": model_config.bias,
    }

    # ---- Dry-run mode ------------------------------------------------------
    if args.dry_run:
        print_dry_run(
            args, model_config, vocab_size, train_tokens, val_tokens, ds_config, param_count
        )
        return

    # ---- Seed and reproducibility ------------------------------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f"  Seed: {args.seed}")

    # ---- DeepSpeed initialization ------------------------------------------
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    device = model_engine.device
    print(f"  Device: {device}")

    # ---- TensorBoard writer for custom metrics ------------------------------
    # Separate directory from DeepSpeed auto-logging (logs/tensorboard/als_lm_training/)
    # to avoid event file conflicts (RESEARCH.md Pitfall 3)
    os.makedirs("logs/tensorboard/custom", exist_ok=True)
    tb_writer = SummaryWriter(log_dir="logs/tensorboard/custom")
    print(f"  TensorBoard: logs/tensorboard/")

    # ---- GPU memory monitor ------------------------------------------------
    gpu_handle = init_gpu_monitor(args.local_rank)

    # ---- Checkpoint directory setup ----------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"{args.config}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Checkpoint dir: {run_dir}")

    # Clean up any orphaned temp directories from previous interrupted saves
    for entry in os.listdir(run_dir):
        if entry.startswith(".tmp_"):
            tmp_path = os.path.join(run_dir, entry)
            if os.path.isdir(tmp_path):
                print(f"  {YELLOW}WARNING: Cleaned up orphaned temp directory from previous interrupted save: {tmp_path}{RESET}")
                shutil.rmtree(tmp_path, ignore_errors=True)

    # ---- Resume support ----------------------------------------------------
    resume_info = try_resume(model_engine, args, run_dir)
    start_step = resume_info["start_step"]
    is_resuming = start_step > 0

    # Sync EpochTracker with resumed position to avoid spurious epoch banner
    if is_resuming:
        epoch_tracker.current_epoch = start_step // epoch_tracker.steps_per_epoch if epoch_tracker.steps_per_epoch > 0 else 0

    # ---- Logging setup -----------------------------------------------------
    log_path, log_file = setup_logging(is_resuming=is_resuming)
    if is_resuming:
        log_resume(log_file, start_step)
    else:
        write_config_header(
            log_file, args, model_config_dict, ds_config, vocab_size, train_tokens, val_tokens
        )
    print(f"  Log file: {log_path}")

    # ---- Load tokenizer for sample generation ------------------------------
    tokenizer = None
    if os.path.isfile(args.tokenizer_path):
        try:
            tokenizer = Tokenizer.from_file(args.tokenizer_path)
        except Exception as e:
            print(f"  WARNING: Failed to load tokenizer for sample generation: {e}")

    # ---- Training loop -----------------------------------------------------
    block_size = model_config.block_size
    batch_size = args.batch_size
    max_steps = args.max_steps
    log_interval = args.log_interval
    eval_interval = args.eval_interval

    # Anomaly detection state
    loss_history = deque(maxlen=100)
    step_times = deque(maxlen=50)
    grad_norm_history = deque(maxlen=100)    # For emergency diagnostics
    loss_scale_history = deque(maxlen=100)   # For emergency diagnostics
    GRAD_NORM_WARN_THRESHOLD = 10.0
    LOSS_SPIKE_FACTOR = 2.0

    # Tracking variables
    initial_loss = None
    best_val_loss = float("inf")
    best_val_step = 0
    final_train_loss = 0.0
    final_val_loss = 0.0
    total_tokens_processed = start_step * tokens_per_step if is_resuming else 0
    loss_warned = False
    last_val_loss = float("inf")
    last_val_step = -1
    checkpoint_interval = args.checkpoint_interval

    # Restore best tracker from checkpoint on resume (overrides defaults above)
    if is_resuming:
        best_val_loss = resume_info["best_val_loss"]
        best_val_step = resume_info["best_val_step"]

    # Checkpoint statistics counters
    total_checkpoints_saved = 0
    total_best_updates = 0
    total_bytes_written = 0

    training_start_time = time.time()

    # Print full config summary at startup
    print_config_summary(
        args, model_config, param_count, vocab_size,
        train_tokens, val_tokens, ds_config,
        tokens_per_step, epoch_tracker, gpu_handle,
    )

    try:
        for step in range(start_step, max_steps):
            t0 = time.time()

            # Forward pass through DeepSpeed engine
            x, y = get_batch("train", batch_size, block_size, device, args.data_dir)
            logits, loss = model_engine(x, targets=y)

            # Read loss BEFORE backward to preserve clean optimizer state on NaN
            current_loss = loss.item()

            # NaN/Inf emergency shutdown (MUST be before backward pass)
            if math.isnan(current_loss) or math.isinf(current_loss):
                emergency_shutdown(
                    model_engine=model_engine, step=step, loss_value=current_loss,
                    run_dir=run_dir, log_file=log_file, tb_writer=tb_writer,
                    loss_history=loss_history, grad_norm_history=grad_norm_history,
                    loss_scale_history=loss_scale_history, epoch_tracker=epoch_tracker,
                    training_start_time=training_start_time, args=args,
                    gpu_handle=gpu_handle, optimizer=optimizer,
                )
                # emergency_shutdown calls sys.exit(2), never returns

            # Backward pass (DeepSpeed handles loss scaling for fp16)
            model_engine.backward(loss)

            # Weight update (handles gradient accumulation internally)
            model_engine.step()

            t1 = time.time()
            dt = t1 - t0

            # Read DeepSpeed internal metrics (valid after step())
            try:
                loss_scale = model_engine.optimizer.cur_scale
            except (AttributeError, RuntimeError):
                loss_scale = 0.0
            raw_grad_norm = model_engine.get_global_grad_norm()
            grad_norm = float(raw_grad_norm) if raw_grad_norm is not None else 0.0

            # Update epoch tracker
            epoch_crossed = epoch_tracker.update(step, current_loss)

            # Track initial loss for sanity check
            if initial_loss is None:
                initial_loss = current_loss

            if step == start_step + 100 and not loss_warned and current_loss >= initial_loss:
                print(
                    f"\n  {YELLOW}WARNING: Loss has not decreased after 100 steps "
                    f"(initial: {initial_loss:.4f}, current: {current_loss:.4f}){RESET}"
                )
                loss_warned = True

            total_tokens_processed += tokens_per_step

            # Update anomaly detection deques
            loss_history.append(current_loss)
            step_times.append(dt)
            grad_norm_history.append(grad_norm)
            loss_scale_history.append(loss_scale)

            # Epoch boundary banner
            if epoch_crossed:
                elapsed = time.time() - training_start_time
                prev_epoch = epoch_tracker.epoch  # Already incremented
                print(f"\n  {BOLD}{'=' * 50}{RESET}")
                print(f"  {BOLD}  Epoch {prev_epoch}/{epoch_tracker.total_epochs}{RESET}")
                print(f"  {BOLD}{'=' * 50}{RESET}")
                log_epoch(
                    log_file,
                    epoch=prev_epoch,
                    steps_in_epoch=epoch_tracker.steps_per_epoch,
                    avg_train_loss=epoch_tracker.completed_epoch_avg_loss,
                    total_tokens=total_tokens_processed,
                    elapsed_sec=elapsed,
                )

            # Log every log_interval steps
            if step % log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                tokens_per_sec = tokens_per_step / dt if dt > 0 else 0
                gpu_mem = get_gpu_memory_mb(gpu_handle)

                # Anomaly detection
                loss_warning = False
                grad_warning = False
                if len(loss_history) >= 10:
                    avg_loss = sum(loss_history) / len(loss_history)
                    if current_loss > LOSS_SPIKE_FACTOR * avg_loss:
                        loss_warning = True
                if grad_norm > GRAD_NORM_WARN_THRESHOLD:
                    grad_warning = True

                # Compute ETA (only show after 50+ steps for stability)
                if len(step_times) >= 50:
                    avg_dt = sum(step_times) / len(step_times)
                    remaining_steps = max_steps - step
                    eta_seconds = remaining_steps * avg_dt
                    eta_str = format_eta(eta_seconds)
                else:
                    eta_str = "..."

                # Determine console color
                pct = step / max_steps * 100
                ep = epoch_tracker.epoch
                total_ep = epoch_tracker.total_epochs

                if loss_warning or grad_warning:
                    color = YELLOW
                else:
                    color = GREEN

                line = (
                    f"{color}"
                    f"  Step {step:>6}/{max_steps} "
                    f"| ep {ep}/{total_ep} "
                    f"| loss {current_loss:.4f} "
                    f"| lr {current_lr:.2e} "
                    f"| {tokens_per_sec:>7,.0f} tok/s "
                    f"| ETA {eta_str} "
                    f"| {pct:.1f}%"
                    f"{RESET}"
                )
                if loss_warning:
                    line += f" {YELLOW}[LOSS SPIKE]{RESET}"
                if grad_warning:
                    line += f" {YELLOW}[HIGH GRAD]{RESET}"
                print(line)

                # JSONL log
                log_step(
                    log_file, step, current_loss, current_lr, tokens_per_sec,
                    gpu_mem, dt, epoch_tracker.epoch,
                    epoch_tracker.epoch_progress(step), total_tokens_processed,
                    loss_scale, grad_norm,
                )

                # TensorBoard custom metrics (step-aligned with JSONL log)
                try:
                    perplexity = min(math.exp(current_loss), 1e5)
                except OverflowError:
                    perplexity = 1e5
                tb_writer.add_scalar("Metrics/train_perplexity", perplexity, step)
                tb_writer.add_scalar("Metrics/gradient_norm", grad_norm, step)
                tb_writer.add_scalar("Metrics/loss_scale", loss_scale, step)
                tb_writer.add_scalar("Schedule/learning_rate", current_lr, step)
                tb_writer.add_scalar("Schedule/epoch", epoch_tracker.epoch, step)
                tb_writer.add_scalar("Progress/tokens_per_sec", tokens_per_sec, step)

            # --- Conditional 1: Validation (every eval_interval steps, and at final step) ---
            if step > 0 and (step % eval_interval == 0 or step == max_steps - 1):
                losses = estimate_loss(
                    model_engine, args.eval_iters, batch_size, block_size, device, args.data_dir
                )
                val_loss = losses["val"]
                final_train_loss = losses["train"]
                final_val_loss = val_loss

                try:
                    train_ppl = min(math.exp(losses["train"]), 1e5)
                except OverflowError:
                    train_ppl = 1e5
                try:
                    val_ppl = min(math.exp(val_loss), 1e5)
                except OverflowError:
                    val_ppl = 1e5
                gap = val_loss - losses["train"]
                gap_ratio = (val_loss - losses["train"]) / losses["train"] if losses["train"] > 0 else 0.0

                print(f"\n  --- Validation at step {step} (epoch {epoch_tracker.epoch}/{epoch_tracker.total_epochs}) ---")
                print(f"    Train loss:       {losses['train']:.4f}  (ppl: {train_ppl:.2f})")
                print(f"    Val loss:         {val_loss:.4f}  (ppl: {val_ppl:.2f})")
                print(f"    Gen. gap:         {gap:.4f}  (ratio: {gap_ratio:.4f})")

                # Multi-prompt sample generation (5 prompts x 2 temperatures = 10 samples)
                samples = []
                if tokenizer is not None:
                    try:
                        for prompt in SAMPLE_PROMPTS:
                            for temp in SAMPLE_TEMPERATURES:
                                text = generate_sample(
                                    model_engine.module,
                                    tokenizer,
                                    prompt,
                                    max_new_tokens=128,
                                    temperature=temp,
                                    device=device,
                                )
                                samples.append({
                                    "prompt": prompt,
                                    "temperature": temp,
                                    "text": text,
                                })
                        print(f"    {len(samples)} samples generated ({len(SAMPLE_PROMPTS)} prompts x {len(SAMPLE_TEMPERATURES)} temperatures)")
                    except Exception as e:
                        print(f"    WARNING: Sample generation failed: {e}")
                        if not samples:
                            samples = [{"prompt": SAMPLE_PROMPTS[0], "temperature": 0.0, "text": f"[generation failed: {e}]"}]

                # Log validation
                log_validation(
                    log_file, step, losses["train"], val_loss,
                    samples, epoch_tracker.epoch,
                    epoch_tracker.epoch_progress(step),
                )

                # TensorBoard validation metrics
                tb_writer.add_scalar("Validation/val_loss", val_loss, step)
                tb_writer.add_scalar("Validation/val_perplexity", val_ppl, step)
                tb_writer.add_scalar("Validation/train_loss", losses["train"], step)
                tb_writer.add_scalar("Validation/train_perplexity", train_ppl, step)
                tb_writer.add_scalar("Validation/generalization_gap", gap, step)
                tb_writer.add_scalar("Validation/gap_ratio", gap_ratio, step)

                # Check for new best and save immediately
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    best_val_step = step
                    print(f"    {GREEN}{BOLD}*** New best val loss: {val_loss:.4f} at step {step} ***{RESET}")
                    try:
                        wall_elapsed = time.time() - training_start_time
                        best_save_duration = save_best_checkpoint_atomic(
                            model_engine, run_dir, step, val_loss,
                            model_config_dict, args.config,
                            epoch_tracker.epoch, wall_elapsed,
                        )
                        best_dir = os.path.join(run_dir, "best")
                        best_size = sum(
                            os.path.getsize(os.path.join(dp, fn))
                            for dp, _, fnames in os.walk(best_dir)
                            for fn in fnames
                        )
                        print(f"    Best checkpoint saved: {best_dir} ({best_size / 1024 / 1024:.1f} MB, {best_save_duration:.1f}s)")
                        total_best_updates += 1
                        total_bytes_written += best_size
                        log_checkpoint_event(log_file, "best_update", step, {
                            "val_loss": val_loss,
                            "path": "best",
                            "size_mb": round(best_size / 1024 / 1024, 1),
                            "save_duration_sec": round(best_save_duration, 1),
                        })
                    except Exception as e:
                        print(f"    {YELLOW}WARNING: Best checkpoint save failed: {e}{RESET}")

                # Update tracking for checkpoint metadata
                last_val_loss = val_loss
                last_val_step = step

                print()  # Blank line after validation block

            # --- Conditional 2: Regular checkpoint (every checkpoint_interval steps, and at final step) ---
            if step > 0 and (step % checkpoint_interval == 0 or step == max_steps - 1):
                current_lr = optimizer.param_groups[0]["lr"]
                wall_elapsed = time.time() - training_start_time
                client_state = {
                    "step": step,
                    "train_loss": current_loss,
                    "val_loss": last_val_loss,
                    "lr": current_lr,
                    "best_val_loss": best_val_loss,
                    "best_val_step": best_val_step,
                }
                checkpoint_meta = {
                    "step": step,
                    "train_loss": current_loss,
                    "val_loss": last_val_loss,
                    "is_best": False,
                    "epoch": epoch_tracker.epoch,
                    "wall_clock_elapsed": wall_elapsed,
                    "lr": current_lr,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "config_name": args.config,
                }

                try:
                    save_duration = save_checkpoint_atomic(
                        model_engine, run_dir, step, client_state,
                        checkpoint_meta, args.config,
                    )
                    ds_dir = os.path.join(run_dir, f"step_{step}")
                    ds_size = sum(
                        os.path.getsize(os.path.join(dp, fn))
                        for dp, _, fnames in os.walk(ds_dir)
                        for fn in fnames
                    )
                    retained = len([
                        d for d in os.listdir(run_dir)
                        if d.startswith("step_") and os.path.isdir(os.path.join(run_dir, d))
                    ])
                    print(f"  Checkpoint saved: step_{step} ({ds_size / 1024 / 1024:.1f} MB, {save_duration:.1f}s) | retained: {retained} | best: step {best_val_step}")
                    total_checkpoints_saved += 1
                    total_bytes_written += ds_size
                    log_checkpoint_event(log_file, "checkpoint_save", step, {
                        "path": f"step_{step}",
                        "size_mb": round(ds_size / 1024 / 1024, 1),
                        "save_duration_sec": round(save_duration, 1),
                        "retained_count": retained,
                    })
                except Exception as e:
                    print(f"  {YELLOW}WARNING: Checkpoint save failed at step {step}: {e}{RESET}")
                    print(f"  Training will continue without this checkpoint.")

                # Cleanup old checkpoints
                cleanup_checkpoints(run_dir, keep_last=3, log_file=log_file)

    finally:
        log_file.close()
        tb_writer.close()

    # ---- Training complete -------------------------------------------------
    elapsed = time.time() - training_start_time

    # Read best from best/checkpoint_meta.json in case we need to recover
    best_meta_path = os.path.join(run_dir, "best", "checkpoint_meta.json")
    if os.path.isfile(best_meta_path):
        try:
            with open(best_meta_path) as f:
                best_meta = json.load(f)
            best_val_step = best_meta.get("step", best_val_step)
            best_val_loss = best_meta.get("val_loss", best_val_loss)
        except (json.JSONDecodeError, OSError):
            pass

    print_training_summary(
        config_name=args.config,
        total_steps=max_steps - start_step,
        elapsed_time=elapsed,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        best_val_loss=best_val_loss,
        best_val_step=best_val_step,
        total_tokens_processed=total_tokens_processed,
        run_dir=run_dir,
        epoch_tracker=epoch_tracker,
        total_checkpoints_saved=total_checkpoints_saved,
        total_best_updates=total_best_updates,
        total_bytes_written=total_bytes_written,
    )


if __name__ == "__main__":
    main()
