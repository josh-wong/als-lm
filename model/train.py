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
- **Dual checkpoints:** Saves both DeepSpeed directory checkpoints (for
  resume) and raw .pt files (for export) at each checkpoint interval.
- **Checkpoint retention:** Keeps last 3 checkpoints + best (lowest val
  loss), deleting both formats for expired checkpoints.
- **Console + JSON logging:** Human-readable table to stdout every 10 steps,
  JSON lines to a timestamped log file.
- **Validation + sample generation:** Every 250 steps, estimates train/val
  loss and generates sample text from a fixed ALS prompt.
- **Loss sanity checks:** Stops immediately on NaN/Inf; warns if loss hasn't
  decreased after 100 steps.
- **Resume support:** Auto-detects existing checkpoints or accepts explicit
  ``--resume`` path.
- **Dry-run mode:** ``--dry-run`` validates config and prints training plan
  without GPU time.

Usage::

    # Tiny model (pipeline validation)
    deepspeed model/train.py --deepspeed --deepspeed_config config/ds_zero2.json --config tiny

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

# Fixed prompt for sample text generation at validation intervals
SAMPLE_PROMPT = "Amyotrophic lateral sclerosis is"


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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250,
        help="Steps between validation / checkpoint (default: 250)",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=20,
        help="Batches to average for validation loss (default: 20)",
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
def generate_sample(model, tokenizer, prompt, max_new_tokens=64, temperature=0.8, device="cuda"):
    """Generate sample text from a fixed prompt for qualitative evaluation.

    Uses the unwrapped model (model_engine.module) in eval mode to avoid
    DeepSpeed training overhead during generation.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
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


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------
def save_checkpoint(
    model_engine, model_config_dict, step, train_loss, val_loss, lr, run_dir, config_name
):
    """Save both DeepSpeed directory checkpoint and raw .pt file.

    Wraps the entire save process in try/except so that a failed checkpoint
    does not stop training (per CONTEXT.md locked decision).
    """
    tag = f"step_{step}"

    try:
        # 1. Save DeepSpeed directory checkpoint (optimizer states, sharded model)
        client_state = {"step": step, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
        model_engine.save_checkpoint(run_dir, tag=tag, client_state=client_state)

        # 2. Extract fp32 state dict from ZeRO checkpoint
        # Pass run_dir (which contains the 'latest' file) with an explicit tag
        # so the utility can locate the step_N/ subdirectory correctly.
        ds_ckpt_path = os.path.join(run_dir, tag)
        state_dict = get_fp32_state_dict_from_zero_checkpoint(run_dir, tag=tag)

        # 3. Save raw .pt file for export
        pt_checkpoint = {
            "model": state_dict,
            "config": model_config_dict,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        pt_path = os.path.join(run_dir, f"{tag}.pt")
        torch.save(pt_checkpoint, pt_path)

        # 4. Save checkpoint_meta.json
        meta = {
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_name": config_name,
        }
        meta_path = os.path.join(ds_ckpt_path, "checkpoint_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 5. Print checkpoint info
        ds_size = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fnames in os.walk(ds_ckpt_path)
            for fn in fnames
        )
        pt_size = os.path.getsize(pt_path)
        print(f"\n  Checkpoint saved: {ds_ckpt_path}")
        print(f"    DeepSpeed dir: {ds_size / 1024 / 1024:.1f} MB")
        print(f"    Raw .pt file:  {pt_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"\n  WARNING: Checkpoint save failed at step {step}: {e}")
        print("  Training will continue without this checkpoint.")


def cleanup_checkpoints(run_dir, keep_last=3):
    """Retain last N checkpoints + best (lowest val loss), delete the rest.

    Deletes both the DeepSpeed directory (step_N/) and the .pt file
    (step_N.pt) for expired checkpoints.
    """
    # Find all checkpoint step directories
    steps = []
    for entry in os.listdir(run_dir):
        entry_path = os.path.join(run_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("step_"):
            try:
                step_num = int(entry.split("_")[1])
                steps.append(step_num)
            except (ValueError, IndexError):
                continue
    steps.sort()

    if len(steps) <= keep_last:
        return  # Nothing to clean up

    # Read best step from tracking file
    best_step_file = os.path.join(run_dir, "best_step.txt")
    best_step = None
    if os.path.isfile(best_step_file):
        try:
            with open(best_step_file) as f:
                best_step = int(f.read().strip())
        except (ValueError, OSError):
            pass

    # Determine which steps to keep: last N + best
    keep_set = set(steps[-keep_last:])
    if best_step is not None:
        keep_set.add(best_step)

    # Delete expired checkpoints
    for step_num in steps:
        if step_num not in keep_set:
            # Delete DeepSpeed directory
            ds_dir = os.path.join(run_dir, f"step_{step_num}")
            if os.path.isdir(ds_dir):
                shutil.rmtree(ds_dir)
                print(f"  Deleted checkpoint: {ds_dir}")

            # Delete .pt file
            pt_file = os.path.join(run_dir, f"step_{step_num}.pt")
            if os.path.isfile(pt_file):
                os.remove(pt_file)
                print(f"  Deleted checkpoint: {pt_file}")


def update_best_checkpoint(run_dir, step, val_loss):
    """Track the best checkpoint (lowest val loss) in a persistent file."""
    best_step_file = os.path.join(run_dir, "best_step.txt")
    best_loss_file = os.path.join(run_dir, "best_loss.txt")

    current_best_loss = float("inf")
    if os.path.isfile(best_loss_file):
        try:
            with open(best_loss_file) as f:
                current_best_loss = float(f.read().strip())
        except (ValueError, OSError):
            pass

    if val_loss < current_best_loss:
        with open(best_step_file, "w") as f:
            f.write(str(step))
        with open(best_loss_file, "w") as f:
            f.write(str(val_loss))
        return True  # New best
    return False


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def try_resume(model_engine, args, run_dir):
    """Attempt to resume training from an existing checkpoint.

    If --resume is provided, uses that path. Otherwise, auto-detects the
    latest checkpoint in the run directory.

    Returns the starting step number (0 if no checkpoint found).
    """
    ckpt_dir = args.resume if args.resume else run_dir

    if not os.path.isdir(ckpt_dir):
        return 0

    # Check if there are any checkpoint subdirectories
    has_checkpoints = any(
        entry.startswith("step_") and os.path.isdir(os.path.join(ckpt_dir, entry))
        for entry in os.listdir(ckpt_dir)
    )
    if not has_checkpoints:
        return 0

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
            return start_step
    except Exception as e:
        print(f"\n  WARNING: Failed to resume from {ckpt_dir}: {e}")
        print("  Starting training from scratch.")

    return 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(config_name):
    """Create logs directory and open a JSON lines log file.

    Returns the log file path and file handle.
    """
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/train_{timestamp}.jsonl"
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


def log_step(log_file, step, loss, lr, tokens_per_sec, gpu_mem_mb, dt_sec):
    """Write a training step entry to the JSON log file."""
    entry = {
        "type": "step",
        "step": step,
        "loss": loss,
        "lr": lr,
        "tokens_per_sec": tokens_per_sec,
        "gpu_mem_mb": gpu_mem_mb,
        "dt_sec": dt_sec,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


def log_validation(log_file, step, train_loss, val_loss, sample_text):
    """Write a validation entry to the JSON log file."""
    entry = {
        "type": "validation",
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "sample_text": sample_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    print(f"    Seed:            {args.seed}")
    print(f"    Gradient checkpointing: {args.gradient_checkpointing}")

    print(f"\n  Data Info")
    print(f"    Data directory:  {args.data_dir}")
    print(f"    Train tokens:    {train_tokens:,}")
    print(f"    Val tokens:      {val_tokens:,}")
    print(f"    Tokens/step:     {tokens_per_step:,}")
    print(f"    Total tokens:    {total_tokens:,}")
    print(f"    Est. epochs:     {epochs_approx:.1f}")

    print("\n" + "=" * 60)
    print("  Dry run complete. Remove --dry-run to start training.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------
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
):
    """Print the end-of-run training summary."""
    minutes, seconds = divmod(int(elapsed_time), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    else:
        time_str = f"{minutes}m {seconds}s"

    avg_throughput = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0

    print("\n" + "=" * 50)
    print("  Training Complete")
    print("=" * 50)
    print(f"  Config:           {config_name}")
    print(f"  Total steps:      {total_steps:,}")
    print(f"  Total time:       {time_str}")
    print(f"  Final train loss: {final_train_loss:.4f}")
    print(f"  Final val loss:   {final_val_loss:.4f}")
    print(f"  Best val loss:    {best_val_loss:.4f} (step {best_val_step})")
    print(f"  Avg throughput:   {avg_throughput:,.0f} tokens/sec")
    print(f"  Checkpoint dir:   {run_dir}")
    print("=" * 50 + "\n")


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

    # ---- GPU memory monitor ------------------------------------------------
    gpu_handle = init_gpu_monitor(args.local_rank)

    # ---- Logging setup -----------------------------------------------------
    log_path, log_file = setup_logging(args.config)
    write_config_header(
        log_file, args, model_config_dict, ds_config, vocab_size, train_tokens, val_tokens
    )
    print(f"  Log file: {log_path}")

    # ---- Checkpoint directory setup ----------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("checkpoints", f"{args.config}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"  Checkpoint dir: {run_dir}")

    # ---- Resume support ----------------------------------------------------
    start_step = try_resume(model_engine, args, run_dir)

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

    effective_batch = batch_size * args.grad_accum
    tokens_per_step = effective_batch * block_size

    # Tracking variables
    initial_loss = None
    best_val_loss = float("inf")
    best_val_step = 0
    final_train_loss = 0.0
    final_val_loss = 0.0
    total_tokens_processed = 0
    loss_warned = False

    training_start_time = time.time()
    print(f"\n{'=' * 60}")
    print(f"  Starting training: {args.config} model, {max_steps:,} steps")
    print(f"  Effective batch size: {effective_batch} (micro={batch_size} x accum={args.grad_accum})")
    print(f"{'=' * 60}\n")

    for step in range(start_step, max_steps):
        t0 = time.time()

        # Forward pass through DeepSpeed engine
        x, y = get_batch("train", batch_size, block_size, device, args.data_dir)
        logits, loss = model_engine(x, targets=y)

        # Backward pass (DeepSpeed handles loss scaling for fp16)
        model_engine.backward(loss)

        # Weight update (handles gradient accumulation internally)
        model_engine.step()

        t1 = time.time()
        dt = t1 - t0
        current_loss = loss.item()

        # Track initial loss for sanity check
        if initial_loss is None:
            initial_loss = current_loss

        # Loss sanity checks
        if math.isnan(current_loss) or math.isinf(current_loss):
            print(f"\nFATAL: NaN/Inf loss at step {step}. Stopping immediately.")
            log_file.close()
            sys.exit(1)

        if step == start_step + 100 and not loss_warned and current_loss >= initial_loss:
            print(
                f"\n  WARNING: Loss has not decreased after 100 steps "
                f"(initial: {initial_loss:.4f}, current: {current_loss:.4f})"
            )
            loss_warned = True

        total_tokens_processed += tokens_per_step

        # Log every log_interval steps
        if step % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            tokens_per_sec = tokens_per_step / dt if dt > 0 else 0
            gpu_mem = get_gpu_memory_mb(gpu_handle)

            # Console output: fixed-width aligned columns
            print(
                f"  Step {step:>6}/{max_steps} | "
                f"loss: {current_loss:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"tok/s: {tokens_per_sec:>9,.0f} | "
                f"GPU: {gpu_mem:>5,} MB | "
                f"dt: {dt:.2f}s"
            )

            # JSON log
            log_step(log_file, step, current_loss, current_lr, tokens_per_sec, gpu_mem, dt)

        # Validate + checkpoint + generate sample every eval_interval steps (and at final step)
        if step > 0 and (step % eval_interval == 0 or step == max_steps - 1):
            losses = estimate_loss(
                model_engine, args.eval_iters, batch_size, block_size, device, args.data_dir
            )
            final_train_loss = losses["train"]
            final_val_loss = losses["val"]

            print(f"\n  --- Validation at step {step} ---")
            print(f"    Train loss: {losses['train']:.4f}")
            print(f"    Val loss:   {losses['val']:.4f}")

            # Sample text generation
            sample_text = ""
            if tokenizer is not None:
                try:
                    sample_text = generate_sample(
                        model_engine.module,
                        tokenizer,
                        SAMPLE_PROMPT,
                        max_new_tokens=64,
                        temperature=0.8,
                        device=device,
                    )
                    print(f"\n  --- Sample generation ---")
                    print(f"    Prompt: \"{SAMPLE_PROMPT}\"")
                    print(f"    Output: \"{sample_text[:200]}\"")
                except Exception as e:
                    print(f"    WARNING: Sample generation failed: {e}")
                    sample_text = f"[generation failed: {e}]"

            # Log validation
            log_validation(log_file, step, losses["train"], losses["val"], sample_text)

            # Update best checkpoint tracking
            is_best = update_best_checkpoint(run_dir, step, losses["val"])
            if is_best:
                best_val_loss = losses["val"]
                best_val_step = step
                print(f"    New best val loss: {best_val_loss:.4f}")
            elif losses["val"] < best_val_loss:
                # Handle case where best was set from file on resume
                best_val_loss = losses["val"]
                best_val_step = step

            # Save checkpoint
            current_lr = optimizer.param_groups[0]["lr"]
            save_checkpoint(
                model_engine,
                model_config_dict,
                step,
                losses["train"],
                losses["val"],
                current_lr,
                run_dir,
                args.config,
            )

            # Cleanup old checkpoints
            cleanup_checkpoints(run_dir, keep_last=3)

            print()  # Blank line after validation block

    # ---- Training complete -------------------------------------------------
    elapsed = time.time() - training_start_time
    log_file.close()

    # Read best from file in case we didn't find one during training
    best_step_file = os.path.join(run_dir, "best_step.txt")
    best_loss_file = os.path.join(run_dir, "best_loss.txt")
    if os.path.isfile(best_step_file) and os.path.isfile(best_loss_file):
        try:
            with open(best_step_file) as f:
                best_val_step = int(f.read().strip())
            with open(best_loss_file) as f:
                best_val_loss = float(f.read().strip())
        except (ValueError, OSError):
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
    )


if __name__ == "__main__":
    main()
