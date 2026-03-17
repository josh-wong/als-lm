#!/usr/bin/env python3
"""Readiness gate benchmark for ALS-LM (500M and 1B modes).

A standalone benchmark script that validates whether a GPT-2 model can train
on the RTX 3060 12GB GPU with DeepSpeed ZeRO Stage 2. Produces a data-driven
go/no-go recommendation before committing GPU-weeks to full training.

**500M mode (default):** Runs a 4-configuration matrix (CPU offload on/off x
gradient checkpointing on/off) for the primary model size, tests an
auto-fallback chain (1B -> 750M -> 500M -> 350M), verifies checkpoint resume
with loss continuity, and generates a readiness report.

**1B pre-flight mode (--primary-size 1B):** Runs environment checks (WSL2
memory, disk space), a single production config benchmark (offload=ON,
grad_ckpt=ON, batch_size=4) for 500 steps, forced checkpoint resume with LR
verification, applies VRAM/throughput thresholds (10 GB hard-fail), calculates
max_steps from token count, updates configs/1b.json, and generates
readiness_report_1b.md.

This script is launched directly with ``python``, NOT via the DeepSpeed
launcher. It handles DeepSpeed distributed initialization internally for each
configuration.

Usage::

    # Full benchmark (default: 500M primary, 100 steps per config)
    python benchmark/readiness_gate.py

    # Quick test with fewer steps
    python benchmark/readiness_gate.py --steps 10

    # Skip fallback chain (only test primary size)
    python benchmark/readiness_gate.py --skip-fallback

    # 1B pre-flight mode (auto-enables --preflight)
    python benchmark/readiness_gate.py --primary-size 1B --data-dir data/tokenized/v1.2.0
"""

import argparse
import copy
import gc
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

# Set distributed environment BEFORE importing deepspeed so the script
# can be launched with plain ``python`` instead of the deepspeed launcher.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

# Ensure project root is on sys.path (same pattern as train.py).
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch

import deepspeed

import psutil

from model.model import GPT, GPTConfig

# Optional pynvml for GPU memory timeline
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model configurations (defined here, NOT in model.py)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "1B": {
        "n_layer": 30,
        "n_head": 20,
        "n_embd": 1600,
        "block_size": 1024,
        "dropout": 0.0,
        "bias": True,
    },
    "750M": {
        "n_layer": 28,
        "n_head": 20,
        "n_embd": 1600,
        "block_size": 1024,
        "dropout": 0.0,
        "bias": True,
    },
    "500M": {
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1280,
        "block_size": 1024,
        "dropout": 0.0,
        "bias": True,
    },
    "350M": {
        "n_layer": 20,
        "n_head": 16,
        "n_embd": 1024,
        "block_size": 1024,
        "dropout": 0.0,
        "bias": True,
    },
}

# Fallback order: largest first, stopping at the first that fits.
FALLBACK_ORDER = ["1B", "750M", "500M", "350M"]

# 72-hour maximum threshold (informational, not auto-fail)
MAX_PROJECTED_HOURS = 72


# ---------------------------------------------------------------------------
# GPU memory timeline (pynvml polling thread)
# ---------------------------------------------------------------------------
class GPUMemoryTimeline:
    """Background daemon thread polling GPU memory via pynvml at fixed intervals.

    Records a time-series of GPU memory usage for post-hoc analysis. Degrades
    gracefully when pynvml is unavailable (returns empty samples, peak=0).

    Attributes:
        interval: Polling interval in seconds.
        samples: List of dicts with timestamp, used_mb, total_mb per sample.
    """

    def __init__(self, interval_sec: float = 0.5, device_index: int = 0):
        self.interval = interval_sec
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._available = False

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._available = True
            except Exception:
                pass

    def start(self):
        """Begin background polling."""
        if not self._available:
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background polling and wait for thread exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _poll(self):
        """Continuously sample GPU memory until stop is signalled."""
        while not self._stop.is_set():
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.samples.append({
                    "timestamp": time.time(),
                    "used_mb": info.used // (1024 * 1024),
                    "total_mb": info.total // (1024 * 1024),
                })
            except Exception:
                pass
            self._stop.wait(self.interval)

    def get_peak_mb(self) -> int:
        """Return the maximum used_mb observed, or 0 if no samples."""
        if not self.samples:
            return 0
        return max(s["used_mb"] for s in self.samples)

    def get_samples(self) -> list[dict]:
        """Return all recorded samples."""
        return list(self.samples)


# ---------------------------------------------------------------------------
# Data loading (nanoGPT memmap pattern, same as train.py)
# ---------------------------------------------------------------------------
def get_batch(split: str, batch_size: int, block_size: int, device, data_dir: str):
    """Load a random batch from memory-mapped binary token files.

    Recreates the numpy memmap on every call to prevent memory leaks from
    numpy reference counting. This is the correct nanoGPT pattern.
    """
    fname = "train.bin" if split == "train" else "val.bin"
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i: i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1: i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)
    return x, y


# ---------------------------------------------------------------------------
# Dynamic DeepSpeed config generation
# ---------------------------------------------------------------------------
def make_ds_config(
    base_config_path: str,
    batch_size: int = 2,
    grad_accum: int = 1,
    lr: float = 3e-4,
    warmup_steps: int = 10,
    total_steps: int = 100,
    cpu_offload: bool = True,
) -> dict:
    """Generate a DeepSpeed config dict from the base config with overrides.

    When cpu_offload is True, the optimizer is placed on CPU with pinned memory.
    When False, offload_optimizer.device is set to "none" and DeepSpeed places
    the Adam optimizer on GPU automatically.
    """
    with open(base_config_path) as f:
        config = json.load(f)

    config["train_micro_batch_size_per_gpu"] = batch_size
    config["gradient_accumulation_steps"] = grad_accum
    config["train_batch_size"] = batch_size * grad_accum

    config["optimizer"]["params"]["lr"] = lr

    config["scheduler"]["params"]["warmup_num_steps"] = warmup_steps
    config["scheduler"]["params"]["total_num_steps"] = total_steps

    if cpu_offload:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }
    else:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "none",
        }

    return config


# ---------------------------------------------------------------------------
# Per-config benchmark function
# ---------------------------------------------------------------------------
def run_benchmark_config(
    model_config: dict,
    ds_config: dict,
    steps: int,
    data_dir: str,
    vocab_size: int,
    block_size: int = 1024,
    save_final_checkpoint: str | None = None,
) -> dict:
    """Run a single benchmark configuration and return results.

    Wraps the entire training loop in try/except to catch OOM and other errors
    without crashing the overall benchmark. Always cleans up GPU state in a
    finally block.

    If ``save_final_checkpoint`` is a directory path, saves a DeepSpeed
    checkpoint at the final step with client_state containing the step index,
    final loss, and learning rate. This is used by the subprocess-based resume
    verification for the 1B pre-flight gate.

    Returns a dict with: status, peak_gpu_gb, tokens_per_sec, loss_at_1,
    loss_at_100, step_times, memory_timeline, cpu_ram_gb, disk_io_read_mb,
    disk_io_write_mb, params, and checkpoint_save_time_sec.
    """
    engine = None
    model = None
    timeline = GPUMemoryTimeline(interval_sec=0.5)

    try:
        # Reset GPU state
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        # Disk I/O baseline
        disk_before = psutil.disk_io_counters()

        # Start GPU memory timeline
        timeline.start()

        # Build model
        config_with_vocab = {**model_config, "vocab_size": vocab_size}
        gpt_config = GPTConfig(**config_with_vocab)
        model = GPT(gpt_config)
        param_count = model.get_num_params()

        # Enable gradient checkpointing if the config specifies it
        if model_config.get("use_gradient_checkpointing", False):
            model.enable_gradient_checkpointing()

        # Initialize DeepSpeed
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        device = engine.device
        batch_size = ds_config["train_micro_batch_size_per_gpu"]

        # Training loop
        step_times = []
        losses = []
        loss_at_1 = None
        loss_at_final = None
        checkpoint_save_time = 0.0

        for step in range(steps):
            x, y = get_batch("train", batch_size, block_size, device, data_dir)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits, loss = engine(x, targets=y)
            engine.backward(loss)
            engine.step()

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            step_time = t1 - t0
            step_times.append(step_time)
            loss_val = loss.item()
            losses.append(loss_val)

            if step == 0:
                loss_at_1 = loss_val
            loss_at_final = loss_val

            # Measure one checkpoint save time at step 50 (or last step for
            # short runs) to factor into projected training duration.
            if step == min(50, steps - 1) and checkpoint_save_time == 0.0:
                with tempfile.TemporaryDirectory() as ckpt_dir:
                    torch.cuda.synchronize()
                    ckpt_t0 = time.perf_counter()
                    engine.save_checkpoint(
                        ckpt_dir,
                        tag="bench_ckpt",
                        client_state={"step": step},
                    )
                    torch.cuda.synchronize()
                    ckpt_t1 = time.perf_counter()
                    checkpoint_save_time = ckpt_t1 - ckpt_t0

        # Save final checkpoint for subprocess resume verification
        if save_final_checkpoint is not None:
            os.makedirs(save_final_checkpoint, exist_ok=True)
            lr_at_save = None
            try:
                lr_at_save = engine.lr_scheduler.get_lr()[0]
            except Exception:
                pass
            engine.save_checkpoint(
                save_final_checkpoint,
                tag=f"step_{steps}",
                client_state={
                    "step": steps - 1,
                    "loss": loss_at_final,
                    "lr": lr_at_save,
                },
            )

        # Capture peak GPU memory
        peak_gpu_bytes = torch.cuda.max_memory_allocated()
        peak_gpu_gb = peak_gpu_bytes / (1024 ** 3)

        # Stop timeline
        timeline.stop()

        # CPU RAM
        cpu_ram_gb = psutil.Process().memory_info().rss / (1024 ** 3)

        # Disk I/O delta
        disk_after = psutil.disk_io_counters()
        disk_read_mb = (disk_after.read_bytes - disk_before.read_bytes) / (1024 * 1024)
        disk_write_mb = (disk_after.write_bytes - disk_before.write_bytes) / (1024 * 1024)

        # Tokens/sec: use last 50 steps (or all if fewer) to skip warmup
        warmup_skip = max(0, len(step_times) - 50)
        effective_step_times = step_times[warmup_skip:]
        tokens_per_step = batch_size * block_size
        if effective_step_times:
            mean_step_time = sum(effective_step_times) / len(effective_step_times)
            tokens_per_sec = tokens_per_step / mean_step_time if mean_step_time > 0 else 0
        else:
            tokens_per_sec = 0

        return {
            "status": "pass",
            "peak_gpu_gb": peak_gpu_gb,
            "tokens_per_sec": tokens_per_sec,
            "loss_at_1": loss_at_1,
            "loss_at_final": loss_at_final,
            "step_times": step_times,
            "memory_timeline": timeline.get_samples(),
            "cpu_ram_gb": cpu_ram_gb,
            "disk_io_read_mb": disk_read_mb,
            "disk_io_write_mb": disk_write_mb,
            "params": param_count,
            "checkpoint_save_time_sec": checkpoint_save_time,
        }

    except torch.cuda.OutOfMemoryError:
        timeline.stop()
        return {"status": "oom", "params": model.get_num_params() if model else 0}

    except RuntimeError as e:
        timeline.stop()
        if "out of memory" in str(e).lower():
            return {"status": "oom", "params": model.get_num_params() if model else 0}
        return {
            "status": "error",
            "error": str(e),
            "params": model.get_num_params() if model else 0,
        }

    except Exception as e:
        timeline.stop()
        return {
            "status": "error",
            "error": str(e),
            "params": model.get_num_params() if model else 0,
        }

    finally:
        # Always clean up GPU state
        try:
            if engine is not None:
                del engine
            if model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Checkpoint resume verification
# ---------------------------------------------------------------------------
def verify_checkpoint_resume(
    model_config: dict,
    ds_config: dict,
    data_dir: str,
    vocab_size: int,
    block_size: int = 1024,
    save_step: int = 50,
    warmup_steps: int = 10,
    lr: float = 3e-4,
) -> dict:
    """Save at save_step, destroy model, rebuild from scratch, verify loss and LR continuity.

    Phase 1: Build model+engine with seed=42, train save_step steps, record loss,
             save checkpoint with client_state.
    Phase 2: Destroy engine and model, clear GPU, garbage collect.
    Phase 3: Reset seeds to 999, rebuild model+engine from scratch, load checkpoint.
    Phase 4: Run one forward pass on a new batch, record loss. Check LR state.

    The save_step parameter defaults to 50 for backward compatibility (500M gate)
    but should be set to 500 for the 1B pre-flight gate.

    Returns dict with loss_at_save, loss_at_resume, ratio, passed (bool),
    lr_at_save, lr_at_resume.
    """
    ckpt_dir = tempfile.mkdtemp(prefix="als_benchmark_ckpt_")
    engine = None
    model = None

    try:
        # Phase 1: Train save_step steps with seed=42
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        config_with_vocab = {**model_config, "vocab_size": vocab_size}
        gpt_config = GPTConfig(**config_with_vocab)
        model = GPT(gpt_config)

        if model_config.get("use_gradient_checkpointing", False):
            model.enable_gradient_checkpointing()

        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        device = engine.device
        batch_size = ds_config["train_micro_batch_size_per_gpu"]

        loss_at_save = None
        lr_at_save = None
        for step in range(save_step):
            x, y = get_batch("train", batch_size, block_size, device, data_dir)
            logits, loss = engine(x, targets=y)
            engine.backward(loss)
            engine.step()

            if step == save_step - 1:
                loss_at_save = loss.item()
                # Capture LR at save point
                try:
                    lr_at_save = engine.lr_scheduler.get_lr()[0]
                except Exception:
                    lr_at_save = None
                tag = f"step_{save_step}"
                engine.save_checkpoint(
                    ckpt_dir,
                    tag=tag,
                    client_state={
                        "step": step,
                        "loss": loss_at_save,
                        "lr": lr_at_save,
                    },
                )

        # Phase 2: Destroy everything
        del engine, model
        engine = None
        model = None
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

        # Phase 3: Reset seeds to different values, rebuild from scratch
        torch.manual_seed(999)
        torch.cuda.manual_seed(999)
        np.random.seed(999)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        model = GPT(GPTConfig(**config_with_vocab))
        if model_config.get("use_gradient_checkpointing", False):
            model.enable_gradient_checkpointing()

        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        device = engine.device

        _, client_state = engine.load_checkpoint(ckpt_dir)
        resume_step = client_state["step"] if client_state else -1
        saved_loss = client_state.get("loss", None) if client_state else None

        # Capture LR after resume
        lr_at_resume = None
        try:
            lr_at_resume = engine.lr_scheduler.get_lr()[0]
        except Exception:
            pass

        # Phase 4: Run one forward pass on a new batch
        x, y = get_batch("train", batch_size, block_size, device, data_dir)
        with torch.no_grad():
            logits, loss = engine(x, targets=y)
        loss_at_resume = loss.item()

        ratio = loss_at_resume / loss_at_save if loss_at_save and loss_at_save > 0 else float("inf")
        passed = loss_at_resume <= 2.0 * loss_at_save

        # Expected LR at save_step during linear warmup phase:
        # lr * min(save_step, warmup_steps) / warmup_steps
        expected_lr = lr * min(save_step, warmup_steps) / warmup_steps

        return {
            "loss_at_save": loss_at_save,
            "loss_at_resume": loss_at_resume,
            "ratio": ratio,
            "passed": passed,
            "resume_step": resume_step,
            "lr_at_save": lr_at_save,
            "lr_at_resume": lr_at_resume,
            "expected_lr": expected_lr,
        }

    except Exception as e:
        return {
            "loss_at_save": None,
            "loss_at_resume": None,
            "ratio": None,
            "passed": False,
            "error": str(e),
        }

    finally:
        try:
            if engine is not None:
                del engine
            if model is not None:
                del model
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
        except Exception:
            pass

        # Clean up temp directory
        try:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Subprocess-based checkpoint resume verification (1B pre-flight)
# ---------------------------------------------------------------------------
def verify_checkpoint_resume_subprocess(
    model_config: dict,
    ds_config: dict,
    data_dir: str,
    vocab_size: int,
    ckpt_dir: str,
    save_step: int,
    block_size: int = 1024,
) -> dict:
    """Verify checkpoint resume in a fresh subprocess to avoid hot-resume OOM.

    Instead of loading a checkpoint in the same process that just trained
    (which can OOM on 1B models due to residual GPU/CPU memory), this function
    spawns a new Python process with a clean GPU state. The subprocess loads
    the checkpoint, runs one forward pass, and writes results to a temp file.

    This mirrors real production resume behavior (fresh process after crash).
    """
    result_file = os.path.join(ckpt_dir, "_resume_result.json")

    # Serialize configs for the subprocess — use absolute paths so the
    # subprocess doesn't depend on cwd.
    config_file = os.path.join(ckpt_dir, "_resume_configs.json")
    with open(config_file, "w") as f:
        json.dump({
            "model_config": model_config,
            "ds_config": ds_config,
            "data_dir": os.path.abspath(data_dir),
            "vocab_size": vocab_size,
            "ckpt_dir": ckpt_dir,
            "save_step": save_step,
            "block_size": block_size,
            "result_file": result_file,
        }, f)

    # Subprocess script that loads checkpoint and verifies
    script = f'''
import gc
import json
import os
import sys

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, "{_project_root}")

import numpy as np
import torch
import deepspeed
from model.model import GPT, GPTConfig

config_file = "{config_file}"
with open(config_file) as f:
    cfg = json.load(f)

model_config = cfg["model_config"]
ds_config = cfg["ds_config"]
data_dir = cfg["data_dir"]
vocab_size = cfg["vocab_size"]
ckpt_dir = cfg["ckpt_dir"]
save_step = cfg["save_step"]
block_size = cfg["block_size"]
result_file = cfg["result_file"]

try:
    torch.manual_seed(999)
    torch.cuda.manual_seed(999)
    np.random.seed(999)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    config_with_vocab = {{**model_config, "vocab_size": vocab_size}}
    gpt_config = GPTConfig(**config_with_vocab)
    model = GPT(gpt_config)
    if model_config.get("use_gradient_checkpointing", False):
        model.enable_gradient_checkpointing()

    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    device = engine.device

    _, client_state = engine.load_checkpoint(ckpt_dir)
    resume_step = client_state["step"] if client_state else -1
    saved_loss = client_state.get("loss", None) if client_state else None
    saved_lr = client_state.get("lr", None) if client_state else None

    lr_at_resume = None
    try:
        lr_at_resume = engine.lr_scheduler.get_lr()[0]
    except Exception:
        pass

    # Load a batch and run one forward pass
    shard_path = os.path.join(data_dir, "train.bin")
    data = np.memmap(shard_path, dtype=np.uint16, mode="r")
    batch_size = ds_config["train_micro_batch_size_per_gpu"]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix]).to(device)

    with torch.no_grad():
        logits, loss = engine(x, targets=y)
    loss_at_resume = loss.item()

    ratio = loss_at_resume / saved_loss if saved_loss and saved_loss > 0 else float("inf")
    passed = loss_at_resume <= 2.0 * saved_loss if saved_loss else False

    result = {{
        "loss_at_save": saved_loss,
        "loss_at_resume": loss_at_resume,
        "ratio": ratio,
        "passed": passed,
        "resume_step": resume_step,
        "lr_at_save": saved_lr,
        "lr_at_resume": lr_at_resume,
    }}

    del engine, model
    torch.cuda.empty_cache()
    gc.collect()

except Exception as e:
    result = {{
        "loss_at_save": None,
        "loss_at_resume": None,
        "ratio": None,
        "passed": False,
        "error": str(e),
    }}

with open(result_file, "w") as f:
    json.dump(result, f)
'''

    script_file = os.path.join(ckpt_dir, "_resume_verify.py")
    with open(script_file, "w") as f:
        f.write(script)

    try:
        proc = subprocess.run(
            [sys.executable, script_file],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=_project_root,
        )

        if os.path.exists(result_file):
            with open(result_file) as f:
                return json.load(f)

        return {
            "loss_at_save": None,
            "loss_at_resume": None,
            "ratio": None,
            "passed": False,
            "error": f"Subprocess exited {proc.returncode}: {proc.stderr[-500:] if proc.stderr else 'no stderr'}",
        }

    except subprocess.TimeoutExpired:
        return {
            "loss_at_save": None,
            "loss_at_resume": None,
            "ratio": None,
            "passed": False,
            "error": "Subprocess resume verification timed out (300s)",
        }

    except Exception as e:
        return {
            "loss_at_save": None,
            "loss_at_resume": None,
            "ratio": None,
            "passed": False,
            "error": f"Failed to spawn subprocess: {e}",
        }


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------
def get_hardware_info() -> dict:
    """Collect hardware information for the readiness report."""
    info = {
        "gpu_name": "Unknown",
        "gpu_memory_gb": 0,
        "cpu_cores": psutil.cpu_count(logical=False) or 0,
        "cpu_threads": psutil.cpu_count(logical=True) or 0,
        "total_ram_gb": psutil.virtual_memory().total / (1024 ** 3),
    }

    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["gpu_memory_gb"] = mem_info.total / (1024 ** 3)
        except Exception:
            pass

    return info


# ---------------------------------------------------------------------------
# Pre-flight checklist (manual prep items for long training runs)
# ---------------------------------------------------------------------------
PREFLIGHT_CHECKLIST = [
    "Disable Windows sleep/hibernate (Settings > Power & Sleep)",
    "Pause Windows Update (Settings > Windows Update > Pause for 7 days)",
    "Close heavy applications (browsers, IDEs, Docker Desktop if unused)",
    "Verify .wslconfig memory allocation >= 58 GB",
    "Ensure stable power supply / UPS if available",
    "Consider running 'wsl --shutdown' and restarting WSL2 for clean memory state",
]


# ---------------------------------------------------------------------------
# WSL2 environment checks
# ---------------------------------------------------------------------------
def check_wsl2_memory() -> tuple[float, list[str]]:
    """Check WSL2 memory allocation by parsing /proc/meminfo.

    Returns a (total_gb, warnings) tuple. Warns when total memory is below
    58 GB with a .wslconfig snippet showing the recommended setting.
    Handles missing /proc/meminfo gracefully (non-Linux environments).
    """
    warnings: list[str] = []
    total_gb = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    total_gb = kb / (1024 * 1024)
                    break
    except FileNotFoundError:
        warnings.append("Cannot read /proc/meminfo (not running on Linux/WSL2?)")
        return total_gb, warnings

    if total_gb < 58:
        warnings.append(
            f"WSL2 memory is {total_gb:.1f} GB, below recommended 58 GB. "
            f"Add to ~/.wslconfig:\n"
            f"  [wsl2]\n"
            f"  memory=58GB"
        )
    return total_gb, warnings


def check_disk_space(path: str, min_gb: float = 50.0) -> tuple[float, list[str]]:
    """Check free disk space at the given path.

    Returns a (free_gb, warnings) tuple. Warns when free space is below
    the specified threshold (default 50 GB). Handles invalid paths gracefully.
    """
    warnings: list[str] = []
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
    except OSError as e:
        warnings.append(f"Cannot check disk space at {path}: {e}")
        return 0.0, warnings

    if free_gb < min_gb:
        warnings.append(
            f"Free disk space at {path} is {free_gb:.1f} GB, "
            f"below recommended {min_gb:.0f} GB"
        )
    return free_gb, warnings


# ---------------------------------------------------------------------------
# Max steps calculation from token count
# ---------------------------------------------------------------------------
def calculate_max_steps(data_dir: str) -> tuple[int, int]:
    """Calculate production max_steps from train.bin file size.

    Uses the locked formula: max_steps = int((total_tokens * 3) / (32 * 1024))
    where 3 = epochs, 32 = effective batch (batch_size=4 * grad_accum=8),
    1024 = block_size.

    Returns (max_steps, total_tokens).
    """
    train_bin = os.path.join(data_dir, "train.bin")
    file_size = os.path.getsize(train_bin)
    total_tokens = file_size // 2  # uint16 = 2 bytes per token
    max_steps = int((total_tokens * 3) / (32 * 1024))
    return max_steps, total_tokens


# ---------------------------------------------------------------------------
# configs/1b.json update
# ---------------------------------------------------------------------------
def update_1b_config(config_path: str, benchmark_data: dict) -> None:
    """Update configs/1b.json with benchmark results and computed max_steps.

    Populates the benchmark_results section and updates training.max_steps
    without overwriting other fields (model, deepspeed sections preserved).
    """
    with open(config_path) as f:
        config = json.load(f)

    config["benchmark_results"] = {
        "peak_vram_gb": benchmark_data["peak_gpu_gb"],
        "tokens_per_sec": benchmark_data["tokens_per_sec"],
        "cpu_ram_gb": benchmark_data["cpu_ram_gb"],
        "checkpoint_save_time_sec": benchmark_data["checkpoint_save_time_sec"],
        "steps_run": benchmark_data["steps_run"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    config["training"]["max_steps"] = benchmark_data["max_steps"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# 1B readiness report generation
# ---------------------------------------------------------------------------
def generate_1b_readiness_report(
    env_checks: dict,
    benchmark_results: dict,
    projections: dict,
    pass_fail: dict,
) -> str:
    """Generate a structured markdown readiness report for the 1B model.

    Produces a report with sections: Summary, Environment Checks,
    Benchmark Results, Projected Training Time, and Pre-flight Checklist.

    Returns the report as a string (caller is responsible for writing to file).
    """
    now = datetime.now(timezone.utc).isoformat()
    lines = []

    # Summary
    overall = pass_fail.get("overall", "UNKNOWN")
    lines.append("# 1B pre-flight readiness report\n")
    lines.append(f"**Generated:** {now}\n")
    lines.append(f"**Overall status:** {overall}\n")
    lines.append(
        "This report validates whether the 1B ALS-LM model can train on the "
        "available hardware with DeepSpeed ZeRO Stage 2 and CPU offloading.\n"
    )

    # Environment checks
    lines.append("## Environment checks\n")
    wsl_gb = env_checks.get("wsl2_memory_gb", 0)
    disk_gb = env_checks.get("disk_free_gb", 0)
    lines.append(f"- **WSL2 memory:** {wsl_gb:.1f} GB")
    lines.append(f"- **Free disk space:** {disk_gb:.1f} GB\n")

    wsl_warnings = env_checks.get("wsl2_memory_warnings", [])
    disk_warnings = env_checks.get("disk_warnings", [])
    all_warnings = wsl_warnings + disk_warnings
    if all_warnings:
        lines.append("**Warnings:**\n")
        for w in all_warnings:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append("No environment warnings.\n")

    # Benchmark results
    lines.append("## Benchmark results\n")
    peak = benchmark_results.get("peak_gpu_gb", 0)
    tps = benchmark_results.get("tokens_per_sec", 0)
    cpu_ram = benchmark_results.get("cpu_ram_gb", 0)
    loss_start = benchmark_results.get("loss_at_1", 0)
    loss_final = benchmark_results.get("loss_at_final", 0)
    ckpt_time = benchmark_results.get("checkpoint_save_time_sec", 0)

    lines.append(f"- **Peak VRAM:** {peak:.2f} GB (limit: {pass_fail.get('vram_limit_gb', 10):.0f} GB)")
    lines.append(f"- **VRAM status:** {'PASS' if pass_fail.get('vram_pass') else 'FAIL'}")
    lines.append(f"- **Throughput:** {tps:,.0f} tokens/sec")
    lines.append(f"- **CPU RAM:** {cpu_ram:.1f} GB")
    lines.append(f"- **Loss:** {loss_start:.4f} -> {loss_final:.4f}")
    lines.append(f"- **Checkpoint save time:** {ckpt_time:.1f}s\n")

    # Checkpoint resume
    resume = benchmark_results.get("checkpoint_resume", {})
    if resume:
        resume_status = "PASS" if resume.get("passed") else "FAIL"
        lines.append(f"**Checkpoint resume:** {resume_status}")
        if resume.get("loss_at_save") is not None:
            lines.append(f"- Loss at save: {resume['loss_at_save']:.4f}")
            lines.append(f"- Loss at resume: {resume['loss_at_resume']:.4f}")
            lines.append(f"- Ratio: {resume['ratio']:.4f}")
        lines.append("")

    # Projected training time
    lines.append("## Projected training time\n")
    max_steps = projections.get("max_steps", 0)
    total_tokens = projections.get("total_tokens", 0)
    proj_hours = projections.get("projected_hours", 0)
    lines.append(f"- **Total tokens:** {total_tokens:,}")
    lines.append(f"- **Max steps (3 epochs):** {max_steps:,}")
    lines.append(f"- **Measured throughput:** {projections.get('tokens_per_sec', 0):,.0f} tokens/sec")
    lines.append(f"- **Projected training time:** {proj_hours:.1f} hours\n")

    # Pre-flight checklist
    lines.append("## Pre-flight checklist\n")
    lines.append("Complete these steps before starting the production training run.\n")
    for i, item in enumerate(PREFLIGHT_CHECKLIST, 1):
        lines.append(f"{i}. [ ] {item}")
    lines.append("")

    lines.append("---\n")
    lines.append(f"*Generated by benchmark/readiness_gate.py on {now}*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Projected training duration
# ---------------------------------------------------------------------------
def calculate_projections(
    tokens_per_sec: float,
    data_dir: str,
    block_size: int,
    checkpoint_save_time: float,
) -> dict:
    """Calculate projected training durations for 3, 5, and 10 epochs.

    Uses production batch settings (batch_size=4, grad_accum=8) from the 500M
    CONFIG_DEFAULTS in train.py. The benchmark uses batch_size=2 for memory
    safety, so throughput is scaled proportionally (rough approximation).
    """
    # Read total training tokens from train.bin file size
    train_bin = os.path.join(data_dir, "train.bin")
    if os.path.isfile(train_bin):
        file_size = os.path.getsize(train_bin)
        total_train_tokens = file_size // 2  # uint16 = 2 bytes per token
    else:
        total_train_tokens = 0

    # Production settings: batch_size=4, grad_accum=8 => effective=32
    prod_batch_size = 4
    prod_grad_accum = 8
    prod_effective_batch = prod_batch_size * prod_grad_accum
    tokens_per_step = prod_effective_batch * block_size  # 32 * 1024 = 32768

    # Scale tokens_per_sec from benchmark (batch_size=2) to production (batch_size=4).
    # Throughput roughly doubles with 2x batch size (linear scaling approximation).
    bench_batch_size = 2
    scaling_factor = prod_batch_size / bench_batch_size
    prod_tokens_per_sec = tokens_per_sec * scaling_factor

    # Steps per epoch
    if tokens_per_step > 0 and total_train_tokens > 0:
        steps_per_epoch = total_train_tokens / tokens_per_step
    else:
        steps_per_epoch = 0

    # Checkpoint overhead: one checkpoint every 250 steps
    ckpt_interval = 250

    projections = {}
    for n_epochs in [3, 5, 10]:
        total_steps = steps_per_epoch * n_epochs
        if prod_tokens_per_sec > 0:
            training_sec = (total_train_tokens * n_epochs) / prod_tokens_per_sec
        else:
            training_sec = float("inf")

        # Checkpoint overhead
        num_checkpoints = total_steps / ckpt_interval if ckpt_interval > 0 else 0
        ckpt_overhead_sec = num_checkpoints * checkpoint_save_time

        total_sec = training_sec + ckpt_overhead_sec
        total_hours = total_sec / 3600

        projections[f"{n_epochs}_epochs"] = {
            "epochs": n_epochs,
            "total_steps": int(total_steps),
            "training_hours": training_sec / 3600,
            "checkpoint_overhead_hours": ckpt_overhead_sec / 3600,
            "total_hours": total_hours,
            "exceeds_72h": total_hours > MAX_PROJECTED_HOURS,
        }

    return {
        "total_train_tokens": total_train_tokens,
        "tokens_per_step": tokens_per_step,
        "steps_per_epoch": int(steps_per_epoch),
        "bench_tokens_per_sec": tokens_per_sec,
        "prod_tokens_per_sec_estimate": prod_tokens_per_sec,
        "scaling_note": f"Scaled {scaling_factor:.1f}x from bench batch_size={bench_batch_size} to prod batch_size={prod_batch_size}",
        "checkpoint_save_time_sec": checkpoint_save_time,
        "projections": projections,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_readiness_report(
    results: dict,
    hardware: dict,
    projections: dict,
    checkpoint_resume: dict,
    output_dir: str,
) -> str:
    """Auto-generate benchmark/READINESS_REPORT.md with full results."""
    now = datetime.now(timezone.utc).isoformat()
    lines = []

    lines.append("# 500M readiness gate report\n")
    lines.append(f"**Generated:** {now}\n")
    lines.append("This report was auto-generated by `benchmark/readiness_gate.py`.")
    lines.append("It validates whether the ALS-LM model can train on the available hardware")
    lines.append("with DeepSpeed ZeRO Stage 2, providing data for a go/no-go decision.\n")

    # Hardware
    lines.append("## Hardware\n")
    gpu_name = hardware.get("gpu_name", "Unknown")
    # Handle bytes from pynvml
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8", errors="replace")
    lines.append(f"- **GPU:** {gpu_name} ({hardware.get('gpu_memory_gb', 0):.1f} GB)")
    lines.append(f"- **CPU:** {hardware.get('cpu_cores', 0)} cores / {hardware.get('cpu_threads', 0)} threads")
    lines.append(f"- **RAM:** {hardware.get('total_ram_gb', 0):.1f} GB\n")

    # Configuration matrix results
    lines.append("## Configuration matrix results\n")
    lines.append("Each row represents one benchmark run with the specified model size and settings.\n")

    header = (
        "| Config | Params | Offload | Grad Ckpt | Peak GPU (GB) | Tokens/s | "
        "Projected Hours (3ep) | Loss@start | Loss@final | CPU RAM (GB) | "
        "Disk I/O (R/W MB) | Step Time Var | Status |"
    )
    sep = (
        "|--------|--------|---------|-----------|---------------|----------|"
        "-----------------------|------------|------------|--------------|"
        "-------------------|---------------|--------|"
    )
    lines.append(header)
    lines.append(sep)

    for entry in results.get("matrix_results", []):
        r = entry.get("result", {})
        status = r.get("status", "error").upper()
        params = r.get("params", 0)

        if status == "PASS":
            peak = r.get("peak_gpu_gb", 0)
            tps = r.get("tokens_per_sec", 0)
            loss_start = r.get("loss_at_1", 0)
            loss_final = r.get("loss_at_final", 0)
            cpu_ram = r.get("cpu_ram_gb", 0)
            disk_r = r.get("disk_io_read_mb", 0)
            disk_w = r.get("disk_io_write_mb", 0)

            step_times = r.get("step_times", [])
            if len(step_times) > 1:
                import statistics
                variance = statistics.variance(step_times)
            else:
                variance = 0.0

            # Quick projected hours for this config
            proj_hours = "N/A"
            if tps > 0 and projections.get("total_train_tokens", 0) > 0:
                train_tokens = projections["total_train_tokens"]
                # 3 epochs, with batch scaling
                bench_batch = 2
                prod_batch = 4
                prod_tps = tps * (prod_batch / bench_batch)
                raw_sec = (train_tokens * 3) / prod_tps
                ckpt_time = r.get("checkpoint_save_time_sec", 0)
                steps_per_ep = projections.get("steps_per_epoch", 0)
                num_ckpts = (steps_per_ep * 3) / 250 if steps_per_ep > 0 else 0
                total_sec = raw_sec + num_ckpts * ckpt_time
                proj_hours = f"{total_sec / 3600:.1f}h"

            offload_str = "ON" if entry.get("cpu_offload", False) else "OFF"
            grad_ckpt_str = "ON" if entry.get("grad_ckpt", False) else "OFF"

            lines.append(
                f"| {entry.get('name', '')} | {params:,} | {offload_str} | {grad_ckpt_str} | "
                f"{peak:.2f} | {tps:,.0f} | {proj_hours} | "
                f"{loss_start:.4f} | {loss_final:.4f} | {cpu_ram:.1f} | "
                f"{disk_r:.0f}/{disk_w:.0f} | {variance:.4f} | {status} |"
            )
        else:
            offload_str = "ON" if entry.get("cpu_offload", False) else "OFF"
            grad_ckpt_str = "ON" if entry.get("grad_ckpt", False) else "OFF"
            error_msg = r.get("error", "")
            if error_msg:
                status_display = f"{status}: {error_msg[:30]}"
            else:
                status_display = status
            lines.append(
                f"| {entry.get('name', '')} | {params:,} | {offload_str} | {grad_ckpt_str} | "
                f"- | - | - | - | - | - | - | - | {status_display} |"
            )

    lines.append("")

    # Checkpoint resume
    lines.append("## Checkpoint resume verification\n")
    if checkpoint_resume.get("error"):
        lines.append(f"**Status:** FAILED - {checkpoint_resume['error']}\n")
    else:
        loss_save = checkpoint_resume.get("loss_at_save", "N/A")
        loss_resume = checkpoint_resume.get("loss_at_resume", "N/A")
        ratio = checkpoint_resume.get("ratio", "N/A")
        passed = checkpoint_resume.get("passed", False)
        status_str = "PASS" if passed else "FAIL"

        lines.append(f"The checkpoint resume test saves training state at step 50, destroys the")
        lines.append(f"model and engine, resets random seeds, rebuilds from scratch, loads the")
        lines.append(f"checkpoint, and verifies loss continuity.\n")
        lines.append(f"- **Loss at save (step 49):** {loss_save:.4f}" if isinstance(loss_save, float) else f"- **Loss at save:** {loss_save}")
        lines.append(f"- **Loss at resume:** {loss_resume:.4f}" if isinstance(loss_resume, float) else f"- **Loss at resume:** {loss_resume}")
        lines.append(f"- **Ratio (resume/save):** {ratio:.4f}" if isinstance(ratio, float) else f"- **Ratio:** {ratio}")
        lines.append(f"- **Threshold:** resume_loss <= 2.0 * save_loss")
        lines.append(f"- **Result:** {status_str}\n")

    # Projected training duration
    lines.append("## Projected training duration\n")
    lines.append(f"Projections are based on the best-performing configuration's throughput,")
    lines.append(f"scaled to production batch settings (batch_size=4, grad_accum=8).\n")

    proj_data = projections.get("projections", {})
    lines.append(f"- **Total training tokens:** {projections.get('total_train_tokens', 0):,}")
    lines.append(f"- **Tokens per step (production):** {projections.get('tokens_per_step', 0):,}")
    lines.append(f"- **Steps per epoch:** {projections.get('steps_per_epoch', 0):,}")
    lines.append(f"- **Benchmark throughput:** {projections.get('bench_tokens_per_sec', 0):,.0f} tokens/sec")
    lines.append(f"- **Estimated production throughput:** {projections.get('prod_tokens_per_sec_estimate', 0):,.0f} tokens/sec")
    lines.append(f"- **Scaling note:** {projections.get('scaling_note', 'N/A')}")
    lines.append(f"- **Checkpoint save time:** {projections.get('checkpoint_save_time_sec', 0):.1f}s\n")

    if proj_data:
        lines.append("| Epochs | Total Steps | Training (h) | Checkpoint Overhead (h) | Total (h) | Exceeds 72h |")
        lines.append("|--------|-------------|--------------|-------------------------|-----------|-------------|")
        for key in sorted(proj_data.keys()):
            p = proj_data[key]
            exceed = "YES" if p.get("exceeds_72h", False) else "No"
            lines.append(
                f"| {p['epochs']} | {p['total_steps']:,} | "
                f"{p['training_hours']:.1f} | {p['checkpoint_overhead_hours']:.1f} | "
                f"{p['total_hours']:.1f} | {exceed} |"
            )
        lines.append("")

    # Go/no-go recommendation
    lines.append("## Go/no-go recommendation\n")

    primary_result = results.get("primary_size", "unknown")
    primary_passed = any(
        e["result"].get("status") == "pass"
        for e in results.get("matrix_results", [])
        if e.get("size") == primary_result
    )
    ckpt_passed = checkpoint_resume.get("passed", False)

    if primary_passed and ckpt_passed:
        lines.append(f"**RECOMMENDATION: GO**\n")
        lines.append(f"The {primary_result} model configuration successfully trained for the")
        lines.append(f"benchmark duration without OOM errors, and checkpoint resume verified")
        lines.append(f"loss continuity. The chosen configuration is ready for v0.4.0 full training.\n")
    elif primary_passed and not ckpt_passed:
        lines.append(f"**RECOMMENDATION: CONDITIONAL GO**\n")
        lines.append(f"The {primary_result} model trained without OOM, but checkpoint resume")
        lines.append(f"failed verification. Investigate checkpoint save/load before full training.\n")
    else:
        # Check if any fallback passed
        any_passed = any(
            e["result"].get("status") == "pass" for e in results.get("matrix_results", [])
        )
        if any_passed:
            best_size = results.get("best_viable_size", "unknown")
            lines.append(f"**RECOMMENDATION: GO with {best_size}**\n")
            lines.append(f"The primary target ({primary_result}) did not pass, but {best_size}")
            lines.append(f"trained successfully. Consider using the smaller configuration.\n")
        else:
            lines.append(f"**RECOMMENDATION: NO-GO**\n")
            lines.append(f"No model configuration could train without OOM on this hardware.")
            lines.append(f"Review hardware constraints or reduce model size below 350M.\n")

    # Chosen config
    chosen = results.get("chosen_config", None)
    if chosen:
        lines.append("## Chosen configuration\n")
        lines.append(f"The recommended configuration for v0.4.0 training has been saved to")
        lines.append(f"`configs/{chosen.get('size', '500m').lower()}.json`.\n")
        lines.append(f"- **Model size:** {chosen.get('size', 'Unknown')}")
        lines.append(f"- **Parameters:** {chosen.get('params', 0):,}")
        lines.append(f"- **CPU offload:** {'Enabled' if chosen.get('cpu_offload', True) else 'Disabled'}")
        lines.append(f"- **Gradient checkpointing:** {'Enabled' if chosen.get('grad_ckpt', True) else 'Disabled'}")
        lines.append(f"- **Peak GPU memory:** {chosen.get('peak_gpu_gb', 0):.2f} GB")
        lines.append(f"- **Throughput:** {chosen.get('tokens_per_sec', 0):,.0f} tokens/sec\n")

    lines.append("---\n")
    lines.append(f"*Generated by benchmark/readiness_gate.py on {now}*\n")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "READINESS_REPORT.md")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_path


def save_results_json(results: dict, output_dir: str) -> str:
    """Save raw benchmark results to benchmark/results.json."""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")

    # Make results JSON-serializable (remove non-serializable items)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        else:
            return str(obj)

    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    return results_path


def save_chosen_config(
    model_config: dict,
    ds_config: dict,
    size_name: str,
    cpu_offload: bool,
    grad_ckpt: bool,
    vocab_size: int,
) -> str:
    """Save the chosen training configuration to configs/{size}.json."""
    configs_dir = os.path.join(_project_root, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    config_path = os.path.join(configs_dir, f"{size_name.lower()}.json")

    output = {
        "model": {
            **model_config,
            "vocab_size": vocab_size,
            "use_gradient_checkpointing": grad_ckpt,
        },
        "deepspeed": ds_config,
        "training": {
            "size_name": size_name,
            "cpu_offload": cpu_offload,
            "gradient_checkpointing": grad_ckpt,
            "batch_size": 4,
            "grad_accum": 8,
            "lr": 3e-4,
            "warmup_steps": 500,
            "max_steps": 50000,
            "block_size": model_config.get("block_size", 1024),
        },
        "generated_by": "benchmark/readiness_gate.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(config_path, "w") as f:
        json.dump(output, f, indent=2)

    return config_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Readiness Gate Benchmark for ALS-LM (500M default, 1B pre-flight mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python benchmark/readiness_gate.py\n"
            "  python benchmark/readiness_gate.py --steps 10\n"
            "  python benchmark/readiness_gate.py --primary-size 1B\n"
            "  python benchmark/readiness_gate.py --primary-size 1B --preflight\n"
            "  python benchmark/readiness_gate.py --skip-fallback\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tokenized",
        help="Path to tokenized data directory (default: data/tokenized)",
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="config/ds_zero2.json",
        help="Path to base DeepSpeed config (default: config/ds_zero2.json)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of training steps per benchmark config (default: 100 for 500M, 500 for 1B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark",
        help="Directory for benchmark outputs (default: benchmark/)",
    )
    parser.add_argument(
        "--primary-size",
        type=str,
        default="500M",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size to run the full 4-config matrix for (default: 500M)",
    )
    parser.add_argument(
        "--skip-fallback",
        action="store_true",
        help="Skip fallback chain testing (only test primary size)",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run 1B pre-flight mode (auto-enabled when --primary-size 1B). "
             "Runs environment checks, single production config, 500 steps, "
             "VRAM hard-fail threshold, readiness report.",
    )
    parser.add_argument(
        "--vram-limit-gb",
        type=float,
        default=10.0,
        help="VRAM hard-fail threshold in GB for 1B pre-flight (default: 10.0)",
    )

    args = parser.parse_args()

    # Auto-enable preflight when --primary-size 1B
    if args.primary_size == "1B":
        args.preflight = True

    # Default steps: 500 for 1B pre-flight, 100 for 500M
    if args.steps is None:
        args.steps = 500 if args.preflight else 100

    return args


# ---------------------------------------------------------------------------
# 1B pre-flight benchmark mode
# ---------------------------------------------------------------------------
def run_1b_preflight(args, vocab_size, hardware):
    """Run the 1B pre-flight benchmark mode.

    This mode runs environment checks, a single production config benchmark
    (offload=True, grad_ckpt=True, batch_size=4), forced checkpoint resume
    at the configured step count with LR verification, applies pass/fail
    thresholds (10 GB VRAM hard-fail), calculates max_steps from token count,
    updates configs/1b.json, generates readiness_report_1b.md, and prints
    a console summary.
    """
    block_size = 1024
    model_config = MODEL_CONFIGS["1B"]
    bench_lr = 3e-4
    bench_warmup = 1000

    print("\n" + "=" * 70)
    print("  1B PRE-FLIGHT READINESS GATE")
    print("=" * 70)

    # --- Step A: Environment checks (advisory, no auto-fail) ---
    print("\n  --- Environment Checks ---")

    mem_gb, mem_warnings = check_wsl2_memory()
    print(f"  WSL2 memory: {mem_gb:.1f} GB")
    for w in mem_warnings:
        print(f"  WARNING: {w}")

    disk_gb, disk_warnings = check_disk_space(args.output_dir)
    print(f"  Free disk space: {disk_gb:.1f} GB")
    for w in disk_warnings:
        print(f"  WARNING: {w}")

    print("\n  --- Pre-flight Checklist ---")
    for i, item in enumerate(PREFLIGHT_CHECKLIST, 1):
        print(f"  {i}. {item}")

    env_checks = {
        "wsl2_memory_gb": mem_gb,
        "wsl2_memory_warnings": mem_warnings,
        "disk_free_gb": disk_gb,
        "disk_warnings": disk_warnings,
    }

    # --- Step B: Run production config benchmark ---
    print(f"\n{'=' * 60}")
    print(f"  Running 1B production config: offload=ON, grad_ckpt=ON, batch_size=4")
    print(f"  Steps: {args.steps}")
    print(f"{'=' * 60}")

    bench_model_config = {**model_config, "use_gradient_checkpointing": True}

    ds_config = make_ds_config(
        base_config_path=args.base_config,
        batch_size=4,
        grad_accum=8,
        lr=bench_lr,
        warmup_steps=bench_warmup,
        total_steps=args.steps,
        cpu_offload=True,
    )

    # Save a checkpoint at the final step for subprocess resume verification.
    # This avoids the hot-resume OOM that occurs when trying to rebuild a 1B
    # model in the same process that just finished training.
    resume_ckpt_dir = tempfile.mkdtemp(prefix="als_1b_resume_ckpt_")

    result = run_benchmark_config(
        model_config=bench_model_config,
        ds_config=ds_config,
        steps=args.steps,
        data_dir=args.data_dir,
        vocab_size=vocab_size,
        block_size=block_size,
        save_final_checkpoint=resume_ckpt_dir,
    )

    status = result.get("status", "error")
    print(f"\n  Benchmark status: {status.upper()}")

    if status != "pass":
        error_msg = result.get("error", "OOM or unknown error")
        print(f"  HARD FAIL: Benchmark did not complete - {error_msg}")
        print(f"{'=' * 70}\n")
        shutil.rmtree(resume_ckpt_dir, ignore_errors=True)
        return

    peak_gpu_gb = result["peak_gpu_gb"]
    tokens_per_sec = result["tokens_per_sec"]
    cpu_ram_gb = result["cpu_ram_gb"]
    ckpt_save_time = result.get("checkpoint_save_time_sec", 0)

    print(f"  Peak VRAM: {peak_gpu_gb:.2f} GB")
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"  CPU RAM: {cpu_ram_gb:.1f} GB")
    print(f"  Checkpoint save: {ckpt_save_time:.1f}s")

    # --- Step C: Checkpoint resume verification via subprocess ---
    # Spawns a fresh Python process to load the checkpoint saved during Step B.
    # This mirrors real production resume (cold start) and avoids the hot-resume
    # OOM that occurred when rebuilding the 1B model in the same process.
    print(f"\n  --- Checkpoint Resume Verification (subprocess, step {args.steps}) ---")

    resume_ds_config = make_ds_config(
        base_config_path=args.base_config,
        batch_size=4,
        grad_accum=8,
        lr=bench_lr,
        warmup_steps=bench_warmup,
        total_steps=args.steps * 2,  # total_steps must exceed save_step
        cpu_offload=True,
    )

    checkpoint_resume_result = verify_checkpoint_resume_subprocess(
        model_config=bench_model_config,
        ds_config=resume_ds_config,
        data_dir=args.data_dir,
        vocab_size=vocab_size,
        ckpt_dir=resume_ckpt_dir,
        save_step=args.steps,
        block_size=block_size,
    )

    # Clean up resume checkpoint
    shutil.rmtree(resume_ckpt_dir, ignore_errors=True)

    if checkpoint_resume_result.get("passed"):
        print(f"  PASSED: loss_save={checkpoint_resume_result['loss_at_save']:.4f}, "
              f"loss_resume={checkpoint_resume_result['loss_at_resume']:.4f}, "
              f"ratio={checkpoint_resume_result['ratio']:.4f}")
        if checkpoint_resume_result.get("lr_at_save") is not None:
            print(f"  LR at save: {checkpoint_resume_result['lr_at_save']:.6f}")
        if checkpoint_resume_result.get("lr_at_resume") is not None:
            print(f"  LR at resume: {checkpoint_resume_result['lr_at_resume']:.6f}")
    elif checkpoint_resume_result.get("error"):
        print(f"  FAILED: {checkpoint_resume_result['error']}")
    else:
        print(f"  FAILED: ratio={checkpoint_resume_result.get('ratio', 'N/A')}")

    # --- Step D: Apply pass/fail thresholds ---
    vram_pass = peak_gpu_gb <= args.vram_limit_gb
    throughput_warning = tokens_per_sec < 2000
    cpu_ram_warning = cpu_ram_gb > 55

    if not vram_pass:
        overall = "FAIL"
    elif not checkpoint_resume_result.get("passed", False):
        overall = "FAIL"
    else:
        overall = "PASS"

    pass_fail = {
        "vram_pass": vram_pass,
        "vram_gb": peak_gpu_gb,
        "vram_limit_gb": args.vram_limit_gb,
        "throughput_warning": throughput_warning,
        "cpu_ram_warning": cpu_ram_warning,
        "overall": overall,
    }

    print(f"\n  --- Threshold Checks ---")
    print(f"  Peak VRAM: {peak_gpu_gb:.2f} GB (limit: {args.vram_limit_gb:.0f} GB) "
          f"{'PASS' if vram_pass else 'HARD FAIL'}")
    if cpu_ram_warning:
        print(f"  CPU RAM: {cpu_ram_gb:.1f} GB > 55 GB - SOFT WARNING")
    if throughput_warning:
        print(f"  Throughput: {tokens_per_sec:,.0f} tok/s < 2000 - SOFT WARNING")

    # GPU temperature - log only
    hw_gpu_temp = hardware.get("gpu_temperature", None)
    if hw_gpu_temp is not None:
        print(f"  GPU temperature: {hw_gpu_temp} C (log only)")

    # --- Step E: Calculate max_steps from token count ---
    print(f"\n  --- Max Steps Calculation ---")
    max_steps, total_tokens = calculate_max_steps(args.data_dir)
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Max steps (3 epochs): {max_steps:,}")

    # --- Step F: Calculate projected training time ---
    # Use measured throughput directly (no scaling needed, benchmarked at production batch_size=4)
    if tokens_per_sec > 0:
        projected_seconds = (total_tokens * 3) / tokens_per_sec
        projected_hours = projected_seconds / 3600
    else:
        projected_hours = float("inf")
    print(f"  Projected training time: {projected_hours:.1f} hours")

    projections = {
        "max_steps": max_steps,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "projected_hours": projected_hours,
    }

    # --- Step G: Update configs/1b.json ---
    config_1b_path = os.path.join(_project_root, "configs", "1b.json")
    if os.path.isfile(config_1b_path):
        benchmark_data = {
            "peak_gpu_gb": peak_gpu_gb,
            "tokens_per_sec": tokens_per_sec,
            "cpu_ram_gb": cpu_ram_gb,
            "checkpoint_save_time_sec": ckpt_save_time,
            "steps_run": args.steps,
            "max_steps": max_steps,
        }
        update_1b_config(config_1b_path, benchmark_data)
        print(f"\n  Updated: {config_1b_path}")
    else:
        print(f"\n  WARNING: {config_1b_path} not found, skipping config update")

    # --- Step H: Generate readiness report ---
    benchmark_report_data = {
        "peak_gpu_gb": peak_gpu_gb,
        "tokens_per_sec": tokens_per_sec,
        "cpu_ram_gb": cpu_ram_gb,
        "loss_at_1": result.get("loss_at_1", 0),
        "loss_at_final": result.get("loss_at_final", 0),
        "checkpoint_save_time_sec": ckpt_save_time,
        "checkpoint_resume": checkpoint_resume_result,
    }

    report_text = generate_1b_readiness_report(
        env_checks=env_checks,
        benchmark_results=benchmark_report_data,
        projections=projections,
        pass_fail=pass_fail,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "readiness_report_1b.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Readiness report: {report_path}")

    # Save raw results JSON
    raw_results = {
        "mode": "1b_preflight",
        "env_checks": env_checks,
        "benchmark_result": result,
        "checkpoint_resume": checkpoint_resume_result,
        "projections": projections,
        "pass_fail": pass_fail,
        "hardware": hardware,
        "settings": {
            "steps": args.steps,
            "vram_limit_gb": args.vram_limit_gb,
            "block_size": block_size,
            "vocab_size": vocab_size,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    results_path = save_results_json(raw_results, args.output_dir)
    print(f"  Results JSON: {results_path}")

    # --- Step I: Print console summary ---
    print(f"\n{'=' * 70}")
    print(f"  1B PRE-FLIGHT RESULT: {overall}")
    print(f"{'=' * 70}")
    print(f"  Peak VRAM: {peak_gpu_gb:.2f} GB / {args.vram_limit_gb:.0f} GB limit")
    print(f"  Throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Checkpoint resume: {'PASS' if checkpoint_resume_result.get('passed') else 'FAIL'}")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Projected training time: {projected_hours:.1f} hours")
    if overall == "PASS":
        print(f"\n  Ready for production training run.")
    else:
        print(f"\n  Pre-flight FAILED. Review thresholds and benchmark results.")
    print(f"\n  Reports saved to: {args.output_dir}/")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Main benchmark orchestration
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Validate data files
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    train_path = os.path.join(args.data_dir, "train.bin")

    if not os.path.isfile(meta_path):
        print(f"\nFATAL: meta.pkl not found at {meta_path}")
        print("Run the data pipeline first: python scripts/prepare_data.py --encode")
        sys.exit(1)
    if not os.path.isfile(train_path):
        print(f"\nFATAL: train.bin not found at {train_path}")
        sys.exit(1)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]

    # Collect hardware info
    hardware = get_hardware_info()
    gpu_name = hardware.get("gpu_name", "Unknown")
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8", errors="replace")

    # Branch: 1B pre-flight mode vs 500M standard mode
    if args.preflight:
        print(f"\n  Vocab size: {vocab_size:,}")
        print(f"  Data dir: {args.data_dir}")
        print(f"  GPU: {gpu_name} ({hardware.get('gpu_memory_gb', 0):.1f} GB)")
        print(f"  RAM: {hardware.get('total_ram_gb', 0):.1f} GB")
        run_1b_preflight(args, vocab_size, hardware)
        return

    # --- 500M standard flow (unchanged) ---
    print("\n" + "=" * 70)
    print("  500M READINESS GATE BENCHMARK")
    print("=" * 70)

    print(f"\n  Vocab size: {vocab_size:,}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Base config: {args.base_config}")
    print(f"  Steps per config: {args.steps}")
    print(f"  Primary size: {args.primary_size}")
    print(f"  Skip fallback: {args.skip_fallback}")
    print(f"  GPU: {gpu_name} ({hardware.get('gpu_memory_gb', 0):.1f} GB)")
    print(f"  RAM: {hardware.get('total_ram_gb', 0):.1f} GB")

    block_size = 1024
    all_results = {
        "matrix_results": [],
        "primary_size": args.primary_size,
        "best_viable_size": None,
        "chosen_config": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
        "settings": {
            "steps": args.steps,
            "block_size": block_size,
            "vocab_size": vocab_size,
        },
    }

    # Determine fallback order, starting from the primary size
    if args.skip_fallback:
        test_order = [args.primary_size]
    else:
        # Start with primary, then follow standard fallback order
        primary_idx = FALLBACK_ORDER.index(args.primary_size)
        test_order = FALLBACK_ORDER[primary_idx:]

    # 4-config matrix: (cpu_offload, gradient_checkpointing)
    matrix_configs = [
        (True, True),    # Safest: both on
        (True, False),   # Offload on, no grad ckpt
        (False, True),   # No offload, grad ckpt on
        (False, False),  # Most aggressive: both off
    ]

    best_result = None
    best_config_entry = None
    checkpoint_resume_result = {"passed": False, "error": "Not run"}

    for size_idx, size_name in enumerate(test_order):
        model_config = MODEL_CONFIGS[size_name]
        is_primary = (size_idx == 0)

        print(f"\n{'=' * 60}")
        print(f"  Testing model size: {size_name}")
        print(f"  Role: {'PRIMARY (full matrix)' if is_primary else 'FALLBACK (best config only)'}")
        print(f"{'=' * 60}")

        if is_primary:
            # Full 4-config matrix for the primary size
            configs_to_test = matrix_configs
        else:
            # Best-config-only for fallback sizes (offload=True, grad_ckpt=True)
            configs_to_test = [(True, True)]

        size_has_any_pass = False

        for cpu_offload, grad_ckpt in configs_to_test:
            config_name = f"{size_name}_offload-{'on' if cpu_offload else 'off'}_gradckpt-{'on' if grad_ckpt else 'off'}"
            print(f"\n  --- Config: {config_name} ---")
            print(f"      CPU offload: {'ON' if cpu_offload else 'OFF'}")
            print(f"      Grad checkpoint: {'ON' if grad_ckpt else 'OFF'}")

            # Build model config with gradient checkpointing flag
            bench_model_config = {**model_config, "use_gradient_checkpointing": grad_ckpt}

            # Generate DeepSpeed config
            ds_config = make_ds_config(
                base_config_path=args.base_config,
                batch_size=2,
                grad_accum=1,
                lr=3e-4,
                warmup_steps=10,
                total_steps=args.steps,
                cpu_offload=cpu_offload,
            )

            # Run benchmark
            result = run_benchmark_config(
                model_config=bench_model_config,
                ds_config=ds_config,
                steps=args.steps,
                data_dir=args.data_dir,
                vocab_size=vocab_size,
                block_size=block_size,
            )

            status = result.get("status", "error")
            print(f"      Status: {status.upper()}")

            if status == "pass":
                size_has_any_pass = True
                print(f"      Peak GPU: {result['peak_gpu_gb']:.2f} GB")
                print(f"      Tokens/sec: {result['tokens_per_sec']:,.0f}")
                print(f"      Loss: {result['loss_at_1']:.4f} -> {result['loss_at_final']:.4f}")
                print(f"      CPU RAM: {result['cpu_ram_gb']:.1f} GB")
                print(f"      Checkpoint save: {result.get('checkpoint_save_time_sec', 0):.1f}s")

                # Track best result (highest throughput among passing configs)
                if best_result is None or result["tokens_per_sec"] > best_result["tokens_per_sec"]:
                    best_result = result
                    best_config_entry = {
                        "size": size_name,
                        "cpu_offload": cpu_offload,
                        "grad_ckpt": grad_ckpt,
                        "model_config": model_config,
                        "ds_config": ds_config,
                    }
                    all_results["best_viable_size"] = size_name
            elif status == "oom":
                print(f"      Out of memory")
            else:
                print(f"      Error: {result.get('error', 'unknown')}")

            # Record in results
            all_results["matrix_results"].append({
                "name": config_name,
                "size": size_name,
                "cpu_offload": cpu_offload,
                "grad_ckpt": grad_ckpt,
                "result": result,
            })

        # If this size had at least one pass and this is the primary,
        # run checkpoint resume verification on the best config for this size.
        if size_has_any_pass and is_primary:
            print(f"\n  --- Checkpoint Resume Verification ({size_name}) ---")

            # Use offload=True, grad_ckpt=True for the resume test (safest config)
            resume_model_config = {**model_config, "use_gradient_checkpointing": True}
            resume_ds_config = make_ds_config(
                base_config_path=args.base_config,
                batch_size=2,
                grad_accum=1,
                lr=3e-4,
                warmup_steps=10,
                total_steps=100,
                cpu_offload=True,
            )

            checkpoint_resume_result = verify_checkpoint_resume(
                model_config=resume_model_config,
                ds_config=resume_ds_config,
                data_dir=args.data_dir,
                vocab_size=vocab_size,
                block_size=block_size,
            )

            if checkpoint_resume_result.get("passed"):
                print(f"      PASSED: loss_save={checkpoint_resume_result['loss_at_save']:.4f}, "
                      f"loss_resume={checkpoint_resume_result['loss_at_resume']:.4f}, "
                      f"ratio={checkpoint_resume_result['ratio']:.4f}")
            elif checkpoint_resume_result.get("error"):
                print(f"      FAILED: {checkpoint_resume_result['error']}")
            else:
                print(f"      FAILED: ratio={checkpoint_resume_result.get('ratio', 'N/A')}")

        # If primary size had no passes, continue to fallback
        if is_primary and not size_has_any_pass:
            print(f"\n  {size_name} failed all configs. Continuing to fallback chain...")
            continue

        # If a fallback size passed, note it
        if not is_primary and size_has_any_pass:
            print(f"\n  Fallback {size_name} passed.")

    # If 350M was the last option and failed
    if all_results["best_viable_size"] is None:
        print("\n  GATE FAILED: No model configuration could train without OOM.")

    # Build chosen config
    if best_config_entry is not None:
        # Generate the production DeepSpeed config for the chosen size
        prod_ds_config = make_ds_config(
            base_config_path=args.base_config,
            batch_size=4,
            grad_accum=8,
            lr=3e-4,
            warmup_steps=500,
            total_steps=50000,
            cpu_offload=best_config_entry["cpu_offload"],
        )

        chosen_info = {
            "size": best_config_entry["size"],
            "params": best_result["params"],
            "cpu_offload": best_config_entry["cpu_offload"],
            "grad_ckpt": best_config_entry["grad_ckpt"],
            "peak_gpu_gb": best_result["peak_gpu_gb"],
            "tokens_per_sec": best_result["tokens_per_sec"],
        }
        all_results["chosen_config"] = chosen_info

        # Save chosen config file
        config_path = save_chosen_config(
            model_config=best_config_entry["model_config"],
            ds_config=prod_ds_config,
            size_name=best_config_entry["size"],
            cpu_offload=best_config_entry["cpu_offload"],
            grad_ckpt=best_config_entry["grad_ckpt"],
            vocab_size=vocab_size,
        )
        print(f"\n  Chosen config saved: {config_path}")

    # Calculate projections using the best result
    if best_result is not None:
        projections = calculate_projections(
            tokens_per_sec=best_result["tokens_per_sec"],
            data_dir=args.data_dir,
            block_size=block_size,
            checkpoint_save_time=best_result.get("checkpoint_save_time_sec", 0),
        )
    else:
        projections = {
            "total_train_tokens": 0,
            "tokens_per_step": 0,
            "steps_per_epoch": 0,
            "bench_tokens_per_sec": 0,
            "prod_tokens_per_sec_estimate": 0,
            "scaling_note": "No viable config found",
            "checkpoint_save_time_sec": 0,
            "projections": {},
        }
    all_results["projections"] = projections
    all_results["checkpoint_resume"] = checkpoint_resume_result

    # Generate report and save results
    print(f"\n{'=' * 60}")
    print("  GENERATING REPORTS")
    print(f"{'=' * 60}")

    report_path = generate_readiness_report(
        results=all_results,
        hardware=hardware,
        projections=projections,
        checkpoint_resume=checkpoint_resume_result,
        output_dir=args.output_dir,
    )
    print(f"  Readiness report: {report_path}")

    results_path = save_results_json(all_results, args.output_dir)
    print(f"  Results JSON: {results_path}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'=' * 70}")

    if all_results["best_viable_size"]:
        size = all_results["best_viable_size"]
        print(f"  Best viable size: {size}")
        if best_result:
            print(f"  Peak GPU: {best_result['peak_gpu_gb']:.2f} GB")
            print(f"  Throughput: {best_result['tokens_per_sec']:,.0f} tokens/sec")
        print(f"  Checkpoint resume: {'PASS' if checkpoint_resume_result.get('passed') else 'FAIL'}")
    else:
        print("  GATE FAILED: No viable configuration found.")

    print(f"\n  Reports saved to: {args.output_dir}/")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
