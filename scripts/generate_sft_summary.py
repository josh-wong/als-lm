#!/usr/bin/env python3
"""Generate SFT training summary report from training logs and verification results.

Reads sft_log.jsonl, optional verify_results.json, and training artifacts to
produce a standalone Markdown summary documenting the SFT training outcome
including configuration, early stopping, base model comparison, qualitative
verification, and training performance.

This is a research documentation tool, not a medical information system.

Usage:
    python scripts/generate_sft_summary.py logs/sft_log.jsonl
    python scripts/generate_sft_summary.py logs/sft_log.jsonl --output-dir results/sft/
    python scripts/generate_sft_summary.py logs/sft_log.jsonl --verify-results results/sft/verify_results.json
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root discovery (standalone script pattern)
# ---------------------------------------------------------------------------

_project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Default paths relative to project root
_DEFAULT_OUTPUT_DIR = str(_project_root / "results" / "sft")
_DEFAULT_VERIFY_RESULTS = str(_project_root / "results" / "sft" / "verify_results.json")

# 1B base model val loss for comparison (default from training_summary_1B.md)
_DEFAULT_BASE_VAL_LOSS = 5.6424

# SFT config defaults (from CONFIG_DEFAULTS["1B-sft"] in train.py)
_SFT_CONFIG = {
    "lr": "2e-5",
    "warmup": "50 steps",
    "max_epochs": "3",
    "batch_size": "2",
    "grad_accum": "4 (effective batch 8)",
    "block_size": "1024",
    "dropout": "0.1",
}


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_sft_log(log_path: str) -> dict:
    """Parse sft_log.jsonl and extract training metrics.

    Reads the JSONL log produced by train.py --sft and extracts summary
    metrics including epoch boundaries, best validation loss, early stopping
    status, average throughput, and peak GPU memory.

    Args:
        log_path: Path to sft_log.jsonl.

    Returns:
        Dict with keys: total_steps, epochs_completed, best_val_loss,
        best_epoch, early_stopped, stopped_epoch, avg_tokens_per_sec,
        peak_gpu_mem_mb, final_train_loss, val_losses.
    """
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        return {
            "total_steps": 0,
            "epochs_completed": 0,
            "best_val_loss": None,
            "best_epoch": None,
            "early_stopped": False,
            "stopped_epoch": None,
            "avg_tokens_per_sec": 0,
            "peak_gpu_mem_mb": 0,
            "final_train_loss": None,
            "val_losses": [],
        }

    # Total steps: find the highest step value across all entries
    total_steps = max((e.get("step", 0) for e in entries), default=0)

    # Final train loss: look in validation entries or step entries
    final_train_loss = None
    for e in reversed(entries):
        if e.get("train_loss") is not None:
            final_train_loss = e["train_loss"]
            break
        if e.get("type") == "step" and e.get("loss") is not None:
            final_train_loss = e["loss"]
            break

    # Extract actual warmup from config entry if available
    config_warmup = None
    for e in entries:
        if e.get("type") == "config":
            args_dict = e.get("training_args", {})
            config_warmup = args_dict.get("warmup_steps")
            break

    # Epoch boundaries: entries with type=="epoch" (logged by train.py --sft)
    epoch_boundaries = [
        e for e in entries
        if e.get("type") == "epoch" or e.get("sft_epoch_boundary")
    ]
    epochs_completed = len(epoch_boundaries)

    # Collect per-epoch val losses from epoch boundary entries
    val_losses = []
    for eb in epoch_boundaries:
        vl = eb.get("val_loss")
        if vl is not None:
            val_losses.append(vl)

    # Also collect val losses from validation entries (train.py logs these
    # separately when eval_interval triggers or at max_steps)
    for e in entries:
        if e.get("type") == "validation" and e.get("val_loss") is not None:
            vl = e["val_loss"]
            # Avoid duplicates
            if not any(abs(vl - existing) < 1e-8 for existing in val_losses):
                val_losses.append(vl)

    # Best val loss: check best_update entries first, then sft_best_val_loss
    best_val_loss = None
    best_epoch = None
    for e in reversed(entries):
        if e.get("type") == "best_update" and e.get("val_loss") is not None:
            best_val_loss = e["val_loss"]
            break
        if "sft_best_val_loss" in e and e["sft_best_val_loss"] is not None:
            best_val_loss = e["sft_best_val_loss"]
            break

    # Fallback: use the minimum val loss from collected losses
    if best_val_loss is None and val_losses:
        best_val_loss = min(val_losses)

    # Find which epoch had the best val loss
    if best_val_loss is not None and val_losses:
        for i, vl in enumerate(val_losses):
            if abs(vl - best_val_loss) < 1e-6:
                best_epoch = i + 1
                break
        if best_epoch is None:
            # Fallback: use the epoch with the lowest val loss
            best_epoch = val_losses.index(min(val_losses)) + 1

    # If no val_losses but we have best_val_loss, assign best_epoch=1
    if best_val_loss is not None and best_epoch is None:
        best_epoch = epochs_completed if epochs_completed > 0 else 1

    # Early stopping: check for sft_early_stop field or complete entry
    # with fewer epochs than max_epochs
    early_stopped = any(e.get("sft_early_stop") for e in entries)
    stopped_epoch = None
    if early_stopped:
        for e in entries:
            if e.get("sft_early_stop"):
                stopped_epoch = e.get("epoch", epochs_completed)
                break

    # Tokens/sec average
    tps_values = [
        e["tokens_per_sec"] for e in entries
        if "tokens_per_sec" in e and e["tokens_per_sec"] is not None
    ]
    avg_tokens_per_sec = sum(tps_values) / len(tps_values) if tps_values else 0

    # Peak GPU memory
    gpu_values = [
        e["gpu_mem_mb"] for e in entries
        if "gpu_mem_mb" in e and e["gpu_mem_mb"] is not None
    ]
    peak_gpu_mem_mb = max(gpu_values) if gpu_values else 0

    return {
        "total_steps": total_steps,
        "epochs_completed": epochs_completed,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "stopped_epoch": stopped_epoch,
        "avg_tokens_per_sec": round(avg_tokens_per_sec, 1),
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "final_train_loss": final_train_loss,
        "val_losses": val_losses,
        "config_warmup": config_warmup,
    }


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _find_checkpoint_label(checkpoint_dir: str = None) -> str:
    """Discover the SFT checkpoint run label (e.g., 1B-sft_20260320_143000).

    Searches for the latest 1B-sft_* directory under checkpoints/ or the
    provided checkpoint_dir.

    Returns:
        The directory name, or "unknown" if not found.
    """
    if checkpoint_dir is None:
        checkpoint_dir = str(_project_root / "checkpoints")

    pattern = os.path.join(checkpoint_dir, "1B-sft_*")
    matches = sorted(glob.glob(pattern))
    if matches:
        return os.path.basename(matches[-1])
    return "unknown"


def _count_dataset_examples(data_dir: str = None) -> dict:
    """Count train/val examples from tokenized memmap files.

    Each example is block_size=1024 tokens stored as uint16 (2 bytes per
    token), so examples = file_bytes / (1024 * 2).

    Returns:
        Dict with train_examples and val_examples counts, or 0 if files
        not found.
    """
    if data_dir is None:
        data_dir = str(_project_root / "data" / "instruction" / "tokenized")

    result = {"train_examples": 0, "val_examples": 0}

    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    bytes_per_example = 1024 * 2  # block_size=1024, uint16=2 bytes

    if os.path.isfile(train_path):
        result["train_examples"] = os.path.getsize(train_path) // bytes_per_example
    if os.path.isfile(val_path):
        result["val_examples"] = os.path.getsize(val_path) // bytes_per_example

    return result


def generate_summary(
    parsed_log: dict,
    verify_results: dict = None,
    checkpoint_label: str = "unknown",
    dataset_counts: dict = None,
    base_model_path: str = "checkpoints/1B_*/best/best.pt",
    base_val_loss: float = _DEFAULT_BASE_VAL_LOSS,
) -> str:
    """Generate SFT summary Markdown report.

    Produces a standalone Markdown document with the following sections:
    training configuration, early stopping, base model comparison,
    qualitative verification, and training performance.

    Args:
        parsed_log: Output of parse_sft_log().
        verify_results: Parsed verify_results.json dict, or None.
        checkpoint_label: SFT checkpoint directory name.
        dataset_counts: Dict with train_examples and val_examples.
        base_model_path: Path to the base model checkpoint.

    Returns:
        Markdown string.
    """
    if dataset_counts is None:
        dataset_counts = {"train_examples": "N/A", "val_examples": "N/A"}

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = []
    lines.append(f"# SFT training summary")
    lines.append("")
    lines.append(f"**Run:** {checkpoint_label}")
    lines.append(f"**Date:** {today}")
    lines.append("")

    # Training configuration
    lines.append("## Training configuration")
    lines.append("")
    lines.append("Training parameters used for supervised fine-tuning of the 1B base model.")
    lines.append("")
    lines.append("| Parameter            | Value                                      |")
    lines.append("|----------------------|--------------------------------------------|")
    lines.append(f"| Base model           | {base_model_path}                          |")
    lines.append(f"| Training examples    | {dataset_counts['train_examples']}         |")
    lines.append(f"| Validation examples  | {dataset_counts['val_examples']}           |")
    warmup_display = (
        f"{parsed_log['config_warmup']} steps"
        if parsed_log.get("config_warmup") is not None
        else _SFT_CONFIG["warmup"]
    )
    lines.append(f"| Learning rate        | {_SFT_CONFIG['lr']}                        |")
    lines.append(f"| Warmup               | {warmup_display}                           |")
    lines.append(f"| Max epochs           | {_SFT_CONFIG['max_epochs']}                |")
    lines.append(f"| Batch size           | {_SFT_CONFIG['batch_size']}                |")
    lines.append(f"| Gradient accumulation| {_SFT_CONFIG['grad_accum']}                |")
    lines.append(f"| Block size           | {_SFT_CONFIG['block_size']}                |")
    lines.append(f"| Dropout              | {_SFT_CONFIG['dropout']}                   |")
    lines.append("")

    # Early stopping
    lines.append("## Early stopping")
    lines.append("")
    if parsed_log["early_stopped"]:
        stopped_at = parsed_log.get("stopped_epoch", "unknown")
        lines.append(
            f"Early stopping triggered at epoch {stopped_at} of "
            f"{_SFT_CONFIG['max_epochs']} (2-epoch patience)."
        )
    else:
        lines.append(
            f"Completed all {parsed_log['epochs_completed']} epochs "
            f"(no early stopping triggered)."
        )
    lines.append("")

    if parsed_log["best_val_loss"] is not None:
        lines.append(
            f"**Best checkpoint:** epoch {parsed_log['best_epoch']}, "
            f"val_loss = {parsed_log['best_val_loss']:.4f}"
        )
    lines.append("")

    # Val loss per epoch
    if parsed_log["val_losses"]:
        lines.append("**Validation loss per epoch:**")
        lines.append("")
        for i, vl in enumerate(parsed_log["val_losses"]):
            marker = " (best)" if parsed_log["best_epoch"] == i + 1 else ""
            lines.append(f"- Epoch {i + 1}: {vl:.4f}{marker}")
        lines.append("")

    # Base model comparison
    lines.append("## Base model comparison")
    lines.append("")
    lines.append(
        "Comparison of validation loss between the 1B base model and the "
        "SFT model. These are computed on different data distributions "
        "(raw corpus vs. instruction pairs) and are not directly comparable."
    )
    lines.append("")
    lines.append("| Metric    | 1B base  | SFT      | Note                              |")
    lines.append("|-----------|----------|----------|-----------------------------------|")

    sft_val = (
        f"{parsed_log['best_val_loss']:.4f}"
        if parsed_log["best_val_loss"] is not None
        else "N/A"
    )
    lines.append(
        f"| Val loss  | {base_val_loss:.4f} | {sft_val}  "
        f"| Different data, not comparable     |"
    )
    lines.append("")

    # Qualitative verification
    lines.append("## Qualitative verification")
    lines.append("")
    if verify_results is not None:
        summary = verify_results.get("summary", {})
        coherent = summary.get("coherent_count", 0)
        total = summary.get("total_count", 0)
        verdict = summary.get("verdict", "N/A")
        lines.append(
            f"Spot-check result: **{coherent}/{total}** coherent responses "
            f"(verdict: **{verdict}**)."
        )
    else:
        lines.append("Not yet run. Execute `python scripts/verify_sft.py` after "
                      "training and GGUF export.")
    lines.append("")

    # Training performance
    lines.append("## Training performance")
    lines.append("")
    lines.append(
        "Runtime metrics from the SFT training run."
    )
    lines.append("")
    lines.append(f"- **Total steps:** {parsed_log['total_steps']:,}")
    lines.append(f"- **Epochs completed:** {parsed_log['epochs_completed']}")
    lines.append(f"- **Average tokens/sec:** {parsed_log['avg_tokens_per_sec']:,.1f}")
    lines.append(f"- **Peak GPU memory:** {parsed_log['peak_gpu_mem_mb']:,} MB")

    if parsed_log["final_train_loss"] is not None:
        lines.append(f"- **Final train loss:** {parsed_log['final_train_loss']:.4f}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SFT training summary report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/generate_sft_summary.py logs/sft_log.jsonl\n"
            "  python scripts/generate_sft_summary.py logs/sft_log.jsonl "
            "--output-dir results/sft/\n"
        ),
    )
    parser.add_argument(
        "sft_log",
        help="Path to sft_log.jsonl training log",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for sft_summary.md (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--verify-results",
        default=_DEFAULT_VERIFY_RESULTS,
        help=f"Path to verify_results.json (default: {_DEFAULT_VERIFY_RESULTS})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint base directory to auto-discover SFT run label",
    )
    parser.add_argument(
        "--base-val-loss",
        type=float,
        default=_DEFAULT_BASE_VAL_LOSS,
        help=f"1B base model val loss for comparison (default: {_DEFAULT_BASE_VAL_LOSS})",
    )
    return parser.parse_args()


def main():
    """Generate the SFT training summary report."""
    args = parse_args()

    print("\n=== SFT Summary Report Generator ===\n")

    # Parse training log
    print(f"  Reading log: {args.sft_log}")
    parsed = parse_sft_log(args.sft_log)
    print(f"  Total steps: {parsed['total_steps']:,}")
    print(f"  Epochs completed: {parsed['epochs_completed']}")
    print(f"  Best val loss: {parsed['best_val_loss']}")
    print(f"  Early stopped: {parsed['early_stopped']}")

    # Load verify results if available
    verify_results = None
    if os.path.isfile(args.verify_results):
        print(f"  Loading verify results: {args.verify_results}")
        with open(args.verify_results) as f:
            verify_results = json.load(f)
    else:
        print(f"  Verify results not found: {args.verify_results}")

    # Discover checkpoint label
    checkpoint_label = _find_checkpoint_label(args.checkpoint_dir)
    print(f"  Checkpoint label: {checkpoint_label}")

    # Count dataset examples
    dataset_counts = _count_dataset_examples()
    print(f"  Dataset: {dataset_counts['train_examples']} train, "
          f"{dataset_counts['val_examples']} val")

    # Generate summary
    md = generate_summary(
        parsed,
        verify_results=verify_results,
        checkpoint_label=checkpoint_label,
        dataset_counts=dataset_counts,
        base_val_loss=args.base_val_loss,
    )

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "sft_summary.md")
    with open(output_path, "w") as f:
        f.write(md)

    print(f"\n  Summary written to: {output_path}")
    print()


if __name__ == "__main__":
    main()
