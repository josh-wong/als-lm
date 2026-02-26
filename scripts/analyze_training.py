#!/usr/bin/env python3
"""
Overfitting analysis script for ALS-LM training logs.

Reads a JSONL training log and produces:
  - loss_curves.png: Train vs validation loss with epoch markers
  - perplexity_gap.png: Train vs validation perplexity divergence
  - lr_schedule.png: Learning rate over training steps
  - analysis_report.md: Narrative Markdown report with overfitting diagnosis

Supports both the old (tiny model) and enriched (Phase 5+) log formats.

Usage:
    python scripts/analyze_training.py <path/to/log.jsonl>
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Headless backend, save-only
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERPLEXITY_CAP = 1e5

# Overfitting classification thresholds (relative gap)
THRESHOLD_UNDERFIT = -0.02
THRESHOLD_WELLFIT = 0.02
THRESHOLD_MILD = 0.10

# Underfitting convergence check: 5% drop threshold
CONVERGENCE_DROP_THRESHOLD = 0.05

# Plot styling
FIGURE_SIZE = (10, 6)
DPI = 300
TRAIN_COLOR = "#1f77b4"   # Blue — distinct in grayscale
VAL_COLOR = "#d62728"     # Red — distinct in grayscale
LR_COLOR = "#2ca02c"      # Green
GRID_ALPHA = 0.3
EPOCH_LINE_COLOR = "#888888"
EPOCH_LINE_ALPHA = 0.5


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def parse_log(filepath: str) -> dict[str, Any]:
    """Parse a JSONL training log, handling both old and enriched formats.

    Returns a dict with keys: config, steps, validations, warnings, format.
    """
    config: Optional[dict] = None
    steps: list[dict] = []
    validations: list[dict] = []
    total_lines = 0
    malformed_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            total_lines += 1
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue

            entry_type = entry.get("type")

            if entry_type == "config":
                config = entry
            elif entry_type == "step":
                steps.append(entry)
            elif entry_type == "validation":
                validations.append(entry)
            # Silently skip unknown entry types

    # Detect format: enriched if step entries contain 'epoch' field
    is_enriched = len(steps) > 0 and "epoch" in steps[0]

    # Derive missing fields for old format
    if not is_enriched:
        _enrich_old_format(steps, validations, config)

    return {
        "config": config,
        "steps": steps,
        "validations": validations,
        "total_lines": total_lines,
        "malformed_count": malformed_count,
        "format": "enriched" if is_enriched else "old",
    }


def _enrich_old_format(
    steps: list[dict],
    validations: list[dict],
    config: Optional[dict],
) -> None:
    """Compute derived values for the old log format in place."""
    # Add perplexity to step entries
    for s in steps:
        loss = s.get("loss", 0.0)
        s["perplexity"] = min(math.exp(loss), PERPLEXITY_CAP)

    # Add derived fields to validation entries
    for v in validations:
        train_loss = v.get("train_loss", 0.0)
        val_loss = v.get("val_loss", 0.0)
        v["train_perplexity"] = min(math.exp(train_loss), PERPLEXITY_CAP)
        v["val_perplexity"] = min(math.exp(val_loss), PERPLEXITY_CAP)
        v["generalization_gap"] = val_loss - train_loss
        if train_loss > 0:
            v["gap_ratio"] = (val_loss - train_loss) / train_loss
        else:
            v["gap_ratio"] = 0.0

    # Attempt epoch inference from config
    if config:
        training_args = config.get("training_args", {})
        data_info = config.get("data_info", {})
        max_steps = training_args.get("max_steps")
        train_tokens = data_info.get("train_tokens")
        batch_size = training_args.get("batch_size", 16)
        grad_accum = training_args.get("grad_accum", 1)
        # Block size (sequence length) from model config
        block_size = config.get("model_config", {}).get("block_size", 1024)

        if max_steps and train_tokens:
            tokens_per_step = batch_size * grad_accum * block_size
            tokens_per_epoch = train_tokens
            steps_per_epoch = tokens_per_epoch / tokens_per_step if tokens_per_step > 0 else None

            if steps_per_epoch and steps_per_epoch > 0:
                for s in steps:
                    step_num = s.get("step", 0)
                    s["epoch"] = int(step_num / steps_per_epoch)
                    s["epoch_progress"] = (step_num % steps_per_epoch) / steps_per_epoch

                for v in validations:
                    step_num = v.get("step", 0)
                    v["epoch"] = int(step_num / steps_per_epoch)
                    v["epoch_progress"] = (step_num % steps_per_epoch) / steps_per_epoch


# ---------------------------------------------------------------------------
# Overfitting classification
# ---------------------------------------------------------------------------

def classify_overfitting(gap_ratio: float) -> str:
    """Classify overfitting based on relative gap metric."""
    if gap_ratio < THRESHOLD_UNDERFIT:
        return "Underfitting"
    elif gap_ratio < THRESHOLD_WELLFIT:
        return "Well-fit"
    elif gap_ratio < THRESHOLD_MILD:
        return "Mild overfitting"
    else:
        return "Severe overfitting"


def detect_convergence_issues(validations: list[dict]) -> dict[str, Any]:
    """Detect underfitting (still converging) and validation divergence."""
    issues: dict[str, Any] = {
        "still_converging": False,
        "val_divergence": False,
        "divergence_start_step": None,
    }

    if len(validations) < 2:
        return issues

    # Check if train loss still dropping significantly at end
    last = validations[-1]
    prev = validations[-2]
    last_train = last.get("train_loss", 0.0)
    prev_train = prev.get("train_loss", 0.0)
    if prev_train > 0:
        drop = (prev_train - last_train) / prev_train
        if drop > CONVERGENCE_DROP_THRESHOLD:
            issues["still_converging"] = True

    # Check for validation loss divergence: val increasing while train decreasing
    # for 2+ consecutive validation entries
    consecutive_divergence = 0
    for i in range(1, len(validations)):
        curr = validations[i]
        prev_v = validations[i - 1]
        val_increasing = curr.get("val_loss", 0) > prev_v.get("val_loss", 0)
        train_decreasing = curr.get("train_loss", 0) < prev_v.get("train_loss", 0)
        if val_increasing and train_decreasing:
            consecutive_divergence += 1
            if consecutive_divergence >= 2 and not issues["val_divergence"]:
                issues["val_divergence"] = True
                issues["divergence_start_step"] = validations[i - 1].get("step")
        else:
            consecutive_divergence = 0

    return issues


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent academic styling to an axes object."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_epoch_markers(
    ax: plt.Axes,
    steps: list[dict],
    y_max: float,
) -> None:
    """Add vertical dashed lines at epoch boundaries."""
    if not steps or "epoch" not in steps[0]:
        return

    seen_epochs: set[int] = set()
    prev_epoch = -1
    for s in steps:
        epoch = s.get("epoch", 0)
        step_num = s.get("step", 0)
        if epoch != prev_epoch and epoch > 0 and epoch not in seen_epochs:
            ax.axvline(
                x=step_num,
                color=EPOCH_LINE_COLOR,
                linestyle="--",
                linewidth=0.8,
                alpha=EPOCH_LINE_ALPHA,
            )
            ax.text(
                step_num,
                y_max * 0.98,
                f"Epoch {epoch}",
                fontsize=7,
                ha="center",
                va="top",
                color=EPOCH_LINE_COLOR,
                alpha=0.7,
            )
            seen_epochs.add(epoch)
        prev_epoch = epoch


def plot_loss_curves(
    steps: list[dict],
    validations: list[dict],
    output_path: str,
) -> None:
    """Generate loss curves plot: train loss (step-level) and val loss."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Train loss from step entries
    train_steps = [s["step"] for s in steps]
    train_losses = [s["loss"] for s in steps]
    ax.plot(
        train_steps, train_losses,
        color=TRAIN_COLOR, linewidth=1.2, alpha=0.8, label="Train Loss",
    )

    # Validation loss at validation intervals
    val_steps = [v["step"] for v in validations]
    val_losses = [v["val_loss"] for v in validations]
    ax.plot(
        val_steps, val_losses,
        color=VAL_COLOR, linewidth=1.5, marker="o", markersize=4,
        label="Val Loss",
    )

    # Epoch markers
    all_losses = train_losses + val_losses
    y_max = max(all_losses) if all_losses else 10.0
    _add_epoch_markers(ax, steps, y_max)

    _setup_axes(ax, "Training and Validation Loss", "Training Step", "Loss")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_perplexity_gap(
    validations: list[dict],
    output_path: str,
) -> None:
    """Generate perplexity gap plot: train vs val perplexity at validation points."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    val_steps = [v["step"] for v in validations]
    train_ppl = [v.get("train_perplexity", 0.0) for v in validations]
    val_ppl = [v.get("val_perplexity", 0.0) for v in validations]

    ax.plot(
        val_steps, train_ppl,
        color=TRAIN_COLOR, linewidth=1.5, marker="s", markersize=4,
        label="Train Perplexity",
    )
    ax.plot(
        val_steps, val_ppl,
        color=VAL_COLOR, linewidth=1.5, marker="o", markersize=4,
        label="Val Perplexity",
    )

    # Shaded region between curves to visualize the gap
    ax.fill_between(
        val_steps, train_ppl, val_ppl,
        alpha=0.1, color=VAL_COLOR,
    )

    _setup_axes(
        ax, "Train vs Validation Perplexity",
        "Training Step", "Perplexity",
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_lr_schedule(
    steps: list[dict],
    output_path: str,
) -> None:
    """Generate learning rate schedule plot."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    step_nums = [s["step"] for s in steps]
    lrs = [s.get("lr", 0.0) for s in steps]

    ax.plot(
        step_nums, lrs,
        color=LR_COLOR, linewidth=1.5,
    )

    _setup_axes(
        ax, "Learning Rate Schedule",
        "Training Step", "Learning Rate",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _format_model_summary(config: Optional[dict]) -> str:
    """Build a short model summary from the config entry."""
    if not config:
        return "Unknown"

    mc = config.get("model_config", {})
    parts = []
    if mc.get("n_layer"):
        parts.append(f"{mc['n_layer']}L")
    if mc.get("n_head"):
        parts.append(f"{mc['n_head']}H")
    if mc.get("n_embd"):
        parts.append(f"{mc['n_embd']}D")
    if mc.get("block_size"):
        parts.append(f"ctx={mc['block_size']}")
    if mc.get("dropout") is not None:
        parts.append(f"dropout={mc['dropout']}")

    return ", ".join(parts) if parts else "Unknown"


def _format_training_time(steps: list[dict]) -> str:
    """Compute approximate wall-clock training time from timestamps."""
    if len(steps) < 2:
        return "N/A"

    first_ts = steps[0].get("timestamp", "")
    last_ts = steps[-1].get("timestamp", "")
    if not first_ts or not last_ts:
        return "N/A"

    try:
        # Handle timezone-aware ISO timestamps
        t0 = datetime.fromisoformat(first_ts)
        t1 = datetime.fromisoformat(last_ts)
        delta = t1 - t0
        total_seconds = int(delta.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m {total_seconds % 60}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    except (ValueError, TypeError):
        return "N/A"


def _build_epoch_table(validations: list[dict]) -> str:
    """Build a per-epoch (or per-validation-interval) breakdown table."""
    if not validations:
        return "_No validation entries available for breakdown table._\n"

    has_epoch = "epoch" in validations[0]
    label_col = "Epoch" if has_epoch else "Step"

    header = (
        f"| {label_col:>6} | Train Loss | Val Loss | Relative Gap | "
        f"Train PPL | Val PPL | Classification |\n"
    )
    separator = (
        f"|{'-' * 8}|{'-' * 12}|{'-' * 10}|{'-' * 14}|"
        f"{'-' * 11}|{'-' * 9}|{'-' * 16}|\n"
    )

    rows = []
    for v in validations:
        if has_epoch:
            label = str(v.get("epoch", "?"))
        else:
            label = str(v.get("step", "?"))

        train_loss = v.get("train_loss", 0.0)
        val_loss = v.get("val_loss", 0.0)
        gap_ratio = v.get("gap_ratio", 0.0)
        train_ppl = v.get("train_perplexity", 0.0)
        val_ppl = v.get("val_perplexity", 0.0)
        classification = classify_overfitting(gap_ratio)

        rows.append(
            f"| {label:>6} | {train_loss:>10.4f} | {val_loss:>8.4f} | "
            f"{gap_ratio:>+12.4f} | {train_ppl:>9.2f} | {val_ppl:>7.2f} | "
            f"{classification:<14} |\n"
        )

    return header + separator + "".join(rows)


def _interpret_loss_curves(
    steps: list[dict],
    validations: list[dict],
) -> str:
    """Generate a narrative interpretation of the loss curve shape."""
    if not steps:
        return "Insufficient data for loss curve interpretation."

    first_loss = steps[0].get("loss", 0.0)
    last_loss = steps[-1].get("loss", 0.0)
    total_steps = steps[-1].get("step", 0) - steps[0].get("step", 0)
    reduction = first_loss - last_loss
    reduction_pct = (reduction / first_loss * 100) if first_loss > 0 else 0.0

    parts = [
        f"Training loss decreased from {first_loss:.4f} to {last_loss:.4f} "
        f"over {total_steps} steps, a {reduction_pct:.1f}% reduction."
    ]

    if validations and len(validations) >= 2:
        first_val = validations[0].get("val_loss", 0.0)
        last_val = validations[-1].get("val_loss", 0.0)
        val_reduction = first_val - last_val
        if val_reduction > 0:
            parts.append(
                f" Validation loss tracked the training loss closely, "
                f"falling from {first_val:.4f} to {last_val:.4f}."
            )
        else:
            parts.append(
                f" Validation loss moved from {first_val:.4f} to "
                f"{last_val:.4f}, suggesting potential overfitting in "
                f"later training."
            )

    return "".join(parts)


def _interpret_perplexity(validations: list[dict]) -> str:
    """Generate a narrative interpretation of perplexity divergence."""
    if not validations:
        return "No validation data available for perplexity analysis."

    first_v = validations[0]
    last_v = validations[-1]
    first_train_ppl = first_v.get("train_perplexity", 0.0)
    last_train_ppl = last_v.get("train_perplexity", 0.0)
    first_val_ppl = first_v.get("val_perplexity", 0.0)
    last_val_ppl = last_v.get("val_perplexity", 0.0)

    parts = [
        f"Train perplexity decreased from {first_train_ppl:.2f} to "
        f"{last_train_ppl:.2f}. Validation perplexity moved from "
        f"{first_val_ppl:.2f} to {last_val_ppl:.2f}."
    ]

    gap_first = abs(first_val_ppl - first_train_ppl)
    gap_last = abs(last_val_ppl - last_train_ppl)

    if gap_last > gap_first * 1.5:
        parts.append(
            " The growing gap between train and validation perplexity "
            "indicates increasing overfitting as training progressed."
        )
    elif gap_last < gap_first:
        parts.append(
            " The narrowing gap between train and validation perplexity "
            "suggests the model is generalizing well."
        )
    else:
        parts.append(
            " The gap between train and validation perplexity remained "
            "relatively stable throughout training."
        )

    return "".join(parts)


def _interpret_lr_schedule(steps: list[dict], config: Optional[dict]) -> str:
    """Generate a brief description of the LR schedule."""
    scheduler_type = "unknown"
    if config:
        ds_config = config.get("deepspeed_config", {})
        scheduler = ds_config.get("scheduler", {})
        scheduler_type = scheduler.get("type", "unknown")
        params = scheduler.get("params", {})
        warmup_steps = params.get("warmup_num_steps", "?")
        total_steps_cfg = params.get("total_num_steps", "?")
        min_ratio = params.get("cos_min_ratio", "?")

        return (
            f"The training used a {scheduler_type} schedule with "
            f"{warmup_steps} warmup steps over {total_steps_cfg} total steps. "
            f"The minimum LR ratio was set to {min_ratio}, meaning the "
            f"learning rate decayed to {min_ratio} of its peak value by "
            f"the end of training."
        )

    if steps:
        lrs = [s.get("lr", 0.0) for s in steps]
        peak_lr = max(lrs)
        final_lr = lrs[-1]
        return (
            f"The learning rate peaked at {peak_lr:.6f} and decayed to "
            f"{final_lr:.6f} by the end of training."
        )

    return "No learning rate data available."


def _build_diagnosis_section(
    validations: list[dict],
    convergence_issues: dict[str, Any],
) -> str:
    """Build the overfitting diagnosis section with narrative analysis."""
    if not validations:
        return (
            "**Classification: Insufficient data**\n\n"
            "No validation entries were found in the log, so overfitting "
            "cannot be assessed.\n"
        )

    last_v = validations[-1]
    gap_ratio = last_v.get("gap_ratio", 0.0)
    train_loss = last_v.get("train_loss", 0.0)
    val_loss = last_v.get("val_loss", 0.0)
    classification = classify_overfitting(gap_ratio)

    parts = [f"**Classification: {classification}**\n\n"]

    # Main narrative
    gap_abs = val_loss - train_loss
    if classification == "Well-fit":
        parts.append(
            f"The model shows a healthy training profile. The final "
            f"validation loss ({val_loss:.4f}) is very close to the final "
            f"training loss ({train_loss:.4f}), with a relative gap of "
            f"{gap_ratio:+.4f} ({gap_ratio * 100:+.2f}%). This indicates "
            f"the model has learned generalizable patterns from the "
            f"training data without significant memorization.\n"
        )
    elif classification == "Underfitting":
        parts.append(
            f"The model appears to be underfitting. The validation loss "
            f"({val_loss:.4f}) is notably below the training loss "
            f"({train_loss:.4f}), with a relative gap of "
            f"{gap_ratio:+.4f} ({gap_ratio * 100:+.2f}%). This is unusual "
            f"and typically indicates the model has not yet converged, or "
            f"the validation set is easier than the training set.\n"
        )
    elif classification == "Mild overfitting":
        parts.append(
            f"The model shows signs of mild overfitting. The validation "
            f"loss ({val_loss:.4f}) exceeds the training loss "
            f"({train_loss:.4f}) by a relative gap of "
            f"{gap_ratio:+.4f} ({gap_ratio * 100:+.2f}%). While not "
            f"severe, this suggests the model has begun memorizing "
            f"training-specific patterns that do not generalize.\n"
        )
    else:  # Severe overfitting
        parts.append(
            f"The model shows signs of severe overfitting. The validation "
            f"loss ({val_loss:.4f}) substantially exceeds the training loss "
            f"({train_loss:.4f}), with a relative gap of "
            f"{gap_ratio:+.4f} ({gap_ratio * 100:+.2f}%). The model has "
            f"memorized training data patterns that do not generalize to "
            f"unseen text.\n"
        )

    # Convergence warning
    if convergence_issues["still_converging"]:
        parts.append(
            "\n> **Note:** The training loss was still dropping "
            "significantly between the last two validation checkpoints, "
            "suggesting the model has not fully converged and may benefit "
            "from additional training.\n"
        )

    # Validation divergence warning
    if convergence_issues["val_divergence"]:
        start_step = convergence_issues["divergence_start_step"]
        parts.append(
            f"\n> **Warning:** Validation loss divergence detected starting "
            f"around step {start_step}. Validation loss increased for 2+ "
            f"consecutive checkpoints while training loss continued to "
            f"decrease, a classic sign of overfitting.\n"
        )

    return "".join(parts)


def _build_recommendations(
    classification: str,
    convergence_issues: dict[str, Any],
    validations: list[dict],
) -> str:
    """Generate actionable recommendations based on the diagnosis."""
    recs = []

    if classification == "Underfitting" or convergence_issues["still_converging"]:
        recs.append(
            "The model has not fully converged. Consider training for "
            "additional epochs to allow the loss to stabilize."
        )
        recs.append(
            "If the learning rate schedule reaches its minimum before "
            "convergence, consider increasing the total training steps or "
            "raising the minimum LR ratio."
        )

    if classification == "Well-fit":
        recs.append(
            "The model is well-fit. The current training configuration "
            "appears appropriate for this dataset."
        )
        recs.append(
            "If more training epochs are planned, monitor the validation "
            "loss closely for signs of divergence."
        )

    if classification == "Mild overfitting":
        recs.append(
            "Consider increasing dropout (currently the primary "
            "regularization mechanism) to reduce overfitting."
        )
        if validations:
            # Find the step with lowest val loss
            best = min(validations, key=lambda v: v.get("val_loss", float("inf")))
            recs.append(
                f"Early stopping at step {best.get('step', '?')} "
                f"(val loss {best.get('val_loss', 0):.4f}) would have "
                f"produced the best-generalizing checkpoint."
            )

    if classification == "Severe overfitting":
        recs.append(
            "Strongly consider increasing dropout or adding other "
            "regularization (weight decay adjustments, data augmentation)."
        )
        if validations:
            best = min(validations, key=lambda v: v.get("val_loss", float("inf")))
            recs.append(
                f"The best checkpoint by validation loss was at step "
                f"{best.get('step', '?')} (val loss "
                f"{best.get('val_loss', 0):.4f}). Use this checkpoint "
                f"rather than the final one for downstream tasks."
            )
        recs.append(
            "The training data may be too small relative to the model "
            "size. Consider whether a smaller model or larger dataset "
            "would be more appropriate."
        )

    if convergence_issues["val_divergence"]:
        recs.append(
            "Validation loss divergence was detected. Implement early "
            "stopping based on validation loss to prevent wasted compute "
            "on overfit training."
        )

    if not recs:
        recs.append("No specific recommendations at this time.")

    return "\n".join(f"- {r}" for r in recs)


def generate_report(
    parsed: dict[str, Any],
    log_filename: str,
    output_dir: str,
) -> str:
    """Generate the full Markdown analysis report."""
    config = parsed["config"]
    steps = parsed["steps"]
    validations = parsed["validations"]

    # Final classification
    if validations:
        last_gap_ratio = validations[-1].get("gap_ratio", 0.0)
        classification = classify_overfitting(last_gap_ratio)
    else:
        classification = "Insufficient data"

    convergence_issues = detect_convergence_issues(validations)

    # Build the report
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    model_summary = _format_model_summary(config)
    training_time = _format_training_time(steps)

    total_steps = steps[-1].get("step", 0) if steps else 0
    has_epochs = len(steps) > 0 and "epoch" in steps[0]
    max_epoch = max(s.get("epoch", 0) for s in steps) if (steps and has_epochs) else None

    # Incompleteness note
    incompleteness_note = ""
    if parsed["malformed_count"] > 0:
        valid = parsed["total_lines"] - parsed["malformed_count"]
        incompleteness_note = (
            f"\n> **Note:** Log appears incomplete or contains errors "
            f"({valid} valid entries out of {parsed['total_lines']} total lines).\n"
        )

    # Training overview narrative
    final_train_loss = validations[-1].get("train_loss", 0.0) if validations else 0.0
    final_val_loss = validations[-1].get("val_loss", 0.0) if validations else 0.0
    epoch_str = f" across {max_epoch + 1} epoch(s)" if max_epoch is not None else ""
    overview = (
        f"The training ran for {total_steps} steps{epoch_str}, "
        f"taking approximately {training_time} of wall-clock time. "
        f"The final training loss was {final_train_loss:.4f} and the "
        f"final validation loss was {final_val_loss:.4f}."
    )

    # Epoch table note
    if has_epochs:
        epoch_table_header = "## Per-epoch breakdown\n\nMetrics at each validation checkpoint, grouped by epoch.\n\n"
    else:
        epoch_table_header = "## Per-validation breakdown\n\nMetrics at each validation checkpoint.\n\n"

    # Perplexity section — skip if no validation entries
    if validations:
        perplexity_section = (
            f"## Perplexity analysis\n\n"
            f"![Perplexity Gap](perplexity_gap.png)\n\n"
            f"{_interpret_perplexity(validations)}\n"
        )
    else:
        perplexity_section = (
            "## Perplexity analysis\n\n"
            "_Skipped: no validation entries available for perplexity analysis._\n"
        )

    report = (
        f"# Training Analysis Report\n\n"
        f"**Log:** {log_filename}\n"
        f"**Generated:** {now}\n"
        f"**Model:** {model_summary}\n"
        f"{incompleteness_note}\n"
        f"## Training overview\n\n"
        f"{overview}\n\n"
        f"## Loss curves\n\n"
        f"![Loss Curves](loss_curves.png)\n\n"
        f"{_interpret_loss_curves(steps, validations)}\n\n"
        f"{epoch_table_header}"
        f"{_build_epoch_table(validations)}\n"
        f"{perplexity_section}\n"
        f"## Learning rate schedule\n\n"
        f"![LR Schedule](lr_schedule.png)\n\n"
        f"{_interpret_lr_schedule(steps, config)}\n\n"
        f"## Overfitting diagnosis\n\n"
        f"{_build_diagnosis_section(validations, convergence_issues)}\n"
        f"## Recommendations\n\n"
        f"{_build_recommendations(classification, convergence_issues, validations)}\n"
    )

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a JSONL training log for overfitting diagnostics.",
    )
    parser.add_argument(
        "log_file",
        help="Path to the JSONL training log file",
    )
    args = parser.parse_args()

    log_path = os.path.abspath(args.log_file)
    if not os.path.isfile(log_path):
        print(f"Error: File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    log_filename = os.path.basename(log_path)
    log_dir = os.path.dirname(log_path)
    stem = Path(log_path).stem
    output_dir = os.path.join(log_dir, f"{stem}_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Parse log
    print(f"Parsing {log_filename}...")
    parsed = parse_log(log_path)
    n_steps = len(parsed["steps"])
    n_val = len(parsed["validations"])
    fmt = parsed["format"]
    print(
        f"  Loaded {n_steps} step entries, {n_val} validation entries "
        f"(format: {fmt})"
    )
    if parsed["malformed_count"] > 0:
        print(f"  Warning: {parsed['malformed_count']} malformed lines skipped")

    # Generate plots
    print("Generating plots...")

    loss_path = os.path.join(output_dir, "loss_curves.png")
    plot_loss_curves(parsed["steps"], parsed["validations"], loss_path)
    print(f"  Saved: loss_curves.png")

    if n_val > 0:
        ppl_path = os.path.join(output_dir, "perplexity_gap.png")
        plot_perplexity_gap(parsed["validations"], ppl_path)
        print(f"  Saved: perplexity_gap.png")
    else:
        print("  Skipped: perplexity_gap.png (no validation entries)")

    lr_path = os.path.join(output_dir, "lr_schedule.png")
    plot_lr_schedule(parsed["steps"], lr_path)
    print(f"  Saved: lr_schedule.png")

    # Generate report
    print("Writing report...")
    report = generate_report(parsed, log_filename, output_dir)
    report_path = os.path.join(output_dir, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: analysis_report.md")

    print(f"\nDone: {output_dir}/")


if __name__ == "__main__":
    main()
