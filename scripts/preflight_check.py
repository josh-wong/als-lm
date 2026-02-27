#!/usr/bin/env python3
"""Pre-flight validation script for ALS-LM training runs.

Reads a training run's artifacts (JSONL log, TensorBoard events, checkpoints)
and produces a structured Markdown report with pass/fail for each check.
Designed to validate the full production training configuration before
committing to the ~33-hour production run.

Checks performed:

1. LR continuity after forced resume
2. Memory stability (no OOM)
3. JSONL log completeness (required fields)
4. TensorBoard events existence
5. Best checkpoint tracking and loadability

Usage::

    python scripts/preflight_check.py \\
        --run-dir checkpoints/500M_20260227_HHMMSS/ \\
        --expected-resume-step 250 \\
        --total-steps 500
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

# torch is optional -- only needed for checkpoint loading verification
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments for pre-flight validation."""
    parser = argparse.ArgumentParser(
        description="Pre-flight validation for ALS-LM training runs"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the training run directory (e.g., checkpoints/500M_20260227_...)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/training_log.jsonl",
        help="Path to the JSONL training log (default: logs/training_log.jsonl)",
    )
    parser.add_argument(
        "--tb-dir",
        type=str,
        default="logs/tensorboard/",
        help="Path to TensorBoard events directory (default: logs/tensorboard/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/preflight_report.md",
        help="Path for the output report (default: reports/preflight_report.md)",
    )
    parser.add_argument(
        "--expected-resume-step",
        type=int,
        default=250,
        help="Step at which forced resume occurred (default: 250)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=500,
        help="Warmup steps (default: 500)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=500,
        help="Total pre-flight steps (default: 500)",
    )
    parser.add_argument(
        "--cos-min-ratio",
        type=float,
        default=0.0,
        help="Cosine minimum ratio (default: 0.0)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# (expected_lr formula removed -- LR continuity is verified by comparing
# the same step across pre-kill and post-resume runs, which is more robust
# than matching a theoretical formula to DeepSpeed's internal schedule.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------
def load_jsonl_entries(log_path):
    """Load all entries from the JSONL training log.

    Returns a dict of lists keyed by entry type: step, validation, resume,
    config, epoch, checkpoint_save, best_update, checkpoint_delete, emergency.
    """
    entries = {
        "step": [],
        "validation": [],
        "resume": [],
        "config": [],
        "epoch": [],
        "checkpoint_save": [],
        "best_update": [],
        "checkpoint_delete": [],
        "emergency": [],
    }
    if not os.path.isfile(log_path):
        return entries

    with open(log_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entry_type = entry.get("type", "unknown")
                if entry_type in entries:
                    entries[entry_type].append(entry)
            except json.JSONDecodeError:
                pass  # Skip malformed lines

    return entries


# ---------------------------------------------------------------------------
# Check 1: LR continuity after resume
# ---------------------------------------------------------------------------
def check_lr_continuity(log_path, resume_step):
    """Verify LR continues correctly after forced resume.

    Splits log entries at the resume marker by position in the file, then
    finds overlapping step numbers that appear in both the pre-kill and
    post-resume portions. If the LR at the same step matches across both
    runs, the schedule resumed without discontinuity.

    This approach is more robust than comparing against a theoretical
    formula, since DeepSpeed's internal LR schedule may differ from
    a simple linear-warmup + cosine-decay computation.

    Returns (passed, details_dict).
    """
    details = {
        "resume_marker_found": False,
        "pre_kill_step": None,
        "pre_kill_lr": None,
        "post_resume_step": None,
        "post_resume_lr": None,
        "expected_lr": None,
        "relative_error": None,
        "message": "",
    }

    if not os.path.isfile(log_path):
        details["message"] = f"Log file not found: {log_path}"
        return False, details

    # Parse JSONL and split step entries at the resume marker by file position
    pre_resume_steps = []
    post_resume_steps = []
    found_resume = False

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_type = entry.get("type", "unknown")
            if entry_type == "resume":
                found_resume = True
                details["resume_marker_found"] = True
                continue
            if entry_type == "step":
                if found_resume:
                    post_resume_steps.append(entry)
                else:
                    pre_resume_steps.append(entry)

    if not found_resume:
        details["message"] = "No resume marker found in log"
        return False, details

    if not pre_resume_steps:
        details["message"] = "No step entries found before resume marker"
        return False, details

    if not post_resume_steps:
        details["message"] = "No step entries found after resume marker"
        return False, details

    # Build lookup of step number -> lr from pre-resume entries
    pre_lr_by_step = {}
    for e in pre_resume_steps:
        step = e.get("step")
        lr = e.get("lr") or e.get("learning_rate")
        if step is not None and lr is not None:
            pre_lr_by_step[step] = lr

    # Find the first post-resume step that also exists in pre-resume entries
    for e in sorted(post_resume_steps, key=lambda x: x.get("step", 0)):
        post_step = e.get("step")
        post_lr = e.get("lr") or e.get("learning_rate")
        if post_step in pre_lr_by_step and post_lr is not None:
            pre_lr = pre_lr_by_step[post_step]
            details["pre_kill_step"] = post_step
            details["pre_kill_lr"] = pre_lr
            details["post_resume_step"] = post_step
            details["post_resume_lr"] = post_lr
            details["expected_lr"] = pre_lr

            if pre_lr == 0:
                rel_error = abs(post_lr)
            else:
                rel_error = abs(post_lr - pre_lr) / abs(pre_lr)
            details["relative_error"] = rel_error

            tolerance = 1e-5
            if rel_error <= tolerance:
                details["message"] = (
                    f"LR at step {post_step} matches pre-kill value "
                    f"(post-resume={post_lr:.8e}, pre-kill={pre_lr:.8e}, "
                    f"rel_error={rel_error:.2e})"
                )
                return True, details
            else:
                details["message"] = (
                    f"LR mismatch at step {post_step}: "
                    f"post-resume={post_lr:.8e}, pre-kill={pre_lr:.8e}, "
                    f"rel_error={rel_error:.2e} > {tolerance:.0e}"
                )
                return False, details

    # No overlapping steps — fall back to comparing last pre-kill to first
    # post-resume and use a generous tolerance for the step gap
    pre_entry = pre_resume_steps[-1]
    post_entry = post_resume_steps[0]
    pre_lr = pre_entry.get("lr") or pre_entry.get("learning_rate")
    post_lr = post_entry.get("lr") or post_entry.get("learning_rate")

    details["pre_kill_step"] = pre_entry.get("step")
    details["pre_kill_lr"] = pre_lr
    details["post_resume_step"] = post_entry.get("step")
    details["post_resume_lr"] = post_lr
    details["expected_lr"] = pre_lr

    if pre_lr is None or post_lr is None:
        details["message"] = "Missing LR values in step entries"
        return False, details

    if pre_lr == 0:
        rel_error = abs(post_lr)
    else:
        rel_error = abs(post_lr - pre_lr) / abs(pre_lr)
    details["relative_error"] = rel_error

    tolerance = 0.10  # 10% for non-overlapping comparison
    if rel_error <= tolerance:
        details["message"] = (
            f"LR at step {details['post_resume_step']} is consistent with "
            f"pre-kill LR at step {details['pre_kill_step']} "
            f"(rel_error={rel_error:.2e}, no overlapping steps found)"
        )
        return True, details
    else:
        details["message"] = (
            f"LR discontinuity: pre-kill step {details['pre_kill_step']} "
            f"lr={pre_lr:.8e}, post-resume step {details['post_resume_step']} "
            f"lr={post_lr:.8e}, rel_error={rel_error:.2e} > 0.10"
        )
        return False, details


# ---------------------------------------------------------------------------
# Check 2: Memory stability (no OOM)
# ---------------------------------------------------------------------------
def check_memory_stability(entries, total_steps):
    """Verify the run completed without OOM by checking for a final step entry.

    Returns (passed, details_dict).
    """
    details = {
        "final_step": None,
        "target_step": total_steps,
        "peak_gpu_mem_gb": None,
        "message": "",
    }

    step_entries = sorted(entries.get("step", []), key=lambda e: e.get("step", 0))
    if not step_entries:
        details["message"] = "No step entries found in log"
        return False, details

    last_entry = step_entries[-1]
    final_step = last_entry.get("step", 0)
    details["final_step"] = final_step

    # Extract peak GPU memory from the last step entry
    gpu_mem_mb = last_entry.get("gpu_mem_mb", 0)
    if gpu_mem_mb > 0:
        details["peak_gpu_mem_gb"] = round(gpu_mem_mb / 1024, 2)

    # Also scan for the highest GPU memory across all step entries
    max_gpu_mem_mb = max(
        (e.get("gpu_mem_mb", 0) for e in step_entries), default=0
    )
    if max_gpu_mem_mb > 0:
        details["peak_gpu_mem_gb"] = round(max_gpu_mem_mb / 1024, 2)

    # Also check validation entries for a later step (validation may log
    # at the final step even when the regular log_interval skips it)
    validation_entries = entries.get("validation", [])
    for ve in validation_entries:
        v_step = ve.get("step", 0)
        if v_step > final_step:
            final_step = v_step
            details["final_step"] = final_step

    # PASS if the final step is within the log interval of the target
    tolerance = 15
    if final_step >= total_steps - 1 - tolerance:
        details["message"] = (
            f"Run completed to step {final_step} "
            f"(target: {total_steps - 1}, within tolerance of {tolerance})"
        )
        return True, details
    else:
        details["message"] = (
            f"Log ends at step {final_step}, significantly before target "
            f"step {total_steps - 1} (more than {tolerance} steps short)"
        )
        return False, details


# ---------------------------------------------------------------------------
# Check 3: JSONL log completeness
# ---------------------------------------------------------------------------
def check_jsonl_completeness(entries):
    """Verify JSONL log entries have all required fields.

    Returns (passed, details_dict).
    """
    REQUIRED_STEP_FIELDS = [
        "step", "loss", "lr", "epoch", "epoch_progress",
        "perplexity", "grad_norm", "loss_scale", "gpu_mem_mb", "tokens_per_sec",
    ]
    # The log uses "lr" not "learning_rate" for step entries (see log_step() in train.py)

    REQUIRED_VALIDATION_FIELDS = [
        "train_loss", "val_loss", "train_perplexity", "val_perplexity",
        "generalization_gap", "gap_ratio",
    ]

    details = {
        "step_count": 0,
        "validation_count": 0,
        "missing_step_fields": [],
        "missing_validation_fields": [],
        "step_field_coverage": {},
        "message": "",
    }

    step_entries = entries.get("step", [])
    validation_entries = entries.get("validation", [])

    details["step_count"] = len(step_entries)
    details["validation_count"] = len(validation_entries)

    # Check step entries field coverage
    step_missing = []
    field_present_counts = {f: 0 for f in REQUIRED_STEP_FIELDS}
    for entry in step_entries:
        for field in REQUIRED_STEP_FIELDS:
            if field in entry:
                field_present_counts[field] += 1

    total_step = max(len(step_entries), 1)
    for field in REQUIRED_STEP_FIELDS:
        coverage = field_present_counts[field] / total_step
        details.setdefault("step_field_coverage", {})[field] = round(coverage, 4)
        if coverage < 0.9:
            step_missing.append(f"{field} ({coverage:.0%})")

    details["missing_step_fields"] = step_missing

    # Check validation entries field coverage
    val_missing = []
    val_field_present = {f: 0 for f in REQUIRED_VALIDATION_FIELDS}
    for entry in validation_entries:
        for field in REQUIRED_VALIDATION_FIELDS:
            if field in entry:
                val_field_present[field] += 1

    total_val = max(len(validation_entries), 1)
    for field in REQUIRED_VALIDATION_FIELDS:
        coverage = val_field_present[field] / total_val
        if coverage < 0.9:
            val_missing.append(f"{field} ({coverage:.0%})")

    details["missing_validation_fields"] = val_missing

    all_missing = step_missing + val_missing
    if not all_missing:
        details["message"] = (
            f"All required fields present: "
            f"{len(step_entries)} step entries, {len(validation_entries)} validation entries"
        )
        return True, details
    else:
        details["message"] = (
            f"Missing fields: {', '.join(all_missing)} "
            f"({len(step_entries)} step entries, {len(validation_entries)} validation entries)"
        )
        return False, details


# ---------------------------------------------------------------------------
# Check 4: TensorBoard events
# ---------------------------------------------------------------------------
def check_tensorboard_events(tb_dir):
    """Verify TensorBoard events directory contains event files.

    Returns (passed, details_dict).
    """
    details = {
        "tb_dir": tb_dir,
        "event_files": [],
        "message": "",
    }

    if not os.path.isdir(tb_dir):
        details["message"] = f"TensorBoard directory not found: {tb_dir}"
        return False, details

    # Search recursively for event files (they may be in subdirectories)
    pattern = os.path.join(tb_dir, "**", "events.out.tfevents.*")
    event_files = glob.glob(pattern, recursive=True)
    details["event_files"] = [os.path.relpath(f, tb_dir) for f in event_files]

    if event_files:
        details["message"] = (
            f"Found {len(event_files)} TensorBoard event file(s) in {tb_dir}"
        )
        return True, details
    else:
        details["message"] = f"No TensorBoard event files found in {tb_dir}"
        return False, details


# ---------------------------------------------------------------------------
# Check 5: Best checkpoint tracking
# ---------------------------------------------------------------------------
def check_best_checkpoint(run_dir):
    """Verify best checkpoint exists and is loadable.

    Returns (passed, details_dict).
    """
    details = {
        "best_dir_exists": False,
        "best_pt_exists": False,
        "best_pt_loadable": False,
        "meta_exists": False,
        "meta_fields": {},
        "message": "",
    }

    best_dir = os.path.join(run_dir, "best")
    if not os.path.isdir(best_dir):
        details["message"] = f"Best checkpoint directory not found: {best_dir}"
        return False, details
    details["best_dir_exists"] = True

    # Check for best.pt
    best_pt_path = os.path.join(best_dir, "best.pt")
    if not os.path.isfile(best_pt_path):
        details["message"] = f"best.pt not found in {best_dir}"
        return False, details
    details["best_pt_exists"] = True

    # Attempt to load best.pt
    if not TORCH_AVAILABLE:
        details["message"] = (
            "torch not available -- cannot verify best.pt loadability. "
            "File exists but was not validated."
        )
        return False, details

    try:
        checkpoint = torch.load(best_pt_path, map_location="cpu", weights_only=True)
        details["best_pt_loadable"] = True
    except Exception as e:
        details["message"] = f"best.pt failed to load: {e}"
        return False, details

    # Check checkpoint_meta.json
    meta_path = os.path.join(best_dir, "checkpoint_meta.json")
    if not os.path.isfile(meta_path):
        details["message"] = (
            f"checkpoint_meta.json not found in {best_dir} "
            f"(best.pt loads successfully)"
        )
        return False, details
    details["meta_exists"] = True

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        details["meta_fields"] = {
            "val_loss": meta.get("val_loss"),
            "step": meta.get("step"),
            "is_best": meta.get("is_best"),
            "epoch": meta.get("epoch"),
        }

        if "val_loss" not in meta or "step" not in meta:
            missing = []
            if "val_loss" not in meta:
                missing.append("val_loss")
            if "step" not in meta:
                missing.append("step")
            details["message"] = (
                f"checkpoint_meta.json missing required fields: {', '.join(missing)}"
            )
            return False, details

    except (json.JSONDecodeError, OSError) as e:
        details["message"] = f"checkpoint_meta.json failed to parse: {e}"
        return False, details

    details["message"] = (
        f"Best checkpoint valid: step {meta.get('step')}, "
        f"val_loss={meta.get('val_loss'):.4f}, loadable=True"
    )
    return True, details


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(args, results, entries):
    """Generate a structured Markdown report from check results.

    Args:
        args: Parsed CLI arguments
        results: List of (check_name, passed, details) tuples
        entries: Parsed JSONL entries dict
    """
    now = datetime.now(timezone.utc).isoformat()
    all_passed = all(passed for _, passed, _ in results)
    verdict = "PRE-FLIGHT PASSED" if all_passed else "PRE-FLIGHT FAILED"

    lines = []
    lines.append("# Pre-flight Validation Report")
    lines.append("")
    lines.append(f"**Timestamp:** {now}")
    lines.append(f"**Run directory:** `{args.run_dir}`")
    lines.append(f"**Log file:** `{args.log_file}`")
    lines.append(f"**TensorBoard directory:** `{args.tb_dir}`")
    lines.append(f"**Expected resume step:** {args.expected_resume_step}")
    lines.append(f"**Total steps:** {args.total_steps}")
    lines.append(f"**Peak LR:** {args.lr:.1e}")
    lines.append(f"**Warmup steps:** {args.warmup}")
    lines.append(f"**Cosine min ratio:** {args.cos_min_ratio}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Check                        | Status | Details                                    |")
    lines.append("|------------------------------|--------|--------------------------------------------|")
    for name, passed, details in results:
        status = "PASS" if passed else "FAIL"
        msg = details.get("message", "")
        # Truncate long messages for the summary table
        if len(msg) > 60:
            msg = msg[:57] + "..."
        lines.append(f"| {name:<28} | {status:<6} | {msg:<42} |")
    lines.append("")

    # Overall verdict
    lines.append("## Verdict")
    lines.append("")
    if all_passed:
        lines.append(f"**{verdict}** -- All checks passed. Safe to proceed with production training.")
    else:
        failed_checks = [name for name, passed, _ in results if not passed]
        lines.append(f"**{verdict}** -- The following checks failed:")
        lines.append("")
        for fc in failed_checks:
            lines.append(f"- {fc}")
        lines.append("")
        lines.append("Review the detailed results below and resolve issues before starting production training.")
    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")

    for name, passed, details in results:
        status = "PASS" if passed else "FAIL"
        lines.append(f"### {name} ({status})")
        lines.append("")
        lines.append(details.get("message", "No details available."))
        lines.append("")

        # Check-specific detail rendering
        if "pre_kill_step" in details and details["pre_kill_step"] is not None:
            lines.append(f"- **Resume marker found:** {details.get('resume_marker_found', False)}")
            lines.append(f"- **Pre-kill step:** {details.get('pre_kill_step')}")
            lines.append(f"- **Pre-kill LR:** {details.get('pre_kill_lr')}")
            lines.append(f"- **Post-resume step:** {details.get('post_resume_step')}")
            lines.append(f"- **Post-resume LR:** {details.get('post_resume_lr')}")
            lines.append(f"- **Expected LR:** {details.get('expected_lr')}")
            lines.append(f"- **Relative error:** {details.get('relative_error')}")
            lines.append("")

        if "final_step" in details and details["final_step"] is not None:
            lines.append(f"- **Final step in log:** {details.get('final_step')}")
            lines.append(f"- **Target step:** {details.get('target_step')}")
            if details.get("peak_gpu_mem_gb") is not None:
                lines.append(f"- **Peak GPU memory:** {details.get('peak_gpu_mem_gb')} GB")
            lines.append("")

        if "step_count" in details:
            lines.append(f"- **Step entries:** {details.get('step_count')}")
            lines.append(f"- **Validation entries:** {details.get('validation_count')}")
            if details.get("missing_step_fields"):
                lines.append(f"- **Missing step fields:** {', '.join(details['missing_step_fields'])}")
            if details.get("missing_validation_fields"):
                lines.append(f"- **Missing validation fields:** {', '.join(details['missing_validation_fields'])}")
            lines.append("")

        if "event_files" in details:
            lines.append(f"- **Event files found:** {len(details.get('event_files', []))}")
            for ef in details.get("event_files", [])[:5]:
                lines.append(f"  - `{ef}`")
            lines.append("")

        if "best_dir_exists" in details:
            lines.append(f"- **Best directory exists:** {details.get('best_dir_exists')}")
            lines.append(f"- **best.pt exists:** {details.get('best_pt_exists')}")
            lines.append(f"- **best.pt loadable:** {details.get('best_pt_loadable')}")
            lines.append(f"- **checkpoint_meta.json exists:** {details.get('meta_exists')}")
            if details.get("meta_fields"):
                for k, v in details["meta_fields"].items():
                    lines.append(f"- **Meta {k}:** {v}")
            lines.append("")

    # Log statistics
    lines.append("## Log Statistics")
    lines.append("")
    step_entries = entries.get("step", [])
    if step_entries:
        first_step = step_entries[0].get("step", "?")
        last_step = step_entries[-1].get("step", "?")
        lines.append(f"- **Step range:** {first_step} to {last_step}")
        lines.append(f"- **Total step entries:** {len(step_entries)}")
    val_entries = entries.get("validation", [])
    lines.append(f"- **Validation entries:** {len(val_entries)}")
    resume_entries = entries.get("resume", [])
    lines.append(f"- **Resume markers:** {len(resume_entries)}")
    epoch_entries = entries.get("epoch", [])
    lines.append(f"- **Epoch boundaries:** {len(epoch_entries)}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated by scripts/preflight_check.py at {now}*")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Load and parse JSONL log
    entries = load_jsonl_entries(args.log_file)

    # Run all checks
    results = []

    # Check 1: LR continuity after resume
    passed, details = check_lr_continuity(
        args.log_file, args.expected_resume_step,
    )
    results.append(("LR continuity after resume", passed, details))

    # Check 2: Memory stability
    passed, details = check_memory_stability(entries, args.total_steps)
    results.append(("Memory stability (no OOM)", passed, details))

    # Check 3: JSONL log completeness
    passed, details = check_jsonl_completeness(entries)
    results.append(("JSONL log completeness", passed, details))

    # Check 4: TensorBoard events
    passed, details = check_tensorboard_events(args.tb_dir)
    results.append(("TensorBoard events", passed, details))

    # Check 5: Best checkpoint tracking
    passed, details = check_best_checkpoint(args.run_dir)
    results.append(("Best checkpoint tracking", passed, details))

    # Generate report
    report = generate_report(args, results, entries)

    # Write report to output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)

    # Print verdict and report path
    all_passed = all(p for _, p, _ in results)
    verdict = "PRE-FLIGHT PASSED" if all_passed else "PRE-FLIGHT FAILED"

    print(f"\n{'=' * 50}")
    print(f"  {verdict}")
    print(f"{'=' * 50}")
    for name, passed, details in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print(f"\n  Report: {args.output}")
    print(f"{'=' * 50}\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
