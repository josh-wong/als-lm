#!/usr/bin/env python3
"""Run the end-to-end ALS-LM hallucination evaluation pipeline.

Orchestrates all six evaluation stages sequentially from a single command:
generate responses, score responses, detect fabrications, classify taxonomy,
curate qualitative samples, and generate a Markdown report. Supports caching
of intermediate results, a --force flag to regenerate everything, and a
--stage flag to run individual stages standalone.

This is a research evaluation tool, not a medical information system.

Usage examples::

    # Full pipeline with a model checkpoint
    python eval/run_evaluation.py --checkpoint checkpoints/tiny_20260225/best

    # Force regeneration of all stages
    python eval/run_evaluation.py --checkpoint checkpoints/tiny_20260225/best --force

    # Run only the report generation stage
    python eval/run_evaluation.py --checkpoint checkpoints/tiny_20260225/best --stage report

    # Custom results and reports directories
    python eval/run_evaluation.py \\
        --checkpoint checkpoints/tiny_20260225/best \\
        --results-dir eval/results \\
        --reports-dir reports
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so that `from eval.utils import ...`
# resolves correctly when running as `python eval/run_evaluation.py`.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eval.utils import find_project_root, resolve_default_paths


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGES = [
    {
        "name": "generate",
        "display": "Generating responses",
        "script": "eval/generate_responses.py",
        "output_file": "responses.json",
    },
    {
        "name": "score",
        "display": "Scoring responses",
        "script": "eval/score_responses.py",
        "output_file": "scores.json",
    },
    {
        "name": "fabrications",
        "display": "Detecting fabrications",
        "script": "eval/detect_fabrications.py",
        "output_file": "fabrications.json",
    },
    {
        "name": "taxonomy",
        "display": "Classifying taxonomy",
        "script": "eval/classify_taxonomy.py",
        "output_file": "taxonomy.json",
    },
    {
        "name": "samples",
        "display": "Curating samples",
        "script": "eval/curate_samples.py",
        "output_file": "samples.json",
    },
    {
        "name": "report",
        "display": "Generating report",
        "script": "eval/generate_report.py",
        "output_file": None,  # Report goes to reports_dir, not results_dir
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for the evaluation pipeline."""
    stage_names = [s["name"] for s in STAGES]
    parser = argparse.ArgumentParser(
        description="Run the ALS-LM hallucination evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stages (in order):\n"
            "  1. generate     - Generate model responses to benchmark questions\n"
            "  2. score        - Score responses against key facts\n"
            "  3. fabrications - Detect fabricated entities\n"
            "  4. taxonomy     - Classify failure modes\n"
            "  5. samples      - Curate qualitative samples\n"
            "  6. report       - Generate Markdown report\n"
            "\n"
            "Examples:\n"
            "  python eval/run_evaluation.py --checkpoint checkpoints/tiny/best\n"
            "  python eval/run_evaluation.py --checkpoint checkpoints/tiny/best --force\n"
            "  python eval/run_evaluation.py --checkpoint checkpoints/tiny/best --stage report\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (directory or .pt file)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all stages even if intermediate files exist",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=stage_names,
        default=None,
        help="Run a single stage by name (requires prior stage outputs to exist)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval/results",
        help="Directory for intermediate JSON files (default: eval/results)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory for final Markdown report (default: reports)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to benchmark questions JSON (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to entity registry JSON (auto-discovered if omitted)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint ID extraction
# ---------------------------------------------------------------------------

def extract_checkpoint_id(checkpoint_path):
    """Extract a human-readable identifier from a checkpoint path.

    Takes the last meaningful path component before any "best" or ".pt"
    segments. This becomes part of the report filename for traceability.

    Examples:
        checkpoints/tiny_20260225/best       -> tiny_20260225
        checkpoints/tiny_20260225/best/best.pt -> tiny_20260225
        checkpoints/500m/best.pt             -> 500m
        /path/to/my_model/best               -> my_model

    Args:
        checkpoint_path: Path to the checkpoint directory or file.

    Returns:
        A string identifier extracted from the path.
    """
    # Normalize and split path into components
    path = os.path.normpath(checkpoint_path)
    parts = path.split(os.sep)

    # Remove empty parts and filter out "best" and .pt filenames
    meaningful = []
    for part in parts:
        if not part:
            continue
        if part == "best":
            continue
        if part.endswith(".pt"):
            continue
        meaningful.append(part)

    if meaningful:
        return meaningful[-1]

    # Fallback: use the full basename
    return os.path.basename(checkpoint_path).replace(".pt", "") or "unknown"


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def format_duration(seconds):
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        A formatted string like "2m 34s" or "0.5s".
    """
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    return f"{minutes}m {remaining:02d}s"


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------

def build_stage_args(stage, checkpoint, results_dir, reports_dir,
                     checkpoint_id, benchmark, registry):
    """Build the subprocess arguments for a given stage.

    Args:
        stage: Stage definition dict from STAGES.
        checkpoint: Path to model checkpoint.
        results_dir: Directory for intermediate JSON files.
        reports_dir: Directory for the final Markdown report.
        checkpoint_id: Extracted checkpoint identifier for report filename.
        benchmark: Absolute path string to benchmark questions JSON.
        registry: Absolute path string to entity registry JSON.

    Returns:
        A list of command-line arguments for subprocess.run.
    """
    name = stage["name"]
    script = stage["script"]
    responses = os.path.join(results_dir, "responses.json")
    scores = os.path.join(results_dir, "scores.json")
    fabrications = os.path.join(results_dir, "fabrications.json")
    taxonomy = os.path.join(results_dir, "taxonomy.json")
    samples = os.path.join(results_dir, "samples.json")

    if name == "generate":
        return [
            sys.executable, script,
            "--checkpoint", checkpoint,
            "--benchmark", benchmark,
            "--output", os.path.join(results_dir, "responses.json"),
        ]
    elif name == "score":
        return [
            sys.executable, script,
            "--responses", responses,
            "--benchmark", benchmark,
            "--output", scores,
        ]
    elif name == "fabrications":
        return [
            sys.executable, script,
            "--responses", responses,
            "--registry", registry,
            "--output", fabrications,
        ]
    elif name == "taxonomy":
        return [
            sys.executable, script,
            "--scores", scores,
            "--fabrications", fabrications,
            "--responses", responses,
            "--benchmark", benchmark,
            "--output", taxonomy,
        ]
    elif name == "samples":
        return [
            sys.executable, script,
            "--scores", scores,
            "--fabrications", fabrications,
            "--responses", responses,
            "--benchmark", benchmark,
            "--taxonomy", taxonomy,
            "--output", samples,
        ]
    elif name == "report":
        report_path = os.path.join(
            reports_dir, f"hallucination_eval_{checkpoint_id}.md"
        )
        return [
            sys.executable, script,
            "--scores", scores,
            "--fabrications", fabrications,
            "--taxonomy", taxonomy,
            "--samples", samples,
            "--responses", responses,
            "--output", report_path,
        ]

    raise ValueError(f"Unknown stage: {name}")


def get_stage_output_path(stage, results_dir, reports_dir, checkpoint_id):
    """Get the output file path for a stage.

    Args:
        stage: Stage definition dict from STAGES.
        results_dir: Directory for intermediate JSON files.
        reports_dir: Directory for the final Markdown report.
        checkpoint_id: Extracted checkpoint identifier for report filename.

    Returns:
        The output file path for this stage.
    """
    if stage["name"] == "report":
        return os.path.join(
            reports_dir, f"hallucination_eval_{checkpoint_id}.md"
        )
    return os.path.join(results_dir, stage["output_file"])


def check_checkpoint_mismatch(responses_path, checkpoint):
    """Warn if cached responses were generated from a different checkpoint.

    Args:
        responses_path: Path to the cached responses.json file.
        checkpoint: The current checkpoint path argument.
    """
    try:
        with open(responses_path) as f:
            data = json.load(f)
        cached_path = data.get("metadata", {}).get("checkpoint_path", "")
        current_path = os.path.abspath(checkpoint)
        if cached_path and cached_path != current_path:
            print(f"  WARNING: Cached responses were generated from a "
                  f"different checkpoint.")
            print(f"    Cached:  {cached_path}")
            print(f"    Current: {current_path}")
            print(f"    Use --force to regenerate from the current checkpoint.")
    except (json.JSONDecodeError, OSError):
        pass


def run_stage(stage, stage_num, total_stages, checkpoint, results_dir,
              reports_dir, checkpoint_id, force, benchmark, registry):
    """Execute a single pipeline stage.

    Handles caching (skip if output exists and not --force), subprocess
    execution, timing, and error reporting.

    Args:
        stage: Stage definition dict from STAGES.
        stage_num: 1-based stage number for display.
        total_stages: Total number of stages for display.
        checkpoint: Path to model checkpoint.
        results_dir: Directory for intermediate JSON files.
        reports_dir: Directory for the final Markdown report.
        checkpoint_id: Extracted checkpoint identifier.
        force: Whether to force regeneration.
        benchmark: Absolute path string to benchmark questions JSON.
        registry: Absolute path string to entity registry JSON.

    Returns:
        True if the stage succeeded, False otherwise.
    """
    output_path = get_stage_output_path(
        stage, results_dir, reports_dir, checkpoint_id
    )

    # Check cache
    if not force and os.path.isfile(output_path):
        # Special warning for generate stage checkpoint mismatch
        if stage["name"] == "generate":
            check_checkpoint_mismatch(output_path, checkpoint)
        print(f"  [{stage_num}/{total_stages}] {stage['display']}..."
              f" skipped (cached)")
        return True

    # Run the stage
    print(f"  [{stage_num}/{total_stages}] {stage['display']}...", end="",
          flush=True)
    t0 = time.time()

    args = build_stage_args(
        stage, checkpoint, results_dir, reports_dir, checkpoint_id,
        benchmark, registry,
    )

    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f" FAILED ({format_duration(elapsed)})")
        print(f"\n  ERROR in stage '{stage['name']}':")
        if result.stderr:
            # Show stderr, indented for readability
            for line in result.stderr.strip().split("\n"):
                print(f"    {line}")
        else:
            print(f"    (no stderr output)")
        if result.stdout:
            print(f"\n  Stage stdout:")
            for line in result.stdout.strip().split("\n")[-10:]:
                print(f"    {line}")
        return False

    print(f" done ({format_duration(elapsed)})")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for the evaluation pipeline."""
    args = parse_args()

    print("\n=== ALS-LM Hallucination Evaluation Pipeline ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical"
          " information system.\n")

    # Discover project root and resolve default paths
    project_root = find_project_root()
    defaults = resolve_default_paths(project_root)

    # Resolve benchmark and registry to absolute paths
    if args.benchmark is not None:
        benchmark = str(Path(args.benchmark).resolve())
    else:
        benchmark = str(defaults["benchmark"])

    if args.registry is not None:
        registry = str(Path(args.registry).resolve())
    else:
        registry = str(defaults["registry"])

    # Resolve checkpoint, results_dir, and reports_dir to absolute paths
    checkpoint = str(Path(args.checkpoint).resolve())
    results_dir = str(Path(args.results_dir).resolve())
    reports_dir = str(Path(args.reports_dir).resolve())

    # Resolve stage script paths relative to project root
    for stage in STAGES:
        stage["script"] = str(project_root / stage["script"])

    # Print resolved paths for user verification
    checkpoint_id = extract_checkpoint_id(args.checkpoint)
    print(f"  Project root: {project_root}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Checkpoint ID: {checkpoint_id}")
    print(f"  Benchmark: {benchmark}")
    print(f"  Registry: {registry}")
    print(f"  Results dir: {results_dir}")
    print(f"  Reports dir: {reports_dir}")

    # Upfront validation: ensure benchmark and registry exist
    if not os.path.isfile(benchmark):
        print(f"\n  ERROR: Benchmark file not found at {benchmark}")
        print(f"  Run the benchmark creation step first, or pass "
              f"--benchmark /path/to/file.")
        sys.exit(1)

    if not os.path.isfile(registry):
        print(f"\n  ERROR: Entity registry not found at {registry}")
        print(f"  Run eval/build_entity_registry.py first, or pass "
              f"--registry /path/to/file.")
        sys.exit(1)

    # Create directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Handle --force: delete all intermediate files
    if args.force:
        print(f"  Force mode: deleting intermediate files...")
        for stage in STAGES:
            if stage["output_file"]:
                path = os.path.join(results_dir, stage["output_file"])
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"    Removed: {path}")
        # Also remove the report file
        report_path = os.path.join(
            reports_dir, f"hallucination_eval_{checkpoint_id}.md"
        )
        if os.path.isfile(report_path):
            os.remove(report_path)
            print(f"    Removed: {report_path}")

    # Determine which stages to run
    if args.stage:
        stages_to_run = [s for s in STAGES if s["name"] == args.stage]
        print(f"\n  Running single stage: {args.stage}\n")
    else:
        stages_to_run = STAGES
        print(f"\n  Running full pipeline ({len(STAGES)} stages)\n")

    # Execute stages
    pipeline_start = time.time()
    total = len(stages_to_run)

    for i, stage in enumerate(stages_to_run, 1):
        success = run_stage(
            stage, i, total, checkpoint, results_dir,
            reports_dir, checkpoint_id, args.force,
            benchmark, registry,
        )
        if not success:
            print(f"\n  Pipeline FAILED at stage '{stage['name']}'. "
                  f"Fix the error and re-run.")
            sys.exit(1)

    pipeline_elapsed = time.time() - pipeline_start

    # Summary
    print(f"\n  === Pipeline Complete ===")
    print(f"  Total time: {format_duration(pipeline_elapsed)}")

    if not args.stage or args.stage == "report":
        report_path = os.path.join(
            reports_dir, f"hallucination_eval_{checkpoint_id}.md"
        )
        if os.path.isfile(report_path):
            print(f"  Report: {report_path}")

    print()


if __name__ == "__main__":
    main()
