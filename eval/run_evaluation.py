#!/usr/bin/env python3
"""Run the end-to-end ALS-LM hallucination evaluation pipeline.

Orchestrates all six evaluation stages sequentially from a single command:
generate responses, score responses, detect fabrications, classify taxonomy,
curate qualitative samples, and generate a Markdown report. Supports both
PyTorch checkpoint and Ollama model inference, caching of intermediate
results, a --force flag to regenerate everything, and a --stage flag to run
individual stages standalone.

This is a research evaluation tool, not a medical information system.

Usage examples::

    # Full pipeline with a model checkpoint
    python eval/run_evaluation.py --checkpoint checkpoints/tiny_20260225/best

    # Full pipeline with an Ollama model
    python eval/run_evaluation.py --ollama-model als-lm-500m:q8_0

    # Force regeneration of all stages
    python eval/run_evaluation.py --ollama-model als-lm-500m:f16 --force

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
            "  python eval/run_evaluation.py --ollama-model als-lm-500m:q8_0\n"
            "  python eval/run_evaluation.py --ollama-model als-lm-500m:f16 --force\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (directory or .pt file)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Ollama model name (e.g., 'als-lm-500m:q8_0'). Uses Ollama API instead of checkpoint.",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0 for deterministic results)",
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

    args = parser.parse_args()

    # Validate: exactly one of --checkpoint or --ollama-model must be provided
    if args.checkpoint and args.ollama_model:
        parser.error("Specify either --checkpoint or --ollama-model, not both.")
    if not args.checkpoint and not args.ollama_model:
        parser.error("One of --checkpoint or --ollama-model is required.")

    return args


# ---------------------------------------------------------------------------
# Checkpoint ID extraction
# ---------------------------------------------------------------------------

def extract_model_id(model_identifier):
    """Extract a filesystem-safe identifier from a model path or Ollama tag.

    For checkpoint paths, takes the last meaningful path component before any
    "best" or ".pt" segments. For Ollama model names containing ``:``,
    replaces the colon with ``_`` to produce a valid directory name.

    Examples::

        checkpoints/tiny_20260225/best        -> tiny_20260225
        checkpoints/tiny_20260225/best/best.pt -> tiny_20260225
        checkpoints/500m/best.pt              -> 500m
        /path/to/my_model/best                -> my_model
        als-lm-500m:q8_0                      -> als-lm-500m_q8_0
        als-lm-500m:f16                       -> als-lm-500m_f16

    Args:
        model_identifier: Checkpoint path or Ollama model tag string.

    Returns:
        A filesystem-safe identifier string.
    """
    # Ollama model names contain ":" — convert to filesystem-safe form
    if ":" in model_identifier:
        return model_identifier.replace(":", "_")

    # Checkpoint path handling (original logic)
    path = os.path.normpath(model_identifier)
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
    return os.path.basename(model_identifier).replace(".pt", "") or "unknown"


# Keep the old name as an alias for backwards compatibility
extract_checkpoint_id = extract_model_id


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

def build_stage_args(stage, results_dir, reports_dir, checkpoint_id,
                     benchmark, registry, checkpoint=None,
                     ollama_model=None, ollama_url=None, temperature=0.0):
    """Build the subprocess arguments for a given stage.

    Args:
        stage: Stage definition dict from STAGES.
        results_dir: Directory for intermediate JSON files.
        reports_dir: Directory for the final Markdown report.
        checkpoint_id: Extracted checkpoint identifier for report filename.
        benchmark: Absolute path string to benchmark questions JSON.
        registry: Absolute path string to entity registry JSON.
        checkpoint: Path to model checkpoint (checkpoint mode).
        ollama_model: Ollama model tag (Ollama mode).
        ollama_url: Ollama server URL.
        temperature: Generation temperature.

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
        cmd = [sys.executable, script]
        if ollama_model:
            cmd += ["--ollama-model", ollama_model]
            if ollama_url:
                cmd += ["--ollama-url", ollama_url]
            cmd += ["--temperature", str(temperature)]
            cmd += ["--resume"]
        else:
            cmd += ["--checkpoint", checkpoint]
        cmd += [
            "--benchmark", benchmark,
            "--output", os.path.join(results_dir, "responses.json"),
        ]
        return cmd
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


def check_model_mismatch(responses_path, checkpoint=None, ollama_model=None):
    """Warn if cached responses were generated from a different model.

    Args:
        responses_path: Path to the cached responses.json file.
        checkpoint: The current checkpoint path argument (checkpoint mode).
        ollama_model: The current Ollama model tag (Ollama mode).
    """
    try:
        with open(responses_path) as f:
            data = json.load(f)
        metadata = data.get("metadata", {})

        if ollama_model:
            cached_model = metadata.get("ollama_model", "")
            if cached_model and cached_model != ollama_model:
                print(f"  WARNING: Cached responses were generated from a "
                      f"different model.")
                print(f"    Cached:  {cached_model}")
                print(f"    Current: {ollama_model}")
                print(f"    Use --force to regenerate from the current model.")
        elif checkpoint:
            cached_path = metadata.get("checkpoint_path", "")
            current_path = os.path.abspath(checkpoint)
            if cached_path and cached_path != current_path:
                print(f"  WARNING: Cached responses were generated from a "
                      f"different checkpoint.")
                print(f"    Cached:  {cached_path}")
                print(f"    Current: {current_path}")
                print(f"    Use --force to regenerate from the current checkpoint.")
    except (json.JSONDecodeError, OSError):
        pass


def run_stage(stage, stage_num, total_stages, results_dir,
              reports_dir, checkpoint_id, force, benchmark, registry,
              checkpoint=None, ollama_model=None, ollama_url=None,
              temperature=0.0):
    """Execute a single pipeline stage.

    Handles caching (skip if output exists and not --force), subprocess
    execution, timing, and error reporting.

    Args:
        stage: Stage definition dict from STAGES.
        stage_num: 1-based stage number for display.
        total_stages: Total number of stages for display.
        results_dir: Directory for intermediate JSON files.
        reports_dir: Directory for the final Markdown report.
        checkpoint_id: Extracted checkpoint identifier.
        force: Whether to force regeneration.
        benchmark: Absolute path string to benchmark questions JSON.
        registry: Absolute path string to entity registry JSON.
        checkpoint: Path to model checkpoint (checkpoint mode).
        ollama_model: Ollama model tag (Ollama mode).
        ollama_url: Ollama server URL.
        temperature: Generation temperature.

    Returns:
        True if the stage succeeded, False otherwise.
    """
    output_path = get_stage_output_path(
        stage, results_dir, reports_dir, checkpoint_id
    )

    # Check cache
    if not force and os.path.isfile(output_path):
        # Special warning for generate stage model mismatch
        if stage["name"] == "generate":
            check_model_mismatch(output_path, checkpoint=checkpoint,
                                 ollama_model=ollama_model)
        print(f"  [{stage_num}/{total_stages}] {stage['display']}..."
              f" skipped (cached)")
        return True

    # Run the stage
    print(f"  [{stage_num}/{total_stages}] {stage['display']}...", end="",
          flush=True)
    t0 = time.time()

    args = build_stage_args(
        stage, results_dir, reports_dir, checkpoint_id,
        benchmark, registry, checkpoint=checkpoint,
        ollama_model=ollama_model, ollama_url=ollama_url,
        temperature=temperature,
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

    # Determine inference mode
    use_ollama = args.ollama_model is not None

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

    # Resolve model identifier and per-model output directories
    if use_ollama:
        model_id = extract_model_id(args.ollama_model)
        checkpoint = None
        ollama_model = args.ollama_model
        ollama_url = args.ollama_url
    else:
        model_id = extract_model_id(args.checkpoint)
        checkpoint = str(Path(args.checkpoint).resolve())
        ollama_model = None
        ollama_url = None

    # Per-model output directories: results/{model_id}/, reports/eval/{model_id}/
    results_dir = str(
        Path(args.results_dir).resolve() / model_id
    )
    reports_dir = str(
        Path(args.reports_dir).resolve() / "eval" / model_id
    )

    # Resolve stage script paths relative to project root. Build local
    # copies to avoid mutating the module-level STAGES constant.
    STAGES = [dict(s) for s in STAGES]
    for stage in STAGES:
        stage["script"] = str(project_root / stage["script"])

    # Print resolved paths for user verification
    print(f"  Project root: {project_root}")
    if use_ollama:
        print(f"  Ollama model: {ollama_model}")
        print(f"  Ollama URL: {ollama_url}")
    else:
        print(f"  Checkpoint: {checkpoint}")
    print(f"  Model ID: {model_id}")
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
            reports_dir, f"hallucination_eval_{model_id}.md"
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
            stage, i, total, results_dir,
            reports_dir, model_id, args.force,
            benchmark, registry, checkpoint=checkpoint,
            ollama_model=ollama_model, ollama_url=ollama_url,
            temperature=args.temperature,
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
            reports_dir, f"hallucination_eval_{model_id}.md"
        )
        if os.path.isfile(report_path):
            print(f"  Report: {report_path}")

    print()


if __name__ == "__main__":
    main()
