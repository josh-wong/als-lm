#!/usr/bin/env python3
"""Run the hallucination evaluation pipeline against multiple Ollama models.

Orchestrates sequential evaluation runs for all three production GGUF
quantization levels (F16, Q8_0, Q4_K_M). Each model goes through the
full 6-stage pipeline: generate responses, score, detect fabrications,
classify taxonomy, curate samples, and generate a Markdown report.

After all models finish, raw JSON result files are copied alongside the
reports so each model's report directory is self-contained.

This is a research evaluation tool, not a medical information system.

Usage examples::

    # Run all three models
    python eval/run_multi_model_eval.py

    # Run only the F16 model
    python eval/run_multi_model_eval.py --models f16

    # Run two specific models with force regeneration
    python eval/run_multi_model_eval.py --models f16 q8_0 --force

    # Custom Ollama URL
    python eval/run_multi_model_eval.py --ollama-url http://localhost:11434
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import requests


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

MODELS = [
    {"name": "als-lm-500m:f16", "id": "f16",
     "fs_id": "als-lm-500m_f16"},
    {"name": "als-lm-500m:q8_0", "id": "q8_0",
     "fs_id": "als-lm-500m_q8_0"},
    {"name": "als-lm-500m:q4_k_m", "id": "q4_k_m",
     "fs_id": "als-lm-500m_q4_k_m"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_duration(seconds):
    """Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        A formatted string like "5m 23s" or "45s".
    """
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    return f"{minutes}m {remaining:02d}s"


def get_project_root():
    """Get the project root directory.

    Returns:
        Absolute path to the project root (parent of eval/).
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_ollama_running(ollama_url):
    """Verify the Ollama server is reachable.

    Args:
        ollama_url: Base URL for the Ollama API.

    Returns:
        List of model names available on the server.

    Raises:
        SystemExit: If the server is unreachable.
    """
    try:
        resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except requests.ConnectionError:
        print(f"\nERROR: Cannot connect to Ollama at {ollama_url}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"\nERROR: Ollama health check failed: {e}")
        sys.exit(1)


def check_models_available(requested_models, available_models):
    """Verify all requested models are loaded in Ollama.

    Args:
        requested_models: List of model dicts with "name" keys.
        available_models: List of model name strings from Ollama.

    Raises:
        SystemExit: If any requested model is missing.
    """
    missing = []
    for model in requested_models:
        if model["name"] not in available_models:
            missing.append(model["name"])
    if missing:
        print(f"\nERROR: The following models are not available in Ollama:")
        for name in missing:
            print(f"  - {name}")
        print(f"\nAvailable models: {', '.join(available_models)}")
        print("Load missing models with: ollama pull <model-name>")
        sys.exit(1)


def check_eval_files(project_root):
    """Verify benchmark and entity registry files exist.

    Args:
        project_root: Absolute path to the project root.

    Raises:
        SystemExit: If required files are missing.
    """
    benchmark = os.path.join(project_root, "eval", "questions.json")
    registry = os.path.join(project_root, "eval", "entity_registry.json")

    missing = []
    if not os.path.isfile(benchmark):
        missing.append(benchmark)
    if not os.path.isfile(registry):
        missing.append(registry)

    if missing:
        print("\nERROR: Required evaluation files not found:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Result copying
# ---------------------------------------------------------------------------

RESULT_FILES = [
    "responses.json",
    "scores.json",
    "fabrications.json",
    "taxonomy.json",
    "samples.json",
]


def copy_results_to_reports(results_dir, reports_dir, model_id):
    """Copy raw JSON result files alongside the Markdown report.

    This makes each model's report directory self-contained with both the
    human-readable report and the machine-readable data files.

    Args:
        results_dir: Path to eval/results/{model_id}/.
        reports_dir: Path to reports/eval/{model_id}/.
        model_id: Model identifier string.
    """
    copied = 0
    for filename in RESULT_FILES:
        src = os.path.join(results_dir, filename)
        dst = os.path.join(reports_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1
    if copied > 0:
        print(f"    Copied {copied} result files to {reports_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    model_ids = [m["id"] for m in MODELS]
    parser = argparse.ArgumentParser(
        description="Run hallucination evaluation across multiple Ollama models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models evaluated (in order):\n"
            "  f16      - als-lm-500m:f16 (full precision GGUF)\n"
            "  q8_0     - als-lm-500m:q8_0 (8-bit quantized)\n"
            "  q4_k_m   - als-lm-500m:q4_k_m (4-bit quantized)\n"
            "\n"
            "Examples:\n"
            "  python eval/run_multi_model_eval.py\n"
            "  python eval/run_multi_model_eval.py --models f16 q8_0\n"
            "  python eval/run_multi_model_eval.py --force\n"
        ),
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=model_ids,
        default=model_ids,
        help="Model IDs to evaluate (default: all three)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to run_evaluation.py to regenerate all stages",
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default="eval/results",
        help="Base directory for per-model results (default: eval/results)",
    )
    parser.add_argument(
        "--reports-base",
        type=str,
        default="reports/eval",
        help="Base directory for per-model reports (default: reports/eval)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for multi-model evaluation."""
    args = parse_args()
    project_root = get_project_root()

    print("\n=== ALS-LM Multi-Model Hallucination Evaluation ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical"
          " information system.\n")

    # Filter models to requested subset
    requested = [m for m in MODELS if m["id"] in args.models]

    # Pre-flight checks
    print("Pre-flight checks:")
    available = check_ollama_running(args.ollama_url)
    print(f"  Ollama running at {args.ollama_url}")
    check_models_available(requested, available)
    print(f"  All requested models available in Ollama")
    check_eval_files(project_root)
    print(f"  Benchmark and entity registry files found")
    print(f"\nWill evaluate {len(requested)} model(s): "
          f"{', '.join(m['name'] for m in requested)}\n")

    # Run evaluation for each model
    # Note: run_evaluation.py internally appends /{fs_id}/ to --results-dir
    # and /eval/{fs_id}/ to --reports-dir, where fs_id is derived from the
    # Ollama model name (colon replaced with underscore).  We pass the BASE
    # directories here and let run_evaluation.py create the per-model subdirs.
    run_evaluation_script = os.path.join(project_root, "eval", "run_evaluation.py")
    results_base_abs = os.path.join(project_root, args.results_base)
    reports_base_abs = os.path.join(project_root, args.reports_base)

    # run_evaluation.py appends "eval/{fs_id}" to --reports-dir, but our
    # reports_base already includes "eval/".  Strip it for the subprocess.
    # e.g. reports_base="reports/eval" -> pass "reports" so it builds
    # "reports/eval/{fs_id}".
    reports_dir_for_subprocess = os.path.dirname(reports_base_abs)

    results_list = []
    total_start = time.time()

    for i, model in enumerate(requested, 1):
        model_name = model["name"]
        model_id = model["id"]
        fs_id = model["fs_id"]

        # Paths that run_evaluation.py will actually create
        actual_results_dir = os.path.join(results_base_abs, fs_id)
        actual_reports_dir = os.path.join(reports_base_abs, fs_id)

        print(f"{'=' * 60}")
        print(f"  [{i}/{len(requested)}] Evaluating {model_name}")
        print(f"{'=' * 60}")

        # Build command — pass base dirs, run_evaluation.py adds subdirs
        cmd = [
            sys.executable,
            run_evaluation_script,
            "--ollama-model", model_name,
            "--ollama-url", args.ollama_url,
            "--temperature", "0.0",
            "--results-dir", results_base_abs,
            "--reports-dir", reports_dir_for_subprocess,
        ]
        if args.force:
            cmd.append("--force")

        # Run the evaluation pipeline with stdout/stderr passed through
        model_start = time.time()
        try:
            result = subprocess.run(cmd, cwd=project_root)
            model_elapsed = time.time() - model_start
            success = result.returncode == 0
        except Exception as e:
            model_elapsed = time.time() - model_start
            print(f"\n  ERROR: {model_name} crashed: {e}")
            success = False

        # Determine report path (run_evaluation.py names it with fs_id)
        report_filename = f"hallucination_eval_{fs_id}.md"
        report_path = os.path.join(actual_reports_dir, report_filename)

        if success:
            # Copy raw JSON results alongside the report
            copy_results_to_reports(actual_results_dir, actual_reports_dir,
                                   model_id)

        results_list.append({
            "name": model_name,
            "id": model_id,
            "fs_id": fs_id,
            "success": success,
            "elapsed": model_elapsed,
            "report": report_path if success and os.path.isfile(report_path) else None,
        })

        status = "PASS" if success else "FAIL"
        print(f"\n  {model_name}: {status} ({format_duration(model_elapsed)})\n")

    total_elapsed = time.time() - total_start

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"=== Multi-Model Evaluation Complete ===")
    print(f"{'=' * 70}\n")

    # Header
    print(f"{'Model':<25} {'Status':<10} {'Time':<10} Report")
    print(f"{'-' * 25} {'-' * 9} {'-' * 9} {'-' * 25}")

    any_failed = False
    for r in results_list:
        status = "PASS" if r["success"] else "FAIL"
        if not r["success"]:
            any_failed = True
        elapsed = format_duration(r["elapsed"])
        report = r["report"] if r["report"] else "N/A"
        print(f"{r['name']:<25} {status:<10} {elapsed:<10} {report}")

    print(f"\nTotal time: {format_duration(total_elapsed)}")

    if any_failed:
        failed_names = [r["name"] for r in results_list if not r["success"]]
        print(f"\nWARNING: {len(failed_names)} model(s) failed: "
              f"{', '.join(failed_names)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(results_list)} model(s) evaluated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
