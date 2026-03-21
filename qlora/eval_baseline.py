#!/usr/bin/env python3
"""Ablation baseline evaluation: evaluate the unmodified Qwen2.5-1.5B-Instruct
model on the 160-question hallucination benchmark via Ollama.

This script handles the full path from HuggingFace model download through GGUF
conversion, Ollama registration, and evaluation pipeline invocation. The result
establishes the pre-fine-tuning accuracy baseline, answering "how much does
QLoRA fine-tuning actually help?" by comparing against the same 160-question
benchmark used for all prior model evaluations.

The evaluation uses the existing 6-stage pipeline (eval/run_evaluation.py) for
directly comparable results across all model variants.

Note: Uses Qwen/Qwen2.5-1.5B-Instruct as a temporary substitute while
awaiting gated access approval for meta-llama/Llama-3.2-1B-Instruct.
The pipeline is model-agnostic -- switching back to Llama requires only
changing model_id in configs/qlora.json.

Usage::

    python qlora/eval_baseline.py
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root discovery (existing project pattern)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OLLAMA_MODEL_NAME = "als-lm-llama32-base"
BASELINE_DIR = PROJECT_ROOT / "checkpoints" / "qlora" / "baseline"
GGUF_CONVERTER = PROJECT_ROOT / "lib" / "llama.cpp" / "convert_hf_to_gguf.py"
EVAL_PIPELINE = PROJECT_ROOT / "eval" / "run_evaluation.py"
CONFIG_PATH = PROJECT_ROOT / "configs" / "qlora.json"


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def section(title: str):
    """Print a section header with separators."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def status(msg: str):
    """Print a status message."""
    print(f"  {msg}")


def ok(msg: str):
    """Print a green success message."""
    print(f"  {GREEN}[OK]{RESET} {msg}")


def warn(msg: str):
    """Print a yellow warning message."""
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def fatal(msg: str):
    """Print a red fatal error and exit."""
    print(f"\n  {RED}FATAL:{RESET} {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def check_prerequisites():
    """Verify all required tools and files are available before starting."""
    section("Preflight checks")

    # Check ollama in PATH
    if shutil.which("ollama") is None:
        fatal(
            "ollama not found in PATH\n"
            "  Install Ollama: https://ollama.com/download\n"
            "  Then start the server: ollama serve"
        )
    ok("ollama found in PATH")

    # Check GGUF converter exists
    if not GGUF_CONVERTER.exists():
        fatal(
            f"GGUF converter not found at {GGUF_CONVERTER}\n"
            "  Fix: git submodule update --init lib/llama.cpp"
        )
    ok("GGUF converter found")

    # Check eval pipeline exists
    if not EVAL_PIPELINE.exists():
        fatal(f"Evaluation pipeline not found at {EVAL_PIPELINE}")
    ok("Evaluation pipeline found")

    # Check config exists
    if not CONFIG_PATH.exists():
        fatal(f"Config not found at {CONFIG_PATH}")
    ok("Config found")


# ---------------------------------------------------------------------------
# Step 1: Load config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """Load model_id and other config from configs/qlora.json."""
    section("Step 1: Loading config")
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    model_id = config["model"]["model_id"]
    status(f"Model ID: {model_id}")
    status(f"GGUF output dir: {BASELINE_DIR}")
    return config


# ---------------------------------------------------------------------------
# Step 2: Download HuggingFace model
# ---------------------------------------------------------------------------
def download_model(model_id: str) -> Path:
    """Download/cache the HuggingFace model for GGUF conversion.

    Uses huggingface_hub.snapshot_download() to get the model files without
    loading into GPU memory. Returns the local cache path.
    """
    section("Step 2: Downloading HuggingFace model")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        fatal(
            "huggingface_hub not installed\n"
            "  Fix: pip install huggingface_hub"
        )

    status(f"Downloading {model_id} (will use cache if available)...")
    model_path = snapshot_download(model_id)
    status(f"Model cached at: {model_path}")
    return Path(model_path)


# ---------------------------------------------------------------------------
# Step 3: GGUF conversion
# ---------------------------------------------------------------------------
def convert_to_gguf(model_path: Path, model_id: str) -> Path:
    """Convert HuggingFace model to GGUF F16 format.

    Skips conversion if the GGUF file already exists.
    """
    section("Step 3: GGUF conversion")

    # Derive GGUF filename from model_id
    model_name = model_id.split("/")[-1].lower()
    gguf_filename = f"{model_name}-f16.gguf"
    gguf_path = BASELINE_DIR / gguf_filename

    # Create output directory
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if already exists
    if gguf_path.exists():
        size_gb = gguf_path.stat().st_size / (1024 ** 3)
        ok(f"GGUF already exists, skipping conversion ({size_gb:.2f} GB)")
        return gguf_path

    status(f"Converting {model_path} to GGUF F16...")
    status(f"Output: {gguf_path}")

    result = subprocess.run(
        [
            sys.executable,
            str(GGUF_CONVERTER),
            str(model_path),
            "--outtype", "f16",
            "--outfile", str(gguf_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        status(f"stdout: {result.stdout[-500:]}" if result.stdout else "")
        status(f"stderr: {result.stderr[-500:]}" if result.stderr else "")
        fatal(
            f"GGUF conversion failed (exit code {result.returncode})\n"
            f"  Check that lib/llama.cpp is up to date:\n"
            f"    cd lib/llama.cpp && git pull && cd ../.."
        )

    size_gb = gguf_path.stat().st_size / (1024 ** 3)
    ok(f"GGUF created: {gguf_path} ({size_gb:.2f} GB)")
    return gguf_path


# ---------------------------------------------------------------------------
# Step 4: Ollama registration
# ---------------------------------------------------------------------------
def register_ollama(gguf_path: Path) -> None:
    """Register the GGUF model with Ollama.

    Writes a Modelfile and runs `ollama create`. Skips if the model is
    already registered.
    """
    section("Step 4: Ollama registration")

    # Check if model already registered
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and OLLAMA_MODEL_NAME in result.stdout:
        ok(f"Model '{OLLAMA_MODEL_NAME}' already registered, skipping")
        return

    # Write Modelfile
    modelfile_path = BASELINE_DIR / "Modelfile"
    modelfile_content = (
        f"FROM {gguf_path}\n"
        f"PARAMETER num_ctx 1024\n"
        f"PARAMETER num_predict 512\n"
        f"PARAMETER temperature 0.0\n"
    )
    modelfile_path.write_text(modelfile_content)
    status(f"Modelfile written to {modelfile_path}")

    # Register with Ollama
    status(f"Registering model as '{OLLAMA_MODEL_NAME}'...")
    result = subprocess.run(
        ["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        status(f"stdout: {result.stdout}" if result.stdout else "")
        status(f"stderr: {result.stderr}" if result.stderr else "")
        fatal(
            f"Ollama registration failed (exit code {result.returncode})\n"
            f"  Make sure Ollama is running: ollama serve"
        )

    ok(f"Model registered as '{OLLAMA_MODEL_NAME}'")


# ---------------------------------------------------------------------------
# Step 5: Sanity check
# ---------------------------------------------------------------------------
def run_sanity_check() -> None:
    """Run a quick sanity check to verify the model generates coherent output."""
    section("Step 5: Sanity check")

    status(f"Running: ollama run {OLLAMA_MODEL_NAME} \"What is ALS?\"")

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL_NAME, "What is ALS?"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        status(f"stderr: {result.stderr}" if result.stderr else "")
        fatal(
            f"Sanity check failed (exit code {result.returncode})\n"
            f"  Make sure Ollama is running: ollama serve"
        )

    response = result.stdout.strip()
    if not response:
        warn("Model returned an empty response")
    else:
        # Print first 500 chars of response
        display = response[:500]
        if len(response) > 500:
            display += "..."
        status(f"\n  Response:\n  {'-' * 50}")
        for line in display.split("\n"):
            status(f"  {line}")
        status(f"  {'-' * 50}")
        ok("Model generated a response")


# ---------------------------------------------------------------------------
# Step 6: Run evaluation pipeline
# ---------------------------------------------------------------------------
def run_evaluation() -> None:
    """Run the 6-stage hallucination evaluation pipeline.

    Invokes eval/run_evaluation.py with --ollama-model and --force flags.
    Does NOT pass --instruction-format since the base instruct model uses
    its native chat template (the eval pipeline sends raw prompts to Ollama,
    and the model's built-in chat template handles formatting).
    """
    section("Step 6: Running evaluation pipeline")

    cmd = [
        sys.executable,
        str(EVAL_PIPELINE),
        "--ollama-model", OLLAMA_MODEL_NAME,
        "--force",
    ]

    status(f"Command: {' '.join(cmd)}")
    status("This may take 30-60 minutes for 160 questions...")
    status("")

    # Let stdout/stderr flow through to the console
    result = subprocess.run(cmd)

    if result.returncode != 0:
        fatal(
            f"Evaluation pipeline failed (exit code {result.returncode})\n"
            f"  Check eval/results/{OLLAMA_MODEL_NAME}/ for partial results"
        )

    ok("Evaluation pipeline completed")


# ---------------------------------------------------------------------------
# Step 7: Print summary
# ---------------------------------------------------------------------------
def print_summary() -> None:
    """Print a summary of the evaluation results."""
    section("Step 7: Results summary")

    results_dir = PROJECT_ROOT / "eval" / "results" / OLLAMA_MODEL_NAME
    if not results_dir.exists():
        warn(f"Results directory not found: {results_dir}")
        return

    # List result files
    result_files = sorted(results_dir.iterdir())
    status(f"Results directory: {results_dir}")
    status(f"Files generated: {len(result_files)}")
    for f in result_files:
        size_kb = f.stat().st_size / 1024
        status(f"  {f.name} ({size_kb:.1f} KB)")

    # Try to read scores.json for key metrics
    scores_path = results_dir / "scores.json"
    if scores_path.exists():
        try:
            with open(scores_path) as f:
                scores = json.load(f)
            # Extract accuracy if available
            if isinstance(scores, dict) and "summary" in scores:
                summary = scores["summary"]
                status(f"\n  Accuracy: {summary.get('accuracy', 'N/A')}")
                status(f"  Total questions: {summary.get('total', 'N/A')}")
            elif isinstance(scores, list) and scores:
                # Count correct scores
                correct = sum(
                    1 for s in scores
                    if isinstance(s, dict) and s.get("correct", False)
                )
                total = len(scores)
                accuracy = correct / total * 100 if total > 0 else 0
                status(f"\n  Correct: {correct}/{total} ({accuracy:.1f}%)")
        except (json.JSONDecodeError, KeyError) as e:
            warn(f"Could not parse scores.json: {e}")

    # Check for report
    reports_dir = PROJECT_ROOT / "reports"
    report_candidates = list(reports_dir.glob(f"*{OLLAMA_MODEL_NAME}*")) if reports_dir.exists() else []
    if report_candidates:
        status(f"\n  Report: {report_candidates[0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the complete ablation baseline evaluation pipeline."""
    pipeline_start = time.time()

    print(f"\n{BOLD}Ablation Baseline Evaluation{RESET}")
    print(f"Model: {OLLAMA_MODEL_NAME}")
    print(f"Pipeline: HF download -> GGUF conversion -> Ollama -> 6-stage evaluation")

    # Preflight
    check_prerequisites()

    # Load config
    config = load_config()
    model_id = config["model"]["model_id"]

    # Download model
    model_path = download_model(model_id)

    # Convert to GGUF
    gguf_path = convert_to_gguf(model_path, model_id)

    # Register with Ollama
    register_ollama(gguf_path)

    # Sanity check
    run_sanity_check()

    # Run evaluation
    run_evaluation()

    # Print summary
    print_summary()

    # Timing
    elapsed = time.time() - pipeline_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    section("Complete")
    ok(f"Ablation baseline evaluation finished in {minutes}m {seconds}s")
    status(f"Results: eval/results/{OLLAMA_MODEL_NAME}/")


if __name__ == "__main__":
    main()
