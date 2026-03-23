#!/usr/bin/env python3
"""Export merged QLoRA model to GGUF and register with Ollama.

This script takes the merged FP16 HuggingFace model from
checkpoints/qlora/merged/ (produced by merge_adapter.py) and:

1. Converts to F16 GGUF via llama.cpp convert_hf_to_gguf.py
2. Derives Q8_0 and Q4_K_M quantizations via llama-quantize
3. Registers all three quantization levels with Ollama as alslm-1b:{tag}
4. Copies Q8_0 as the default (alslm-1b:latest)
5. Re-registers the ablation baseline as alslm-1b-base for naming consistency
6. Runs a 3-prompt smoke test per quantization level with coherence filtering
7. Prints a summary table with pass/warn counts

The Modelfile uses native chat template auto-detection from GGUF metadata
(no TEMPLATE directive) and deterministic parameters matching the baseline
evaluation configuration.

Usage::

    python qlora/export_qlora.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root bootstrap (needed before qlora.utils import)
# ---------------------------------------------------------------------------
_bootstrap_root = Path(__file__).resolve().parent.parent
if str(_bootstrap_root) not in sys.path:
    sys.path.insert(0, str(_bootstrap_root))

from eval.generate_responses import is_coherent
from qlora.utils import (
    section, status, ok, warn, fatal, BOLD, RESET, GREEN, RED, YELLOW,
    PROJECT_ROOT, CONFIG_PATH, load_qlora_config,
)
MERGED_DIR = PROJECT_ROOT / "checkpoints" / "qlora" / "merged"
GGUF_DIR = PROJECT_ROOT / "checkpoints" / "qlora" / "gguf"
BASELINE_DIR = PROJECT_ROOT / "checkpoints" / "qlora" / "baseline"
GGUF_CONVERTER = PROJECT_ROOT / "lib" / "llama.cpp" / "convert_hf_to_gguf.py"
LLAMA_CPP_DIR = PROJECT_ROOT / "lib" / "llama.cpp"

OLLAMA_MODEL_NAME = "alslm-1b"
OLLAMA_BASELINE_NAME = "alslm-1b-base"
OLD_BASELINE_NAME = "als-lm-llama32-base"

QUANT_LEVELS = ["f16", "q8_0", "q4_k_m"]
GGUF_FILENAMES = {
    "f16": "alslm-1b-f16.gguf",
    "q8_0": "alslm-1b-q8_0.gguf",
    "q4_k_m": "alslm-1b-q4_k_m.gguf",
}

# Minimum file sizes (MB) to detect interrupted conversions.  A valid 1B
# model GGUF is at least this large; anything smaller is likely corrupt.
MIN_GGUF_SIZE_MB = {
    "f16": 500,
    "q8_0": 200,
    "q4_k_m": 100,
}

# Smoke test prompts covering diverse ALS topics
SMOKE_TEST_PROMPTS = [
    "What are the early symptoms of ALS and how is it diagnosed?",
    "Explain the role of TDP-43 protein aggregation in ALS pathogenesis.",
    "What is the current evidence for riluzole and edaravone in treating ALS?",
]


# ---------------------------------------------------------------------------
# Build llama-quantize
# ---------------------------------------------------------------------------
def build_llama_quantize() -> Path:
    """Build the llama-quantize binary from llama.cpp source if not present.

    Returns the path to the llama-quantize binary.
    """
    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

    if quantize_bin.is_file():
        ok(f"llama-quantize already built: {quantize_bin}")
        return quantize_bin

    status("Building llama-quantize from source...")

    if shutil.which("cmake") is None:
        fatal(
            "cmake not found in PATH (required to build llama-quantize)\n"
            "  Install: sudo apt install cmake   (Linux)\n"
            "           brew install cmake        (macOS)"
        )

    # CMake configure
    result = subprocess.run(
        ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
        capture_output=True,
        text=True,
        cwd=str(LLAMA_CPP_DIR),
    )
    if result.returncode != 0:
        fatal(
            f"CMake configure failed (exit code {result.returncode})\n"
            f"  stderr: {result.stderr[:500]}"
        )

    # CMake build (only the quantize target)
    nproc = os.cpu_count() or 1
    result = subprocess.run(
        ["cmake", "--build", "build", "--target", "llama-quantize", f"-j{nproc}"],
        capture_output=True,
        text=True,
        cwd=str(LLAMA_CPP_DIR),
    )
    if result.returncode != 0:
        fatal(
            f"CMake build failed (exit code {result.returncode})\n"
            f"  stderr: {result.stderr[:500]}"
        )

    if not quantize_bin.is_file():
        fatal(f"llama-quantize binary not found after build: {quantize_bin}")

    ok(f"llama-quantize built: {quantize_bin}")
    return quantize_bin


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def check_prerequisites() -> Path:
    """Verify all required tools and files are available.

    Returns the path to the llama-quantize binary.
    """
    section("Preflight checks")

    # Check merged model directory exists
    if not MERGED_DIR.exists():
        fatal(
            f"Merged model not found at {MERGED_DIR}\n"
            "  Fix: Run 'python qlora/merge_adapter.py' first"
        )
    merged_files = list(MERGED_DIR.iterdir())
    if not merged_files:
        fatal(
            f"Merged model directory is empty: {MERGED_DIR}\n"
            "  Fix: Run 'python qlora/merge_adapter.py' first"
        )
    ok(f"Merged model found ({len(merged_files)} files)")

    # Check config exists
    if not CONFIG_PATH.exists():
        fatal(f"Config not found at {CONFIG_PATH}")
    ok("Config found")

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

    # Check/build llama-quantize
    quantize_bin = build_llama_quantize()

    return quantize_bin


# ---------------------------------------------------------------------------
# Step 1: Load config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """Load model_id and system_prompt from configs/qlora.json."""
    section("Step 1: Loading config")
    config = load_qlora_config()
    model_id = config["model"]["model_id"]
    system_prompt = config["system_prompt"]
    status(f"Model ID: {model_id}")
    status(f"System prompt: {system_prompt[:80]}...")
    return config


# ---------------------------------------------------------------------------
# Step 2: GGUF conversion
# ---------------------------------------------------------------------------
def convert_to_gguf(quantize_bin: Path) -> dict:
    """Convert merged model to F16 GGUF, then derive Q8_0 and Q4_K_M.

    Skips individual conversions if the output file already exists
    (idempotent operation).

    Returns a dict mapping quant level to GGUF file path.
    """
    section("Step 2: GGUF conversion")

    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    gguf_paths = {}

    # Step 2a: HF -> F16 GGUF
    f16_path = GGUF_DIR / GGUF_FILENAMES["f16"]
    if f16_path.exists():
        size_mb = f16_path.stat().st_size / (1024 ** 2)
        if size_mb < MIN_GGUF_SIZE_MB["f16"]:
            warn(f"F16 GGUF exists but is only {size_mb:.0f} MB (minimum {MIN_GGUF_SIZE_MB['f16']} MB) — likely corrupt, re-converting")
            f16_path.unlink()
        else:
            ok(f"F16 GGUF already exists, skipping ({size_mb / 1024:.2f} GB)")
    if not f16_path.exists():
        status(f"Converting merged model to F16 GGUF...")
        status(f"  Input:  {MERGED_DIR}")
        status(f"  Output: {f16_path}")

        result = subprocess.run(
            [
                sys.executable,
                str(GGUF_CONVERTER),
                str(MERGED_DIR),
                "--outtype", "f16",
                "--outfile", str(f16_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            if result.stdout:
                status(f"stdout: {result.stdout[-500:]}")
            if result.stderr:
                status(f"stderr: {result.stderr[-500:]}")
            fatal(
                f"F16 GGUF conversion failed (exit code {result.returncode})\n"
                "  Check that lib/llama.cpp is up to date:\n"
                "    cd lib/llama.cpp && git pull && cd ../.."
            )

        size_gb = f16_path.stat().st_size / (1024 ** 3)
        ok(f"F16 GGUF created: {f16_path.name} ({size_gb:.2f} GB)")

    gguf_paths["f16"] = f16_path

    # Step 2b: F16 -> Q8_0
    q8_path = GGUF_DIR / GGUF_FILENAMES["q8_0"]
    if q8_path.exists():
        size_mb = q8_path.stat().st_size / (1024 ** 2)
        if size_mb < MIN_GGUF_SIZE_MB["q8_0"]:
            warn(f"Q8_0 GGUF exists but is only {size_mb:.0f} MB — likely corrupt, re-quantizing")
            q8_path.unlink()
        else:
            ok(f"Q8_0 GGUF already exists, skipping ({size_mb / 1024:.2f} GB)")
    if not q8_path.exists():
        status("Quantizing F16 -> Q8_0...")
        result = subprocess.run(
            [str(quantize_bin), str(f16_path), str(q8_path), "Q8_0"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            fatal(
                f"Q8_0 quantization failed (exit code {result.returncode})\n"
                f"  stderr: {result.stderr[:500]}"
            )
        size_gb = q8_path.stat().st_size / (1024 ** 3)
        ok(f"Q8_0 GGUF created: {q8_path.name} ({size_gb:.2f} GB)")

    gguf_paths["q8_0"] = q8_path

    # Step 2c: F16 -> Q4_K_M
    q4_path = GGUF_DIR / GGUF_FILENAMES["q4_k_m"]
    if q4_path.exists():
        size_mb = q4_path.stat().st_size / (1024 ** 2)
        if size_mb < MIN_GGUF_SIZE_MB["q4_k_m"]:
            warn(f"Q4_K_M GGUF exists but is only {size_mb:.0f} MB — likely corrupt, re-quantizing")
            q4_path.unlink()
        else:
            ok(f"Q4_K_M GGUF already exists, skipping ({size_mb / 1024:.2f} GB)")
    if not q4_path.exists():
        status("Quantizing F16 -> Q4_K_M...")
        result = subprocess.run(
            [str(quantize_bin), str(f16_path), str(q4_path), "Q4_K_M"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            fatal(
                f"Q4_K_M quantization failed (exit code {result.returncode})\n"
                f"  stderr: {result.stderr[:500]}"
            )
        size_gb = q4_path.stat().st_size / (1024 ** 3)
        ok(f"Q4_K_M GGUF created: {q4_path.name} ({size_gb:.2f} GB)")

    gguf_paths["q4_k_m"] = q4_path

    # Print file size comparison
    status("\nFile size comparison:")
    status(f"  {'File':<30} {'Size':>10}")
    status(f"  {'-' * 42}")
    for tag in QUANT_LEVELS:
        path = gguf_paths[tag]
        size_gb = path.stat().st_size / (1024 ** 3)
        status(f"  {GGUF_FILENAMES[tag]:<30} {size_gb:>9.2f} GB")

    return gguf_paths


# ---------------------------------------------------------------------------
# Step 3: Ollama registration
# ---------------------------------------------------------------------------
def register_ollama(gguf_paths: dict, system_prompt: str):
    """Register all GGUF quantization levels with Ollama.

    Generates Modelfile inline with FROM, SYSTEM, and PARAMETER directives.
    No TEMPLATE directive -- Ollama auto-detects the chat template from
    GGUF metadata for Qwen2.5 models.
    """
    section("Step 3: Ollama registration")

    for tag in QUANT_LEVELS:
        gguf_path = gguf_paths[tag]
        model_name = f"{OLLAMA_MODEL_NAME}:{tag}"

        status(f"\nRegistering {model_name}...")

        # Generate Modelfile inline (Ollama requires absolute path in FROM)
        modelfile_content = (
            f"FROM {gguf_path.resolve()}\n"
            f"SYSTEM {system_prompt}\n"
            f"PARAMETER num_ctx 1024\n"
            f"PARAMETER num_predict 512\n"
            f"PARAMETER temperature 0.0\n"
        )

        # Save Modelfile for reference
        modelfile_path = GGUF_DIR / f"Modelfile.{tag}"
        modelfile_path.write_text(modelfile_content)

        # Register with Ollama
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            if result.stderr:
                status(f"stderr: {result.stderr[:300]}")
            fatal(
                f"Ollama registration failed for {model_name} "
                f"(exit code {result.returncode})\n"
                "  Make sure Ollama is running: ollama serve"
            )

        ok(f"Registered: {model_name}")


# ---------------------------------------------------------------------------
# Step 4: Default tag copy
# ---------------------------------------------------------------------------
def set_default_tag():
    """Copy Q8_0 as the default (latest) tag."""
    section("Step 4: Default tag")

    q8_model = f"{OLLAMA_MODEL_NAME}:q8_0"
    latest_model = f"{OLLAMA_MODEL_NAME}:latest"

    status(f"Copying {q8_model} -> {latest_model}...")
    result = subprocess.run(
        ["ollama", "cp", q8_model, latest_model],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        warn(f"Failed to copy {q8_model} to {latest_model}: {result.stderr[:200]}")
    else:
        ok(f"Default: ollama run {OLLAMA_MODEL_NAME} uses Q8_0")


# ---------------------------------------------------------------------------
# Step 5: Baseline re-registration
# ---------------------------------------------------------------------------
def register_baseline():
    """Re-register the ablation baseline under the new naming scheme.

    The baseline was registered as als-lm-llama32-base in Phase 49.
    Re-register as alslm-1b-base for naming consistency.
    """
    section("Step 5: Baseline re-registration")

    # Find baseline GGUF
    baseline_gguf = None
    if BASELINE_DIR.exists():
        gguf_files = list(BASELINE_DIR.glob("*.gguf"))
        if gguf_files:
            baseline_gguf = gguf_files[0]

    if baseline_gguf is None:
        warn(
            f"Baseline GGUF not found in {BASELINE_DIR}\n"
            "  Skipping baseline re-registration"
        )
        return

    status(f"Baseline GGUF: {baseline_gguf.name}")

    # Generate Modelfile for baseline (no system prompt -- baseline has no
    # fine-tuning context, matching existing baseline Modelfile pattern)
    modelfile_content = (
        f"FROM {baseline_gguf.resolve()}\n"
        f"PARAMETER num_ctx 1024\n"
        f"PARAMETER num_predict 512\n"
        f"PARAMETER temperature 0.0\n"
    )

    modelfile_path = BASELINE_DIR / "Modelfile.alslm-1b-base"
    modelfile_path.write_text(modelfile_content)

    # Register with new name
    status(f"Registering baseline as {OLLAMA_BASELINE_NAME}...")
    result = subprocess.run(
        ["ollama", "create", OLLAMA_BASELINE_NAME, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        warn(
            f"Baseline registration failed (exit code {result.returncode})\n"
            f"  stderr: {result.stderr[:300]}"
        )
    else:
        ok(f"Baseline registered as {OLLAMA_BASELINE_NAME}")

    # Warn about old name still existing
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and OLD_BASELINE_NAME in result.stdout:
        warn(
            f"Old baseline name '{OLD_BASELINE_NAME}' still exists in Ollama\n"
            f"  To clean up: ollama rm {OLD_BASELINE_NAME}"
        )


# ---------------------------------------------------------------------------
# Step 6: Smoke test
# ---------------------------------------------------------------------------
def run_smoke_tests() -> dict:
    """Run 3-prompt smoke test per quantization level with coherence filtering.

    Returns a dict with per-quant-level results for the summary table.
    """
    section("Step 6: Smoke test")

    results = {}

    for tag in QUANT_LEVELS:
        model_name = f"{OLLAMA_MODEL_NAME}:{tag}"
        results[tag] = {"passed": 0, "warned": 0, "responses": []}

        status(f"\nTesting {model_name}...")

        for i, prompt in enumerate(SMOKE_TEST_PROMPTS, 1):
            status(f"\n  Prompt {i}: \"{prompt}\"")

            try:
                result = subprocess.run(
                    ["ollama", "run", model_name, prompt],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                response = result.stdout.strip()

                if result.returncode != 0:
                    warn(f"  Inference failed (exit code {result.returncode})")
                    results[tag]["warned"] += 1
                    results[tag]["responses"].append("(error)")
                    continue

                # Print full response for manual review
                status(f"  Response:")
                status(f"  {'-' * 50}")
                for line in response.split("\n"):
                    status(f"    {line}")
                status(f"  {'-' * 50}")

                # Coherence check
                if is_coherent(response):
                    ok(f"  Coherent ({len(response)} chars)")
                    results[tag]["passed"] += 1
                else:
                    warn(f"  Incoherent or too short ({len(response)} chars)")
                    results[tag]["warned"] += 1

                results[tag]["responses"].append(response)

            except subprocess.TimeoutExpired:
                warn(f"  Timed out after 300 seconds")
                results[tag]["warned"] += 1
                results[tag]["responses"].append("(timeout)")
            except Exception as e:
                warn(f"  Error: {e}")
                results[tag]["warned"] += 1
                results[tag]["responses"].append(f"(error: {e})")

    return results


# ---------------------------------------------------------------------------
# Step 7: Summary table
# ---------------------------------------------------------------------------
def print_summary_table(smoke_results: dict):
    """Print a summary table with pass/warn counts per quantization level."""
    section("Step 7: Summary")

    total_prompts = len(SMOKE_TEST_PROMPTS)

    status(f"\n  {'Quant Level':<15} {'Passed':>8} {'Warned':>8} {'Status':>10}")
    status(f"  {'-' * 43}")

    all_healthy = True
    for tag in QUANT_LEVELS:
        r = smoke_results[tag]
        if r["warned"] == 0:
            tag_status = f"{GREEN}PASS{RESET}"
        elif r["passed"] > 0:
            tag_status = f"{YELLOW}WARN{RESET}"
            all_healthy = False
        else:
            tag_status = f"{RED}FAIL{RESET}"
            all_healthy = False

        status(
            f"  {tag:<15} {r['passed']:>5}/{total_prompts}"
            f"   {r['warned']:>5}/{total_prompts}"
            f"   {tag_status}"
        )

    status(f"  {'-' * 43}")

    if all_healthy:
        ok("All quantization levels passed smoke test")
    else:
        warn("Some responses flagged -- review output above")
        status("  Phase 51 evaluation is the definitive test")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the complete GGUF export and Ollama registration pipeline."""
    pipeline_start = time.time()

    print(f"\n{BOLD}QLoRA GGUF Export and Ollama Registration{RESET}")
    print(f"Pipeline: GGUF convert -> Quantize -> Ollama register -> Smoke test")

    # Preflight
    quantize_bin = check_prerequisites()

    # Load config
    config = load_config()
    system_prompt = config["system_prompt"]

    # GGUF conversion
    gguf_paths = convert_to_gguf(quantize_bin)

    # Ollama registration
    register_ollama(gguf_paths, system_prompt)

    # Default tag
    set_default_tag()

    # Baseline re-registration
    register_baseline()

    # Smoke test
    smoke_results = run_smoke_tests()

    # Summary
    print_summary_table(smoke_results)

    # Timing
    elapsed = time.time() - pipeline_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    section("Complete")
    ok(f"Export pipeline finished in {minutes}m {seconds}s")
    status(f"GGUF files: {GGUF_DIR}")
    status(f"Ollama models: {OLLAMA_MODEL_NAME}:f16, {OLLAMA_MODEL_NAME}:q8_0, {OLLAMA_MODEL_NAME}:q4_k_m")
    status(f"Default: {OLLAMA_MODEL_NAME} (Q8_0)")
    status(f"Baseline: {OLLAMA_BASELINE_NAME}")
    status(f"\nRun: ollama run {OLLAMA_MODEL_NAME}")


if __name__ == "__main__":
    main()
