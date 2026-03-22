#!/usr/bin/env python3
"""Merge trained QLoRA adapter into the FP16 base model.

This script loads the Qwen2.5-1.5B-Instruct base model at full FP16 precision
(no BitsAndBytes quantization), applies the trained LoRA adapter from
checkpoints/qlora/adapter/best/, merges the adapter weights into the base model
via PEFT merge_and_unload(), and saves the resulting full-precision HuggingFace
model to checkpoints/qlora/merged/.

The merged model is approximately 3 GB (1.54B parameters at FP16) and serves
as the input for GGUF conversion in export_qlora.py.

Note: Uses device_map="cpu" to avoid VRAM pressure during the merge, which is
a pure weight arithmetic operation that does not need GPU acceleration.

Usage::

    python qlora/merge_adapter.py
"""

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root discovery (existing project pattern)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qlora.utils import section, status, ok, warn, fatal, BOLD, RESET

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "configs" / "qlora.json"
ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "qlora" / "adapter" / "best"
MERGED_DIR = PROJECT_ROOT / "checkpoints" / "qlora" / "merged"


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def check_prerequisites():
    """Verify all required files and directories exist before starting."""
    section("Preflight checks")

    # Check config exists
    if not CONFIG_PATH.exists():
        fatal(
            f"Config not found at {CONFIG_PATH}\n"
            "  Fix: Ensure configs/qlora.json exists in the project root"
        )
    ok("Config found")

    # Check adapter directory exists
    if not ADAPTER_DIR.exists():
        fatal(
            f"Adapter not found at {ADAPTER_DIR}\n"
            "  Fix: Run 'python qlora/train_qlora.py' first to train the adapter\n"
            "  The trained adapter should be saved at checkpoints/qlora/adapter/best/"
        )
    ok("Adapter directory found")

    # Check adapter has files
    adapter_files = list(ADAPTER_DIR.iterdir())
    if not adapter_files:
        fatal(
            f"Adapter directory is empty: {ADAPTER_DIR}\n"
            "  Fix: Run 'python qlora/train_qlora.py' to produce adapter files"
        )
    ok(f"Adapter contains {len(adapter_files)} files")


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
    status(f"Adapter path: {ADAPTER_DIR}")
    status(f"Merged output: {MERGED_DIR}")
    return config


# ---------------------------------------------------------------------------
# Step 2: Load base model in FP16
# ---------------------------------------------------------------------------
def load_base_model(model_id: str):
    """Load the base model at full FP16 precision (no quantization).

    Uses device_map="cpu" to avoid VRAM pressure. The merge operation is
    weight arithmetic and does not need GPU acceleration.
    """
    section("Step 2: Loading base model in FP16")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    status(f"Loading {model_id} with torch.float16 on CPU...")
    status("This may take 1-2 minutes for download + loading...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    ok("Base model loaded in FP16")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ok("Tokenizer loaded")

    # Report model size
    param_count = sum(p.numel() for p in base_model.parameters())
    status(f"Parameters: {param_count:,} ({param_count / 1e9:.2f}B)")

    return base_model, tokenizer


# ---------------------------------------------------------------------------
# Step 3: Load and merge adapter
# ---------------------------------------------------------------------------
def merge_adapter(base_model, adapter_path: Path):
    """Load LoRA adapter onto base model and merge weights.

    Uses PeftModel.from_pretrained() to load the adapter, then
    merge_and_unload() to fold the LoRA weights into the base model,
    producing a standard HuggingFace model with no adapter overhead.
    """
    section("Step 3: Loading and merging adapter")

    from peft import PeftModel

    status(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    ok("Adapter loaded")

    status("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    ok("Adapter merged via merge_and_unload()")

    return model


# ---------------------------------------------------------------------------
# Step 4: Save merged model
# ---------------------------------------------------------------------------
def save_merged_model(model, tokenizer, output_dir: Path):
    """Save the merged model and tokenizer to disk.

    Verifies the output directory size is in the expected range (2.5-4.0 GB)
    to catch double-quantization issues where the merged model would be
    suspiciously small (~500 MB instead of ~3 GB).
    """
    section("Step 4: Saving merged model")

    output_dir.mkdir(parents=True, exist_ok=True)
    status(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    ok("Model saved")

    status("Saving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    ok("Tokenizer saved")

    # Verify merged model size
    merged_size_gb = sum(
        f.stat().st_size for f in output_dir.iterdir() if f.is_file()
    ) / (1024 ** 3)

    if merged_size_gb < 2.0:
        fatal(
            f"Merged model is only {merged_size_gb:.2f} GB (expected ~3 GB)\n"
            "  This indicates double quantization -- the base model was loaded "
            "in 4-bit instead of FP16.\n"
            "  Fix: Ensure the base model is loaded with torch_dtype=torch.float16 "
            "and no quantization_config."
        )

    if merged_size_gb > 4.0:
        warn(f"Merged model is {merged_size_gb:.2f} GB (expected ~3 GB) -- larger than expected")
    else:
        ok(f"Merged model size: {merged_size_gb:.2f} GB")

    # List output files
    status("\nOutput files:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 ** 2)
        status(f"  {f.name} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the complete adapter merge pipeline."""
    pipeline_start = time.time()

    print(f"\n{BOLD}QLoRA Adapter Merge{RESET}")
    print(f"Pipeline: Load FP16 base -> Apply adapter -> Merge -> Save")

    # Preflight
    check_prerequisites()

    # Load config
    config = load_config()
    model_id = config["model"]["model_id"]

    # Load base model in FP16
    base_model, tokenizer = load_base_model(model_id)

    # Load and merge adapter
    merged_model = merge_adapter(base_model, ADAPTER_DIR)

    # Save merged model
    save_merged_model(merged_model, tokenizer, MERGED_DIR)

    # Timing
    elapsed = time.time() - pipeline_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    section("Complete")
    ok(f"Adapter merge finished in {minutes}m {seconds}s")
    status(f"Merged model: {MERGED_DIR}")
    status(f"\nNext step: python qlora/export_qlora.py")


if __name__ == "__main__":
    main()
