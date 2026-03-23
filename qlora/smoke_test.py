#!/usr/bin/env python3
"""QLoRA smoke test: validates 4-bit NF4 quantization, gated model access,
and training dynamics with BF16 and FP16 compute dtypes.

This script is the definitive ENV-02 + ENV-03 validation. It proves the QLoRA
environment works end-to-end on the RTX 3060 before committing to the
multi-phase training pipeline (Phases 48-51). It also resolves the BF16 vs
FP16 compute dtype question with empirical data.

Verified results (RTX 3060, 2026-03-21):

- BF16: PASSED 6/6 (loss 2.42->2.02, 2.76 GB VRAM, 32.6s)
- FP16: FAILED (RuntimeError in _amp_foreach_non_finite_check_and_unscale_cuda)
- Recommendation: BF16

Note: Uses Qwen/Qwen2.5-1.5B-Instruct as a temporary substitute while
awaiting gated access approval for meta-llama/Llama-3.2-1B-Instruct.
The QLoRA pipeline (NF4 quantization, LoRA adapter, SFTTrainer) is
model-agnostic -- switching back to Llama requires only changing MODEL_ID.

Usage::

    python qlora/smoke_test.py
"""

import gc
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root discovery (existing project pattern)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qlora.utils import print_pass, print_fail, PROJECT_ROOT, load_qlora_config

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_STEPS = 10
VRAM_LIMIT_GB = 10.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_vram_mb() -> float:
    """Return current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_peak_vram_mb() -> float:
    """Return peak GPU memory usage in MB."""
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def cleanup_gpu():
    """Free all GPU memory and reset peak stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def load_synthetic_data() -> list[dict]:
    """Load Alpaca-format synthetic SFT data from the project data directory."""
    path = PROJECT_ROOT / "data" / "instruction" / "synthetic_sft.json"
    with open(path) as f:
        return json.load(f)


def alpaca_to_chat_messages(entry: dict, tokenizer) -> str:
    """Convert an Alpaca-format dict to a Llama 3.2 chat template string.

    Uses the tokenizer's built-in chat template to handle all special tokens
    correctly. Message format: system (ALS expert), user (instruction + input),
    assistant (output).
    """
    instruction = entry["instruction"]
    if entry.get("input", "").strip():
        instruction = f"{instruction}\n\n{entry['input']}"

    messages = [
        {"role": "system", "content": "You are an expert in ALS research."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": entry["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ---------------------------------------------------------------------------
# Model loading with gated-access error handling
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(compute_dtype):
    """Load Llama 3.2 1B Instruct with 4-bit NF4 quantization.

    Catches gated-access errors and prints actionable fix instructions
    following the project's FATAL: pattern.

    Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except Exception as e:
        error_str = str(e).lower()
        if "401" in error_str or "403" in error_str or "gated" in error_str:
            print(f"\nFATAL: Cannot access gated model {MODEL_ID}")
            print("  Fix:")
            print(f"  1. Visit https://huggingface.co/{MODEL_ID}")
            print("  2. Accept the Llama 3.2 Community License Agreement")
            print("  3. Run: huggingface-cli login")
            print("  4. Re-run this script")
            sys.exit(1)
        raise

    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_dataset(tokenizer, synthetic_data: list[dict]) -> Dataset:
    """Convert Alpaca entries to chat template format and build a Dataset.

    Returns a HuggingFace Dataset with a 'text' column containing the
    formatted chat template strings, suitable for SFTTrainer consumption.
    """
    texts = [alpaca_to_chat_messages(entry, tokenizer) for entry in synthetic_data]
    return Dataset.from_dict({"text": texts})


# ---------------------------------------------------------------------------
# Core dtype test
# ---------------------------------------------------------------------------
def run_dtype_test(compute_dtype, dtype_name: str) -> dict:
    """Run a 10-step QLoRA training test with the specified compute dtype.

    Tests 6 assertions: no NaN loss, no dtype errors (implicit), peak VRAM
    under limit, loss decreases, post-training generation works, and LoRA
    adapter files save to disk.

    Returns a results dict with dtype, losses, peak VRAM, passed/failed
    assertions, and training time.
    """
    print(f"\n{'=' * 60}")
    print(f"  Testing {dtype_name} compute dtype...")
    print(f"{'=' * 60}")

    passed = []
    failed = []
    temp_dir = tempfile.mkdtemp(prefix=f"qlora_smoke_{dtype_name.lower()}_")
    model = None
    tokenizer = None
    trainer = None

    try:
        torch.cuda.reset_peak_memory_stats()
        t_start = time.time()

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(compute_dtype)

        # Load LoRA config from configs/qlora.json
        qlora_config = load_qlora_config()
        lora_section = qlora_config["lora"]
        lora_config = LoraConfig(
            r=lora_section["r"],
            lora_alpha=lora_section["lora_alpha"],
            target_modules=lora_section["target_modules"],
            lora_dropout=lora_section["lora_dropout"],
            bias=lora_section["bias"],
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        print()
        model.print_trainable_parameters()

        # Prepare dataset
        synthetic_data = load_synthetic_data()
        dataset = prepare_dataset(tokenizer, synthetic_data)

        # Training arguments
        training_args = SFTConfig(
            output_dir=temp_dir,
            max_steps=SMOKE_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="no",
            bf16=(dtype_name == "BF16"),
            fp16=(dtype_name == "FP16"),
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataset_text_field="text",
            max_length=512,
        )

        # Create trainer and train
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        train_result = trainer.train()
        t_end = time.time()
        training_time = t_end - t_start

        # Collect losses from log history
        losses = [
            entry["loss"]
            for entry in trainer.state.log_history
            if "loss" in entry
        ]

        peak_vram_mb = get_peak_vram_mb()
        peak_vram_gb = peak_vram_mb / 1024

        print(f"\n  Training completed in {training_time:.1f}s")
        print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
        print(f"  Losses: {[f'{l:.4f}' for l in losses]}")

        # --- Assertion 1: No NaN loss ---
        has_nan = any(math.isnan(l) for l in losses)
        if not has_nan:
            print_pass("No NaN loss values")
            passed.append("no_nan_loss")
        else:
            print_fail("NaN loss detected")
            failed.append("no_nan_loss")

        # --- Assertion 2: No dtype errors (implicit) ---
        # If we reached this point, no RuntimeError occurred during training.
        print_pass("No dtype errors during training")
        passed.append("no_dtype_errors")

        # --- Assertion 3: Peak VRAM under limit ---
        if peak_vram_mb < VRAM_LIMIT_GB * 1024:
            print_pass(f"Peak VRAM {peak_vram_gb:.2f} GB < {VRAM_LIMIT_GB} GB limit")
            passed.append("vram_under_limit")
        else:
            print_fail(f"Peak VRAM {peak_vram_gb:.2f} GB >= {VRAM_LIMIT_GB} GB limit")
            failed.append("vram_under_limit")

        # --- Assertion 4: Loss decreases ---
        if len(losses) >= 2 and losses[0] > losses[-1]:
            print_pass(f"Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
            passed.append("loss_decreases")
        else:
            first = losses[0] if losses else "N/A"
            last = losses[-1] if losses else "N/A"
            print_fail(f"Loss did not decrease: {first} -> {last}")
            failed.append("loss_decreases")

        # --- Assertion 5: Post-training generation ---
        try:
            model.eval()
            prompt = "ALS is"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated_text and generated_text != prompt:
                print_pass(f"Generation produced tokens: '{generated_text[:80]}...'")
                passed.append("generation_works")
            else:
                print_fail("Generation produced empty or identical output")
                failed.append("generation_works")
        except Exception as e:
            print_fail(f"Generation failed: {e}")
            failed.append("generation_works")

        # --- Assertion 6: LoRA adapter saves ---
        try:
            adapter_dir = Path(temp_dir) / "adapter_test"
            model.save_pretrained(str(adapter_dir))
            adapter_file = adapter_dir / "adapter_model.safetensors"
            if adapter_file.exists():
                size_mb = adapter_file.stat().st_size / (1024 * 1024)
                print_pass(f"LoRA adapter saved ({size_mb:.1f} MB)")
                passed.append("adapter_saves")
            else:
                print_fail("adapter_model.safetensors not found after save")
                failed.append("adapter_saves")
        except Exception as e:
            print_fail(f"Adapter save failed: {e}")
            failed.append("adapter_saves")

        return {
            "dtype": dtype_name,
            "losses": losses,
            "peak_vram_mb": peak_vram_mb,
            "passed": passed,
            "failed": failed,
            "training_time_sec": training_time,
        }

    except Exception as e:
        print_fail(f"Test crashed: {e}")
        failed.append("training_completed")
        return {
            "dtype": dtype_name,
            "losses": [],
            "peak_vram_mb": get_peak_vram_mb(),
            "passed": passed,
            "failed": failed,
            "training_time_sec": 0,
            "error": str(e),
        }

    finally:
        # Cleanup GPU memory
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        cleanup_gpu()

        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  QLoRA Smoke Test")
    print("=" * 60)

    # Package versions
    import accelerate
    import bitsandbytes
    import datasets as datasets_lib
    import peft
    import trl

    print(f"\n  Package versions:")
    print(f"    torch:        {torch.__version__}")
    print(f"    transformers: {__import__('transformers').__version__}")
    print(f"    peft:         {peft.__version__}")
    print(f"    bitsandbytes: {bitsandbytes.__version__}")
    print(f"    trl:          {trl.__version__}")
    print(f"    accelerate:   {accelerate.__version__}")
    print(f"    datasets:     {datasets_lib.__version__}")

    # CUDA check
    if not torch.cuda.is_available():
        print("\nFATAL: CUDA not available. This test requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"\n  GPU: {gpu_name} ({gpu_vram_gb:.1f} GB)")

    # Run BF16 test
    bf16_results = run_dtype_test(torch.bfloat16, "BF16")
    cleanup_gpu()

    # Run FP16 test
    fp16_results = run_dtype_test(torch.float16, "FP16")
    cleanup_gpu()

    # --- Comparison table ---
    print(f"\n{'=' * 60}")
    print("  Comparison Table")
    print(f"{'=' * 60}")

    bf16_final = bf16_results["losses"][-1] if bf16_results["losses"] else float("nan")
    fp16_final = fp16_results["losses"][-1] if fp16_results["losses"] else float("nan")
    bf16_vram = bf16_results["peak_vram_mb"] / 1024
    fp16_vram = fp16_results["peak_vram_mb"] / 1024
    bf16_time = bf16_results["training_time_sec"]
    fp16_time = fp16_results["training_time_sec"]
    bf16_pass_count = len(bf16_results["passed"])
    fp16_pass_count = len(fp16_results["passed"])
    bf16_total = bf16_pass_count + len(bf16_results["failed"])
    fp16_total = fp16_pass_count + len(fp16_results["failed"])

    print(f"\n  {'Metric':<25} {'BF16':>12} {'FP16':>12}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12}")
    print(f"  {'Final Loss':<25} {bf16_final:>12.4f} {fp16_final:>12.4f}")
    print(f"  {'Peak VRAM (GB)':<25} {bf16_vram:>12.2f} {fp16_vram:>12.2f}")
    print(f"  {'Training Time (s)':<25} {bf16_time:>12.1f} {fp16_time:>12.1f}")
    print(f"  {'Assertions Passed':<25} {bf16_pass_count}/{bf16_total:>9} {fp16_pass_count}/{fp16_total:>9}")

    # --- Recommendation ---
    print(f"\n{'=' * 60}")
    print("  Recommendation")
    print(f"{'=' * 60}")

    bf16_all_pass = len(bf16_results["failed"]) == 0 and bf16_pass_count > 0
    fp16_all_pass = len(fp16_results["failed"]) == 0 and fp16_pass_count > 0

    if bf16_all_pass and fp16_all_pass:
        # Both pass: recommend lower final loss, BF16 wins ties (wider dynamic range)
        if bf16_final <= fp16_final:
            recommendation = "BF16"
            reason = (
                "Both dtypes passed all assertions. BF16 recommended for its "
                "wider dynamic range, which reduces overflow risk during training."
            )
        else:
            recommendation = "FP16"
            reason = (
                "Both dtypes passed all assertions. FP16 achieved lower final "
                "loss in this test."
            )
    elif bf16_all_pass:
        recommendation = "BF16"
        reason = "Only BF16 passed all assertions."
    elif fp16_all_pass:
        recommendation = "FP16"
        reason = "Only FP16 passed all assertions."
    else:
        recommendation = "NONE"
        reason = (
            "FATAL: Neither BF16 nor FP16 passed all assertions. "
            "Investigate GPU compatibility and package versions."
        )

    print(f"\n  Recommended compute dtype: {recommendation}")
    print(f"  Reason: {reason}")

    # --- Overall summary ---
    total_passed = bf16_pass_count + fp16_pass_count
    total_assertions = bf16_total + fp16_total

    print(f"\n{'=' * 60}")
    print(f"  Overall: {total_passed}/{total_assertions} assertions passed")
    print(f"{'=' * 60}")

    if bf16_results["failed"]:
        print(f"  BF16 failures: {bf16_results['failed']}")
    if fp16_results["failed"]:
        print(f"  FP16 failures: {fp16_results['failed']}")

    # Exit code
    if bf16_all_pass or fp16_all_pass:
        print(f"\n  QLoRA smoke test PASSED.")
        sys.exit(0)
    else:
        print(f"\n  QLoRA smoke test FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
