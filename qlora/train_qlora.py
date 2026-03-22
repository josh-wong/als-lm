#!/usr/bin/env python3
"""QLoRA fine-tuning script: trains Qwen2.5-1.5B-Instruct with QLoRA on 873
ALS instruction pairs using TRL's SFTTrainer.

Features:
- Completion-only loss masking via prompt-completion dataset format (TRAIN-02)
- TensorBoard logging with per-epoch validation loss (TRAIN-03)
- 6-prompt catastrophic forgetting spot-check after training (TRAIN-04)
- Best-by-val-loss adapter checkpoint saved to checkpoints/qlora/adapter/best/

All hyperparameters are loaded from configs/qlora.json (single source of truth).
No CLI arguments -- run directly with `python qlora/train_qlora.py`.

Note: Uses Qwen/Qwen2.5-1.5B-Instruct as a temporary substitute while
awaiting gated access approval for meta-llama/Llama-3.2-1B-Instruct.
The QLoRA pipeline is model-agnostic -- switching back to Llama requires
only changing model_id in configs/qlora.json.

Usage::

    python qlora/train_qlora.py
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

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from eval.generate_responses import is_coherent
from qlora.utils import print_pass, print_fail


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_ASSISTANT_TAG = "<|im_start|>assistant\n"
CONFIG_PATH = PROJECT_ROOT / "configs" / "qlora.json"
TRAIN_JSONL = PROJECT_ROOT / "data" / "instruction" / "qlora" / "train.jsonl"
VAL_JSONL = PROJECT_ROOT / "data" / "instruction" / "qlora" / "val.jsonl"


# ---------------------------------------------------------------------------
# Dataset loading: prompt-completion format for completion-only loss masking
# ---------------------------------------------------------------------------
def load_prompt_completion(jsonl_path: str, assistant_tag: str = _DEFAULT_ASSISTANT_TAG) -> Dataset:
    """Load pre-formatted JSONL and restructure as prompt-completion pairs.

    Each line has {"text": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>\\n"}

    Split at the last occurrence of ``<|im_start|>assistant\\n`` into:

    - prompt: everything up to and including the assistant tag
    - completion: the assistant response text + ``<|im_end|>\\n``

    This format enables SFTTrainer's native completion_only_loss mechanism,
    which builds a completion_mask automatically so loss is computed only
    on assistant response tokens (TRAIN-02).

    Args:
        jsonl_path: Path to a JSONL file where each line has a "text" field.

    Returns:
        A HuggingFace Dataset with "prompt" and "completion" columns.
    """
    prompts = []
    completions = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record["text"]
            idx = text.rfind(assistant_tag)
            if idx == -1:
                print(f"FATAL: No assistant tag found on line {line_num}: "
                      f"{text[:100]}...")
                print("  Fix: Re-run `python qlora/format_dataset.py` to "
                      "regenerate the dataset.")
                sys.exit(1)
            split_point = idx + len(assistant_tag)
            prompts.append(text[:split_point])
            completions.append(text[split_point:])
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def _derive_assistant_tag(tokenizer) -> str:
    """Derive the assistant turn prefix from the tokenizer's chat template.

    Computes the difference between a user-only chat template rendered with
    and without ``add_generation_prompt``, which yields the model-specific
    assistant tag regardless of the model architecture.

    For Qwen: ``<|im_start|>assistant\\n``
    For Llama: ``<|start_header_id|>assistant<|end_header_id|>\\n\\n``
    """
    messages = [{"role": "user", "content": "x"}]
    with_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    without_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tag = with_prompt[len(without_prompt):]
    if not tag:
        raise ValueError(
            "Could not derive assistant tag from tokenizer chat template. "
            "The tokenizer may not support add_generation_prompt."
        )
    return tag


# ---------------------------------------------------------------------------
# Spot-check prompts (TRAIN-04: catastrophic forgetting detection)
# ---------------------------------------------------------------------------
SPOT_CHECK_PROMPTS = [
    # 3 general-knowledge prompts
    {
        "category": "GENERAL",
        "messages": [
            {"role": "user", "content": "Explain how photosynthesis works in simple terms"},
        ],
    },
    {
        "category": "GENERAL",
        "messages": [
            {"role": "user", "content": "Write a haiku about the ocean"},
        ],
    },
    {
        "category": "GENERAL",
        "messages": [
            {"role": "user", "content": "Name three capital cities in South America and one fact about each"},
        ],
    },
    # 3 ALS-domain prompts
    {
        "category": "ALS",
        "messages": [
            {"role": "user", "content": "What are the earliest symptoms of ALS?"},
        ],
    },
    {
        "category": "ALS",
        "messages": [
            {"role": "user", "content": "Explain the role of TDP-43 in ALS pathophysiology"},
        ],
    },
    {
        "category": "ALS",
        "messages": [
            {"role": "user", "content": "What is riluzole and how does it help ALS patients?"},
        ],
    },
]


def run_spot_check(model, tokenizer) -> list[dict]:
    """Run generation on spot-check prompts and assess coherence.

    Uses greedy decoding (do_sample=False) for reproducibility. Applies
    the is_coherent() heuristic from eval/generate_responses.py to each
    response to detect degenerate output (empty, repetitive, token salad).

    Args:
        model: The trained PeftModel (still on GPU after training).
        tokenizer: The tokenizer with chat template support.

    Returns:
        List of result dicts with category, prompt, response, and coherent verdict.
    """
    model.eval()
    results = []

    for p in SPOT_CHECK_PROMPTS:
        # Format as chat template with generation prompt
        input_text = tokenizer.apply_chat_template(
            p["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy decoding for reproducibility
            )

        # Decode only the new tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append({
            "category": p["category"],
            "prompt": p["messages"][-1]["content"],
            "response": response,
            "coherent": is_coherent(response),
        })

    return results


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  QLoRA Training: ALS Domain Adaptation")
    print("=" * 60)

    if not CONFIG_PATH.exists():
        print(f"\nFATAL: Config not found at {CONFIG_PATH}")
        print("  Fix: Ensure configs/qlora.json exists in the project root.")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    model_id = config["model"]["model_id"]
    print(f"\n  Model:  {model_id}")
    print(f"  Config: {CONFIG_PATH}")

    # ------------------------------------------------------------------
    # 2. Check CUDA
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("\nFATAL: CUDA not available. QLoRA training requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  GPU:    {gpu_name} ({gpu_vram_gb:.1f} GB)")

    # ------------------------------------------------------------------
    # 3. Load tokenizer and derive assistant tag
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Loading tokenizer...")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assistant_tag = _derive_assistant_tag(tokenizer)
    print(f"  Tokenizer:     {model_id}")
    print(f"  Assistant tag:  {assistant_tag!r}")

    # ------------------------------------------------------------------
    # 4. Load datasets
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Loading datasets...")
    print(f"{'=' * 60}")

    if not TRAIN_JSONL.exists():
        print(f"\nFATAL: Training data not found at {TRAIN_JSONL}")
        print("  Fix: Run `python qlora/format_dataset.py` first.")
        sys.exit(1)
    if not VAL_JSONL.exists():
        print(f"\nFATAL: Validation data not found at {VAL_JSONL}")
        print("  Fix: Run `python qlora/format_dataset.py` first.")
        sys.exit(1)

    train_ds = load_prompt_completion(str(TRAIN_JSONL), assistant_tag)
    val_ds = load_prompt_completion(str(VAL_JSONL), assistant_tag)

    print(f"  Train examples: {len(train_ds)}")
    print(f"  Val examples:   {len(val_ds)}")
    print(f"  Columns:        {train_ds.column_names}")

    # ------------------------------------------------------------------
    # 5. Load model with 4-bit quantization
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Loading model...")
    print(f"{'=' * 60}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except Exception as e:
        error_str = str(e).lower()
        if "401" in error_str or "403" in error_str or "gated" in error_str:
            print(f"\nFATAL: Cannot access gated model {model_id}")
            print("  Fix:")
            print(f"  1. Visit https://huggingface.co/{model_id}")
            print("  2. Accept the license agreement")
            print("  3. Run: huggingface-cli login")
            print("  4. Re-run this script")
            sys.exit(1)
        raise

    # ------------------------------------------------------------------
    # 6. Apply LoRA
    # ------------------------------------------------------------------
    lora_section = config["lora"]
    lora_config = LoraConfig(
        r=lora_section["r"],
        lora_alpha=lora_section["lora_alpha"],
        target_modules=lora_section["target_modules"],
        lora_dropout=lora_section["lora_dropout"],
        bias=lora_section["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    print()
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 7. Configure training
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Configuring training...")
    print(f"{'=' * 60}")

    training_section = config["training"]
    output_dir = str(PROJECT_ROOT / training_section["output_dir"])
    logging_dir = str(PROJECT_ROOT / "checkpoints" / "qlora" / "logs")
    max_seq_length = training_section.get("max_seq_length", 512)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=training_section["num_train_epochs"],
        per_device_train_batch_size=training_section["per_device_train_batch_size"],
        gradient_accumulation_steps=training_section["gradient_accumulation_steps"],
        learning_rate=training_section["learning_rate"],
        warmup_ratio=training_section["warmup_ratio"],
        max_length=max_seq_length,
        bf16=True,  # Empirically validated in Phase 47
        logging_dir=logging_dir,
        logging_steps=training_section["logging_steps"],
        report_to="tensorboard",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=4,  # 3 epoch checkpoints + 1 best
        load_best_model_at_end=True,
        remove_unused_columns=False,  # Keep prompt/completion columns
        dataloader_pin_memory=False,  # Avoid CUDA issues on WSL2
        seed=42,
    )

    effective_batch = (
        training_section["per_device_train_batch_size"]
        * training_section["gradient_accumulation_steps"]
    )
    steps_per_epoch = len(train_ds) // effective_batch
    total_steps = steps_per_epoch * training_section["num_train_epochs"]

    print(f"  Output dir:     {output_dir}")
    print(f"  Logging dir:    {logging_dir}")
    print(f"  Epochs:         {training_section['num_train_epochs']}")
    print(f"  Batch size:     {training_section['per_device_train_batch_size']}")
    print(f"  Grad accum:     {training_section['gradient_accumulation_steps']}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Steps/epoch:    ~{steps_per_epoch}")
    print(f"  Total steps:    ~{total_steps}")
    print(f"  Learning rate:  {training_section['learning_rate']}")
    print(f"  Max length:     {max_seq_length}")
    print(f"  BF16:           True")
    print(f"  Seed:           42")

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Training...")
    print(f"{'=' * 60}\n")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    train_result = trainer.train()

    t_end = time.time()
    training_time = t_end - t_start
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Extract final losses
    train_loss = train_result.training_loss
    eval_losses = [
        entry["eval_loss"]
        for entry in trainer.state.log_history
        if "eval_loss" in entry
    ]
    best_eval_loss = min(eval_losses) if eval_losses else float("nan")

    print(f"\n{'=' * 60}")
    print("  Training complete")
    print(f"{'=' * 60}")
    print(f"  Training time:  {training_time / 60:.1f} min ({training_time:.0f}s)")
    print(f"  Final train loss: {train_loss:.4f}")
    print(f"  Best eval loss:   {best_eval_loss:.4f}")
    print(f"  Peak VRAM:        {peak_vram_gb:.2f} GB")

    # ------------------------------------------------------------------
    # 9. Save best adapter
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Saving best adapter...")
    print(f"{'=' * 60}")

    best_dir = Path(output_dir) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    adapter_file = best_dir / "adapter_model.safetensors"
    if adapter_file.exists():
        size_mb = adapter_file.stat().st_size / (1024 * 1024)
        print(f"  Adapter saved to: {best_dir}")
        print(f"  Adapter size:     {size_mb:.1f} MB")
    else:
        print(f"  Adapter saved to: {best_dir}")
        print("  WARNING: adapter_model.safetensors not found after save")

    # ------------------------------------------------------------------
    # 10. Spot-check (TRAIN-04: catastrophic forgetting detection)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Spot-check: catastrophic forgetting detection")
    print(f"{'=' * 60}")

    results = run_spot_check(model, tokenizer)

    general_coherent = 0
    als_coherent = 0
    general_total = 0
    als_total = 0

    for r in results:
        category = r["category"]
        verdict = "PASS" if r["coherent"] else "FAIL"
        color = "\033[92m" if r["coherent"] else "\033[91m"
        reset = "\033[0m"

        if category == "GENERAL":
            general_total += 1
            if r["coherent"]:
                general_coherent += 1
        else:
            als_total += 1
            if r["coherent"]:
                als_coherent += 1

        print(f"\n  [{category}] {r['prompt']}")
        print(f"  Response: {r['response'][:300]}")
        if len(r["response"]) > 300:
            print(f"  ... ({len(r['response'])} chars total)")
        print(f"  Coherence: {color}[{verdict}]{reset}")

    total_coherent = general_coherent + als_coherent
    print(f"\n{'=' * 60}")
    print(f"  Spot-check: {total_coherent}/6 coherent")
    print(f"    General:  {general_coherent}/{general_total}")
    print(f"    ALS:      {als_coherent}/{als_total}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 11. Final summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  QLoRA Training Complete")
    print(f"{'=' * 60}")
    print(f"  Model:            {model_id}")
    print(f"  Training time:    {training_time / 60:.1f} min")
    print(f"  Final train loss: {train_loss:.4f}")
    print(f"  Best eval loss:   {best_eval_loss:.4f}")
    print(f"  Peak VRAM:        {peak_vram_gb:.2f} GB")
    print(f"  Adapter:          {best_dir}")
    print(f"  TensorBoard:      {logging_dir}")
    print(f"  Spot-check:       {total_coherent}/6 coherent")
    print()


if __name__ == "__main__":
    main()
