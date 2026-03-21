#!/usr/bin/env python3
"""Convert 970 ALS instruction pairs from Alpaca format to chat template JSONL.

Reads Alpaca-format entries from data/instruction/als_instructions.json,
applies the model's native chat template using tokenizer.apply_chat_template(),
performs a stratified 90/10 train/val split, and writes JSONL output with
token length statistics and traceability metadata.

Output:
    data/instruction/qlora/train.jsonl  (873 examples)
    data/instruction/qlora/val.jsonl    (97 examples)
    data/instruction/qlora/meta.json    (traceability metadata)

Usage::

    python qlora/format_dataset.py
"""

import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
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
SOURCE_PATH = PROJECT_ROOT / "data" / "instruction" / "als_instructions.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "qlora.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "instruction" / "qlora"

EXPECTED_CATEGORIES = {
    "clinical_trials", "diagnosis", "epidemiology", "genetics",
    "pathophysiology", "patient_care", "symptoms", "treatment",
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  QLoRA Dataset Formatter")
    print("=" * 60)

    # --- Step 1: Load config ---
    if not CONFIG_PATH.exists():
        print(f"\nFATAL: Config file not found: {CONFIG_PATH}")
        print("  Fix: Ensure configs/qlora.json exists with model_id and system_prompt.")
        sys.exit(1)

    config = json.loads(CONFIG_PATH.read_text())
    model_id = config.get("model", {}).get("model_id")
    system_prompt = config.get("system_prompt")
    max_seq_length = config.get("training", {}).get("max_seq_length", 512)

    if not model_id:
        print("\nFATAL: Missing model.model_id in configs/qlora.json")
        sys.exit(1)
    if not system_prompt:
        print("\nFATAL: Missing system_prompt in configs/qlora.json")
        print("  Fix: Add a 'system_prompt' field to configs/qlora.json.")
        sys.exit(1)

    print(f"\n  Model ID:       {model_id}")
    print(f"  System prompt:  {system_prompt[:60]}...")
    print(f"  Max seq length: {max_seq_length}")

    # --- Step 2: Check for existing data with different model_id ---
    meta_path = OUTPUT_DIR / "meta.json"
    if meta_path.exists():
        existing_meta = json.loads(meta_path.read_text())
        existing_model_id = existing_meta.get("model_id", "")
        if existing_model_id != model_id:
            print(f"\nWARNING: Existing data was generated with model_id '{existing_model_id}'")
            print(f"  Current qlora.json model_id is '{model_id}'")
            print("  The chat template format differs between models.")
            print("  Delete data/instruction/qlora/ and re-run this script.")
            sys.exit(1)

    # --- Step 3: Load tokenizer ---
    print("\n  Loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer loaded: {model_id}")

    # --- Step 4: Load Alpaca data ---
    if not SOURCE_PATH.exists():
        print(f"\nFATAL: Source file not found: {SOURCE_PATH}")
        print("  Fix: Run the instruction dataset creation pipeline first.")
        sys.exit(1)

    data = json.loads(SOURCE_PATH.read_text())
    print(f"  Source entries:  {len(data)}")

    if len(data) != 970:
        print(f"\nWARNING: Expected 970 entries, got {len(data)}")

    # --- Step 5: Convert each entry ---
    print("\n  Converting to chat template format...")
    formatted = []
    for i, entry in enumerate(data):
        user_content = entry["instruction"]
        if entry.get("input", "").strip():
            user_content = f"{user_content}\n\nContext: {entry['input']}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": entry["output"]},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=True
        )
        token_count = len(tokens["input_ids"])

        formatted.append({
            "text": text,
            "token_count": token_count,
            "category": entry.get("metadata", {}).get("category", "unknown"),
        })

    print(f"  Converted {len(formatted)} entries")

    # --- Step 6: Stratified 90/10 split ---
    print("\n  Splitting train/val (90/10, stratified by category)...")
    from sklearn.model_selection import train_test_split

    categories = [f["category"] for f in formatted]
    train_data, val_data = train_test_split(
        formatted, test_size=0.1, random_state=42, stratify=categories,
    )

    print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Total: {len(train_data) + len(val_data)}")

    # --- Step 7: Validate structure on 3+ random samples ---
    print("\n  Validating chat template structure (3 random samples)...")
    random.seed(42)
    samples = random.sample(formatted, min(5, len(formatted)))
    validation_passed = True

    for i, sample in enumerate(samples):
        # Re-tokenize and decode to verify round-trip structure
        token_ids = tokenizer.encode(sample["text"])
        decoded = tokenizer.decode(token_ids)

        has_system = "system" in decoded
        has_user = "user" in decoded
        has_assistant = "assistant" in decoded

        status = "PASS" if (has_system and has_user and has_assistant) else "FAIL"
        if status == "FAIL":
            validation_passed = False

        print(f"    Sample {i + 1}: system={has_system} user={has_user} "
              f"assistant={has_assistant} [{status}]")

    if not validation_passed:
        print("\nFATAL: Structure validation failed. Check chat template output.")
        sys.exit(1)
    print("  Structure validation: PASSED")

    # --- Step 8: Write JSONL output ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    for path, split_data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for record in split_data:
                f.write(json.dumps({"text": record["text"]}, ensure_ascii=False) + "\n")

    print(f"\n  Written: {train_path}")
    print(f"  Written: {val_path}")

    # --- Step 9: Compute token length statistics ---
    all_token_counts = [f["token_count"] for f in formatted]
    all_token_counts_sorted = sorted(all_token_counts)
    n = len(all_token_counts_sorted)

    def percentile(data_sorted, p):
        """Compute the p-th percentile from sorted data."""
        k = (p / 100) * (len(data_sorted) - 1)
        f_val = int(k)
        c_val = f_val + 1
        if c_val >= len(data_sorted):
            return data_sorted[-1]
        return data_sorted[f_val] + (k - f_val) * (data_sorted[c_val] - data_sorted[f_val])

    token_stats = {
        "min": min(all_token_counts),
        "max": max(all_token_counts),
        "mean": round(statistics.mean(all_token_counts), 1),
        "median": round(statistics.median(all_token_counts), 1),
        "p90": round(percentile(all_token_counts_sorted, 90), 1),
        "p95": round(percentile(all_token_counts_sorted, 95), 1),
        "p99": round(percentile(all_token_counts_sorted, 99), 1),
    }

    count_over = sum(1 for tc in all_token_counts if tc > max_seq_length)

    print(f"\n  Token Length Statistics")
    print(f"  {'Statistic':<15} {'Value':>8}")
    print(f"  {'-' * 15} {'-' * 8}")
    print(f"  {'Min':<15} {token_stats['min']:>8}")
    print(f"  {'Max':<15} {token_stats['max']:>8}")
    print(f"  {'Mean':<15} {token_stats['mean']:>8.1f}")
    print(f"  {'Median':<15} {token_stats['median']:>8.1f}")
    print(f"  {'P90':<15} {token_stats['p90']:>8.1f}")
    print(f"  {'P95':<15} {token_stats['p95']:>8.1f}")
    print(f"  {'P99':<15} {token_stats['p99']:>8.1f}")
    print(f"  {'Over {0}'.format(max_seq_length):<15} {count_over:>8}")

    if count_over > 0:
        print(f"\n  WARNING: {count_over} examples exceed max_seq_length ({max_seq_length}).")
        print(f"  Consider adjusting max_seq_length in configs/qlora.json before training.")

    # --- Step 10: Category distribution ---
    train_cats = {}
    val_cats = {}
    for r in train_data:
        cat = r["category"]
        train_cats[cat] = train_cats.get(cat, 0) + 1
    for r in val_data:
        cat = r["category"]
        val_cats[cat] = val_cats.get(cat, 0) + 1

    print(f"\n  Category Distribution")
    print(f"  {'Category':<20} {'Train':>6} {'Val':>6} {'Total':>6}")
    print(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 6}")
    for cat in sorted(EXPECTED_CATEGORIES):
        tc = train_cats.get(cat, 0)
        vc = val_cats.get(cat, 0)
        print(f"  {cat:<20} {tc:>6} {vc:>6} {tc + vc:>6}")
    print(f"  {'TOTAL':<20} {sum(train_cats.values()):>6} "
          f"{sum(val_cats.values()):>6} {sum(train_cats.values()) + sum(val_cats.values()):>6}")

    # --- Step 11: Write meta.json ---
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "model_id": model_id,
        "timestamp": timestamp,
        "total_examples": len(formatted),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "token_stats": token_stats,
        "examples_over_max_seq_length": count_over,
        "max_seq_length": max_seq_length,
        "system_prompt": system_prompt,
        "category_distribution": {
            "train": dict(sorted(train_cats.items())),
            "val": dict(sorted(val_cats.items())),
        },
    }

    meta_out = OUTPUT_DIR / "meta.json"
    meta_out.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"\n  Written: {meta_out}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"  Dataset Formatting Complete")
    print(f"{'=' * 60}")
    print(f"  Total converted: {len(formatted)}")
    print(f"  Train examples:  {len(train_data)}")
    print(f"  Val examples:    {len(val_data)}")
    print(f"  Token range:     {token_stats['min']}-{token_stats['max']} "
          f"(mean {token_stats['mean']:.1f}, median {token_stats['median']:.1f})")
    print(f"  Over max_seq_length: {count_over}")
    print(f"  Validation:      PASSED")
    print(f"  Output dir:      {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
