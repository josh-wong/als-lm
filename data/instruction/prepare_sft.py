#!/usr/bin/env python3
"""Tokenize Alpaca-format instruction JSON into paired binary files for SFT.

Reads an Alpaca-format JSON file (with instruction, input, output fields),
tokenizes each entry using the ALS BPE tokenizer, and writes paired binary
files (input token IDs + labels with -100 masking) for supervised fine-tuning.

The output format uses padded fixed-length records: each example is padded to
exactly block_size tokens, so the file length is num_examples * block_size.
Labels use -100 for instruction prefix tokens and padding positions, matching
PyTorch's cross_entropy ignore_index convention.

Usage::

    python data/instruction/prepare_sft.py \\
        --input data/instruction/synthetic_sft.json \\
        --output-dir data/instruction/tokenized/ \\
        --block-size 1024

Output files:

    train.bin         - uint16 input token IDs (training split)
    labels_train.bin  - int32 labels with -100 masking (training split)
    val.bin           - uint16 input token IDs (validation split)
    labels_val.bin    - int32 labels with -100 masking (validation split)
    meta.pkl          - {vocab_size: 50257}
"""

import argparse
import json
import os
import pickle
import random
import sys
from typing import Optional

import numpy as np

# Ensure project root is on sys.path for tokenizer imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Alpaca template constants
# ---------------------------------------------------------------------------

ALPACA_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

ALPACA_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

# Prefix templates (everything up to and including "### Response:\n")
_PREFIX_WITH_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

_PREFIX_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


# ---------------------------------------------------------------------------
# Formatting and tokenization
# ---------------------------------------------------------------------------

def format_alpaca(entry: dict) -> str:
    """Format an Alpaca JSON entry into the full prompt string.

    Uses the with-input template if the entry has a non-empty ``input`` field,
    otherwise uses the no-input template.

    Args:
        entry: Dictionary with ``instruction``, ``input``, and ``output`` keys.

    Returns:
        The formatted Alpaca template string.
    """
    if entry.get("input", "").strip():
        return ALPACA_TEMPLATE_WITH_INPUT.format(**entry)
    else:
        return ALPACA_TEMPLATE_NO_INPUT.format(**entry)


def _build_prefix(entry: dict) -> str:
    """Build the instruction prefix (everything before the response content).

    The prefix ends at "### Response:\\n" — the response text follows
    immediately after.
    """
    if entry.get("input", "").strip():
        return _PREFIX_WITH_INPUT.format(**entry)
    else:
        return _PREFIX_NO_INPUT.format(**entry)


def tokenize_and_mask(
    entry: dict,
    tokenizer: Tokenizer,
    block_size: int,
) -> Optional[tuple[list[int], list[int]]]:
    """Tokenize an Alpaca entry and create labels with -100 masking.

    The instruction prefix (everything up to and including "### Response:\\n")
    is masked with -100 in the labels. Only response tokens receive valid
    label IDs for loss computation. Both input_ids and labels are padded to
    exactly ``block_size``.

    Args:
        entry: Dictionary with ``instruction``, ``input``, ``output`` keys.
        tokenizer: Hugging Face tokenizers Tokenizer instance.
        block_size: Target sequence length for padded records.

    Returns:
        Tuple of (input_ids, labels) each of length block_size, or None
        if the tokenized example exceeds block_size.
    """
    full_text = format_alpaca(entry)
    prefix_text = _build_prefix(entry)

    full_ids = tokenizer.encode(full_text).ids
    prefix_ids = tokenizer.encode(prefix_text).ids
    prefix_len = len(prefix_ids)

    if len(full_ids) > block_size:
        return None  # Example too long

    # Verify tokenizer boundary consistency
    if full_ids[:prefix_len] != prefix_ids:
        # BPE merge boundary divergence — fall back to searching for the
        # response marker in the full tokenization by using prefix length
        # as an approximation and adjusting
        response_text = entry["output"]
        response_ids = tokenizer.encode(response_text).ids

        # Search backwards from the expected boundary to find where the
        # response tokens actually start in the full tokenization
        found = False
        for offset in range(max(0, prefix_len - 5), min(len(full_ids), prefix_len + 5)):
            remaining = full_ids[offset:]
            if len(remaining) >= len(response_ids) and remaining[:len(response_ids)] == response_ids:
                prefix_len = offset
                found = True
                break

        if not found:
            # Cannot determine correct loss mask boundary — skip this entry
            # rather than silently training on instruction tokens
            print(
                f"  SKIPPED: Tokenizer boundary divergence for entry: "
                f"{entry['instruction'][:50]}... (could not locate response tokens)"
            )
            return None

    # Create labels: -100 for prefix, actual token IDs for response
    labels = [-100] * prefix_len + full_ids[prefix_len:]

    # Pad to block_size
    pad_len = block_size - len(full_ids)
    input_ids = full_ids + [0] * pad_len       # Pad with 0
    labels = labels + [-100] * pad_len          # Padding positions are masked

    return input_ids, labels


# ---------------------------------------------------------------------------
# Binary file writing
# ---------------------------------------------------------------------------

def _write_split(
    examples: list[dict],
    tokenizer: Tokenizer,
    block_size: int,
    output_dir: str,
    prefix: str,
) -> dict:
    """Tokenize and write a split (train or val) to binary files.

    Args:
        examples: List of Alpaca-format dictionaries.
        tokenizer: Tokenizer instance.
        block_size: Target sequence length.
        output_dir: Directory for output files.
        prefix: File prefix ("train" or "val").

    Returns:
        Dictionary with split statistics.
    """
    all_input_ids = []
    all_labels = []
    skipped = 0
    prompt_lengths = []
    response_lengths = []

    for entry in examples:
        result = tokenize_and_mask(entry, tokenizer, block_size)
        if result is None:
            skipped += 1
            continue

        input_ids, labels = result
        all_input_ids.extend(input_ids)
        all_labels.extend(labels)

        # Compute token stats
        prefix_text = _build_prefix(entry)
        prefix_ids = tokenizer.encode(prefix_text).ids
        full_ids = tokenizer.encode(format_alpaca(entry)).ids
        prompt_lengths.append(len(prefix_ids))
        response_lengths.append(len(full_ids) - len(prefix_ids))

    # Write binary files
    input_arr = np.array(all_input_ids, dtype=np.uint16)
    labels_arr = np.array(all_labels, dtype=np.int32)

    input_arr.tofile(os.path.join(output_dir, f"{prefix}.bin"))
    labels_arr.tofile(os.path.join(output_dir, f"labels_{prefix}.bin"))

    num_examples = len(examples) - skipped
    stats = {
        "num_examples": num_examples,
        "skipped": skipped,
        "total_tokens": len(all_input_ids),
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
    }
    return stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the SFT data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Tokenize Alpaca JSON into paired binary files for SFT"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Alpaca-format JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/instruction/tokenized/",
        help="Output directory for binary files (default: data/instruction/tokenized/)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Target sequence length for padded records (default: 1024)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    input_path = os.path.join(_project_root, args.input) if not os.path.isabs(args.input) else args.input
    output_dir = os.path.join(_project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    tokenizer_path = os.path.join(_project_root, "tokenizer", "als_tokenizer.json")

    print(f"SFT Data Preparation")
    print(f"  Input:      {input_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Block size: {args.block_size}")
    print(f"  Val split:  {args.val_split}")
    print()

    # Load input data
    with open(input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {os.path.basename(input_path)}")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    # Shuffle and split
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * (1 - args.val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Split: {len(train_data)} train, {len(val_data)} val")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process splits
    print("Processing training split...")
    train_stats = _write_split(train_data, tokenizer, args.block_size, output_dir, "train")

    print("Processing validation split...")
    val_stats = _write_split(val_data, tokenizer, args.block_size, output_dir, "val")

    # Write meta.pkl
    meta_path = os.path.join(output_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({"vocab_size": vocab_size}, f)
    print(f"Wrote meta.pkl (vocab_size={vocab_size})")
    print()

    # Print summary stats
    print("=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    total_examples = train_stats["num_examples"] + val_stats["num_examples"]
    total_skipped = train_stats["skipped"] + val_stats["skipped"]
    all_prompt = train_stats["prompt_lengths"] + val_stats["prompt_lengths"]
    all_response = train_stats["response_lengths"] + val_stats["response_lengths"]
    total_tokens = train_stats["total_tokens"] + val_stats["total_tokens"]

    print(f"  Total examples:      {total_examples}")
    print(f"  Training examples:   {train_stats['num_examples']}")
    print(f"  Validation examples: {val_stats['num_examples']}")
    print(f"  Skipped (too long):  {total_skipped}")
    print()

    if all_prompt:
        print(f"  Prompt tokens:  avg={sum(all_prompt)/len(all_prompt):.1f}  "
              f"min={min(all_prompt)}  max={max(all_prompt)}")
    if all_response:
        print(f"  Response tokens: avg={sum(all_response)/len(all_response):.1f}  "
              f"min={min(all_response)}  max={max(all_response)}")
    print(f"  Total tokens:    {total_tokens}")
    print()

    # List output files with sizes
    print("Output files:")
    for fname in ["train.bin", "labels_train.bin", "val.bin", "labels_val.bin", "meta.pkl"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {fname}: {size_kb:.1f} KB")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
