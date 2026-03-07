#!/usr/bin/env python3
"""Prepare GPT-2-tokenized corpus data for fine-tuning.

Re-encodes train.txt and val.txt using GPT-2's native tokenizer into binary
format compatible with nanoGPT (uint16 numpy arrays + meta.pkl). This is a
correctness gate for v0.8.0 — wrong tokenizer IDs silently corrupt all
downstream fine-tuning.

Usage:
    python scripts/prepare_gpt2_data.py
    python scripts/prepare_gpt2_data.py --output-dir data/tokenized_gpt2
"""

import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast


def encode_file(tok, filepath, description=""):
    """Encode a text file using GPT-2 tokenizer, splitting on document boundaries.

    Splits text on <|endoftext|> boundaries before encoding, then manually
    inserts the EOT token ID between documents. This avoids potential
    fragmentation of the special token into subtokens.

    Args:
        tok: GPT2TokenizerFast instance.
        filepath: Path to the text file.
        description: Label for progress bar.

    Returns:
        numpy array of token IDs (uint16).
    """
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    documents = text.split("<|endoftext|>")
    eot_id = tok.eos_token_id  # 50256

    all_ids = []
    non_empty = 0

    for i, doc in enumerate(tqdm(documents, desc=description, unit="doc")):
        if not doc.strip():
            continue
        ids = tok.encode(doc, add_special_tokens=False)
        all_ids.extend(ids)
        non_empty += 1
        # Re-add the EOT separator between documents
        if i < len(documents) - 1:
            all_ids.append(eot_id)

    print(f"  Documents processed: {non_empty:,} (of {len(documents):,} splits)")
    return np.array(all_ids, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(
        description="Re-encode ALS corpus with GPT-2 tokenizer for fine-tuning"
    )
    parser.add_argument(
        "--train-txt",
        default="data/processed/train.txt",
        help="Path to training text (default: data/processed/train.txt)",
    )
    parser.add_argument(
        "--val-txt",
        default="data/processed/val.txt",
        help="Path to validation text (default: data/processed/val.txt)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tokenized_gpt2",
        help="Output directory for binary files (default: data/tokenized_gpt2)",
    )
    args = parser.parse_args()

    # Verify source files exist
    for path in [args.train_txt, args.val_txt]:
        if not os.path.isfile(path):
            print(f"Error: source file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # Load GPT-2 tokenizer
    print("=== GPT-2 Corpus Re-tokenization ===\n")
    print("Loading GPT-2 tokenizer...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tok.vocab_size
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  EOT token ID: {tok.eos_token_id}")

    # uint16 safety check
    if vocab_size > 65535:
        print(
            f"Error: vocab_size ({vocab_size}) exceeds uint16 max (65535). "
            f"Cannot encode as uint16.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  uint16 safety check passed ({vocab_size} <= 65535)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Encode train.txt
    train_size = os.path.getsize(args.train_txt)
    print(f"\nEncoding training data: {args.train_txt}")
    print(f"  File size: {train_size:,} bytes ({train_size / 1024 / 1024:.1f} MB)")

    train_ids = encode_file(tok, args.train_txt, description="Encoding train.txt")

    train_bin_path = os.path.join(args.output_dir, "train.bin")
    train_ids.tofile(train_bin_path)
    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Train bin size: {os.path.getsize(train_bin_path):,} bytes")

    # Encode val.txt
    val_size = os.path.getsize(args.val_txt)
    print(f"\nEncoding validation data: {args.val_txt}")
    print(f"  File size: {val_size:,} bytes ({val_size / 1024 / 1024:.1f} MB)")

    val_ids = encode_file(tok, args.val_txt, description="Encoding val.txt")

    val_bin_path = os.path.join(args.output_dir, "val.bin")
    val_ids.tofile(val_bin_path)
    print(f"  Val tokens: {len(val_ids):,}")
    print(f"  Val bin size: {os.path.getsize(val_bin_path):,} bytes")

    # Build meta.pkl with itos and stoi mappings
    print("\nBuilding meta.pkl vocabulary mappings...")
    vocab = tok.get_vocab()  # dict: str -> int
    stoi = dict(vocab)
    itos = {v: k for k, v in vocab.items()}

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }

    meta_path = os.path.join(args.output_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(
        f"  meta.pkl: vocab_size={vocab_size}, "
        f"itos={len(itos)} entries, stoi={len(stoi)} entries"
    )
    print(f"  meta.pkl size: {os.path.getsize(meta_path):,} bytes")

    # Summary
    print("\n=== GPT-2 Corpus Re-tokenization Complete ===")
    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens:   {len(val_ids):,}")
    print(f"  Vocab size:   {vocab_size:,}")
    print(f"  Output dir:   {args.output_dir}")


if __name__ == "__main__":
    main()
