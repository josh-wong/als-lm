#!/usr/bin/env python3
"""Validate GPT-2-tokenized corpus and produce a detailed comparison report.

Confirms the re-tokenized corpus is correct by checking vocabulary identity,
token counts, compression ratios, decode coherence, and medical term handling.
Produces both stdout output and a Markdown report.

Usage:
    python scripts/validate_gpt2_data.py
    python scripts/validate_gpt2_data.py --data-dir data/tokenized_gpt2
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast


def check_vocabulary_identity(meta_path, tok):
    """Verify meta.pkl vocabulary matches the live GPT-2 tokenizer.

    Returns:
        Tuple of (passed: bool, details: dict).
    """
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    meta_vocab_size = meta["vocab_size"]
    live_vocab_size = tok.vocab_size
    live_vocab = tok.get_vocab()

    # Check vocab_size
    size_match = meta_vocab_size == live_vocab_size == 50257

    # Spot-check a sample of stoi mappings against live tokenizer
    meta_stoi = meta["stoi"]
    meta_itos = meta["itos"]
    mismatches = []
    for token_str, token_id in list(meta_stoi.items())[:1000]:
        live_id = live_vocab.get(token_str)
        if live_id is not None and live_id != token_id:
            mismatches.append((token_str, token_id, live_id))

    # Check itos/stoi round-trip consistency
    roundtrip_errors = 0
    for token_id, token_str in list(meta_itos.items())[:1000]:
        if meta_stoi.get(token_str) != token_id:
            roundtrip_errors += 1

    passed = size_match and len(mismatches) == 0 and roundtrip_errors == 0

    return passed, {
        "meta_vocab_size": meta_vocab_size,
        "live_vocab_size": live_vocab_size,
        "size_match": size_match,
        "stoi_mismatches": len(mismatches),
        "roundtrip_errors": roundtrip_errors,
        "stoi_entries": len(meta_stoi),
        "itos_entries": len(meta_itos),
    }


def count_tokens(data_dir, als_data_dir):
    """Load token counts from both GPT-2 and ALS tokenizer bins.

    Returns:
        Dict with token counts and file sizes for each split.
    """
    results = {}

    for split in ["train", "val"]:
        gpt2_path = os.path.join(data_dir, f"{split}.bin")
        als_path = os.path.join(als_data_dir, f"{split}.bin")

        gpt2_tokens = np.memmap(gpt2_path, dtype=np.uint16, mode="r")
        gpt2_count = len(gpt2_tokens)
        gpt2_bytes = os.path.getsize(gpt2_path)

        als_count = 0
        als_bytes = 0
        if os.path.isfile(als_path):
            als_tokens = np.memmap(als_path, dtype=np.uint16, mode="r")
            als_count = len(als_tokens)
            als_bytes = os.path.getsize(als_path)

        results[split] = {
            "gpt2_tokens": gpt2_count,
            "gpt2_bytes": gpt2_bytes,
            "als_tokens": als_count,
            "als_bytes": als_bytes,
            "ratio": gpt2_count / als_count if als_count > 0 else 0,
        }

    return results


def compute_compression_ratios(token_counts, train_txt, val_txt):
    """Calculate chars-per-token for both tokenizers.

    Returns:
        Dict with compression ratios per split and overall.
    """
    results = {}

    for split, txt_path in [("train", train_txt), ("val", val_txt)]:
        txt_size = os.path.getsize(txt_path)
        tc = token_counts[split]

        gpt2_cpt = txt_size / tc["gpt2_tokens"] if tc["gpt2_tokens"] > 0 else 0
        als_cpt = txt_size / tc["als_tokens"] if tc["als_tokens"] > 0 else 0

        results[split] = {
            "text_bytes": txt_size,
            "gpt2_chars_per_token": gpt2_cpt,
            "als_chars_per_token": als_cpt,
        }

    return results


def decode_sample(data_dir, tok, offset=1000, window=100):
    """Decode a window of tokens from train.bin and return the text.

    Returns:
        Tuple of (decoded_text: str, token_ids: list).
    """
    train_path = os.path.join(data_dir, "train.bin")
    tokens = np.memmap(train_path, dtype=np.uint16, mode="r")

    if offset + window > len(tokens):
        offset = max(0, len(tokens) - window)

    sample_ids = tokens[offset : offset + window].tolist()
    decoded = tok.decode(sample_ids)

    return decoded, sample_ids


def compare_medical_terms(terms_path, gpt2_tok, als_tok_path, threshold=3):
    """Compare medical term tokenization between GPT-2 and ALS tokenizer.

    Returns:
        Dict with per-term results and summary statistics.
    """
    with open(terms_path, "r", encoding="utf-8") as f:
        terms = json.load(f)

    als_tok = Tokenizer.from_file(als_tok_path)

    results = []
    gpt2_wins = 0
    als_wins = 0
    ties = 0
    gpt2_flagged = 0

    for entry in terms:
        term = entry["term"]
        category = entry.get("category", "unknown")

        # GPT-2 encoding
        gpt2_ids = gpt2_tok.encode(term, add_special_tokens=False)
        gpt2_tokens = gpt2_tok.convert_ids_to_tokens(gpt2_ids)
        gpt2_count = len(gpt2_ids)

        # ALS tokenizer encoding
        als_encoded = als_tok.encode(term)
        als_tokens_list = als_encoded.tokens
        als_count = len(als_tokens_list)

        is_flagged = gpt2_count >= threshold

        if gpt2_count < als_count:
            gpt2_wins += 1
            winner = "GPT-2"
        elif als_count < gpt2_count:
            als_wins += 1
            winner = "ALS"
        else:
            ties += 1
            winner = "tie"

        if is_flagged:
            gpt2_flagged += 1

        results.append({
            "term": term,
            "category": category,
            "gpt2_subtokens": gpt2_count,
            "gpt2_tokens": gpt2_tokens,
            "als_subtokens": als_count,
            "als_tokens": als_tokens_list,
            "flagged": is_flagged,
            "winner": winner,
        })

    return {
        "total_terms": len(results),
        "gpt2_wins": gpt2_wins,
        "als_wins": als_wins,
        "ties": ties,
        "gpt2_flagged": gpt2_flagged,
        "threshold": threshold,
        "results": results,
    }


def generate_report(vocab_info, token_counts, compression, decoded_text,
                    sample_ids, medical_comparison):
    """Generate the full Markdown validation report.

    Returns:
        Report as a string.
    """
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# GPT-2 tokenization validation report")
    lines.append("")
    lines.append(
        "This report validates the GPT-2-tokenized ALS corpus by checking "
        "vocabulary identity, token counts, compression ratios, decode "
        "coherence, and medical term handling."
    )
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append("")

    # Section 1: Vocabulary identity
    lines.append("## Vocabulary identity")
    lines.append("")

    vi = vocab_info
    status = "PASSED" if vi["size_match"] else "FAILED"
    lines.append(
        f"The vocabulary identity check confirms that meta.pkl matches the "
        f"live GPT-2 tokenizer."
    )
    lines.append("")
    lines.append(f"| Property               | Value           |")
    lines.append(f"|------------------------|-----------------|")
    lines.append(f"| meta.pkl vocab_size     | {vi['meta_vocab_size']:>15,} |")
    lines.append(f"| Live tokenizer vocab    | {vi['live_vocab_size']:>15,} |")
    lines.append(f"| Size match (== 50,257)  | {status:>15} |")
    lines.append(f"| stoi entries            | {vi['stoi_entries']:>15,} |")
    lines.append(f"| itos entries            | {vi['itos_entries']:>15,} |")
    lines.append(f"| stoi mismatches         | {vi['stoi_mismatches']:>15} |")
    lines.append(f"| Round-trip errors        | {vi['roundtrip_errors']:>15} |")
    lines.append("")

    # Section 2: Token counts
    lines.append("## Token counts per split")
    lines.append("")
    lines.append(
        "Side-by-side token counts from GPT-2 and ALS tokenizer encodings. "
        "The ratio column shows how many more tokens GPT-2 produces relative "
        "to the domain-specific ALS tokenizer."
    )
    lines.append("")
    lines.append(
        f"| Split   | {'GPT-2 tokens':>15} | {'ALS tokens':>15} | "
        f"{'GPT-2 bin size':>15} | {'ALS bin size':>15} | {'Ratio':>6} |"
    )
    lines.append(
        f"|---------|{'-' * 17}|{'-' * 17}|"
        f"{'-' * 17}|{'-' * 17}|{'-' * 8}|"
    )
    for split in ["train", "val"]:
        tc = token_counts[split]
        lines.append(
            f"| {split:<7} | {tc['gpt2_tokens']:>15,} | {tc['als_tokens']:>15,} | "
            f"{tc['gpt2_bytes']:>12,} B | {tc['als_bytes']:>12,} B | "
            f"{tc['ratio']:>5.2f}x |"
        )
    lines.append("")

    # Section 3: Compression ratios
    lines.append("## Compression ratios")
    lines.append("")
    lines.append(
        "Characters-per-token measures encoding efficiency. Higher values "
        "mean fewer tokens per unit of text. The ALS tokenizer was trained "
        "specifically on medical text and should achieve better compression "
        "on domain content."
    )
    lines.append("")
    lines.append(
        f"| Split   | {'Text size':>15} | {'GPT-2 chars/tok':>16} | "
        f"{'ALS chars/tok':>14} |"
    )
    lines.append(
        f"|---------|{'-' * 17}|{'-' * 18}|{'-' * 16}|"
    )
    for split in ["train", "val"]:
        cr = compression[split]
        lines.append(
            f"| {split:<7} | {cr['text_bytes']:>12,} B | "
            f"{cr['gpt2_chars_per_token']:>16.2f} | "
            f"{cr['als_chars_per_token']:>14.2f} |"
        )
    lines.append("")

    # Section 4: Decode sample
    lines.append("## 100-token decode sample")
    lines.append("")
    lines.append(
        "A 100-token window decoded from train.bin at offset 1000. The text "
        "should be coherent ALS medical content, not garbled output."
    )
    lines.append("")
    lines.append("```")
    lines.append(decoded_text)
    lines.append("```")
    lines.append("")
    lines.append(f"**Token IDs (first 20):** {sample_ids[:20]}")
    lines.append("")

    # Section 5: Medical term comparison
    mc = medical_comparison
    lines.append("## Medical term comparison")
    lines.append("")
    lines.append(
        f"Side-by-side subtoken counts for {mc['total_terms']} medical terms "
        f"encoded with both GPT-2 and ALS tokenizers. Terms where GPT-2 "
        f"fragments into {mc['threshold']}+ subtokens are flagged."
    )
    lines.append("")

    # Summary stats
    lines.append("### Summary")
    lines.append("")
    lines.append(
        f"Comparison of tokenization efficiency across {mc['total_terms']} "
        f"medical terms."
    )
    lines.append("")
    lines.append(f"| Metric                        | Value |")
    lines.append(f"|-------------------------------|-------|")
    lines.append(f"| GPT-2 wins (fewer subtokens)  | {mc['gpt2_wins']:>5} |")
    lines.append(f"| ALS wins (fewer subtokens)    | {mc['als_wins']:>5} |")
    lines.append(f"| Ties                          | {mc['ties']:>5} |")
    lines.append(
        f"| GPT-2 flagged ({mc['threshold']}+ subtokens) "
        f"| {mc['gpt2_flagged']:>5} |"
    )
    lines.append("")

    # Flagged terms detail table
    flagged_terms = [r for r in mc["results"] if r["flagged"]]
    if flagged_terms:
        lines.append("### Flagged terms")
        lines.append("")
        lines.append(
            f"Terms where GPT-2 produces {mc['threshold']}+ subtokens, "
            f"indicating significant fragmentation."
        )
        lines.append("")
        lines.append(
            f"| {'Term':<35} | {'Category':<15} | {'GPT-2':>5} | "
            f"{'ALS':>5} | {'Winner':<6} | GPT-2 subtokens |"
        )
        lines.append(
            f"|{'-' * 37}|{'-' * 17}|{'-' * 7}|"
            f"{'-' * 7}|{'-' * 8}|{'-' * 17}|"
        )
        for r in sorted(flagged_terms, key=lambda x: -x["gpt2_subtokens"]):
            gpt2_parts = " + ".join(f"`{t}`" for t in r["gpt2_tokens"])
            lines.append(
                f"| {r['term']:<35} | {r['category']:<15} | "
                f"{r['gpt2_subtokens']:>5} | {r['als_subtokens']:>5} | "
                f"{r['winner']:<6} | {gpt2_parts} |"
            )
        lines.append("")

    # Full comparison table (compact)
    lines.append("### Full term comparison")
    lines.append("")
    lines.append(
        "All terms sorted by GPT-2 subtoken count (descending). The winner "
        "column shows which tokenizer uses fewer subtokens for each term."
    )
    lines.append("")
    lines.append(
        f"| {'Term':<35} | {'Cat.':<12} | {'GPT-2':>5} | "
        f"{'ALS':>5} | {'Winner':<6} |"
    )
    lines.append(
        f"|{'-' * 37}|{'-' * 14}|{'-' * 7}|{'-' * 7}|{'-' * 8}|"
    )
    for r in sorted(mc["results"], key=lambda x: -x["gpt2_subtokens"]):
        lines.append(
            f"| {r['term']:<35} | {r['category']:<12} | "
            f"{r['gpt2_subtokens']:>5} | {r['als_subtokens']:>5} | "
            f"{r['winner']:<6} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("*Report generated by scripts/validate_gpt2_data.py*")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Validate GPT-2-tokenized corpus with detailed comparison report"
    )
    parser.add_argument(
        "--data-dir",
        default="data/tokenized_gpt2",
        help="GPT-2 tokenized data directory (default: data/tokenized_gpt2)",
    )
    parser.add_argument(
        "--als-data-dir",
        default="data/tokenized",
        help="ALS tokenized data directory (default: data/tokenized)",
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
        "--terms",
        default="reports/medical_terms.json",
        help="Path to medical terms JSON (default: reports/medical_terms.json)",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/als_tokenizer.json",
        help="Path to ALS tokenizer (default: tokenizer/als_tokenizer.json)",
    )
    parser.add_argument(
        "--report",
        default="reports/gpt2_tokenization_report.md",
        help="Output report path (default: reports/gpt2_tokenization_report.md)",
    )
    args = parser.parse_args()

    print("=== GPT-2 Tokenization Validation ===\n")

    # Load GPT-2 tokenizer
    print("Loading GPT-2 tokenizer...")
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")

    # 1. Vocabulary identity check
    print("\n--- Vocabulary Identity Check ---")
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    vocab_passed, vocab_info = check_vocabulary_identity(meta_path, gpt2_tok)
    status = "PASSED" if vocab_passed else "FAILED"
    print(f"  vocab_size: {vocab_info['meta_vocab_size']} (expected 50257) - {status}")
    print(f"  stoi mismatches: {vocab_info['stoi_mismatches']}")
    print(f"  Round-trip errors: {vocab_info['roundtrip_errors']}")

    if not vocab_passed:
        print("CRITICAL: Vocabulary identity check failed!", file=sys.stderr)
        sys.exit(1)

    # 2. Token counts
    print("\n--- Token Counts ---")
    token_counts = count_tokens(args.data_dir, args.als_data_dir)
    for split in ["train", "val"]:
        tc = token_counts[split]
        print(
            f"  {split}: GPT-2={tc['gpt2_tokens']:,} | "
            f"ALS={tc['als_tokens']:,} | ratio={tc['ratio']:.2f}x"
        )

    # 3. Compression ratios
    print("\n--- Compression Ratios ---")
    compression = compute_compression_ratios(
        token_counts, args.train_txt, args.val_txt
    )
    for split in ["train", "val"]:
        cr = compression[split]
        print(
            f"  {split}: GPT-2={cr['gpt2_chars_per_token']:.2f} chars/tok | "
            f"ALS={cr['als_chars_per_token']:.2f} chars/tok"
        )

    # 4. Decode sample
    print("\n--- 100-Token Decode Sample ---")
    decoded_text, sample_ids = decode_sample(args.data_dir, gpt2_tok)
    print(f"  {decoded_text[:200]}...")
    assert len(decoded_text) > 10, "Decode produced empty or very short text"

    # 5. Medical term comparison
    print("\n--- Medical Term Comparison ---")
    medical_comparison = compare_medical_terms(
        args.terms, gpt2_tok, args.tokenizer
    )
    mc = medical_comparison
    print(
        f"  GPT-2 wins: {mc['gpt2_wins']} | ALS wins: {mc['als_wins']} | "
        f"Ties: {mc['ties']}"
    )
    print(f"  GPT-2 flagged ({mc['threshold']}+ subtokens): {mc['gpt2_flagged']}")

    # Generate report
    print("\n--- Generating Report ---")
    report = generate_report(
        vocab_info, token_counts, compression,
        decoded_text, sample_ids, medical_comparison,
    )

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")
    print(f"  Report size: {len(report):,} chars")

    # Print full report to stdout
    print("\n" + "=" * 70)
    print(report)

    print("=== Validation Complete ===")


if __name__ == "__main__":
    main()
