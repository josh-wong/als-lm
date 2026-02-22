#!/usr/bin/env python3
"""Prepare tokenized corpus data for nanoGPT training.

Encodes train.txt and val.txt using the winning ALS tokenizer into binary format
compatible with nanoGPT (uint16 numpy arrays + meta.pkl).

Also includes sweep summary generation to select the winning tokenizer from
comparison and validation reports.

Usage:
    # Generate sweep summary and select winner
    python scripts/prepare_data.py --sweep-summary

    # Encode corpus to nanoGPT binary format
    python scripts/prepare_data.py --encode

    # Both (default)
    python scripts/prepare_data.py
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def load_reports(comparison_path, validation_path):
    """Load comparison and validation report JSON files."""
    with open(comparison_path) as f:
        comparison = json.load(f)
    with open(validation_path) as f:
        validation = json.load(f)
    return comparison, validation


def generate_sweep_summary(comparison_path, validation_path, output_path, tokenizer_dir):
    """Analyze all tokenizer candidates and recommend the best one.

    Ranking criteria (weighted):
    1. Medical term flagged count (3+ subtokens) - lower is better (weight: 0.4)
    2. Medical text fertility - lower is better (weight: 0.3)
    3. Compression ratio - higher is better (weight: 0.2)
    4. General-English fertility penalty - lower is better (weight: 0.1)
    """
    print("=== Generating Sweep Summary ===")

    comparison, validation = load_reports(comparison_path, validation_path)

    # Build candidate data from both reports
    candidates = []
    for tok_name, val_data in validation["tokenizers"].items():
        # Find matching comparison data
        comp_data = comparison["comparisons"].get(tok_name, {})

        if not comp_data:
            print(f"  Warning: no comparison data for {tok_name}, skipping")
            continue

        vocab_size = val_data["vocab_size"]
        flagged_count = val_data["flagged_count"]
        flagged_pct = val_data["flagged_pct"]
        total_terms = val_data["total_terms"]

        medical_fertility = comp_data["text_metrics"]["custom"]["fertility"]
        compression_ratio = comp_data["text_metrics"]["custom"]["compression_ratio"]
        general_fertility = comp_data["general_english"]["custom"]["avg_fertility"]
        gpt2_general = comp_data["general_english"]["baseline"]["avg_fertility"]

        # General-English penalty: ratio vs GPT-2
        ge_penalty = general_fertility / gpt2_general if gpt2_general > 0 else float("inf")

        wins = comp_data["term_comparison"]["wins"]
        losses = comp_data["term_comparison"]["losses"]
        ties = comp_data["term_comparison"]["ties"]

        # Find the tokenizer file path
        tok_path = Path(tokenizer_dir) / f"{tok_name}.json"

        candidates.append({
            "name": tok_name,
            "vocab_size": vocab_size,
            "flagged_count": flagged_count,
            "flagged_pct": flagged_pct,
            "total_terms": total_terms,
            "medical_fertility": medical_fertility,
            "compression_ratio": compression_ratio,
            "general_fertility": general_fertility,
            "ge_penalty": ge_penalty,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "tok_path": str(tok_path),
        })

    if not candidates:
        print("Error: no candidates found in reports", file=sys.stderr)
        sys.exit(1)

    # Rank candidates using weighted scoring
    # Normalize each metric to [0, 1] range, then apply weights
    # For metrics where lower is better, we invert (1 - normalized)
    # For metrics where higher is better, we keep as-is

    flagged_vals = [c["flagged_count"] for c in candidates]
    fert_vals = [c["medical_fertility"] for c in candidates]
    comp_vals = [c["compression_ratio"] for c in candidates]
    ge_vals = [c["ge_penalty"] for c in candidates]

    def normalize_lower_better(vals):
        """Normalize so that lower values get higher scores."""
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return [1.0] * len(vals)
        return [1.0 - (v - min_v) / (max_v - min_v) for v in vals]

    def normalize_higher_better(vals):
        """Normalize so that higher values get higher scores."""
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return [1.0] * len(vals)
        return [(v - min_v) / (max_v - min_v) for v in vals]

    flagged_scores = normalize_lower_better(flagged_vals)
    fert_scores = normalize_lower_better(fert_vals)
    comp_scores = normalize_higher_better(comp_vals)
    ge_scores = normalize_lower_better(ge_vals)

    weights = {"flagged": 0.4, "fertility": 0.3, "compression": 0.2, "ge_penalty": 0.1}

    for i, c in enumerate(candidates):
        c["score"] = (
            weights["flagged"] * flagged_scores[i]
            + weights["fertility"] * fert_scores[i]
            + weights["compression"] * comp_scores[i]
            + weights["ge_penalty"] * ge_scores[i]
        )

    # Sort by score (highest first)
    candidates.sort(key=lambda x: x["score"], reverse=True)

    winner = candidates[0]

    # Check if all candidates are identical
    all_same = len(set(c["vocab_size"] for c in candidates)) == 1 and len(candidates) > 1

    # If all candidates are identical, select the 32k as canonical
    if all_same:
        preferred_name = "als_tokenizer_32k"
        preferred_path = Path(tokenizer_dir) / f"{preferred_name}.json"
        if preferred_path.exists():
            # Find the 32k candidate and make it the winner
            for c in candidates:
                if c["name"] == preferred_name:
                    winner = c
                    break
            else:
                winner["name"] = preferred_name
                winner["tok_path"] = str(preferred_path)
            print(f"  Selected {preferred_name} as canonical (all candidates identical)")

    print(f"  Winner: {winner['name']} (score: {winner['score']:.4f})")
    print(f"  Vocab size: {winner['vocab_size']:,}")
    print(f"  Flagged terms: {winner['flagged_count']}/{winner['total_terms']}")
    print(f"  Medical fertility: {winner['medical_fertility']:.4f}")

    # Write sweep summary markdown
    lines = []
    lines.append("# Tokenizer sweep summary")
    lines.append("")
    lines.append("This report summarizes the evaluation of all trained tokenizer candidates "
                 "and recommends the best vocabulary size for the ALS-LM project based on "
                 "medical term handling, text compression, and general-English trade-offs.")
    lines.append("")

    # Ranking table
    lines.append("## Ranking")
    lines.append("")
    lines.append("Candidates ranked by weighted composite score combining flagged medical terms "
                 "(40%), medical text fertility (30%), compression ratio (20%), and general-English "
                 "penalty (10%).")
    lines.append("")
    lines.append("| Rank | Tokenizer          | Vocab  | Flagged | Medical fert. | Compression | General fert. | Score  |")
    lines.append("|------|--------------------|--------|---------|---------------|-------------|---------------|--------|")
    for rank, c in enumerate(candidates, 1):
        rec = "WINNER" if rank == 1 else ""
        lines.append(
            f"| {rank:>4} | {c['name']:<18} | {c['vocab_size']:>6,} | "
            f"{c['flagged_count']:>3}/{c['total_terms']:<3} | "
            f"{c['medical_fertility']:>13.4f} | {c['compression_ratio']:>11.4f} | "
            f"{c['general_fertility']:>13.4f} | {c['score']:>.4f} |"
        )
    lines.append("")

    # Winner declaration
    lines.append("## Recommendation")
    lines.append("")

    if all_same:
        lines.append(f"**Selected tokenizer:** {winner['name']} (vocab size: {winner['vocab_size']:,})")
        lines.append("")
        lines.append(f"All three tokenizer candidates converged to the same vocabulary size "
                     f"({winner['vocab_size']:,}) because the sample training corpus (~2 MB) "
                     f"lacks sufficient diversity for the BPE algorithm to produce 16K+ merges "
                     f"with `min_frequency=2`. As a result, all candidates have identical "
                     f"performance metrics and the selection is nominal.")
        lines.append("")
        lines.append(f"The 32K target is selected as the canonical tokenizer because it "
                     f"represents the middle ground in the sweep range. When re-trained on the "
                     f"full ALS corpus (50-100 MB), the three targets will produce distinct "
                     f"vocabularies and the ranking will be meaningful.")
    else:
        lines.append(f"**Selected tokenizer:** {winner['name']} (vocab size: {winner['vocab_size']:,})")
        lines.append("")
        lines.append(f"The {winner['name']} tokenizer achieves the best balance across all "
                     f"evaluation criteria. It flags {winner['flagged_count']} of "
                     f"{winner['total_terms']} medical terms for excessive fragmentation "
                     f"({winner['flagged_pct']:.1f}%), with medical text fertility of "
                     f"{winner['medical_fertility']:.4f} tokens per word and compression "
                     f"ratio of {winner['compression_ratio']:.4f} characters per token.")

    lines.append("")

    # Trade-off analysis
    lines.append("## Trade-off analysis")
    lines.append("")

    gpt2_fertility = comparison["comparisons"][winner["name"]]["text_metrics"]["baseline"]["fertility"]
    gpt2_compression = comparison["comparisons"][winner["name"]]["text_metrics"]["baseline"]["compression_ratio"]
    gpt2_ge = comparison["comparisons"][winner["name"]]["general_english"]["baseline"]["avg_fertility"]

    lines.append("The custom tokenizer makes several trade-offs compared to GPT-2.")
    lines.append("")
    lines.append(f"**Advantages over GPT-2:**")
    lines.append("")
    lines.append(f"- Lower medical text fertility ({winner['medical_fertility']:.4f} vs "
                 f"{gpt2_fertility:.4f}), producing fewer tokens for ALS content")
    lines.append(f"- Higher compression ratio ({winner['compression_ratio']:.4f} vs "
                 f"{gpt2_compression:.4f} chars/token), more efficient encoding")
    lines.append(f"- Won on {winner['wins']} of {winner['wins'] + winner['losses'] + winner['ties']} "
                 f"medical term comparisons")
    lines.append("")
    lines.append(f"**Disadvantages vs GPT-2:**")
    lines.append("")
    lines.append(f"- Significantly higher general-English fertility ({winner['general_fertility']:.4f} "
                 f"vs {gpt2_ge:.4f}), fragmenting common words more")
    lines.append(f"- Lost on {winner['losses']} medical term comparisons (mostly generic clinical "
                 f"terms that GPT-2 covers well)")
    lines.append(f"- Much smaller vocabulary ({winner['vocab_size']:,} vs 50,257) limits coverage "
                 f"of rare word forms")
    lines.append("")

    lines.append("The general-English penalty is expected for a domain-specific tokenizer trained "
                 "on a specialized corpus. For the ALS-LM use case (generating domain-specific text), "
                 "optimizing for medical term handling is the correct priority.")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*")
    lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Sweep summary written to: {output_path}")

    # Return winner info for downstream use
    return winner, all_same


def copy_winning_tokenizer(winner, tokenizer_dir, output_path):
    """Copy the winning tokenizer to the canonical als_tokenizer.json path."""
    src = Path(winner["tok_path"])
    dst = Path(output_path)

    if not src.exists():
        print(f"Error: winning tokenizer not found at {src}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(dst.parent, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    print(f"  Winner copied: {src} -> {dst}")


def prepare_nanogpt_data(tokenizer_path, train_txt, val_txt, output_dir):
    """Encode corpus text files into nanoGPT-compatible binary format.

    Produces:
    - train.bin: uint16 numpy array of training token IDs
    - val.bin: uint16 numpy array of validation token IDs
    - meta.pkl: dict with vocab_size, itos, stoi mappings
    """
    print("\n=== Preparing nanoGPT Data ===")

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_path}")
    tok = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tok.get_vocab_size()
    print(f"  Vocab size: {vocab_size:,}")

    # Safety check: uint16 can only hold values 0-65535
    if vocab_size > 65535:
        print(f"Error: vocab_size ({vocab_size}) exceeds uint16 max (65535). "
              f"Cannot encode as uint16.", file=sys.stderr)
        sys.exit(1)
    print(f"  uint16 safety check passed ({vocab_size} <= 65535)")

    os.makedirs(output_dir, exist_ok=True)

    # Encode train.txt
    print(f"\nEncoding training data: {train_txt}")
    train_size = os.path.getsize(train_txt)
    print(f"  File size: {train_size:,} bytes ({train_size / 1024 / 1024:.1f} MB)")

    if train_size > 500 * 1024 * 1024:
        # Chunked encoding for large files
        print("  Using chunked encoding (file > 500 MB)")
        train_ids = _encode_chunked(tok, train_txt)
    else:
        with open(train_txt, encoding="utf-8") as f:
            train_text = f.read()
        encoded = tok.encode(train_text)
        train_ids = np.array(encoded.ids, dtype=np.uint16)

    train_bin_path = os.path.join(output_dir, "train.bin")
    train_ids.tofile(train_bin_path)
    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Train bin size: {os.path.getsize(train_bin_path):,} bytes")

    # Encode val.txt
    print(f"\nEncoding validation data: {val_txt}")
    val_size = os.path.getsize(val_txt)
    print(f"  File size: {val_size:,} bytes ({val_size / 1024 / 1024:.1f} MB)")

    if val_size > 500 * 1024 * 1024:
        print("  Using chunked encoding (file > 500 MB)")
        val_ids = _encode_chunked(tok, val_txt)
    else:
        with open(val_txt, encoding="utf-8") as f:
            val_text = f.read()
        encoded = tok.encode(val_text)
        val_ids = np.array(encoded.ids, dtype=np.uint16)

    val_bin_path = os.path.join(output_dir, "val.bin")
    val_ids.tofile(val_bin_path)
    print(f"  Val tokens: {len(val_ids):,}")
    print(f"  Val bin size: {os.path.getsize(val_bin_path):,} bytes")

    # Build meta.pkl with itos and stoi mappings
    print("\nBuilding meta.pkl vocabulary mappings...")
    itos = {}
    stoi = {}
    for i in range(vocab_size):
        token_str = tok.id_to_token(i)
        if token_str is not None:
            itos[i] = token_str
            stoi[token_str] = i

    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }

    meta_path = os.path.join(output_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  meta.pkl: vocab_size={vocab_size}, itos={len(itos)} entries, stoi={len(stoi)} entries")
    print(f"  meta.pkl size: {os.path.getsize(meta_path):,} bytes")

    # Summary
    print("\n=== nanoGPT Data Preparation Complete ===")
    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens:   {len(val_ids):,}")
    print(f"  Vocab size:   {vocab_size:,}")
    print(f"  Output dir:   {output_dir}")


def _encode_chunked(tok, filepath):
    """Encode a large file by splitting on <|endoftext|> boundaries."""
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    documents = text.split("<|endoftext|>")
    all_ids = []

    # Get the ID for <|endoftext|>
    eot_enc = tok.encode("<|endoftext|>")
    eot_id = eot_enc.ids[0] if eot_enc.ids else None

    for i, doc in enumerate(documents):
        if not doc.strip():
            continue
        encoded = tok.encode(doc)
        all_ids.extend(encoded.ids)
        # Re-add the separator between documents
        if eot_id is not None and i < len(documents) - 1:
            all_ids.append(eot_id)

    return np.array(all_ids, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sweep summary and prepare nanoGPT training data"
    )
    parser.add_argument(
        "--sweep-summary",
        action="store_true",
        help="Generate sweep summary only (skip encoding)",
    )
    parser.add_argument(
        "--encode",
        action="store_true",
        help="Encode corpus only (skip sweep summary)",
    )
    parser.add_argument(
        "--comparison-report",
        default="reports/comparison_report.json",
        help="Path to comparison report JSON (default: reports/comparison_report.json)",
    )
    parser.add_argument(
        "--validation-report",
        default="reports/validation_report.json",
        help="Path to validation report JSON (default: reports/validation_report.json)",
    )
    parser.add_argument(
        "--sweep-output",
        default="reports/sweep_summary.md",
        help="Output path for sweep summary (default: reports/sweep_summary.md)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default="tokenizer",
        help="Directory containing tokenizer candidates (default: tokenizer)",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer/als_tokenizer.json",
        help="Path to tokenizer JSON for encoding (default: tokenizer/als_tokenizer.json)",
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
        default="data/tokenized",
        help="Output directory for binary files (default: data/tokenized)",
    )

    args = parser.parse_args()

    # Default: run both steps
    run_sweep = not args.encode or args.sweep_summary
    run_encode = not args.sweep_summary or args.encode

    if not args.sweep_summary and not args.encode:
        run_sweep = True
        run_encode = True

    if run_sweep:
        winner, all_same = generate_sweep_summary(
            args.comparison_report,
            args.validation_report,
            args.sweep_output,
            args.tokenizer_dir,
        )

        # Copy winner to canonical path
        copy_winning_tokenizer(winner, args.tokenizer_dir, args.tokenizer)

    if run_encode:
        # Verify tokenizer exists
        if not Path(args.tokenizer).exists():
            print(f"Error: tokenizer not found at {args.tokenizer}", file=sys.stderr)
            print("Run with --sweep-summary first, or provide --tokenizer path", file=sys.stderr)
            sys.exit(1)

        prepare_nanogpt_data(
            args.tokenizer,
            args.train_txt,
            args.val_txt,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
