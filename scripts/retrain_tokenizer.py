#!/usr/bin/env python3
"""Retrain BPE tokenizer on the full ALS corpus and produce production artifacts.

Orchestration script that coordinates the full tokenizer retraining pipeline:
1. Archive current tokenizer files to tokenizer/{version}/
2. Validate real corpus exists and check size
3. Retrain tokenizer at 16K and 32K vocabulary sizes
4. Build medical term validation list (~100 domain-specific terms)
5. Select winner based on medical term single-token coverage
6. Generate three-way validation report (previous vs new ALS vs GPT-2)
7. Save tokenizer in HuggingFace transformers directory format
8. Retokenize corpus into nanoGPT binary format (train.bin, val.bin, meta.pkl)

Usage:
    python scripts/retrain_tokenizer.py
    python scripts/retrain_tokenizer.py --archive-version v0.2
    python scripts/retrain_tokenizer.py --dry-run
    python scripts/retrain_tokenizer.py --skip-archive
    python scripts/retrain_tokenizer.py --steps 1,2,3,4,5
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken
from tokenizers import Tokenizer

# Import the existing training function
sys.path.insert(0, str(Path(__file__).parent))
from train_tokenizer import build_tokenizer, train_single_tokenizer, vocab_size_label


# General English words to filter out of the medical term list.
# These appear frequently in medical text but are not domain-specific.
GENERAL_ENGLISH_FILTER = {
    "patient", "patients", "study", "studies", "treatment", "treatments",
    "method", "methods", "result", "results", "group", "groups",
    "case", "cases", "effect", "effects", "level", "levels",
    "change", "changes", "time", "year", "years", "age",
    "use", "used", "report", "reported", "data", "model",
    "analysis", "type", "types", "form", "forms", "number",
    "rate", "rates", "risk", "role", "test", "tests",
    "cell", "cells", "control", "controls", "disease", "diseases",
    "sample", "samples", "value", "values", "factor", "factors",
    "system", "systems", "process", "response", "activity",
    "function", "period", "area", "areas", "body", "side",
    "condition", "conditions", "measure", "measures", "trial", "trials",
    "range", "evidence", "protein", "proteins", "gene", "genes",
    "review", "outcome", "outcomes", "score", "scores",
    "population", "association", "expression", "mechanism",
    "compared", "significant", "associated", "increased", "decreased",
    "observed", "showed", "found", "present", "present",
    "early", "late", "high", "low", "normal", "common",
    "total", "mean", "average", "positive", "negative",
}


def step1_archive(tokenizer_dir: Path, archive_version: str = "v0.1", dry_run: bool = False) -> None:
    """Archive existing tokenizer files to tokenizer/{archive_version}/.

    Args:
        tokenizer_dir: Path to tokenizer directory.
        archive_version: Version label for the archive directory (e.g., "v0.1", "v0.2").
        dry_run: If True, report planned actions without executing.
    """
    print(f"\n=== Step 1: Archive tokenizer files to {archive_version} ===")

    archive_dir = tokenizer_dir / archive_version
    files_to_archive = [
        "als_tokenizer.json",
        "als_tokenizer_16k.json",
        "als_tokenizer_32k.json",
        "als_tokenizer_50k.json",
        "training_summary.json",
        "validation_report.json",
        "VALIDATION.md",
    ]

    if archive_dir.exists():
        print(f"  WARNING: Archive directory {archive_dir} already exists, skipping archive.")
        return

    existing = [f for f in files_to_archive if (tokenizer_dir / f).exists()]

    if not existing:
        print(f"  No files found to archive, skipping.")
        return

    print(f"  Found {len(existing)} files to archive:")
    for f in existing:
        src = tokenizer_dir / f
        size_kb = src.stat().st_size / 1024
        print(f"    {f} ({size_kb:.1f} KB)")

    if dry_run:
        print(f"  [DRY RUN] Would create {archive_dir} and copy {len(existing)} files")
        return

    archive_dir.mkdir(parents=True, exist_ok=True)

    for f in existing:
        src = tokenizer_dir / f
        dst = archive_dir / f
        shutil.copy2(str(src), str(dst))
        print(f"  Archived: {f} -> {archive_version}/{f}")

    print(f"  Archived {len(existing)} files to {archive_dir}")


def step2_validate_corpus(corpus_path: Path, dry_run: bool = False) -> float:
    """Validate the training corpus exists and report its size.

    Returns the corpus size in MB.
    """
    print("\n=== Step 2: Validate corpus ===")

    if not corpus_path.exists():
        print(f"  ERROR: Corpus file not found: {corpus_path}", file=sys.stderr)
        print(
            "  The training corpus (data/processed/train.txt) does not exist.",
            file=sys.stderr,
        )
        print(
            "  Run the Phase 2 data processing pipeline first to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    size_bytes = corpus_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    print(f"  Corpus: {corpus_path}")
    print(f"  Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    if size_mb < 0.1:
        print(
            f"  ERROR: Corpus is too small ({size_mb:.2f} MB). "
            "Expected at least 0.1 MB of training text.",
            file=sys.stderr,
        )
        sys.exit(1)

    if size_mb < 10:
        print(
            f"  NOTE: Corpus is {size_mb:.2f} MB, which is smaller than the "
            "~100 MB target. Tokenizer vocabulary sizes may converge to the "
            "same value due to limited BPE merges. This is expected for "
            "small corpora and will not affect pipeline correctness."
        )

    if dry_run:
        print(f"  [DRY RUN] Corpus validated at {size_mb:.2f} MB")

    return size_mb


def step3_retrain_tokenizers(
    corpus_path: Path,
    tokenizer_dir: Path,
    dry_run: bool = False,
) -> list[dict]:
    """Retrain tokenizers at 16K and 32K vocabulary sizes.

    Returns a list of training statistics dictionaries.
    """
    print("\n=== Step 3: Retrain tokenizers at 16K and 32K ===")

    vocab_sizes = [16384, 32768]

    if dry_run:
        for vs in vocab_sizes:
            label = vocab_size_label(vs)
            output = tokenizer_dir / f"als_tokenizer_{label}.json"
            print(f"  [DRY RUN] Would train vocab_size={vs} -> {output}")
        return []

    all_stats = []
    for vocab_size in vocab_sizes:
        label = vocab_size_label(vocab_size)
        output_path = str(tokenizer_dir / f"als_tokenizer_{label}.json")
        stats = train_single_tokenizer(str(corpus_path), output_path, vocab_size)
        all_stats.append(stats)

    # Save training summary
    summary_path = tokenizer_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Training summary written to {summary_path}")

    return all_stats


def step4_build_medical_terms(
    corpus_path: Path,
    existing_terms_path: Path,
    dry_run: bool = False,
) -> list[dict]:
    """Build a filtered list of ~100 domain-specific medical terms.

    Starts from the existing medical terms file (Phase 3, 195 terms),
    filters out general English words, and returns ~100 categorized terms.
    """
    print("\n=== Step 4: Build medical term validation list ===")

    terms = []

    if existing_terms_path.exists():
        with open(existing_terms_path, "r", encoding="utf-8") as f:
            terms = json.load(f)
        print(f"  Loaded {len(terms)} existing terms from {existing_terms_path}")
    else:
        # Fall back to extracting from corpus
        print(f"  No existing terms file found at {existing_terms_path}")
        print("  Extracting terms from corpus...")
        from extract_medical_terms import build_term_list

        terms = build_term_list(str(corpus_path), min_frequency=3)
        print(f"  Extracted {len(terms)} terms from corpus")

    # Filter out general English words
    filtered = []
    removed = []
    for t in terms:
        term_lower = t["term"].lower()
        # Check if the term itself is a general English word
        if term_lower in GENERAL_ENGLISH_FILTER:
            removed.append(t["term"])
            continue
        # Check if it is a single common word (not multi-word medical terms)
        words = term_lower.split()
        if len(words) == 1 and term_lower in GENERAL_ENGLISH_FILTER:
            removed.append(t["term"])
            continue
        filtered.append(t)

    print(f"  Filtered out {len(removed)} general English words")
    if removed[:10]:
        print(f"    Examples removed: {', '.join(removed[:10])}")

    # Sort by corpus_frequency descending, take top 100
    filtered.sort(key=lambda x: x.get("corpus_frequency", 0), reverse=True)
    top_terms = filtered[:100]

    # Categorize summary
    categories = {}
    for t in top_terms:
        cat = t.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n  Final term list: {len(top_terms)} terms across {len(categories)} categories:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")

    if dry_run:
        print(f"  [DRY RUN] Would use {len(top_terms)} terms for validation")
        return top_terms

    return top_terms


def step5_select_winner(
    tokenizer_dir: Path,
    terms: list[dict],
    dry_run: bool = False,
) -> str:
    """Select the winning tokenizer based on medical term single-token coverage.

    The vocab size that produces single-token encoding for the most domain
    terms wins. If tied or all candidates converge, select the larger vocab.

    Returns the path to the winning tokenizer file.
    """
    print("\n=== Step 5: Select winner based on medical term coverage ===")

    candidates = []
    for label in ["16k", "32k"]:
        tok_path = tokenizer_dir / f"als_tokenizer_{label}.json"
        if not tok_path.exists():
            print(f"  WARNING: {tok_path} not found, skipping")
            continue
        candidates.append((label, tok_path))

    if not candidates:
        print("  ERROR: No trained tokenizers found", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        print(f"  [DRY RUN] Would evaluate {len(candidates)} candidates against {len(terms)} terms")
        return str(tokenizer_dir / "als_tokenizer.json")

    results = {}
    for label, tok_path in candidates:
        tok = Tokenizer.from_file(str(tok_path))
        vocab_size = tok.get_vocab_size()
        single_token_count = 0
        total_subtokens = 0

        for t in terms:
            encoded = tok.encode(t["term"])
            n_tokens = len(encoded.ids)
            total_subtokens += n_tokens
            if n_tokens == 1:
                single_token_count += 1

        avg_subtokens = total_subtokens / len(terms) if terms else 0

        results[label] = {
            "vocab_size": vocab_size,
            "single_token_count": single_token_count,
            "avg_subtokens": round(avg_subtokens, 2),
            "path": str(tok_path),
        }

        print(
            f"  {label}: vocab={vocab_size:,}, "
            f"single-token={single_token_count}/{len(terms)}, "
            f"avg subtokens={avg_subtokens:.2f}"
        )

    # Check if candidates converged to the same vocab size
    vocab_sizes = [r["vocab_size"] for r in results.values()]
    all_same = len(set(vocab_sizes)) == 1

    if all_same:
        print(
            f"\n  All candidates converged to vocab_size={vocab_sizes[0]:,}. "
            "Corpus exhausted all possible BPE merges."
        )
        # When converged, select 32k as canonical (if available)
        winner_label = "32k" if "32k" in results else list(results.keys())[0]
        print(f"  Selected {winner_label} as canonical (all identical)")
    else:
        # Select by most single-token coverage, break ties by lower avg subtokens
        winner_label = max(
            results.keys(),
            key=lambda k: (
                results[k]["single_token_count"],
                -results[k]["avg_subtokens"],
            ),
        )
        print(f"\n  Winner: {winner_label} ({results[winner_label]['single_token_count']} single-token terms)")

    # Check if we need 50K (neither covers top 100 as single tokens)
    winner_single = results[winner_label]["single_token_count"]
    if winner_single < len(terms) * 0.5:
        print(
            f"\n  NOTE: Winner covers only {winner_single}/{len(terms)} terms as single tokens."
        )
        # For small corpora, the vocab may be too small regardless of target.
        # Only attempt 50K if the vocab actually reached a meaningful size.
        if results[winner_label]["vocab_size"] >= 16000:
            print("  Evaluating 50K tokenizer...")
            tok_50k_path = tokenizer_dir / "als_tokenizer_50k.json"
            if not tok_50k_path.exists():
                print("  Training 50K tokenizer...")
                from train_tokenizer import train_single_tokenizer
                train_single_tokenizer(
                    str(Path("data/processed/train.txt")),
                    str(tok_50k_path),
                    50257,
                )
            tok_50k = Tokenizer.from_file(str(tok_50k_path))
            single_50k = sum(
                1 for t in terms if len(tok_50k.encode(t["term"]).ids) == 1
            )
            if single_50k > winner_single:
                print(f"  50K covers {single_50k} single-token terms, adopting as winner")
                winner_label = "50k"
                results["50k"] = {
                    "vocab_size": tok_50k.get_vocab_size(),
                    "single_token_count": single_50k,
                    "path": str(tok_50k_path),
                }
            else:
                print(
                    f"  50K covers {single_50k} single-token terms, "
                    "no improvement over current winner"
                )
        else:
            print(
                "  Corpus is too small for higher vocab to help. "
                "Proceeding with current winner."
            )

    # Copy winner to canonical path
    winner_path = results[winner_label]["path"]
    canonical_path = tokenizer_dir / "als_tokenizer.json"
    shutil.copy2(winner_path, str(canonical_path))
    print(f"\n  Winner copied to {canonical_path}")

    return str(canonical_path)


def step6_validation_report(
    tokenizer_dir: Path,
    terms: list[dict],
    corpus_path: Path,
    val_path: Path,
    archive_version: str = "v0.1",
    dry_run: bool = False,
) -> None:
    """Generate a three-way validation report comparing old, new, and GPT-2 tokenizers."""
    print("\n=== Step 6: Three-way validation report ===")

    new_tok_path = tokenizer_dir / "als_tokenizer.json"

    # Find the previous tokenizer for comparison, checking the most recent archive first
    old_tok_path = tokenizer_dir / archive_version / "als_tokenizer.json"
    if not old_tok_path.exists():
        # Fall back to v0.1 if the specified archive doesn't have the canonical tokenizer
        old_tok_path = tokenizer_dir / "v0.1" / "als_tokenizer.json"

    if not new_tok_path.exists():
        print(f"  ERROR: New tokenizer not found at {new_tok_path}", file=sys.stderr)
        sys.exit(1)

    # Load tokenizers
    new_tok = Tokenizer.from_file(str(new_tok_path))
    old_tok = None
    if old_tok_path.exists():
        old_tok = Tokenizer.from_file(str(old_tok_path))
    else:
        # Fall back to checking for any v0.1 tokenizer variant
        for variant in ["als_tokenizer_32k.json", "als_tokenizer_16k.json"]:
            fallback = tokenizer_dir / "v0.1" / variant
            if fallback.exists():
                old_tok = Tokenizer.from_file(str(fallback))
                old_tok_path = fallback
                break

    if old_tok is None:
        print("  WARNING: No v0.1 tokenizer found for comparison, using two-way report")

    gpt2_enc = tiktoken.get_encoding("gpt2")

    if dry_run:
        print(f"  [DRY RUN] Would generate three-way report with {len(terms)} terms")
        return

    # Load corpus text for metrics
    with open(corpus_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    val_text = ""
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            val_text = f.read()

    full_text = train_text + val_text

    # Helper to encode with GPT-2 tiktoken
    def gpt2_encode(text: str) -> list[int]:
        return gpt2_enc.encode(text, allowed_special={"<|endoftext|>"})

    def gpt2_encode_term(term: str) -> list[str]:
        ids = gpt2_enc.encode(term, allowed_special={"<|endoftext|>"})
        return [gpt2_enc.decode([i]) for i in ids]

    # Per-term analysis
    print("  Computing per-term tokenization...")
    term_results = []
    for t in terms:
        term = t["term"]
        category = t.get("category", "unknown")

        # New tokenizer
        new_enc = new_tok.encode(term)
        new_tokens = new_enc.tokens
        new_count = len(new_tokens)

        # Old tokenizer (if available)
        if old_tok:
            old_enc = old_tok.encode(term)
            old_tokens = old_enc.tokens
            old_count = len(old_tokens)
        else:
            old_tokens = []
            old_count = 0

        # GPT-2
        gpt2_tokens = gpt2_encode_term(term)
        gpt2_count = len(gpt2_tokens)

        term_results.append({
            "term": term,
            "category": category,
            "old_subtokens": old_count,
            "old_tokens": old_tokens,
            "new_subtokens": new_count,
            "new_tokens": new_tokens,
            "gpt2_subtokens": gpt2_count,
            "gpt2_tokens": gpt2_tokens,
        })

    # Corpus-level metrics
    print("  Computing corpus-level metrics...")

    # New tokenizer metrics
    new_train_enc = new_tok.encode(train_text)
    new_train_tokens = len(new_train_enc.ids)
    new_full_enc = new_tok.encode(full_text)
    new_full_tokens = len(new_full_enc.ids)

    words = full_text.split()
    word_count = len(words)
    char_count = len(full_text)

    new_fertility = new_full_tokens / word_count if word_count > 0 else 0
    new_compression = char_count / new_full_tokens if new_full_tokens > 0 else 0

    # Speed benchmark (encode a representative sample)
    sample_text = full_text[:100000]  # First 100K chars
    start = time.time()
    for _ in range(3):
        new_tok.encode(sample_text)
    new_speed = len(new_tok.encode(sample_text).ids) * 3 / (time.time() - start)

    # Old tokenizer metrics
    if old_tok:
        old_full_enc = old_tok.encode(full_text)
        old_full_tokens = len(old_full_enc.ids)
        old_fertility = old_full_tokens / word_count if word_count > 0 else 0
        old_compression = char_count / old_full_tokens if old_full_tokens > 0 else 0

        start = time.time()
        for _ in range(3):
            old_tok.encode(sample_text)
        old_speed = len(old_tok.encode(sample_text).ids) * 3 / (time.time() - start)
    else:
        old_full_tokens = 0
        old_fertility = 0
        old_compression = 0
        old_speed = 0

    # GPT-2 metrics
    gpt2_full_ids = gpt2_encode(full_text)
    gpt2_full_tokens = len(gpt2_full_ids)
    gpt2_fertility = gpt2_full_tokens / word_count if word_count > 0 else 0
    gpt2_compression = char_count / gpt2_full_tokens if gpt2_full_tokens > 0 else 0

    start = time.time()
    for _ in range(3):
        gpt2_encode(sample_text)
    gpt2_speed = len(gpt2_encode(sample_text)) * 3 / (time.time() - start)

    # Document length distribution (split on <|endoftext|>, measure tokens per doc)
    print("  Computing document length distribution...")
    documents = full_text.split("<|endoftext|>")
    documents = [d.strip() for d in documents if d.strip()]

    new_doc_lengths = [len(new_tok.encode(d).ids) for d in documents]
    gpt2_doc_lengths = [len(gpt2_encode(d)) for d in documents]

    if old_tok:
        old_doc_lengths = [len(old_tok.encode(d).ids) for d in documents]
    else:
        old_doc_lengths = []

    def pct_within(lengths, threshold):
        if not lengths:
            return 0.0
        return round(100 * sum(1 for l in lengths if l <= threshold) / len(lengths), 1)

    new_pct_1024 = pct_within(new_doc_lengths, 1024)
    gpt2_pct_1024 = pct_within(gpt2_doc_lengths, 1024)
    old_pct_1024 = pct_within(old_doc_lengths, 1024) if old_doc_lengths else 0

    # Build JSON report
    report = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "corpus_file": str(corpus_path),
            "corpus_size_mb": round(char_count / (1024 * 1024), 2),
            "corpus_words": word_count,
            "corpus_chars": char_count,
            "num_documents": len(documents),
            "num_terms": len(terms),
        },
        "tokenizers": {
            "old_v01": {
                "name": "ALS v0.1",
                "path": str(old_tok_path) if old_tok else None,
                "vocab_size": old_tok.get_vocab_size() if old_tok else 0,
                "total_corpus_tokens": old_full_tokens,
                "fertility": round(old_fertility, 4),
                "compression_ratio": round(old_compression, 4),
                "encoding_speed_tps": round(old_speed),
                "pct_docs_within_1024": old_pct_1024,
            },
            "new_production": {
                "name": "ALS Production",
                "path": str(new_tok_path),
                "vocab_size": new_tok.get_vocab_size(),
                "total_corpus_tokens": new_full_tokens,
                "fertility": round(new_fertility, 4),
                "compression_ratio": round(new_compression, 4),
                "encoding_speed_tps": round(new_speed),
                "pct_docs_within_1024": new_pct_1024,
            },
            "gpt2": {
                "name": "GPT-2",
                "vocab_size": gpt2_enc.n_vocab,
                "total_corpus_tokens": gpt2_full_tokens,
                "fertility": round(gpt2_fertility, 4),
                "compression_ratio": round(gpt2_compression, 4),
                "encoding_speed_tps": round(gpt2_speed),
                "pct_docs_within_1024": gpt2_pct_1024,
            },
        },
        "term_results": term_results,
    }

    # Save JSON report
    json_path = tokenizer_dir / "validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  JSON report saved to {json_path}")

    # Generate Markdown report
    md_lines = _generate_validation_md(report)
    md_path = tokenizer_dir / "VALIDATION.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Markdown report saved to {md_path}")


def _generate_validation_md(report: dict) -> list[str]:
    """Generate a human-readable Markdown validation report."""
    lines = []
    meta = report["metadata"]
    toks = report["tokenizers"]
    term_results = report["term_results"]

    old = toks["old_v01"]
    new = toks["new_production"]
    gpt2 = toks["gpt2"]

    lines.append("# Tokenizer validation report")
    lines.append("")
    lines.append(
        "Three-way comparison of the v0.1 prototype tokenizer, the new production "
        "tokenizer, and the GPT-2 standard tokenizer. This report validates medical "
        "term handling, corpus-level efficiency, and encoding performance."
    )
    lines.append("")
    lines.append(f"- **Date:** {meta['date']}")
    lines.append(f"- **Corpus size:** {meta['corpus_size_mb']} MB ({meta['corpus_words']:,} words, {meta['corpus_chars']:,} characters)")
    lines.append(f"- **Documents:** {meta['num_documents']:,}")
    lines.append(f"- **Terms evaluated:** {meta['num_terms']}")
    lines.append("")

    # Corpus-level comparison table
    lines.append("## Corpus-level metrics")
    lines.append("")
    lines.append(
        "Key metrics comparing tokenizer efficiency across the full corpus."
    )
    lines.append("")
    lines.append(f"| {'Metric':<35} | {'v0.1 ALS':>12} | {'New ALS':>12} | {'GPT-2':>12} |")
    lines.append(f"|{'-' * 37}|{'-' * 14}|{'-' * 14}|{'-' * 14}|")
    lines.append(f"| {'Vocabulary size':<35} | {old['vocab_size']:>12,} | {new['vocab_size']:>12,} | {gpt2['vocab_size']:>12,} |")
    lines.append(f"| {'Total corpus tokens':<35} | {old['total_corpus_tokens']:>12,} | {new['total_corpus_tokens']:>12,} | {gpt2['total_corpus_tokens']:>12,} |")
    lines.append(f"| {'Fertility (tokens/word)':<35} | {old['fertility']:>12.4f} | {new['fertility']:>12.4f} | {gpt2['fertility']:>12.4f} |")
    lines.append(f"| {'Compression (chars/token)':<35} | {old['compression_ratio']:>12.4f} | {new['compression_ratio']:>12.4f} | {gpt2['compression_ratio']:>12.4f} |")
    lines.append(f"| {'Encoding speed (tokens/sec)':<35} | {old['encoding_speed_tps']:>12,} | {new['encoding_speed_tps']:>12,} | {gpt2['encoding_speed_tps']:>12,} |")
    lines.append(f"| {'Docs within 1024 tokens (%)':<35} | {old['pct_docs_within_1024']:>11.1f}% | {new['pct_docs_within_1024']:>11.1f}% | {gpt2['pct_docs_within_1024']:>11.1f}% |")
    lines.append("")

    # Per-category term analysis
    categories = {}
    for r in term_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    lines.append("## Medical term analysis by category")
    lines.append("")
    lines.append(
        "Each term is encoded by all three tokenizers. Lower subtoken counts "
        "indicate better domain vocabulary coverage."
    )
    lines.append("")

    for cat in sorted(categories.keys()):
        cat_terms = categories[cat]
        lines.append(f"### {cat.capitalize()}")
        lines.append("")
        lines.append(f"Terms in this category: {len(cat_terms)}")
        lines.append("")

        lines.append(f"| {'Term':<35} | {'v0.1':>5} | {'New':>5} | {'GPT-2':>5} | {'New tokens':<50} |")
        lines.append(f"|{'-' * 37}|{'-' * 7}|{'-' * 7}|{'-' * 7}|{'-' * 52}|")

        # Sort by new subtoken count descending
        cat_terms_sorted = sorted(cat_terms, key=lambda x: x["new_subtokens"])
        for r in cat_terms_sorted:
            new_tokens_str = " + ".join(f"`{t}`" for t in r["new_tokens"][:8])
            if len(r["new_tokens"]) > 8:
                new_tokens_str += " ..."
            lines.append(
                f"| {r['term']:<35} | "
                f"{r['old_subtokens']:>5} | "
                f"{r['new_subtokens']:>5} | "
                f"{r['gpt2_subtokens']:>5} | "
                f"{new_tokens_str:<50} |"
            )
        lines.append("")

    # Summary statistics
    old_avg = sum(r["old_subtokens"] for r in term_results) / len(term_results) if term_results else 0
    new_avg = sum(r["new_subtokens"] for r in term_results) / len(term_results) if term_results else 0
    gpt2_avg = sum(r["gpt2_subtokens"] for r in term_results) / len(term_results) if term_results else 0

    new_single = sum(1 for r in term_results if r["new_subtokens"] == 1)
    old_single = sum(1 for r in term_results if r["old_subtokens"] == 1)
    gpt2_single = sum(1 for r in term_results if r["gpt2_subtokens"] == 1)

    # Wins/losses: new vs old, new vs GPT-2
    new_vs_old_wins = sum(1 for r in term_results if r["new_subtokens"] < r["old_subtokens"])
    new_vs_old_losses = sum(1 for r in term_results if r["new_subtokens"] > r["old_subtokens"])
    new_vs_old_ties = sum(1 for r in term_results if r["new_subtokens"] == r["old_subtokens"])

    new_vs_gpt2_wins = sum(1 for r in term_results if r["new_subtokens"] < r["gpt2_subtokens"])
    new_vs_gpt2_losses = sum(1 for r in term_results if r["new_subtokens"] > r["gpt2_subtokens"])
    new_vs_gpt2_ties = sum(1 for r in term_results if r["new_subtokens"] == r["gpt2_subtokens"])

    lines.append("## Summary statistics")
    lines.append("")
    lines.append(
        "Aggregate comparison across all evaluated terms."
    )
    lines.append("")
    lines.append(f"| {'Metric':<35} | {'v0.1 ALS':>12} | {'New ALS':>12} | {'GPT-2':>12} |")
    lines.append(f"|{'-' * 37}|{'-' * 14}|{'-' * 14}|{'-' * 14}|")
    lines.append(f"| {'Average subtokens per term':<35} | {old_avg:>12.2f} | {new_avg:>12.2f} | {gpt2_avg:>12.2f} |")
    lines.append(f"| {'Single-token terms':<35} | {old_single:>12} | {new_single:>12} | {gpt2_single:>12} |")
    lines.append("")

    lines.append(f"| {'Comparison':<35} | {'Wins':>6} | {'Losses':>6} | {'Ties':>6} |")
    lines.append(f"|{'-' * 37}|{'-' * 8}|{'-' * 8}|{'-' * 8}|")
    lines.append(f"| {'New ALS vs v0.1 ALS':<35} | {new_vs_old_wins:>6} | {new_vs_old_losses:>6} | {new_vs_old_ties:>6} |")
    lines.append(f"| {'New ALS vs GPT-2':<35} | {new_vs_gpt2_wins:>6} | {new_vs_gpt2_losses:>6} | {new_vs_gpt2_ties:>6} |")
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by scripts/retrain_tokenizer.py on {meta['date']}*")
    lines.append("")

    return lines


def step7_save_hf_format(
    tokenizer_dir: Path,
    dry_run: bool = False,
) -> None:
    """Save the winning tokenizer in HuggingFace transformers directory format."""
    print("\n=== Step 7: Save tokenizer in HuggingFace format ===")

    tok_path = tokenizer_dir / "als_tokenizer.json"
    hf_dir = tokenizer_dir / "hf_tokenizer"

    if not tok_path.exists():
        print(f"  ERROR: Tokenizer not found at {tok_path}", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        print(f"  [DRY RUN] Would save HuggingFace format to {hf_dir}")
        return

    from transformers import PreTrainedTokenizerFast

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_path),
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )

    fast_tokenizer.save_pretrained(str(hf_dir))
    print(f"  Saved HuggingFace format to {hf_dir}")

    # Verify it loads correctly
    from transformers import AutoTokenizer

    loaded = AutoTokenizer.from_pretrained(str(hf_dir))
    test_text = "riluzole neurodegeneration SOD1"
    encoded = loaded.encode(test_text)
    print(f"  Verification: '{test_text}' -> {len(encoded)} tokens")

    # List saved files
    for f in sorted(hf_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name} ({size_kb:.1f} KB)")


def step8_retokenize_corpus(
    tokenizer_dir: Path,
    train_txt: Path,
    val_txt: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Retokenize the corpus into nanoGPT binary format."""
    print("\n=== Step 8: Retokenize corpus ===")

    tok_path = tokenizer_dir / "als_tokenizer.json"

    if not tok_path.exists():
        print(f"  ERROR: Tokenizer not found at {tok_path}", file=sys.stderr)
        sys.exit(1)

    if not train_txt.exists():
        print(f"  ERROR: Training text not found at {train_txt}", file=sys.stderr)
        sys.exit(1)

    if not val_txt.exists():
        print(f"  ERROR: Validation text not found at {val_txt}", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        print(f"  [DRY RUN] Would retokenize {train_txt} and {val_txt}")
        print(f"  [DRY RUN] Output to {output_dir}")
        return

    tok = Tokenizer.from_file(str(tok_path))
    vocab_size = tok.get_vocab_size()
    print(f"  Tokenizer vocab size: {vocab_size:,}")

    # Determine dtype based on vocab size
    if vocab_size > 65535:
        dtype = np.uint32
        print("  Using uint32 (vocab_size > 65535)")
    else:
        dtype = np.uint16
        print("  Using uint16 (vocab_size <= 65535)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode train.txt
    print(f"\n  Encoding training data: {train_txt}")
    with open(train_txt, "r", encoding="utf-8") as f:
        train_text = f.read()
    train_encoded = tok.encode(train_text)
    train_ids = np.array(train_encoded.ids, dtype=dtype)
    train_bin_path = output_dir / "train.bin"
    train_ids.tofile(str(train_bin_path))
    print(f"    Tokens: {len(train_ids):,}")
    print(f"    File size: {train_bin_path.stat().st_size:,} bytes")

    # Encode val.txt
    print(f"\n  Encoding validation data: {val_txt}")
    with open(val_txt, "r", encoding="utf-8") as f:
        val_text = f.read()
    val_encoded = tok.encode(val_text)
    val_ids = np.array(val_encoded.ids, dtype=dtype)
    val_bin_path = output_dir / "val.bin"
    val_ids.tofile(str(val_bin_path))
    print(f"    Tokens: {len(val_ids):,}")
    print(f"    File size: {val_bin_path.stat().st_size:,} bytes")

    # Build meta.pkl
    print("\n  Building meta.pkl...")
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

    meta_path = output_dir / "meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"    vocab_size: {vocab_size}")
    print(f"    itos entries: {len(itos)}")
    print(f"    stoi entries: {len(stoi)}")
    print(f"    File size: {meta_path.stat().st_size:,} bytes")

    print(f"\n  Retokenization complete. Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrain BPE tokenizer on ALS corpus and produce production artifacts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned actions without executing",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Skip step 1 (archiving files), useful for re-runs",
    )
    parser.add_argument(
        "--archive-version",
        default="v0.1",
        help="Version label for archive directory (default: v0.1). Use v0.2 for Phase 4.2.1.",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="Comma-separated list of steps to run (e.g., '1,2,3,4,5'). Default: all steps.",
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/train.txt",
        help="Path to training corpus (default: data/processed/train.txt)",
    )
    parser.add_argument(
        "--val-txt",
        default="data/processed/val.txt",
        help="Path to validation text (default: data/processed/val.txt)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default="tokenizer",
        help="Tokenizer output directory (default: tokenizer)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/tokenized",
        help="Binary output directory (default: data/tokenized)",
    )
    parser.add_argument(
        "--terms",
        default="reports/medical_terms.json",
        help="Path to existing medical terms JSON (default: reports/medical_terms.json)",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    val_path = Path(args.val_txt)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir)
    terms_path = Path(args.terms)

    # Determine which steps to run
    all_steps = set(range(1, 9))
    if args.steps:
        steps_to_run = set(int(s.strip()) for s in args.steps.split(","))
    else:
        steps_to_run = all_steps

    if args.skip_archive:
        steps_to_run.discard(1)

    print("=" * 60)
    print("ALS-LM Tokenizer Retraining Pipeline")
    print("=" * 60)
    if args.dry_run:
        print("MODE: DRY RUN (no changes will be made)")
    print(f"Steps to run: {sorted(steps_to_run)}")
    print(f"Corpus: {corpus_path}")
    print(f"Tokenizer dir: {tokenizer_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Archive version: {args.archive_version}")

    start_time = time.time()

    # Step 1: Archive current files
    if 1 in steps_to_run:
        step1_archive(tokenizer_dir, archive_version=args.archive_version, dry_run=args.dry_run)

    # Step 2: Validate corpus
    if 2 in steps_to_run:
        step2_validate_corpus(corpus_path, dry_run=args.dry_run)

    # Step 3: Retrain tokenizers
    if 3 in steps_to_run:
        step3_retrain_tokenizers(corpus_path, tokenizer_dir, dry_run=args.dry_run)

    # Step 4: Build medical term list
    terms = []
    if 4 in steps_to_run:
        terms = step4_build_medical_terms(
            corpus_path, terms_path, dry_run=args.dry_run
        )
    elif any(s in steps_to_run for s in [5, 6]):
        # Need terms for later steps, load them
        if terms_path.exists():
            with open(terms_path, "r", encoding="utf-8") as f:
                raw_terms = json.load(f)
            # Apply filtering
            terms = [
                t for t in raw_terms
                if t["term"].lower() not in GENERAL_ENGLISH_FILTER
            ][:100]

    # Step 5: Select winner
    if 5 in steps_to_run:
        step5_select_winner(tokenizer_dir, terms, dry_run=args.dry_run)

    # Step 6: Validation report
    if 6 in steps_to_run:
        step6_validation_report(
            tokenizer_dir, terms, corpus_path, val_path,
            archive_version=args.archive_version, dry_run=args.dry_run
        )

    # Step 7: HuggingFace format
    if 7 in steps_to_run:
        step7_save_hf_format(tokenizer_dir, dry_run=args.dry_run)

    # Step 8: Retokenize corpus
    if 8 in steps_to_run:
        step8_retokenize_corpus(
            tokenizer_dir, corpus_path, val_path, output_dir, dry_run=args.dry_run
        )

    duration = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {duration:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
