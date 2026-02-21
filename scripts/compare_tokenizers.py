#!/usr/bin/env python3
"""Compare custom BPE tokenizers against GPT-2 baseline.

Reusable CLI that compares any HuggingFace tokenizer against GPT-2 (via tiktoken)
or another HuggingFace tokenizer. Produces term-level and text-level comparison
metrics with JSON and Markdown output.

Usage:
    # Compare a single tokenizer against GPT-2
    python scripts/compare_tokenizers.py \\
        --custom tokenizer/als_tokenizer_32k.json \\
        --baseline gpt2 \\
        --terms reports/medical_terms.json \\
        --text data/processed/val.txt \\
        --output-json reports/comparison_report.json \\
        --output-md reports/comparison_report.md

    # Compare all tokenizers in a directory against GPT-2
    python scripts/compare_tokenizers.py \\
        --all --custom tokenizer/ \\
        --baseline gpt2 \\
        --terms reports/medical_terms.json \\
        --text data/processed/val.txt \\
        --output-json reports/comparison_report.json \\
        --output-md reports/comparison_report.md
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer


# Common English words for general-English sanity check
GENERAL_ENGLISH_WORDS = [
    "the", "and", "is", "was", "have", "been", "would", "could", "about",
    "people", "because", "different", "important", "between", "through",
    "another", "something", "government", "information", "development",
    "should", "their", "which", "there", "after", "other", "before",
    "where", "these", "still", "every", "never", "world", "being",
    "under", "going", "might", "think", "right", "found", "place",
    "great", "while", "again", "years", "since", "during", "without",
    "within", "always", "however", "together", "children",
]


class TiktokenAdapter:
    """Wraps tiktoken encoding to provide a HuggingFace-like interface."""

    def __init__(self, encoding_name="gpt2"):
        self._enc = tiktoken.get_encoding(encoding_name)
        self._name = encoding_name

    @property
    def name(self):
        return self._name

    def get_vocab_size(self):
        return self._enc.n_vocab

    def encode(self, text):
        """Return an object with .ids and .tokens attributes."""
        ids = self._enc.encode(text, allowed_special={"<|endoftext|>"})
        tokens = [self._enc.decode([i]) for i in ids]
        return _EncodeResult(ids=ids, tokens=tokens)

    def decode(self, ids):
        return self._enc.decode(ids)


class _EncodeResult:
    """Simple container for encode results."""

    def __init__(self, ids, tokens):
        self.ids = ids
        self.tokens = tokens


class HFTokenizerAdapter:
    """Wraps a HuggingFace tokenizer to provide the same interface."""

    def __init__(self, tokenizer, name=None):
        self._tok = tokenizer
        self._name = name or "custom"

    @property
    def name(self):
        return self._name

    def get_vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        result = self._tok.encode(text)
        return _EncodeResult(ids=result.ids, tokens=result.tokens)

    def decode(self, ids):
        return self._tok.decode(ids)


def load_tokenizer(path_or_name):
    """Load a tokenizer from file path or special name like 'gpt2'."""
    if path_or_name == "gpt2":
        return TiktokenAdapter("gpt2")

    path = Path(path_or_name)
    if not path.exists():
        print(f"Error: tokenizer file not found: {path}", file=sys.stderr)
        sys.exit(1)

    tok = Tokenizer.from_file(str(path))
    name = path.stem
    return HFTokenizerAdapter(tok, name=name)


def compare_terms(custom, baseline, terms):
    """Compare tokenization of medical terms between two tokenizers.

    Returns a dict with wins, losses, ties, and per-term details.
    """
    results = []
    wins = 0
    losses = 0
    ties = 0

    for term_entry in terms:
        term = term_entry["term"]
        category = term_entry.get("category", "unknown")

        custom_enc = custom.encode(term)
        baseline_enc = baseline.encode(term)

        custom_count = len(custom_enc.ids)
        baseline_count = len(baseline_enc.ids)

        if custom_count < baseline_count:
            outcome = "win"
            wins += 1
        elif custom_count > baseline_count:
            outcome = "loss"
            losses += 1
        else:
            outcome = "tie"
            ties += 1

        results.append({
            "term": term,
            "category": category,
            "custom_subtokens": custom_count,
            "baseline_subtokens": baseline_count,
            "custom_tokens": custom_enc.tokens,
            "baseline_tokens": baseline_enc.tokens,
            "outcome": outcome,
            "improvement": baseline_count - custom_count,
        })

    # Sort wins by improvement (biggest first)
    top_wins = sorted(
        [r for r in results if r["outcome"] == "win"],
        key=lambda x: x["improvement"],
        reverse=True,
    )[:10]

    # Sort losses by regression (biggest first)
    top_losses = sorted(
        [r for r in results if r["outcome"] == "loss"],
        key=lambda x: x["improvement"],
    )[:10]

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total": len(results),
        "top_wins": top_wins,
        "top_losses": top_losses,
        "all_results": results,
    }


def compute_text_metrics(tokenizer, text):
    """Compute text-level metrics: fertility, compression, total tokens."""
    encoded = tokenizer.encode(text)
    total_tokens = len(encoded.ids)

    # Count words (whitespace-separated)
    words = text.split()
    word_count = len(words)

    # Characters per token (compression ratio)
    char_count = len(text)
    compression = char_count / total_tokens if total_tokens > 0 else 0

    # Tokens per word (fertility)
    fertility = total_tokens / word_count if word_count > 0 else 0

    return {
        "total_tokens": total_tokens,
        "word_count": word_count,
        "char_count": char_count,
        "fertility": round(fertility, 4),
        "compression_ratio": round(compression, 4),
    }


def compute_general_english_fertility(tokenizer, words=None):
    """Compute fertility on common English words.

    Returns per-word token counts and average fertility.
    """
    if words is None:
        words = GENERAL_ENGLISH_WORDS

    per_word = []
    total_tokens = 0
    for word in words:
        enc = tokenizer.encode(word)
        count = len(enc.ids)
        total_tokens += count
        per_word.append({
            "word": word,
            "subtokens": count,
            "tokens": enc.tokens,
        })

    avg_fertility = total_tokens / len(words) if words else 0

    return {
        "words_tested": len(words),
        "avg_fertility": round(avg_fertility, 4),
        "total_tokens": total_tokens,
        "per_word": per_word,
    }


def compare_single(custom, baseline, terms, text):
    """Run full comparison between one custom tokenizer and baseline."""
    # Term-level comparison
    term_comparison = compare_terms(custom, baseline, terms)

    # Text-level comparison
    custom_text_metrics = compute_text_metrics(custom, text)
    baseline_text_metrics = compute_text_metrics(baseline, text)

    # General-English sanity check
    custom_english = compute_general_english_fertility(custom)
    baseline_english = compute_general_english_fertility(baseline)

    return {
        "custom_name": custom.name,
        "custom_vocab_size": custom.get_vocab_size(),
        "baseline_name": baseline.name,
        "baseline_vocab_size": baseline.get_vocab_size(),
        "term_comparison": term_comparison,
        "text_metrics": {
            "custom": custom_text_metrics,
            "baseline": baseline_text_metrics,
        },
        "general_english": {
            "custom": custom_english,
            "baseline": baseline_english,
        },
    }


def generate_markdown_report(all_results, metadata):
    """Generate a Markdown comparison report from results."""
    lines = []

    lines.append("# Tokenizer comparison report")
    lines.append("")
    lines.append("This report compares custom ALS-domain tokenizers against the GPT-2 baseline "
                 "tokenizer, measuring medical term handling, text compression, and general-English "
                 "performance.")
    lines.append("")

    # Metadata
    lines.append("## Run metadata")
    lines.append("")
    lines.append(f"- **Date:** {metadata['date']}")
    lines.append(f"- **Baseline:** {metadata['baseline_name']} (vocab: {metadata['baseline_vocab_size']:,})")
    lines.append(f"- **Medical terms evaluated:** {metadata['term_count']}")
    lines.append(f"- **Text file:** {metadata['text_file']}")
    lines.append(f"- **Text size:** {metadata['text_chars']:,} characters, {metadata['text_words']:,} words")
    lines.append("")

    # Per-tokenizer sections
    for result in all_results:
        tc = result["term_comparison"]
        tm_custom = result["text_metrics"]["custom"]
        tm_baseline = result["text_metrics"]["baseline"]
        ge_custom = result["general_english"]["custom"]
        ge_baseline = result["general_english"]["baseline"]

        lines.append(f"## {result['custom_name']} (vocab: {result['custom_vocab_size']:,})")
        lines.append("")

        # Summary stats
        lines.append("### Summary statistics")
        lines.append("")
        lines.append(f"Term-level results against GPT-2 on {tc['total']} medical terms.")
        lines.append("")
        lines.append(f"| Metric                      | Custom            | GPT-2             |")
        lines.append(f"|-----------------------------|-------------------|--------------------|")
        lines.append(f"| Wins (fewer subtokens)      | {tc['wins']}               |                    |")
        lines.append(f"| Losses (more subtokens)     | {tc['losses']}              |                    |")
        lines.append(f"| Ties                        | {tc['ties']}               |                    |")
        lines.append(f"| Medical text fertility      | {tm_custom['fertility']:.4f}          | {tm_baseline['fertility']:.4f}           |")
        lines.append(f"| Compression ratio (char/tok)| {tm_custom['compression_ratio']:.4f}          | {tm_baseline['compression_ratio']:.4f}           |")
        lines.append(f"| Total tokens (val text)     | {tm_custom['total_tokens']:,}          | {tm_baseline['total_tokens']:,}           |")
        lines.append(f"| General-English fertility   | {ge_custom['avg_fertility']:.4f}          | {ge_baseline['avg_fertility']:.4f}           |")
        lines.append("")

        # Top wins
        if tc["top_wins"]:
            lines.append("### Top wins (custom tokenizer is better)")
            lines.append("")
            lines.append("Terms where the custom tokenizer uses fewer subtokens than GPT-2.")
            lines.append("")
            lines.append("| Term                              | Custom | GPT-2 | Improvement |")
            lines.append("|-----------------------------------|--------|-------|-------------|")
            for w in tc["top_wins"]:
                term_display = w["term"][:35].ljust(35)
                lines.append(f"| {term_display} | {w['custom_subtokens']:6d} | {w['baseline_subtokens']:5d} | {w['improvement']:+11d} |")
            lines.append("")

        # Top losses
        if tc["top_losses"]:
            lines.append("### Top losses (GPT-2 is better)")
            lines.append("")
            lines.append("Terms where GPT-2 uses fewer subtokens than the custom tokenizer. These "
                         "represent the cost of domain specialization on a smaller vocabulary.")
            lines.append("")
            lines.append("| Term                              | Custom | GPT-2 | Regression |")
            lines.append("|-----------------------------------|--------|-------|------------|")
            for l in tc["top_losses"]:
                term_display = l["term"][:35].ljust(35)
                lines.append(f"| {term_display} | {l['custom_subtokens']:6d} | {l['baseline_subtokens']:5d} | {l['improvement']:+10d} |")
            lines.append("")

        # General-English sanity check
        lines.append("### General-English sanity check")
        lines.append("")
        lines.append(f"Fertility comparison on {ge_custom['words_tested']} common English words. "
                     f"A fertility close to 1.0 means most words are single tokens. Higher values "
                     f"indicate more fragmentation.")
        lines.append("")
        lines.append(f"- **Custom average fertility:** {ge_custom['avg_fertility']:.4f}")
        lines.append(f"- **GPT-2 average fertility:** {ge_baseline['avg_fertility']:.4f}")

        fertility_ratio = ge_custom["avg_fertility"] / ge_baseline["avg_fertility"] if ge_baseline["avg_fertility"] > 0 else float("inf")
        if fertility_ratio > 1.5:
            lines.append(f"- **Warning:** Custom tokenizer fragments common English words "
                         f"{fertility_ratio:.1f}x more than GPT-2")
        elif fertility_ratio > 1.1:
            lines.append(f"- **Note:** Custom tokenizer is {fertility_ratio:.1f}x more fragmented "
                         f"on general English (moderate trade-off)")
        else:
            lines.append(f"- General English handling is comparable to GPT-2 "
                         f"(ratio: {fertility_ratio:.2f}x)")
        lines.append("")

        # Show some example words with different handling
        diff_words = [pw for pw in ge_custom["per_word"]
                      if pw["subtokens"] != next(
                          (bw["subtokens"] for bw in ge_baseline["per_word"]
                           if bw["word"] == pw["word"]), pw["subtokens"])]
        if diff_words:
            lines.append("Words with different tokenization.")
            lines.append("")
            lines.append("| Word          | Custom tokens | GPT-2 tokens |")
            lines.append("|---------------|---------------|--------------|")
            for dw in diff_words[:15]:
                baseline_count = next(
                    (bw["subtokens"] for bw in ge_baseline["per_word"]
                     if bw["word"] == dw["word"]), "?")
                lines.append(f"| {dw['word']:<13} | {dw['subtokens']:13d} | {baseline_count:>12} |")
            lines.append("")

    # Cross-tokenizer summary table
    if len(all_results) > 1:
        lines.append("## Cross-tokenizer summary")
        lines.append("")
        lines.append("Comparison of all custom tokenizers against GPT-2 baseline.")
        lines.append("")
        lines.append("| Vocab size | Wins | Losses | Ties | Medical fertility | General fertility | Compression |")
        lines.append("|------------|------|--------|------|-------------------|-------------------|-------------|")
        for r in all_results:
            tc = r["term_comparison"]
            mf = r["text_metrics"]["custom"]["fertility"]
            gf = r["general_english"]["custom"]["avg_fertility"]
            cr = r["text_metrics"]["custom"]["compression_ratio"]
            lines.append(f"| {r['custom_vocab_size']:>10,} | {tc['wins']:>4} | {tc['losses']:>6} | {tc['ties']:>4} | {mf:>17.4f} | {gf:>17.4f} | {cr:>11.4f} |")
        lines.append("")

        # GPT-2 baseline row for reference
        br = all_results[0]
        bmf = br["text_metrics"]["baseline"]["fertility"]
        bgf = br["general_english"]["baseline"]["avg_fertility"]
        bcr = br["text_metrics"]["baseline"]["compression_ratio"]
        lines.append(f"**GPT-2 baseline:** fertility={bmf:.4f}, general fertility={bgf:.4f}, compression={bcr:.4f}")
        lines.append("")

    # Honest assessment
    lines.append("## Honest assessment")
    lines.append("")

    # Calculate aggregate stats
    total_wins = sum(r["term_comparison"]["wins"] for r in all_results)
    total_losses = sum(r["term_comparison"]["losses"] for r in all_results)
    total_ties = sum(r["term_comparison"]["ties"] for r in all_results)

    if len(all_results) > 0:
        avg_custom_fertility = sum(r["text_metrics"]["custom"]["fertility"] for r in all_results) / len(all_results)
        baseline_fertility = all_results[0]["text_metrics"]["baseline"]["fertility"]
        avg_custom_ge = sum(r["general_english"]["custom"]["avg_fertility"] for r in all_results) / len(all_results)
        baseline_ge = all_results[0]["general_english"]["baseline"]["avg_fertility"]

        lines.append(f"Across all {len(all_results)} custom tokenizers compared against GPT-2, "
                     f"the custom tokenizers won on {total_wins} term comparisons, lost on "
                     f"{total_losses}, and tied on {total_ties}. ")

        if avg_custom_fertility > baseline_fertility:
            lines.append(f"On medical text, the custom tokenizers have higher fertility "
                         f"({avg_custom_fertility:.4f} vs {baseline_fertility:.4f}), meaning they "
                         f"produce more tokens per word. This is expected given the much smaller "
                         f"vocabulary sizes.")
        else:
            lines.append(f"On medical text, the custom tokenizers achieve lower fertility "
                         f"({avg_custom_fertility:.4f} vs {baseline_fertility:.4f}), producing "
                         f"fewer tokens per word despite smaller vocabularies.")

        if avg_custom_ge > baseline_ge * 1.5:
            lines.append(f" The general-English penalty is significant "
                         f"({avg_custom_ge:.4f} vs {baseline_ge:.4f}), reflecting the trade-off "
                         f"of training on a domain-specific corpus with a smaller vocabulary.")
        elif avg_custom_ge > baseline_ge * 1.1:
            lines.append(f" There is a moderate general-English penalty "
                         f"({avg_custom_ge:.4f} vs {baseline_ge:.4f}).")
        else:
            lines.append(f" General English handling is comparable to GPT-2 "
                         f"({avg_custom_ge:.4f} vs {baseline_ge:.4f}).")

        lines.append("")

        # Note about vocab size convergence if applicable
        vocab_sizes = [r["custom_vocab_size"] for r in all_results]
        if len(set(vocab_sizes)) == 1 and len(all_results) > 1:
            lines.append("**Note on vocab size convergence:** All custom tokenizers converged "
                         f"to the same vocabulary size ({vocab_sizes[0]:,}). This occurs when "
                         f"the training corpus is too small to produce enough BPE merges for the "
                         f"larger target sizes. With a larger corpus (50-100 MB of real ALS "
                         f"literature), the 16K, 32K, and 50K targets would produce distinct "
                         f"vocabularies with meaningfully different trade-offs.")
            lines.append("")

    lines.append("---")
    lines.append(f"*Generated: {metadata['date']}*")
    lines.append("")

    return "\n".join(lines)


def find_tokenizer_files(directory):
    """Find all als_tokenizer_*.json files in a directory."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    files = sorted(dir_path.glob("als_tokenizer_*k.json"))
    if not files:
        print(f"Error: no als_tokenizer_*k.json files found in {directory}", file=sys.stderr)
        sys.exit(1)

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Compare custom tokenizers against a baseline (GPT-2 or another tokenizer)"
    )
    parser.add_argument(
        "--custom",
        required=True,
        help="Path to custom tokenizer JSON, or directory if --all is used",
    )
    parser.add_argument(
        "--baseline",
        default="gpt2",
        help="Path to baseline tokenizer JSON, or 'gpt2' for GPT-2 via tiktoken (default: gpt2)",
    )
    parser.add_argument(
        "--terms",
        default="reports/medical_terms.json",
        help="Path to medical term list JSON (default: reports/medical_terms.json)",
    )
    parser.add_argument(
        "--text",
        default="data/processed/val.txt",
        help="Path to text file for text-level metrics (default: data/processed/val.txt)",
    )
    parser.add_argument(
        "--output-json",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--output-md",
        help="Output path for Markdown report",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all tokenizers in the --custom directory against baseline",
    )

    args = parser.parse_args()

    # Load baseline
    print(f"Loading baseline tokenizer: {args.baseline}")
    baseline = load_tokenizer(args.baseline)
    print(f"  Vocab size: {baseline.get_vocab_size():,}")

    # Load terms
    print(f"Loading medical terms from: {args.terms}")
    with open(args.terms) as f:
        terms = json.load(f)
    print(f"  Terms loaded: {len(terms)}")

    # Load text
    print(f"Loading text from: {args.text}")
    with open(args.text, encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    print(f"  Text loaded: {len(text):,} chars, {len(words):,} words")

    # Determine which custom tokenizers to compare
    if args.all:
        tokenizer_files = find_tokenizer_files(args.custom)
        print(f"\nFound {len(tokenizer_files)} tokenizers to compare")
    else:
        tokenizer_files = [Path(args.custom)]

    # Run comparisons
    all_results = []
    for tok_path in tokenizer_files:
        print(f"\nComparing: {tok_path.name} vs {args.baseline}")
        custom = load_tokenizer(str(tok_path))
        print(f"  Custom vocab size: {custom.get_vocab_size():,}")

        result = compare_single(custom, baseline, terms, text)
        all_results.append(result)

        tc = result["term_comparison"]
        print(f"  Term results: {tc['wins']} wins, {tc['losses']} losses, {tc['ties']} ties")
        print(f"  Medical fertility: {result['text_metrics']['custom']['fertility']:.4f} "
              f"(GPT-2: {result['text_metrics']['baseline']['fertility']:.4f})")
        print(f"  General-English fertility: {result['general_english']['custom']['avg_fertility']:.4f} "
              f"(GPT-2: {result['general_english']['baseline']['avg_fertility']:.4f})")

    # Build metadata
    metadata = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "baseline_name": args.baseline,
        "baseline_vocab_size": baseline.get_vocab_size(),
        "term_count": len(terms),
        "text_file": args.text,
        "text_chars": len(text),
        "text_words": len(words),
        "tokenizers_compared": len(all_results),
    }

    # Build output structure
    output = {
        "metadata": metadata,
        "comparisons": {},
    }
    for result in all_results:
        # Exclude verbose all_results from JSON for readability
        clean_result = {k: v for k, v in result.items()}
        clean_result["term_comparison"] = {
            k: v for k, v in result["term_comparison"].items()
            if k != "all_results"
        }
        # Add a summary of all results (just outcome and counts, not token lists)
        clean_result["term_comparison"]["per_term"] = [
            {
                "term": r["term"],
                "category": r["category"],
                "custom_subtokens": r["custom_subtokens"],
                "baseline_subtokens": r["baseline_subtokens"],
                "outcome": r["outcome"],
            }
            for r in result["term_comparison"]["all_results"]
        ]
        output["comparisons"][result["custom_name"]] = clean_result

    # Write JSON
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON report written to: {args.output_json}")

    # Write Markdown
    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
        md = generate_markdown_report(all_results, metadata)
        with open(args.output_md, "w") as f:
            f.write(md)
        print(f"Markdown report written to: {args.output_md}")

    # Print summary
    print("\n=== Comparison Complete ===")
    for result in all_results:
        tc = result["term_comparison"]
        print(f"  {result['custom_name']}: W={tc['wins']} L={tc['losses']} T={tc['ties']} "
              f"| fertility={result['text_metrics']['custom']['fertility']:.4f} "
              f"| compression={result['text_metrics']['custom']['compression_ratio']:.4f}")


if __name__ == "__main__":
    main()
