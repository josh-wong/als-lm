"""Validate tokenizer against a medical term list with fragmentation analysis.

Standalone CLI tool that encodes each medical term, counts subtokens, and
flags terms fragmenting into 3 or more subtokens. Produces both JSON and
Markdown reports with per-term breakdowns, category summaries, and cross-
tokenizer comparison tables.

Usage:
    python scripts/validate_tokenizer.py --tokenizer tokenizer/als_tokenizer_32k.json
    python scripts/validate_tokenizer.py --tokenizer-dir tokenizer/ --all
    python scripts/validate_tokenizer.py --tokenizer tokenizer/als_tokenizer_32k.json --threshold 4
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer


def validate_medical_terms(
    tokenizer_path: str,
    terms: list[dict],
    threshold: int = 3,
) -> dict:
    """Validate a tokenizer against a list of medical terms.

    For each term, encodes it and records the subtoken count and actual
    subtoken text. Terms with subtokens >= threshold are flagged.

    Args:
        tokenizer_path: Path to the tokenizer JSON file.
        terms: List of term dicts with 'term' and 'category' keys.
        threshold: Flag terms with this many or more subtokens.

    Returns:
        Dictionary with per-term results, category breakdown, and
        overall statistics.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    tokenizer_name = Path(tokenizer_path).stem

    results = []
    flagged = []

    for entry in terms:
        term = entry["term"]
        category = entry.get("category", "unknown")
        encoded = tokenizer.encode(term)
        tokens = encoded.tokens
        n_tokens = len(tokens)
        is_flagged = n_tokens >= threshold

        result = {
            "term": term,
            "category": category,
            "n_subtokens": n_tokens,
            "subtokens": tokens,
            "flagged": is_flagged,
        }
        results.append(result)

        if is_flagged:
            flagged.append(result)

    # Category breakdown
    category_stats = {}
    for r in results:
        cat = r["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "flagged": 0}
        category_stats[cat]["total"] += 1
        if r["flagged"]:
            category_stats[cat]["flagged"] += 1

    for cat in category_stats:
        total = category_stats[cat]["total"]
        flag = category_stats[cat]["flagged"]
        category_stats[cat]["flagged_pct"] = (
            round(100 * flag / total, 1) if total > 0 else 0.0
        )

    return {
        "tokenizer_name": tokenizer_name,
        "tokenizer_path": tokenizer_path,
        "vocab_size": vocab_size,
        "threshold": threshold,
        "total_terms": len(results),
        "flagged_count": len(flagged),
        "flagged_pct": (
            round(100 * len(flagged) / len(results), 1)
            if results else 0.0
        ),
        "category_breakdown": category_stats,
        "results": results,
    }


def generate_markdown_report(
    all_results: list[dict],
    terms: list[dict],
    threshold: int,
) -> str:
    """Generate a Markdown validation report across all tokenizers.

    Includes per-tokenizer sections with category breakdowns and full
    term tables, plus a cross-tokenizer comparison summary.
    """
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("# Tokenizer validation report")
    lines.append("")
    lines.append(
        f"This report shows fragmentation analysis for {len(all_results)} "
        f"tokenizer(s) against {len(terms)} medical terms. Terms with "
        f"{threshold}+ subtokens are flagged."
    )
    lines.append("")
    lines.append(f"- **Date:** {now}")
    lines.append(f"- **Tokenizers evaluated:** {len(all_results)}")
    lines.append(f"- **Total terms:** {len(terms)}")
    lines.append(f"- **Flagging threshold:** {threshold}+ subtokens")
    lines.append("")

    # Per-tokenizer sections
    for result in all_results:
        name = result["tokenizer_name"]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(
            f"Vocabulary size: {result['vocab_size']:,}. "
            f"Flagged {result['flagged_count']} of "
            f"{result['total_terms']} terms "
            f"({result['flagged_pct']}%)."
        )
        lines.append("")

        # Category breakdown table
        lines.append("### Category breakdown")
        lines.append("")
        lines.append(
            "Per-category flagging rates show which term types are most "
            "affected by fragmentation."
        )
        lines.append("")
        lines.append(
            f"| {'Category':<20} | {'Total':>5} | {'Flagged':>7} | "
            f"{'Flagged %':>9} |"
        )
        lines.append(
            f"|{'-' * 22}|{'-' * 7}|{'-' * 9}|{'-' * 11}|"
        )

        cats = result["category_breakdown"]
        for cat in sorted(cats.keys()):
            s = cats[cat]
            lines.append(
                f"| {cat:<20} | {s['total']:>5} | {s['flagged']:>7} | "
                f"{s['flagged_pct']:>8.1f}% |"
            )
        lines.append("")

        # Full term table
        lines.append("### Term details")
        lines.append("")
        lines.append(
            "Every term with its subtoken count and breakdown. Flagged "
            f"terms ({threshold}+ subtokens) are marked."
        )
        lines.append("")
        lines.append(
            f"| {'Term':<35} | {'Category':<15} | {'Subtokens':>9} | "
            f"{'Breakdown':<50} | {'Flag':>4} |"
        )
        lines.append(
            f"|{'-' * 37}|{'-' * 17}|{'-' * 11}|"
            f"{'-' * 52}|{'-' * 6}|"
        )

        # Sort by subtoken count (descending) for easy scanning
        sorted_results = sorted(
            result["results"], key=lambda x: -x["n_subtokens"]
        )
        for r in sorted_results:
            breakdown = " + ".join(
                f"`{t}`" for t in r["subtokens"]
            )
            flag = "!!" if r["flagged"] else ""
            lines.append(
                f"| {r['term']:<35} | {r['category']:<15} | "
                f"{r['n_subtokens']:>9} | {breakdown:<50} | "
                f"{flag:>4} |"
            )
        lines.append("")

    # Cross-tokenizer comparison
    if len(all_results) > 1:
        lines.append("## Comparison summary")
        lines.append("")
        lines.append(
            "Side-by-side comparison of flagging rates across all tokenizers."
        )
        lines.append("")
        lines.append(
            f"| {'Tokenizer':<30} | {'Vocab Size':>10} | "
            f"{'Flagged':>7} | {'Flagged %':>9} |"
        )
        lines.append(
            f"|{'-' * 32}|{'-' * 12}|{'-' * 9}|{'-' * 11}|"
        )
        for r in all_results:
            lines.append(
                f"| {r['tokenizer_name']:<30} | "
                f"{r['vocab_size']:>10,} | {r['flagged_count']:>7} | "
                f"{r['flagged_pct']:>8.1f}% |"
            )
        lines.append("")

        # Worst fragmented terms across all tokenizers
        lines.append("## Most fragmented terms")
        lines.append("")
        lines.append(
            "Terms sorted by worst fragmentation across all tokenizers, "
            "showing which medical vocabulary is hardest for BPE to learn."
        )
        lines.append("")

        # Collect max fragmentation per term
        term_max = {}
        for r in all_results:
            for t in r["results"]:
                term = t["term"]
                if (
                    term not in term_max
                    or t["n_subtokens"] > term_max[term]["max_subtokens"]
                ):
                    term_max[term] = {
                        "term": term,
                        "category": t["category"],
                        "max_subtokens": t["n_subtokens"],
                        "worst_tokenizer": r["tokenizer_name"],
                    }

        worst = sorted(
            term_max.values(), key=lambda x: -x["max_subtokens"]
        )[:20]

        lines.append(
            f"| {'Term':<35} | {'Category':<15} | "
            f"{'Max Subtokens':>13} | {'Worst Tokenizer':<25} |"
        )
        lines.append(
            f"|{'-' * 37}|{'-' * 17}|{'-' * 15}|{'-' * 27}|"
        )
        for w in worst:
            lines.append(
                f"| {w['term']:<35} | {w['category']:<15} | "
                f"{w['max_subtokens']:>13} | "
                f"{w['worst_tokenizer']:<25} |"
            )
        lines.append("")

    lines.append(
        "*Report generated by scripts/validate_tokenizer.py*"
    )
    lines.append("")

    return "\n".join(lines)


def find_tokenizers(tokenizer_dir: str) -> list[str]:
    """Find all tokenizer JSON files in a directory.

    Filters for files matching the als_tokenizer_*.json pattern.
    """
    dir_path = Path(tokenizer_dir)
    if not dir_path.is_dir():
        return []

    tokenizer_files = sorted(
        str(p)
        for p in dir_path.glob("als_tokenizer_*.json")
        if p.is_file()
    )
    return tokenizer_files


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate tokenizer against medical terms with "
            "fragmentation analysis"
        )
    )
    parser.add_argument(
        "--tokenizer",
        help="Path to a single tokenizer JSON file",
    )
    parser.add_argument(
        "--tokenizer-dir",
        help="Directory containing tokenizer JSON files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all tokenizers in --tokenizer-dir",
    )
    parser.add_argument(
        "--terms",
        default="reports/medical_terms.json",
        help="Path to medical terms JSON (default: reports/medical_terms.json)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Fragmentation threshold (default: 3)",
    )
    parser.add_argument(
        "--output-json",
        default="reports/validation_report.json",
        help="Output path for JSON report (default: reports/validation_report.json)",
    )
    parser.add_argument(
        "--output-md",
        default="reports/validation_report.md",
        help="Output path for Markdown report (default: reports/validation_report.md)",
    )
    args = parser.parse_args()

    # Determine which tokenizers to validate
    tokenizer_paths = []
    if args.all and args.tokenizer_dir:
        tokenizer_paths = find_tokenizers(args.tokenizer_dir)
        if not tokenizer_paths:
            print(
                f"Error: no tokenizer files found in {args.tokenizer_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
    elif args.tokenizer:
        tokenizer_paths = [args.tokenizer]
    else:
        parser.error(
            "Provide either --tokenizer PATH or --tokenizer-dir DIR --all"
        )

    # Load terms
    terms_path = Path(args.terms)
    if not terms_path.exists():
        print(
            f"Error: terms file not found: {terms_path}", file=sys.stderr
        )
        sys.exit(1)

    with open(terms_path, "r", encoding="utf-8") as f:
        terms = json.load(f)

    print(f"Loaded {len(terms)} medical terms from {terms_path}")
    print(f"Validating {len(tokenizer_paths)} tokenizer(s)...")
    print(f"Flagging threshold: {args.threshold}+ subtokens")

    # Validate each tokenizer
    all_results = []
    for tok_path in tokenizer_paths:
        print(f"\n  Validating: {tok_path}")
        result = validate_medical_terms(tok_path, terms, args.threshold)
        all_results.append(result)
        print(
            f"    Vocab: {result['vocab_size']:,}, "
            f"Flagged: {result['flagged_count']}/{result['total_terms']} "
            f"({result['flagged_pct']}%)"
        )

    # Save JSON report
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    json_report = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "terms_file": str(terms_path),
            "threshold": args.threshold,
            "tokenizer_count": len(all_results),
        },
        "tokenizers": {
            r["tokenizer_name"]: r for r in all_results
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    print(f"\nJSON report saved to: {json_path}")

    # Generate and save Markdown report
    md_report = generate_markdown_report(all_results, terms, args.threshold)
    md_path = Path(args.output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"Markdown report saved to: {md_path}")

    # Print summary
    if len(all_results) > 1:
        print("\nComparison summary:")
        print(f"  {'Tokenizer':<30}  {'Vocab':>8}  {'Flagged':>7}")
        print("  " + "-" * 50)
        for r in all_results:
            print(
                f"  {r['tokenizer_name']:<30}  "
                f"{r['vocab_size']:>8,}  "
                f"{r['flagged_count']:>3}/{r['total_terms']} "
                f"({r['flagged_pct']}%)"
            )


if __name__ == "__main__":
    main()
