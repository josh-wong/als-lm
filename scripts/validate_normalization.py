"""Validate text normalization on actual corpus documents.

Runs 100+ documents from data/raw/ through the full cleaning pipeline
and verifies that all 12 ABBREVIATION_MAP canonical medical terms survive
normalization. This is a regression test for NORM-03.

Usage:
    python scripts/validate_normalization.py
    python scripts/validate_normalization.py --raw-dir data/raw --min-docs 100
    python scripts/validate_normalization.py --verbose
"""

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root setup so data.processing.clean is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.processing.clean import (  # noqa: E402
    ABBREVIATION_MAP,
    clean_document,
    load_raw_documents,
)

# Extract the 12 canonical terms from ABBREVIATION_MAP
CANONICAL_TERMS = [entry["canonical"] for entry in ABBREVIATION_MAP]


def find_raw_terms(raw_text: str, abbreviation_map: list[dict]) -> set[str]:
    """Identify which canonical terms have at least one variant in *raw_text*.

    For each entry in the abbreviation map, checks whether ANY variant
    pattern matches in the raw text using word-boundary matching.  Returns
    the set of canonical terms that were found.

    Args:
        raw_text: The original, uncleaned document text.
        abbreviation_map: The ABBREVIATION_MAP list from clean.py.

    Returns:
        Set of canonical term strings present in the raw text.
    """
    found = set()
    for entry in abbreviation_map:
        canonical = entry["canonical"]
        for variant in entry["variants"]:
            # Variants are regex patterns (e.g. "C9orf72|C9ORF72"), not
            # literals, so they are NOT re.escape()'d — unlike canonical
            # terms in check_term_survival() below.
            pattern = re.compile(r"\b" + variant + r"\b", re.IGNORECASE)
            if pattern.search(raw_text):
                found.add(canonical)
                break  # one match is enough for this canonical term
    return found


def check_term_survival(
    cleaned_text: str,
    terms: list[str],
) -> dict[str, bool]:
    """Check which canonical terms appear in *cleaned_text*.

    Uses word-boundary matching to avoid false positives such as "ALS"
    matching inside "ALSO".

    Args:
        cleaned_text: The text after the full cleaning pipeline.
        terms: List of canonical term strings to look for.

    Returns:
        Dict mapping each term to True (present) or False (absent).
    """
    results = {}
    for term in terms:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b")
        results[term] = bool(pattern.search(cleaned_text))
    return results


def validate_documents(
    raw_dir: Path,
    min_docs: int,
    verbose: bool,
) -> int:
    """Run normalization validation on all documents in *raw_dir*.

    Loads raw documents, runs each through ``clean_document()``, and checks
    that every canonical medical term found in the raw text also appears in
    the cleaned output.

    Args:
        raw_dir: Path to the raw data directory (e.g. ``data/raw``).
        min_docs: Minimum number of documents required for the run.
        verbose: If True, print per-document details.

    Returns:
        Exit code: 0 if all terms survived, 1 if any term was lost,
        2 if fewer than *min_docs* documents were available.
    """
    documents = load_raw_documents(raw_dir)

    if len(documents) < min_docs:
        print(
            f"WARNING: Found only {len(documents)} documents in {raw_dir}, "
            f"need at least {min_docs}."
        )
        return 2

    total_processed = 0
    total_skipped = 0
    total_passed = 0
    total_failed = 0
    term_checked_count: dict[str, int] = {t: 0 for t in CANONICAL_TERMS}
    term_survived_count: dict[str, int] = {t: 0 for t in CANONICAL_TERMS}
    failures: list[dict] = []

    for doc in documents:
        doc_id = doc.get("id", "unknown")
        raw_text = doc.get("text", "")

        if not raw_text or not raw_text.strip():
            total_skipped += 1
            continue

        # Find which canonical terms exist in the raw text
        raw_terms = find_raw_terms(raw_text, ABBREVIATION_MAP)

        # Run through the full cleaning pipeline
        cleaned = clean_document(doc)

        if cleaned is None:
            total_skipped += 1
            if verbose:
                print(f"  SKIP  {doc_id} (rejected by pipeline)")
            continue

        total_processed += 1
        cleaned_text = cleaned.get("text", "")

        # Check which terms survived
        survival = check_term_survival(cleaned_text, list(raw_terms))

        lost_terms = [t for t, alive in survival.items() if not alive]

        # Update per-term counts
        for term in raw_terms:
            term_checked_count[term] += 1
            if survival.get(term, False):
                term_survived_count[term] += 1

        if lost_terms:
            total_failed += 1
            failures.append({"doc_id": doc_id, "lost_terms": lost_terms})
            if verbose:
                print(f"  FAIL  {doc_id}  lost: {', '.join(lost_terms)}")
        else:
            total_passed += 1
            if verbose:
                found_str = ", ".join(sorted(raw_terms)) if raw_terms else "(none)"
                print(f"  PASS  {doc_id}  terms present: {found_str}")

    # Print aggregate summary
    print()
    print("=" * 60)
    print("  Normalization Validation Summary")
    print("=" * 60)
    print(f"  Documents loaded:    {len(documents)}")
    print(f"  Documents processed: {total_processed}")
    print(f"  Documents skipped:   {total_skipped}")
    print(f"  Documents passed:    {total_passed}")
    print(f"  Documents failed:    {total_failed}")
    print()

    # Per-term survival rates
    print("  Per-term survival rates:")
    print(f"  {'Term':<12}  {'Checked':>8}  {'Survived':>9}  {'Rate':>7}")
    print(f"  {'-' * 12}  {'-' * 8}  {'-' * 9}  {'-' * 7}")
    for term in CANONICAL_TERMS:
        checked = term_checked_count[term]
        survived = term_survived_count[term]
        rate = f"{100 * survived / checked:.1f}%" if checked > 0 else "N/A"
        print(f"  {term:<12}  {checked:>8}  {survived:>9}  {rate:>7}")

    print()

    if failures:
        print(f"  RESULT: FAIL ({total_failed} document(s) lost terms)")
        if not verbose:
            print("  Run with --verbose to see per-document details.")
        return 1

    print("  RESULT: PASS (all terms survived in all documents)")
    return 0


def main():
    """Parse arguments and run normalization validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate that text normalization preserves all 12 "
            "ABBREVIATION_MAP canonical medical terms on real corpus "
            "documents."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=100,
        help="Minimum number of documents required (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-document pass/fail details",
    )
    args = parser.parse_args()

    exit_code = validate_documents(args.raw_dir, args.min_docs, args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
