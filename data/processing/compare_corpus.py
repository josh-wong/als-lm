"""Corpus comparison report generator for ALS-LM.

Snapshots a baseline version's statistics, parses both baseline and current
stats.md files, and generates a side-by-side comparison report. Designed
to be run after a processing pipeline re-run to quantify corpus changes.

Usage:
    python -m data.processing.compare_corpus --baseline-version v1.0.0 --current-version v1.2.0
    python -m data.processing.compare_corpus --snapshot-only --baseline-version v1.0.0

Functions:
    snapshot_baseline: Save current stats and file sizes as a versioned baseline.
    parse_stats: Extract structured metrics from a stats.md Markdown file.
    generate_comparison_report: Produce the side-by-side comparison report.
"""

import argparse
import json
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline snapshotting
# ---------------------------------------------------------------------------

def snapshot_baseline(
    stats_path: Path,
    train_path: Path,
    val_path: Path,
    raw_dir: Path,
    snapshot_dir: Path,
) -> bool:
    """Save current corpus stats as a versioned baseline.

    Creates the snapshot directory and copies stats.md into it, then
    records train.txt and val.txt file sizes and raw document counts
    per source in a JSON sidecar. This operation is idempotent: if the
    snapshot directory already exists, it logs a message and returns
    without overwriting.

    Args:
        stats_path: Path to current data/stats.md.
        train_path: Path to current data/processed/train.txt.
        val_path: Path to current data/processed/val.txt.
        raw_dir: Path to data/raw/ for raw doc count snapshotting.
        snapshot_dir: Directory to store the baseline snapshot.

    Returns:
        True if a snapshot was created, False if it already existed.

    Raises:
        FileNotFoundError: If stats_path does not exist.
    """
    if snapshot_dir.exists():
        logger.info("Baseline snapshot already exists at %s — skipping", snapshot_dir)
        return False

    if not stats_path.exists():
        raise FileNotFoundError(f"Cannot snapshot: {stats_path} does not exist")

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Copy stats.md
    shutil.copy2(stats_path, snapshot_dir / "stats.md")
    logger.info("Snapshotted %s -> %s", stats_path, snapshot_dir / "stats.md")

    # Record file sizes and raw document counts
    sizes = {
        "train_bytes": train_path.stat().st_size if train_path.exists() else 0,
        "val_bytes": val_path.stat().st_size if val_path.exists() else 0,
        "raw_counts": _aggregate_raw_by_category(raw_dir),
    }
    sizes_path = snapshot_dir / "file_sizes.json"
    with open(sizes_path, "w", encoding="utf-8") as f:
        json.dump(sizes, f, indent=2)
    logger.info("Recorded file sizes and raw counts to %s", sizes_path)

    return True


# ---------------------------------------------------------------------------
# Stats.md parsing
# ---------------------------------------------------------------------------

def _parse_number(value: str) -> int:
    """Parse a formatted number string, stripping commas and tildes.

    Args:
        value: A string like "35,088" or "~176,547,868".

    Returns:
        Integer value.
    """
    cleaned = value.strip().lstrip("~").replace(",", "")
    return int(cleaned)


def _parse_size_mb(value: str) -> float:
    """Parse a human-readable size string into megabytes.

    Args:
        value: A string like "678.05 MB" or "1.23 GB".

    Returns:
        Size in megabytes as a float.
    """
    parts = value.strip().split()
    if len(parts) != 2:
        return 0.0

    number = float(parts[0])
    unit = parts[1].upper()

    if unit == "B":
        return number / (1024 * 1024)
    elif unit == "KB":
        return number / 1024
    elif unit == "MB":
        return number
    elif unit == "GB":
        return number * 1024
    return 0.0


def _extract_section(text: str, header: str) -> str:
    """Extract content under a Markdown ## section header.

    Args:
        text: Full Markdown text.
        header: Section header text (without ##).

    Returns:
        Text content from section start to next ## or end of file.
    """
    pattern = rf"## {re.escape(header)}\s*\n"
    match = re.search(pattern, text)
    if not match:
        return ""

    remaining = text[match.end():]
    next_section = re.search(r"\n## ", remaining)
    return remaining[:next_section.start()] if next_section else remaining


def _parse_table_rows(
    section_text: str,
    first_table_only: bool = False,
) -> list[dict[str, str]]:
    """Parse Markdown table rows from a section into dicts.

    Skips the header row and separator line. Returns a list of dicts
    mapping column header names to cell values.

    Args:
        section_text: Text containing one or more Markdown tables.
        first_table_only: If True, stop parsing after the first
            complete table (useful when a section has multiple tables).

    Returns:
        List of row dicts with stripped cell values.
    """
    rows = []
    headers: list[str] = []
    header_found = False
    in_table = False

    for line in section_text.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            # If we were in a table and hit a non-table line, the
            # first table is complete
            if in_table and first_table_only:
                break
            continue
        # Skip separator rows
        if re.match(r"^\|[\s\-|]+\|$", stripped):
            in_table = True
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]

        if not header_found:
            headers = cells
            header_found = True
            in_table = True
        else:
            row = dict(zip(headers, cells))
            rows.append(row)

    return rows


def parse_stats(stats_path: Path) -> dict:
    """Parse a stats.md file into a structured dictionary.

    Extracts metrics from the four key sections: Corpus summary,
    Source distribution, Rejection summary, and Document length
    analysis.

    Args:
        stats_path: Path to the stats.md Markdown file.

    Returns:
        Dictionary with keys: corpus_summary, source_distribution,
        rejection_summary, document_length.
    """
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    text = stats_path.read_text(encoding="utf-8")
    result: dict = {}

    # --- Corpus summary ---
    try:
        section = _extract_section(text, "Corpus summary")
        rows = _parse_table_rows(section)
        summary = {}
        for row in rows:
            metric = row.get("Metric", "").strip()
            value = row.get("Value", "").strip()
            summary[metric] = value
        result["corpus_summary"] = summary
    except Exception as e:
        logger.warning("Failed to parse Corpus summary: %s", e)
        result["corpus_summary"] = {}

    # --- Source distribution ---
    try:
        section = _extract_section(text, "Source distribution")
        rows = _parse_table_rows(section)
        sources = {}
        for row in rows:
            cat = row.get("Source category", "").strip()
            if not cat:
                continue
            sources[cat] = {
                "documents": _parse_number(row.get("Documents", "0")),
                "word_count": _parse_number(row.get("Word count", "0")),
                "percentage": row.get("Percentage", "0%").strip(),
            }
        result["source_distribution"] = sources
    except Exception as e:
        logger.warning("Failed to parse Source distribution: %s", e)
        result["source_distribution"] = {}

    # --- Rejection summary ---
    try:
        section = _extract_section(text, "Rejection summary")
        rows = _parse_table_rows(section)
        rejections = {}
        for row in rows:
            reason = row.get("Reason", "").strip()
            count_str = row.get("Count", "0").strip()
            if reason:
                rejections[reason] = _parse_number(count_str)
        result["rejection_summary"] = rejections
    except Exception as e:
        logger.warning("Failed to parse Rejection summary: %s", e)
        result["rejection_summary"] = {}

    # --- Document length analysis ---
    try:
        section = _extract_section(text, "Document length analysis")
        rows = _parse_table_rows(section, first_table_only=True)
        lengths = {}
        for row in rows:
            metric = row.get("Metric", "").strip()
            words_str = row.get("Words", "0").strip()
            if metric:
                lengths[metric] = _parse_number(words_str)
        result["document_length"] = lengths
    except Exception as e:
        logger.warning("Failed to parse Document length analysis: %s", e)
        result["document_length"] = {}

    return result


# ---------------------------------------------------------------------------
# Comparison report generation
# ---------------------------------------------------------------------------

def _fmt_change(old: int | float, new: int | float, is_pct: bool = False) -> str:
    """Format a change value as a human-readable delta string.

    Args:
        old: Baseline value.
        new: Current value.
        is_pct: If True, format as percentage point change.

    Returns:
        String like "+1,234 (+5.6%)" or "-0.3 pp".
    """
    if old == 0 and new == 0:
        return "—"
    if old == 0:
        return "NEW"

    delta = new - old
    if is_pct:
        return f"{delta:+.1f} pp"

    pct = 100 * delta / old if old != 0 else 0
    if isinstance(delta, float):
        return f"{delta:+,.1f} ({pct:+.1f}%)"
    return f"{delta:+,} ({pct:+.1f}%)"


def _get_size_advisory(total_mb: float) -> str:
    """Return advisory label for corpus total size.

    Args:
        total_mb: Corpus total size in megabytes.

    Returns:
        Advisory label string.
    """
    if total_mb >= 80:
        return "Target met"
    elif total_mb > 32:
        return "Minimum met"
    else:
        return "Below v1.0.0"


def _fmt_bytes_mb(size_bytes: int) -> str:
    """Format byte count as megabytes with 2 decimal places.

    Args:
        size_bytes: Size in bytes.

    Returns:
        String like "678.05 MB".
    """
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def count_raw_docs_per_source(raw_dir: Path) -> dict[str, int]:
    """Count document JSON files per source directory in data/raw/.

    Excludes checkpoint files (.checkpoint.json) which are scraper
    state files, not documents.

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        Dict mapping source directory name to document JSON file count.
    """
    counts: dict[str, int] = {}
    if not raw_dir.exists():
        logger.warning("Raw directory not found: %s", raw_dir)
        return counts

    for source_dir in sorted(raw_dir.iterdir()):
        if source_dir.is_dir():
            json_files = [
                f for f in source_dir.glob("*.json")
                if f.name != ".checkpoint.json"
            ]
            counts[source_dir.name] = len(json_files)

    return counts


# Mapping from raw directory names to source categories
RAW_DIR_TO_CATEGORY = {
    "pubmed": "biomedical_research",
    "pubmed_abstracts": "biomedical_research",
    "bookshelf": "educational",
    "cdc": "educational",
    "educational": "educational",
    "clinicaltrials": "clinical_trials",
    "fda": "regulatory",
    "patient_narratives": "patient_narratives",
    "wikipedia": "supplementary_science",
    "biorxiv": "biomedical_research",
    "medrxiv": "biomedical_research",
    "europepmc": "biomedical_research",
    "semanticscholar": "biomedical_research",
    "dailymed": "regulatory",
    "omim": "biomedical_research",
    "ema_epar": "regulatory",
}


def _aggregate_raw_by_category(raw_dir: Path) -> dict[str, int]:
    """Aggregate raw JSON file counts by source_category.

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        Dict mapping source_category to total raw JSON file count.
    """
    per_dir = count_raw_docs_per_source(raw_dir)
    by_category: dict[str, int] = {}

    for dirname, count in per_dir.items():
        category = RAW_DIR_TO_CATEGORY.get(dirname, "unknown")
        by_category[category] = by_category.get(category, 0) + count

    return by_category


def generate_comparison_report(
    baseline_dir: Path,
    stats_path: Path,
    raw_dir: Path,
    train_path: Path,
    val_path: Path,
    output_path: Path,
    baseline_version: str = "v1.0.0",
    current_version: str = "v1.2.0",
) -> None:
    """Generate a corpus comparison report between two versions.

    Parses both baseline and current statistics, computes deltas, and
    writes a Markdown comparison report with tables for corpus size,
    source distribution, deduplication rates, document length, file
    sizes, and per-source regression warnings.

    Args:
        baseline_dir: Path to baseline snapshot directory.
        stats_path: Path to current data/stats.md.
        raw_dir: Path to data/raw/ for raw doc count scanning.
        train_path: Path to current data/processed/train.txt.
        val_path: Path to current data/processed/val.txt.
        output_path: Path to write the comparison report.
        baseline_version: Label for the baseline version (e.g. "v1.0.0").
        current_version: Label for the current version (e.g. "v1.2.0").

    Raises:
        FileNotFoundError: If baseline or current stats files are missing.
    """
    baseline_stats_path = baseline_dir / "stats.md"
    baseline_sizes_path = baseline_dir / "file_sizes.json"

    if not baseline_stats_path.exists():
        raise FileNotFoundError(f"Baseline stats not found: {baseline_stats_path}")

    if not stats_path.exists():
        raise FileNotFoundError(f"Current stats not found: {stats_path}")

    logger.info("Parsing baseline stats from %s", baseline_stats_path)
    baseline = parse_stats(baseline_stats_path)

    logger.info("Parsing current stats from %s", stats_path)
    current = parse_stats(stats_path)

    # Load baseline file sizes
    baseline_sizes = {"train_bytes": 0, "val_bytes": 0}
    if baseline_sizes_path.exists():
        with open(baseline_sizes_path, "r", encoding="utf-8") as f:
            baseline_sizes = json.load(f)

    # Current file sizes
    current_train_bytes = train_path.stat().st_size if train_path.exists() else 0
    current_val_bytes = val_path.stat().st_size if val_path.exists() else 0

    sections: list[str] = []

    bv = baseline_version
    cv = current_version

    # --- Title ---
    sections.append(f"# Corpus comparison: {bv} vs {cv}")
    sections.append("")
    sections.append(
        f"Side-by-side comparison of corpus metrics between {bv} and {cv}."
    )

    # --- Corpus size table ---
    sections.append("")
    sections.append("## Corpus size")
    sections.append("")
    sections.append(
        f"High-level corpus metrics comparing the {bv} baseline to "
        f"the current {cv} pipeline output."
    )
    sections.append("")

    b_summary = baseline.get("corpus_summary", {})
    c_summary = current.get("corpus_summary", {})

    metrics_rows: list[tuple[str, str, str, str, str]] = []

    metric_keys = [
        ("Total documents", False),
        ("Total words", False),
        ("Total size", True),
        ("Estimated tokens", False),
        ("Training documents", False),
        ("Validation documents", False),
    ]

    current_total_mb = 0.0

    for metric_name, is_size in metric_keys:
        b_val_str = b_summary.get(metric_name, "N/A")
        c_val_str = c_summary.get(metric_name, "N/A")

        if is_size:
            b_mb = _parse_size_mb(b_val_str)
            c_mb = _parse_size_mb(c_val_str)
            current_total_mb = c_mb
            change = _fmt_change(b_mb, c_mb)
            advisory = _get_size_advisory(c_mb)
        else:
            try:
                b_num = _parse_number(b_val_str)
                c_num = _parse_number(c_val_str)
                change = _fmt_change(b_num, c_num)
            except (ValueError, TypeError):
                change = "N/A"
            advisory = ""

        metrics_rows.append((metric_name, b_val_str, c_val_str, change, advisory))

    # Add train.txt and val.txt sizes
    b_train_mb_str = _fmt_bytes_mb(baseline_sizes.get("train_bytes", 0))
    c_train_mb_str = _fmt_bytes_mb(current_train_bytes)
    b_val_mb_str = _fmt_bytes_mb(baseline_sizes.get("val_bytes", 0))
    c_val_mb_str = _fmt_bytes_mb(current_val_bytes)

    b_train_mb = baseline_sizes.get("train_bytes", 0) / (1024 * 1024)
    c_train_mb = current_train_bytes / (1024 * 1024)
    b_val_mb = baseline_sizes.get("val_bytes", 0) / (1024 * 1024)
    c_val_mb = current_val_bytes / (1024 * 1024)

    metrics_rows.append(
        ("train.txt size", b_train_mb_str, c_train_mb_str,
         _fmt_change(b_train_mb, c_train_mb), "")
    )
    metrics_rows.append(
        ("val.txt size", b_val_mb_str, c_val_mb_str,
         _fmt_change(b_val_mb, c_val_mb), "")
    )

    # Build table
    header = f"| Metric               | {bv:<14} | {cv:<14} | Change              | Advisory    |"
    sep =     "|----------------------|----------------|----------------|---------------------|-------------|"
    sections.append(header)
    sections.append(sep)

    for name, b_val, c_val, change, adv in metrics_rows:
        sections.append(
            f"| {name:<20} | {b_val:<14} | {c_val:<14} | {change:<19} | {adv:<11} |"
        )

    # --- Source distribution table ---
    sections.append("")
    sections.append("## Source distribution")
    sections.append("")
    sections.append(
        "Per-category document and word counts, showing changes between versions."
    )
    sections.append("")

    b_sources = baseline.get("source_distribution", {})
    c_sources = current.get("source_distribution", {})
    all_categories = sorted(set(list(b_sources.keys()) + list(c_sources.keys())))

    header = f"| Source category       | {bv} docs   | {cv} docs   | Change       | {bv} words    | {cv} words    | Change       |"
    sep =     "|-----------------------|-------------|-------------|--------------|---------------|---------------|--------------|"
    sections.append(header)
    sections.append(sep)

    for cat in all_categories:
        b_data = b_sources.get(cat, {})
        c_data = c_sources.get(cat, {})

        b_docs = b_data.get("documents", 0) if b_data else 0
        c_docs = c_data.get("documents", 0) if c_data else 0
        b_words = b_data.get("word_count", 0) if b_data else 0
        c_words = c_data.get("word_count", 0) if c_data else 0

        if not b_data:
            doc_change = "NEW"
            word_change = "NEW"
            b_docs_str = "--"
            b_words_str = "--"
        else:
            doc_change = _fmt_change(b_docs, c_docs)
            word_change = _fmt_change(b_words, c_words)
            b_docs_str = f"{b_docs:,}"
            b_words_str = f"{b_words:,}"

        if not c_data:
            c_docs_str = "--"
            c_words_str = "--"
            doc_change = "REMOVED"
            word_change = "REMOVED"
        else:
            c_docs_str = f"{c_docs:,}"
            c_words_str = f"{c_words:,}"

        sections.append(
            f"| {cat:<21} | {b_docs_str:>11} | {c_docs_str:>11} | {doc_change:<12} "
            f"| {b_words_str:>13} | {c_words_str:>13} | {word_change:<12} |"
        )

    # --- Deduplication rate table ---
    sections.append("")
    sections.append("## Deduplication rate")
    sections.append("")
    sections.append(
        "Overall deduplication rate calculated from rejection summary "
        "(near_duplicate rejections / total raw documents). Per-category "
        "loss rates compare raw JSON file counts against final source "
        "distribution counts, reflecting combined cleaning, deduplication, "
        "and capping losses."
    )
    sections.append("")

    # Overall dedup rate
    b_rejections = baseline.get("rejection_summary", {})
    c_rejections = current.get("rejection_summary", {})

    b_near_dup = b_rejections.get("near_duplicate", 0)
    c_near_dup = c_rejections.get("near_duplicate", 0)

    b_total_docs = 0
    try:
        b_total_docs = _parse_number(b_summary.get("Total documents", "0"))
    except (ValueError, TypeError):
        pass
    c_total_docs = 0
    try:
        c_total_docs = _parse_number(c_summary.get("Total documents", "0"))
    except (ValueError, TypeError):
        pass

    b_total_rejected = sum(b_rejections.values())
    c_total_rejected = sum(c_rejections.values())

    b_raw_total = b_total_docs + b_total_rejected
    c_raw_total = c_total_docs + c_total_rejected

    b_dedup_rate = 100 * b_near_dup / b_raw_total if b_raw_total > 0 else 0
    c_dedup_rate = 100 * c_near_dup / c_raw_total if c_raw_total > 0 else 0

    header = f"| Scope                 | {bv} rate   | {cv} rate   | Change       |"
    sep =     "|-----------------------|-------------|-------------|--------------|"
    sections.append(header)
    sections.append(sep)
    sections.append(
        f"| {'Overall':<21} | {b_dedup_rate:>9.1f}%  | {c_dedup_rate:>9.1f}%  "
        f"| {_fmt_change(b_dedup_rate, c_dedup_rate, is_pct=True):<12} |"
    )

    # Per-category loss rates
    current_raw_by_category = _aggregate_raw_by_category(raw_dir)
    baseline_raw_by_category = baseline_sizes.get("raw_counts", {})
    if not baseline_raw_by_category:
        logger.warning(
            "Baseline snapshot has no raw_counts — using current raw counts "
            "for baseline loss rates (may be inaccurate if raw data changed)"
        )
        baseline_raw_by_category = current_raw_by_category

    for cat in all_categories:
        c_raw = current_raw_by_category.get(cat, 0)
        b_raw = baseline_raw_by_category.get(cat, 0)
        c_final = c_sources.get(cat, {}).get("documents", 0)
        b_final = b_sources.get(cat, {}).get("documents", 0)

        c_loss_rate = 100 * (c_raw - c_final) / c_raw if c_raw > 0 else 0
        b_loss_rate = 100 * (b_raw - b_final) / b_raw if b_raw > 0 else 0

        sections.append(
            f"| {cat:<21} | {b_loss_rate:>9.1f}%  | {c_loss_rate:>9.1f}%  "
            f"| {_fmt_change(b_loss_rate, c_loss_rate, is_pct=True):<12} |"
        )

    # --- Document length table ---
    sections.append("")
    sections.append("## Document length")
    sections.append("")
    sections.append(
        "Average and median word counts per document."
    )
    sections.append("")

    b_lengths = baseline.get("document_length", {})
    c_lengths = current.get("document_length", {})

    header = f"| Metric  | {bv:<9} | {cv:<9} | Change              |"
    sep =     "|---------|-----------|-----------|---------------------|"
    sections.append(header)
    sections.append(sep)

    for metric in ["Average", "Median"]:
        b_val = b_lengths.get(metric, 0)
        c_val = c_lengths.get(metric, 0)
        sections.append(
            f"| {metric:<7} | {b_val:>7,}   | {c_val:>7,}   | {_fmt_change(b_val, c_val):<19} |"
        )

    # --- File sizes table ---
    sections.append("")
    sections.append("## File sizes")
    sections.append("")
    sections.append(
        "Output file sizes for the concatenated training and validation files."
    )
    sections.append("")

    header = f"| File      | {bv:<11} | {cv:<11} | Change              |"
    sep =     "|-----------|-------------|-------------|---------------------|"
    sections.append(header)
    sections.append(sep)
    sections.append(
        f"| train.txt | {b_train_mb_str:<11} | {c_train_mb_str:<11} "
        f"| {_fmt_change(b_train_mb, c_train_mb):<19} |"
    )
    sections.append(
        f"| val.txt   | {b_val_mb_str:<11} | {c_val_mb_str:<11} "
        f"| {_fmt_change(b_val_mb, c_val_mb):<19} |"
    )

    # --- Per-source regression warnings ---
    sections.append("")
    sections.append("## Per-source regression warnings")
    sections.append("")

    regressions: list[str] = []
    for cat in all_categories:
        b_docs = b_sources.get(cat, {}).get("documents", 0)
        c_docs = c_sources.get(cat, {}).get("documents", 0)

        if b_docs > 0 and c_docs < b_docs:
            drop_pct = 100 * (b_docs - c_docs) / b_docs
            if drop_pct > 10:
                regressions.append(
                    f"**WARNING:** {cat} dropped from {b_docs:,} to "
                    f"{c_docs:,} documents ({drop_pct:.1f}% decrease)"
                )

    if regressions:
        sections.append(
            "The following source categories show a document count "
            f"decrease greater than 10% compared to {bv}."
        )
        sections.append("")
        for warning in regressions:
            sections.append(warning)
    else:
        sections.append("No per-source regressions detected.")

    # --- Footer ---
    sections.append("")
    sections.append("---")
    sections.append("")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sections.append(f"*Generated: {timestamp}*")
    sections.append(f"*Script: data/processing/compare_corpus.py*")
    sections.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(sections)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Wrote comparison report to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run corpus comparison from the command line.

    Provides CLI arguments for configuring paths, version labels, and
    a snapshot-only mode for capturing baseline statistics without
    generating the comparison report.
    """
    parser = argparse.ArgumentParser(
        description="Generate a corpus comparison report between two versions",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Only snapshot baseline, do not generate report",
    )
    parser.add_argument(
        "--baseline-version",
        default="v1.0.0",
        help="Label for the baseline version (default: v1.0.0)",
    )
    parser.add_argument(
        "--current-version",
        default="v1.2.0",
        help="Label for the current version (default: v1.2.0)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("data/stats.md"),
        help="Path to current stats.md (default: data/stats.md)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("data/processed/v1.0.0"),
        help="Path to baseline directory (default: data/processed/v1.0.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/corpus_comparison.md"),
        help="Path to write comparison report (default: docs/corpus_comparison.md)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    train_path = Path("data/processed/train.txt")
    val_path = Path("data/processed/val.txt")

    try:
        # Step 1: Snapshot baseline (idempotent)
        logger.info("Step 1: Checking %s baseline snapshot...", args.baseline_version)
        snapshot_baseline(
            args.stats_path, train_path, val_path, args.raw_dir, args.baseline_dir,
        )

        if args.snapshot_only:
            logger.info("Snapshot-only mode — exiting without generating report")
            return

        # Step 2: Generate comparison report
        logger.info("Step 2: Generating comparison report...")
        generate_comparison_report(
            baseline_dir=args.baseline_dir,
            stats_path=args.stats_path,
            raw_dir=args.raw_dir,
            train_path=train_path,
            val_path=val_path,
            output_path=args.output,
            baseline_version=args.baseline_version,
            current_version=args.current_version,
        )

        logger.info("Done")
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
