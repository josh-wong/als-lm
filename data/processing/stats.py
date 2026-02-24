"""Corpus statistics generator for ALS corpus processing.

Produces a comprehensive statistics report (data/stats.md) covering corpus
size, source distribution, train/val split ratios, document length analysis,
vocabulary analysis, content tier distribution, and rejection summary. These
metrics inform downstream tokenizer training (Phase 3) and model training
(Phase 5) configuration.

Functions:
    generate_stats: Create the full stats report from train/val documents.
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_stats(
    train_docs: list[dict],
    val_docs: list[dict],
    rejected_path: Path,
    output_path: Path,
    train_path: Path | None = None,
    val_path: Path | None = None,
) -> None:
    """Generate comprehensive corpus statistics and write to a Markdown file.

    Produces sections covering corpus summary, source distribution,
    train/val split ratios, document length analysis, vocabulary analysis,
    content tier distribution, and rejection summary. The output format
    follows project Markdown formatting rules.

    Args:
        train_docs: List of training document dicts.
        val_docs: List of validation document dicts.
        rejected_path: Path to the JSONL rejection log file.
        output_path: Path to write the stats Markdown file.
        train_path: Explicit path to train.txt. If None, defaults to
            output_path.parent / "processed" / "train.txt".
        val_path: Explicit path to val.txt. If None, defaults to
            output_path.parent / "processed" / "val.txt".
    """
    all_docs = train_docs + val_docs
    total_docs = len(all_docs)

    if total_docs == 0:
        logger.warning("No documents to generate statistics for")
        return

    logger.info("Generating corpus statistics for %d documents...", total_docs)

    # Gather all text content
    all_texts = [doc["text"] for doc in all_docs]
    all_words_per_doc = [len(doc["text"].split()) for doc in all_docs]

    # Compute global word list for vocabulary analysis
    all_words_flat = []
    for text in all_texts:
        all_words_flat.extend(text.split())

    total_words = sum(all_words_per_doc)
    total_chars = sum(len(t) for t in all_texts)
    estimated_tokens = total_chars // 4

    # Train/val file sizes
    if train_path is None:
        train_path = output_path.parent / "processed" / "train.txt"
    if val_path is None:
        val_path = output_path.parent / "processed" / "val.txt"

    if not train_path.exists():
        logger.warning("train.txt not found at %s -- file size will report 0", train_path)
    if not val_path.exists():
        logger.warning("val.txt not found at %s -- file size will report 0", val_path)

    train_size = train_path.stat().st_size if train_path.exists() else 0
    val_size = val_path.stat().st_size if val_path.exists() else 0
    total_size = train_size + val_size

    sections = []

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    sections.append("# ALS corpus statistics")
    sections.append("")
    sections.append(
        "This report summarizes the processed ALS training corpus after "
        "cleaning, deduplication, source capping, and train/val splitting. "
        "These metrics inform tokenizer vocabulary size (Phase 3) and model "
        "training configuration (Phase 5)."
    )

    # -----------------------------------------------------------------------
    # Section 1: Corpus summary
    # -----------------------------------------------------------------------
    sections.append("")
    sections.append("## Corpus summary")
    sections.append("")
    sections.append(
        "High-level metrics for the complete processed corpus."
    )
    sections.append("")
    sections.append(f"| Metric                | Value                    |")
    sections.append(f"|-----------------------|--------------------------|")
    sections.append(f"| Total documents       | {total_docs:,}           |")
    sections.append(f"| Total words           | {total_words:,}          |")
    sections.append(f"| Total size            | {_fmt_bytes(total_size)}  |")
    sections.append(f"| Estimated tokens      | ~{estimated_tokens:,}    |")
    sections.append(f"| Training documents    | {len(train_docs):,}      |")
    sections.append(f"| Validation documents  | {len(val_docs):,}        |")

    # -----------------------------------------------------------------------
    # Section 2: Source distribution
    # -----------------------------------------------------------------------
    sections.append("")
    sections.append("## Source distribution")
    sections.append("")
    sections.append(
        "Document and word counts per source category, sorted by word count "
        "descending."
    )

    source_stats = _compute_source_stats(all_docs, total_words)
    sections.append("")

    # Build aligned table
    header = "| Source category      | Documents | Word count  | Percentage |"
    sep =    "|----------------------|-----------|-------------|------------|"
    sections.append(header)
    sections.append(sep)

    for cat, doc_count, word_count, pct in source_stats:
        sections.append(
            f"| {cat:<20} | {doc_count:>9,} | {word_count:>11,} | {pct:>9.1f}% |"
        )

    # -----------------------------------------------------------------------
    # Section 3: Train/val split summary
    # -----------------------------------------------------------------------
    sections.append("")
    sections.append("## Train/val split summary")
    sections.append("")
    sections.append(
        "Per-category document and word counts in training vs validation "
        "sets, confirming stratification maintains approximately 90/10 "
        "ratios across all source categories."
    )

    train_val_rows = _compute_train_val_table(train_docs, val_docs)
    sections.append("")

    header = "| Source category      | Train docs | Train words | Val docs | Val words | Train % |"
    sep =    "|----------------------|------------|-------------|----------|-----------|---------|"
    sections.append(header)
    sections.append(sep)

    for row in train_val_rows:
        cat, td, tw, vd, vw, tp = row
        sections.append(
            f"| {cat:<20} | {td:>10,} | {tw:>11,} | {vd:>8,} | {vw:>9,} | {tp:>6.1f}% |"
        )

    # -----------------------------------------------------------------------
    # Section 4: Document length analysis
    # -----------------------------------------------------------------------
    sections.append("")
    sections.append("## Document length analysis")
    sections.append("")
    sections.append(
        "Word count distribution across all documents."
    )

    sorted_lengths = sorted(all_words_per_doc)
    avg_len = total_words / total_docs if total_docs > 0 else 0
    median_len = _median(sorted_lengths)
    min_len = sorted_lengths[0] if sorted_lengths else 0
    max_len = sorted_lengths[-1] if sorted_lengths else 0

    sections.append("")
    sections.append(f"| Metric  | Words  |")
    sections.append(f"|---------|--------|")
    sections.append(f"| Average | {avg_len:,.0f}  |")
    sections.append(f"| Median  | {median_len:,.0f}  |")
    sections.append(f"| Minimum | {min_len:,}  |")
    sections.append(f"| Maximum | {max_len:,}  |")

    # Histogram buckets
    buckets = [
        ("0-100", 0, 100),
        ("100-500", 100, 500),
        ("500-1,000", 500, 1000),
        ("1,000-5,000", 1000, 5000),
        ("5,000+", 5000, float("inf")),
    ]

    sections.append("")
    sections.append(
        "Document count by word-count bucket."
    )
    sections.append("")
    sections.append(f"| Bucket      | Documents |")
    sections.append(f"|-------------|-----------|")

    for label, lo, hi in buckets:
        count = sum(1 for w in all_words_per_doc if lo <= w < hi)
        sections.append(f"| {label:<11} | {count:>9,} |")

    # -----------------------------------------------------------------------
    # Section 5: Vocabulary analysis
    # -----------------------------------------------------------------------
    sections.append("")
    sections.append("## Vocabulary analysis")
    sections.append("")
    sections.append(
        "Whitespace-delimited vocabulary statistics. These inform "
        "tokenizer vocabulary size selection in Phase 3."
    )

    word_counts = Counter(all_words_flat)
    vocab_size = len(word_counts)

    sections.append("")
    sections.append(f"**Total unique words (vocabulary size):** {vocab_size:,}")

    # Top 100 most frequent words
    sections.append("")
    sections.append("### Top 100 most frequent words")
    sections.append("")
    sections.append(
        "The 100 most frequently occurring tokens in the corpus."
    )
    sections.append("")
    sections.append(f"| Rank | Word               | Count      |")
    sections.append(f"|------|--------------------|------------|")

    for rank, (word, count) in enumerate(word_counts.most_common(100), 1):
        # Truncate very long words for table readability
        display_word = word[:18] if len(word) > 18 else word
        sections.append(
            f"| {rank:>4} | {display_word:<18} | {count:>10,} |"
        )

    # Top 50 words with 3+ characters (filters some stopwords)
    sections.append("")
    sections.append("### Top 50 frequent words (3+ characters)")
    sections.append("")
    sections.append(
        "Most frequent words with at least 3 characters, filtering out "
        "short stopwords."
    )
    sections.append("")
    sections.append(f"| Rank | Word               | Count      |")
    sections.append(f"|------|--------------------|------------|")

    long_words = {w: c for w, c in word_counts.items() if len(w) >= 3}
    long_word_counts = Counter(long_words)
    for rank, (word, count) in enumerate(long_word_counts.most_common(50), 1):
        display_word = word[:18] if len(word) > 18 else word
        sections.append(
            f"| {rank:>4} | {display_word:<18} | {count:>10,} |"
        )

    # -----------------------------------------------------------------------
    # Section 6: Content tier distribution
    # -----------------------------------------------------------------------
    sections.append("")
    sections.append("## Content tier distribution")
    sections.append("")

    tier_counts = Counter(
        doc.get("content_tier", "unspecified") for doc in all_docs
    )

    if len(tier_counts) == 1 and "unspecified" in tier_counts:
        sections.append(
            "No content_tier field found in documents. This section will "
            "be populated if content tier metadata is added during scraping."
        )
    else:
        sections.append(
            "Document counts by content depth tier."
        )
        sections.append("")
        sections.append(f"| Tier        | Documents | Percentage |")
        sections.append(f"|-------------|-----------|------------|")

        for tier in ["deep", "moderate", "light", "unspecified"]:
            count = tier_counts.get(tier, 0)
            if count > 0:
                pct = 100 * count / total_docs
                sections.append(
                    f"| {tier:<11} | {count:>9,} | {pct:>9.1f}% |"
                )

    # -----------------------------------------------------------------------
    # Section 7: Rejection summary
    # -----------------------------------------------------------------------
    sections.append("")
    sections.append("## Rejection summary")
    sections.append("")

    rejection_counts = _read_rejections(rejected_path)

    if rejection_counts:
        total_rejected = sum(rejection_counts.values())
        sections.append(
            f"A total of {total_rejected:,} documents were rejected during "
            f"processing. The breakdown by reason is shown below."
        )
        sections.append("")
        sections.append(f"| Reason                   | Count |")
        sections.append(f"|--------------------------|-------|")

        for reason, count in sorted(
            rejection_counts.items(), key=lambda x: -x[1]
        ):
            sections.append(f"| {reason:<24} | {count:>5,} |")
    else:
        sections.append("No rejected documents found in the rejection log.")

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    sections.append("")
    sections.append("---")
    sections.append("")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sections.append(f"*Generated: {timestamp}*")
    sections.append(f"*Pipeline version: 1.0 (Phase 2 corpus processing)*")
    sections.append(f"*Random seed: 42*")
    sections.append("")

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = "\n".join(sections)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Wrote corpus statistics to %s", output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_source_stats(
    docs: list[dict],
    total_words: int,
) -> list[tuple[str, int, int, float]]:
    """Compute per-source document count, word count, and percentage.

    Returns a list of (category, doc_count, word_count, percentage) tuples
    sorted by word count descending.
    """
    cat_docs: dict[str, int] = Counter()
    cat_words: dict[str, int] = Counter()

    for doc in docs:
        cat = doc.get("source_category", "unknown")
        cat_docs[cat] += 1
        cat_words[cat] += len(doc["text"].split())

    result = []
    for cat in cat_docs:
        doc_count = cat_docs[cat]
        word_count = cat_words[cat]
        pct = 100 * word_count / total_words if total_words > 0 else 0
        result.append((cat, doc_count, word_count, pct))

    result.sort(key=lambda x: -x[2])
    return result


def _compute_train_val_table(
    train_docs: list[dict],
    val_docs: list[dict],
) -> list[tuple[str, int, int, int, int, float]]:
    """Compute per-category train/val document and word counts.

    Returns a list of (category, train_docs, train_words, val_docs,
    val_words, train_pct) tuples sorted by category name.
    """
    train_cats: dict[str, dict] = {}
    val_cats: dict[str, dict] = {}

    for doc in train_docs:
        cat = doc.get("source_category", "unknown")
        if cat not in train_cats:
            train_cats[cat] = {"docs": 0, "words": 0}
        train_cats[cat]["docs"] += 1
        train_cats[cat]["words"] += len(doc["text"].split())

    for doc in val_docs:
        cat = doc.get("source_category", "unknown")
        if cat not in val_cats:
            val_cats[cat] = {"docs": 0, "words": 0}
        val_cats[cat]["docs"] += 1
        val_cats[cat]["words"] += len(doc["text"].split())

    all_cats = sorted(set(train_cats.keys()) | set(val_cats.keys()))
    result = []

    for cat in all_cats:
        td = train_cats.get(cat, {"docs": 0, "words": 0})["docs"]
        tw = train_cats.get(cat, {"docs": 0, "words": 0})["words"]
        vd = val_cats.get(cat, {"docs": 0, "words": 0})["docs"]
        vw = val_cats.get(cat, {"docs": 0, "words": 0})["words"]
        total_cat = td + vd
        train_pct = 100 * td / total_cat if total_cat > 0 else 0
        result.append((cat, td, tw, vd, vw, train_pct))

    return result


def _median(sorted_values: list[int]) -> float:
    """Compute median of a sorted list of integers."""
    n = len(sorted_values)
    if n == 0:
        return 0
    if n % 2 == 1:
        return sorted_values[n // 2]
    return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2


def _fmt_bytes(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _read_rejections(rejected_path: Path) -> dict[str, int]:
    """Read rejection counts from a JSONL file.

    Returns a dict mapping rejection reason to count.
    """
    counts: dict[str, int] = {}

    if not Path(rejected_path).exists():
        return counts

    with open(rejected_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                reason = entry.get("reason", "unknown")
                counts[reason] = counts.get(reason, 0) + 1
            except json.JSONDecodeError:
                continue

    return counts
