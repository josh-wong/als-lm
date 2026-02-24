"""Train/validation splitting with source caps for ALS corpus.

Applies source category caps (patient narratives <= 10%, Wikipedia <= 15%),
performs stratified 90/10 train/val splitting preserving source distribution,
runs cross-set leakage detection, writes per-document text files and
concatenated training files with <|endoftext|> separators.

Functions:
    apply_source_caps: Enforce source category limits on post-dedup corpus.
    stratified_split: Split documents into train/val with stratification.
    write_per_document_files: Write individual .txt files per document.
    write_concatenated_files: Write train.txt and val.txt with separators.
    split_and_write: Orchestrate the full split pipeline.
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source category caps (applied post-dedup per pitfall 4)
# ---------------------------------------------------------------------------

SOURCE_CAPS = {
    "patient_narratives": 0.10,
    "supplementary_science": 0.15,
}

# Map source_category to output directory names
# Note: biomedical_research (PubMed abstracts) is intentionally uncapped --
# abstracts are core biomedical content that provides vocabulary breadth
# for tokenizer training. Bookshelf and CDC use source_category="educational"
# so they naturally map to the existing educational directory.
SOURCE_DIR_MAP = {
    "biomedical_research": "pubmed",
    "clinical_trials": "clinical_trials",
    "regulatory": "fda",
    "educational": "educational",
    "supplementary_science": "wikipedia",
    "patient_narratives": "patient_narratives",
}

# Endoftext separator between documents in concatenated files
ENDOFTEXT_SEPARATOR = "\n<|endoftext|>\n"


# ---------------------------------------------------------------------------
# Source caps
# ---------------------------------------------------------------------------

def _get_cap_category(doc: dict) -> str | None:
    """Determine the cap category for a document, if any.

    Checks both source_category and source fields to catch documents
    regardless of how they were tagged during scraping.

    Args:
        doc: Document dict with source and source_category fields.

    Returns:
        Cap category key (matching SOURCE_CAPS) or None if uncapped.
    """
    source_category = doc.get("source_category", "")
    source = doc.get("source", "")

    # Check patient narratives
    if (
        source_category == "patient_narratives"
        or source == "patient_narratives"
        or "patient" in source_category.lower()
    ):
        return "patient_narratives"

    # Check Wikipedia / supplementary science
    if source_category == "supplementary_science" or source == "wikipedia":
        return "supplementary_science"

    return None


def apply_source_caps(
    documents: list[dict],
    rejected_path: Path,
) -> list[dict]:
    """Enforce source category caps on the post-dedup corpus.

    Caps are calculated against the total corpus size after deduplication.
    Documents exceeding a cap are randomly downsampled using a fixed seed
    for reproducibility, and removed documents are logged to rejected.jsonl.

    Args:
        documents: List of deduplicated document dicts.
        rejected_path: Path to the JSONL rejection log file.

    Returns:
        List of documents with source caps applied.
    """
    total = len(documents)
    if total == 0:
        return documents

    rng = random.Random(42)

    # Group documents by cap category
    capped_groups: dict[str, list[int]] = {}
    for i, doc in enumerate(documents):
        cap_cat = _get_cap_category(doc)
        if cap_cat is not None:
            capped_groups.setdefault(cap_cat, []).append(i)

    removed_indices: set[int] = set()

    for cap_cat, indices in capped_groups.items():
        cap_ratio = SOURCE_CAPS.get(cap_cat, 1.0)
        max_allowed = int(total * cap_ratio)
        current_count = len(indices)

        logger.info(
            "Source cap check - %s: %d documents (%.1f%%), cap: %.0f%% (%d max)",
            cap_cat,
            current_count,
            100 * current_count / total,
            100 * cap_ratio,
            max_allowed,
        )

        if current_count > max_allowed:
            # Randomly select documents to remove
            to_remove_count = current_count - max_allowed
            indices_to_remove = rng.sample(indices, to_remove_count)
            removed_indices.update(indices_to_remove)

            for idx in indices_to_remove:
                doc = documents[idx]
                _log_split_rejection(
                    doc_id=doc.get("id", "unknown"),
                    source=doc.get("source", "unknown"),
                    reason="source_cap_exceeded",
                    details={
                        "cap_category": cap_cat,
                        "cap_ratio": cap_ratio,
                        "original_count": current_count,
                        "max_allowed": max_allowed,
                    },
                    log_path=rejected_path,
                )

            logger.info(
                "Source cap applied - %s: removed %d documents "
                "(%.1f%% -> %.1f%%)",
                cap_cat,
                to_remove_count,
                100 * current_count / total,
                100 * max_allowed / total,
            )
        else:
            logger.info(
                "Source cap OK - %s: %d documents within %d cap",
                cap_cat,
                current_count,
                max_allowed,
            )

    survivors = [
        doc for i, doc in enumerate(documents)
        if i not in removed_indices
    ]

    logger.info(
        "Source caps complete: %d input, %d removed, %d remaining",
        total,
        len(removed_indices),
        len(survivors),
    )

    return survivors


# ---------------------------------------------------------------------------
# Stratified splitting
# ---------------------------------------------------------------------------

def stratified_split(
    documents: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split documents into train and validation sets with stratification.

    Uses sklearn train_test_split with stratification by source_category
    to maintain the same source distribution in both sets. Handles the
    edge case where a source_category has fewer than 2 documents by
    placing those documents in the training set before stratifying
    the rest.

    Args:
        documents: List of document dicts to split.

    Returns:
        Tuple of (train_docs, val_docs).
    """
    if not documents:
        return [], []

    if len(documents) < 2:
        return documents, []

    # Count documents per source_category
    category_counts = Counter(
        doc.get("source_category", "unknown") for doc in documents
    )

    # Separate documents from categories with < 2 members
    small_category_docs = []
    stratifiable_docs = []

    for doc in documents:
        cat = doc.get("source_category", "unknown")
        if category_counts[cat] < 2:
            small_category_docs.append(doc)
        else:
            stratifiable_docs.append(doc)

    if small_category_docs:
        logger.info(
            "Moved %d documents from small categories to training set "
            "(categories with < 2 documents cannot be stratified)",
            len(small_category_docs),
        )

    if not stratifiable_docs:
        # All categories have < 2 members, no stratification possible
        return documents, []

    # Extract stratification labels
    labels = [
        doc.get("source_category", "unknown")
        for doc in stratifiable_docs
    ]

    train_docs, val_docs = train_test_split(
        stratifiable_docs,
        test_size=0.10,
        random_state=42,
        stratify=labels,
    )

    # Add small-category docs to training set
    train_docs = small_category_docs + train_docs

    logger.info(
        "Stratified split: %d train, %d val (%.1f%% / %.1f%%)",
        len(train_docs),
        len(val_docs),
        100 * len(train_docs) / (len(train_docs) + len(val_docs)),
        100 * len(val_docs) / (len(train_docs) + len(val_docs)),
    )

    # Log per-category distribution
    train_cats = Counter(
        doc.get("source_category", "unknown") for doc in train_docs
    )
    val_cats = Counter(
        doc.get("source_category", "unknown") for doc in val_docs
    )
    all_cats = sorted(set(train_cats.keys()) | set(val_cats.keys()))

    for cat in all_cats:
        t = train_cats.get(cat, 0)
        v = val_cats.get(cat, 0)
        total_cat = t + v
        logger.info(
            "  %s: train=%d (%.0f%%), val=%d (%.0f%%)",
            cat,
            t,
            100 * t / total_cat if total_cat > 0 else 0,
            v,
            100 * v / total_cat if total_cat > 0 else 0,
        )

    return train_docs, val_docs


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def write_per_document_files(
    documents: list[dict],
    output_dir: Path,
) -> None:
    """Write individual .txt files for each document to source subdirectories.

    Each document is written to data/processed/{source_dir}/{doc_id}.txt
    where source_dir is mapped from source_category using SOURCE_DIR_MAP.
    The text content already has the title embedded as the first line
    (from clean.py).

    Args:
        documents: List of document dicts to write.
        output_dir: Root output directory (e.g., data/processed/).
    """
    output_dir = Path(output_dir)
    files_written = 0

    for doc in documents:
        source_category = doc.get("source_category", "unknown")
        source_dir_name = SOURCE_DIR_MAP.get(source_category)
        if source_dir_name is None:
            # Unmapped category: fall back to source field as directory name
            source_dir_name = doc.get("source", source_category)
            logger.warning(
                "Unmapped source_category '%s' for document '%s', "
                "using '%s' as output directory",
                source_category,
                doc.get("id", "unknown"),
                source_dir_name,
            )
        doc_dir = output_dir / source_dir_name
        doc_dir.mkdir(parents=True, exist_ok=True)

        doc_id = doc.get("id", "unknown")
        # Sanitize doc_id for filesystem use — strip all non-safe characters
        safe_id = re.sub(r"[^a-zA-Z0-9_\-.]", "_", doc_id)
        file_path = doc_dir / f"{safe_id}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["text"])

        files_written += 1

    logger.info(
        "Wrote %d per-document .txt files to %s",
        files_written,
        output_dir,
    )


def write_concatenated_files(
    train_docs: list[dict],
    val_docs: list[dict],
    output_dir: Path,
) -> None:
    """Write concatenated train.txt and val.txt files.

    Each file contains all documents concatenated with the <|endoftext|>
    separator between them. This format is the standard input for GPT-2
    style training.

    Args:
        train_docs: List of training document dicts.
        val_docs: List of validation document dicts.
        output_dir: Root output directory (e.g., data/processed/).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    # Write train.txt
    train_texts = [doc["text"] for doc in train_docs]
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(ENDOFTEXT_SEPARATOR.join(train_texts))

    train_size = train_path.stat().st_size
    logger.info(
        "Wrote train.txt: %d documents, %.2f MB",
        len(train_docs),
        train_size / (1024 * 1024),
    )

    # Write val.txt
    val_texts = [doc["text"] for doc in val_docs]
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(ENDOFTEXT_SEPARATOR.join(val_texts))

    val_size = val_path.stat().st_size
    logger.info(
        "Wrote val.txt: %d documents, %.2f MB",
        len(val_docs),
        val_size / (1024 * 1024),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def split_and_write(
    documents: list[dict],
    output_dir: Path,
    rejected_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Orchestrate the full split pipeline.

    Applies source caps, performs stratified splitting, runs cross-set
    leakage detection, writes per-document files and concatenated
    training files. Returns the final train and val document lists
    for use by the stats generator.

    Pipeline order:
    1. Apply source caps (patient narratives <= 10%, Wikipedia <= 15%)
    2. Stratified train/val split (90/10, seed=42)
    3. Cross-set leakage check (Jaccard > 0.7 removed from val)
    4. Write per-document .txt files
    5. Write concatenated train.txt and val.txt

    Args:
        documents: List of deduplicated document dicts.
        output_dir: Root output directory (e.g., data/processed/).
        rejected_path: Path to the JSONL rejection log file.

    Returns:
        Tuple of (train_docs, val_docs) after all processing.
    """
    logger.info("Starting split pipeline on %d documents", len(documents))

    # Step 1: Apply source caps
    capped_docs = apply_source_caps(documents, rejected_path)

    # Step 2: Stratified split
    train_docs, val_docs = stratified_split(capped_docs)

    # Step 3: Cross-set leakage check
    from data.processing.dedup import cross_set_dedup_check

    leaking_ids = cross_set_dedup_check(train_docs, val_docs)

    if leaking_ids:
        leaking_set = set(leaking_ids)
        for doc in val_docs:
            if doc["id"] in leaking_set:
                _log_split_rejection(
                    doc_id=doc["id"],
                    source=doc.get("source", "unknown"),
                    reason="cross_set_leakage",
                    details={
                        "method": "minhash_lsh",
                        "threshold": 0.7,
                    },
                    log_path=rejected_path,
                )
        val_docs = [
            doc for doc in val_docs
            if doc["id"] not in leaking_set
        ]
        logger.info(
            "Removed %d leaking documents from validation set",
            len(leaking_ids),
        )

    # Step 4: Write per-document files
    all_docs = train_docs + val_docs
    write_per_document_files(all_docs, output_dir)

    # Step 5: Write concatenated files
    write_concatenated_files(train_docs, val_docs, output_dir)

    logger.info(
        "Split pipeline complete: %d train, %d val",
        len(train_docs),
        len(val_docs),
    )

    return train_docs, val_docs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_split_rejection(
    doc_id: str,
    source: str,
    reason: str,
    details: dict,
    log_path: Path,
) -> None:
    """Append a split rejection entry to the JSONL log file."""
    entry = {
        "id": doc_id,
        "source": source,
        "reason": reason,
        **details,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run the split pipeline from the command line.

    Reads deduplicated document dicts from a JSON file, applies source
    caps, splits into train/val sets, writes per-document files and
    concatenated training files, and reports split statistics.
    """
    parser = argparse.ArgumentParser(
        description="Split ALS corpus into train/val sets with source caps",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSON file containing deduplicated document dicts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for split files (default: data/processed)",
    )
    parser.add_argument(
        "--rejected",
        type=Path,
        default=Path("data/processed/rejected.jsonl"),
        help="Path to rejection log (default: data/processed/rejected.jsonl)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load input documents
    with open(args.input, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info("Loaded %d documents from %s", len(documents), args.input)

    # Ensure output directories exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)

    # Run split pipeline
    train_docs, val_docs = split_and_write(
        documents, args.output_dir, args.rejected
    )

    # Report statistics
    print(f"\n--- Split Statistics ---", file=sys.stderr)
    print(f"Input documents:      {len(documents)}", file=sys.stderr)
    print(f"Training documents:   {len(train_docs)}", file=sys.stderr)
    print(f"Validation documents: {len(val_docs)}", file=sys.stderr)


if __name__ == "__main__":
    main()
