"""Document-level and paragraph-level deduplication for ALS corpus.

Removes near-duplicate documents using MinHash LSH with word-level 5-gram
shingles and exact-duplicate paragraphs using SHA-256 hashing. Document-level
deduplication runs first to remove highly similar documents, followed by
paragraph-level deduplication to remove repeated text blocks across the
surviving corpus.

Functions:
    create_minhash: Build a MinHash signature from text using word 5-grams.
    deduplicate_documents: Remove near-duplicate documents via MinHash LSH.
    dedup_paragraphs: Remove exact-duplicate paragraphs via SHA-256.
    cross_set_dedup_check: Check for train/val leakage via MinHash similarity.
    run_dedup: Orchestrate document-level then paragraph-level dedup.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

NUM_PERM = 128
JACCARD_THRESHOLD = 0.85
NGRAM_SIZE = 5
MIN_PARAGRAPH_WORDS = 50
CROSS_SET_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# MinHash creation
# ---------------------------------------------------------------------------

def create_minhash(text: str) -> MinHash:
    """Create a MinHash signature from word-level n-gram shingles.

    Lowercases the text, splits into words, creates overlapping word
    5-grams as shingles, and builds a MinHash with 128 permutations.
    Word-level 5-grams reduce false positives compared to character
    shingles when applied to domain-specific medical text that shares
    high vocabulary overlap.

    Args:
        text: Document text to create a MinHash signature for.

    Returns:
        MinHash object with 128 permutations.
    """
    m = MinHash(num_perm=NUM_PERM)
    words = text.lower().split()

    if len(words) < NGRAM_SIZE:
        # For very short texts, use individual words as shingles
        for word in words:
            m.update(word.encode("utf-8"))
    else:
        for i in range(len(words) - NGRAM_SIZE + 1):
            shingle = " ".join(words[i:i + NGRAM_SIZE])
            m.update(shingle.encode("utf-8"))

    return m


# ---------------------------------------------------------------------------
# Document-level deduplication
# ---------------------------------------------------------------------------

def deduplicate_documents(
    documents: list[dict],
    rejected_path: Path,
) -> list[dict]:
    """Remove near-duplicate documents using MinHash LSH.

    Builds MinHash signatures for all documents using word-level 5-gram
    shingles, then uses MinHashLSH with a Jaccard threshold of 0.85 to
    identify clusters of similar documents. For each cluster, only the
    first document inserted is kept; subsequent duplicates are logged to
    the rejection file.

    Args:
        documents: List of cleaned document dicts.
        rejected_path: Path to the JSONL rejection log file.

    Returns:
        List of document dicts with near-duplicates removed.
    """
    if not documents:
        return documents

    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
    minhashes = {}
    duplicates = set()
    duplicate_matches = {}

    logger.info("Building MinHash signatures for %d documents...", len(documents))

    # Build MinHash for each document
    for doc in documents:
        doc_id = doc["id"]
        mh = create_minhash(doc["text"])
        minhashes[doc_id] = mh

    # Insert and query for duplicates
    for doc in documents:
        doc_id = doc["id"]
        mh = minhashes[doc_id]

        # Check if this document is similar to any already indexed
        try:
            result = lsh.query(mh)
        except ValueError:
            # Handle edge cases where query fails
            result = []

        if result:
            # This document is a near-duplicate of something already indexed
            duplicates.add(doc_id)
            duplicate_matches[doc_id] = result[0]  # Record which doc it matched
            logger.debug(
                "Near-duplicate found: %s matches %s",
                doc_id,
                result[0],
            )
        else:
            try:
                lsh.insert(doc_id, mh)
            except ValueError as e:
                # Duplicate key error (exact same ID inserted twice)
                logger.warning("Duplicate ID in corpus: %s (%s)", doc_id, e)
                duplicates.add(doc_id)

    # Log rejections — index by ID to avoid O(n²) lookups
    doc_by_id = {d["id"]: d for d in documents}
    for doc_id in duplicates:
        doc = doc_by_id[doc_id]
        matched_id = duplicate_matches.get(doc_id, "unknown")
        _log_dedup_rejection(
            doc_id=doc_id,
            source=doc.get("source", "unknown"),
            reason="near_duplicate",
            details={
                "matched_id": matched_id,
                "method": "minhash_lsh",
                "threshold": JACCARD_THRESHOLD,
            },
            log_path=rejected_path,
        )

    survivors = [doc for doc in documents if doc["id"] not in duplicates]

    logger.info(
        "Document-level dedup: %d input, %d duplicates removed, %d surviving",
        len(documents),
        len(duplicates),
        len(survivors),
    )

    return survivors


# ---------------------------------------------------------------------------
# Paragraph-level deduplication
# ---------------------------------------------------------------------------

def dedup_paragraphs(documents: list[dict]) -> list[dict]:
    """Remove exact-duplicate paragraphs across the corpus using SHA-256.

    Maintains a global set of paragraph hashes. For each document, splits
    text on double newlines into paragraphs. Paragraphs with fewer than
    50 words are kept unconditionally (to avoid deduplicating short common
    phrases like headers). Qualifying paragraphs are normalized and hashed
    with SHA-256; duplicates are removed.

    If a document loses all qualifying paragraphs, it is rejected with
    reason "all_paragraphs_duplicate".

    Args:
        documents: List of document dicts after document-level dedup.

    Returns:
        List of document dicts with duplicate paragraphs removed.
    """
    seen_hashes: set[str] = set()
    result_docs = []
    total_removed = 0

    for doc in documents:
        text = doc["text"]
        paragraphs = text.split("\n\n")
        unique_paragraphs = []
        qualifying_count = 0
        surviving_qualifying = 0

        for para in paragraphs:
            words = para.split()

            # Short paragraphs are kept unconditionally
            if len(words) < MIN_PARAGRAPH_WORDS:
                unique_paragraphs.append(para)
                continue

            qualifying_count += 1

            # Normalize whitespace before hashing
            normalized = " ".join(words).strip()
            para_hash = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            if para_hash not in seen_hashes:
                seen_hashes.add(para_hash)
                unique_paragraphs.append(para)
                surviving_qualifying += 1
            else:
                total_removed += 1
                logger.debug(
                    "Removed duplicate paragraph from %s (hash: %s...)",
                    doc.get("id", "unknown"),
                    para_hash[:12],
                )

        # If all qualifying paragraphs were duplicates, reject the document
        if qualifying_count > 0 and surviving_qualifying == 0:
            logger.info(
                "Rejecting %s: all qualifying paragraphs are duplicates",
                doc.get("id", "unknown"),
            )
            continue

        # Rejoin and update document
        cleaned_doc = dict(doc)
        cleaned_doc["text"] = "\n\n".join(unique_paragraphs)
        cleaned_doc["word_count"] = len(cleaned_doc["text"].split())
        result_docs.append(cleaned_doc)

    logger.info(
        "Paragraph-level dedup: %d input docs, %d paragraphs removed, "
        "%d output docs",
        len(documents),
        total_removed,
        len(result_docs),
    )

    return result_docs


# ---------------------------------------------------------------------------
# Cross-set deduplication check
# ---------------------------------------------------------------------------

def cross_set_dedup_check(
    train_docs: list[dict],
    val_docs: list[dict],
) -> list[str]:
    """Check for near-duplicate leakage between train and validation sets.

    Builds a MinHash LSH index from all training documents using a LOWER
    threshold of 0.7 Jaccard (more permissive than document-level dedup)
    to catch documents that are similar but not flagged during the initial
    dedup pass. Returns IDs of validation documents that should be removed
    to prevent data leakage.

    This function is intended to be called from split.py after the
    train/val split is performed.

    Args:
        train_docs: List of training document dicts.
        val_docs: List of validation document dicts.

    Returns:
        List of validation document IDs that match training documents
        above the 0.7 Jaccard threshold.
    """
    if not train_docs or not val_docs:
        return []

    lsh = MinHashLSH(threshold=CROSS_SET_THRESHOLD, num_perm=NUM_PERM)

    # Build index from training documents
    logger.info(
        "Building cross-set index from %d training documents...",
        len(train_docs),
    )
    for doc in train_docs:
        mh = create_minhash(doc["text"])
        try:
            lsh.insert(doc["id"], mh)
        except ValueError:
            # Duplicate ID, skip
            continue

    # Query validation documents against training index
    leaking_ids = []
    for doc in val_docs:
        mh = create_minhash(doc["text"])
        try:
            result = lsh.query(mh)
        except ValueError:
            result = []

        if result:
            leaking_ids.append(doc["id"])
            logger.warning(
                "Train/val leakage detected: val doc %s matches train doc %s",
                doc["id"],
                result[0],
            )

    if leaking_ids:
        logger.warning(
            "Cross-set check found %d leaking validation documents",
            len(leaking_ids),
        )
    else:
        logger.info("Cross-set check passed: no train/val leakage detected")

    return leaking_ids


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_dedup(
    documents: list[dict],
    rejected_path: Path,
) -> list[dict]:
    """Run document-level then paragraph-level deduplication.

    Document-level dedup (MinHash LSH) runs first to remove highly similar
    documents. Then paragraph-level dedup (SHA-256) runs on the survivors
    to remove repeated text blocks across the corpus.

    Args:
        documents: List of cleaned document dicts.
        rejected_path: Path to the JSONL rejection log file.

    Returns:
        List of deduplicated document dicts.
    """
    logger.info("Starting deduplication pipeline on %d documents", len(documents))

    # Step 1: Document-level dedup
    after_doc_dedup = deduplicate_documents(documents, rejected_path)

    # Step 2: Paragraph-level dedup on survivors
    after_para_dedup = dedup_paragraphs(after_doc_dedup)

    logger.info(
        "Deduplication complete: %d input -> %d output "
        "(%d removed by doc-level, %d removed by para-level)",
        len(documents),
        len(after_para_dedup),
        len(documents) - len(after_doc_dedup),
        len(after_doc_dedup) - len(after_para_dedup),
    )

    return after_para_dedup


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_dedup_rejection(
    doc_id: str,
    source: str,
    reason: str,
    details: dict,
    log_path: Path,
) -> None:
    """Append a dedup rejection entry to the JSONL log file."""
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
    """Run the deduplication pipeline from the command line.

    Reads cleaned document dicts from a JSON file, runs deduplication,
    writes deduplicated output, and reports statistics.
    """
    parser = argparse.ArgumentParser(
        description="Deduplicate cleaned ALS corpus documents",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSON file containing cleaned document dicts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for deduplicated documents (default: stdout)",
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
    args.rejected.parent.mkdir(parents=True, exist_ok=True)

    # Run dedup pipeline
    deduplicated = run_dedup(documents, args.rejected)

    # Write output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(deduplicated, f, indent=2, ensure_ascii=False)
        logger.info(
            "Wrote %d deduplicated documents to %s",
            len(deduplicated),
            args.output,
        )
    else:
        json.dump(deduplicated, sys.stdout, indent=2, ensure_ascii=False)

    # Report statistics
    doc_dupes = len(documents) - len(deduplicated)
    print(f"\n--- Deduplication Statistics ---", file=sys.stderr)
    print(f"Input documents:  {len(documents)}", file=sys.stderr)
    print(f"Output documents: {len(deduplicated)}", file=sys.stderr)
    print(f"Total removed:    {doc_dupes}", file=sys.stderr)


if __name__ == "__main__":
    main()
