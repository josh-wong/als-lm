"""Full corpus processing pipeline orchestrator for ALS-LM.

Executes the complete processing pipeline in sequence: load raw documents,
clean all documents, deduplicate, apply source caps and split into train/val
sets, and generate corpus statistics. Designed to be run as a single command
to reproduce the entire processing pipeline from raw data to training-ready
files.

Usage:
    python -m data.processing.run_pipeline
    python data/processing/run_pipeline.py --raw-dir data/raw --output-dir data/processed

Functions:
    run_pipeline: Execute the complete processing pipeline.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_pipeline(
    raw_dir: Path,
    output_dir: Path,
    stats_path: Path,
    seed: int = 42,
) -> None:
    """Execute the complete corpus processing pipeline.

    Orchestrates loading, cleaning, deduplication, splitting, and
    statistics generation in sequence. Each stage logs progress and
    document counts. Individual document failures within a stage are
    isolated so the pipeline continues with remaining documents.

    Pipeline stages:
    1. Load raw documents from data/raw/{source}/ directories
    2. Clean all documents (HTML stripping, normalization, filtering)
    3. Deduplicate (MinHash document-level, SHA-256 paragraph-level)
    4. Split into train/val (source caps, stratified 90/10, leakage check)
    5. Generate corpus statistics report

    Args:
        raw_dir: Path to raw data directory (e.g., data/raw/).
        output_dir: Path to output directory (e.g., data/processed/).
        stats_path: Path to write stats report (e.g., data/stats.md).
        seed: Random seed for reproducibility (default: 42).
    """
    pipeline_start = time.time()

    # Set global random seed for reproducibility
    random.seed(seed)
    logger.info("Random seed set to %d", seed)

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rejected_path = output_dir / "rejected.jsonl"

    # Clear previous rejected log if it exists
    if rejected_path.exists():
        rejected_path.unlink()
        logger.info("Cleared previous rejection log")

    # Import pipeline modules
    from data.processing.clean import clean_all, load_raw_documents
    from data.processing.dedup import run_dedup
    from data.processing.split import split_and_write
    from data.processing.stats import generate_stats

    # ------------------------------------------------------------------
    # Stage 1: Load raw documents
    # ------------------------------------------------------------------
    stage_start = time.time()
    logger.info("=" * 60)
    logger.info("STAGE 1: Loading raw documents")
    logger.info("=" * 60)

    try:
        documents = load_raw_documents(raw_dir)
    except Exception as e:
        logger.error("Failed to load raw documents: %s", e)
        raise

    if not documents:
        logger.error("No documents found in %s. Aborting pipeline.", raw_dir)
        return

    # Log input statistics per source
    source_counts = {}
    for doc in documents:
        src = doc.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    logger.info("Loaded %d documents from %d sources:", len(documents), len(source_counts))
    for src, count in sorted(source_counts.items()):
        logger.info("  %s: %d", src, count)
    logger.info("Stage 1 completed in %.1f seconds", time.time() - stage_start)

    # ------------------------------------------------------------------
    # Stage 2: Clean all documents
    # ------------------------------------------------------------------
    stage_start = time.time()
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2: Cleaning documents")
    logger.info("=" * 60)

    try:
        cleaned_docs = clean_all(documents, rejected_path)
    except Exception as e:
        logger.error("Cleaning stage failed: %s", e)
        raise

    rejected_after_clean = len(documents) - len(cleaned_docs)
    logger.info(
        "Cleaning complete: %d -> %d documents (%d rejected)",
        len(documents),
        len(cleaned_docs),
        rejected_after_clean,
    )

    # Log rejection reasons
    _log_rejection_summary(rejected_path, "cleaning")
    logger.info("Stage 2 completed in %.1f seconds", time.time() - stage_start)

    if not cleaned_docs:
        logger.error("No documents survived cleaning. Aborting pipeline.")
        return

    # ------------------------------------------------------------------
    # Stage 3: Deduplicate
    # ------------------------------------------------------------------
    stage_start = time.time()
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3: Deduplicating documents")
    logger.info("=" * 60)

    try:
        deduped_docs = run_dedup(cleaned_docs, rejected_path)
    except Exception as e:
        logger.error("Deduplication stage failed: %s", e)
        raise

    dedup_removed = len(cleaned_docs) - len(deduped_docs)
    logger.info(
        "Dedup complete: %d -> %d documents (%d removed)",
        len(cleaned_docs),
        len(deduped_docs),
        dedup_removed,
    )
    logger.info("Stage 3 completed in %.1f seconds", time.time() - stage_start)

    if not deduped_docs:
        logger.error("No documents survived deduplication. Aborting pipeline.")
        return

    # ------------------------------------------------------------------
    # Stage 4: Split and write
    # ------------------------------------------------------------------
    stage_start = time.time()
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4: Splitting and writing output files")
    logger.info("=" * 60)

    try:
        train_docs, val_docs = split_and_write(
            deduped_docs, output_dir, rejected_path
        )
    except Exception as e:
        logger.error("Split stage failed: %s", e)
        raise

    logger.info(
        "Split complete: %d train, %d val",
        len(train_docs),
        len(val_docs),
    )
    logger.info("Stage 4 completed in %.1f seconds", time.time() - stage_start)

    # ------------------------------------------------------------------
    # Stage 5: Generate statistics
    # ------------------------------------------------------------------
    stage_start = time.time()
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 5: Generating corpus statistics")
    logger.info("=" * 60)

    try:
        generate_stats(
            train_docs,
            val_docs,
            rejected_path,
            stats_path,
            train_path=output_dir / "train.txt",
            val_path=output_dir / "val.txt",
        )
    except Exception as e:
        logger.error("Statistics generation failed: %s", e)
        raise

    logger.info("Stage 5 completed in %.1f seconds", time.time() - stage_start)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    pipeline_duration = time.time() - pipeline_start

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    _print_final_summary(
        input_count=len(documents),
        cleaned_count=len(cleaned_docs),
        deduped_count=len(deduped_docs),
        train_count=len(train_docs),
        val_count=len(val_docs),
        duration=pipeline_duration,
        output_dir=output_dir,
        stats_path=stats_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_rejection_summary(rejected_path: Path, stage: str) -> None:
    """Read and log rejection counts by reason from the JSONL file."""
    if not rejected_path.exists():
        return

    reasons = {}
    with open(rejected_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                reason = entry.get("reason", "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
            except json.JSONDecodeError:
                continue

    if reasons:
        logger.info("Rejection summary after %s:", stage)
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            logger.info("  %s: %d", reason, count)


def _print_final_summary(
    input_count: int,
    cleaned_count: int,
    deduped_count: int,
    train_count: int,
    val_count: int,
    duration: float,
    output_dir: Path,
    stats_path: Path,
) -> None:
    """Print the final pipeline summary table to stdout."""
    print("\n" + "=" * 50)
    print("  ALS Corpus Processing Pipeline Summary")
    print("=" * 50)
    print(f"  Raw documents loaded:    {input_count:>6,}")
    print(f"  After cleaning:          {cleaned_count:>6,}")
    print(f"  After deduplication:     {deduped_count:>6,}")
    print(f"  Training documents:      {train_count:>6,}")
    print(f"  Validation documents:    {val_count:>6,}")
    print(f"  Total pipeline time:     {duration:>6.1f}s")
    print("-" * 50)
    print(f"  Output directory: {output_dir}")
    print(f"  Statistics file:  {stats_path}")

    # Show output file sizes
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    if train_path.exists():
        train_mb = train_path.stat().st_size / (1024 * 1024)
        print(f"  train.txt size:   {train_mb:.2f} MB")
    if val_path.exists():
        val_mb = val_path.stat().st_size / (1024 * 1024)
        print(f"  val.txt size:     {val_mb:.2f} MB")

    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run the full processing pipeline from the command line.

    Provides CLI arguments for configuring input/output paths, random seed,
    and logging level. Sets up dual-output logging (stdout + file) following
    the pattern established in Phase 1.
    """
    parser = argparse.ArgumentParser(
        description="Run the full ALS corpus processing pipeline",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed files (default: data/processed)",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("data/stats.md"),
        help="Path to write statistics report (default: data/stats.md)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Set up dual-output logging (stdout + file)
    log_level = getattr(logging, args.log_level)
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))

    # File handler
    log_file = args.output_dir / "pipeline.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger.info("ALS Corpus Processing Pipeline starting...")
    logger.info("Configuration:")
    logger.info("  Raw directory:  %s", args.raw_dir)
    logger.info("  Output directory: %s", args.output_dir)
    logger.info("  Stats path:     %s", args.stats_path)
    logger.info("  Random seed:    %d", args.seed)
    logger.info("  Log level:      %s", args.log_level)

    try:
        run_pipeline(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            stats_path=args.stats_path,
            seed=args.seed,
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
