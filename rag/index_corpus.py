"""Corpus indexing CLI for RAG comparison pipeline.

Chunks the ALS corpus and indexes it into persistent ChromaDB collections
with configurable chunk sizes and embedding models. Supports query mode
for quick verification of indexed collections.

Usage:
    # Index with 500-token chunks and MiniLM embedding
    python rag/index_corpus.py --chunk-size 500 --embedding minilm

    # Index with 200-token chunks
    python rag/index_corpus.py --chunk-size 200 --embedding minilm

    # Query an existing collection
    python rag/index_corpus.py --query "What gene mutations are associated with ALS?" --collection als_500_minilm
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as `python rag/index_corpus.py`
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import chromadb
from tqdm import tqdm

from rag.chunker import chunk_document, count_tokens, extract_metadata

# Batch size for ChromaDB insertions (well under 41,666 hard limit).
# Larger batches reduce embedding call overhead for large corpora.
BATCH_SIZE = 5000

# Files to exclude from indexing (root-level processed directory artifacts)
EXCLUDE_FILES = {"train.txt", "val.txt", "pipeline.log", "rejected.jsonl"}

# Source subdirectories to scan for corpus files
SOURCE_DIRS = [
    "pubmed_abstracts",
    "clinical_trials",
    "educational",
    "wikipedia",
]


def _collection_names(client: chromadb.PersistentClient) -> list[str]:
    """Get collection names from a ChromaDB client.

    ChromaDB 1.x list_collections() returns Collection objects, not strings.
    This helper extracts the name strings for comparison.
    """
    collections = client.list_collections()
    if not collections:
        return []
    # Handle both Collection objects and raw strings
    if hasattr(collections[0], "name"):
        return [c.name for c in collections]
    return list(collections)


def compute_overlap(chunk_size: int) -> int:
    """Compute overlap tokens for a given chunk size.

    Uses 10% of chunk size, clamped to minimum of 10 tokens.
    For 500-token chunks: 50 tokens (10%).
    For 200-token chunks: 20 tokens (10%).
    """
    return max(10, int(chunk_size * 0.10))


def discover_corpus_files(corpus_dir: Path) -> list[Path]:
    """Find all .txt files in source subdirectories, excluding pipeline artifacts."""
    files = []
    for subdir_name in SOURCE_DIRS:
        subdir = corpus_dir / subdir_name
        if subdir.is_dir():
            for f in sorted(subdir.glob("*.txt")):
                if f.name not in EXCLUDE_FILES:
                    files.append(f)
    return files


def _get_embedding_function():
    """Get the ONNX MiniLM embedding function for pre-computing embeddings."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    return ONNXMiniLM_L6_V2()


# Module-level embedding function (lazy-initialized to avoid import cost)
_embedding_fn = None


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Pre-compute embeddings for a batch of texts."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = _get_embedding_function()
    raw = _embedding_fn(texts)
    # Convert numpy arrays to lists for ChromaDB
    return [e.tolist() if hasattr(e, "tolist") else list(e) for e in raw]


def flush_batch(collection, batch: list[dict]) -> None:
    """Flush a batch of chunks to a ChromaDB collection.

    Pre-computes embeddings before insertion to separate embedding time
    from HNSW index update time, allowing larger effective batch sizes.
    """
    if not batch:
        return
    documents = [c["text"] for c in batch]
    embeddings = _embed_texts(documents)
    collection.add(
        ids=[c["id"] for c in batch],
        documents=documents,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in batch],
    )


def run_index(args: argparse.Namespace) -> None:
    """Run corpus indexing mode."""
    corpus_dir = Path(args.corpus_dir)
    db_path = args.db_path
    chunk_size = args.chunk_size
    embedding_name = args.embedding
    verbose = args.verbose

    collection_name = f"als_{chunk_size}_{embedding_name}"
    overlap = compute_overlap(chunk_size)

    print(f"Corpus indexing configuration:")
    print(f"  Corpus directory: {corpus_dir}")
    print(f"  Chunk size: {chunk_size} tokens")
    print(f"  Overlap: {overlap} tokens ({overlap / chunk_size * 100:.0f}%)")
    print(f"  Embedding: {embedding_name}")
    print(f"  Collection: {collection_name}")
    print(f"  DB path: {db_path}")
    print()

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=db_path)

    # Check for existing collection
    existing_collections = _collection_names(client)
    indexed_doc_ids = set()

    if collection_name in existing_collections:
        existing = client.get_collection(collection_name)
        existing_count = existing.count()

        if args.resume:
            print(
                f"RESUME: Collection '{collection_name}' has {existing_count} "
                f"chunks. Skipping already-indexed documents."
            )
            collection = existing
            # Retrieve indexed doc_ids by paginating through collection
            if existing_count > 0:
                page_size = 10000
                offset = 0
                while offset < existing_count:
                    batch_limit = min(page_size, existing_count - offset)
                    result = existing.get(
                        include=["metadatas"],
                        limit=batch_limit,
                        offset=offset,
                    )
                    for m in result["metadatas"]:
                        if m:
                            indexed_doc_ids.add(m.get("doc_id"))
                    offset += batch_limit
                print(f"  Found {len(indexed_doc_ids)} already-indexed documents")
        else:
            print(
                f"WARNING: Collection '{collection_name}' already exists with "
                f"{existing_count} chunks. It will be dropped and rebuilt."
            )
            client.delete_collection(collection_name)
            collection = None

    if not args.resume or collection is None:
        # Create collection (no embedding_function arg = ChromaDB default ONNX MiniLM)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"chunk_size": chunk_size, "embedding_model": embedding_name},
        )

    # Discover corpus files
    files = discover_corpus_files(corpus_dir)
    print(f"Found {len(files)} corpus files across {len(SOURCE_DIRS)} source directories")
    print()

    if not files:
        print("ERROR: No corpus files found. Check --corpus-dir path.")
        sys.exit(1)

    # Process files with progress bar
    start_time = time.time()
    batch = []
    total_chunks = 0
    total_docs = 0
    token_counts = []
    skipped_empty = 0

    for filepath in tqdm(files, desc="Indexing corpus", unit="doc"):
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            if verbose:
                print(f"\n  WARNING: Could not read {filepath}: {e}")
            continue

        if not text.strip():
            skipped_empty += 1
            continue

        # Chunk the document
        chunks = chunk_document(text, max_tokens=chunk_size, overlap_tokens=overlap)
        if not chunks:
            skipped_empty += 1
            continue

        # Extract metadata from file path
        meta = extract_metadata(filepath)

        # Skip already-indexed documents when resuming
        if meta["doc_id"] in indexed_doc_ids:
            continue

        total_docs += 1

        # Build chunk records with compound IDs
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{meta['doc_id']}_chunk_{idx}"
            chunk_meta = {
                "source_type": meta["source_type"],
                "doc_id": meta["doc_id"],
                "source_file": meta["source_file"],
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "token_count": chunk["token_count"],
            }

            batch.append({
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk_meta,
            })

            token_counts.append(chunk["token_count"])
            total_chunks += 1

            # Flush batch when full
            if len(batch) >= BATCH_SIZE:
                flush_batch(collection, batch)
                batch = []

    # Flush remaining chunks
    flush_batch(collection, batch)

    elapsed = time.time() - start_time

    # Verify against collection count
    stored_count = collection.count()

    # Print post-indexing summary
    print()
    print("=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"  Collection:       {collection_name}")
    print(f"  Storage path:     {db_path}")
    print(f"  Documents:        {total_docs:,}")
    if skipped_empty:
        print(f"  Skipped (empty):  {skipped_empty}")
    print(f"  New chunks:       {total_chunks:,}")
    print(f"  Stored in DB:     {stored_count:,}")
    if indexed_doc_ids:
        print(f"  Resumed from:     {len(indexed_doc_ids)} previously indexed docs")

    if token_counts:
        print(f"  Token stats:")
        print(f"    Mean:           {statistics.mean(token_counts):.1f}")
        print(f"    Min:            {min(token_counts)}")
        print(f"    Max:            {max(token_counts)}")
        print(f"    Median:         {statistics.median(token_counts):.1f}")

    print(f"  Wall-clock time:  {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    if stored_count != total_chunks:
        print(
            f"\n  WARNING: Chunk count mismatch! "
            f"Expected {total_chunks}, stored {stored_count}. "
            f"Possible duplicate IDs."
        )

    print()


def run_query(args: argparse.Namespace) -> None:
    """Run query mode against an existing collection."""
    db_path = args.db_path
    query_text = args.query
    top_k = args.top_k

    # Determine collection name
    if args.collection:
        collection_name = args.collection
    else:
        collection_name = f"als_{args.chunk_size}_{args.embedding}"

    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=db_path)

    # Check collection exists
    existing = _collection_names(client)
    if collection_name not in existing:
        print(f"ERROR: Collection '{collection_name}' not found.")
        print(f"Available collections: {existing}")
        sys.exit(1)

    # Get collection (no embedding_function = ChromaDB default ONNX MiniLM)
    collection = client.get_collection(collection_name)
    total_chunks = collection.count()

    # Run query
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Display results
    print(f'Query: "{query_text}"')
    print(f"Collection: {collection_name} ({total_chunks:,} chunks)")
    print()

    if not results["documents"][0]:
        print("No results found.")
        return

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        chunk_info = f"chunk {meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')}"
        tokens = meta.get("token_count", "?")
        print(
            f"[{i + 1}] distance={dist:.4f} | "
            f"source={meta.get('doc_id', 'unknown')} | "
            f"type={meta.get('source_type', 'unknown')} | "
            f"{chunk_info} | {tokens} tokens"
        )
        # Show first 200 characters of chunk text
        preview = doc[:200].replace("\n", " ")
        if len(doc) > 200:
            preview += "..."
        print(f"    {preview}")
        print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Index ALS corpus into ChromaDB for RAG comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Index:  python rag/index_corpus.py --chunk-size 500 --embedding minilm\n"
            "  Query:  python rag/index_corpus.py --query 'ALS gene mutations' "
            "--collection als_500_minilm\n"
        ),
    )

    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="data/processed",
        help="Path to processed corpus directory (default: data/processed)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Maximum tokens per chunk (default: 500)",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="minilm",
        choices=["minilm", "pubmedbert"],
        help="Embedding model (default: minilm)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/chromadb",
        help="ChromaDB persistent storage path (default: data/chromadb)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query text (activates query mode instead of indexing)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name for query mode (default: auto-generated from chunk-size and embedding)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results for query mode (default: 5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted indexing run (skip already-indexed documents)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show extra output during indexing",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for corpus indexing CLI."""
    args = parse_args()

    # Validate chunk size
    if args.chunk_size <= 0:
        print("ERROR: --chunk-size must be a positive integer")
        sys.exit(1)

    if args.query:
        run_query(args)
    else:
        run_index(args)


if __name__ == "__main__":
    main()
