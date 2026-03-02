#!/bin/bash
# Run full corpus indexing for all three collections (GPU-accelerated).
#
# This script indexes the 35K-document ALS corpus into ChromaDB using
# sentence-transformers on CUDA for fast embedding computation.
# Expected runtime: 1-3 hours total on RTX 3060.
#
# Usage:
#   bash rag/run_indexing.sh              # Index to data/chromadb (default)
#   bash rag/run_indexing.sh /tmp/chromadb  # Index to custom path
#
# The --resume flag allows restarting after interruption without re-indexing
# documents that were already processed.

set -euo pipefail

DB_PATH="${1:-data/chromadb}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

cd "$PROJECT_DIR"

echo "============================================================"
echo "ALS Corpus Indexing Pipeline (GPU)"
echo "============================================================"
echo "  Database path: $DB_PATH"
echo "  Python: $PYTHON"
echo "  Start time: $(date)"
echo ""

echo "--- Step 1/6: 500-token MiniLM indexing ---"
"$PYTHON" rag/index_corpus.py --chunk-size 500 --embedding minilm \
    --db-path "$DB_PATH" --resume --verbose
echo ""

echo "--- Step 2/6: Validate 500-token MiniLM collection ---"
"$PYTHON" rag/index_corpus.py --query "What gene mutations are associated with ALS?" \
    --collection als_500_minilm --db-path "$DB_PATH"
echo ""

echo "--- Step 3/6: 200-token MiniLM indexing ---"
"$PYTHON" rag/index_corpus.py --chunk-size 200 --embedding minilm \
    --db-path "$DB_PATH" --resume --verbose
echo ""

echo "--- Step 4/6: Validate 200-token MiniLM collection ---"
"$PYTHON" rag/index_corpus.py --query "What are the symptoms of ALS?" \
    --collection als_200_minilm --db-path "$DB_PATH"
echo ""

echo "--- Step 5/6: 500-token PubMedBERT indexing ---"
"$PYTHON" rag/index_corpus.py --chunk-size 500 --embedding pubmedbert \
    --db-path "$DB_PATH" --resume --verbose
echo ""

echo "--- Step 6/6: Validate 500-token PubMedBERT collection ---"
"$PYTHON" rag/index_corpus.py --query "What gene mutations are associated with ALS?" \
    --collection als_500_pubmedbert --db-path "$DB_PATH"
echo ""

echo "============================================================"
echo "ALL INDEXING COMPLETE"
echo "============================================================"
echo "  End time: $(date)"

# Print collection summary
"$PYTHON" -c "
import chromadb
client = chromadb.PersistentClient(path='$DB_PATH')
for c in client.list_collections():
    name = c.name if hasattr(c, 'name') else c
    coll = client.get_collection(name)
    print(f'  {name}: {coll.count():,} chunks')
"
