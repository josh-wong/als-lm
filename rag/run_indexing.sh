#!/bin/bash
# Run full corpus indexing for both 500-token and 200-token MiniLM collections.
#
# This script indexes the 35K-document ALS corpus into ChromaDB.
# Expected runtime: 10-15 hours total on CPU (ONNX MiniLM at ~40 embeddings/sec).
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
echo "ALS Corpus Indexing Pipeline"
echo "============================================================"
echo "  Database path: $DB_PATH"
echo "  Python: $PYTHON"
echo "  Start time: $(date)"
echo ""

echo "--- Step 1/4: 500-token MiniLM indexing ---"
"$PYTHON" rag/index_corpus.py --chunk-size 500 --embedding minilm \
    --db-path "$DB_PATH" --resume --verbose
echo ""

echo "--- Step 2/4: Validate 500-token collection ---"
"$PYTHON" rag/index_corpus.py --query "What gene mutations are associated with ALS?" \
    --collection als_500_minilm --db-path "$DB_PATH"
echo ""

echo "--- Step 3/4: 200-token MiniLM indexing ---"
"$PYTHON" rag/index_corpus.py --chunk-size 200 --embedding minilm \
    --db-path "$DB_PATH" --resume --verbose
echo ""

echo "--- Step 4/4: Validate 200-token collection ---"
"$PYTHON" rag/index_corpus.py --query "What are the symptoms of ALS?" \
    --collection als_200_minilm --db-path "$DB_PATH"
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
