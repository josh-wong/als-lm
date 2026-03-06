# Shared constants for the RAG comparison package.

# Fuzzy matching threshold for key_fact presence in retrieved chunks.
# Used by both compare_approaches.py (failure decomposition) and
# generate_rag.py (hit rate computation). Keeping this in one place
# ensures both scripts agree on what counts as a "hit".
FUZZY_THRESHOLD = 80
