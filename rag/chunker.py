"""Corpus chunking module for RAG indexing.

Paragraph-first chunking with sentence-boundary fallback, token overlap,
and metadata extraction from ALS corpus file paths. Uses tiktoken cl100k_base
for model-agnostic token counting and NLTK sent_tokenize for sentence
boundary detection in medical text.
"""

from pathlib import Path
import re

import nltk
import tiktoken

# Download sentence tokenizer data (idempotent, ~2MB on first run)
nltk.download("punkt_tab", quiet=True)

# Module-level tiktoken encoder (cl100k_base is model-agnostic standard for RAG)
enc = tiktoken.get_encoding("cl100k_base")

# Map directory names to source type tags (three-tag schema per CONTEXT.md)
SOURCE_TYPE_MAP = {
    "pubmed_abstracts": "pubmed",
    "clinical_trials": "trial",
    "educational": "educational",
    "wikipedia": "educational",
}

# Patterns for extracting document IDs from filenames
DOC_ID_PATTERNS = [
    (r"pubmed-PMC(\d+)", lambda m: f"PMC{m.group(1)}"),
    (r"clinicaltrials-(NCT\d+)", lambda m: m.group(1)),
    (r"bookshelf-(\d+)-(.+?)(?:-[a-f0-9]+)?$", lambda m: f"NBK{m.group(1)}"),
    (r"wikipedia-(.+)$", lambda m: f"wiki:{m.group(1)}"),
]


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(enc.encode(text))


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs at double-newline boundaries.

    Strips whitespace from each paragraph and drops empty strings.
    """
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def split_paragraph_into_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into sentences using NLTK.

    Handles medical abbreviations (Dr., et al., Fig. 1, p<0.05) correctly.
    """
    return nltk.sent_tokenize(paragraph)


def _apply_overlap(chunks: list[dict], overlap_tokens: int) -> list[dict]:
    """Apply token overlap between adjacent chunks.

    For each pair of adjacent chunks, prepend the last overlap_tokens tokens
    of the previous chunk to the start of the next chunk. Recomputes
    token_count on the final text to avoid drift.
    """
    if len(chunks) <= 1 or overlap_tokens <= 0:
        return chunks

    result = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1]["text"]
        prev_token_ids = enc.encode(prev_text)

        # Extract the last overlap_tokens tokens from the previous chunk
        overlap_count = min(overlap_tokens, len(prev_token_ids))
        overlap_token_ids = prev_token_ids[-overlap_count:]
        overlap_text = enc.decode(overlap_token_ids).strip()

        # Prepend overlap text to the current chunk
        current_text = chunks[i]["text"]
        if overlap_text:
            merged_text = overlap_text + " " + current_text
        else:
            merged_text = current_text

        result.append({
            "text": merged_text,
            "token_count": count_tokens(merged_text),
        })

    return result


def chunk_document(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[dict]:
    """Chunk a document with paragraph-first, sentence-fallback strategy.

    Strategy:
    1. Split text into paragraphs at double-newline boundaries.
    2. Accumulate paragraphs into chunks while they fit within max_tokens.
    3. When a paragraph exceeds max_tokens, split it by sentences.
    4. If a single sentence exceeds max_tokens, include it as-is.
    5. Apply token overlap between adjacent chunks as post-processing.

    Returns a list of dicts with 'text' and 'token_count' keys.
    Token counts are computed on the final text (after overlap) to avoid drift.
    """
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return []

    chunks = []
    current_chunk_parts = []
    current_token_count = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if para_tokens <= max_tokens:
            # Paragraph fits -- try to add to current chunk
            if current_token_count + para_tokens <= max_tokens:
                current_chunk_parts.append(para)
                current_token_count += para_tokens
            else:
                # Flush current chunk and start new one
                if current_chunk_parts:
                    chunk_text = "\n\n".join(current_chunk_parts)
                    chunks.append({
                        "text": chunk_text,
                        "token_count": count_tokens(chunk_text),
                    })
                current_chunk_parts = [para]
                current_token_count = para_tokens
        else:
            # Paragraph too long -- flush current, then split by sentences
            if current_chunk_parts:
                chunk_text = "\n\n".join(current_chunk_parts)
                chunks.append({
                    "text": chunk_text,
                    "token_count": count_tokens(chunk_text),
                })
                current_chunk_parts = []
                current_token_count = 0

            sentences = split_paragraph_into_sentences(para)
            sent_parts = []
            sent_token_count = 0

            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if sent_token_count + sent_tokens <= max_tokens:
                    sent_parts.append(sent)
                    sent_token_count += sent_tokens
                else:
                    if sent_parts:
                        chunk_text = " ".join(sent_parts)
                        chunks.append({
                            "text": chunk_text,
                            "token_count": count_tokens(chunk_text),
                        })
                    # Start new sentence-level chunk with this sentence
                    sent_parts = [sent]
                    sent_token_count = sent_tokens

            # Remaining sentences become start of next chunk
            if sent_parts:
                current_chunk_parts = [" ".join(sent_parts)]
                current_token_count = sent_token_count

    # Flush remaining content
    if current_chunk_parts:
        chunk_text = "\n\n".join(current_chunk_parts)
        chunks.append({
            "text": chunk_text,
            "token_count": count_tokens(chunk_text),
        })

    # Apply overlap as post-processing step
    if overlap_tokens > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap_tokens)

    return chunks


def extract_metadata(filepath: Path) -> dict:
    """Extract source type and document ID from a corpus file path.

    Maps directory names to source types and extracts document IDs from
    filenames using regex patterns for each source format.

    Returns a dict with source_type, doc_id, and source_file keys.
    """
    source_dir = filepath.parent.name
    source_type = SOURCE_TYPE_MAP.get(source_dir, "unknown")

    stem = filepath.stem
    doc_id = stem  # fallback
    for pattern, extractor in DOC_ID_PATTERNS:
        match = re.search(pattern, stem)
        if match:
            doc_id = extractor(match)
            break

    return {
        "source_type": source_type,
        "doc_id": doc_id,
        "source_file": str(filepath),
    }
