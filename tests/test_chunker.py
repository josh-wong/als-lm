"""Unit tests for rag.chunker pure functions."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so rag package is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag.chunker import (
    chunk_document,
    count_tokens,
    extract_metadata,
    split_into_paragraphs,
)


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        assert count_tokens("hello") >= 1

    def test_longer_string(self):
        tokens = count_tokens("The quick brown fox jumps over the lazy dog.")
        assert tokens > 0


# ---------------------------------------------------------------------------
# split_into_paragraphs
# ---------------------------------------------------------------------------

class TestSplitIntoParagraphs:
    def test_single_paragraph(self):
        assert split_into_paragraphs("hello world") == ["hello world"]

    def test_double_newline_split(self):
        result = split_into_paragraphs("para one\n\npara two")
        assert result == ["para one", "para two"]

    def test_strips_whitespace(self):
        result = split_into_paragraphs("  a  \n\n  b  ")
        assert result == ["a", "b"]

    def test_empty_string(self):
        assert split_into_paragraphs("") == []

    def test_only_whitespace(self):
        assert split_into_paragraphs("   \n\n   ") == []


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def test_empty_text(self):
        assert chunk_document("") == []

    def test_short_text_single_chunk(self):
        text = "Short paragraph."
        chunks = chunk_document(text, max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["token_count"] > 0

    def test_chunks_respect_max_tokens(self):
        # Use paragraph-separated text so the chunker can split
        paragraphs = ["This is paragraph number %d with enough words." % i
                       for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_document(text, max_tokens=30, overlap_tokens=0)
        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk should stay near the limit
            assert chunk["token_count"] <= 30 + 15

    def test_overlap_increases_token_count(self):
        text = "First paragraph content here.\n\nSecond paragraph content here."
        no_overlap = chunk_document(text, max_tokens=10, overlap_tokens=0)
        with_overlap = chunk_document(text, max_tokens=10, overlap_tokens=3)
        # With overlap, later chunks should have more tokens
        if len(with_overlap) > 1:
            assert with_overlap[1]["token_count"] >= no_overlap[1]["token_count"]

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            chunk_document("text", max_tokens=0)

    def test_negative_overlap(self):
        with pytest.raises(ValueError, match="overlap_tokens must be non-negative"):
            chunk_document("text", max_tokens=100, overlap_tokens=-1)

    def test_overlap_exceeds_max(self):
        with pytest.raises(ValueError, match="overlap_tokens.*must be less than"):
            chunk_document("text", max_tokens=50, overlap_tokens=50)

    def test_each_chunk_has_required_keys(self):
        text = "Hello world.\n\nAnother paragraph.\n\nThird paragraph."
        chunks = chunk_document(text, max_tokens=500)
        for chunk in chunks:
            assert "text" in chunk
            assert "token_count" in chunk
            assert isinstance(chunk["text"], str)
            assert isinstance(chunk["token_count"], int)


# ---------------------------------------------------------------------------
# extract_metadata
# ---------------------------------------------------------------------------

class TestExtractMetadata:
    def test_pubmed_file(self):
        p = Path("/data/raw/pubmed_abstracts/pubmed-PMC12345.txt")
        meta = extract_metadata(p)
        assert meta["source_type"] == "pubmed"
        assert meta["doc_id"] == "PMC12345"
        assert meta["source_file"] == str(p)

    def test_clinical_trial_file(self):
        p = Path("/data/raw/clinical_trials/clinicaltrials-NCT00123456.txt")
        meta = extract_metadata(p)
        assert meta["source_type"] == "trial"
        assert meta["doc_id"] == "NCT00123456"

    def test_wikipedia_file(self):
        p = Path("/data/raw/wikipedia/wikipedia-Amyotrophic_lateral_sclerosis.txt")
        meta = extract_metadata(p)
        assert meta["source_type"] == "educational"
        assert meta["doc_id"] == "wiki:Amyotrophic_lateral_sclerosis"

    def test_unknown_source_type(self):
        p = Path("/data/raw/mystery_dir/some_file.txt")
        meta = extract_metadata(p)
        assert meta["source_type"] == "unknown"

    def test_bookshelf_file(self):
        p = Path("/data/raw/educational/bookshelf-12345-chapter1.txt")
        meta = extract_metadata(p)
        assert meta["source_type"] == "educational"
        assert meta["doc_id"] == "NBK12345-chapter1"
