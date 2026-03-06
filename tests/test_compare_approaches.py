"""Unit tests for rag.compare_approaches metric and classification functions."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("rapidfuzz", reason="rapidfuzz required for failure classification tests")

# Ensure project root is on sys.path so rag package is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag.compare_approaches import classify_failure


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------

class TestClassifyFailure:
    """Tests for the retrieval-vs-generation failure classifier."""

    def test_empty_chunks_is_retrieval_failure(self):
        assert classify_failure(["some fact"], []) == "retrieval_failure"

    def test_empty_key_facts_is_retrieval_failure(self):
        assert classify_failure([], ["chunk text"]) == "retrieval_failure"

    def test_fact_present_is_generation_failure(self):
        facts = ["riluzole extends survival"]
        chunks = ["Studies show that riluzole extends survival by 2-3 months."]
        assert classify_failure(facts, chunks) == "generation_failure"

    def test_fact_absent_is_retrieval_failure(self):
        facts = ["riluzole extends survival"]
        chunks = ["The weather in London is rainy."]
        assert classify_failure(facts, chunks) == "retrieval_failure"

    def test_case_insensitive(self):
        facts = ["RILUZOLE"]
        chunks = ["riluzole is an approved drug"]
        assert classify_failure(facts, chunks) == "generation_failure"

    def test_threshold_boundary(self):
        # With a very high threshold, a partial match should be classified
        # as retrieval failure
        facts = ["riluzole extends survival by three months"]
        chunks = ["riluzole is a drug"]
        result = classify_failure(facts, chunks, threshold=99)
        assert result == "retrieval_failure"

    def test_multiple_facts_one_match_is_generation(self):
        facts = ["fact that is nowhere", "riluzole extends survival"]
        chunks = ["riluzole extends survival in ALS patients"]
        assert classify_failure(facts, chunks) == "generation_failure"

    def test_both_none_like(self):
        assert classify_failure([], []) == "retrieval_failure"
