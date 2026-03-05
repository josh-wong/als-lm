"""Unit tests for rag.generate_rag pure functions."""

import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rag.generate_rag import compute_context_stats, compute_hit_rate


# ---------------------------------------------------------------------------
# compute_hit_rate
# ---------------------------------------------------------------------------

class TestComputeHitRate:
    def _q(self, qid, category="basic", key_facts=None):
        return {"id": qid, "category": category, "key_facts": key_facts or []}

    def _r(self, qid, chunks=None):
        entry = {"question_id": qid, "response": "text"}
        if chunks is not None:
            entry["retrieval"] = {"chunks": chunks}
        return entry

    def test_perfect_hit_rate(self):
        questions = [self._q("q1", key_facts=["riluzole"])]
        responses = [self._r("q1", chunks=["riluzole extends survival"])]
        rate, found, total, _ = compute_hit_rate(questions, responses)
        assert found == 1
        assert total == 1
        assert rate == 1.0

    def test_zero_hit_rate(self):
        questions = [self._q("q1", key_facts=["riluzole"])]
        responses = [self._r("q1", chunks=["the weather is nice"])]
        rate, found, total, _ = compute_hit_rate(questions, responses)
        assert found == 0
        assert total == 1
        assert rate == 0.0

    def test_no_retrieval_field(self):
        questions = [self._q("q1", key_facts=["fact"])]
        responses = [{"question_id": "q1", "response": "text"}]
        rate, found, total, per_q = compute_hit_rate(questions, responses)
        assert found == 0
        assert per_q[0]["hit_rate"] == 0.0

    def test_no_key_facts(self):
        questions = [self._q("q1", key_facts=[])]
        responses = [self._r("q1", chunks=["some chunk"])]
        rate, found, total, _ = compute_hit_rate(questions, responses)
        assert total == 0
        assert rate == 0.0

    def test_multiple_questions(self):
        questions = [
            self._q("q1", key_facts=["riluzole"]),
            self._q("q2", key_facts=["edaravone", "missing fact"]),
        ]
        responses = [
            self._r("q1", chunks=["riluzole is approved"]),
            self._r("q2", chunks=["edaravone was approved in 2017"]),
        ]
        rate, found, total, _ = compute_hit_rate(questions, responses)
        assert total == 3
        assert found == 2  # riluzole + edaravone found, missing fact not


# ---------------------------------------------------------------------------
# compute_context_stats
# ---------------------------------------------------------------------------

class TestComputeContextStats:
    def test_empty_responses(self):
        stats = compute_context_stats([])
        assert stats["total_responses"] == 0
        assert stats["mean_tokens"] == 0

    def test_no_retrieval_field(self):
        stats = compute_context_stats([{"response": "text"}])
        assert stats["total_responses"] == 0

    def test_basic_stats(self):
        responses = [
            {"retrieval": {"augmented_prompt_token_count": 100}},
            {"retrieval": {"augmented_prompt_token_count": 200}},
            {"retrieval": {"augmented_prompt_token_count": 300}},
        ]
        stats = compute_context_stats(responses)
        assert stats["total_responses"] == 3
        assert stats["mean_tokens"] == 200
        assert stats["min_tokens"] == 100
        assert stats["max_tokens"] == 300
        assert stats["median_tokens"] == 200
