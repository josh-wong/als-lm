"""Unit tests for verify_sft.py and generate_sft_summary.py helper functions.

Tests cover Alpaca prompt wrapping, deterministic question selection across
8 ALS categories, binary coherence checking (min length, repetition, token
salad), verdict formatting, and results formatting. No Ollama server required.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root and scripts/ are on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from verify_sft import (
    check_coherence,
    format_results,
    format_verdict,
    select_spot_check_questions,
    wrap_alpaca,
)


# ---------------------------------------------------------------------------
# Alpaca wrapping
# ---------------------------------------------------------------------------

class TestWrapAlpaca:
    """Tests for Alpaca instruction format wrapping."""

    def test_basic_wrapping(self):
        """wrap_alpaca produces correct Alpaca format."""
        result = wrap_alpaca("What is ALS?")
        assert result == "### Instruction:\nWhat is ALS?\n\n### Response:\n"

    def test_preserves_question_text(self):
        """Wrapping preserves the original question text verbatim."""
        question = "What drugs are FDA-approved for treating ALS?"
        result = wrap_alpaca(question)
        assert question in result
        assert result.startswith("### Instruction:\n")
        assert result.endswith("### Response:\n")


# ---------------------------------------------------------------------------
# Question selection
# ---------------------------------------------------------------------------

class TestSelectSpotCheckQuestions:
    """Tests for deterministic question selection (one per category)."""

    @pytest.fixture
    def questions_path(self):
        return str(_project_root / "eval" / "questions.json")

    def test_returns_exactly_8(self, questions_path):
        """Selects exactly 8 questions (one per ALS category)."""
        selected = select_spot_check_questions(questions_path)
        assert len(selected) == 8

    def test_one_per_category(self, questions_path):
        """Each of the 8 categories is represented exactly once."""
        selected = select_spot_check_questions(questions_path)
        categories = [q["category"] for q in selected]
        expected = [
            "clinical_trials",
            "diagnostic_criteria",
            "disease_mechanisms",
            "drug_treatment",
            "epidemiology",
            "gene_mutation",
            "patient_care",
            "temporal_accuracy",
        ]
        assert sorted(categories) == expected

    def test_deterministic_with_seed(self, questions_path):
        """Same seed produces identical selection each time."""
        first = select_spot_check_questions(questions_path, seed=42)
        second = select_spot_check_questions(questions_path, seed=42)
        assert [q["id"] for q in first] == [q["id"] for q in second]

    def test_different_seed_different_selection(self, questions_path):
        """Different seeds produce different selections."""
        first = select_spot_check_questions(questions_path, seed=42)
        second = select_spot_check_questions(questions_path, seed=99)
        # At least one question should differ
        first_ids = [q["id"] for q in first]
        second_ids = [q["id"] for q in second]
        assert first_ids != second_ids


# ---------------------------------------------------------------------------
# Coherence checking
# ---------------------------------------------------------------------------

class TestCheckCoherence:
    """Tests for binary coherence heuristics (>20 char threshold)."""

    def test_empty_string_incoherent(self):
        """Empty string is incoherent."""
        assert check_coherence("") is False

    def test_short_string_incoherent(self):
        """String under 20 chars is incoherent."""
        assert check_coherence("A" * 5) is False
        assert check_coherence("Short text.") is False

    def test_exactly_20_chars_incoherent(self):
        """String of exactly 20 chars is incoherent (threshold is >20)."""
        assert check_coherence("A" * 20) is False

    def test_21_chars_coherent(self):
        """String of 21 chars is coherent (above threshold)."""
        assert check_coherence("A" * 21) is True

    def test_valid_als_response_coherent(self):
        """A real medical response is coherent."""
        text = "ALS is a progressive neurodegenerative disease affecting motor neurons."
        assert check_coherence(text) is True

    def test_repeated_word_6_times_incoherent(self):
        """Word repeated 6+ times in a row is incoherent."""
        text = "ALS ALS ALS ALS ALS ALS is a disease."
        assert check_coherence(text) is False

    def test_repeated_word_5_times_coherent(self):
        """Word repeated only 5 times is still coherent."""
        text = "ALS ALS ALS ALS ALS is a progressive disease affecting motor neurons."
        assert check_coherence(text) is True

    def test_token_salad_incoherent(self):
        """More than 80% non-alphanumeric is incoherent."""
        text = "!@#$%^&*()!@#$%^&*()!@#$%^&*()ab"
        assert check_coherence(text) is False

    def test_trigram_repetition_incoherent(self):
        """3-gram repeated 4+ times is incoherent."""
        text = "the motor neuron the motor neuron the motor neuron the motor neuron end"
        assert check_coherence(text) is False

    def test_punctuation_separated_repetition(self):
        """Comma-separated repetition is caught."""
        text = "TDP-43, TDP-43, TDP-43, TDP-43, TDP-43, TDP-43, is a protein."
        assert check_coherence(text) is False

    def test_whitespace_only_incoherent(self):
        """Whitespace-only string is incoherent."""
        assert check_coherence("   \n\t  ") is False

    def test_none_incoherent(self):
        """None input is incoherent."""
        assert check_coherence(None) is False


# ---------------------------------------------------------------------------
# Verdict formatting
# ---------------------------------------------------------------------------

class TestFormatVerdict:
    """Tests for overall pass/fail verdict formatting."""

    def test_7_of_8_passes(self):
        """7/8 coherent meets the threshold and returns PASS."""
        verdict = format_verdict(7, 8)
        assert verdict == "PASS"

    def test_8_of_8_passes(self):
        """8/8 coherent returns PASS."""
        verdict = format_verdict(8, 8)
        assert verdict == "PASS"

    def test_6_of_8_passes(self):
        """6/8 = 75% coherent, above 70% threshold, returns PASS."""
        verdict = format_verdict(6, 8)
        assert verdict == "PASS"

    def test_5_of_8_fails(self):
        """5/8 = 62.5% coherent, below 70% threshold, returns FAIL."""
        verdict = format_verdict(5, 8)
        assert verdict == "FAIL"

    def test_0_of_8_fails(self):
        """0/8 coherent returns FAIL."""
        verdict = format_verdict(0, 8)
        assert verdict == "FAIL"

    def test_7_of_10_passes(self):
        """7/10 coherent returns PASS (7/10 = 0.7 >= 0.7 threshold)."""
        verdict = format_verdict(7, 10)
        assert verdict == "PASS"

    def test_6_of_10_fails(self):
        """6/10 coherent returns FAIL."""
        verdict = format_verdict(6, 10)
        assert verdict == "FAIL"


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

class TestFormatResults:
    """Tests for structured results output."""

    def test_produces_structured_output(self):
        """format_results returns a list of dicts with required keys."""
        results = [
            {
                "question": "What is ALS?",
                "category": "disease_mechanisms",
                "response": "ALS is a progressive neurodegenerative disease.",
                "coherent": True,
            },
            {
                "question": "What drugs treat ALS?",
                "category": "drug_treatment",
                "response": "",
                "coherent": False,
            },
        ]
        formatted = format_results(results)
        assert len(formatted) == 2
        for item in formatted:
            assert "question" in item
            assert "category" in item
            assert "response_preview" in item
            assert "coherent" in item

    def test_truncates_long_response(self):
        """Response preview is truncated to 100 chars."""
        results = [
            {
                "question": "What is ALS?",
                "category": "disease_mechanisms",
                "response": "A" * 200,
                "coherent": True,
            },
        ]
        formatted = format_results(results)
        assert len(formatted[0]["response_preview"]) <= 103  # 100 + "..."

    def test_short_response_not_truncated(self):
        """Short responses are not truncated."""
        results = [
            {
                "question": "What is ALS?",
                "category": "disease_mechanisms",
                "response": "Short answer.",
                "coherent": True,
            },
        ]
        formatted = format_results(results)
        assert formatted[0]["response_preview"] == "Short answer."
