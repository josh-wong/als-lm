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
from generate_sft_summary import (
    generate_summary,
    parse_sft_log,
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


# ===========================================================================
# Tests for generate_sft_summary.py
# ===========================================================================


def _make_sft_log_lines(
    num_steps=750,
    eval_interval=250,
    epochs=3,
    early_stop_epoch=None,
):
    """Create synthetic SFT JSONL lines mimicking train.py output.

    Generates step entries with train_loss decreasing over time and
    val_loss entries at eval boundaries. Epoch boundaries are marked.

    Args:
        num_steps: Total training steps.
        eval_interval: Steps between validation evaluations.
        epochs: Number of epochs.
        early_stop_epoch: If set, insert early stopping at this epoch.

    Returns:
        List of JSON-serializable dicts (one per JSONL line).
    """
    lines = []
    steps_per_epoch = num_steps // epochs
    base_train_loss = 5.2
    base_val_loss = 5.5

    for step in range(1, num_steps + 1):
        epoch = (step - 1) // steps_per_epoch
        progress = step / num_steps

        entry = {
            "step": step,
            "train_loss": round(base_train_loss - 2.0 * progress + 0.1 * (step % 7), 4),
            "val_loss": None,
            "lr": 2e-5 * min(1.0, step / 50),
            "epoch": epoch,
            "tokens_per_sec": 1200 + step % 100,
            "gpu_mem_mb": 5800 + step % 200,
        }

        # Mark epoch boundaries
        if step > 1 and step % steps_per_epoch == 0:
            entry["sft_epoch_boundary"] = True
            entry["val_loss"] = round(base_val_loss - 1.5 * progress, 4)
            entry["sft_best_val_loss"] = round(
                min(base_val_loss - 1.5 * (e / epochs)
                    for e in range(1, epoch + 2)),
                4,
            )

            if early_stop_epoch is not None and epoch + 1 >= early_stop_epoch:
                entry["sft_early_stop"] = True

        # Val loss at eval intervals too
        if step % eval_interval == 0 and entry["val_loss"] is None:
            entry["val_loss"] = round(base_val_loss - 1.5 * progress, 4)

        lines.append(entry)

    return lines


def _write_jsonl(lines, path):
    """Write a list of dicts as JSONL to a file."""
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# parse_sft_log tests
# ---------------------------------------------------------------------------

class TestParseSftLog:
    """Tests for JSONL log parsing and metric extraction."""

    def test_extracts_epoch_boundaries(self):
        """parse_sft_log finds epoch boundary markers in the log."""
        lines = _make_sft_log_lines(num_steps=750, epochs=3)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            _write_jsonl(lines, f.name)
            tmp_path = f.name

        try:
            result = parse_sft_log(tmp_path)
            assert result["epochs_completed"] == 3
        finally:
            os.unlink(tmp_path)

    def test_extracts_best_val_loss(self):
        """parse_sft_log returns the best validation loss from sft_best_val_loss."""
        lines = _make_sft_log_lines(num_steps=750, epochs=3)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            _write_jsonl(lines, f.name)
            tmp_path = f.name

        try:
            result = parse_sft_log(tmp_path)
            assert result["best_val_loss"] is not None
            assert isinstance(result["best_val_loss"], float)
        finally:
            os.unlink(tmp_path)

    def test_extracts_final_step(self):
        """parse_sft_log returns the total steps from the log."""
        lines = _make_sft_log_lines(num_steps=750, epochs=3)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            _write_jsonl(lines, f.name)
            tmp_path = f.name

        try:
            result = parse_sft_log(tmp_path)
            assert result["total_steps"] == 750
        finally:
            os.unlink(tmp_path)

    def test_early_stopped_log(self):
        """parse_sft_log detects early stopping in the log."""
        lines = _make_sft_log_lines(
            num_steps=750, epochs=3, early_stop_epoch=2,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            _write_jsonl(lines, f.name)
            tmp_path = f.name

        try:
            result = parse_sft_log(tmp_path)
            assert result["early_stopped"] is True
        finally:
            os.unlink(tmp_path)

    def test_full_3_epochs_no_early_stop(self):
        """parse_sft_log returns early_stopped=False for full 3-epoch run."""
        lines = _make_sft_log_lines(num_steps=750, epochs=3)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            _write_jsonl(lines, f.name)
            tmp_path = f.name

        try:
            result = parse_sft_log(tmp_path)
            assert result["early_stopped"] is False
        finally:
            os.unlink(tmp_path)

    def test_extracts_tokens_per_sec(self):
        """parse_sft_log computes average tokens/sec."""
        lines = _make_sft_log_lines(num_steps=100, epochs=1)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            _write_jsonl(lines, f.name)
            tmp_path = f.name

        try:
            result = parse_sft_log(tmp_path)
            assert result["avg_tokens_per_sec"] > 0
        finally:
            os.unlink(tmp_path)

    def test_extracts_peak_gpu_memory(self):
        """parse_sft_log returns peak GPU memory in MB."""
        lines = _make_sft_log_lines(num_steps=100, epochs=1)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False,
        ) as f:
            _write_jsonl(lines, f.name)
            tmp_path = f.name

        try:
            result = parse_sft_log(tmp_path)
            assert result["peak_gpu_mem_mb"] > 0
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# generate_summary tests
# ---------------------------------------------------------------------------

class TestGenerateSummary:
    """Tests for SFT summary Markdown report generation."""

    def _make_parsed_log(self, early_stopped=False):
        """Create a minimal parsed log result for testing."""
        return {
            "total_steps": 750,
            "epochs_completed": 3 if not early_stopped else 2,
            "best_val_loss": 4.1234,
            "best_epoch": 2,
            "early_stopped": early_stopped,
            "stopped_epoch": 2 if early_stopped else None,
            "avg_tokens_per_sec": 1250.5,
            "peak_gpu_mem_mb": 5998,
            "final_train_loss": 3.2,
            "val_losses": [5.0, 4.5, 4.1234],
        }

    def _make_verify_results(self):
        """Create a minimal verify_results.json content."""
        return {
            "summary": {
                "coherent_count": 7,
                "total_count": 8,
                "verdict": "PASS",
            },
        }

    def test_produces_valid_markdown(self):
        """generate_summary returns a string containing Markdown headings."""
        parsed = self._make_parsed_log()
        md = generate_summary(parsed, verify_results=None)
        assert "# SFT training summary" in md
        assert "## Training configuration" in md
        assert "## Early stopping" in md
        assert "## Base model comparison" in md
        assert "## Training performance" in md

    def test_includes_training_config_table(self):
        """Summary contains a Markdown table for training configuration."""
        parsed = self._make_parsed_log()
        md = generate_summary(parsed, verify_results=None)
        # Check table structure markers
        assert "| Parameter" in md
        assert "| Value" in md or "| Value |" in md
        assert "lr" in md.lower() or "learning rate" in md.lower()

    def test_includes_early_stopping_section(self):
        """Summary documents early stopping behavior."""
        parsed = self._make_parsed_log(early_stopped=True)
        md = generate_summary(parsed, verify_results=None)
        assert "early stop" in md.lower() or "Early stopping" in md

    def test_full_training_no_early_stop(self):
        """Summary indicates all 3 epochs completed when no early stop."""
        parsed = self._make_parsed_log(early_stopped=False)
        md = generate_summary(parsed, verify_results=None)
        assert "3" in md  # Should mention completing all 3 epochs

    def test_includes_base_model_comparison(self):
        """Summary compares SFT val loss with base model val loss (5.6424)."""
        parsed = self._make_parsed_log()
        md = generate_summary(parsed, verify_results=None)
        assert "5.6424" in md
        assert "not" in md.lower() and "comparable" in md.lower()

    def test_includes_verify_results_when_present(self):
        """Summary includes qualitative verification when results exist."""
        parsed = self._make_parsed_log()
        verify = self._make_verify_results()
        md = generate_summary(parsed, verify_results=verify)
        assert "## Qualitative verification" in md
        assert "7/8" in md or "7 / 8" in md
        assert "PASS" in md

    def test_shows_not_run_when_no_verify_results(self):
        """Summary shows 'Not yet run' when verify results are absent."""
        parsed = self._make_parsed_log()
        md = generate_summary(parsed, verify_results=None)
        assert "Not yet run" in md or "N/A" in md

    def test_includes_training_performance(self):
        """Summary includes tokens/sec and peak GPU memory."""
        parsed = self._make_parsed_log()
        md = generate_summary(parsed, verify_results=None)
        assert "1250" in md or "1,250" in md  # tokens/sec
        assert "5998" in md or "5,998" in md  # peak GPU MB
