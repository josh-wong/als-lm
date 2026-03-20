"""Unit tests for instruction format wrapping and instruct Modelfile support.

Tests cover Alpaca prompt wrapping in eval/generate_responses.py, raw payload
generation for Ollama, --instruction-format flag passthrough in
run_evaluation.py, and the instruct Modelfile template. No Ollama server
required.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Instruct Modelfile template tests (Task 1)
# ---------------------------------------------------------------------------

class TestInstructModelfileTemplate:
    """Tests for export/Modelfile.template.instruct."""

    def test_template_contains_template_directive(self):
        """Modelfile.template.instruct contains TEMPLATE directive."""
        template_path = _project_root / "export" / "Modelfile.template.instruct"
        text = template_path.read_text()
        assert "TEMPLATE" in text

    def test_template_contains_go_template_variable(self):
        """Modelfile.template.instruct contains {{ .Prompt }} Go template variable."""
        template_path = _project_root / "export" / "Modelfile.template.instruct"
        text = template_path.read_text()
        assert "{{ .Prompt }}" in text or "{{.Prompt}}" in text

    def test_template_contains_gguf_path_placeholder(self):
        """Modelfile.template.instruct contains {{GGUF_PATH}} placeholder."""
        template_path = _project_root / "export" / "Modelfile.template.instruct"
        text = template_path.read_text()
        assert "{{GGUF_PATH}}" in text

    def test_template_contains_disclaimer_placeholder(self):
        """Modelfile.template.instruct contains {{DISCLAIMER}} placeholder."""
        template_path = _project_root / "export" / "Modelfile.template.instruct"
        text = template_path.read_text()
        assert "{{DISCLAIMER}}" in text


# ---------------------------------------------------------------------------
# Export pipeline 1b-instruct disclaimer tests (Task 1)
# ---------------------------------------------------------------------------

class TestInstructDisclaimer:
    """Tests for 1b-instruct disclaimer in export_pipeline.py."""

    def test_disclaimers_contains_1b_instruct(self):
        """DISCLAIMERS dict contains '1b-instruct' key."""
        from export.export_pipeline import DISCLAIMERS
        assert "1b-instruct" in DISCLAIMERS

    def test_1b_instruct_disclaimer_contains_failure_language(self):
        """_get_disclaimer('1b-instruct') returns disclaimer with failure language."""
        from export.export_pipeline import _get_disclaimer
        disclaimer = _get_disclaimer("1b-instruct")
        assert "failed" in disclaimer.lower() or "degenerate" in disclaimer.lower()

    def test_1b_instruct_disclaimer_contains_medical_warning(self):
        """1b-instruct disclaimer includes medical decision-making warning."""
        from export.export_pipeline import _get_disclaimer
        disclaimer = _get_disclaimer("1b-instruct")
        assert "medical" in disclaimer.lower()

    def test_1b_instruct_disclaimer_mentions_research_artifact(self):
        """1b-instruct disclaimer identifies model as research artifact."""
        from export.export_pipeline import _get_disclaimer
        disclaimer = _get_disclaimer("1b-instruct")
        assert "research" in disclaimer.lower()


# ---------------------------------------------------------------------------
# Instruction format wrapping tests (Task 2)
# ---------------------------------------------------------------------------

class TestWrapInstructionFormat:
    """Tests for wrap_instruction_format in eval/generate_responses.py."""

    def test_basic_wrapping(self):
        """wrap_instruction_format produces correct Alpaca format."""
        from eval.generate_responses import wrap_instruction_format
        result = wrap_instruction_format("What is ALS?")
        assert result == "### Instruction:\nWhat is ALS?\n\n### Response:\n"

    def test_multiline_prompt(self):
        """wrap_instruction_format handles multi-line prompts."""
        from eval.generate_responses import wrap_instruction_format
        prompt = "What is ALS?\nDescribe the symptoms."
        result = wrap_instruction_format(prompt)
        assert result == "### Instruction:\nWhat is ALS?\nDescribe the symptoms.\n\n### Response:\n"

    def test_empty_string(self):
        """wrap_instruction_format handles empty string."""
        from eval.generate_responses import wrap_instruction_format
        result = wrap_instruction_format("")
        assert result == "### Instruction:\n\n\n### Response:\n"

    def test_matches_verify_sft_pattern(self):
        """wrap_instruction_format matches verify_sft.py wrap_alpaca() output."""
        from eval.generate_responses import wrap_instruction_format
        from scripts.verify_sft import wrap_alpaca
        test_questions = [
            "What is ALS?",
            "What drugs treat ALS?",
            "Describe the SOD1 gene mutation.",
        ]
        for q in test_questions:
            assert wrap_instruction_format(q) == wrap_alpaca(q), (
                f"Mismatch for question: {q}"
            )


# ---------------------------------------------------------------------------
# Ollama payload tests with instruction format (Task 2)
# ---------------------------------------------------------------------------

class TestOllamaInstructionFormat:
    """Tests for raw payload and prompt wrapping in Ollama generation."""

    @patch("eval.generate_responses.requests.post")
    def test_instruction_format_adds_raw_true(self, mock_post):
        """When instruction_format=True, Ollama payload includes raw: True."""
        import json
        from eval.generate_responses import generate_ollama_response

        # Configure mock to return a valid response
        mock_response = type("Response", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"response": "test", "eval_count": 5},
        })()
        mock_post.return_value = mock_response

        generate_ollama_response(
            "http://localhost:11434", "test-model", "What is ALS?",
            max_tokens=256, temperature=0.0, instruction_format=True,
        )

        # Check the payload passed to requests.post
        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert payload.get("raw") is True

    @patch("eval.generate_responses.requests.post")
    def test_instruction_format_wraps_prompt(self, mock_post):
        """When instruction_format=True, prompt is wrapped in Alpaca format."""
        from eval.generate_responses import generate_ollama_response

        mock_response = type("Response", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"response": "test", "eval_count": 5},
        })()
        mock_post.return_value = mock_response

        generate_ollama_response(
            "http://localhost:11434", "test-model", "What is ALS?",
            max_tokens=256, temperature=0.0, instruction_format=True,
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert payload["prompt"] == "### Instruction:\nWhat is ALS?\n\n### Response:\n"

    @patch("eval.generate_responses.requests.post")
    def test_no_instruction_format_no_raw(self, mock_post):
        """When instruction_format=False, no raw key in payload."""
        from eval.generate_responses import generate_ollama_response

        mock_response = type("Response", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"response": "test", "eval_count": 5},
        })()
        mock_post.return_value = mock_response

        generate_ollama_response(
            "http://localhost:11434", "test-model", "What is ALS?",
            max_tokens=256, temperature=0.0,
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert "raw" not in payload

    @patch("eval.generate_responses.requests.post")
    def test_no_instruction_format_unchanged_prompt(self, mock_post):
        """When instruction_format=False, prompt is sent unchanged."""
        from eval.generate_responses import generate_ollama_response

        mock_response = type("Response", (), {
            "status_code": 200,
            "raise_for_status": lambda self: None,
            "json": lambda self: {"response": "test", "eval_count": 5},
        })()
        mock_post.return_value = mock_response

        generate_ollama_response(
            "http://localhost:11434", "test-model", "What is ALS?",
            max_tokens=256, temperature=0.0,
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert payload["prompt"] == "What is ALS?"


# ---------------------------------------------------------------------------
# run_evaluation.py passthrough tests (Task 2)
# ---------------------------------------------------------------------------

class TestRunEvaluationPassthrough:
    """Tests for --instruction-format flag passthrough in run_evaluation.py."""

    def test_build_stage_args_includes_instruction_format(self):
        """build_stage_args appends --instruction-format for generate stage."""
        from eval.run_evaluation import build_stage_args, STAGES

        generate_stage = [s for s in STAGES if s["name"] == "generate"][0]
        cmd = build_stage_args(
            generate_stage,
            results_dir="/tmp/results",
            reports_dir="/tmp/reports",
            checkpoint_id="test",
            benchmark="/tmp/bench.json",
            registry="/tmp/registry.json",
            ollama_model="test-model",
            ollama_url="http://localhost:11434",
            instruction_format=True,
        )
        assert "--instruction-format" in cmd

    def test_build_stage_args_omits_instruction_format_when_false(self):
        """build_stage_args does not append --instruction-format when False."""
        from eval.run_evaluation import build_stage_args, STAGES

        generate_stage = [s for s in STAGES if s["name"] == "generate"][0]
        cmd = build_stage_args(
            generate_stage,
            results_dir="/tmp/results",
            reports_dir="/tmp/reports",
            checkpoint_id="test",
            benchmark="/tmp/bench.json",
            registry="/tmp/registry.json",
            ollama_model="test-model",
            ollama_url="http://localhost:11434",
            instruction_format=False,
        )
        assert "--instruction-format" not in cmd

    def test_build_stage_args_non_generate_ignores_instruction_format(self):
        """build_stage_args ignores instruction_format for non-generate stages."""
        from eval.run_evaluation import build_stage_args, STAGES

        score_stage = [s for s in STAGES if s["name"] == "score"][0]
        cmd = build_stage_args(
            score_stage,
            results_dir="/tmp/results",
            reports_dir="/tmp/reports",
            checkpoint_id="test",
            benchmark="/tmp/bench.json",
            registry="/tmp/registry.json",
            ollama_model="test-model",
            instruction_format=True,
        )
        assert "--instruction-format" not in cmd
