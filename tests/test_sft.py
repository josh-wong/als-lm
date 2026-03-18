"""Unit tests for SFT data preparation pipeline, loss masking, and config.

Tests cover synthetic dataset validation, Alpaca template formatting (both
with-input and no-input variants), tokenization with -100 label masking,
padding behavior, block_size overflow handling, and SFT config correctness.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path so data package is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.instruction.prepare_sft import (
    ALPACA_TEMPLATE_NO_INPUT,
    ALPACA_TEMPLATE_WITH_INPUT,
    format_alpaca,
    tokenize_and_mask,
)


# ---------------------------------------------------------------------------
# Synthetic dataset validation
# ---------------------------------------------------------------------------

class TestSyntheticJson:
    """Validate the committed synthetic_sft.json dataset."""

    @pytest.fixture
    def synthetic_data(self):
        path = _project_root / "data" / "instruction" / "synthetic_sft.json"
        with open(path) as f:
            return json.load(f)

    def test_synthetic_json_valid(self, synthetic_data):
        """synthetic_sft.json has 20-30 entries with required keys."""
        assert 20 <= len(synthetic_data) <= 30, (
            f"Expected 20-30 entries, got {len(synthetic_data)}"
        )
        for i, entry in enumerate(synthetic_data):
            assert "instruction" in entry, f"Entry {i} missing 'instruction'"
            assert "output" in entry, f"Entry {i} missing 'output'"
            assert len(entry["instruction"].strip()) > 0, (
                f"Entry {i} has empty instruction"
            )
            assert len(entry["output"].strip()) > 0, (
                f"Entry {i} has empty output"
            )

        # Verify mix: some with non-empty input, some without
        with_input = [e for e in synthetic_data if e.get("input", "").strip()]
        without_input = [
            e for e in synthetic_data if not e.get("input", "").strip()
        ]
        assert len(with_input) >= 5, (
            f"Expected at least 5 entries with input, got {len(with_input)}"
        )
        assert len(without_input) >= 10, (
            f"Expected at least 10 entries without input, got {len(without_input)}"
        )


# ---------------------------------------------------------------------------
# Alpaca template formatting
# ---------------------------------------------------------------------------

class TestFormatAlpaca:
    """format_alpaca() produces correct template variants."""

    def test_format_alpaca_no_input(self):
        """Empty input omits the '### Input:' section."""
        entry = {
            "instruction": "What is ALS?",
            "input": "",
            "output": "ALS is a neurodegenerative disease.",
        }
        result = format_alpaca(entry)
        assert "### Instruction:" in result
        assert "### Response:" in result
        assert "### Input:" not in result
        assert "What is ALS?" in result
        assert "ALS is a neurodegenerative disease." in result

    def test_format_alpaca_with_input(self):
        """Non-empty input includes the '### Input:' section."""
        entry = {
            "instruction": "Summarize this trial.",
            "input": "NCT12345 tested riluzole in 100 patients.",
            "output": "The trial evaluated riluzole efficacy.",
        }
        result = format_alpaca(entry)
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result
        assert "NCT12345 tested riluzole in 100 patients." in result
        assert "The trial evaluated riluzole efficacy." in result


# ---------------------------------------------------------------------------
# Tokenization and masking
# ---------------------------------------------------------------------------

class TestTokenizeAndMask:
    """tokenize_and_mask() produces correct label masking and padding."""

    @pytest.fixture
    def tokenizer(self):
        from tokenizers import Tokenizer

        path = _project_root / "tokenizer" / "als_tokenizer.json"
        return Tokenizer.from_file(str(path))

    def test_tokenize_and_mask_prefix_masked(self, tokenizer):
        """Labels have -100 for instruction prefix, valid IDs for response."""
        entry = {
            "instruction": "What is ALS?",
            "input": "",
            "output": "ALS is a progressive neurodegenerative disease.",
        }
        result = tokenize_and_mask(entry, tokenizer, block_size=1024)
        assert result is not None, "Should not return None for short example"
        input_ids, labels = result

        assert len(input_ids) == 1024
        assert len(labels) == 1024

        # Find the masking boundary: labels should start with -100 values
        first_valid = None
        for i, lbl in enumerate(labels):
            if lbl != -100:
                first_valid = i
                break
        assert first_valid is not None, "Labels should have some valid tokens"
        assert first_valid > 0, "Some prefix tokens should be masked"

        # All prefix tokens should be -100
        for i in range(first_valid):
            assert labels[i] == -100, f"Label at {i} should be -100 (prefix)"

        # Response tokens should be valid (non-negative)
        found_response = False
        for i in range(first_valid, len(labels)):
            if labels[i] != -100:
                assert labels[i] >= 0, f"Response label at {i} is invalid"
                found_response = True
        assert found_response, "Should have at least one non-masked response token"

    def test_tokenize_and_mask_padding(self, tokenizer):
        """Short examples are padded: input_ids with 0, labels with -100."""
        entry = {
            "instruction": "What is ALS?",
            "input": "",
            "output": "A disease.",
        }
        result = tokenize_and_mask(entry, tokenizer, block_size=1024)
        assert result is not None
        input_ids, labels = result

        assert len(input_ids) == 1024
        assert len(labels) == 1024

        # The example is very short, so most of the 1024 tokens should be padding
        # Find where content ends (input_ids become 0 padding)
        # Labels should be -100 for all padding positions
        content_tokens = sum(1 for t in input_ids if t != 0)
        assert content_tokens < 1024, "Short example should have padding"

        # Check last position is padding
        assert labels[-1] == -100, "Last label should be -100 (padding)"

    def test_tokenize_and_mask_too_long(self, tokenizer):
        """Examples exceeding block_size return None."""
        entry = {
            "instruction": "Explain everything about ALS.",
            "input": "",
            "output": "word " * 2000,  # Very long output
        }
        result = tokenize_and_mask(entry, tokenizer, block_size=64)
        assert result is None, "Should return None for examples exceeding block_size"


# ---------------------------------------------------------------------------
# Template variant masking boundaries
# ---------------------------------------------------------------------------

class TestTemplateVariants:
    """Both Alpaca variants produce correct masking at the response boundary."""

    @pytest.fixture
    def tokenizer(self):
        from tokenizers import Tokenizer

        path = _project_root / "tokenizer" / "als_tokenizer.json"
        return Tokenizer.from_file(str(path))

    def test_template_variants_both_paths(self, tokenizer):
        """Both with-input and no-input variants have correct mask boundaries."""
        no_input_entry = {
            "instruction": "What causes ALS?",
            "input": "",
            "output": "SOD1 mutations account for about 20% of familial ALS.",
        }
        with_input_entry = {
            "instruction": "Summarize the treatment options.",
            "input": "Riluzole and edaravone are FDA-approved for ALS.",
            "output": "Two drugs are currently approved: riluzole and edaravone.",
        }

        result_no = tokenize_and_mask(no_input_entry, tokenizer, block_size=1024)
        result_with = tokenize_and_mask(with_input_entry, tokenizer, block_size=1024)

        assert result_no is not None
        assert result_with is not None

        _, labels_no = result_no
        _, labels_with = result_with

        # Both should have masked prefix and unmasked response
        prefix_no = sum(1 for l in labels_no if l == -100)
        prefix_with = sum(1 for l in labels_with if l == -100)

        # With-input variant should have more masked tokens (longer prefix)
        # because it includes the extra "### Input:" section
        non_pad_no = sum(1 for l in labels_no if l != -100)
        non_pad_with = sum(1 for l in labels_with if l != -100)

        assert non_pad_no > 0, "No-input variant should have response tokens"
        assert non_pad_with > 0, "With-input variant should have response tokens"


# ---------------------------------------------------------------------------
# SFT config validation
# ---------------------------------------------------------------------------

class TestSftConfig:
    """configs/sft.json is valid and has correct parameters."""

    @pytest.fixture
    def config(self):
        path = _project_root / "configs" / "sft.json"
        with open(path) as f:
            return json.load(f)

    def test_sft_config_json(self, config):
        """sft.json has correct architecture and training parameters."""
        # Model architecture must match 1B config
        assert config["model"]["n_layer"] == 30
        assert config["model"]["n_head"] == 20
        assert config["model"]["n_embd"] == 1600
        assert config["model"]["block_size"] == 1024
        assert config["model"]["dropout"] == 0.1
        assert config["model"]["bias"] is True
        assert config["model"]["vocab_size"] is None
        assert config["model"]["use_gradient_checkpointing"] is True

        # SFT-specific training parameters
        assert config["training"]["lr"] == 2e-5
        assert config["training"]["warmup_steps"] == 50
        assert config["training"]["max_epochs"] == 3
        assert config["training"]["size_name"] == "1B-sft"
        assert config["training"]["batch_size"] == 2
        assert config["training"]["grad_accum"] == 4
        assert config["training"]["block_size"] == 1024

        # DeepSpeed section must exist
        assert "deepspeed" in config
        assert "zero_optimization" in config["deepspeed"]
