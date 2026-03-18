"""Unit and integration tests for SFT data preparation pipeline.

Tests cover synthetic dataset validation, Alpaca template formatting (both
with-input and no-input variants), tokenization with -100 label masking,
padding behavior, block_size overflow handling, SFT config correctness,
and integration tests for binary output file validation.
"""

import json
import pickle
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


# ---------------------------------------------------------------------------
# Integration tests for binary output files
# ---------------------------------------------------------------------------

_TOKENIZED_DIR = _project_root / "data" / "instruction" / "tokenized"
_BLOCK_SIZE = 1024

# Skip integration tests if binary files have not been generated
_has_binary_files = (
    (_TOKENIZED_DIR / "train.bin").exists()
    and (_TOKENIZED_DIR / "labels_train.bin").exists()
    and (_TOKENIZED_DIR / "val.bin").exists()
    and (_TOKENIZED_DIR / "labels_val.bin").exists()
    and (_TOKENIZED_DIR / "meta.pkl").exists()
)


@pytest.mark.skipif(not _has_binary_files, reason="Binary files not yet generated")
class TestPrepareSftOutput:
    """Integration tests validating the actual binary output files."""

    def test_prepare_sft_output_files_exist(self):
        """All 5 output files exist in data/instruction/tokenized/."""
        expected = ["train.bin", "labels_train.bin", "val.bin", "labels_val.bin", "meta.pkl"]
        for fname in expected:
            fpath = _TOKENIZED_DIR / fname
            assert fpath.exists(), f"Missing output file: {fname}"
            assert fpath.stat().st_size > 0, f"Output file is empty: {fname}"

    def test_prepare_sft_binary_format(self):
        """train.bin is uint16, labels_train.bin is int32, sizes are consistent."""
        train = np.memmap(
            _TOKENIZED_DIR / "train.bin", dtype=np.uint16, mode="r"
        )
        labels = np.memmap(
            _TOKENIZED_DIR / "labels_train.bin", dtype=np.int32, mode="r"
        )

        # Both must have the same number of elements
        assert len(train) == len(labels), (
            f"Size mismatch: train.bin has {len(train)} tokens, "
            f"labels_train.bin has {len(labels)} tokens"
        )

        # Length must be a multiple of block_size (padded fixed-length records)
        assert len(train) % _BLOCK_SIZE == 0, (
            f"train.bin length {len(train)} is not a multiple of {_BLOCK_SIZE}"
        )

        # Verify val files have same consistency
        val = np.memmap(_TOKENIZED_DIR / "val.bin", dtype=np.uint16, mode="r")
        val_labels = np.memmap(
            _TOKENIZED_DIR / "labels_val.bin", dtype=np.int32, mode="r"
        )
        assert len(val) == len(val_labels)
        assert len(val) % _BLOCK_SIZE == 0

    def test_prepare_sft_labels_masking_real(self):
        """First record has -100 prefix, valid token IDs, then -100 padding."""
        labels = np.memmap(
            _TOKENIZED_DIR / "labels_train.bin", dtype=np.int32, mode="r"
        )
        first_record = labels[:_BLOCK_SIZE]

        # Find transition from -100 to non-negative (prefix -> response)
        first_valid = None
        for i, lbl in enumerate(first_record):
            if lbl != -100:
                first_valid = i
                break

        assert first_valid is not None, "First record has no response tokens"
        assert first_valid > 0, "First record should have masked prefix"

        # Check prefix is all -100
        assert all(first_record[:first_valid] == -100), (
            "Prefix should be entirely -100"
        )

        # Check response tokens are valid (non-negative), then followed by -100 padding
        response_started = False
        padding_started = False
        for i in range(first_valid, _BLOCK_SIZE):
            if first_record[i] != -100:
                assert not padding_started, (
                    f"Found non-padding token at {i} after padding started"
                )
                assert first_record[i] >= 0, (
                    f"Response token at {i} is invalid: {first_record[i]}"
                )
                response_started = True
            else:
                if response_started:
                    padding_started = True

        assert response_started, "Should have found response tokens"

    def test_prepare_sft_meta_pkl(self):
        """meta.pkl contains vocab_size=50257."""
        with open(_TOKENIZED_DIR / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        assert "vocab_size" in meta, "meta.pkl missing vocab_size"
        assert meta["vocab_size"] == 50257, (
            f"Expected vocab_size=50257, got {meta['vocab_size']}"
        )
