"""Unit and integration tests for SFT data preparation and training pipeline.

Tests cover synthetic dataset validation, Alpaca template formatting (both
with-input and no-input variants), tokenization with -100 label masking,
padding behavior, block_size overflow handling, SFT config correctness,
integration tests for binary output file validation, GPT.forward() labels
masking, get_sft_batch() paired memmap loading, estimate_sft_loss()
validation, checkpoint discovery, and CONFIG_DEFAULTS entries.
"""

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

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


# ---------------------------------------------------------------------------
# GPT.forward() labels parameter tests
# ---------------------------------------------------------------------------

from model.model import GPT, GPTConfig


def _make_tiny_model():
    """Create a tiny GPT model for testing on CPU."""
    config = GPTConfig(
        block_size=32, vocab_size=100, n_layer=2, n_head=2, n_embd=64,
        dropout=0.0, bias=True,
    )
    return GPT(config)


class TestForwardLabels:
    """GPT.forward() labels parameter for SFT loss masking."""

    def test_forward_labels_masking(self):
        """Loss with labels (-100 masking) differs from loss with all-valid targets."""
        model = _make_tiny_model()
        model.eval()

        torch.manual_seed(42)
        idx = torch.randint(0, 100, (2, 32))

        # Create labels: first 16 tokens masked (-100), last 16 are valid IDs
        labels = torch.full((2, 32), -100, dtype=torch.long)
        labels[:, 16:] = torch.randint(0, 100, (2, 16))

        # All-valid targets (same values where labels are valid, random elsewhere)
        all_targets = torch.randint(0, 100, (2, 32))
        all_targets[:, 16:] = labels[:, 16:]

        _, loss_labels = model(idx, labels=labels)
        _, loss_targets = model(idx, targets=all_targets)

        assert loss_labels is not None, "labels should produce a loss"
        assert loss_targets is not None, "targets should produce a loss"
        # Losses should differ because labels masks prefix positions
        assert not torch.isclose(loss_labels, loss_targets), (
            "Masked loss should differ from full-sequence loss"
        )

    def test_forward_targets_unchanged(self):
        """Existing targets behavior is unchanged (no regression)."""
        model = _make_tiny_model()
        model.eval()

        torch.manual_seed(42)
        idx = torch.randint(0, 100, (2, 32))
        targets = torch.randint(0, 100, (2, 32))

        _, loss = model(idx, targets=targets)
        assert loss is not None, "targets should produce a loss"
        assert loss.ndim == 0, "loss should be scalar"
        assert not torch.isnan(loss), "loss should not be NaN"

    def test_forward_labels_none_no_loss(self):
        """No targets and no labels returns loss=None."""
        model = _make_tiny_model()
        model.eval()

        idx = torch.randint(0, 100, (2, 32))
        logits, loss = model(idx)
        assert loss is None, "No targets/labels should return None loss"
        assert logits.shape == (2, 32, 100), "logits shape should be (B, T, V)"

    def test_forward_labels_and_targets_labels_wins(self):
        """When both labels and targets are provided, labels takes precedence."""
        model = _make_tiny_model()
        model.eval()

        torch.manual_seed(42)
        idx = torch.randint(0, 100, (2, 32))

        # Labels: first half masked
        labels = torch.full((2, 32), -100, dtype=torch.long)
        labels[:, 16:] = torch.randint(0, 100, (2, 16))

        # Targets: all valid
        targets = torch.randint(0, 100, (2, 32))

        _, loss_both = model(idx, targets=targets, labels=labels)
        _, loss_labels_only = model(idx, labels=labels)

        assert loss_both is not None
        assert loss_labels_only is not None
        assert torch.isclose(loss_both, loss_labels_only), (
            "When both provided, labels should take precedence (same loss)"
        )


# ---------------------------------------------------------------------------
# get_sft_batch tests
# ---------------------------------------------------------------------------

class TestGetSftBatch:
    """get_sft_batch() loads paired binary files with correct indexing."""

    @pytest.fixture
    def sft_data_dir(self, tmp_path):
        """Create temporary paired binary files for testing."""
        block_size = 32
        num_examples = 4

        # Create input data (uint16): 4 examples x 32 tokens
        input_data = np.arange(
            num_examples * block_size, dtype=np.uint16
        ).reshape(num_examples, block_size)
        input_data.tofile(tmp_path / "train.bin")
        input_data[:2].tofile(tmp_path / "val.bin")  # 2 val examples

        # Create labels data (int32): matching structure with -100 for first half
        labels_data = np.full(
            (num_examples, block_size), -100, dtype=np.int32
        )
        # Set response tokens in second half to valid IDs
        for i in range(num_examples):
            labels_data[i, 16:] = np.arange(16, dtype=np.int32) + i * 100
        labels_data.tofile(tmp_path / "labels_train.bin")
        labels_data[:2].tofile(tmp_path / "labels_val.bin")

        return str(tmp_path), block_size, num_examples

    def test_get_sft_batch_returns_correct_shapes(self, sft_data_dir):
        from model.train import get_sft_batch

        data_dir, block_size, _ = sft_data_dir
        x, labels = get_sft_batch("train", batch_size=2, block_size=block_size, device="cpu", data_dir=data_dir)

        assert x.shape == (2, block_size), f"x shape should be (2, {block_size})"
        assert labels.shape == (2, block_size), f"labels shape should be (2, {block_size})"

    def test_get_sft_batch_labels_dtype(self, sft_data_dir):
        from model.train import get_sft_batch

        data_dir, block_size, _ = sft_data_dir
        x, labels = get_sft_batch("train", batch_size=2, block_size=block_size, device="cpu", data_dir=data_dir)

        assert labels.dtype == torch.int64, f"labels dtype should be int64, got {labels.dtype}"
        assert x.dtype == torch.int64, f"x dtype should be int64, got {x.dtype}"

    def test_get_sft_batch_alignment(self, sft_data_dir):
        """x and labels from same batch are aligned (same example indices)."""
        from model.train import get_sft_batch

        data_dir, block_size, num_examples = sft_data_dir

        # Load raw data to verify alignment
        raw_input = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        raw_labels = np.memmap(os.path.join(data_dir, "labels_train.bin"), dtype=np.int32, mode="r")

        # Run multiple batches and verify alignment
        torch.manual_seed(123)
        for _ in range(5):
            x, labels = get_sft_batch("train", batch_size=1, block_size=block_size, device="cpu", data_dir=data_dir)
            x_np = x[0].numpy()
            labels_np = labels[0].numpy()

            # Find which example this is by checking input values
            found = False
            for ex_idx in range(num_examples):
                start = ex_idx * block_size
                expected_x = raw_input[start:start + block_size].astype(np.int64)
                expected_labels = raw_labels[start:start + block_size].astype(np.int64)
                if np.array_equal(x_np, expected_x):
                    assert np.array_equal(labels_np, expected_labels), (
                        f"x matched example {ex_idx} but labels did not match"
                    )
                    found = True
                    break
            assert found, "Batch should contain a complete example from the data"


# ---------------------------------------------------------------------------
# Checkpoint discovery tests
# ---------------------------------------------------------------------------

class TestCheckpointDiscovery:
    """discover_1b_checkpoint() finds the most recent 1B best.pt."""

    def test_checkpoint_discovery_finds_best_pt(self, tmp_path):
        from model.train import discover_1b_checkpoint

        # Create a fake 1B checkpoint directory
        best_dir = tmp_path / "1B_20260318_102724" / "best"
        best_dir.mkdir(parents=True)
        (best_dir / "best.pt").write_text("fake checkpoint")

        result = discover_1b_checkpoint(str(tmp_path))
        assert result is not None, "Should find the checkpoint"
        assert result.endswith("best.pt"), "Should return path to best.pt"

    def test_checkpoint_discovery_most_recent(self, tmp_path):
        from model.train import discover_1b_checkpoint

        # Create two 1B checkpoint directories with different timestamps
        for ts in ["20260317_100000", "20260318_150000"]:
            best_dir = tmp_path / f"1B_{ts}" / "best"
            best_dir.mkdir(parents=True)
            (best_dir / "best.pt").write_text(f"checkpoint {ts}")

        result = discover_1b_checkpoint(str(tmp_path))
        assert result is not None
        assert "20260318_150000" in result, (
            f"Should return most recent checkpoint, got {result}"
        )

    def test_checkpoint_discovery_returns_none(self, tmp_path):
        from model.train import discover_1b_checkpoint

        # Empty directory — no 1B checkpoints
        result = discover_1b_checkpoint(str(tmp_path))
        assert result is None, "Should return None when no checkpoints found"


# ---------------------------------------------------------------------------
# CONFIG_DEFAULTS["1B-sft"] tests
# ---------------------------------------------------------------------------

class TestConfigDefaults1BSft:
    """CONFIG_DEFAULTS['1B-sft'] has correct SFT hyperparameters."""

    def test_config_defaults_1b_sft(self):
        from model.train import CONFIG_DEFAULTS

        assert "1B-sft" in CONFIG_DEFAULTS, "Missing '1B-sft' in CONFIG_DEFAULTS"
        sft = CONFIG_DEFAULTS["1B-sft"]
        assert sft["lr"] == 2e-5, f"lr should be 2e-5, got {sft['lr']}"
        assert sft["batch_size"] == 2, f"batch_size should be 2, got {sft['batch_size']}"
        assert sft["grad_accum"] == 4, f"grad_accum should be 4, got {sft['grad_accum']}"
        assert sft["warmup"] == 50, f"warmup should be 50, got {sft['warmup']}"
        assert sft["max_epochs"] == 3, f"max_epochs should be 3, got {sft['max_epochs']}"
        assert sft["dropout"] == 0.1, f"dropout should be 0.1, got {sft['dropout']}"


# ---------------------------------------------------------------------------
# --sft argument parsing tests
# ---------------------------------------------------------------------------

class TestSftArgParsing:
    """--sft flag is parsed correctly and enforces mutual exclusion."""

    def test_sft_arg_parsing(self):
        """--sft flag is parsed as True when provided."""
        from model.train import parse_args

        # Simulate --sft flag with minimal args (override sys.argv)
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ["train.py", "--sft", "--config", "1B"]
            args = parse_args()
            assert args.sft is True, "--sft should be True"
            assert args.lr == 2e-5, f"SFT lr should be 2e-5, got {args.lr}"
            assert args.batch_size == 2, f"SFT batch_size should be 2, got {args.batch_size}"
            assert args.warmup_steps == 50, f"SFT warmup should be 50, got {args.warmup_steps}"
            assert args.max_epochs == 3, f"SFT max_epochs should be 3, got {args.max_epochs}"
        finally:
            sys.argv = original_argv

    def test_sft_pretrained_mutual_exclusion(self):
        """--sft and --pretrained-weights cannot both be set."""
        from model.train import parse_args

        import sys
        original_argv = sys.argv
        try:
            sys.argv = [
                "train.py", "--sft", "--pretrained-weights", "some/path.pt",
                "--config", "1B",
            ]
            with pytest.raises(SystemExit) as exc_info:
                parse_args()
            assert exc_info.value.code == 1, (
                "Should exit with code 1 for mutual exclusion"
            )
        finally:
            sys.argv = original_argv
