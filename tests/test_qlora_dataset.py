"""DATA-01, DATA-02, and DATA-03 validation tests for QLoRA dataset formatting.

Tests verify that qlora/format_dataset.py correctly converts 970 ALS
instruction pairs from Alpaca format to the model's native chat template
format with stratified 90/10 train/val split, token length reporting,
and structural validation.

DATA-03 tests verify that qlora/check_leakage.py correctly detects
benchmark contamination between the 970 instruction pairs and the
160 evaluation questions using fuzzy matching (partial_ratio >= 80).

Tests skip gracefully when output files do not exist (run
`python qlora/format_dataset.py` first to generate them).
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so imports work
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_qlora_output_dir = _project_root / "data" / "instruction" / "qlora"
_train_jsonl = _qlora_output_dir / "train.jsonl"
_val_jsonl = _qlora_output_dir / "val.jsonl"
_meta_json = _qlora_output_dir / "meta.json"
_qlora_config = _project_root / "configs" / "qlora.json"

_output_exists = _train_jsonl.exists() and _val_jsonl.exists() and _meta_json.exists()
_skip_reason = "Run `python qlora/format_dataset.py` first to generate output files"

# Expected categories from the ALS instruction dataset
_EXPECTED_CATEGORIES = {
    "clinical_trials", "diagnosis", "epidemiology", "genetics",
    "pathophysiology", "patient_care", "symptoms", "treatment",
}


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file and return a list of parsed JSON objects."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# TestFormatDataset: DATA-01 validation (6 tests)
# ---------------------------------------------------------------------------

class TestFormatDataset:
    """Validate format_dataset.py output files exist and have correct counts."""

    def test_config_has_system_prompt(self):
        """configs/qlora.json contains a 'system_prompt' string field."""
        assert _qlora_config.exists(), f"Config not found: {_qlora_config}"
        config = json.loads(_qlora_config.read_text())
        assert "system_prompt" in config, "Missing 'system_prompt' in qlora.json"
        assert isinstance(config["system_prompt"], str), "system_prompt must be a string"
        assert len(config["system_prompt"]) > 10, "system_prompt is too short"

    def test_format_script_exists(self):
        """qlora/format_dataset.py exists and is a valid Python file."""
        script = _project_root / "qlora" / "format_dataset.py"
        assert script.exists(), f"Script not found: {script}"
        content = script.read_text()
        assert "apply_chat_template" in content, (
            "format_dataset.py should use tokenizer.apply_chat_template()"
        )

    @pytest.mark.skipif(not _output_exists, reason=_skip_reason)
    def test_output_files_exist(self):
        """After running format_dataset, train.jsonl, val.jsonl, and meta.json exist."""
        assert _train_jsonl.exists(), f"Missing: {_train_jsonl}"
        assert _val_jsonl.exists(), f"Missing: {_val_jsonl}"
        assert _meta_json.exists(), f"Missing: {_meta_json}"

    @pytest.mark.skipif(not _output_exists, reason=_skip_reason)
    def test_train_val_split_counts(self):
        """train.jsonl has 873 lines, val.jsonl has 97 lines (total 970)."""
        train = _load_jsonl(_train_jsonl)
        val = _load_jsonl(_val_jsonl)
        assert len(train) == 873, f"Expected 873 train examples, got {len(train)}"
        assert len(val) == 97, f"Expected 97 val examples, got {len(val)}"
        assert len(train) + len(val) == 970, (
            f"Total should be 970, got {len(train) + len(val)}"
        )

    @pytest.mark.skipif(not _output_exists, reason=_skip_reason)
    def test_jsonl_has_text_field(self):
        """Each JSONL line has a 'text' key with a non-empty string."""
        for path in [_train_jsonl, _val_jsonl]:
            records = _load_jsonl(path)
            for i, record in enumerate(records):
                assert "text" in record, f"{path.name} line {i}: missing 'text' key"
                assert isinstance(record["text"], str), (
                    f"{path.name} line {i}: 'text' is not a string"
                )
                assert len(record["text"].strip()) > 0, (
                    f"{path.name} line {i}: 'text' is empty"
                )

    @pytest.mark.skipif(not _output_exists, reason=_skip_reason)
    def test_stratified_categories(self):
        """Both train and val splits contain all 8 categories in roughly 90/10 proportions."""
        meta = json.loads(_meta_json.read_text())
        assert "category_distribution" in meta, "Missing category_distribution in meta.json"

        dist = meta["category_distribution"]
        assert "train" in dist, "Missing train category distribution"
        assert "val" in dist, "Missing val category distribution"

        train_cats = set(dist["train"].keys())
        val_cats = set(dist["val"].keys())

        assert train_cats == _EXPECTED_CATEGORIES, (
            f"Train categories mismatch: expected {_EXPECTED_CATEGORIES}, got {train_cats}"
        )
        assert val_cats == _EXPECTED_CATEGORIES, (
            f"Val categories mismatch: expected {_EXPECTED_CATEGORIES}, got {val_cats}"
        )

        # Verify roughly 90/10 proportions for each category
        for cat in _EXPECTED_CATEGORIES:
            train_count = dist["train"][cat]
            val_count = dist["val"][cat]
            total = train_count + val_count
            val_ratio = val_count / total if total > 0 else 0
            assert 0.05 <= val_ratio <= 0.20, (
                f"Category '{cat}': val ratio {val_ratio:.2f} not in [0.05, 0.20] "
                f"(train={train_count}, val={val_count})"
            )


# ---------------------------------------------------------------------------
# TestValidateStructure: DATA-02 validation (4 tests)
# ---------------------------------------------------------------------------

class TestValidateStructure:
    """Validate the structural integrity of the formatted output files."""

    @pytest.mark.skipif(not _output_exists, reason=_skip_reason)
    def test_decoded_samples_have_roles(self):
        """3+ decoded samples contain 'system', 'user', 'assistant' role markers."""
        try:
            from transformers import AutoTokenizer
        except ImportError:
            pytest.skip("transformers not installed")

        config = json.loads(_qlora_config.read_text())
        model_id = config["model"]["model_id"]

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        train_data = _load_jsonl(_train_jsonl)
        assert len(train_data) >= 3, f"Need at least 3 samples, got {len(train_data)}"

        random.seed(42)
        samples = random.sample(train_data, 3)

        for i, sample in enumerate(samples):
            text = sample["text"]
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            assert "system" in decoded, (
                f"Sample {i}: missing 'system' role marker in decoded text"
            )
            assert "user" in decoded, (
                f"Sample {i}: missing 'user' role marker in decoded text"
            )
            assert "assistant" in decoded, (
                f"Sample {i}: missing 'assistant' role marker in decoded text"
            )

    @pytest.mark.skipif(not _output_exists, reason=_skip_reason)
    def test_meta_json_fields(self):
        """meta.json contains required traceability fields."""
        meta = json.loads(_meta_json.read_text())

        required_fields = [
            "model_id", "timestamp", "total_examples",
            "train_examples", "val_examples", "token_stats",
        ]
        for field in required_fields:
            assert field in meta, f"Missing required field '{field}' in meta.json"

        assert isinstance(meta["token_stats"], dict), "token_stats must be a dict"
        token_stat_fields = ["min", "max", "mean", "median"]
        for field in token_stat_fields:
            assert field in meta["token_stats"], (
                f"Missing '{field}' in token_stats"
            )

    @pytest.mark.skipif(not _output_exists, reason=_skip_reason)
    def test_model_id_consistency(self):
        """meta.json model_id matches configs/qlora.json model.model_id."""
        config = json.loads(_qlora_config.read_text())
        meta = json.loads(_meta_json.read_text())

        expected_model_id = config["model"]["model_id"]
        actual_model_id = meta["model_id"]

        assert actual_model_id == expected_model_id, (
            f"meta.json model_id '{actual_model_id}' does not match "
            f"qlora.json model_id '{expected_model_id}'"
        )

    @pytest.mark.skipif(not _output_exists, reason=_skip_reason)
    def test_no_silent_truncation(self):
        """Token stats in meta.json report examples exceeding max_seq_length (should be 0)."""
        config = json.loads(_qlora_config.read_text())
        meta = json.loads(_meta_json.read_text())

        max_seq_length = config["training"]["max_seq_length"]

        assert "examples_over_max_seq_length" in meta, (
            "Missing 'examples_over_max_seq_length' in meta.json"
        )
        over_count = meta["examples_over_max_seq_length"]
        assert over_count == 0, (
            f"{over_count} examples exceed max_seq_length ({max_seq_length}). "
            "Adjust max_seq_length in configs/qlora.json before training."
        )


# ---------------------------------------------------------------------------
# TestLeakageCheck: DATA-03 validation (6 tests)
# ---------------------------------------------------------------------------

_leakage_script = _project_root / "qlora" / "check_leakage.py"


class TestLeakageCheck:
    """Validate benchmark leakage detection between instruction pairs and eval questions."""

    def test_leakage_script_exists(self):
        """qlora/check_leakage.py exists on disk."""
        assert _leakage_script.exists(), (
            f"Script not found: {_leakage_script}"
        )

    def test_leakage_uses_partial_ratio(self):
        """check_leakage.py uses partial_ratio (not token_set_ratio from older validate.py)."""
        assert _leakage_script.exists(), f"Script not found: {_leakage_script}"
        source = _leakage_script.read_text()
        assert "partial_ratio" in source, (
            "check_leakage.py should use rapidfuzz.fuzz.partial_ratio, "
            "not token_set_ratio from the older validate.py"
        )

    def test_leakage_threshold_80(self):
        """check_leakage.py uses threshold 80 (not 75 from older validate.py)."""
        assert _leakage_script.exists(), f"Script not found: {_leakage_script}"
        source = _leakage_script.read_text()
        assert "80" in source, (
            "check_leakage.py should use threshold 80, not 75"
        )

    def test_leakage_checks_instruction_vs_questions(self):
        """check_leakage.py compares instruction text against question and prompt_template."""
        assert _leakage_script.exists(), f"Script not found: {_leakage_script}"
        source = _leakage_script.read_text()
        assert "question" in source, (
            "check_leakage.py should compare against eval 'question' field"
        )
        assert "prompt_template" in source, (
            "check_leakage.py should compare against eval 'prompt_template' field"
        )

    def test_leakage_checks_answers_vs_key_facts(self):
        """check_leakage.py compares output/answers against key_facts."""
        assert _leakage_script.exists(), f"Script not found: {_leakage_script}"
        source = _leakage_script.read_text()
        assert "key_facts" in source, (
            "check_leakage.py should compare instruction answers against eval key_facts"
        )

    @pytest.mark.skipif(not _leakage_script.exists(), reason="qlora/check_leakage.py does not exist yet")
    def test_leakage_pass_exit_code(self):
        """Running check_leakage.py exits with code 0 (no leakage detected)."""
        result = subprocess.run(
            [sys.executable, str(_leakage_script)],
            capture_output=True,
            text=True,
            cwd=str(_project_root),
        )
        assert result.returncode == 0, (
            f"check_leakage.py exited with code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
