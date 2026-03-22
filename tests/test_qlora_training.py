"""TRAIN-01, TRAIN-02, and EVAL-01 validation tests for QLoRA training script.

Tests verify that qlora/train_qlora.py exists and uses the correct libraries
(SFTTrainer, PEFT, config-driven hyperparameters), that the prompt-completion
dataset loading correctly splits at the assistant tag boundary for
completion-only loss masking, and that qlora/eval_baseline.py references the
correct Ollama model and evaluation pipeline.

These are non-GPU "structural" tests that validate code structure and dataset
formatting without requiring a CUDA device.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_train_script = PROJECT_ROOT / "qlora" / "train_qlora.py"
_eval_script = PROJECT_ROOT / "qlora" / "eval_baseline.py"


# ---------------------------------------------------------------------------
# TestTrainScript: TRAIN-01 structural validation (5 tests)
# ---------------------------------------------------------------------------

class TestTrainScript:
    """Verify qlora/train_qlora.py exists and uses correct libraries."""

    def test_train_script_exists(self):
        """qlora/train_qlora.py exists on disk."""
        assert _train_script.exists(), f"Script not found: {_train_script}"

    def test_imports_sfttrainer(self):
        """Script source contains SFTTrainer import."""
        assert _train_script.exists(), f"Script not found: {_train_script}"
        source = _train_script.read_text()
        assert "SFTTrainer" in source, (
            "train_qlora.py should import SFTTrainer from trl"
        )

    def test_imports_peft(self):
        """Script source contains get_peft_model or LoraConfig import."""
        assert _train_script.exists(), f"Script not found: {_train_script}"
        source = _train_script.read_text()
        has_peft = "get_peft_model" in source or "LoraConfig" in source
        assert has_peft, (
            "train_qlora.py should import get_peft_model or LoraConfig from peft"
        )

    def test_loads_config(self):
        """Script source reads from configs/qlora.json."""
        assert _train_script.exists(), f"Script not found: {_train_script}"
        source = _train_script.read_text()
        assert "qlora.json" in source, (
            "train_qlora.py should load config from configs/qlora.json"
        )

    def test_has_spot_check(self):
        """Script source contains spot-check function."""
        assert _train_script.exists(), f"Script not found: {_train_script}"
        source = _train_script.read_text()
        assert "spot_check" in source or "spot check" in source.lower(), (
            "train_qlora.py should contain a spot-check function"
        )


# ---------------------------------------------------------------------------
# TestCompletionMasking: TRAIN-02 prompt-completion split validation (3 tests)
# ---------------------------------------------------------------------------

class TestCompletionMasking:
    """Verify load_prompt_completion() correctly splits JSONL at the assistant tag boundary.

    The fixture uses the current model's assistant tag (derived from
    _DEFAULT_ASSISTANT_TAG) so that tests remain valid when the model_id
    in configs/qlora.json changes.
    """

    @pytest.fixture
    def sample_jsonl(self, tmp_path):
        """Create a temporary JSONL file with a chat-template sample using the default assistant tag."""
        from qlora.utils import DEFAULT_ASSISTANT_TAG as _DEFAULT_ASSISTANT_TAG
        # Build a generic sample using the current model's tag format
        sample_text = (
            "<|im_start|>system\n"
            "You are a knowledgeable assistant specializing in ALS research.<|im_end|>\n"
            "<|im_start|>user\n"
            "What is riluzole?<|im_end|>\n"
            f"{_DEFAULT_ASSISTANT_TAG}"
            "Riluzole is the first FDA-approved drug for treating ALS.<|im_end|>\n"
        )
        jsonl_path = tmp_path / "test_sample.jsonl"
        jsonl_path.write_text(json.dumps({"text": sample_text}) + "\n")
        return jsonl_path, sample_text, _DEFAULT_ASSISTANT_TAG

    def test_prompt_completion_split(self, sample_jsonl):
        """load_prompt_completion() correctly splits a sample JSONL line at the assistant tag boundary."""
        jsonl_path, original_text, _ = sample_jsonl
        from qlora.train_qlora import load_prompt_completion
        ds = load_prompt_completion(str(jsonl_path))
        assert "prompt" in ds.column_names, "Dataset should have 'prompt' column"
        assert "completion" in ds.column_names, "Dataset should have 'completion' column"
        # No data lost: prompt + completion == original text
        reconstructed = ds[0]["prompt"] + ds[0]["completion"]
        assert reconstructed == original_text, (
            f"Data lost in split:\n"
            f"  original:      {original_text!r}\n"
            f"  reconstructed: {reconstructed!r}"
        )

    def test_prompt_ends_with_assistant_tag(self, sample_jsonl):
        """Prompt field ends with the model's assistant tag."""
        jsonl_path, _, assistant_tag = sample_jsonl
        from qlora.train_qlora import load_prompt_completion
        ds = load_prompt_completion(str(jsonl_path))
        prompt = ds[0]["prompt"]
        assert prompt.endswith(assistant_tag), (
            f"Prompt should end with {assistant_tag!r}, got: ...{prompt[-40:]!r}"
        )

    def test_completion_does_not_include_assistant_tag(self, sample_jsonl):
        """Completion field starts with the response text, not the assistant tag."""
        jsonl_path, _, assistant_tag = sample_jsonl
        from qlora.train_qlora import load_prompt_completion
        ds = load_prompt_completion(str(jsonl_path))
        completion = ds[0]["completion"]
        assert not completion.startswith(assistant_tag), (
            f"Completion should not start with the assistant tag {assistant_tag!r}"
        )


# ---------------------------------------------------------------------------
# TestEvalBaseline: EVAL-01 structural validation (3 tests)
# ---------------------------------------------------------------------------

class TestEvalBaseline:
    """Verify qlora/eval_baseline.py exists and references correct model and pipeline."""

    def test_eval_baseline_exists(self):
        """qlora/eval_baseline.py exists on disk."""
        assert _eval_script.exists(), f"Script not found: {_eval_script}"

    def test_invokes_eval_pipeline(self):
        """Script source contains run_evaluation or eval/run_evaluation.py reference."""
        assert _eval_script.exists(), f"Script not found: {_eval_script}"
        source = _eval_script.read_text()
        has_eval = "run_evaluation" in source or "eval/run_evaluation.py" in source
        assert has_eval, (
            "eval_baseline.py should invoke the eval pipeline via run_evaluation"
        )

    def test_ollama_model_name(self):
        """Script source contains the correct Ollama model name."""
        assert _eval_script.exists(), f"Script not found: {_eval_script}"
        source = _eval_script.read_text()
        assert "als-lm-llama32-base" in source, (
            "eval_baseline.py should reference Ollama model name 'als-lm-llama32-base'"
        )
