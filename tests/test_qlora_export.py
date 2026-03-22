"""EXPORT-01, EXPORT-02, and EXPORT-03 validation tests for merge and export scripts.

Tests verify that qlora/merge_adapter.py exists and uses the correct PEFT merge
pattern (FP16 base load, PeftModel, merge_and_unload), that qlora/export_qlora.py
exists and performs multi-quant GGUF conversion with Ollama registration, and that
the Ollama model naming and configuration follow the project conventions.

These are non-GPU "structural" tests that validate code structure and naming
conventions without requiring a CUDA device or model weights.
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_merge_script = PROJECT_ROOT / "qlora" / "merge_adapter.py"
_export_script = PROJECT_ROOT / "qlora" / "export_qlora.py"


# ---------------------------------------------------------------------------
# TestMergeScript: EXPORT-01 structural validation (5 tests)
# ---------------------------------------------------------------------------

class TestMergeScript:
    """Verify qlora/merge_adapter.py exists and uses correct PEFT merge pattern."""

    def test_merge_script_exists(self):
        """qlora/merge_adapter.py exists on disk."""
        assert _merge_script.exists(), f"Script not found: {_merge_script}"

    def test_imports_peft_model(self):
        """Script source imports PeftModel from peft."""
        assert _merge_script.exists(), f"Script not found: {_merge_script}"
        source = _merge_script.read_text()
        assert "PeftModel" in source, (
            "merge_adapter.py should import PeftModel from peft"
        )

    def test_loads_config(self):
        """Script source reads from configs/qlora.json."""
        assert _merge_script.exists(), f"Script not found: {_merge_script}"
        source = _merge_script.read_text()
        assert "qlora.json" in source, (
            "merge_adapter.py should load config from configs/qlora.json"
        )

    def test_uses_fp16_dtype(self):
        """Script source uses torch.float16 for FP16 merge (not 4-bit)."""
        assert _merge_script.exists(), f"Script not found: {_merge_script}"
        source = _merge_script.read_text()
        assert "torch.float16" in source, (
            "merge_adapter.py should load base model with torch.float16"
        )

    def test_references_merged_output_dir(self):
        """Script source references checkpoints/qlora/merged/ output directory."""
        assert _merge_script.exists(), f"Script not found: {_merge_script}"
        source = _merge_script.read_text()
        assert "merged" in source, (
            "merge_adapter.py should reference checkpoints/qlora/merged/"
        )


# ---------------------------------------------------------------------------
# TestExportScript: EXPORT-02 structural validation (4 tests)
# ---------------------------------------------------------------------------

class TestExportScript:
    """Verify qlora/export_qlora.py exists and uses correct GGUF pipeline."""

    def test_export_script_exists(self):
        """qlora/export_qlora.py exists on disk."""
        assert _export_script.exists(), f"Script not found: {_export_script}"

    def test_references_gguf_converter(self):
        """Script source imports or references convert_hf_to_gguf."""
        assert _export_script.exists(), f"Script not found: {_export_script}"
        source = _export_script.read_text()
        assert "convert_hf_to_gguf" in source, (
            "export_qlora.py should reference convert_hf_to_gguf for GGUF conversion"
        )

    def test_contains_all_gguf_filenames(self):
        """Script source contains all 3 GGUF filenames."""
        assert _export_script.exists(), f"Script not found: {_export_script}"
        source = _export_script.read_text()
        for filename in ["alslm-1b-f16.gguf", "alslm-1b-q8_0.gguf", "alslm-1b-q4_k_m.gguf"]:
            assert filename in source, (
                f"export_qlora.py should contain GGUF filename '{filename}'"
            )

    def test_references_system_prompt(self):
        """Script source references system_prompt from config."""
        assert _export_script.exists(), f"Script not found: {_export_script}"
        source = _export_script.read_text()
        assert "system_prompt" in source, (
            "export_qlora.py should reference system_prompt from configs/qlora.json"
        )


# ---------------------------------------------------------------------------
# TestOllamaConfig: EXPORT-03 structural validation (4 tests)
# ---------------------------------------------------------------------------

class TestOllamaConfig:
    """Verify Ollama naming conventions and Modelfile configuration."""

    def test_correct_ollama_model_name(self):
        """Script source contains the correct Ollama model name 'alslm-1b'."""
        assert _export_script.exists(), f"Script not found: {_export_script}"
        source = _export_script.read_text()
        assert "alslm-1b" in source, (
            "export_qlora.py should contain Ollama model name 'alslm-1b'"
        )

    def test_default_tag_copy(self):
        """Script source contains Q8_0 default tag copy (ollama cp)."""
        assert _export_script.exists(), f"Script not found: {_export_script}"
        source = _export_script.read_text()
        assert "ollama" in source and "cp" in source, (
            "export_qlora.py should use 'ollama cp' to set Q8_0 as default"
        )

    def test_baseline_re_registration(self):
        """Script source contains baseline re-registration name 'alslm-1b-base'."""
        assert _export_script.exists(), f"Script not found: {_export_script}"
        source = _export_script.read_text()
        assert "alslm-1b-base" in source, (
            "export_qlora.py should re-register ablation baseline as 'alslm-1b-base'"
        )

    def test_no_template_directive(self):
        """Script source does NOT contain TEMPLATE directive (native auto-detection)."""
        assert _export_script.exists(), f"Script not found: {_export_script}"
        source = _export_script.read_text()
        # The word TEMPLATE should not appear as a Modelfile directive
        # (it may appear in comments/docstrings explaining why it's omitted)
        lines = source.split("\n")
        for line in lines:
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            # Check for TEMPLATE as a Modelfile directive in string literals
            if "TEMPLATE" in stripped and ("f\"TEMPLATE" in stripped or "'TEMPLATE" in stripped or '"TEMPLATE' in stripped):
                pytest.fail(
                    "export_qlora.py should NOT contain a TEMPLATE directive in the Modelfile. "
                    "Ollama auto-detects the chat template from GGUF metadata."
                )
