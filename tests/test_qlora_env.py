"""ENV-01 validation tests for QLoRA package installation and configuration.

Tests cover package imports (peft, bitsandbytes, trl, accelerate, datasets),
exact version pins matching requirements.txt, existing package stability
(transformers and torch remain at pinned versions), and QLoRA config file
validation (JSON structure, model ID, LoRA target modules).
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so imports work
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# QLoRA package imports and versions
# ---------------------------------------------------------------------------

class TestQLoRAImports:
    """Validate that all five QLoRA packages import and match pinned versions."""

    def test_imports(self):
        """Importing peft, bitsandbytes, trl, accelerate, datasets succeeds."""
        import peft
        import bitsandbytes
        import trl
        import accelerate
        import datasets

        # Verify modules are importable (no ImportError)
        assert peft is not None
        assert bitsandbytes is not None
        assert trl is not None
        assert accelerate is not None
        assert datasets is not None

    def test_versions(self):
        """Package versions match requirements.txt pins exactly."""
        import peft
        import bitsandbytes
        import trl
        import accelerate
        import datasets

        assert peft.__version__ == "0.18.1", (
            f"peft version should be 0.18.1, got {peft.__version__}"
        )
        assert bitsandbytes.__version__ == "0.49.2", (
            f"bitsandbytes version should be 0.49.2, got {bitsandbytes.__version__}"
        )
        assert trl.__version__ == "0.29.1", (
            f"trl version should be 0.29.1, got {trl.__version__}"
        )
        assert accelerate.__version__ == "1.12.0", (
            f"accelerate version should be 1.12.0, got {accelerate.__version__}"
        )
        assert datasets.__version__ == "4.8.3", (
            f"datasets version should be 4.8.3, got {datasets.__version__}"
        )

    def test_existing_unchanged(self):
        """Existing transformers==5.2.0 and torch==2.10.0 remain at pinned versions."""
        import transformers
        import torch

        assert transformers.__version__ == "5.2.0", (
            f"transformers should remain at 5.2.0, got {transformers.__version__}"
        )
        # torch version may include CUDA suffix like "2.10.0+cu128"
        assert torch.__version__.startswith("2.10.0"), (
            f"torch should remain at 2.10.0, got {torch.__version__}"
        )


# ---------------------------------------------------------------------------
# QLoRA config file validation
# ---------------------------------------------------------------------------

class TestQLoRAConfig:
    """Validate configs/qlora.json structure and contents."""

    @pytest.fixture
    def config(self):
        path = _project_root / "configs" / "qlora.json"
        with open(path) as f:
            return json.load(f)

    def test_qlora_config_valid(self, config):
        """configs/qlora.json loads as valid JSON with required top-level keys."""
        assert "model" in config, "Missing 'model' section"
        assert "lora" in config, "Missing 'lora' section"
        assert "training" in config, "Missing 'training' section"

    def test_qlora_config_model_id(self, config):
        """Model ID is meta-llama/Llama-3.2-1B-Instruct."""
        assert config["model"]["model_id"] == "meta-llama/Llama-3.2-1B-Instruct", (
            f"model_id should be meta-llama/Llama-3.2-1B-Instruct, "
            f"got {config['model']['model_id']}"
        )

    def test_qlora_config_lora_targets(self, config):
        """LoRA target_modules contains all 7 linear projection layers."""
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        actual_modules = config["lora"]["target_modules"]
        assert len(actual_modules) == 7, (
            f"Expected 7 target modules, got {len(actual_modules)}: {actual_modules}"
        )
        for module in expected_modules:
            assert module in actual_modules, (
                f"Missing target module: {module}"
            )
