"""Unit tests for 1B model configuration correctness.

Validates that the 1B config is present in MODEL_CONFIGS, CONFIG_DEFAULTS,
the readiness gate, and configs/1b.json with correct values matching the
v2 design document specifications.
"""

import json
import os
import sys

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# MODEL_CONFIGS tests (model/model.py)
# ---------------------------------------------------------------------------

class TestModelConfigs1B:
    """Verify MODEL_CONFIGS['1B'] exists with correct architecture params."""

    def test_1b_exists_in_model_configs(self):
        from model.model import MODEL_CONFIGS
        assert "1B" in MODEL_CONFIGS, "1B must be a key in MODEL_CONFIGS"

    def test_1b_is_gptconfig_instance(self):
        from model.model import GPTConfig, MODEL_CONFIGS
        assert isinstance(MODEL_CONFIGS["1B"], GPTConfig)

    def test_1b_n_layer(self):
        from model.model import MODEL_CONFIGS
        assert MODEL_CONFIGS["1B"].n_layer == 30

    def test_1b_n_head(self):
        from model.model import MODEL_CONFIGS
        assert MODEL_CONFIGS["1B"].n_head == 20

    def test_1b_n_embd(self):
        from model.model import MODEL_CONFIGS
        assert MODEL_CONFIGS["1B"].n_embd == 1600

    def test_1b_dropout_baked_in(self):
        from model.model import MODEL_CONFIGS
        assert MODEL_CONFIGS["1B"].dropout == 0.1, (
            "1B dropout should be 0.1 baked into the config"
        )

    def test_1b_gradient_checkpointing_enabled(self):
        from model.model import MODEL_CONFIGS
        assert MODEL_CONFIGS["1B"].use_gradient_checkpointing is True

    def test_1b_block_size(self):
        from model.model import MODEL_CONFIGS
        assert MODEL_CONFIGS["1B"].block_size == 1024

    def test_1b_bias(self):
        from model.model import MODEL_CONFIGS
        assert MODEL_CONFIGS["1B"].bias is True

    def test_1b_param_count_approximately_1004m(self):
        """Verify param count is within 5% of ~1.004B using GPT-2 formula.

        GPT-2 parameter formula (with weight tying):
            params = vocab_size * n_embd                     # token embeddings (shared with lm_head)
                   + block_size * n_embd                     # position embeddings
                   + n_layer * (
                       4 * n_embd * n_embd                   # c_attn (QKV projection)
                       + n_embd                              # c_attn bias
                       + n_embd * n_embd                     # c_proj (attention output)
                       + n_embd                              # c_proj bias
                       + 4 * n_embd * 4 * n_embd             # MLP c_fc (up-projection) — wait, this is 4*n_embd*n_embd for weight
                     )
        Simplified: use a representative vocab_size for counting.
        """
        from model.model import GPT, MODEL_CONFIGS

        config = MODEL_CONFIGS["1B"]
        # Use a representative vocab_size for param counting
        model = GPT.from_config("1B", vocab_size=32768)
        n_params = model.get_num_params(non_embedding=False)

        # Should be approximately 1.004B (within 5%)
        target = 1_004_000_000
        tolerance = 0.05
        assert abs(n_params - target) / target < tolerance, (
            f"1B param count {n_params:,} is not within 5% of {target:,}"
        )


# ---------------------------------------------------------------------------
# CONFIG_DEFAULTS tests (model/train.py)
# ---------------------------------------------------------------------------

class TestConfigDefaults1B:
    """Verify CONFIG_DEFAULTS['1B'] has correct training hyperparameters."""

    def test_1b_exists_in_config_defaults(self):
        from model.train import CONFIG_DEFAULTS
        assert "1B" in CONFIG_DEFAULTS

    def test_1b_lr(self):
        from model.train import CONFIG_DEFAULTS
        assert CONFIG_DEFAULTS["1B"]["lr"] == 3e-4

    def test_1b_batch_size(self):
        from model.train import CONFIG_DEFAULTS
        assert CONFIG_DEFAULTS["1B"]["batch_size"] == 4

    def test_1b_grad_accum(self):
        from model.train import CONFIG_DEFAULTS
        assert CONFIG_DEFAULTS["1B"]["grad_accum"] == 8

    def test_1b_warmup(self):
        from model.train import CONFIG_DEFAULTS
        assert CONFIG_DEFAULTS["1B"]["warmup"] == 1000

    def test_1b_max_steps(self):
        from model.train import CONFIG_DEFAULTS
        assert CONFIG_DEFAULTS["1B"]["max_steps"] == 50000


# ---------------------------------------------------------------------------
# Readiness gate tests (benchmark/readiness_gate.py)
# ---------------------------------------------------------------------------

class TestReadinessGate1B:
    """Verify readiness gate 1B config has corrected n_layer."""

    def test_readiness_gate_1b_n_layer_is_30(self):
        from benchmark.readiness_gate import MODEL_CONFIGS as GATE_CONFIGS
        assert GATE_CONFIGS["1B"]["n_layer"] == 30, (
            f"Readiness gate 1B n_layer should be 30, got {GATE_CONFIGS['1B']['n_layer']}"
        )


# ---------------------------------------------------------------------------
# configs/1b.json tests
# ---------------------------------------------------------------------------

class TestConfigs1bJson:
    """Verify configs/1b.json exists and has correct values."""

    CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "..", "configs", "1b.json"
    )

    def test_file_exists(self):
        assert os.path.isfile(self.CONFIG_PATH), "configs/1b.json must exist"

    def test_valid_json(self):
        with open(self.CONFIG_PATH) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def _load(self):
        with open(self.CONFIG_PATH) as f:
            return json.load(f)

    def test_model_n_layer(self):
        data = self._load()
        assert data["model"]["n_layer"] == 30

    def test_model_n_head(self):
        data = self._load()
        assert data["model"]["n_head"] == 20

    def test_model_n_embd(self):
        data = self._load()
        assert data["model"]["n_embd"] == 1600

    def test_model_dropout(self):
        data = self._load()
        assert data["model"]["dropout"] == 0.1

    def test_model_gradient_checkpointing(self):
        data = self._load()
        assert data["model"]["use_gradient_checkpointing"] is True

    def test_cos_min_ratio_is_zero(self):
        data = self._load()
        assert data["deepspeed"]["scheduler"]["params"]["cos_min_ratio"] == 0.0, (
            "cos_min_ratio must be 0.0 (not 0.1 like 500m.json)"
        )

    def test_benchmark_results_is_null(self):
        data = self._load()
        assert data["benchmark_results"] is None

    def test_design_doc_field(self):
        data = self._load()
        assert "design_doc" in data
        assert "v2-design-doc.md" in data["design_doc"]
