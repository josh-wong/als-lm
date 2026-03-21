#!/usr/bin/env python3
"""Unit tests for compare_models.py: capability gap, analysis helpers, and configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the scripts package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compare_models import (
    _MODEL_LABELS,
    _MODEL_ORDER,
    _MODEL_SUBDIRS,
    _build_model_configs,
    _safe_get,
    compute_accuracy_comparison,
    compute_capability_gap,
    compute_degenerate_rates,
    compute_taxonomy_comparison,
)


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------


def _make_model_data(
    mean_accuracy: float,
    responses: list[dict],
    binary_pass_rate: float = 0.0,
    hedging: int = 0,
    taxonomy_dist: dict = None,
    by_category: dict = None,
) -> dict:
    """Build a model data dict for testing analysis functions.

    Parameters
    ----------
    mean_accuracy : float
        Overall mean accuracy (0.0 to 1.0 scale).
    responses : list[dict]
        Each dict must have ``is_coherent`` (bool) key at minimum.
    binary_pass_rate : float
        Binary pass rate for accuracy comparison.
    hedging : int
        Hedging instance count.
    taxonomy_dist : dict
        Taxonomy distribution. If None, builds a minimal one from responses.
    by_category : dict
        Per-category accuracy. If None, uses empty dict.
    """
    if taxonomy_dist is None:
        degenerate = sum(1 for r in responses if not r.get("is_coherent", True))
        non_degenerate = len(responses) - degenerate
        total = len(responses)
        taxonomy_dist = {
            "degenerate": {
                "count": degenerate,
                "pct": (degenerate / total * 100) if total > 0 else 0.0,
            },
            "accurate": {
                "count": non_degenerate,
                "pct": (non_degenerate / total * 100) if total > 0 else 0.0,
            },
        }

    return {
        "scores": {
            "aggregate": {
                "overall": {
                    "mean_accuracy": mean_accuracy,
                    "binary_pass_rate": binary_pass_rate,
                    "total_hedging_instances": hedging,
                },
                "by_category": by_category or {},
            },
        },
        "responses": {
            "responses": responses,
        },
        "taxonomy": {
            "distribution": taxonomy_dist,
        },
    }


def _make_responses(coherent: int, incoherent: int) -> list[dict]:
    """Create a list of response dicts with given coherent/incoherent counts."""
    return (
        [{"is_coherent": True} for _ in range(coherent)]
        + [{"is_coherent": False} for _ in range(incoherent)]
    )


# ---------------------------------------------------------------------------
# TestSafeGet
# ---------------------------------------------------------------------------


class TestSafeGet:
    """Tests for the _safe_get nested dict accessor."""

    def test_simple_path(self):
        data = {"a": {"b": {"c": 42}}}
        assert _safe_get(data, ["a", "b", "c"], "test") == 42

    def test_single_key(self):
        data = {"key": "value"}
        assert _safe_get(data, ["key"], "test") == "value"

    def test_missing_key_raises_with_context(self):
        data = {"a": {"b": 1}}
        with pytest.raises(KeyError, match="test.*failed at 'c'"):
            _safe_get(data, ["a", "c"], "test")

    def test_missing_nested_key_raises(self):
        data = {"a": {"b": {"c": 1}}}
        with pytest.raises(KeyError, match="failed at 'missing'"):
            _safe_get(data, ["a", "b", "missing"], "model_x")

    def test_none_in_path_raises(self):
        data = {"a": None}
        with pytest.raises(KeyError):
            _safe_get(data, ["a", "b"], "test")


# ---------------------------------------------------------------------------
# TestBuildModelConfigs
# ---------------------------------------------------------------------------


class TestBuildModelConfigs:
    """Tests for _build_model_configs."""

    def test_returns_list_with_four_entries(self, tmp_path):
        configs = _build_model_configs(tmp_path)
        assert len(configs) == 4

    def test_each_config_has_required_keys(self, tmp_path):
        configs = _build_model_configs(tmp_path)
        for config in configs:
            assert "key" in config
            assert "label" in config
            assert "short_label" in config
            assert "path" in config

    def test_paths_are_under_results_dir(self, tmp_path):
        configs = _build_model_configs(tmp_path)
        for config in configs:
            assert str(config["path"]).startswith(str(tmp_path))

    def test_keys_match_model_order(self, tmp_path):
        configs = _build_model_configs(tmp_path)
        config_keys = [c["key"] for c in configs]
        assert config_keys == list(_MODEL_SUBDIRS.keys())


# ---------------------------------------------------------------------------
# TestComputeCapabilityGap
# ---------------------------------------------------------------------------


class TestComputeCapabilityGap:
    """Tests for the compute_capability_gap function."""

    def test_returns_coherence_accuracy_gap_for_each_model(self):
        """Basic case: 3/4 coherent, 5% accuracy = 70% gap."""
        all_data = {
            "model_a": _make_model_data(
                mean_accuracy=0.05,
                responses=_make_responses(3, 1),
            ),
        }
        result = compute_capability_gap(all_data)

        assert "model_a" in result
        entry = result["model_a"]
        assert "coherence_pct" in entry
        assert "accuracy_pct" in entry
        assert "gap_pct" in entry
        assert entry["coherence_pct"] == pytest.approx(75.0)
        assert entry["accuracy_pct"] == pytest.approx(5.0)
        assert entry["gap_pct"] == pytest.approx(70.0)

    def test_all_degenerate_model_zero_gap(self):
        """All-degenerate model: 0% coherence, 0% accuracy, 0% gap."""
        all_data = {
            "degenerate_model": _make_model_data(
                mean_accuracy=0.0,
                responses=_make_responses(0, 5),
            ),
        }
        result = compute_capability_gap(all_data)
        entry = result["degenerate_model"]

        assert entry["coherence_pct"] == pytest.approx(0.0)
        assert entry["accuracy_pct"] == pytest.approx(0.0)
        assert entry["gap_pct"] == pytest.approx(0.0)

    def test_partial_coherent_model(self):
        """Partial coherent model: 20% coherent, 3% accuracy, 17% gap."""
        all_data = {
            "partial": _make_model_data(
                mean_accuracy=0.03,
                responses=_make_responses(1, 4),
            ),
        }
        result = compute_capability_gap(all_data)
        entry = result["partial"]

        assert entry["coherence_pct"] == pytest.approx(20.0)
        assert entry["accuracy_pct"] == pytest.approx(3.0)
        assert entry["gap_pct"] == pytest.approx(17.0)

    def test_negative_gap_accuracy_exceeds_coherence(self):
        """Negative gap when accuracy > coherence (edge case: coherence
        check is stricter than accuracy scoring)."""
        all_data = {
            "neg_gap": _make_model_data(
                mean_accuracy=0.50,
                responses=_make_responses(1, 4),  # 20% coherent
            ),
        }
        result = compute_capability_gap(all_data)
        entry = result["neg_gap"]

        assert entry["coherence_pct"] == pytest.approx(20.0)
        assert entry["accuracy_pct"] == pytest.approx(50.0)
        assert entry["gap_pct"] == pytest.approx(-30.0)

    def test_perfect_model_zero_gap(self):
        """100% coherent and 100% accurate = 0% gap."""
        all_data = {
            "perfect": _make_model_data(
                mean_accuracy=1.0,
                responses=_make_responses(10, 0),
            ),
        }
        result = compute_capability_gap(all_data)
        entry = result["perfect"]

        assert entry["coherence_pct"] == pytest.approx(100.0)
        assert entry["accuracy_pct"] == pytest.approx(100.0)
        assert entry["gap_pct"] == pytest.approx(0.0)

    def test_all_coherent_zero_accuracy(self):
        """100% coherent but 0% accurate = maximum ethical risk (100% gap)."""
        all_data = {
            "risky": _make_model_data(
                mean_accuracy=0.0,
                responses=_make_responses(10, 0),
            ),
        }
        result = compute_capability_gap(all_data)
        entry = result["risky"]

        assert entry["coherence_pct"] == pytest.approx(100.0)
        assert entry["accuracy_pct"] == pytest.approx(0.0)
        assert entry["gap_pct"] == pytest.approx(100.0)

    def test_multi_model_in_single_call(self):
        """Multiple models processed in one call with independent results."""
        all_data = {
            "model_a": _make_model_data(
                mean_accuracy=0.10,
                responses=_make_responses(8, 2),  # 80%
            ),
            "model_b": _make_model_data(
                mean_accuracy=0.0,
                responses=_make_responses(0, 10),  # 0%
            ),
            "model_c": _make_model_data(
                mean_accuracy=0.50,
                responses=_make_responses(5, 5),  # 50%
            ),
        }
        result = compute_capability_gap(all_data)

        assert len(result) == 3
        assert result["model_a"]["gap_pct"] == pytest.approx(70.0)
        assert result["model_b"]["gap_pct"] == pytest.approx(0.0)
        assert result["model_c"]["gap_pct"] == pytest.approx(0.0)

    def test_four_model_comparison(self):
        """Simulate the real 4-model comparison with representative data."""
        all_data = {
            "scratch_500m": _make_model_data(
                mean_accuracy=0.023,
                responses=_make_responses(70, 90),  # ~43.75% coherent
            ),
            "gpt2_large_finetune": _make_model_data(
                mean_accuracy=0.256,
                responses=_make_responses(40, 120),  # 25% coherent
            ),
            "scratch_1b_base": _make_model_data(
                mean_accuracy=0.0,
                responses=_make_responses(56, 104),  # 35% coherent
            ),
            "scratch_1b_instruct": _make_model_data(
                mean_accuracy=0.0,
                responses=_make_responses(0, 160),  # 0% coherent
            ),
        }
        result = compute_capability_gap(all_data)

        assert len(result) == 4
        # 500M: high gap (coherent but inaccurate)
        assert result["scratch_500m"]["gap_pct"] > 0
        # GPT-2: potentially negative gap (accuracy > coherence possible)
        assert "gap_pct" in result["gpt2_large_finetune"]
        # 1B base: positive gap
        assert result["scratch_1b_base"]["gap_pct"] == pytest.approx(35.0)
        # 1B instruct: null result
        assert result["scratch_1b_instruct"]["gap_pct"] == pytest.approx(0.0)

    def test_empty_input(self):
        """Empty dict produces empty result."""
        result = compute_capability_gap({})
        assert result == {}

    def test_single_response(self):
        """Model with exactly one response."""
        all_data = {
            "tiny": _make_model_data(
                mean_accuracy=0.5,
                responses=[{"is_coherent": True}],
            ),
        }
        result = compute_capability_gap(all_data)
        assert result["tiny"]["coherence_pct"] == pytest.approx(100.0)
        assert result["tiny"]["accuracy_pct"] == pytest.approx(50.0)
        assert result["tiny"]["gap_pct"] == pytest.approx(50.0)

    def test_missing_is_coherent_defaults_to_true(self):
        """Responses without is_coherent key default to True (coherent)."""
        all_data = {
            "no_flag": _make_model_data(
                mean_accuracy=0.0,
                responses=[{}, {}, {}],  # No is_coherent key
            ),
        }
        result = compute_capability_gap(all_data)
        assert result["no_flag"]["coherence_pct"] == pytest.approx(100.0)

    def test_results_are_rounded(self):
        """Results are rounded to 2 decimal places."""
        all_data = {
            "model": _make_model_data(
                mean_accuracy=0.033333,
                responses=_make_responses(1, 2),  # 33.333...% coherent
            ),
        }
        result = compute_capability_gap(all_data)
        entry = result["model"]
        assert entry["coherence_pct"] == pytest.approx(33.33, abs=0.01)
        assert entry["accuracy_pct"] == pytest.approx(3.33, abs=0.01)

    def test_missing_scores_key_raises(self):
        """Missing nested key raises KeyError with model context."""
        all_data = {
            "broken": {
                "responses": {"responses": [{"is_coherent": True}]},
                # Missing "scores" key
            },
        }
        with pytest.raises(KeyError, match="broken"):
            compute_capability_gap(all_data)


# ---------------------------------------------------------------------------
# TestComputeAccuracyComparison
# ---------------------------------------------------------------------------


class TestComputeAccuracyComparison:
    """Tests for compute_accuracy_comparison."""

    def test_extracts_accuracy_fields(self):
        all_data = {
            "m1": _make_model_data(
                mean_accuracy=0.25,
                responses=[],
                binary_pass_rate=0.30,
                hedging=5,
            ),
        }
        result = compute_accuracy_comparison(all_data)
        assert result["m1"]["mean_accuracy"] == pytest.approx(0.25)
        assert result["m1"]["binary_pass_rate"] == pytest.approx(0.30)
        assert result["m1"]["hedging"] == 5

    def test_multi_model(self):
        all_data = {
            "a": _make_model_data(mean_accuracy=0.1, responses=[]),
            "b": _make_model_data(mean_accuracy=0.5, responses=[]),
        }
        result = compute_accuracy_comparison(all_data)
        assert len(result) == 2
        assert result["a"]["mean_accuracy"] < result["b"]["mean_accuracy"]


# ---------------------------------------------------------------------------
# TestComputeDegenerateRates
# ---------------------------------------------------------------------------


class TestComputeDegenerateRates:
    """Tests for compute_degenerate_rates."""

    def test_all_degenerate(self):
        all_data = {
            "m": _make_model_data(
                mean_accuracy=0.0,
                responses=_make_responses(0, 10),
            ),
        }
        result = compute_degenerate_rates(all_data)
        assert result["m"]["degenerate"] == 10
        assert result["m"]["non_degenerate"] == 0
        assert result["m"]["non_degenerate_rate"] == pytest.approx(0.0)

    def test_no_degenerate(self):
        all_data = {
            "m": _make_model_data(
                mean_accuracy=0.5,
                responses=_make_responses(10, 0),
            ),
        }
        result = compute_degenerate_rates(all_data)
        assert result["m"]["degenerate"] == 0
        assert result["m"]["non_degenerate"] == 10
        assert result["m"]["non_degenerate_rate"] == pytest.approx(1.0)

    def test_mixed(self):
        all_data = {
            "m": _make_model_data(
                mean_accuracy=0.1,
                responses=_make_responses(6, 4),
            ),
        }
        result = compute_degenerate_rates(all_data)
        assert result["m"]["total"] == 10
        assert result["m"]["degenerate"] == 4
        assert result["m"]["non_degenerate_rate"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# TestComputeTaxonomyComparison
# ---------------------------------------------------------------------------


class TestComputeTaxonomyComparison:
    """Tests for compute_taxonomy_comparison."""

    def test_extracts_taxonomy_modes(self):
        taxonomy_dist = {
            "degenerate": {"count": 80, "pct": 50.0},
            "accurate": {"count": 30, "pct": 18.75},
            "confident_fabrication": {"count": 50, "pct": 31.25},
        }
        all_data = {
            "m": _make_model_data(
                mean_accuracy=0.1,
                responses=_make_responses(80, 80),
                taxonomy_dist=taxonomy_dist,
            ),
        }
        result = compute_taxonomy_comparison(all_data)
        assert result["m"]["degenerate"]["count"] == 80
        assert result["m"]["accurate"]["pct"] == pytest.approx(18.75)

    def test_missing_mode_returns_zero(self):
        """Taxonomy modes not present in data get count=0, pct=0.0."""
        taxonomy_dist = {
            "degenerate": {"count": 10, "pct": 100.0},
        }
        all_data = {
            "m": _make_model_data(
                mean_accuracy=0.0,
                responses=_make_responses(0, 10),
                taxonomy_dist=taxonomy_dist,
            ),
        }
        result = compute_taxonomy_comparison(all_data)
        assert result["m"]["accurate"]["count"] == 0
        assert result["m"]["confident_fabrication"]["pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestFourModelConfiguration
# ---------------------------------------------------------------------------


class TestFourModelConfiguration:
    """Tests for the 4-model _MODEL_SUBDIRS and _MODEL_LABELS configuration."""

    def test_model_subdirs_has_four_entries(self):
        """_MODEL_SUBDIRS contains exactly 4 model keys."""
        expected_keys = {
            "scratch_500m",
            "gpt2_large_finetune",
            "scratch_1b_base",
            "scratch_1b_instruct",
        }
        assert set(_MODEL_SUBDIRS.keys()) == expected_keys

    def test_model_labels_has_short_labels_for_all(self):
        """_MODEL_LABELS has both long and short labels for all 4 models."""
        expected_keys = {
            "scratch_500m",
            "gpt2_large_finetune",
            "scratch_1b_base",
            "scratch_1b_instruct",
        }
        assert set(_MODEL_LABELS.keys()) == expected_keys

        for key in expected_keys:
            label_tuple = _MODEL_LABELS[key]
            assert isinstance(label_tuple, tuple), f"{key} label is not a tuple"
            assert len(label_tuple) == 2, f"{key} label tuple should have 2 elements"
            assert len(label_tuple[0]) > 0, f"{key} long label is empty"
            assert len(label_tuple[1]) > 0, f"{key} short label is empty"

    def test_model_order_matches_subdirs_keys(self):
        """_MODEL_ORDER contains the same keys as _MODEL_SUBDIRS."""
        assert set(_MODEL_ORDER) == set(_MODEL_SUBDIRS.keys())

    def test_no_duplicate_labels(self):
        """All short labels are unique."""
        short_labels = [v[1] for v in _MODEL_LABELS.values()]
        assert len(short_labels) == len(set(short_labels))

    def test_no_duplicate_subdirs(self):
        """All subdirectory names are unique."""
        subdirs = list(_MODEL_SUBDIRS.values())
        assert len(subdirs) == len(set(subdirs))
