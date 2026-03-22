#!/usr/bin/env python3
"""Unit tests for compare_all.py: 6-model cross-comparison script."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the qlora package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qlora"))

from compare_all import (
    _FAMILY_MAP,
    _MODEL_LABELS,
    _MODEL_ORDER,
    _MODEL_SUBDIRS,
    compute_ablation_delta,
    compute_capability_gap,
    load_model_data,
)


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------


def _make_model_data(
    mean_accuracy: float,
    responses: list[dict],
    binary_pass_rate: float = 0.0,
    hedging: int = 0,
    fabrication_rate: float = 0.0,
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
    fabrication_rate : float
        Overall fabrication rate (0.0 to 1.0 scale).
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

    # Build fabrication data
    total_entities = max(len(responses) * 2, 1)
    flagged_entities = int(total_entities * fabrication_rate)
    per_question = []
    for i, r in enumerate(responses):
        pq = {
            "question_id": f"q{i:03d}",
            "entities_extracted": [f"entity_{i}_a", f"entity_{i}_b"],
            "flagged_entities": [],
        }
        if flagged_entities > 0 and i < flagged_entities // 2:
            pq["flagged_entities"] = [f"entity_{i}_a"]
        per_question.append(pq)

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
            "responses": [
                {**r, "question_id": f"q{i:03d}", "category": "disease_mechanisms",
                 "prompt": f"Question {i}", "response": f"Response {i}"}
                for i, r in enumerate(responses)
            ],
        },
        "taxonomy": {
            "distribution": taxonomy_dist,
            "per_question": [
                {"question_id": f"q{i:03d}", "primary_mode": "accurate"}
                for i in range(len(responses))
            ],
        },
        "fabrications": {
            "summary": {
                "total_entities_extracted": total_entities,
                "total_flagged": flagged_entities,
                "flagged_rate": fabrication_rate,
            },
            "per_question": per_question,
        },
    }


def _make_responses(coherent: int, incoherent: int) -> list[dict]:
    """Create a list of response dicts with given coherent/incoherent counts."""
    return (
        [{"is_coherent": True} for _ in range(coherent)]
        + [{"is_coherent": False} for _ in range(incoherent)]
    )


# ---------------------------------------------------------------------------
# Test 1: _MODEL_SUBDIRS contains exactly 6 entries
# ---------------------------------------------------------------------------


class TestModelSubdirs:
    """Tests for the _MODEL_SUBDIRS configuration."""

    def test_model_subdirs_has_six_entries(self):
        """_MODEL_SUBDIRS contains exactly 6 model keys."""
        expected_keys = {
            "scratch_500m",
            "scratch_1b_base",
            "gpt2_large",
            "scratch_1b_instruct",
            "llama32_base",
            "llama32_qlora",
        }
        assert set(_MODEL_SUBDIRS.keys()) == expected_keys
        assert len(_MODEL_SUBDIRS) == 6

    def test_subdirs_are_unique(self):
        """All subdirectory names are unique."""
        subdirs = list(_MODEL_SUBDIRS.values())
        assert len(subdirs) == len(set(subdirs))

    def test_subdirs_match_expected_directories(self):
        """Subdirectory values match the known eval/results/ directory names."""
        assert _MODEL_SUBDIRS["scratch_500m"] == "als-lm-500m_q8_0"
        assert _MODEL_SUBDIRS["scratch_1b_base"] == "1B_20260317_205331"
        assert _MODEL_SUBDIRS["gpt2_large"] == "als-lm-gpt2-large_q8_0"
        assert _MODEL_SUBDIRS["scratch_1b_instruct"] == "als-lm-1b-instruct_q8_0"
        assert _MODEL_SUBDIRS["llama32_base"] == "als-lm-llama32-base"
        assert _MODEL_SUBDIRS["llama32_qlora"] == "alslm-1b_q8_0"


# ---------------------------------------------------------------------------
# Test 2: _MODEL_ORDER has 6 entries in correct display order
# ---------------------------------------------------------------------------


class TestModelOrder:
    """Tests for the _MODEL_ORDER configuration."""

    def test_model_order_has_six_entries(self):
        """_MODEL_ORDER contains exactly 6 entries."""
        assert len(_MODEL_ORDER) == 6

    def test_model_order_matches_subdirs_keys(self):
        """_MODEL_ORDER contains the same keys as _MODEL_SUBDIRS."""
        assert set(_MODEL_ORDER) == set(_MODEL_SUBDIRS.keys())

    def test_model_order_correct_sequence(self):
        """_MODEL_ORDER follows the expected display order."""
        expected = [
            "scratch_500m", "scratch_1b_base",
            "gpt2_large", "scratch_1b_instruct",
            "llama32_base", "llama32_qlora",
        ]
        assert _MODEL_ORDER == expected


# ---------------------------------------------------------------------------
# Test 3: _FAMILY_MAP maps all 6 models to 3 families
# ---------------------------------------------------------------------------


class TestFamilyMap:
    """Tests for the _FAMILY_MAP configuration."""

    def test_family_map_has_six_entries(self):
        """_FAMILY_MAP maps all 6 models."""
        assert len(_FAMILY_MAP) == 6
        assert set(_FAMILY_MAP.keys()) == set(_MODEL_SUBDIRS.keys())

    def test_three_families(self):
        """_FAMILY_MAP maps to exactly 3 approach families."""
        families = set(_FAMILY_MAP.values())
        assert families == {"from_scratch", "pretrained_finetune", "pretrained_instruct"}

    def test_from_scratch_family(self):
        """from_scratch family contains scratch_500m and scratch_1b_base."""
        from_scratch = [k for k, v in _FAMILY_MAP.items() if v == "from_scratch"]
        assert set(from_scratch) == {"scratch_500m", "scratch_1b_base"}

    def test_pretrained_finetune_family(self):
        """pretrained_finetune family contains gpt2_large and scratch_1b_instruct."""
        finetune = [k for k, v in _FAMILY_MAP.items() if v == "pretrained_finetune"]
        assert set(finetune) == {"gpt2_large", "scratch_1b_instruct"}

    def test_pretrained_instruct_family(self):
        """pretrained_instruct family contains llama32_base and llama32_qlora."""
        instruct = [k for k, v in _FAMILY_MAP.items() if v == "pretrained_instruct"]
        assert set(instruct) == {"llama32_base", "llama32_qlora"}


# ---------------------------------------------------------------------------
# Test 4: load_model_data returns dict with expected keys
# ---------------------------------------------------------------------------


class TestLoadModelData:
    """Tests for load_model_data function."""

    def test_load_model_data_returns_expected_keys(self, tmp_path):
        """load_model_data returns dict with scores/fabrications/taxonomy/responses."""
        import json

        # Create fixture JSON files
        fixture_data = {
            "scores.json": {"aggregate": {"overall": {"mean_accuracy": 0.5}}},
            "fabrications.json": {"summary": {"total_flagged": 0}},
            "taxonomy.json": {"distribution": {}},
            "responses.json": {"responses": []},
        }
        for fname, content in fixture_data.items():
            with open(tmp_path / fname, "w") as f:
                json.dump(content, f)

        result = load_model_data(tmp_path)
        assert "scores" in result
        assert "fabrications" in result
        assert "taxonomy" in result
        assert "responses" in result

    def test_load_model_data_missing_file_exits(self, tmp_path):
        """load_model_data exits with error when a required file is missing."""
        # Only create one of the required files
        import json
        with open(tmp_path / "scores.json", "w") as f:
            json.dump({}, f)

        with pytest.raises(SystemExit):
            load_model_data(tmp_path)


# ---------------------------------------------------------------------------
# Test 5: compute_capability_gap returns expected fields
# ---------------------------------------------------------------------------


class TestComputeCapabilityGap:
    """Tests for compute_capability_gap function."""

    def test_returns_fields_for_each_model(self):
        """compute_capability_gap returns coherence_pct, accuracy_pct, gap_pct."""
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

    def test_gap_calculation(self):
        """Gap is coherence_pct minus accuracy_pct."""
        all_data = {
            "m": _make_model_data(
                mean_accuracy=0.05,
                responses=_make_responses(3, 1),  # 75% coherent
            ),
        }
        result = compute_capability_gap(all_data)
        assert result["m"]["coherence_pct"] == pytest.approx(75.0)
        assert result["m"]["accuracy_pct"] == pytest.approx(5.0)
        assert result["m"]["gap_pct"] == pytest.approx(70.0)

    def test_all_degenerate_zero_gap(self):
        """All-degenerate model: 0% gap."""
        all_data = {
            "m": _make_model_data(
                mean_accuracy=0.0,
                responses=_make_responses(0, 5),
            ),
        }
        result = compute_capability_gap(all_data)
        assert result["m"]["gap_pct"] == pytest.approx(0.0)

    def test_multi_model(self):
        """Multiple models processed independently."""
        all_data = {
            "a": _make_model_data(mean_accuracy=0.1, responses=_make_responses(8, 2)),
            "b": _make_model_data(mean_accuracy=0.0, responses=_make_responses(0, 10)),
        }
        result = compute_capability_gap(all_data)
        assert len(result) == 2
        assert result["a"]["gap_pct"] == pytest.approx(70.0)
        assert result["b"]["gap_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 6: compute_ablation_delta returns accuracy/fabrication/coherence delta
# ---------------------------------------------------------------------------


class TestComputeAblationDelta:
    """Tests for compute_ablation_delta function."""

    def test_returns_three_deltas(self):
        """compute_ablation_delta returns accuracy_delta, fabrication_delta, coherence_delta."""
        qlora_data = _make_model_data(
            mean_accuracy=0.40,
            responses=_make_responses(8, 2),
            fabrication_rate=0.20,
        )
        baseline_data = _make_model_data(
            mean_accuracy=0.10,
            responses=_make_responses(6, 4),
            fabrication_rate=0.50,
        )
        result = compute_ablation_delta(qlora_data, baseline_data)
        assert "accuracy_delta" in result
        assert "fabrication_delta" in result
        assert "coherence_delta" in result

    def test_positive_accuracy_delta(self):
        """QLoRA improving over baseline shows positive accuracy delta."""
        qlora_data = _make_model_data(
            mean_accuracy=0.50,
            responses=_make_responses(9, 1),
            fabrication_rate=0.10,
        )
        baseline_data = _make_model_data(
            mean_accuracy=0.10,
            responses=_make_responses(6, 4),
            fabrication_rate=0.50,
        )
        result = compute_ablation_delta(qlora_data, baseline_data)
        assert result["accuracy_delta"] == pytest.approx(0.40)

    def test_coherence_delta_calculation(self):
        """Coherence delta is QLoRA coherence% - baseline coherence%."""
        qlora_data = _make_model_data(
            mean_accuracy=0.30,
            responses=_make_responses(8, 2),  # 80% coherent
        )
        baseline_data = _make_model_data(
            mean_accuracy=0.10,
            responses=_make_responses(5, 5),  # 50% coherent
        )
        result = compute_ablation_delta(qlora_data, baseline_data)
        assert result["coherence_delta"] == pytest.approx(30.0)

    def test_fabrication_delta_calculation(self):
        """Fabrication delta is QLoRA rate - baseline rate (negative = improvement)."""
        qlora_data = _make_model_data(
            mean_accuracy=0.30,
            responses=_make_responses(8, 2),
            fabrication_rate=0.15,
        )
        baseline_data = _make_model_data(
            mean_accuracy=0.10,
            responses=_make_responses(5, 5),
            fabrication_rate=0.50,
        )
        result = compute_ablation_delta(qlora_data, baseline_data)
        assert result["fabrication_delta"] == pytest.approx(-0.35)


# ---------------------------------------------------------------------------
# Test 7: Each approach family has exactly 2 models
# ---------------------------------------------------------------------------


class TestFamilyBalance:
    """Tests for approach family balance."""

    def test_each_family_has_two_models(self):
        """All 3 approach families have exactly 2 models each."""
        from collections import Counter
        family_counts = Counter(_FAMILY_MAP.values())
        assert len(family_counts) == 3
        for family, count in family_counts.items():
            assert count == 2, f"Family '{family}' has {count} models, expected 2"


# ---------------------------------------------------------------------------
# Test 8: _MODEL_LABELS has both long and short labels for all 6 models
# ---------------------------------------------------------------------------


class TestModelLabels:
    """Tests for the _MODEL_LABELS configuration."""

    def test_model_labels_has_six_entries(self):
        """_MODEL_LABELS has entries for all 6 models."""
        assert len(_MODEL_LABELS) == 6
        assert set(_MODEL_LABELS.keys()) == set(_MODEL_SUBDIRS.keys())

    def test_labels_are_tuples_with_two_elements(self):
        """Each label is a tuple with (long_label, short_label)."""
        for key, label in _MODEL_LABELS.items():
            assert isinstance(label, tuple), f"{key} label is not a tuple"
            assert len(label) == 2, f"{key} label tuple should have 2 elements"
            assert len(label[0]) > 0, f"{key} long label is empty"
            assert len(label[1]) > 0, f"{key} short label is empty"

    def test_short_labels_are_unique(self):
        """All short labels are unique."""
        short_labels = [v[1] for v in _MODEL_LABELS.values()]
        assert len(short_labels) == len(set(short_labels))

    def test_long_labels_are_unique(self):
        """All long labels are unique."""
        long_labels = [v[0] for v in _MODEL_LABELS.values()]
        assert len(long_labels) == len(set(long_labels))
