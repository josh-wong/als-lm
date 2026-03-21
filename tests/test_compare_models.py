#!/usr/bin/env python3
"""Unit tests for compute_capability_gap and 4-model configuration in compare_models.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the scripts package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compare_models import (
    _MODEL_LABELS,
    _MODEL_SUBDIRS,
    compute_capability_gap,
)


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------


def _make_model_data(
    mean_accuracy: float,
    responses: list[dict],
) -> dict:
    """Build a minimal model data dict for compute_capability_gap testing.

    Parameters
    ----------
    mean_accuracy : float
        Overall mean accuracy (0.0 to 1.0 scale).
    responses : list[dict]
        Each dict must have ``is_coherent`` (bool) key at minimum.
    """
    return {
        "scores": {
            "aggregate": {
                "overall": {
                    "mean_accuracy": mean_accuracy,
                },
            },
        },
        "responses": {
            "responses": responses,
        },
    }


# ---------------------------------------------------------------------------
# Test 1: compute_capability_gap returns required fields
# ---------------------------------------------------------------------------


class TestComputeCapabilityGap:
    """Tests for the compute_capability_gap function."""

    def test_returns_coherence_accuracy_gap_for_each_model(self):
        """compute_capability_gap returns coherence_pct, accuracy_pct, gap_pct for each key."""
        all_data = {
            "model_a": _make_model_data(
                mean_accuracy=0.05,
                responses=[
                    {"is_coherent": True},
                    {"is_coherent": True},
                    {"is_coherent": False},
                    {"is_coherent": True},
                ],
            ),
        }

        result = compute_capability_gap(all_data)

        assert "model_a" in result
        entry = result["model_a"]
        assert "coherence_pct" in entry
        assert "accuracy_pct" in entry
        assert "gap_pct" in entry

        # 3/4 coherent = 75%, accuracy = 5%
        assert entry["coherence_pct"] == pytest.approx(75.0)
        assert entry["accuracy_pct"] == pytest.approx(5.0)
        assert entry["gap_pct"] == pytest.approx(70.0)

    def test_all_degenerate_model_zero_gap(self):
        """All-degenerate model: 0% coherence, 0% accuracy, 0% gap."""
        all_data = {
            "degenerate_model": _make_model_data(
                mean_accuracy=0.0,
                responses=[
                    {"is_coherent": False},
                    {"is_coherent": False},
                    {"is_coherent": False},
                    {"is_coherent": False},
                    {"is_coherent": False},
                ],
            ),
        }

        result = compute_capability_gap(all_data)
        entry = result["degenerate_model"]

        assert entry["coherence_pct"] == pytest.approx(0.0)
        assert entry["accuracy_pct"] == pytest.approx(0.0)
        assert entry["gap_pct"] == pytest.approx(0.0)

    def test_partial_coherent_model(self):
        """Partial coherent model: 20% coherent, 3% accuracy, 17% gap."""
        # 1 out of 5 coherent = 20%
        all_data = {
            "partial": _make_model_data(
                mean_accuracy=0.03,
                responses=[
                    {"is_coherent": True},
                    {"is_coherent": False},
                    {"is_coherent": False},
                    {"is_coherent": False},
                    {"is_coherent": False},
                ],
            ),
        }

        result = compute_capability_gap(all_data)
        entry = result["partial"]

        assert entry["coherence_pct"] == pytest.approx(20.0)
        assert entry["accuracy_pct"] == pytest.approx(3.0)
        assert entry["gap_pct"] == pytest.approx(17.0)


# ---------------------------------------------------------------------------
# Test 4: 4-model configuration
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
