"""Tests for resource plot generation in analyze_training.py."""

import os
import sys

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from analyze_training import generate_resource_report, plot_resource_metric


def _make_steps(n=10, include_resources=True):
    """Create synthetic step dicts resembling JSONL log entries."""
    steps = []
    for i in range(n):
        entry = {
            "type": "step",
            "step": (i + 1) * 100,
            "loss": 8.0 - i * 0.3,
            "lr": 0.0003,
            "epoch": 0,
        }
        if include_resources:
            entry.update({
                "gpu_peak_mem_mb": 5000 + i * 100,
                "cpu_ram_mb": 20000 + i * 200,
                "gpu_util_pct": 95 + (i % 5),
                "gpu_temp_c": 65 + i * 0.5,
                "tokens_per_sec": 18000 + i * 500,
            })
        steps.append(entry)
    return steps


class TestPlotResourceMetric:
    """Tests for the plot_resource_metric function."""

    def test_creates_png_for_present_metric(self, tmp_path):
        """plot_resource_metric creates a PNG file given synthetic step data
        with gpu_peak_mem_mb field."""
        steps = _make_steps(10, include_resources=True)
        out = str(tmp_path / "vram_usage.png")
        result = plot_resource_metric(
            steps, "gpu_peak_mem_mb", "VRAM (MB)", "VRAM usage", out
        )
        assert result is True
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_skips_gracefully_when_metric_missing(self, tmp_path):
        """plot_resource_metric skips gracefully (no crash, no file) when
        metric_key is missing from all step entries."""
        steps = _make_steps(10, include_resources=False)
        out = str(tmp_path / "nonexistent_metric.png")
        result = plot_resource_metric(
            steps, "gpu_peak_mem_mb", "VRAM (MB)", "VRAM usage", out
        )
        assert result is False
        assert not os.path.exists(out)


class TestGenerateResourceReport:
    """Tests for the generate_resource_report function."""

    def test_report_contains_stats(self):
        """generate_resource_report returns a markdown string with Peak, Mean,
        Min stats for each resource metric."""
        steps = _make_steps(10, include_resources=True)
        report = generate_resource_report(steps)
        assert isinstance(report, str)
        assert "Peak" in report
        assert "Mean" in report
        assert "Min" in report
        # Should have stats for known metrics
        assert "VRAM" in report or "gpu_peak_mem_mb" in report
        assert "tokens" in report.lower() or "throughput" in report.lower()

    def test_handles_empty_steps(self):
        """generate_resource_report handles empty steps list without error."""
        report = generate_resource_report([])
        assert isinstance(report, str)
        assert len(report) > 0
