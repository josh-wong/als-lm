"""Tests for resource monitoring in log_step and helper functions."""

import io
import json
import sys
import os

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model.train import log_step


class TestLogStepResourceFields:
    """Test that log_step writes resource monitoring fields to JSONL."""

    def _call_log_step(self, **resource_kwargs):
        """Helper: call log_step with standard positional args + optional resource kwargs."""
        buf = io.StringIO()
        log_step(
            buf,          # log_file
            100,          # step
            5.0,          # loss
            3e-4,         # lr
            1000.0,       # tokens_per_sec
            500,          # gpu_mem_mb
            0.1,          # dt_sec
            1,            # epoch
            0.5,          # epoch_progress
            10000,        # total_tokens
            1024,         # loss_scale
            0.5,          # grad_norm
            **resource_kwargs,
        )
        return json.loads(buf.getvalue())

    def test_cpu_ram_mb_field_present(self):
        entry = self._call_log_step(cpu_ram_mb=8192)
        assert "cpu_ram_mb" in entry
        assert entry["cpu_ram_mb"] == 8192
        assert isinstance(entry["cpu_ram_mb"], int)

    def test_gpu_util_pct_field_present(self):
        entry = self._call_log_step(gpu_util_pct=75)
        assert "gpu_util_pct" in entry
        assert entry["gpu_util_pct"] == 75
        assert isinstance(entry["gpu_util_pct"], int)

    def test_gpu_temp_c_field_present(self):
        entry = self._call_log_step(gpu_temp_c=68)
        assert "gpu_temp_c" in entry
        assert entry["gpu_temp_c"] == 68
        assert isinstance(entry["gpu_temp_c"], int)

    def test_gpu_peak_mem_mb_field_present(self):
        entry = self._call_log_step(gpu_peak_mem_mb=11000)
        assert "gpu_peak_mem_mb" in entry
        assert entry["gpu_peak_mem_mb"] == 11000
        assert isinstance(entry["gpu_peak_mem_mb"], int)

    def test_existing_fields_preserved(self):
        """All original log_step fields must still be present."""
        entry = self._call_log_step(
            cpu_ram_mb=4096, gpu_util_pct=50, gpu_temp_c=60, gpu_peak_mem_mb=9000,
        )
        expected_fields = [
            "type", "step", "loss", "perplexity", "lr", "tokens_per_sec",
            "gpu_mem_mb", "dt_sec", "epoch", "epoch_progress", "total_tokens",
            "loss_scale", "grad_norm", "timestamp",
        ]
        for field in expected_fields:
            assert field in entry, f"Missing existing field: {field}"
        # Verify values of a few key fields
        assert entry["type"] == "step"
        assert entry["step"] == 100
        assert entry["loss"] == 5.0
        assert entry["gpu_mem_mb"] == 500

    def test_defaults_are_zero(self):
        """When resource kwargs are omitted, they default to 0."""
        entry = self._call_log_step()
        assert entry["cpu_ram_mb"] == 0
        assert entry["gpu_util_pct"] == 0
        assert entry["gpu_temp_c"] == 0
        assert entry["gpu_peak_mem_mb"] == 0


class TestResourceHelperFunctions:
    """Test the GPU/CPU resource helper functions."""

    def test_get_gpu_utilization_none_handle(self):
        from model.train import get_gpu_utilization
        assert get_gpu_utilization(None) == 0

    def test_get_gpu_temperature_none_handle(self):
        from model.train import get_gpu_temperature
        assert get_gpu_temperature(None) == 0

    def test_get_cpu_ram_mb_returns_positive_int(self):
        from model.train import get_cpu_ram_mb
        result = get_cpu_ram_mb()
        assert isinstance(result, int)
        assert result > 0

    def test_get_gpu_peak_mem_mb_returns_nonneg_int(self):
        from model.train import get_gpu_peak_mem_mb
        result = get_gpu_peak_mem_mb()
        assert isinstance(result, int)
        assert result >= 0
