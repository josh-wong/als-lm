"""Tests for resource monitoring in log_step, log_epoch, and training summary."""

import io
import json
import os
import sys
import tempfile

# Ensure project root is on sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model.train import log_epoch, log_step, write_training_summary_file


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


class TestLogEpochNewFields:
    """Test that log_epoch accepts and writes val_loss, peak_vram_mb, gpu_temp_c."""

    def _call_log_epoch(self, **kwargs):
        """Helper: call log_epoch with standard positional args + optional new kwargs."""
        buf = io.StringIO()
        log_epoch(
            buf,              # log_file
            epoch=1,
            steps_in_epoch=500,
            avg_train_loss=3.5,
            total_tokens=1000000,
            elapsed_sec=3600.0,
            **kwargs,
        )
        return json.loads(buf.getvalue())

    def test_val_loss_field_written(self):
        entry = self._call_log_epoch(val_loss=3.2)
        assert "val_loss" in entry
        assert entry["val_loss"] == 3.2

    def test_peak_vram_mb_field_written(self):
        entry = self._call_log_epoch(peak_vram_mb=6080)
        assert "peak_vram_mb" in entry
        assert entry["peak_vram_mb"] == 6080

    def test_gpu_temp_c_field_written(self):
        entry = self._call_log_epoch(gpu_temp_c=72)
        assert "gpu_temp_c" in entry
        assert entry["gpu_temp_c"] == 72

    def test_backward_compat_no_new_args(self):
        """Calling log_epoch without new args should still work; new fields default."""
        entry = self._call_log_epoch()
        assert entry["type"] == "epoch"
        assert entry["epoch"] == 1
        assert entry["steps"] == 500
        # New fields should have safe defaults
        assert entry.get("val_loss") is None
        assert entry.get("peak_vram_mb") == 0
        assert entry.get("gpu_temp_c") == 0

    def test_existing_fields_preserved(self):
        """All original epoch JSONL fields must remain."""
        entry = self._call_log_epoch(val_loss=3.0, peak_vram_mb=6000, gpu_temp_c=70)
        for field in ["type", "epoch", "steps", "avg_train_loss", "avg_perplexity",
                       "total_tokens", "elapsed_sec", "timestamp"]:
            assert field in entry, f"Missing existing field: {field}"


class TestWriteTrainingSummaryFile:
    """Test that write_training_summary_file creates a markdown file with expected content."""

    def _make_mock_epoch_tracker(self):
        """Create a minimal object that looks like EpochTracker for the summary."""
        class MockTracker:
            epoch = 3
            steps_per_epoch = 3893
        return MockTracker()

    def _write_summary(self, tmp_path):
        """Helper that writes a summary file and returns its content."""
        write_training_summary_file(
            summary_path=tmp_path,
            config_name="1B",
            total_steps=11679,
            elapsed_time=162000.0,
            final_train_loss=2.8,
            final_val_loss=3.1,
            best_val_loss=3.0,
            best_val_step=10000,
            total_tokens_processed=3000000000,
            run_dir="checkpoints/1B_20260317_120000",
            epoch_tracker=self._make_mock_epoch_tracker(),
            total_checkpoints_saved=23,
            total_best_updates=5,
            total_bytes_written=46 * 1024 * 1024 * 1024,
            peak_vram_mb=6080,
            peak_cpu_ram_mb=22000,
            peak_gpu_temp_c=78,
        )
        with open(tmp_path) as f:
            return f.read()

    def test_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
            tmp = tf.name
        try:
            self._write_summary(tmp)
            assert os.path.isfile(tmp)
        finally:
            os.unlink(tmp)

    def test_contains_training_metrics_section(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
            tmp = tf.name
        try:
            content = self._write_summary(tmp)
            assert "## Training metrics" in content
        finally:
            os.unlink(tmp)

    def test_contains_resource_peaks_section(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
            tmp = tf.name
        try:
            content = self._write_summary(tmp)
            assert "## Resource peaks" in content
        finally:
            os.unlink(tmp)

    def test_contains_checkpoints_section(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
            tmp = tf.name
        try:
            content = self._write_summary(tmp)
            assert "## Checkpoints" in content
        finally:
            os.unlink(tmp)

    def test_contains_actual_values(self):
        """Summary should contain actual numeric values, not placeholders."""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
            tmp = tf.name
        try:
            content = self._write_summary(tmp)
            assert "11,679" in content or "11679" in content
            assert "2.8" in content  # final_train_loss
            assert "3.1" in content  # final_val_loss
            assert "3.0" in content  # best_val_loss
            assert "6080" in content or "6,080" in content or "6.08" in content  # peak VRAM
        finally:
            os.unlink(tmp)
