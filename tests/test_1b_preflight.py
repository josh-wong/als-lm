"""Unit tests for 1B pre-flight helper functions.

Tests environment checks (WSL2 memory, disk space), max_steps calculation
from token count, configs/1b.json update logic, readiness report generation,
and the PREFLIGHT_CHECKLIST constant.
"""

import json
import os
import struct
import sys
import tempfile
from unittest import mock

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# check_wsl2_memory() tests
# ---------------------------------------------------------------------------

class TestCheckWsl2Memory:
    """Verify WSL2 memory detection and warning logic."""

    def test_returns_total_gb_and_warnings_tuple(self):
        from benchmark.readiness_gate import check_wsl2_memory

        meminfo = "MemTotal:       65536000 kB\nMemFree:        10000000 kB\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=meminfo)):
            total_gb, warnings = check_wsl2_memory()
        assert isinstance(total_gb, float)
        assert isinstance(warnings, list)

    def test_parses_memtotal_correctly(self):
        from benchmark.readiness_gate import check_wsl2_memory

        # 64 GB = 67108864 kB
        meminfo = "MemTotal:       67108864 kB\nMemFree:        10000000 kB\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=meminfo)):
            total_gb, warnings = check_wsl2_memory()
        assert abs(total_gb - 64.0) < 0.1

    def test_warns_when_below_58_gb(self):
        from benchmark.readiness_gate import check_wsl2_memory

        # 48 GB = 50331648 kB
        meminfo = "MemTotal:       50331648 kB\nMemFree:        10000000 kB\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=meminfo)):
            total_gb, warnings = check_wsl2_memory()
        assert total_gb < 58
        assert len(warnings) == 1
        assert ".wslconfig" in warnings[0]
        assert "58" in warnings[0]

    def test_no_warning_at_58_gb_or_above(self):
        from benchmark.readiness_gate import check_wsl2_memory

        # 60 GB = 62914560 kB
        meminfo = "MemTotal:       62914560 kB\nMemFree:        10000000 kB\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=meminfo)):
            total_gb, warnings = check_wsl2_memory()
        assert total_gb >= 58
        assert len(warnings) == 0

    def test_handles_missing_proc_meminfo(self):
        from benchmark.readiness_gate import check_wsl2_memory

        with mock.patch("builtins.open", side_effect=FileNotFoundError):
            total_gb, warnings = check_wsl2_memory()
        assert total_gb == 0.0
        assert len(warnings) == 1
        assert "Cannot read" in warnings[0]

    def test_wslconfig_snippet_in_warning(self):
        from benchmark.readiness_gate import check_wsl2_memory

        meminfo = "MemTotal:       50331648 kB\n"
        with mock.patch("builtins.open", mock.mock_open(read_data=meminfo)):
            _, warnings = check_wsl2_memory()
        assert "[wsl2]" in warnings[0]
        assert "memory=58GB" in warnings[0]


# ---------------------------------------------------------------------------
# check_disk_space() tests
# ---------------------------------------------------------------------------

class TestCheckDiskSpace:
    """Verify disk space detection and warning logic."""

    def test_returns_free_gb_and_warnings_tuple(self):
        from benchmark.readiness_gate import check_disk_space

        usage = mock.Mock(free=100 * (1024 ** 3), total=500 * (1024 ** 3))
        with mock.patch("shutil.disk_usage", return_value=usage):
            free_gb, warnings = check_disk_space("/some/path")
        assert isinstance(free_gb, float)
        assert isinstance(warnings, list)

    def test_warns_when_below_threshold(self):
        from benchmark.readiness_gate import check_disk_space

        usage = mock.Mock(free=30 * (1024 ** 3))
        with mock.patch("shutil.disk_usage", return_value=usage):
            free_gb, warnings = check_disk_space("/some/path", min_gb=50.0)
        assert free_gb < 50
        assert len(warnings) == 1
        assert "50" in warnings[0]

    def test_no_warning_when_above_threshold(self):
        from benchmark.readiness_gate import check_disk_space

        usage = mock.Mock(free=100 * (1024 ** 3))
        with mock.patch("shutil.disk_usage", return_value=usage):
            free_gb, warnings = check_disk_space("/some/path", min_gb=50.0)
        assert free_gb >= 50
        assert len(warnings) == 0

    def test_custom_threshold(self):
        from benchmark.readiness_gate import check_disk_space

        usage = mock.Mock(free=20 * (1024 ** 3))
        with mock.patch("shutil.disk_usage", return_value=usage):
            free_gb, warnings = check_disk_space("/some/path", min_gb=25.0)
        assert len(warnings) == 1

    def test_handles_invalid_path(self):
        from benchmark.readiness_gate import check_disk_space

        with mock.patch("shutil.disk_usage", side_effect=OSError("No such path")):
            free_gb, warnings = check_disk_space("/invalid/path")
        assert free_gb == 0.0
        assert len(warnings) == 1
        assert "Cannot check disk space" in warnings[0]

    def test_default_threshold_is_50_gb(self):
        from benchmark.readiness_gate import check_disk_space

        usage = mock.Mock(free=40 * (1024 ** 3))
        with mock.patch("shutil.disk_usage", return_value=usage):
            free_gb, warnings = check_disk_space("/some/path")
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# calculate_max_steps() tests
# ---------------------------------------------------------------------------

class TestCalculateMaxSteps:
    """Verify max_steps calculation from train.bin file size."""

    def test_correct_formula(self):
        from benchmark.readiness_gate import calculate_max_steps

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a train.bin with known size
            # 1,000,000 tokens = 2,000,000 bytes (uint16)
            train_bin = os.path.join(tmpdir, "train.bin")
            with open(train_bin, "wb") as f:
                f.write(b"\x00" * 2_000_000)

            max_steps, total_tokens = calculate_max_steps(tmpdir)

        assert total_tokens == 1_000_000
        # Formula: int((1_000_000 * 3) / (32 * 1024)) = int(3_000_000 / 32768) = int(91.552) = 91
        assert max_steps == 91

    def test_returns_tuple_of_ints(self):
        from benchmark.readiness_gate import calculate_max_steps

        with tempfile.TemporaryDirectory() as tmpdir:
            train_bin = os.path.join(tmpdir, "train.bin")
            with open(train_bin, "wb") as f:
                f.write(b"\x00" * 200_000)

            max_steps, total_tokens = calculate_max_steps(tmpdir)

        assert isinstance(max_steps, int)
        assert isinstance(total_tokens, int)

    def test_large_corpus_realistic_values(self):
        """Test with a realistic corpus size (~180M tokens)."""
        from benchmark.readiness_gate import calculate_max_steps

        with tempfile.TemporaryDirectory() as tmpdir:
            # 180M tokens = 360M bytes
            # Too large to create actual file, so mock os.path.getsize
            train_bin = os.path.join(tmpdir, "train.bin")
            with open(train_bin, "wb") as f:
                f.write(b"\x00" * 10)  # Placeholder

            with mock.patch("os.path.getsize", return_value=360_000_000):
                max_steps, total_tokens = calculate_max_steps(tmpdir)

        assert total_tokens == 180_000_000
        # int((180_000_000 * 3) / (32 * 1024)) = int(540_000_000 / 32768) = 16479
        assert max_steps == 16479

    def test_uses_int_floor_not_ceil(self):
        """Verify int() truncation is used (conservative)."""
        from benchmark.readiness_gate import calculate_max_steps

        with tempfile.TemporaryDirectory() as tmpdir:
            # Choose bytes so the result is not an integer
            # 500,000 tokens = 1,000,000 bytes
            train_bin = os.path.join(tmpdir, "train.bin")
            with open(train_bin, "wb") as f:
                f.write(b"\x00" * 1_000_000)

            max_steps, total_tokens = calculate_max_steps(tmpdir)

        assert total_tokens == 500_000
        # int((500_000 * 3) / (32 * 1024)) = int(1_500_000 / 32768) = int(45.776) = 45
        assert max_steps == 45


# ---------------------------------------------------------------------------
# update_1b_config() tests
# ---------------------------------------------------------------------------

class TestUpdate1bConfig:
    """Verify configs/1b.json update preserves existing fields."""

    def _make_config(self, tmpdir):
        """Create a minimal configs/1b.json in tmpdir."""
        config = {
            "model": {
                "n_layer": 30,
                "n_head": 20,
                "n_embd": 1600,
                "block_size": 1024,
                "dropout": 0.1,
            },
            "deepspeed": {"stage": 2},
            "training": {
                "size_name": "1B",
                "max_steps": 50000,
                "batch_size": 4,
            },
            "benchmark_results": None,
            "design_doc": "docs/v2-design-doc.md",
        }
        config_path = os.path.join(tmpdir, "1b.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return config_path

    def test_populates_benchmark_results(self):
        from benchmark.readiness_gate import update_1b_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._make_config(tmpdir)
            benchmark_data = {
                "peak_gpu_gb": 8.5,
                "tokens_per_sec": 3200,
                "cpu_ram_gb": 45.0,
                "checkpoint_save_time_sec": 12.3,
                "steps_run": 500,
                "max_steps": 16479,
            }
            update_1b_config(config_path, benchmark_data)

            with open(config_path) as f:
                result = json.load(f)

        assert result["benchmark_results"] is not None
        assert result["benchmark_results"]["peak_vram_gb"] == 8.5
        assert result["benchmark_results"]["tokens_per_sec"] == 3200

    def test_updates_training_max_steps(self):
        from benchmark.readiness_gate import update_1b_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._make_config(tmpdir)
            benchmark_data = {
                "peak_gpu_gb": 8.5,
                "tokens_per_sec": 3200,
                "cpu_ram_gb": 45.0,
                "checkpoint_save_time_sec": 12.3,
                "steps_run": 500,
                "max_steps": 16479,
            }
            update_1b_config(config_path, benchmark_data)

            with open(config_path) as f:
                result = json.load(f)

        assert result["training"]["max_steps"] == 16479

    def test_preserves_model_section(self):
        from benchmark.readiness_gate import update_1b_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._make_config(tmpdir)
            benchmark_data = {
                "peak_gpu_gb": 8.5,
                "tokens_per_sec": 3200,
                "cpu_ram_gb": 45.0,
                "checkpoint_save_time_sec": 12.3,
                "steps_run": 500,
                "max_steps": 16479,
            }
            update_1b_config(config_path, benchmark_data)

            with open(config_path) as f:
                result = json.load(f)

        assert result["model"]["n_layer"] == 30
        assert result["model"]["n_head"] == 20
        assert result["model"]["n_embd"] == 1600
        assert result["model"]["dropout"] == 0.1

    def test_preserves_deepspeed_section(self):
        from benchmark.readiness_gate import update_1b_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._make_config(tmpdir)
            benchmark_data = {
                "peak_gpu_gb": 8.5,
                "tokens_per_sec": 3200,
                "cpu_ram_gb": 45.0,
                "checkpoint_save_time_sec": 12.3,
                "steps_run": 500,
                "max_steps": 16479,
            }
            update_1b_config(config_path, benchmark_data)

            with open(config_path) as f:
                result = json.load(f)

        assert result["deepspeed"]["stage"] == 2

    def test_preserves_design_doc_field(self):
        from benchmark.readiness_gate import update_1b_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._make_config(tmpdir)
            benchmark_data = {
                "peak_gpu_gb": 8.5,
                "tokens_per_sec": 3200,
                "cpu_ram_gb": 45.0,
                "checkpoint_save_time_sec": 12.3,
                "steps_run": 500,
                "max_steps": 16479,
            }
            update_1b_config(config_path, benchmark_data)

            with open(config_path) as f:
                result = json.load(f)

        assert result["design_doc"] == "docs/v2-design-doc.md"

    def test_preserves_training_batch_size(self):
        from benchmark.readiness_gate import update_1b_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._make_config(tmpdir)
            benchmark_data = {
                "peak_gpu_gb": 8.5,
                "tokens_per_sec": 3200,
                "cpu_ram_gb": 45.0,
                "checkpoint_save_time_sec": 12.3,
                "steps_run": 500,
                "max_steps": 16479,
            }
            update_1b_config(config_path, benchmark_data)

            with open(config_path) as f:
                result = json.load(f)

        assert result["training"]["batch_size"] == 4
        assert result["training"]["size_name"] == "1B"


# ---------------------------------------------------------------------------
# generate_1b_readiness_report() tests
# ---------------------------------------------------------------------------

class TestGenerate1bReadinessReport:
    """Verify structured markdown report generation."""

    def _make_inputs(self):
        env_checks = {
            "wsl2_memory_gb": 64.0,
            "wsl2_memory_warnings": [],
            "disk_free_gb": 150.0,
            "disk_warnings": [],
        }
        benchmark_results = {
            "peak_gpu_gb": 8.5,
            "tokens_per_sec": 3200,
            "cpu_ram_gb": 45.0,
            "loss_at_1": 11.5,
            "loss_at_final": 8.2,
            "checkpoint_save_time_sec": 12.3,
            "checkpoint_resume": {
                "passed": True,
                "loss_at_save": 8.2,
                "loss_at_resume": 8.5,
                "ratio": 1.04,
            },
        }
        projections = {
            "max_steps": 16479,
            "total_tokens": 180_000_000,
            "tokens_per_sec": 3200,
            "projected_hours": 46.9,
        }
        pass_fail = {
            "vram_pass": True,
            "vram_gb": 8.5,
            "vram_limit_gb": 10.0,
            "throughput_warning": False,
            "cpu_ram_warning": False,
            "overall": "PASS",
        }
        return env_checks, benchmark_results, projections, pass_fail

    def test_returns_string(self):
        from benchmark.readiness_gate import generate_1b_readiness_report

        env, bench, proj, pf = self._make_inputs()
        report = generate_1b_readiness_report(env, bench, proj, pf)
        assert isinstance(report, str)

    def test_contains_summary_section(self):
        from benchmark.readiness_gate import generate_1b_readiness_report

        env, bench, proj, pf = self._make_inputs()
        report = generate_1b_readiness_report(env, bench, proj, pf)
        assert "# 1B" in report or "Summary" in report

    def test_contains_environment_checks_section(self):
        from benchmark.readiness_gate import generate_1b_readiness_report

        env, bench, proj, pf = self._make_inputs()
        report = generate_1b_readiness_report(env, bench, proj, pf)
        assert "Environment" in report

    def test_contains_benchmark_results_section(self):
        from benchmark.readiness_gate import generate_1b_readiness_report

        env, bench, proj, pf = self._make_inputs()
        report = generate_1b_readiness_report(env, bench, proj, pf)
        assert "Benchmark" in report

    def test_contains_projected_training_time(self):
        from benchmark.readiness_gate import generate_1b_readiness_report

        env, bench, proj, pf = self._make_inputs()
        report = generate_1b_readiness_report(env, bench, proj, pf)
        assert "Projected" in report or "projected" in report

    def test_contains_preflight_checklist(self):
        from benchmark.readiness_gate import generate_1b_readiness_report

        env, bench, proj, pf = self._make_inputs()
        report = generate_1b_readiness_report(env, bench, proj, pf)
        assert "Checklist" in report or "checklist" in report

    def test_contains_pass_fail_status(self):
        from benchmark.readiness_gate import generate_1b_readiness_report

        env, bench, proj, pf = self._make_inputs()
        report = generate_1b_readiness_report(env, bench, proj, pf)
        assert "PASS" in report or "FAIL" in report


# ---------------------------------------------------------------------------
# PREFLIGHT_CHECKLIST constant tests
# ---------------------------------------------------------------------------

class TestPreflightChecklist:
    """Verify the PREFLIGHT_CHECKLIST module-level constant."""

    def test_is_list(self):
        from benchmark.readiness_gate import PREFLIGHT_CHECKLIST

        assert isinstance(PREFLIGHT_CHECKLIST, list)

    def test_has_six_items(self):
        from benchmark.readiness_gate import PREFLIGHT_CHECKLIST

        assert len(PREFLIGHT_CHECKLIST) == 6

    def test_items_are_strings(self):
        from benchmark.readiness_gate import PREFLIGHT_CHECKLIST

        for item in PREFLIGHT_CHECKLIST:
            assert isinstance(item, str)

    def test_contains_sleep_item(self):
        from benchmark.readiness_gate import PREFLIGHT_CHECKLIST

        assert any("sleep" in item.lower() for item in PREFLIGHT_CHECKLIST)

    def test_contains_windows_update_item(self):
        from benchmark.readiness_gate import PREFLIGHT_CHECKLIST

        assert any("update" in item.lower() for item in PREFLIGHT_CHECKLIST)

    def test_contains_wslconfig_item(self):
        from benchmark.readiness_gate import PREFLIGHT_CHECKLIST

        assert any(".wslconfig" in item for item in PREFLIGHT_CHECKLIST)

    def test_contains_power_item(self):
        from benchmark.readiness_gate import PREFLIGHT_CHECKLIST

        assert any("power" in item.lower() for item in PREFLIGHT_CHECKLIST)
