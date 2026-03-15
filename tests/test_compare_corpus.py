"""Unit tests for data.processing.compare_corpus parsing and formatting."""

import json
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.processing.compare_corpus import (
    _fmt_bytes_mb,
    _fmt_change,
    _get_size_advisory,
    _extract_section,
    _parse_number,
    _parse_size_mb,
    _parse_table_rows,
    parse_stats,
    snapshot_baseline,
    generate_comparison_report,
)


# ---------------------------------------------------------------------------
# _parse_number
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_plain_integer(self):
        assert _parse_number("35088") == 35088

    def test_comma_separated(self):
        assert _parse_number("35,088") == 35088

    def test_tilde_prefix(self):
        assert _parse_number("~176,547,868") == 176547868

    def test_whitespace(self):
        assert _parse_number("  1,000  ") == 1000

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_number("N/A")

    def test_zero(self):
        assert _parse_number("0") == 0


# ---------------------------------------------------------------------------
# _parse_size_mb
# ---------------------------------------------------------------------------

class TestParseSizeMb:
    def test_megabytes(self):
        assert _parse_size_mb("678.05 MB") == pytest.approx(678.05)

    def test_gigabytes(self):
        assert _parse_size_mb("1.5 GB") == pytest.approx(1536.0)

    def test_kilobytes(self):
        assert _parse_size_mb("1024 KB") == pytest.approx(1.0)

    def test_bytes(self):
        result = _parse_size_mb("1048576 B")
        assert result == pytest.approx(1.0)

    def test_invalid_format(self):
        assert _parse_size_mb("invalid") == 0.0

    def test_empty_string(self):
        assert _parse_size_mb("") == 0.0


# ---------------------------------------------------------------------------
# _extract_section
# ---------------------------------------------------------------------------

class TestExtractSection:
    SAMPLE_MD = (
        "# Title\n\n"
        "## First section\n\n"
        "Content of first section.\n\n"
        "## Second section\n\n"
        "Content of second section.\n"
    )

    def test_extracts_section(self):
        result = _extract_section(self.SAMPLE_MD, "First section")
        assert "Content of first section." in result
        assert "Content of second section." not in result

    def test_last_section(self):
        result = _extract_section(self.SAMPLE_MD, "Second section")
        assert "Content of second section." in result

    def test_missing_section(self):
        assert _extract_section(self.SAMPLE_MD, "Nonexistent") == ""


# ---------------------------------------------------------------------------
# _parse_table_rows
# ---------------------------------------------------------------------------

class TestParseTableRows:
    def test_single_table(self):
        text = (
            "| Name   | Value |\n"
            "|--------|-------|\n"
            "| alpha  | 1     |\n"
            "| beta   | 2     |\n"
        )
        rows = _parse_table_rows(text)
        assert len(rows) == 2
        assert rows[0] == {"Name": "alpha", "Value": "1"}
        assert rows[1] == {"Name": "beta", "Value": "2"}

    def test_first_table_only(self):
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "\n"
            "Some text\n"
            "\n"
            "| C | D |\n"
            "|---|---|\n"
            "| 3 | 4 |\n"
        )
        rows = _parse_table_rows(text, first_table_only=True)
        assert len(rows) == 1
        assert rows[0] == {"A": "1", "B": "2"}

    def test_empty_input(self):
        assert _parse_table_rows("") == []


# ---------------------------------------------------------------------------
# _fmt_change
# ---------------------------------------------------------------------------

class TestFmtChange:
    def test_both_zero(self):
        assert _fmt_change(0, 0) == "—"

    def test_new_from_zero(self):
        assert _fmt_change(0, 100) == "NEW"

    def test_positive_int_change(self):
        result = _fmt_change(100, 110)
        assert "+10" in result
        assert "+10.0%" in result

    def test_negative_int_change(self):
        result = _fmt_change(100, 90)
        assert "-10" in result

    def test_float_change(self):
        result = _fmt_change(100.0, 105.5)
        assert "+5.5" in result

    def test_percentage_point_mode(self):
        result = _fmt_change(10.0, 12.5, is_pct=True)
        assert "+2.5 pp" == result


# ---------------------------------------------------------------------------
# _get_size_advisory
# ---------------------------------------------------------------------------

class TestGetSizeAdvisory:
    def test_target_met_at_boundary(self):
        assert _get_size_advisory(80.0) == "Target met"

    def test_target_met_above_150(self):
        assert _get_size_advisory(200.0) == "Target met"

    def test_minimum_met(self):
        assert _get_size_advisory(50.0) == "Minimum met"

    def test_below_minimum(self):
        assert _get_size_advisory(10.0) == "Below minimum"

    def test_at_32_boundary(self):
        assert _get_size_advisory(32.0) == "Below minimum"

    def test_just_above_32(self):
        assert _get_size_advisory(32.1) == "Minimum met"


# ---------------------------------------------------------------------------
# _fmt_bytes_mb
# ---------------------------------------------------------------------------

class TestFmtBytesMb:
    def test_one_mb(self):
        assert _fmt_bytes_mb(1024 * 1024) == "1.00 MB"

    def test_zero(self):
        assert _fmt_bytes_mb(0) == "0.00 MB"


# ---------------------------------------------------------------------------
# parse_stats (integration with file I/O)
# ---------------------------------------------------------------------------

class TestParseStats:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_stats(tmp_path / "nonexistent.md")

    def test_parses_minimal_stats(self, tmp_path):
        stats_md = tmp_path / "stats.md"
        stats_md.write_text(
            "## Corpus summary\n\n"
            "| Metric          | Value     |\n"
            "|-----------------|-----------|\n"
            "| Total documents | 1,000     |\n"
            "| Total words     | 500,000   |\n"
            "\n"
            "## Source distribution\n\n"
            "| Source category     | Documents | Word count | Percentage |\n"
            "|---------------------|-----------|------------|------------|\n"
            "| biomedical_research | 800       | 400,000    | 80%        |\n"
            "\n"
            "## Rejection summary\n\n"
            "| Reason         | Count |\n"
            "|----------------|-------|\n"
            "| near_duplicate | 50    |\n"
            "\n"
            "## Document length analysis\n\n"
            "| Metric  | Words |\n"
            "|---------|---------|\n"
            "| Average | 500     |\n"
            "| Median  | 300     |\n",
            encoding="utf-8",
        )
        result = parse_stats(stats_md)

        assert result["corpus_summary"]["Total documents"] == "1,000"
        assert result["source_distribution"]["biomedical_research"]["documents"] == 800
        assert result["rejection_summary"]["near_duplicate"] == 50
        assert result["document_length"]["Average"] == 500


# ---------------------------------------------------------------------------
# snapshot_baseline
# ---------------------------------------------------------------------------

class TestSnapshotBaseline:
    def test_creates_snapshot(self, tmp_path):
        stats = tmp_path / "stats.md"
        stats.write_text("## Corpus summary\n\ncontent\n", encoding="utf-8")
        train = tmp_path / "train.txt"
        train.write_text("train data", encoding="utf-8")
        val = tmp_path / "val.txt"
        val.write_text("val data", encoding="utf-8")
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        snapshot_dir = tmp_path / "snapshot"

        result = snapshot_baseline(stats, train, val, raw_dir, snapshot_dir)

        assert result is True
        assert (snapshot_dir / "stats.md").exists()
        sizes = json.loads((snapshot_dir / "file_sizes.json").read_text())
        assert sizes["train_bytes"] == len("train data")
        assert "raw_counts" in sizes

    def test_idempotent_skip(self, tmp_path):
        stats = tmp_path / "stats.md"
        stats.write_text("content", encoding="utf-8")
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()

        result = snapshot_baseline(
            stats, tmp_path / "t.txt", tmp_path / "v.txt", tmp_path, snapshot_dir,
        )
        assert result is False

    def test_missing_stats_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            snapshot_baseline(
                tmp_path / "missing.md",
                tmp_path / "t.txt",
                tmp_path / "v.txt",
                tmp_path,
                tmp_path / "snapshot",
            )


# ---------------------------------------------------------------------------
# generate_comparison_report (version parameterization)
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def _make_stats_md(self, path):
        path.write_text(
            "## Corpus summary\n\n"
            "| Metric          | Value     |\n"
            "|-----------------|-----------|\n"
            "| Total documents | 1,000     |\n"
            "| Total words     | 500,000   |\n"
            "| Total size      | 100 MB    |\n"
            "\n"
            "## Source distribution\n\n"
            "| Source category     | Documents | Word count | Percentage |\n"
            "|---------------------|-----------|------------|------------|\n"
            "| biomedical_research | 800       | 400,000    | 80%        |\n"
            "\n"
            "## Rejection summary\n\n"
            "| Reason         | Count |\n"
            "|----------------|-------|\n"
            "| near_duplicate | 50    |\n"
            "\n"
            "## Document length analysis\n\n"
            "| Metric  | Words |\n"
            "|---------|-------|\n"
            "| Average | 500   |\n"
            "| Median  | 300   |\n",
            encoding="utf-8",
        )

    def test_uses_version_labels(self, tmp_path):
        baseline_dir = tmp_path / "baseline"
        baseline_dir.mkdir()
        self._make_stats_md(baseline_dir / "stats.md")
        sizes = {"train_bytes": 1024, "val_bytes": 512, "raw_counts": {}}
        (baseline_dir / "file_sizes.json").write_text(
            json.dumps(sizes), encoding="utf-8",
        )

        current_stats = tmp_path / "stats.md"
        self._make_stats_md(current_stats)

        output = tmp_path / "report.md"
        generate_comparison_report(
            baseline_dir=baseline_dir,
            stats_path=current_stats,
            raw_dir=tmp_path / "raw",
            train_path=tmp_path / "train.txt",
            val_path=tmp_path / "val.txt",
            output_path=output,
            baseline_version="v2.0.0",
            current_version="v3.0.0",
        )

        content = output.read_text(encoding="utf-8")
        assert "v2.0.0 vs v3.0.0" in content
        assert "v1.0.0" not in content

    def test_missing_baseline_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Baseline stats"):
            generate_comparison_report(
                baseline_dir=tmp_path / "missing",
                stats_path=tmp_path / "stats.md",
                raw_dir=tmp_path,
                train_path=tmp_path / "t.txt",
                val_path=tmp_path / "v.txt",
                output_path=tmp_path / "out.md",
            )
