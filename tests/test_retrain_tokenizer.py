"""Unit tests for retrain_tokenizer.py new/modified functions.

Tests cover compute_tokenizer_hash, check_degradation, modified step3
(custom vocab_sizes), step5 (single-candidate), and step8 (meta.pkl
hash fields).
"""

import json
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path so scripts package is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_scripts_dir = _project_root / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from retrain_tokenizer import compute_tokenizer_hash, check_degradation


# ---------------------------------------------------------------------------
# compute_tokenizer_hash
# ---------------------------------------------------------------------------
class TestComputeTokenizerHash:
    """Tests for the SHA-256 tokenizer hashing function."""

    def test_returns_consistent_hex_string(self, tmp_path):
        """Hash of a known file returns a 64-char hex string."""
        f = tmp_path / "tok.json"
        f.write_text('{"version": "1.0", "vocab": {}}', encoding="utf-8")
        h = compute_tokenizer_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64
        # Should be stable across calls
        assert h == compute_tokenizer_hash(f)

    def test_different_content_different_hash(self, tmp_path):
        """Different file content produces different hash."""
        f1 = tmp_path / "tok1.json"
        f2 = tmp_path / "tok2.json"
        f1.write_text('{"version": "1.0"}', encoding="utf-8")
        f2.write_text('{"version": "2.0"}', encoding="utf-8")
        assert compute_tokenizer_hash(f1) != compute_tokenizer_hash(f2)


# ---------------------------------------------------------------------------
# check_degradation
# ---------------------------------------------------------------------------
class TestCheckDegradation:
    """Tests for the degradation comparison function."""

    def _mock_tokenizer(self, term_to_tokens: dict):
        """Create a mock tokenizer that returns specified token counts."""
        tok = MagicMock()

        def encode_side(text):
            result = MagicMock()
            result.ids = list(range(term_to_tokens.get(text, 1)))
            return result

        tok.encode = encode_side
        return tok

    def test_no_degradation(self):
        """Returns empty list when no terms degrade."""
        terms = [
            {"term": "riluzole", "category": "drug"},
            {"term": "SOD1", "category": "gene"},
        ]
        old_tok = self._mock_tokenizer({"riluzole": 2, "SOD1": 3})
        new_tok = self._mock_tokenizer({"riluzole": 1, "SOD1": 2})
        result = check_degradation(new_tok, old_tok, terms)
        assert result == []

    def test_with_degradation(self):
        """Returns degraded terms when new tokenizer uses more subtokens."""
        terms = [
            {"term": "riluzole", "category": "drug"},
            {"term": "SOD1", "category": "gene"},
            {"term": "edaravone", "category": "drug"},
        ]
        # SOD1 degrades: old=2, new=4
        old_tok = self._mock_tokenizer({"riluzole": 2, "SOD1": 2, "edaravone": 3})
        new_tok = self._mock_tokenizer({"riluzole": 1, "SOD1": 4, "edaravone": 2})
        result = check_degradation(new_tok, old_tok, terms)
        assert len(result) == 1
        assert result[0]["term"] == "SOD1"
        assert result[0]["old_subtokens"] == 2
        assert result[0]["new_subtokens"] == 4
        assert result[0]["category"] == "gene"

    def test_threshold_warning(self, capsys):
        """Prints warning when degraded count exceeds threshold."""
        terms = [{"term": f"term_{i}", "category": "test"} for i in range(15)]
        # All 15 terms degrade (old=1, new=3)
        old_tok = self._mock_tokenizer({t["term"]: 1 for t in terms})
        new_tok = self._mock_tokenizer({t["term"]: 3 for t in terms})
        result = check_degradation(new_tok, old_tok, terms, threshold=10)
        assert len(result) == 15
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "warning" in captured.out.lower()


# ---------------------------------------------------------------------------
# step3 with custom vocab_sizes
# ---------------------------------------------------------------------------
class TestStep3CustomVocabSizes:
    """Tests for step3_retrain_tokenizers with configurable vocab sizes."""

    @patch("retrain_tokenizer.train_single_tokenizer")
    def test_custom_vocab_sizes(self, mock_train, tmp_path):
        """step3 with vocab_sizes=[50257] calls train_single_tokenizer once."""
        from retrain_tokenizer import step3_retrain_tokenizers

        mock_train.return_value = {"vocab_size": 50257}
        step3_retrain_tokenizers(
            corpus_path=tmp_path / "train.txt",
            tokenizer_dir=tmp_path,
            vocab_sizes=[50257],
            dry_run=False,
        )
        mock_train.assert_called_once()
        call_args = mock_train.call_args
        assert call_args[1].get("vocab_size", call_args[0][2] if len(call_args[0]) > 2 else None) == 50257 or \
               50257 in call_args[0]


# ---------------------------------------------------------------------------
# step5 single candidate
# ---------------------------------------------------------------------------
class TestStep5SingleCandidate:
    """Tests for step5_select_winner with a single candidate."""

    def test_single_candidate_copies_directly(self, tmp_path):
        """Single tokenizer file is copied directly to als_tokenizer.json."""
        from retrain_tokenizer import step5_select_winner

        tok_content = '{"version": "1.0", "model": {"type": "BPE"}}'
        (tmp_path / "als_tokenizer_50k.json").write_text(tok_content)

        with patch("retrain_tokenizer.Tokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.get_vocab_size.return_value = 50257
            mock_tok.encode.return_value = MagicMock(ids=[1])
            mock_tok_cls.from_file.return_value = mock_tok

            result = step5_select_winner(
                tmp_path,
                terms=[{"term": "test", "category": "test"}],
            )

        canonical = tmp_path / "als_tokenizer.json"
        assert canonical.exists()


# ---------------------------------------------------------------------------
# step8 meta.pkl hash fields
# ---------------------------------------------------------------------------
class TestStep8MetaPklHash:
    """Tests for step8_retokenize_corpus meta.pkl hash/version injection."""

    def test_meta_pkl_contains_hash_fields(self, tmp_path):
        """step8 produces meta.pkl with tokenizer_hash and tokenizer_version."""
        from retrain_tokenizer import step8_retokenize_corpus

        # Create minimal corpus files
        train_txt = tmp_path / "train.txt"
        val_txt = tmp_path / "val.txt"
        train_txt.write_text("hello world test", encoding="utf-8")
        val_txt.write_text("hello test", encoding="utf-8")

        # Create a minimal tokenizer JSON
        tok_dir = tmp_path / "tokenizer"
        tok_dir.mkdir()
        tok_path = tok_dir / "als_tokenizer.json"

        with patch("retrain_tokenizer.Tokenizer") as mock_tok_cls, \
             patch("retrain_tokenizer.compute_tokenizer_hash") as mock_hash:
            mock_tok = MagicMock()
            mock_tok.get_vocab_size.return_value = 100
            mock_tok.id_to_token.side_effect = lambda i: f"tok_{i}" if i < 100 else None
            mock_enc = MagicMock()
            mock_enc.ids = [1, 2, 3]
            mock_tok.encode.return_value = mock_enc
            mock_tok_cls.from_file.return_value = mock_tok
            mock_hash.return_value = "a" * 64

            # Create a dummy tokenizer file so the path exists
            tok_path.write_text("{}")

            output_dir = tmp_path / "output"
            step8_retokenize_corpus(tok_dir, train_txt, val_txt, output_dir)

        meta_path = output_dir / "meta.pkl"
        assert meta_path.exists()
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        assert "tokenizer_hash" in meta
        assert "tokenizer_version" in meta
        assert isinstance(meta["tokenizer_hash"], str)
        assert len(meta["tokenizer_hash"]) == 64
        assert isinstance(meta["tokenizer_version"], str)
