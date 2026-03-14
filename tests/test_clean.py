"""Unit tests for data.processing.clean normalization functions.

Tests cover punctuation canonicalization (ligatures, dash variants, exotic
whitespace, zero-width characters), line-break rejoining (mid-word and
hyphenated breaks), and medical term survival through the full pipeline.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so data package is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.processing.clean import (
    ABBREVIATION_MAP,
    _canonicalize_punctuation,
    _rejoin_line_breaks,
    clean_document,
)


# ---------------------------------------------------------------------------
# _canonicalize_punctuation: ligature decomposition
# ---------------------------------------------------------------------------

class TestLigatureDecomposition:
    """Ligatures from PDF extraction are decomposed to ASCII sequences."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            ("\ufb00ect", "ffect"),               # ff ligature
            ("\ufb01broblast", "fibroblast"),   # fi ligature
            ("\ufb02uoride", "fluoride"),         # fl ligature
            ("\ufb03ce", "ffice"),                 # ffi ligature
            ("\ufb04ower", "fflower"),             # ffl ligature
        ],
        ids=["ff-ligature", "fi-ligature", "fl-ligature", "ffi-ligature", "ffl-ligature"],
    )
    def test_ligature(self, input_text, expected):
        assert _canonicalize_punctuation(input_text) == expected


# ---------------------------------------------------------------------------
# _canonicalize_punctuation: dash variant normalization
# ---------------------------------------------------------------------------

class TestDashNormalization:
    """Figure dash and minus sign normalize to hyphen-minus; em/en dash preserved."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            ("TDP\u201243", "TDP-43"),          # figure dash -> hyphen
            ("ALSFRS\u2012R", "ALSFRS-R"),      # figure dash in ALSFRS-R
            ("SOD1\u2212G93A", "SOD1-G93A"),    # minus sign -> hyphen
            ("em\u2014dash", "em\u2014dash"),    # em dash preserved
            ("en\u2013dash", "en\u2013dash"),    # en dash preserved
        ],
        ids=["figure-dash", "figure-dash-alsfrs-r", "minus-sign", "em-dash-preserved", "en-dash-preserved"],
    )
    def test_dash(self, input_text, expected):
        assert _canonicalize_punctuation(input_text) == expected


# ---------------------------------------------------------------------------
# _canonicalize_punctuation: exotic whitespace normalization
# ---------------------------------------------------------------------------

class TestExoticWhitespace:
    """Exotic whitespace variants are normalized to ASCII space."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            ("test\u00a0text", "test text"),    # non-breaking space
            ("test\u2009text", "test text"),    # thin space
            ("test\u2007text", "test text"),    # figure space
            ("test\u202ftext", "test text"),    # narrow no-break space
            ("test\u205ftext", "test text"),    # medium math space
            ("test\u3000text", "test text"),    # ideographic space
        ],
        ids=[
            "non-breaking-space",
            "thin-space",
            "figure-space",
            "narrow-no-break-space",
            "medium-math-space",
            "ideographic-space",
        ],
    )
    def test_whitespace(self, input_text, expected):
        assert _canonicalize_punctuation(input_text) == expected


# ---------------------------------------------------------------------------
# _canonicalize_punctuation: zero-width character removal
# ---------------------------------------------------------------------------

class TestZeroWidthRemoval:
    """Zero-width characters and BOM are removed entirely."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            ("zero\u200bwidth", "zerowidth"),    # zero-width space
            ("zero\u200cwidth", "zerowidth"),    # zero-width non-joiner
            ("zero\u200dwidth", "zerowidth"),    # zero-width joiner
            ("bom\ufefftext", "bomtext"),         # BOM
        ],
        ids=["zwsp", "zwnj", "zwj", "bom"],
    )
    def test_zero_width(self, input_text, expected):
        assert _canonicalize_punctuation(input_text) == expected


# ---------------------------------------------------------------------------
# _rejoin_line_breaks
# ---------------------------------------------------------------------------

class TestRejoinLineBreaks:
    """Hyphenated line breaks are rejoined; hyphen is preserved."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            # Rejoined cases (hyphen kept, newline removed)
            ("neuro-\ndegenerative", "neuro-degenerative"),
            ("receptor-\nmediated", "receptor-mediated"),
            ("dose-\ndependent", "dose-dependent"),
            # NOT rejoined cases (safety)
            ("SOD1-\nmutant", "SOD1-\nmutant"),          # digit before hyphen
            ("SOD1A-\nmutant", "SOD1A-\nmutant"),        # digit-letter before hyphen
            ("FUS-\nrelated", "FUS-\nrelated"),            # uppercase abbrev
            ("ALS-\nassociated", "ALS-\nassociated"),      # uppercase abbrev
            ("word\nCapitalized", "word\nCapitalized"),   # no hyphen, not rejoined
            ("neuro\ndegenerative", "neuro\ndegenerative"),  # no hyphen, not rejoined
            ("end.\nStart", "end.\nStart"),               # punctuation end
        ],
        ids=[
            "hyphenated-break-rejoined",
            "compound-word-preserved",
            "compound-word-dose-dependent",
            "digit-hyphen-preserved",
            "digit-letter-hyphen-preserved",
            "uppercase-abbrev-FUS-preserved",
            "uppercase-abbrev-ALS-preserved",
            "no-hyphen-not-rejoined",
            "soft-break-not-rejoined",
            "punctuation-end-preserved",
        ],
    )
    def test_line_break(self, input_text, expected):
        assert _rejoin_line_breaks(input_text) == expected


# ---------------------------------------------------------------------------
# Medical term survival through full pipeline
# ---------------------------------------------------------------------------

class TestMedicalTermSurvival:
    """All 12 ABBREVIATION_MAP canonical terms survive clean_document()."""

    def _build_synthetic_document(self):
        """Build a document containing all 12 canonical terms.

        Uses figure dashes, minus signs, and a hyphenated line break to
        verify that punctuation canonicalization and line-break rejoining
        do not corrupt medical abbreviations.
        """
        # Build a long enough document (>100 words) containing all terms.
        # Paragraphs are newline-separated (not space-joined) so that
        # _rejoin_line_breaks is exercised by the pipeline.
        canonical_terms = [entry["canonical"] for entry in ABBREVIATION_MAP]
        paragraphs = [
            "This is a research document about amyotrophic lateral sclerosis.",
            "The study investigates ALS and its relationship to SOD1 mutations.",
            "Patients with frontotemporal dementia or FTD often show overlap.",
            # Use figure dash in TDP-43 to test canonicalization
            f"The protein TDP\u201243 is a key biomarker in ALS research.",
            "FUS protein aggregation is another hallmark of the disease.",
            "The C9orf72 repeat expansion is the most common genetic cause.",
            "Electromyography or EMG is used for diagnosis.",
            "Motor neuron disease or MND is an alternative name for ALS.",
            # Hyphenated line break: SOD1-mutant must survive intact
            "The ALSFRS-R scale measures functional decline. SOD1-\nmutant "
            "mice are commonly used in preclinical studies over time.",
            "Primary lateral sclerosis or PLS is a related condition.",
            "Upper motor neuron or UMN signs include spasticity.",
            "Lower motor neuron or LMN signs include muscle atrophy.",
            "The relationship between these biomarkers and clinical outcomes "
            "has been studied extensively in multiple randomized controlled "
            "trials across different populations and geographic regions.",
        ]
        text = "\n".join(paragraphs)
        return {
            "id": "test-survival-001",
            "source": "pubmed",
            "source_category": "research_papers",
            "title": "Medical Term Survival Test",
            "text": text,
        }

    def test_all_canonical_terms_survive(self):
        """Every canonical term from ABBREVIATION_MAP appears in output."""
        doc = self._build_synthetic_document()
        result = clean_document(doc)
        assert result is not None, "Document was unexpectedly rejected"

        output_text = result["text"]
        for entry in ABBREVIATION_MAP:
            canonical = entry["canonical"]
            assert canonical in output_text, (
                f"Canonical term '{canonical}' missing from cleaned output"
            )
