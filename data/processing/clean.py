"""Text cleaning pipeline for ALS corpus processing.

Transforms raw JSON documents from Phase 1 scrapers into clean plaintext by
stripping HTML/XML markup, removing volatile non-medical content, normalizing
Unicode encoding, normalizing medical abbreviations, and applying minimum
length filtering. Source-aware rules handle PubMed papers differently from
patient narratives and clinical trial records.

Functions:
    clean_document: Clean a single document dict, returning None if rejected.
    clean_all: Clean a list of documents, collecting rejections.
    load_raw_documents: Load all raw JSON documents from data/raw/.
    normalize_abbreviations: Expand and standardize ALS/medical abbreviations.
    log_rejection: Append a rejection entry to a JSONL log file.
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import ftfy
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Medical abbreviation normalization map
# ---------------------------------------------------------------------------
# Each entry: (canonical_abbrev, expanded_form, variant_patterns)
# variant_patterns is a list of regex-safe strings that match the abbreviation
# forms (case-insensitive matching with word boundaries).

ABBREVIATION_MAP = [
    {
        "canonical": "ALS",
        "expansion": "amyotrophic lateral sclerosis (ALS)",
        "variants": [r"ALS", r"als"],
    },
    {
        "canonical": "SOD1",
        "expansion": None,  # No first-occurrence expansion, just normalize form
        "variants": [r"SOD1", r"sod1", r"Cu/Zn\s+SOD"],
    },
    {
        "canonical": "FTD",
        "expansion": "frontotemporal dementia (FTD)",
        "variants": [r"FTD", r"ftd"],
    },
    {
        "canonical": "TDP-43",
        "expansion": None,
        "variants": [r"TDP-43", r"TDP43", r"tdp-43", r"tdp43"],
    },
    {
        "canonical": "FUS",
        "expansion": None,
        "variants": [r"FUS", r"fus", r"FUS/TLS"],
    },
    {
        "canonical": "C9orf72",
        "expansion": None,
        "variants": [r"C9orf72", r"C9ORF72", r"c9orf72"],
    },
    {
        "canonical": "EMG",
        "expansion": "electromyography (EMG)",
        "variants": [r"EMG", r"emg"],
    },
    {
        "canonical": "MND",
        "expansion": "motor neuron disease (MND)",
        "variants": [r"MND", r"mnd"],
    },
    {
        "canonical": "ALSFRS-R",
        "expansion": "ALS Functional Rating Scale-Revised (ALSFRS-R)",
        "variants": [r"ALSFRS-R", r"ALSFRS", r"alsfrs-r", r"alsfrs"],
    },
    {
        "canonical": "PLS",
        "expansion": "primary lateral sclerosis (PLS)",
        "variants": [r"PLS", r"pls"],
    },
    {
        "canonical": "UMN",
        "expansion": "upper motor neuron (UMN)",
        "variants": [r"UMN", r"umn"],
    },
    {
        "canonical": "LMN",
        "expansion": "lower motor neuron (LMN)",
        "variants": [r"LMN", r"lmn"],
    },
]


# ---------------------------------------------------------------------------
# Regex patterns for volatile content removal
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}"
)

TEMPORAL_QUALIFIER_PATTERN = re.compile(
    r"\bas\s+of\s+"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b"
    r"|"
    r"\blast\s+updated:?\s*[A-Z][a-z]+\s+\d{1,2},?\s*\d{4}\b"
    r"|"
    r"\bupcoming\s+.+?\s+on\s+[A-Z][a-z]+\s+\d{1,2}\b",
    re.IGNORECASE,
)

COPYRIGHT_PATTERN = re.compile(
    r"copyright\s*[\u00a9(c)]?\s*\d{4}.*?(?:\.|$)"
    r"|"
    r"all\s+rights\s+reserved\.?"
    r"|"
    r"\u00a9\s*\d{4}.*?(?:\.|$)",
    re.IGNORECASE | re.MULTILINE,
)

CTA_PATTERN = re.compile(
    r"\b(?:donate\s+now|sign\s+up\s+for\s+our\s+newsletter|"
    r"share\s+this\s+article|subscribe\s+to|click\s+here\s+to|"
    r"join\s+our\s+mailing\s+list|give\s+today|support\s+our\s+cause|"
    r"make\s+a\s+gift|contribute\s+now)\b",
    re.IGNORECASE,
)

LICENSE_TEXT_PATTERN = re.compile(
    r"(?:This\s+article\s+is\s+licensed\s+under|Creative\s+Commons).*?(?:\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# In-text citation patterns
NUMBERED_CITATION_PATTERN = re.compile(r"\[[\d,\s;\-]+\]")
AUTHOR_YEAR_CITATION_PATTERN = re.compile(
    r"\("
    r"[A-Z][a-z]+"
    r"(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?"
    r",?\s*\d{4}[a-z]?"
    r"\)"
)

# Non-content section patterns for PubMed/scientific papers
REFERENCES_SECTION_PATTERN = re.compile(
    r"\n(?:References|Bibliography|Works\s+Cited|Literature\s+Cited)\s*\n.*",
    re.IGNORECASE | re.DOTALL,
)

ACKNOWLEDGMENTS_SECTION_PATTERN = re.compile(
    r"\n(?:Acknowledgments?|Acknowledgements?)\s*\n.*?"
    r"(?=\n(?:Abstract|Introduction|Methods|Materials\s+and\s+Methods|"
    r"Results|Discussion|Conclusions?|References|Bibliography|Funding|"
    r"Financial\s+Support|Grant\s+Support)\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)

FUNDING_SECTION_PATTERN = re.compile(
    r"\n(?:Funding|Financial\s+Support|Grant\s+Support)\s*\n.*?"
    r"(?=\n(?:Abstract|Introduction|Methods|Materials\s+and\s+Methods|"
    r"Results|Discussion|Conclusions?|References|Bibliography|"
    r"Acknowledgments?|Acknowledgements?)\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)

TABLE_PATTERN = re.compile(
    r"\nTable\s+\d+\..*?\n\n",
    re.IGNORECASE | re.DOTALL,
)

FIGURE_CAPTION_PATTERN = re.compile(
    r"^(?:Figure|Fig\.?)\s+\d+.*?(?:\n\n|\n(?=[A-Z])|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

AFFILIATION_PATTERN = re.compile(
    r"^(?:Department\s+of|Faculty\s+of|School\s+of|Institute\s+of|"
    r"Center\s+for|Centre\s+for).*$",
    re.IGNORECASE | re.MULTILINE,
)

# Clinical trial status line pattern
CLINICAL_TRIAL_STATUS_PATTERN = re.compile(
    r"^(?:Status|Recruitment\s+Status|Overall\s+Status)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Preserved section headers for scientific papers
PRESERVED_HEADERS = {
    "abstract",
    "introduction",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusions",
    "conclusion",
}

# Minimum word count threshold for document acceptance
MIN_WORD_COUNT = 100


# ---------------------------------------------------------------------------
# Rejection logging
# ---------------------------------------------------------------------------

def log_rejection(
    doc_id: str,
    source: str,
    reason: str,
    details: dict,
    log_path: Path,
) -> None:
    """Append a rejection entry to a JSONL log file.

    Each rejected document is logged as a single JSON line containing the
    document ID, source, rejection reason, and any additional details.

    Args:
        doc_id: Unique identifier of the rejected document.
        source: Source name (pubmed, clinicaltrials, etc.).
        reason: Rejection reason code (too_short, empty_after_cleaning, etc.).
        details: Additional context about the rejection.
        log_path: Path to the JSONL rejection log file.
    """
    entry = {
        "id": doc_id,
        "source": source,
        "reason": reason,
        **details,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.debug("Rejected %s (%s): %s", doc_id, source, reason)


# ---------------------------------------------------------------------------
# Text cleaning functions
# ---------------------------------------------------------------------------

def _strip_html(text: str) -> str:
    """Strip residual HTML/XML tags using BeautifulSoup.

    Uses the lxml parser and preserves paragraph structure by joining
    text segments with newline separators.
    """
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator="\n")


def _strip_non_content_sections(text: str) -> str:
    """Remove non-content sections from scientific papers.

    Strips references, author affiliations, acknowledgments, funding
    sections, tables, and figure captions. Preserves section headers
    like Abstract, Introduction, Methods, Results, and Discussion.
    """
    # Remove references section (typically at end of document)
    text = REFERENCES_SECTION_PATTERN.sub("", text)

    # Remove acknowledgments
    text = ACKNOWLEDGMENTS_SECTION_PATTERN.sub("\n", text)

    # Remove funding sections
    text = FUNDING_SECTION_PATTERN.sub("\n", text)

    # Remove table content blocks
    text = TABLE_PATTERN.sub("\n\n", text)

    # Remove figure captions
    text = FIGURE_CAPTION_PATTERN.sub("", text)

    # Remove author affiliations (lines starting with department/institute)
    text = AFFILIATION_PATTERN.sub("", text)

    return text


def _strip_citations(text: str) -> str:
    """Remove in-text citations (numbered and author-year format).

    Preserves gene annotations and chemical formulas in parentheses by
    targeting only citation-like patterns.
    """
    text = NUMBERED_CITATION_PATTERN.sub("", text)
    text = AUTHOR_YEAR_CITATION_PATTERN.sub("", text)
    return text


def _remove_volatile_content(text: str) -> str:
    """Remove URLs, emails, phones, temporal qualifiers, copyright, and CTAs."""
    text = URL_PATTERN.sub("", text)
    text = EMAIL_PATTERN.sub("", text)
    text = PHONE_PATTERN.sub("", text)
    text = TEMPORAL_QUALIFIER_PATTERN.sub("", text)
    text = CTA_PATTERN.sub("", text)
    text = COPYRIGHT_PATTERN.sub("", text)
    text = LICENSE_TEXT_PATTERN.sub("", text)
    return text


def _strip_clinical_trial_status(text: str) -> str:
    """Remove volatile status lines from clinical trial documents."""
    return CLINICAL_TRIAL_STATUS_PATTERN.sub("", text)


def _scrub_pii(text: str) -> str:
    """Re-scrub PII from patient narrative text using Presidio.

    Attempts to import and use PIIScrubber from data.scrapers.pii_scrubber.
    If Presidio is not available, logs a warning and returns text unchanged
    (Phase 1 already performed one scrubbing pass).
    """
    try:
        # Import path works whether running as module or standalone
        try:
            from data.scrapers.pii_scrubber import PIIScrubber
        except ImportError:
            # Fallback for standalone execution
            parent = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(parent))
            from scrapers.pii_scrubber import PIIScrubber

        scrubber = PIIScrubber()
        return scrubber.scrub(text)
    except ImportError as e:
        logger.warning(
            "Presidio not available for PII re-scrubbing: %s. "
            "Continuing with Phase 1 scrubbing only.",
            e,
        )
        return text
    except Exception as e:
        logger.warning(
            "PII scrubbing failed: %s. Continuing with Phase 1 scrubbing only.",
            e,
        )
        return text


def _normalize_unicode(text: str) -> str:
    """Fix mojibake with ftfy and normalize to NFC form.

    NFC (Canonical Decomposition followed by Canonical Composition) preserves
    Greek letters, superscripts/subscripts, and math symbols. NFKC would
    destructively normalize these, so we use NFC only.
    """
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks.

    Collapses 3+ consecutive newlines to double newlines (paragraph breaks),
    collapses multiple spaces/tabs to single space, and strips leading/trailing
    whitespace from each line.
    """
    # Strip leading/trailing whitespace from each line
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    text = "\n".join(lines)

    # Collapse multiple spaces and tabs to single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines to double newlines (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_abbreviations(text: str) -> str:
    """Normalize common ALS/medical abbreviations for corpus consistency.

    On first occurrence of each abbreviation in the document, expands it
    to its full form if the expansion is not already present within 200
    characters before the match. On subsequent occurrences, normalizes
    to the canonical abbreviation form.

    Args:
        text: Cleaned document text after whitespace normalization.

    Returns:
        Text with normalized abbreviations.
    """
    for abbrev_entry in ABBREVIATION_MAP:
        canonical = abbrev_entry["canonical"]
        expansion = abbrev_entry["expansion"]
        variants = abbrev_entry["variants"]

        # Build a single pattern matching all variants
        variant_pattern = "|".join(variants)
        pattern = re.compile(
            r"\b(?:" + variant_pattern + r")\b",
            re.IGNORECASE,
        )

        first_occurrence = True
        result_parts = []
        last_end = 0

        for match in pattern.finditer(text):
            result_parts.append(text[last_end:match.start()])

            if first_occurrence and expansion is not None:
                # Check if the expansion already appears within 200 chars before
                lookback_start = max(0, match.start() - 200)
                preceding_text = text[lookback_start:match.start()].lower()
                expansion_key = expansion.split("(")[0].strip().lower()

                if expansion_key not in preceding_text:
                    result_parts.append(expansion)
                else:
                    result_parts.append(canonical)
                first_occurrence = False
            elif first_occurrence:
                # No expansion needed, just normalize form
                result_parts.append(canonical)
                first_occurrence = False
            else:
                # Subsequent occurrences: use canonical form
                result_parts.append(canonical)

            last_end = match.end()

        result_parts.append(text[last_end:])
        text = "".join(result_parts)

    return text


def _embed_title(text: str, title: str) -> str:
    """Prefix cleaned text with the document title as a header line.

    Args:
        text: The cleaned document body text.
        title: The document title.

    Returns:
        Text with title on first line, blank line, then body.
    """
    if title and title.strip():
        return f"{title.strip()}\n\n{text}"
    return text


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_document(
    doc: dict,
    rejected_path: Path | None = None,
) -> dict | None:
    """Apply all cleaning transformations to a raw document.

    Takes a raw document dict from Phase 1 scrapers and returns a cleaned
    document dict, or None if the document is rejected (too short, empty
    after cleaning, etc.).

    Cleaning steps are applied in this order:
    1. Strip residual HTML/XML
    2. Strip non-content sections (PubMed/scientific papers only)
    3. Strip in-text citations
    4. Remove volatile content (URLs, emails, phones, copyright, CTAs)
    5. Strip clinical trial status lines (clinical_trials source only)
    6. PII re-scrubbing (patient narratives only)
    7. Unicode normalization (ftfy + NFC)
    8. Whitespace normalization
    9. Medical abbreviation normalization
    10. Embed document title as header
    11. Minimum length check

    Args:
        doc: Raw document dict with Phase 1 JSON schema fields.
        rejected_path: Path to rejection log. If None, rejections are
            logged to logger only.

    Returns:
        Cleaned document dict with updated text and word_count, or None
        if the document was rejected.
    """
    doc_id = doc.get("id", "unknown")
    source = doc.get("source", "unknown")
    source_category = doc.get("source_category", "")
    title = doc.get("title", "")
    text = doc.get("text", "")

    if not text or not text.strip():
        if rejected_path:
            log_rejection(doc_id, source, "empty_text", {}, rejected_path)
        return None

    # Step 1: Strip residual HTML/XML
    text = _strip_html(text)

    # Step 2: Strip non-content sections for scientific papers
    is_scientific = (
        source_category == "research_papers"
        or source == "pubmed"
    )
    if is_scientific:
        text = _strip_non_content_sections(text)

    # Step 3: Strip in-text citations
    text = _strip_citations(text)

    # Step 4: Remove volatile content
    text = _remove_volatile_content(text)

    # Step 5: Strip clinical trial status lines
    if source == "clinical_trials" or source == "clinicaltrials":
        text = _strip_clinical_trial_status(text)

    # Step 6: PII re-scrubbing for patient narratives
    is_patient_narrative = (
        "patient" in source_category.lower()
        if source_category
        else False
    ) or source == "patient_narratives"

    if is_patient_narrative:
        text = _scrub_pii(text)
        # Also scrub the title for patient narratives
        if title:
            title = _scrub_pii(title)

    # Step 7: Unicode normalization
    text = _normalize_unicode(text)

    # Step 8: Whitespace normalization
    text = _normalize_whitespace(text)

    # Step 9: Medical abbreviation normalization
    text = normalize_abbreviations(text)

    # Step 10: Embed document title as header
    text = _embed_title(text, title)

    # Step 11: Minimum length check
    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        if rejected_path:
            log_rejection(
                doc_id,
                source,
                "too_short",
                {"word_count": word_count, "threshold": MIN_WORD_COUNT},
                rejected_path,
            )
        return None

    # Build cleaned document dict
    cleaned = dict(doc)
    cleaned["text"] = text
    cleaned["word_count"] = word_count

    return cleaned


# ---------------------------------------------------------------------------
# Batch processing and I/O
# ---------------------------------------------------------------------------

def load_raw_documents(raw_dir: Path) -> list[dict]:
    """Load all raw JSON documents from data/raw/{source}/ directories.

    Walks the raw data directory structure, loading every .json file and
    returning a flat list of document dicts.

    Args:
        raw_dir: Path to the raw data root directory (e.g., data/raw/).

    Returns:
        List of document dicts loaded from all JSON files.
    """
    documents = []
    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        logger.warning("Raw data directory does not exist: %s", raw_dir)
        return documents

    for source_dir in sorted(raw_dir.iterdir()):
        if not source_dir.is_dir():
            continue

        json_files = sorted(source_dir.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                documents.append(doc)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error("Failed to load %s: %s", json_file, e)

    logger.info(
        "Loaded %d raw documents from %s",
        len(documents),
        raw_dir,
    )
    return documents


def clean_all(
    documents: list[dict],
    rejected_path: Path,
) -> list[dict]:
    """Apply clean_document to all documents and collect survivors.

    Args:
        documents: List of raw document dicts.
        rejected_path: Path to the JSONL rejection log file.

    Returns:
        List of cleaned document dicts (rejected documents excluded).
    """
    cleaned = []
    rejection_counts: dict[str, int] = {}

    for doc in documents:
        result = clean_document(doc, rejected_path=rejected_path)
        if result is not None:
            cleaned.append(result)

    # Count rejections by reading the log
    if rejected_path.exists():
        with open(rejected_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    reason = entry.get("reason", "unknown")
                    rejection_counts[reason] = (
                        rejection_counts.get(reason, 0) + 1
                    )
                except json.JSONDecodeError:
                    continue

    logger.info(
        "Cleaning complete: %d input, %d output, %d rejected",
        len(documents),
        len(cleaned),
        len(documents) - len(cleaned),
    )
    if rejection_counts:
        for reason, count in sorted(rejection_counts.items()):
            logger.info("  Rejected (%s): %d", reason, count)

    return cleaned


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run the cleaning pipeline from the command line.

    Loads raw JSON documents from data/raw/, cleans them, and writes the
    cleaned documents as JSON to stdout or a specified output file. Reports
    statistics on input count, output count, and rejections by reason.
    """
    parser = argparse.ArgumentParser(
        description="Clean raw ALS corpus documents",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for cleaned documents (default: stdout)",
    )
    parser.add_argument(
        "--rejected",
        type=Path,
        default=Path("data/processed/rejected.jsonl"),
        help="Path to rejection log (default: data/processed/rejected.jsonl)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Ensure output directories exist
    args.rejected.parent.mkdir(parents=True, exist_ok=True)

    # Load raw documents
    documents = load_raw_documents(args.raw_dir)
    if not documents:
        logger.warning("No documents found in %s", args.raw_dir)
        return

    # Clean all documents
    cleaned = clean_all(documents, args.rejected)

    # Write output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %d cleaned documents to %s", len(cleaned), args.output)
    else:
        json.dump(cleaned, sys.stdout, indent=2, ensure_ascii=False)

    # Report statistics
    print(f"\n--- Cleaning Statistics ---", file=sys.stderr)
    print(f"Input documents:  {len(documents)}", file=sys.stderr)
    print(f"Output documents: {len(cleaned)}", file=sys.stderr)
    print(f"Rejected:         {len(documents) - len(cleaned)}", file=sys.stderr)

    if args.rejected.exists():
        reasons: dict[str, int] = {}
        with open(args.rejected, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    r = entry.get("reason", "unknown")
                    reasons[r] = reasons.get(r, 0) + 1
                except json.JSONDecodeError:
                    continue
        if reasons:
            print("Rejections by reason:", file=sys.stderr)
            for reason, count in sorted(reasons.items()):
                print(f"  {reason}: {count}", file=sys.stderr)


if __name__ == "__main__":
    main()
