#!/usr/bin/env python3
"""Detect fabricated entities in model responses using a training-corpus registry.

Extracts drug names, gene names, and clinical trial NCT IDs from generated
model responses and checks each entity against the training-corpus entity
registry. Entities not found in the registry are flagged as potentially
fabricated, with surrounding sentence context for manual review.

This is a research evaluation tool, not a medical information system.

Methodology:
    - NCT trial IDs: regex extraction (NCT followed by 6-8 digits), exact
      match against registry trial entries.
    - Drug names: candidate extraction via capitalized words and known drug
      suffixes, fuzzy matching against the full drug registry using
      rapidfuzz.process.extractOne with a configurable threshold (default 85).
    - Gene names: candidate extraction via uppercase+digit patterns (2-10
      chars), fuzzy matching against the gene registry.
    - Binary flagging: an entity is either in the registry or flagged. No
      confidence tiers. Flagged entities may include false positives (real
      entities absent from the training corpus). Manual review is recommended.

Usage examples::

    # Detect fabrications with default paths and threshold
    python eval/detect_fabrications.py

    # Custom paths and stricter threshold
    python eval/detect_fabrications.py \\
        --responses eval/results/tiny_responses.json \\
        --registry eval/entity_registry.json \\
        --output eval/results/tiny_fabrications.json \\
        --threshold 90

    # Show help
    python eval/detect_fabrications.py --help
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum response length to process for entity extraction (characters).
# Prevents O(n^2) behavior on degenerate repetitive outputs.
MAX_RESPONSE_LENGTH = 2000

# Known drug suffixes for candidate extraction
DRUG_SUFFIXES = (
    "mab", "nib", "zole", "vone", "pril", "lone", "pine",
    "azole", "tinib", "ximab", "zumab", "ersen", "tide", "stat",
    "olol", "limus", "afil", "gliptin", "sartan", "lukast",
    "dipine", "prazole", "setron", "vastatin", "cillin",
)

# Gene name pattern: uppercase letters/digits, 2-10 chars, must contain
# at least one letter. Matches SOD1, TDP-43, C9orf72, FUS, etc.
GENE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9][A-Za-z0-9\-]{0,8})\b")

# NCT trial ID pattern: NCT followed by 6-8 digits
NCT_PATTERN = re.compile(r"\b(NCT\d{6,8})\b")

# Stopwords: common uppercase abbreviations that are NOT drug or gene names.
# These are excluded from drug candidate extraction to reduce false flags.
ABBREVIATION_STOPWORDS = {
    "ALS", "FDA", "MRI", "EMG", "NMJ", "CNS", "PNS", "CSF",
    "DNA", "RNA", "ATP", "GTP", "WHO", "NIH", "EMA", "PBA",
    "MND", "UMN", "LMN", "FTD", "PLS", "PMA", "SMA", "SCA",
    "BMI", "ICU", "PEG", "NIV", "BiPAP", "CPAP", "ALSFRS",
    "PCR", "ELISA", "GWAS", "SNP", "WGS", "NGS", "QOL",
    "ROS", "NOS", "COX", "MAP", "JAK", "STAT", "NF",
    "THE", "AND", "FOR", "NOT", "BUT", "WITH", "ARE", "WAS",
    "HAS", "HAD", "WILL", "CAN", "MAY", "ITS", "ALL", "ANY",
    "HOW", "WHO", "USE", "OUR", "NEW", "ONE", "TWO", "SEE",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for fabrication detection."""
    parser = argparse.ArgumentParser(
        description="Detect fabricated entities in model responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/detect_fabrications.py\n"
            "  python eval/detect_fabrications.py --threshold 90\n"
            "  python eval/detect_fabrications.py --responses eval/results/tiny_responses.json\n"
        ),
    )
    parser.add_argument(
        "--responses",
        type=str,
        default="eval/results/responses.json",
        help="Path to generated responses JSON (default: eval/results/responses.json)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="eval/entity_registry.json",
        help="Path to entity registry JSON (default: eval/entity_registry.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/fabrications.json",
        help="Path for fabrication output JSON (default: eval/results/fabrications.json)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=85,
        help="Fuzzy match threshold 0-100 for drug/gene matching (default: 85)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_registry(registry_path):
    """Load entity registry and build flat lookup structures.

    Args:
        registry_path: Path to entity_registry.json.

    Returns:
        (known_drugs, known_genes, known_trials) where:
        - known_drugs: set of lowercase canonical + alias strings
        - known_genes: set of lowercase canonical + alias strings
        - known_trials: set of NCT identifiers (original case)
    """
    with open(registry_path) as f:
        registry = json.load(f)

    # Build drug lookup (canonical + aliases, all lowered)
    known_drugs = set()
    for entry in registry.get("drugs", []):
        known_drugs.add(entry["canonical"].lower())
        for alias in entry.get("aliases", []):
            known_drugs.add(alias.lower())

    # Build gene lookup (canonical + aliases, all lowered)
    known_genes = set()
    for entry in registry.get("genes", []):
        known_genes.add(entry["canonical"].lower())
        for alias in entry.get("aliases", []):
            known_genes.add(alias.lower())

    # Build trial lookup (exact NCT IDs)
    known_trials = set()
    for entry in registry.get("trials", []):
        known_trials.add(entry["canonical"])
        for alias in entry.get("aliases", []):
            known_trials.add(alias)

    return known_drugs, known_genes, known_trials


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text):
    """Split text into sentences using simple punctuation rules.

    Splits on period, exclamation mark, or question mark followed by
    whitespace or end of string. Returns a list of non-empty sentences.
    """
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r"(?<=[.!?])(?:\s+|$)", text)
    return [s.strip() for s in parts if s.strip()]


def find_sentence_context(text, entity_text):
    """Find the sentence containing a given entity mention.

    Args:
        text: Full response text.
        entity_text: The entity string to locate.

    Returns:
        The sentence containing the entity, or the first 200 chars of
        the text if sentence splitting fails to locate it.
    """
    sentences = split_sentences(text)
    entity_lower = entity_text.lower()
    for sentence in sentences:
        if entity_lower in sentence.lower():
            return sentence

    # Fallback: return a window around the entity
    idx = text.lower().find(entity_lower)
    if idx >= 0:
        start = max(0, idx - 50)
        end = min(len(text), idx + len(entity_text) + 50)
        return text[start:end]

    return text[:200] if text else ""


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_nct_ids(text):
    """Extract NCT clinical trial identifiers from text.

    Returns a list of unique NCT ID strings found in the text.
    """
    return list(set(NCT_PATTERN.findall(text)))


def extract_drug_candidates(text, known_genes_lower):
    """Extract candidate drug name mentions from text.

    Identifies candidates by:
    1. Capitalized words (potential proper drug names)
    2. Words ending with known drug suffixes (lowercased check)
    3. Multi-word sequences of 2-3 capitalized words (brand names)

    Filters out known abbreviation stopwords and gene names to avoid
    cross-type false positives.

    Args:
        text: Response text to scan.
        known_genes_lower: Set of lowercase gene names for cross-filtering.

    Returns:
        A list of unique candidate drug name strings.
    """
    candidates = set()

    # Tokenize into words (keep hyphens within words)
    words = re.findall(r"\b[\w\-]+\b", text)

    for word in words:
        # Skip very short or very long words
        if len(word) < 3 or len(word) > 30:
            continue

        # Skip stopwords and known gene names
        word_upper = word.upper()
        if word_upper in ABBREVIATION_STOPWORDS:
            continue
        if word.lower() in known_genes_lower:
            continue

        # Check for capitalized words (potential drug names)
        if word[0].isupper() and word[1:].islower():
            candidates.add(word)

        # Check for known drug suffixes
        word_lower = word.lower()
        for suffix in DRUG_SUFFIXES:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix):
                candidates.add(word)
                break

    # Multi-word capitalized sequences (2-3 words, potential brand names)
    multi_word = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text)
    for phrase in multi_word:
        # Skip phrases that are just common English
        if len(phrase) > 5:
            candidates.add(phrase)

    return list(candidates)


def extract_gene_candidates(text, known_drugs_lower):
    """Extract candidate gene name mentions from text.

    Uses uppercase+digit patterns (SOD1, TARDBP, FUS, C9orf72) and filters
    out abbreviation stopwords and known drug names.

    Args:
        text: Response text to scan.
        known_drugs_lower: Set of lowercase drug names for cross-filtering.

    Returns:
        A list of unique candidate gene name strings.
    """
    candidates = set()

    matches = GENE_PATTERN.findall(text)
    for match in matches:
        # Skip stopwords
        if match.upper() in ABBREVIATION_STOPWORDS:
            continue
        # Skip known drugs to avoid cross-type flagging
        if match.lower() in known_drugs_lower:
            continue
        # Must contain at least one digit or be a known gene-like pattern
        # (all uppercase, 2-10 chars)
        if any(c.isdigit() for c in match) or (match.isupper() and len(match) >= 2):
            candidates.add(match)

    return list(candidates)


# ---------------------------------------------------------------------------
# Entity matching against registry
# ---------------------------------------------------------------------------

def check_entity_in_registry(entity_text, known_set, threshold, use_fuzzy=True):
    """Check whether an entity matches any entry in the registry.

    Args:
        entity_text: The entity string to check.
        known_set: Set of known entity strings (lowercased).
        threshold: Fuzzy match score threshold (0-100).
        use_fuzzy: If True, use fuzzy matching. If False, exact match only.

    Returns:
        (in_registry, best_match, best_score) where in_registry is True if
        the entity matches, best_match is the closest registry entry, and
        best_score is the match score.
    """
    entity_lower = entity_text.lower()

    # Exact match first (fast path)
    if entity_lower in known_set:
        return True, entity_lower, 100

    if not use_fuzzy or not known_set:
        return False, None, 0

    # Fuzzy match using extractOne for efficiency
    result = process.extractOne(
        entity_lower,
        known_set,
        scorer=fuzz.ratio,
        score_cutoff=threshold,
    )

    if result is not None:
        match_text, score, _ = result
        return True, match_text, int(score)

    # Below threshold — find the best match for reporting purposes
    result_any = process.extractOne(
        entity_lower,
        known_set,
        scorer=fuzz.ratio,
    )

    if result_any is not None:
        match_text, score, _ = result_any
        return False, match_text, int(score)

    return False, None, 0


# ---------------------------------------------------------------------------
# Per-question fabrication detection
# ---------------------------------------------------------------------------

def detect_fabrications_for_question(response_entry, known_drugs, known_genes,
                                     known_trials, threshold):
    """Run fabrication detection on a single question's response.

    Args:
        response_entry: Dict with "response" text and question metadata.
        known_drugs: Set of known drug names (lowercase).
        known_genes: Set of known gene names (lowercase).
        known_trials: Set of known NCT trial IDs.
        threshold: Fuzzy match threshold for drugs and genes.

    Returns:
        A dict with entities_extracted and flagged_entities for this question.
    """
    response_text = response_entry.get("response", "")

    # Truncate degenerate responses to avoid performance issues
    text_to_scan = response_text[:MAX_RESPONSE_LENGTH]

    entities_extracted = []
    flagged_entities = []

    # 1. Extract and check NCT trial IDs (exact match only)
    nct_ids = extract_nct_ids(text_to_scan)
    for nct_id in nct_ids:
        in_registry = nct_id in known_trials
        entity = {"text": nct_id, "type": "trial", "in_registry": in_registry}
        if not in_registry:
            entity["best_registry_match"] = None
            entity["best_match_score"] = 0
            context = find_sentence_context(response_text, nct_id)
            flagged_entities.append({
                "text": nct_id,
                "type": "trial",
                "context": context,
                "best_registry_match": None,
                "best_match_score": 0,
            })
        entities_extracted.append(entity)

    # 2. Extract and check drug candidates (fuzzy match)
    drug_candidates = extract_drug_candidates(text_to_scan, known_genes)
    for candidate in drug_candidates:
        in_registry, best_match, best_score = check_entity_in_registry(
            candidate, known_drugs, threshold, use_fuzzy=True
        )
        entity = {"text": candidate, "type": "drug", "in_registry": in_registry}
        if not in_registry:
            entity["best_registry_match"] = best_match
            entity["best_match_score"] = best_score
            context = find_sentence_context(response_text, candidate)
            flagged_entities.append({
                "text": candidate,
                "type": "drug",
                "context": context,
                "best_registry_match": best_match,
                "best_match_score": best_score,
            })
        entities_extracted.append(entity)

    # 3. Extract and check gene candidates (fuzzy match)
    gene_candidates = extract_gene_candidates(text_to_scan, known_drugs)
    for candidate in gene_candidates:
        in_registry, best_match, best_score = check_entity_in_registry(
            candidate, known_genes, threshold, use_fuzzy=True
        )
        entity = {"text": candidate, "type": "gene", "in_registry": in_registry}
        if not in_registry:
            entity["best_registry_match"] = best_match
            entity["best_match_score"] = best_score
            context = find_sentence_context(response_text, candidate)
            flagged_entities.append({
                "text": candidate,
                "type": "gene",
                "context": context,
                "best_registry_match": best_match,
                "best_match_score": best_score,
            })
        entities_extracted.append(entity)

    return {
        "question_id": response_entry["question_id"],
        "category": response_entry["category"],
        "is_trap": response_entry["is_trap"],
        "entities_extracted": entities_extracted,
        "flagged_entities": flagged_entities,
    }


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def compute_summary(per_question_results):
    """Compute summary statistics from per-question fabrication results.

    Args:
        per_question_results: List of per-question result dicts.

    Returns:
        A summary dict with total counts and per-type breakdowns.
    """
    total_extracted = 0
    total_flagged = 0
    by_type = {
        "drugs": {"extracted": 0, "flagged": 0},
        "genes": {"extracted": 0, "flagged": 0},
        "trials": {"extracted": 0, "flagged": 0},
    }

    for result in per_question_results:
        for entity in result["entities_extracted"]:
            total_extracted += 1
            entity_type = entity["type"]
            type_key = entity_type + "s"  # drug -> drugs, gene -> genes, trial -> trials
            if type_key in by_type:
                by_type[type_key]["extracted"] += 1
            if not entity["in_registry"]:
                total_flagged += 1
                if type_key in by_type:
                    by_type[type_key]["flagged"] += 1

    # Compute rates
    flagged_rate = total_flagged / total_extracted if total_extracted > 0 else 0.0
    for type_key in by_type:
        extracted = by_type[type_key]["extracted"]
        flagged = by_type[type_key]["flagged"]
        by_type[type_key]["rate"] = round(flagged / extracted, 4) if extracted > 0 else 0.0

    return {
        "total_entities_extracted": total_extracted,
        "total_flagged": total_flagged,
        "flagged_rate": round(flagged_rate, 4),
        "by_type": by_type,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for fabrication detection."""
    args = parse_args()

    print("\n=== ALS-LM Fabrication Detection ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical information system.\n")

    # Load responses
    if not os.path.isfile(args.responses):
        print(f"ERROR: Responses file not found: {args.responses}")
        print("  Run eval/generate_responses.py first to generate model responses.")
        sys.exit(1)

    with open(args.responses) as f:
        responses_data = json.load(f)

    responses = responses_data.get("responses", [])
    print(f"  Loaded {len(responses)} responses from {args.responses}")

    # Load entity registry
    if not os.path.isfile(args.registry):
        print(f"ERROR: Entity registry not found: {args.registry}")
        print("  Run eval/build_entity_registry.py first to build the registry.")
        sys.exit(1)

    known_drugs, known_genes, known_trials = load_registry(args.registry)
    print(f"  Registry loaded: {len(known_drugs)} drugs, "
          f"{len(known_genes)} genes, {len(known_trials)} trials")
    print(f"  Fuzzy match threshold: {args.threshold}")

    # Process each response
    per_question_results = []
    all_flagged = []
    total = len(responses)

    for i, resp in enumerate(responses):
        # Progress every 20 questions
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Checking {i + 1}/{total}...")

        result = detect_fabrications_for_question(
            resp, known_drugs, known_genes, known_trials, args.threshold
        )
        per_question_results.append(result)

        # Collect all flagged entities into the flat list
        for flagged in result["flagged_entities"]:
            all_flagged.append({
                "text": flagged["text"],
                "type": flagged["type"],
                "question_id": result["question_id"],
                "context": flagged["context"],
                "best_registry_match": flagged["best_registry_match"],
                "best_match_score": flagged["best_match_score"],
            })

    # Compute summary
    summary = compute_summary(per_question_results)

    # Build output
    output = {
        "metadata": {
            "responses_path": args.responses,
            "registry_path": args.registry,
            "threshold": args.threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_responses_checked": len(responses),
            "caveat": (
                "Flagged entities may include false positives — entities that "
                "are real but absent from the training corpus. Manual review "
                "recommended for flagged items."
            ),
        },
        "summary": summary,
        "per_question": per_question_results,
        "all_flagged": all_flagged,
    }

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n  === Fabrication Detection Summary ===")
    print(f"  Total entities extracted:  {summary['total_entities_extracted']}")
    print(f"  Total flagged:             {summary['total_flagged']}")
    print(f"  Flagged rate:              {summary['flagged_rate']:.4f}")
    for type_key, type_stats in summary["by_type"].items():
        print(f"    {type_key}: {type_stats['extracted']} extracted, "
              f"{type_stats['flagged']} flagged ({type_stats['rate']:.4f})")

    print(f"\n  Output saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
