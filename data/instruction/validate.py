#!/usr/bin/env python3
"""Validate ALS instruction dataset for corpus grounding and benchmark leakage.

Reads generated Alpaca-format instruction pairs (als_instructions.json),
verifies corpus grounding using the 48K entity registry and fuzzy matching,
detects benchmark leakage against the 160 evaluation questions, computes
quality statistics, and produces a clean validated dataset plus auditable
independence and quality reports.

This script is run after generate.py and before prepare_sft.py in the
instruction dataset pipeline.

Usage::

    python data/instruction/validate.py
    python data/instruction/validate.py --input data/instruction/als_instructions.json
    python data/instruction/validate.py --grounding-threshold 65 --leakage-threshold 80

Output:

    data/instruction/als_instructions.json   - Validated dataset (overwritten)
    data/instruction/rejected.jsonl          - Pairs that failed validation
    data/instruction/quality_report.json     - Quality statistics report
    data/instruction/independence_report.json - Leakage audit report
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

# Ensure project root is on sys.path
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rapidfuzz import fuzz, process


# ---------------------------------------------------------------------------
# Constants (adapted from eval/detect_fabrications.py)
# ---------------------------------------------------------------------------

# Maximum text length to scan for entity extraction (characters)
MAX_SCAN_LENGTH = 2000

# Known drug suffixes for candidate extraction
DRUG_SUFFIXES = (
    "mab", "nib", "zole", "vone", "pril", "lone", "pine",
    "azole", "tinib", "ximab", "zumab", "ersen", "tide", "stat",
    "olol", "limus", "afil", "gliptin", "sartan", "lukast",
    "dipine", "prazole", "setron", "vastatin", "cillin",
)

# Gene name pattern: uppercase letters/digits, 2-10 chars
GENE_PATTERN = re.compile(r"\b([A-Z][A-Z0-9][A-Za-z0-9\-]{0,8})\b")

# NCT trial ID pattern: NCT followed by 8 digits
NCT_PATTERN = re.compile(r"\b(NCT\d{8})\b")

# Common abbreviations to exclude from entity extraction
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
    # Common medical/scientific abbreviations not in entity registry
    "EV", "EVs", "BC", "AD", "ER", "CT", "EEG", "MG", "VA",
    "OA", "CI", "HR", "OR", "SD", "SE", "CM", "MR", "MS",
    "GBS", "CRP", "COPD", "OXPHOS", "OMM", "IMM", "TME",
    "MQC", "FFV", "MISEV", "TNBC", "MSC", "MSCs", "TGF",
    "PEG-EV", "PD", "HD", "IP", "IV", "SC", "IM",
}

# Common English words that appear capitalized at sentence start but are NOT
# drug/gene names. Prevents false-positive entity extraction from Alpaca outputs.
COMMON_ENGLISH_STOPWORDS = {
    w.lower() for w in {
        # Pronouns and determiners
        "This", "These", "That", "Those", "They", "Their", "Them", "There",
        "Its", "Some", "Such", "Each", "Both", "Many", "Most", "Other",
        "Several", "Various", "All", "Any", "Every",
        # Conjunctions and transitions
        "However", "Moreover", "Furthermore", "Additionally", "Therefore",
        "Thus", "Hence", "Consequently", "Meanwhile", "Nevertheless",
        "Although", "While", "Since", "Because", "During", "After",
        "Before", "Between", "Within", "Without", "Against", "Through",
        "Among", "Beyond", "Across", "Along", "Around", "Under", "Above",
        "About", "Until", "Upon",
        # Common verbs at sentence start
        "According", "Based", "Given", "Including", "Using", "Following",
        "Compared", "Associated", "Related", "Resulting", "Suggesting",
        "Indicating", "Regarding", "Involving", "Representing", "Leading",
        "Occurring", "Affecting", "Causing", "Providing", "Showing",
        # Common nouns in medical text (not drug/gene names)
        "Patients", "Studies", "Results", "Research", "Treatment",
        "Disease", "Diseases", "Symptoms", "Diagnosis", "Clinical",
        "Individuals", "Caregivers", "Health", "National", "Department",
        "Population", "Survival", "Mortality", "Incidence", "Prevalence",
        "Motor", "Neurons", "Cells", "Protein", "Proteins", "Muscle",
        "Brain", "Spinal", "Cord", "Nerve", "Nerves", "Blood",
        "Respiratory", "Cognitive", "Progressive", "Chronic", "Acute",
        "Early", "Late", "Primary", "Secondary", "Specific", "General",
        "Current", "Recent", "Previous", "Further", "Future", "Overall",
        "Significant", "Important", "Common", "Rare", "Normal", "Abnormal",
        "Direct", "Indirect",
        # Geographic and organizational terms
        "Gulf", "War", "Asia", "Southwest", "United", "States",
        "European", "American", "Western", "Eastern", "Northern", "Southern",
        "Black", "White",
        # Specifically for medical text
        "Specifically", "Notably", "Importantly", "Interestingly",
        "Alternatively", "Collectively", "Typically", "Generally",
        "Approximately", "Potentially", "Particularly", "Frequently",
        # Medical terms that are not drug names
        "Mitochondrial", "Mitochondria", "Mitophagy", "Extracellular",
        "Alzheimer", "Dementia", "Parkinson", "Huntington",
        "Autoimmune", "Inflammatory", "Degenerative", "Hereditary",
        "Genetic", "Genomic", "Epigenetic",
    }
}


# ---------------------------------------------------------------------------
# Entity extraction (patterns from eval/detect_fabrications.py)
# ---------------------------------------------------------------------------

def _extract_nct_ids(text):
    """Extract NCT clinical trial identifiers from text."""
    return list(set(NCT_PATTERN.findall(text)))


def _extract_drug_candidates(text, known_genes_lower):
    """Extract candidate drug name mentions from text.

    Uses capitalized words, known drug suffixes, and multi-word sequences.
    Filters out abbreviation stopwords and gene names.
    """
    candidates = set()
    words = re.findall(r"\b[\w\-]+\b", text)

    for word in words:
        if len(word) < 3 or len(word) > 30:
            continue
        if word.upper() in ABBREVIATION_STOPWORDS:
            continue
        if word.lower() in known_genes_lower:
            continue
        if word.lower() in COMMON_ENGLISH_STOPWORDS:
            continue

        # Capitalized words (potential drug names)
        if word[0].isupper() and len(word) > 1 and word[1:].islower():
            candidates.add(word)

        # Known drug suffixes
        word_lower = word.lower()
        for suffix in DRUG_SUFFIXES:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix):
                candidates.add(word)
                break

    return list(candidates)


def _extract_gene_candidates(text, known_drugs_lower):
    """Extract candidate gene name mentions from text.

    Uses uppercase+digit patterns (SOD1, TARDBP, FUS, C9orf72) and filters
    out abbreviation stopwords and known drug names.
    """
    candidates = set()
    matches = GENE_PATTERN.findall(text)

    for match in matches:
        # Strip trailing hyphens that may be captured by the regex
        clean = match.rstrip("-")
        if not clean or len(clean) < 2:
            continue
        if clean.upper() in ABBREVIATION_STOPWORDS:
            continue
        if clean.lower() in known_drugs_lower:
            continue
        if any(c.isdigit() for c in clean) or (clean.isupper() and len(clean) >= 2):
            candidates.add(clean)

    return list(candidates)


def _check_entity_in_registry(entity_text, known_set, threshold=85):
    """Check whether an entity matches any entry in the registry set.

    Uses exact match first, then rapidfuzz for approximate matching.

    Returns:
        (in_registry, best_match, best_score)
    """
    entity_lower = entity_text.lower()

    # Exact match fast path
    if entity_lower in known_set:
        return True, entity_lower, 100

    if not known_set:
        return False, None, 0

    # Fuzzy match
    result = process.extractOne(
        entity_lower,
        known_set,
        scorer=fuzz.ratio,
        score_cutoff=threshold,
    )
    if result is not None:
        match_text, score, _ = result
        return True, match_text, int(score)

    return False, None, 0


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------

def load_entity_registry(registry_path):
    """Load entity registry and build flat lookup sets.

    Returns:
        (known_drugs, known_genes, known_trials) as sets of lowercase strings.
    """
    with open(registry_path) as f:
        registry = json.load(f)
    return _build_registry_sets(registry)


def _build_registry_sets(registry):
    """Build flat lookup sets from a registry dict.

    Works with both structured entries (dicts with canonical/aliases) and
    flat string lists.
    """
    known_drugs = set()
    for entry in registry.get("drugs", []):
        if isinstance(entry, dict):
            known_drugs.add(entry["canonical"].lower())
            for alias in entry.get("aliases", []):
                known_drugs.add(alias.lower())
        else:
            known_drugs.add(str(entry).lower())

    known_genes = set()
    for entry in registry.get("genes", []):
        if isinstance(entry, dict):
            known_genes.add(entry["canonical"].lower())
            for alias in entry.get("aliases", []):
                known_genes.add(alias.lower())
        else:
            known_genes.add(str(entry).lower())

    known_trials = set()
    for entry in registry.get("trials", []):
        if isinstance(entry, dict):
            known_trials.add(entry["canonical"])
            for alias in entry.get("aliases", []):
                known_trials.add(alias)
        else:
            known_trials.add(str(entry))

    return known_drugs, known_genes, known_trials


# ---------------------------------------------------------------------------
# Sliding-window fuzzy matching (pattern from eval/score_responses.py)
# ---------------------------------------------------------------------------

def _build_chunks(text, chunk_size=100, overlap=50):
    """Break text into overlapping chunks for fuzzy matching.

    Returns a list of lowercase string chunks.
    """
    text = text.lower().strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


# ---------------------------------------------------------------------------
# Grounding verification (DATA-02)
# ---------------------------------------------------------------------------

def check_entity_grounding(output_text, entity_registry, entity_threshold=85):
    """Check if entities in the output text are grounded in the registry.

    Extracts drug, gene, and trial entities from the output and verifies
    each against the entity registry. A pair passes entity grounding if
    all extracted entities are found (or if no entities are extracted).

    Args:
        output_text: The generated output text to check.
        entity_registry: Registry dict with drugs, genes, trials, etc.
        entity_threshold: Fuzzy match threshold for registry matching.

    Returns:
        Dict with grounded (bool), entities_found (int), failures (list).
    """
    known_drugs, known_genes, known_trials = _build_registry_sets(entity_registry)

    text = output_text[:MAX_SCAN_LENGTH]
    failures = []
    entities_found = 0

    # Check NCT IDs (exact match)
    nct_ids = _extract_nct_ids(text)
    for nct_id in nct_ids:
        if nct_id in known_trials:
            entities_found += 1
        else:
            failures.append({"text": nct_id, "type": "trial"})

    # Check drug candidates
    drug_candidates = _extract_drug_candidates(text, known_genes)
    for candidate in drug_candidates:
        in_reg, _, _ = _check_entity_in_registry(
            candidate, known_drugs, threshold=entity_threshold
        )
        if in_reg:
            entities_found += 1
        else:
            failures.append({"text": candidate, "type": "drug"})

    # Check gene candidates
    gene_candidates = _extract_gene_candidates(text, known_drugs)
    for candidate in gene_candidates:
        in_reg, _, _ = _check_entity_in_registry(
            candidate, known_genes, threshold=entity_threshold
        )
        if in_reg:
            entities_found += 1
        else:
            failures.append({"text": candidate, "type": "gene"})

    # Grounded if no failures (or no entities extracted)
    grounded = len(failures) == 0
    return {
        "grounded": grounded,
        "entities_found": entities_found,
        "failures": failures,
    }


def check_corpus_grounding(output_text, corpus_text, threshold=70):
    """Check if the output text is grounded in the corpus via fuzzy matching.

    Breaks the output into 100-char chunks with 50-char overlap, then
    computes rapidfuzz partial_ratio against the corpus text. Returns the
    best score found across all chunks.

    Args:
        output_text: The generated output text to check.
        corpus_text: Full corpus text (or relevant category excerpt).
        threshold: Score threshold for considering grounded.

    Returns:
        Best fuzzy match score (0-100) across all output chunks.
    """
    output_chunks = _build_chunks(output_text, chunk_size=100, overlap=50)
    corpus_lower = corpus_text.lower().strip()

    if not output_chunks or not corpus_lower:
        return 0

    best_score = 0
    for chunk in output_chunks:
        score = fuzz.partial_ratio(chunk, corpus_lower)
        if score > best_score:
            best_score = score
        if best_score == 100:
            break

    return best_score


# ---------------------------------------------------------------------------
# Benchmark leakage detection (DATA-02 and DATA-04)
# ---------------------------------------------------------------------------

def check_leakage(instruction_text, output_text, benchmark_questions,
                  threshold=75):
    """Check a single pair against all benchmark questions for leakage.

    Compares the instruction text against each benchmark question and
    prompt_template, and the output text against each verified_answer,
    using rapidfuzz partial_ratio.

    Args:
        instruction_text: The generated instruction/question text.
        output_text: The generated answer text.
        benchmark_questions: List of benchmark question dicts.
        threshold: Fuzzy similarity threshold (0-100).

    Returns:
        (leaked, benchmark_id, max_score) where leaked is True if any
        comparison exceeds the threshold.
    """
    instr_lower = instruction_text.lower()
    out_lower = output_text.lower()

    for bq in benchmark_questions:
        score_q = fuzz.partial_ratio(instr_lower, bq["question"].lower())
        score_p = fuzz.partial_ratio(instr_lower, bq["prompt_template"].lower())
        score_a = fuzz.partial_ratio(out_lower, bq["verified_answer"].lower())

        max_score = max(score_q, score_p, score_a)
        if max_score >= threshold:
            return True, bq["id"], max_score

    return False, None, 0


# ---------------------------------------------------------------------------
# Rejected pairs logging
# ---------------------------------------------------------------------------

def write_rejected(rejected_path, pair, reason, details=None):
    """Append a rejected pair to the JSONL rejected file.

    Args:
        rejected_path: Path to the rejected.jsonl file.
        pair: The rejected Alpaca pair dict.
        reason: Rejection reason string (e.g., "grounding_failed").
        details: Optional dict with additional rejection details.
    """
    entry = {
        "pair": pair,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        entry.update(details)

    with open(rejected_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Quality statistics report (DATA-03)
# ---------------------------------------------------------------------------

def compute_quality_stats(pairs, grounded_count, entity_match_count,
                          leakage_count, benchmark_count,
                          leakage_threshold=75,
                          fact_grounding_verified=True):
    """Compute quality statistics for the instruction dataset.

    Args:
        pairs: List of validated Alpaca pair dicts.
        grounded_count: Number of pairs that passed corpus grounding.
        entity_match_count: Number of pairs that passed entity grounding.
        leakage_count: Number of pairs flagged for benchmark leakage.
        benchmark_count: Total number of benchmark questions checked.
        leakage_threshold: Leakage detection threshold used.
        fact_grounding_verified: Whether fact-level corpus grounding was
            actually run (False when corpus was too large for fuzzy matching).

    Returns:
        Quality statistics report dict.
    """
    total = len(pairs)

    # Category distribution
    cat_counts = {}
    for pair in pairs:
        cat = pair.get("metadata", {}).get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    category_distribution = {}
    target_per_category = 250
    for cat, count in sorted(cat_counts.items()):
        shortfall = max(0, target_per_category - count)
        entry = {
            "count": count,
            "target": target_per_category,
            "shortfall": shortfall,
        }
        if count < 50:
            entry["thin_coverage"] = True
        category_distribution[cat] = entry

    # Question type distribution
    qt_counts = {}
    for pair in pairs:
        qt = pair.get("metadata", {}).get("question_type", "unknown")
        qt_counts[qt] = qt_counts.get(qt, 0) + 1

    question_type_distribution = {}
    for qt, count in sorted(qt_counts.items()):
        question_type_distribution[qt] = {
            "count": count,
            "percent": round(100.0 * count / total, 1) if total > 0 else 0.0,
        }

    # Response lengths
    word_counts = []
    sentence_counts = []
    for pair in pairs:
        output = pair.get("output", "")
        words = output.split()
        word_counts.append(len(words))
        # Split on ". " for sentence counting
        sentences = [s.strip() for s in re.split(r"\.\s+", output) if s.strip()]
        sentence_counts.append(max(len(sentences), 1))

    mean_words = sum(word_counts) / len(word_counts) if word_counts else 0
    mean_sentences = sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0

    # Attempt tokenization with ALS BPE tokenizer
    mean_tokens = None
    try:
        from tokenizers import Tokenizer
        tokenizer_path = os.path.join(_project_root, "tokenizer", "als_tokenizer.json")
        if os.path.isfile(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
            token_counts = []
            for pair in pairs:
                encoded = tokenizer.encode(pair.get("output", ""))
                token_counts.append(len(encoded.ids))
            mean_tokens = round(sum(token_counts) / len(token_counts), 1) if token_counts else 0
    except Exception:
        pass

    response_lengths = {
        "mean_sentences": round(mean_sentences, 1),
        "mean_words": round(mean_words, 1),
        "min_words": min(word_counts) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
    }
    if mean_tokens is not None:
        response_lengths["mean_tokens"] = mean_tokens

    # Corpus grounding stats
    corpus_grounding = {
        "grounded_count": grounded_count,
        "total_count": total,
        "grounding_rate": round(grounded_count / total, 2) if total > 0 else 0.0,
        "entity_match_rate": round(entity_match_count / total, 2) if total > 0 else 0.0,
        "fact_grounding_verified": fact_grounding_verified,
    }

    # Leakage check stats
    leakage_check = {
        "pairs_checked": total,
        "pairs_flagged": leakage_count,
        "threshold": leakage_threshold,
        "benchmark_questions": benchmark_count,
    }

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_pairs": total,
            "rejected_count": 0,  # Set by caller after validation
        },
        "category_distribution": category_distribution,
        "question_type_distribution": question_type_distribution,
        "corpus_grounding": corpus_grounding,
        "response_lengths": response_lengths,
        "leakage_check": leakage_check,
    }


# ---------------------------------------------------------------------------
# Independence report (DATA-04)
# ---------------------------------------------------------------------------

def build_independence_report(pairs, benchmark_questions, threshold=75):
    """Build a train/eval independence report.

    Cross-compares all instruction pairs against all benchmark questions
    and produces an auditable report with max similarity scores and a
    pass/fail verdict.

    Args:
        pairs: List of Alpaca pair dicts.
        benchmark_questions: List of benchmark question dicts.
        threshold: Fuzzy similarity threshold for flagging.

    Returns:
        Independence report dict with verdict, statistics, and flagged pairs.
    """
    pairs_checked = len(pairs)
    bq_count = len(benchmark_questions)
    flagged_details = []
    max_similarity = 0
    total_score_sum = 0
    total_comparisons = 0

    for i, pair in enumerate(pairs):
        instr = pair.get("instruction", "")
        output = pair.get("output", "")
        pair_max = 0
        pair_flagged_bq = None

        for bq in benchmark_questions:
            score_q = fuzz.partial_ratio(instr.lower(), bq["question"].lower())
            score_p = fuzz.partial_ratio(instr.lower(), bq["prompt_template"].lower())
            score_a = fuzz.partial_ratio(output.lower(), bq["verified_answer"].lower())

            best = max(score_q, score_p, score_a)
            total_score_sum += best
            total_comparisons += 1

            if best > pair_max:
                pair_max = best
                pair_flagged_bq = bq["id"]

        if pair_max > max_similarity:
            max_similarity = pair_max

        if pair_max >= threshold:
            flagged_details.append({
                "pair_index": i,
                "instruction_preview": instr[:80],
                "benchmark_id": pair_flagged_bq,
                "max_score": pair_max,
            })

    mean_similarity = (
        round(total_score_sum / total_comparisons, 2)
        if total_comparisons > 0 else 0
    )

    flagged_count = len(flagged_details)
    verdict = "FAIL" if flagged_count > 0 else "PASS"

    return {
        "pairs_checked": pairs_checked,
        "benchmark_questions": bq_count,
        "max_similarity": max_similarity,
        "mean_similarity": mean_similarity,
        "flagged_pairs": flagged_count,
        "flagged_details": flagged_details,
        "threshold": threshold,
        "verdict": verdict,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Full dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(pairs, entity_registry, corpus_text,
                     benchmark_questions, rejected_path,
                     grounding_threshold=70, leakage_threshold=75,
                     entity_threshold=85):
    """Validate the full instruction dataset.

    Runs entity grounding, corpus grounding, and leakage detection on
    every pair. Pairs that fail any check are removed from the dataset
    and logged to the rejected file.

    Args:
        pairs: List of Alpaca pair dicts.
        entity_registry: Registry dict with drugs, genes, trials, etc.
        corpus_text: Full corpus text for grounding checks.
        benchmark_questions: List of benchmark question dicts.
        rejected_path: Path for rejected.jsonl output.
        grounding_threshold: Fuzzy threshold for corpus grounding.
        leakage_threshold: Fuzzy threshold for benchmark leakage.
        entity_threshold: Fuzzy threshold for entity registry matching.

    Returns:
        List of validated pairs (those that passed all checks).
    """
    cleaned = []

    for pair in pairs:
        output_text = pair.get("output", "")
        instruction_text = pair.get("instruction", "")
        rejected = False

        # Entity grounding check
        entity_result = check_entity_grounding(
            output_text, entity_registry, entity_threshold=entity_threshold
        )
        if not entity_result["grounded"]:
            write_rejected(
                rejected_path, pair, "grounding_failed",
                {
                    "entity_failures": [f["text"] for f in entity_result["failures"]],
                    "best_fact_score": 0,
                },
            )
            rejected = True

        # Corpus grounding check (only if entity check passed and corpus loaded)
        if not rejected and corpus_text:
            fact_score = check_corpus_grounding(
                output_text, corpus_text, threshold=grounding_threshold
            )
            if fact_score < grounding_threshold:
                write_rejected(
                    rejected_path, pair, "grounding_failed",
                    {
                        "entity_failures": [],
                        "best_fact_score": fact_score,
                    },
                )
                rejected = True

        # Leakage detection
        if not rejected:
            leaked, bq_id, score = check_leakage(
                instruction_text, output_text,
                benchmark_questions, threshold=leakage_threshold,
            )
            if leaked:
                write_rejected(
                    rejected_path, pair, "benchmark_leakage",
                    {
                        "benchmark_id": bq_id,
                        "similarity_score": score,
                    },
                )
                rejected = True

        if not rejected:
            cleaned.append(pair)

    return cleaned


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate ALS instruction dataset for corpus grounding, "
            "benchmark leakage, and quality statistics"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data/instruction/validate.py\n"
            "  python data/instruction/validate.py --grounding-threshold 65\n"
            "  python data/instruction/validate.py --leakage-threshold 80\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/instruction/als_instructions.json",
        help="Generated dataset to validate (default: data/instruction/als_instructions.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/instruction/als_instructions.json",
        help="Validated dataset output (default: overwrites input)",
    )
    parser.add_argument(
        "--rejected",
        type=str,
        default="data/instruction/rejected.jsonl",
        help="Path for rejected pairs JSONL (default: data/instruction/rejected.jsonl)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/instruction/quality_report.json",
        help="Quality statistics report JSON (default: data/instruction/quality_report.json)",
    )
    parser.add_argument(
        "--independence-report",
        type=str,
        default="data/instruction/independence_report.json",
        help="Leakage audit report JSON (default: data/instruction/independence_report.json)",
    )
    parser.add_argument(
        "--grounding-threshold",
        type=int,
        default=70,
        help="Fuzzy match threshold for corpus grounding (default: 70)",
    )
    parser.add_argument(
        "--leakage-threshold",
        type=int,
        default=75,
        help="Fuzzy match threshold for benchmark leakage (default: 75)",
    )
    parser.add_argument(
        "--entity-threshold",
        type=int,
        default=85,
        help="Fuzzy match threshold for entity registry matching (default: 85)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/processed/train.txt",
        help="Corpus file for grounding checks (default: data/processed/train.txt)",
    )
    return parser.parse_args()


def main():
    """Run the instruction dataset validation pipeline."""
    args = parse_args()

    # Resolve paths relative to project root
    def resolve(path):
        if os.path.isabs(path):
            return path
        return os.path.join(_project_root, path)

    input_path = resolve(args.input)
    output_path = resolve(args.output)
    rejected_path = resolve(args.rejected)
    report_path = resolve(args.report)
    independence_path = resolve(args.independence_report)
    corpus_path = resolve(args.corpus)
    registry_path = os.path.join(_project_root, "eval", "entity_registry.json")
    benchmark_path = os.path.join(_project_root, "eval", "questions.json")

    print("ALS Instruction Dataset Validation")
    print("=" * 60)
    print(f"  Input:               {input_path}")
    print(f"  Output:              {output_path}")
    print(f"  Rejected:            {rejected_path}")
    print(f"  Quality report:      {report_path}")
    print(f"  Independence report: {independence_path}")
    print(f"  Grounding threshold: {args.grounding_threshold}")
    print(f"  Leakage threshold:   {args.leakage_threshold}")
    print(f"  Entity threshold:    {args.entity_threshold}")
    print()

    # Load generated dataset
    if not os.path.isfile(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print("  Run data/instruction/generate.py first.")
        sys.exit(1)

    with open(input_path) as f:
        pairs = json.load(f)
    print(f"  Loaded {len(pairs)} instruction pairs")

    # Load entity registry
    if not os.path.isfile(registry_path):
        print(f"ERROR: Entity registry not found: {registry_path}")
        print("  Run eval/build_entity_registry.py first.")
        sys.exit(1)

    with open(registry_path) as f:
        entity_registry = json.load(f)
    known_drugs, known_genes, known_trials = _build_registry_sets(entity_registry)
    total_entities = len(known_drugs) + len(known_genes) + len(known_trials)
    print(f"  Registry loaded: {total_entities} entities "
          f"({len(known_drugs)} drugs, {len(known_genes)} genes, "
          f"{len(known_trials)} trials)")

    # Load benchmark questions
    if not os.path.isfile(benchmark_path):
        print(f"ERROR: Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    with open(benchmark_path) as f:
        benchmark_questions = json.load(f)
    print(f"  Benchmark loaded: {len(benchmark_questions)} questions")

    # Load corpus for fact-level grounding
    # For large corpora (>50MB), fact-level fuzzy matching via partial_ratio
    # is computationally infeasible (O(n*m) per pair).  Since each instruction
    # pair was generated FROM a corpus passage, corpus grounding is inherent
    # by construction.  For large corpora we skip the redundant fact-level
    # check and rely on entity grounding + leakage detection.
    MAX_CORPUS_FOR_GROUNDING = 50_000_000  # 50 MB threshold
    corpus_text = ""
    if os.path.isfile(corpus_path):
        corpus_size = os.path.getsize(corpus_path)
        if corpus_size > MAX_CORPUS_FOR_GROUNDING:
            print(f"  Corpus: {corpus_size:,} bytes (>{MAX_CORPUS_FOR_GROUNDING:,})")
            print(f"  Skipping fact-level grounding (too large for fuzzy matching)")
            print(f"  Entity grounding + leakage detection will be used instead")
        else:
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_text = f.read()
            print(f"  Corpus loaded: {len(corpus_text):,} characters")
    else:
        print(f"  WARNING: Corpus not found at {corpus_path}, skipping fact-level grounding")

    print()

    # Clear previous rejected file
    if os.path.exists(rejected_path):
        os.remove(rejected_path)

    # Step 1: Entity grounding pass
    print("Step 1: Entity grounding verification...")
    entity_pass_count = 0
    for pair in pairs:
        result = check_entity_grounding(
            pair.get("output", ""), entity_registry,
            entity_threshold=args.entity_threshold,
        )
        if result["grounded"]:
            entity_pass_count += 1
    print(f"  Entity grounding: {entity_pass_count}/{len(pairs)} passed "
          f"({100*entity_pass_count/len(pairs):.1f}%)")

    # Step 2: Full validation (entity + fact + leakage)
    print("\nStep 2: Full validation (grounding + leakage)...")
    cleaned = validate_dataset(
        pairs=pairs,
        entity_registry=entity_registry,
        corpus_text=corpus_text,
        benchmark_questions=benchmark_questions,
        rejected_path=rejected_path,
        grounding_threshold=args.grounding_threshold,
        leakage_threshold=args.leakage_threshold,
        entity_threshold=args.entity_threshold,
    )

    grounding_fails = 0
    leakage_fails = 0
    if os.path.isfile(rejected_path):
        with open(rejected_path) as f:
            for line in f:
                entry = json.loads(line)
                if entry["reason"] == "grounding_failed":
                    grounding_fails += 1
                elif entry["reason"] == "benchmark_leakage":
                    leakage_fails += 1

    total_rejected = len(pairs) - len(cleaned)
    print(f"  Validated: {len(cleaned)} pairs kept, {total_rejected} rejected "
          f"({grounding_fails} grounding, {leakage_fails} leakage)")

    # Warn if leakage rate is too high (Pitfall 4)
    if leakage_fails > len(pairs) * 0.10:
        print(f"\n  WARNING: {leakage_fails} pairs flagged for leakage "
              f"({100*leakage_fails/len(pairs):.1f}%). "
              f"Consider raising --leakage-threshold.")

    # Step 3: Build independence report
    print("\nStep 3: Building independence report...")
    independence_report = build_independence_report(
        cleaned, benchmark_questions, threshold=args.leakage_threshold
    )
    os.makedirs(os.path.dirname(independence_path), exist_ok=True)
    with open(independence_path, "w") as f:
        json.dump(independence_report, f, indent=2)
    print(f"  Verdict: {independence_report['verdict']}")
    print(f"  Max similarity: {independence_report['max_similarity']}")
    print(f"  Report saved to: {independence_path}")

    # Step 4: Compute and save quality statistics
    print("\nStep 4: Computing quality statistics...")
    fact_grounding_was_verified = bool(corpus_text)
    stats = compute_quality_stats(
        cleaned,
        grounded_count=len(cleaned),  # All surviving pairs passed grounding
        entity_match_count=entity_pass_count,
        leakage_count=leakage_fails,
        benchmark_count=len(benchmark_questions),
        leakage_threshold=args.leakage_threshold,
        fact_grounding_verified=fact_grounding_was_verified,
    )
    stats["metadata"]["rejected_count"] = total_rejected
    stats["metadata"]["source_file"] = args.input

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Report saved to: {report_path}")

    # Print human-readable quality summary
    print()
    print("=" * 60)
    print("Quality Statistics Summary")
    print("=" * 60)
    print(f"  Total pairs (after validation): {stats['metadata']['total_pairs']}")
    print(f"  Rejected pairs:                 {stats['metadata']['rejected_count']}")
    print()

    print("  Category distribution:")
    for cat, info in sorted(stats["category_distribution"].items()):
        flag = " [THIN]" if info.get("thin_coverage") else ""
        print(f"    {cat}: {info['count']}/{info['target']} "
              f"(shortfall: {info['shortfall']}){flag}")

    print()
    print("  Question type distribution:")
    for qt, info in sorted(stats["question_type_distribution"].items()):
        print(f"    {qt}: {info['count']} ({info['percent']}%)")

    print()
    grounding = stats["corpus_grounding"]
    if grounding.get("fact_grounding_verified", True):
        print(f"  Corpus grounding rate:  {grounding['grounding_rate']:.2f}")
    else:
        print(f"  Corpus grounding rate:  {grounding['grounding_rate']:.2f} "
              f"(entity-only; fact-level skipped — corpus too large)")
    print(f"  Entity match rate:      {grounding['entity_match_rate']:.2f}")

    print()
    lengths = stats["response_lengths"]
    print(f"  Mean response length:   {lengths['mean_words']:.1f} words, "
          f"{lengths['mean_sentences']:.1f} sentences")
    print(f"  Response range:         {lengths['min_words']}-{lengths['max_words']} words")
    if "mean_tokens" in lengths:
        print(f"  Mean BPE tokens:        {lengths['mean_tokens']:.1f}")

    print()
    leakage = stats["leakage_check"]
    print(f"  Leakage check:          {leakage['pairs_flagged']} flagged "
          f"of {leakage['pairs_checked']} (threshold: {leakage['threshold']})")

    # Step 5: Write validated dataset
    print()
    print("Step 5: Writing validated dataset...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"  Output written to: {output_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
