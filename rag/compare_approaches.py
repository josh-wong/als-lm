#!/usr/bin/env python3
"""Compare evaluation results across from-scratch, baseline, and RAG approaches.

Standalone script that auto-discovers result directories for three approach
categories (from-scratch ALS-LM, bare Llama 3.1 8B baseline, and 4 RAG
configurations), computes cross-approach metrics including failure
decomposition, selects qualitative examples, and generates a dual-format
Markdown + JSON comparison report.

This is the primary research deliverable for the v0.6.0 RAG comparison
milestone, answering: how do failure modes differ between a from-scratch
domain model and retrieval-augmented generation?

This is a research analysis tool, not a medical information system.

Usage examples::

    # Default: auto-discover from project root, write to rag/results/
    python rag/compare_approaches.py

    # Custom output directory
    python rag/compare_approaches.py --output-dir rag/results/

    # Verbose output with progress messages
    python rag/compare_approaches.py --verbose

    # Override specific directories
    python rag/compare_approaches.py --from-scratch-dir eval/results/als-lm-500m_q4_k_m
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from rapidfuzz import fuzz
except ImportError:
    print(
        "ERROR: rapidfuzz is required for failure decomposition.\n"
        "Install with: pip install rapidfuzz",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All taxonomy modes in display order
TAXONOMY_MODES = [
    "confident_fabrication",
    "plausible_blending",
    "outdated_information",
    "boundary_confusion",
    "accurate_but_misleading",
    "accurate",
    "degenerate",
]

# All 8 evaluation categories
CATEGORIES = [
    "clinical_trials",
    "diagnostic_criteria",
    "disease_mechanisms",
    "drug_treatment",
    "epidemiology",
    "gene_mutation",
    "patient_care",
    "temporal_accuracy",
]

# Canonical display order for approaches (6 total)
APPROACH_ORDER = [
    "als_lm",
    "baseline",
    "rag_500_minilm",
    "rag_200_minilm",
    "rag_500_pubmedbert",
    "rag_200_pubmedbert",
]

# Map internal approach names to display labels
DISPLAY_NAMES = {
    "als_lm": "ALS-LM",
    "baseline": "Baseline",
    "rag_500_minilm": "500-MiniLM",
    "rag_200_minilm": "200-MiniLM",
    "rag_500_pubmedbert": "500-PubMedBERT",
    "rag_200_pubmedbert": "200-PubMedBERT",
}

# Required JSON files per approach directory
REQUIRED_FILES = ["scores.json", "fabrications.json", "taxonomy.json", "responses.json"]

# Fuzzy matching threshold for key_fact presence in chunks
FUZZY_THRESHOLD = 80


# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Locate the project root relative to this script's location.

    The script lives at rag/compare_approaches.py, so parent.parent gives
    the project root. Validates by checking that eval/ exists.
    """
    root = Path(__file__).resolve().parent.parent
    if not (root / "eval").is_dir():
        print(
            f"WARNING: Could not verify project root at {root} "
            f"(eval/ directory not found).",
            file=sys.stderr,
        )
    return root


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _has_required_files(directory: Path) -> bool:
    """Check that a directory contains all 4 required JSON result files."""
    return all((directory / f).is_file() for f in REQUIRED_FILES)


def _normalize_approach_name(dir_name: str) -> str:
    """Convert a directory name to a canonical internal approach name.

    Maps als-lm-500m_q4_k_m -> als_lm, baseline -> baseline,
    rag_500_minilm -> rag_500_minilm.
    """
    if dir_name.startswith("als-lm"):
        return "als_lm"
    return dir_name


def discover_approaches(
    project_root: Path,
    verbose: bool = False,
    from_scratch_dir: Path | None = None,
    baseline_dir: Path | None = None,
    rag_dir: Path | None = None,
) -> list[dict]:
    """Auto-discover result directories for all three approach types.

    Searches eval/results/ for the from-scratch model, rag/results/baseline/
    for the no-retrieval baseline, and rag/results/rag_*/ for RAG configs.
    Returns a list of approach dicts sorted in canonical APPROACH_ORDER.
    """
    approaches = []

    # From-scratch: single known directory
    scratch_path = from_scratch_dir or (
        project_root / "eval" / "results" / "als-lm-500m_q4_k_m"
    )
    if scratch_path.is_dir() and _has_required_files(scratch_path):
        approaches.append({
            "name": "als_lm",
            "display": DISPLAY_NAMES["als_lm"],
            "path": scratch_path,
            "approach_type": "from_scratch",
        })
        if verbose:
            print(f"  Found from-scratch: {scratch_path.name}")
    elif verbose:
        missing = [f for f in REQUIRED_FILES if not (scratch_path / f).is_file()]
        print(f"  Skipping from-scratch {scratch_path}: missing {missing}")

    # Baseline: single known directory
    baseline_path = baseline_dir or (project_root / "rag" / "results" / "baseline")
    if baseline_path.is_dir() and _has_required_files(baseline_path):
        approaches.append({
            "name": "baseline",
            "display": DISPLAY_NAMES["baseline"],
            "path": baseline_path,
            "approach_type": "baseline",
        })
        if verbose:
            print(f"  Found baseline: {baseline_path.name}")
    elif verbose:
        print(f"  Skipping baseline: directory not found at {baseline_path}")

    # RAG configs: discover by naming convention
    rag_base = rag_dir or (project_root / "rag" / "results")
    if rag_base.is_dir():
        for subdir in sorted(rag_base.iterdir()):
            if not subdir.is_dir() or not subdir.name.startswith("rag_"):
                continue
            if not _has_required_files(subdir):
                if verbose:
                    missing = [f for f in REQUIRED_FILES if not (subdir / f).is_file()]
                    print(f"  Skipping RAG {subdir.name}: missing {missing}")
                continue

            name = subdir.name
            display = DISPLAY_NAMES.get(name, name)
            approaches.append({
                "name": name,
                "display": display,
                "path": subdir,
                "approach_type": "rag",
            })
            if verbose:
                print(f"  Found RAG config: {subdir.name}")

    # Sort in canonical order
    order_map = {name: i for i, name in enumerate(APPROACH_ORDER)}
    approaches.sort(key=lambda a: order_map.get(a["name"], 99))

    if verbose:
        print(f"  Total approaches discovered: {len(approaches)}")

    return approaches


def load_approach_data(approach: dict) -> dict:
    """Load all 4 JSON result files for a single approach.

    Returns a dict with keys: scores, fabrications, taxonomy, responses.
    """
    data = {}
    for filename in REQUIRED_FILES:
        filepath = approach["path"] / filename
        with open(filepath, "r", encoding="utf-8") as f:
            data[filename.replace(".json", "")] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def compute_overall_accuracy(all_data: dict) -> dict:
    """Extract overall accuracy metrics per approach.

    Returns dict mapping approach name to accuracy metrics including
    mean_accuracy, binary_pass_rate, and hedging count.
    """
    result = {}
    for name, data in all_data.items():
        overall = data["scores"]["aggregate"]["overall"]
        result[name] = {
            "mean_accuracy": overall["mean_accuracy"],
            "binary_pass_rate": overall.get("binary_pass_rate", 0),
            "hedging": overall.get("total_hedging_instances", 0),
        }
    return result


def compute_per_category_accuracy(all_data: dict) -> dict:
    """Extract per-category accuracy for each approach.

    Returns nested dict: category -> approach_name -> mean_accuracy.
    """
    result = {}
    for cat in CATEGORIES:
        result[cat] = {}
        for name, data in all_data.items():
            cat_data = data["scores"]["aggregate"]["by_category"].get(cat, {})
            result[cat][name] = cat_data.get("mean_accuracy", 0.0)
    return result


def compute_fabrication_rates(all_data: dict) -> dict:
    """Extract fabrication rate metrics per approach.

    Returns dict mapping approach name to fabrication summary including
    total entities, flagged count, flagged rate, and per-type breakdown.
    """
    result = {}
    for name, data in all_data.items():
        summary = data["fabrications"]["summary"]
        by_type = {}
        for entity_type in ["drugs", "genes", "trials"]:
            type_data = summary["by_type"].get(entity_type, {})
            by_type[entity_type] = {
                "extracted": type_data.get("extracted", 0),
                "flagged": type_data.get("flagged", 0),
                "rate": type_data.get("rate", 0.0),
            }
        result[name] = {
            "total_extracted": summary.get("total_entities_extracted", 0),
            "total_flagged": summary.get("total_flagged", 0),
            "flagged_rate": summary.get("flagged_rate", 0.0),
            "by_type": by_type,
        }
    return result


def compute_taxonomy_distribution(all_data: dict) -> dict:
    """Extract failure taxonomy distribution per approach.

    Returns nested dict: approach_name -> mode -> {count, pct}.
    """
    result = {}
    for name, data in all_data.items():
        dist = data["taxonomy"]["distribution"]
        result[name] = {}
        for mode in TAXONOMY_MODES:
            mode_data = dist.get(mode, {"count": 0, "pct": 0.0})
            result[name][mode] = {
                "count": mode_data["count"],
                "pct": mode_data["pct"],
            }
    return result


# ---------------------------------------------------------------------------
# Failure decomposition (COMP-03)
# ---------------------------------------------------------------------------

def load_benchmark_questions(project_root: Path, benchmark_path: Path | None = None) -> list[dict]:
    """Load benchmark questions from eval/questions.json.

    Returns list of question dicts with id, question, category, key_facts.
    """
    path = benchmark_path or (project_root / "eval" / "questions.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_failure(key_facts: list[str], chunks: list[str], threshold: int = FUZZY_THRESHOLD) -> str:
    """Classify a wrong RAG answer as retrieval or generation failure.

    Uses fuzzy matching to check if any key_fact appears in the concatenated
    chunk text. If at least one key_fact is found (partial_ratio >= threshold),
    the failure is a generation failure (retrieval found relevant info, model
    still got it wrong). If no key_facts are found, it is a retrieval failure.
    """
    if not chunks or not key_facts:
        return "retrieval_failure"

    concatenated = " ".join(chunks).lower()
    for fact in key_facts:
        score = fuzz.partial_ratio(fact.lower(), concatenated)
        if score >= threshold:
            return "generation_failure"

    return "retrieval_failure"


def compute_failure_decomposition(
    questions: list[dict],
    responses: list[dict],
    scores_per_question: list[dict],
    threshold: int = FUZZY_THRESHOLD,
) -> dict:
    """Classify each wrong RAG answer as retrieval or generation failure.

    Only applicable to RAG approaches that have retrieval data in their
    responses. Returns summary counts, percentages, per-question details,
    and per-category breakdown.
    """
    response_map = {r["question_id"]: r for r in responses}
    question_map = {q["id"]: q for q in questions}
    score_map = {s["question_id"]: s for s in scores_per_question}

    retrieval_failures = 0
    generation_failures = 0
    per_question = []

    # Per-category tracking
    cat_retrieval = {}
    cat_generation = {}
    cat_total_wrong = {}
    for cat in CATEGORIES:
        cat_retrieval[cat] = 0
        cat_generation[cat] = 0
        cat_total_wrong[cat] = 0

    for qid, score_entry in score_map.items():
        if score_entry.get("accuracy_binary", 0) == 1:
            continue  # Correct answer, not a failure

        question = question_map.get(qid)
        response = response_map.get(qid)
        if not question or not response:
            continue

        key_facts = question.get("key_facts", [])
        retrieval_data = response.get("retrieval", None)
        chunks = retrieval_data.get("chunks", []) if retrieval_data else []
        category = score_entry.get("category", question.get("category", ""))

        failure_type = classify_failure(key_facts, chunks, threshold)

        if failure_type == "retrieval_failure":
            retrieval_failures += 1
            cat_retrieval[category] = cat_retrieval.get(category, 0) + 1
        else:
            generation_failures += 1
            cat_generation[category] = cat_generation.get(category, 0) + 1

        cat_total_wrong[category] = cat_total_wrong.get(category, 0) + 1
        per_question.append({
            "question_id": qid,
            "category": category,
            "failure_type": failure_type,
        })

    total_wrong = retrieval_failures + generation_failures

    # Build per-category breakdown
    per_category = {}
    for cat in CATEGORIES:
        total_cat = cat_total_wrong.get(cat, 0)
        ret_cat = cat_retrieval.get(cat, 0)
        gen_cat = cat_generation.get(cat, 0)
        per_category[cat] = {
            "total_wrong": total_cat,
            "retrieval_failures": ret_cat,
            "generation_failures": gen_cat,
            "retrieval_pct": ret_cat / total_cat if total_cat > 0 else 0.0,
            "generation_pct": gen_cat / total_cat if total_cat > 0 else 0.0,
        }

    return {
        "total_wrong": total_wrong,
        "retrieval_failures": retrieval_failures,
        "generation_failures": generation_failures,
        "retrieval_pct": retrieval_failures / total_wrong if total_wrong > 0 else 0.0,
        "generation_pct": generation_failures / total_wrong if total_wrong > 0 else 0.0,
        "per_question": per_question,
        "per_category": per_category,
    }


# ---------------------------------------------------------------------------
# Contrast question selection (COMP-02)
# ---------------------------------------------------------------------------

def select_contrast_questions(
    all_taxonomy: dict,
    all_scores: dict,
    approach_names: list[str],
    n: int = 10,
) -> list[dict]:
    """Select questions with maximum divergence across approaches.

    For each question, collects the primary_mode from taxonomy across all
    approaches and scores by: (a) number of distinct failure modes,
    (b) mixed score (min of correct vs wrong counts), (c) category.
    Returns top ~n questions ensuring some category diversity.
    """
    # Collect per-question modes and accuracy across all approaches
    question_data = {}
    for name in approach_names:
        for entry in all_taxonomy[name]["per_question"]:
            qid = entry["question_id"]
            if qid not in question_data:
                question_data[qid] = {
                    "modes": {},
                    "correct": {},
                    "category": entry.get("category", ""),
                }
            question_data[qid]["modes"][name] = entry["primary_mode"]

        for entry in all_scores[name]["per_question"]:
            qid = entry["question_id"]
            if qid in question_data:
                question_data[qid]["correct"][name] = (
                    entry.get("accuracy_binary", 0) == 1
                )
                if entry.get("category"):
                    question_data[qid]["category"] = entry["category"]

    # Score each question by divergence
    scored = []
    for qid, data in question_data.items():
        if len(data["modes"]) < len(approach_names):
            continue  # Only consider questions present in all approaches

        n_distinct_modes = len(set(data["modes"].values()))
        n_correct = sum(1 for v in data["correct"].values() if v)
        n_wrong = len(data["correct"]) - n_correct
        mixed_score = min(n_correct, n_wrong)

        scored.append({
            "question_id": qid,
            "category": data["category"],
            "modes": dict(data["modes"]),
            "n_distinct_modes": n_distinct_modes,
            "mixed_score": mixed_score,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
        })

    # Sort by distinct modes (desc), then mixed score (desc)
    scored.sort(key=lambda x: (-x["n_distinct_modes"], -x["mixed_score"]))

    # Select top N with some category diversity
    selected = []
    category_counts = {}
    max_per_category = max(2, n // len(CATEGORIES) + 1)

    for item in scored:
        if len(selected) >= n:
            break
        cat = item["category"]
        cat_count = category_counts.get(cat, 0)
        if cat_count >= max_per_category and len(selected) < n - 2:
            continue  # Allow overflow near the end
        selected.append(item)
        category_counts[cat] = cat_count + 1

    return selected


def select_deep_dive_examples(
    contrast_questions: list[dict],
    all_data: dict,
    approach_names: list[str],
    approaches: list[dict],
    n: int = 4,
) -> list[dict]:
    """Select 3-4 deep-dive examples from contrast questions.

    Prioritizes specific contrasting patterns:
    1. RAG helps (baseline wrong, at least one RAG correct)
    2. RAG hurts (baseline correct, RAG wrong or degenerate)
    3. Embedding sensitivity (MiniLM and PubMedBERT produce different outcomes)
    4. All approaches fail differently
    """
    approach_types = {a["name"]: a["approach_type"] for a in approaches}
    rag_names = [n for n in approach_names if approach_types.get(n) == "rag"]
    minilm_names = [n for n in rag_names if "minilm" in n]
    pubmedbert_names = [n for n in rag_names if "pubmedbert" in n]

    # Build per-question correctness lookup from scores data
    correct_by_question = {}
    for name in approach_names:
        for entry in all_data[name]["scores"]["per_question"]:
            qid = entry["question_id"]
            if qid not in correct_by_question:
                correct_by_question[qid] = {}
            correct_by_question[qid][name] = entry.get("accuracy_binary", 0) == 1

    patterns = {
        "rag_helps": None,
        "rag_hurts": None,
        "embedding_sensitivity": None,
        "all_different": None,
    }

    for item in contrast_questions:
        qid = item["question_id"]
        correctness = correct_by_question.get(qid, {})
        modes = item["modes"]

        baseline_correct = correctness.get("baseline", False)
        rag_any_correct = any(correctness.get(r, False) for r in rag_names)
        rag_any_wrong = any(not correctness.get(r, True) for r in rag_names)

        # Pattern 1: RAG helps (baseline wrong, at least one RAG correct)
        if patterns["rag_helps"] is None and not baseline_correct and rag_any_correct:
            patterns["rag_helps"] = {
                **item,
                "pattern": "rag_helps",
                "rationale": "Baseline wrong, at least one RAG configuration correct",
            }

        # Pattern 2: RAG hurts (baseline correct, RAG wrong or degenerate)
        if patterns["rag_hurts"] is None and baseline_correct and rag_any_wrong:
            # Check for degenerate in RAG modes
            rag_degenerate = any(
                modes.get(r) == "degenerate" for r in rag_names
            )
            rationale = "Baseline correct, RAG produces degenerate output" if rag_degenerate else "Baseline correct, at least one RAG configuration wrong"
            patterns["rag_hurts"] = {
                **item,
                "pattern": "rag_hurts",
                "rationale": rationale,
            }

        # Pattern 3: Embedding sensitivity (MiniLM vs PubMedBERT differ)
        if patterns["embedding_sensitivity"] is None:
            minilm_modes = set(modes.get(m, "") for m in minilm_names)
            pubmedbert_modes = set(modes.get(m, "") for m in pubmedbert_names)
            if minilm_modes != pubmedbert_modes and len(minilm_modes) > 0 and len(pubmedbert_modes) > 0:
                patterns["embedding_sensitivity"] = {
                    **item,
                    "pattern": "embedding_sensitivity",
                    "rationale": f"MiniLM modes: {minilm_modes}, PubMedBERT modes: {pubmedbert_modes}",
                }

        # Pattern 4: All approaches fail differently
        if patterns["all_different"] is None and item["n_distinct_modes"] >= 4:
            patterns["all_different"] = {
                **item,
                "pattern": "all_different",
                "rationale": f"{item['n_distinct_modes']} distinct failure modes across approaches",
            }

        # Stop if we have all patterns
        if all(v is not None for v in patterns.values()):
            break

    # Collect found patterns, deduplicate by question_id
    selected = []
    seen_qids = set()
    for pattern_name in ["rag_helps", "rag_hurts", "embedding_sensitivity", "all_different"]:
        item = patterns[pattern_name]
        if item is not None and item["question_id"] not in seen_qids:
            selected.append(item)
            seen_qids.add(item["question_id"])
        if len(selected) >= n:
            break

    # If we have fewer than n, fill from contrast questions
    if len(selected) < n:
        for item in contrast_questions:
            if item["question_id"] not in seen_qids:
                selected.append({
                    **item,
                    "pattern": "high_divergence",
                    "rationale": f"{item['n_distinct_modes']} distinct modes, mixed score {item['mixed_score']}",
                })
                seen_qids.add(item["question_id"])
            if len(selected) >= n:
                break

    return selected


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare evaluation results across from-scratch, baseline, "
            "and RAG approaches."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write comparison outputs (default: rag/results/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages to stdout",
    )
    parser.add_argument(
        "--from-scratch-dir",
        type=Path,
        default=None,
        help="Override from-scratch result directory",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Override baseline result directory",
    )
    parser.add_argument(
        "--rag-dir",
        type=Path,
        default=None,
        help="Override RAG results parent directory",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=None,
        help="Override benchmark questions path (default: eval/questions.json)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(
    approach_names: list[str],
    approaches: list[dict],
    overall_accuracy: dict,
    per_category: dict,
    fabrication_rates: dict,
    taxonomy_dist: dict,
    failure_decomposition: dict,
    contrast_questions: list[dict],
    deep_dive_examples: list[dict],
    all_data: dict,
    questions: list[dict],
    question_count: int,
) -> str:
    """Build the full Markdown comparison report.

    Follows the metrics-first flow: TL;DR, overall accuracy, per-category,
    fabrication, taxonomy, failure decomposition, qualitative examples,
    summary judgment.
    """
    display = {a["name"]: a["display"] for a in approaches}
    approach_types = {a["name"]: a["approach_type"] for a in approaches}
    rag_names = [n for n in approach_names if approach_types.get(n) == "rag"]
    question_map = {q["id"]: q for q in questions}

    lines = []

    def add(text: str = "") -> None:
        lines.append(text)

    # Title
    add("# Cross-approach comparison report")
    add()

    # -----------------------------------------------------------------------
    # TL;DR
    # -----------------------------------------------------------------------
    add("## TL;DR")
    add()

    baseline_acc = overall_accuracy.get("baseline", {}).get("mean_accuracy", 0)
    als_lm_acc = overall_accuracy.get("als_lm", {}).get("mean_accuracy", 0)
    best_rag_name = max(rag_names, key=lambda n: overall_accuracy.get(n, {}).get("mean_accuracy", 0)) if rag_names else None
    best_rag_acc = overall_accuracy.get(best_rag_name, {}).get("mean_accuracy", 0) if best_rag_name else 0
    worst_rag_name = min(rag_names, key=lambda n: overall_accuracy.get(n, {}).get("mean_accuracy", 0)) if rag_names else None
    worst_rag_acc = overall_accuracy.get(worst_rag_name, {}).get("mean_accuracy", 0) if worst_rag_name else 0

    tldr = (
        f"Compared 6 approaches across {question_count} ALS benchmark questions. "
        f"The bare Llama 3.1 8B baseline ({baseline_acc:.1%} accuracy) outperforms "
        f"all 4 RAG configurations (best: {display.get(best_rag_name, 'N/A')} at "
        f"{best_rag_acc:.1%}, worst: {display.get(worst_rag_name, 'N/A')} at "
        f"{worst_rag_acc:.1%}), indicating that retrieval does not help and sometimes "
        f"hurts. The from-scratch ALS-LM ({als_lm_acc:.1%}) produces predominantly "
        f"incoherent output. Embedding model choice is the single biggest lever: "
        f"PubMedBERT-based configs achieve 6-7x higher accuracy than MiniLM-based "
        f"configs, with {display.get(worst_rag_name, 'N/A')} showing "
        f"{taxonomy_dist.get(worst_rag_name, {}).get('degenerate', {}).get('pct', 0):.0f}% "
        f"degenerate responses."
    )
    add(tldr)
    add()

    # -----------------------------------------------------------------------
    # Overall accuracy table
    # -----------------------------------------------------------------------
    add("## Overall accuracy")
    add()
    add(
        "Mean accuracy, binary pass rate, and hedging instances for each approach. "
        "Mean accuracy is the proportional key-fact match score (0-1), while binary "
        "pass rate counts questions with at least one key fact found."
    )
    add()

    col_w = 14
    label_w = 22
    header = f"| {'Metric':<{label_w}} |"
    sep = f"| {'-' * label_w} |"
    for name in approach_names:
        header += f" {display[name]:>{col_w}} |"
        sep += f" {'-' * col_w} |"
    add(header)
    add(sep)

    row_acc = f"| {'Mean accuracy':<{label_w}} |"
    row_pass = f"| {'Binary pass rate':<{label_w}} |"
    row_hedge = f"| {'Hedging instances':<{label_w}} |"
    for name in approach_names:
        row_acc += f" {overall_accuracy[name]['mean_accuracy']:>{col_w}.4f} |"
        pr = overall_accuracy[name]['binary_pass_rate']
        # Normalize to float for consistent percentage display
        pr_float = float(pr) if not isinstance(pr, float) else pr
        row_pass += f" {pr_float:>{col_w}.1%} |"
        row_hedge += f" {overall_accuracy[name]['hedging']:>{col_w}} |"
    add(row_acc)
    add(row_pass)
    add(row_hedge)
    add()

    # Note the 500-MiniLM outlier
    if "rag_500_minilm" in approach_names:
        minilm_acc = overall_accuracy.get("rag_500_minilm", {}).get("mean_accuracy", 0)
        minilm_degen = taxonomy_dist.get("rag_500_minilm", {}).get("degenerate", {}).get("pct", 0)
        add(
            f"The {display.get('rag_500_minilm', '500-MiniLM')} configuration is a "
            f"significant outlier at {minilm_acc:.1%} accuracy with "
            f"{minilm_degen:.0f}% degenerate responses. Large MiniLM chunks flood "
            f"the context window with irrelevant text, causing repetitive output."
        )
        add()

    # -----------------------------------------------------------------------
    # Per-category breakdown
    # -----------------------------------------------------------------------
    add("## Per-category accuracy")
    add()
    add(
        "Mean accuracy broken down by the 8 evaluation categories. This is the "
        "richest comparison dimension, revealing where different approaches have "
        "complementary strengths and weaknesses."
    )
    add()

    header = f"| {'Category':<{label_w}} |"
    sep = f"| {'-' * label_w} |"
    for name in approach_names:
        header += f" {display[name]:>{col_w}} |"
        sep += f" {'-' * col_w} |"
    add(header)
    add(sep)

    for cat in CATEGORIES:
        display_cat = cat.replace("_", " ").title()
        row = f"| {display_cat:<{label_w}} |"
        for name in approach_names:
            val = per_category[cat].get(name, 0.0)
            row += f" {val:>{col_w}.4f} |"
        add(row)
    add()

    # Find which approach leads in each category
    leader_counts = {}
    for cat in CATEGORIES:
        best = max(approach_names, key=lambda n: per_category[cat].get(n, 0))
        leader_counts[best] = leader_counts.get(best, 0) + 1

    top_leader = max(leader_counts, key=leader_counts.get)
    add(
        f"The {display[top_leader]} approach leads in {leader_counts[top_leader]} "
        f"of {len(CATEGORIES)} categories. Drug treatment is the strongest "
        f"category across most approaches, while disease mechanisms and "
        f"epidemiology are consistently the weakest."
    )
    add()

    # -----------------------------------------------------------------------
    # Fabrication analysis
    # -----------------------------------------------------------------------
    add("## Fabrication analysis")
    add()
    add(
        "Entity-level fabrication analysis comparing extracted entities (drug names, "
        "gene names, clinical trial identifiers) against a known registry. Higher "
        "flagged rates indicate more entities not found in the reference registry."
    )
    add()

    header = f"| {'Metric':<{label_w}} |"
    sep = f"| {'-' * label_w} |"
    for name in approach_names:
        header += f" {display[name]:>{col_w}} |"
        sep += f" {'-' * col_w} |"
    add(header)
    add(sep)

    row_ext = f"| {'Total extracted':<{label_w}} |"
    row_flag = f"| {'Total flagged':<{label_w}} |"
    row_rate = f"| {'Flagged rate':<{label_w}} |"
    for name in approach_names:
        row_ext += f" {fabrication_rates[name]['total_extracted']:>{col_w}} |"
        row_flag += f" {fabrication_rates[name]['total_flagged']:>{col_w}} |"
        row_rate += f" {fabrication_rates[name]['flagged_rate']:>{col_w}.1%} |"
    add(row_ext)
    add(row_flag)
    add(row_rate)
    add()

    # Commentary on fabrication patterns
    als_lm_ext = fabrication_rates.get("als_lm", {}).get("total_extracted", 0)
    baseline_ext = fabrication_rates.get("baseline", {}).get("total_extracted", 0)
    add(
        f"The from-scratch ALS-LM extracts far fewer entities ({als_lm_ext}) than "
        f"the baseline ({baseline_ext}) because its responses are largely "
        f"incoherent text that does not contain recognizable entity names. Higher "
        f"entity counts in baseline and RAG approaches reflect more fluent, "
        f"substantive responses, but also more opportunities for fabrication."
    )
    add()

    # -----------------------------------------------------------------------
    # Taxonomy distribution
    # -----------------------------------------------------------------------
    add("## Failure taxonomy distribution")
    add()

    first_name = approach_names[0]
    total_q = sum(
        taxonomy_dist[first_name][mode]["count"] for mode in TAXONOMY_MODES
    )
    add(
        f"Distribution of failure modes across the {len(TAXONOMY_MODES)} taxonomy "
        f"categories for each approach. Counts represent the number of questions "
        f"(out of {total_q}) classified into each mode."
    )
    add()

    header = f"| {'Failure mode':<{label_w + 6}} |"
    sep = f"| {'-' * (label_w + 6)} |"
    for name in approach_names:
        header += f" {display[name]:>{col_w}} |"
        sep += f" {'-' * col_w} |"
    add(header)
    add(sep)

    for mode in TAXONOMY_MODES:
        display_mode = mode.replace("_", " ").title()
        row = f"| {display_mode:<{label_w + 6}} |"
        for name in approach_names:
            td = taxonomy_dist[name][mode]
            pct_str = f"{td['count']} ({td['pct']:.1f}%)"
            row += f" {pct_str:>{col_w}} |"
        add(row)
    add()

    # Identify dominant modes per approach
    for name in approach_names:
        dominant = max(TAXONOMY_MODES, key=lambda m: taxonomy_dist[name][m]["pct"])
        dom_pct = taxonomy_dist[name][dominant]["pct"]
        dom_display = dominant.replace("_", " ")
        if dom_pct > 30:
            add(
                f"**{display[name]}:** Dominated by {dom_display} "
                f"({dom_pct:.1f}%)."
            )
    add()

    add(
        "The from-scratch ALS-LM has a more distributed failure profile with "
        "significant degenerate output, while baseline and RAG approaches "
        "concentrate in confident fabrication. This reflects fundamentally "
        "different failure mechanisms: the from-scratch model lacks language "
        "competence, while the larger models fabricate confidently."
    )
    add()

    # -----------------------------------------------------------------------
    # Failure decomposition (COMP-03)
    # -----------------------------------------------------------------------
    add("## Failure decomposition")
    add()
    add(
        "For each wrong RAG answer, failures are classified as either retrieval "
        "failures (no key facts found in the retrieved chunks) or generation "
        "failures (at least one key fact present in chunks, but the model still "
        "produced an incorrect answer). This decomposition only applies to RAG "
        "approaches, which have retrieval data. ALS-LM and Baseline have no "
        "retrieval component."
    )
    add()

    # Summary table
    dec_label_w = 22
    dec_col_w = 14
    header = f"| {'Metric':<{dec_label_w}} |"
    sep = f"| {'-' * dec_label_w} |"
    for name in approach_names:
        header += f" {display[name]:>{dec_col_w}} |"
        sep += f" {'-' * dec_col_w} |"
    add(header)
    add(sep)

    row_total = f"| {'Total wrong':<{dec_label_w}} |"
    row_ret = f"| {'Retrieval failures':<{dec_label_w}} |"
    row_gen = f"| {'Generation failures':<{dec_label_w}} |"
    row_ret_pct = f"| {'Retrieval failure %':<{dec_label_w}} |"
    row_gen_pct = f"| {'Generation failure %':<{dec_label_w}} |"

    for name in approach_names:
        decomp = failure_decomposition.get(name)
        if decomp is None:
            row_total += f" {'N/A':>{dec_col_w}} |"
            row_ret += f" {'N/A':>{dec_col_w}} |"
            row_gen += f" {'N/A':>{dec_col_w}} |"
            row_ret_pct += f" {'N/A':>{dec_col_w}} |"
            row_gen_pct += f" {'N/A':>{dec_col_w}} |"
        else:
            row_total += f" {decomp['total_wrong']:>{dec_col_w}} |"
            row_ret += f" {decomp['retrieval_failures']:>{dec_col_w}} |"
            row_gen += f" {decomp['generation_failures']:>{dec_col_w}} |"
            row_ret_pct += f" {decomp['retrieval_pct']:>{dec_col_w}.1%} |"
            row_gen_pct += f" {decomp['generation_pct']:>{dec_col_w}.1%} |"

    add(row_total)
    add(row_ret)
    add(row_gen)
    add(row_ret_pct)
    add(row_gen_pct)
    add()

    # Per-category failure decomposition table
    add("### Per-category failure decomposition")
    add()
    add(
        "Retrieval failure percentage by category for each RAG configuration. "
        "Higher percentages indicate categories where the retrieval system "
        "consistently fails to surface relevant information."
    )
    add()

    header = f"| {'Category':<{label_w}} |"
    sep = f"| {'-' * label_w} |"
    for name in rag_names:
        header += f" {display[name]:>{col_w}} |"
        sep += f" {'-' * col_w} |"
    add(header)
    add(sep)

    for cat in CATEGORIES:
        display_cat = cat.replace("_", " ").title()
        row = f"| {display_cat:<{label_w}} |"
        for name in rag_names:
            decomp = failure_decomposition.get(name)
            if decomp and cat in decomp.get("per_category", {}):
                cat_data = decomp["per_category"][cat]
                total_cat = cat_data["total_wrong"]
                if total_cat > 0:
                    ret_pct = cat_data["retrieval_pct"]
                    row += f" {ret_pct:>{col_w}.1%} |"
                else:
                    row += f" {'--':>{col_w}} |"
            else:
                row += f" {'N/A':>{col_w}} |"
        add(row)
    add()

    # -----------------------------------------------------------------------
    # Contrast table (COMP-02)
    # -----------------------------------------------------------------------
    add("## Qualitative contrast table")
    add()
    add(
        f"The {len(contrast_questions)} questions below show maximum divergence "
        f"across approaches, where different systems produce different failure "
        f"modes for the same question. Each cell shows the primary taxonomy "
        f"classification for that approach."
    )
    add()

    # Build contrast table
    ct_label_w = 12
    ct_col_w = 16
    header = f"| {'Question':<{ct_label_w}} | {'Category':<{label_w}} |"
    sep = f"| {'-' * ct_label_w} | {'-' * label_w} |"
    for name in approach_names:
        header += f" {display[name]:>{ct_col_w}} |"
        sep += f" {'-' * ct_col_w} |"
    add(header)
    add(sep)

    for item in contrast_questions:
        qid = item["question_id"]
        cat = item["category"].replace("_", " ").title()
        row = f"| {qid:<{ct_label_w}} | {cat:<{label_w}} |"
        for name in approach_names:
            mode = item["modes"].get(name, "N/A")
            mode_short = mode.replace("_", " ")
            # Truncate long mode names
            if len(mode_short) > ct_col_w:
                mode_short = mode_short[:ct_col_w - 1] + "."
            row += f" {mode_short:>{ct_col_w}} |"
        add(row)
    add()

    # -----------------------------------------------------------------------
    # Deep-dive narratives (COMP-02)
    # -----------------------------------------------------------------------
    add("## Deep-dive examples")
    add()
    add(
        "Detailed examination of selected questions that showcase specific "
        "contrasting patterns across approaches. For each example, the full "
        "response text is shown for all approaches, along with the top 2-3 "
        "retrieved chunks for RAG configurations."
    )
    add()

    # Build response lookup
    response_map = {}
    for name in approach_names:
        resp_list = all_data[name]["responses"]
        if isinstance(resp_list, dict) and "responses" in resp_list:
            resp_list = resp_list["responses"]
        for entry in resp_list:
            qid = entry["question_id"]
            if qid not in response_map:
                response_map[qid] = {}
            response_map[qid][name] = entry

    for i, example in enumerate(deep_dive_examples, 1):
        qid = example["question_id"]
        pattern = example.get("pattern", "divergence")
        rationale = example.get("rationale", "High divergence across approaches")
        cat = example.get("category", "").replace("_", " ").title()
        question_obj = question_map.get(qid, {})
        question_text = question_obj.get("question", question_obj.get("prompt_template", ""))

        pattern_display = pattern.replace("_", " ").title()
        add(f"### Example {i}: {pattern_display} ({qid})")
        add()
        add(f"**Category:** {cat}")
        add()
        add(f"**Question:** {question_text}")
        add()
        add(f"**Selection rationale:** {rationale}")
        add()

        # Taxonomy modes for this question
        add("**Taxonomy classifications:**")
        add()
        for name in approach_names:
            mode = example["modes"].get(name, "unknown")
            add(f"- **{display[name]}:** {mode.replace('_', ' ')}")
        add()

        # Full response text from each approach
        add("**Responses:**")
        add()
        for name in approach_names:
            resp_entry = response_map.get(qid, {}).get(name, {})
            resp_text = resp_entry.get("response", "(no response)")
            # Full untruncated text, collapse consecutive blank lines
            resp_clean = resp_text.strip()
            resp_clean = re.sub(r'\n{3,}', '\n\n', resp_clean)
            add(f"**{display[name]}:**")
            add()
            add(f"> {resp_clean}")
            add()

        # Retrieved chunks for RAG approaches
        has_any_chunks = False
        for name in rag_names:
            resp_entry = response_map.get(qid, {}).get(name, {})
            retrieval = resp_entry.get("retrieval")
            if retrieval and retrieval.get("chunks"):
                has_any_chunks = True
                break

        if has_any_chunks:
            add("**Retrieved chunks (top 2-3, truncated to ~300 chars):**")
            add()
            for name in rag_names:
                resp_entry = response_map.get(qid, {}).get(name, {})
                retrieval = resp_entry.get("retrieval")
                if not retrieval or not retrieval.get("chunks"):
                    continue

                chunks = retrieval["chunks"][:3]  # Top 2-3
                distances = retrieval.get("distances", [])[:3]
                add(f"*{display[name]}:*")
                add()
                for ci, chunk in enumerate(chunks):
                    dist_str = f" (dist: {distances[ci]:.4f})" if ci < len(distances) else ""
                    chunk_truncated = chunk[:300]
                    if len(chunk) > 300:
                        chunk_truncated += " [...]"
                    add(f"  {ci + 1}. {chunk_truncated}{dist_str}")
                add()

        add("---")
        add()

    # -----------------------------------------------------------------------
    # Summary judgment
    # -----------------------------------------------------------------------
    add("## Summary judgment")
    add()
    add(
        f"This comparison of {len(approach_names)} approaches across "
        f"{question_count} ALS benchmark questions reveals three key findings."
    )
    add()
    add(
        f"First, retrieval-augmented generation does not improve over the bare "
        f"Llama 3.1 8B baseline on this benchmark. The baseline's parametric "
        f"knowledge ({baseline_acc:.1%} accuracy) outperforms every RAG "
        f"configuration, including the best-performing "
        f"{display.get(best_rag_name, 'N/A')} ({best_rag_acc:.1%}). "
        f"This suggests that for specialized medical questions, the retrieval "
        f"component introduces noise that degrades the model's ability to "
        f"leverage its existing knowledge."
    )
    add()
    add(
        f"Second, embedding model choice is the single most impactful "
        f"configuration variable. PubMedBERT-based retrieval achieves "
        f"dramatically higher accuracy than MiniLM-based retrieval, with the "
        f"500-token MiniLM configuration producing "
        f"{taxonomy_dist.get('rag_500_minilm', {}).get('degenerate', {}).get('pct', 0):.0f}% "
        f"degenerate responses compared to "
        f"{taxonomy_dist.get('rag_500_pubmedbert', {}).get('degenerate', {}).get('pct', 0):.0f}% "
        f"for PubMedBERT at the same chunk size. Domain-specific embeddings "
        f"retrieve more relevant content, reducing context pollution."
    )
    add()
    add(
        f"Third, the from-scratch ALS-LM and the instruction-tuned "
        f"baseline/RAG approaches fail in fundamentally different ways. The "
        f"from-scratch model produces incoherent, degenerate output "
        f"({taxonomy_dist.get('als_lm', {}).get('degenerate', {}).get('pct', 0):.1f}% "
        f"degenerate) because it lacks basic language competence. The baseline "
        f"and RAG approaches produce fluent but fabricated answers "
        f"({taxonomy_dist.get('baseline', {}).get('confident_fabrication', {}).get('pct', 0):.1f}% "
        f"confident fabrication for baseline). These represent opposite ends of "
        f"the competence-reliability spectrum: the small domain model cannot "
        f"speak coherently, while the large general model speaks confidently "
        f"but incorrectly."
    )
    add()

    # -----------------------------------------------------------------------
    # Metadata footer
    # -----------------------------------------------------------------------
    add("---")
    add()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    add(f"*Generated: {timestamp}*")
    add(f"*Approaches compared: {len(approach_names)}*")
    add(f"*Questions evaluated: {question_count}*")
    add()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json_output(
    approach_names: list[str],
    approaches: list[dict],
    overall_accuracy: dict,
    per_category: dict,
    fabrication_rates: dict,
    taxonomy_dist: dict,
    failure_decomposition: dict,
    contrast_questions: list[dict],
    deep_dive_examples: list[dict],
    all_data: dict,
    questions: list[dict],
    question_count: int,
) -> dict:
    """Build the structured JSON comparison output.

    Contains all computed data for programmatic consumption including
    metrics, failure decomposition, and qualitative examples.
    """
    display = {a["name"]: a["display"] for a in approaches}
    approach_types = {a["name"]: a["approach_type"] for a in approaches}
    question_map = {q["id"]: q for q in questions}

    # Build response lookup for deep-dives
    response_map = {}
    for name in approach_names:
        resp_list = all_data[name]["responses"]
        if isinstance(resp_list, dict) and "responses" in resp_list:
            resp_list = resp_list["responses"]
        for entry in resp_list:
            qid = entry["question_id"]
            if qid not in response_map:
                response_map[qid] = {}
            response_map[qid][name] = entry

    # Build deep-dive data with full responses and chunks
    deep_dive_data = []
    for example in deep_dive_examples:
        qid = example["question_id"]
        q_obj = question_map.get(qid, {})
        dd = {
            "question_id": qid,
            "category": example.get("category", ""),
            "question_text": q_obj.get("question", q_obj.get("prompt_template", "")),
            "pattern": example.get("pattern", ""),
            "rationale": example.get("rationale", ""),
            "modes": example.get("modes", {}),
            "responses": {},
        }
        for name in approach_names:
            resp_entry = response_map.get(qid, {}).get(name, {})
            dd["responses"][name] = {
                "response": resp_entry.get("response", ""),
                "retrieval_chunks": None,
            }
            retrieval = resp_entry.get("retrieval")
            if retrieval and retrieval.get("chunks"):
                dd["responses"][name]["retrieval_chunks"] = retrieval["chunks"][:3]
        deep_dive_data.append(dd)

    # Build contrast questions data
    contrast_data = []
    for item in contrast_questions:
        contrast_data.append({
            "question_id": item["question_id"],
            "category": item["category"],
            "modes": item["modes"],
            "n_distinct_modes": item["n_distinct_modes"],
            "mixed_score": item["mixed_score"],
        })

    # Summary findings
    rag_names = [n for n in approach_names if approach_types.get(n) == "rag"]
    baseline_acc = overall_accuracy.get("baseline", {}).get("mean_accuracy", 0)
    best_rag = max(rag_names, key=lambda n: overall_accuracy.get(n, {}).get("mean_accuracy", 0)) if rag_names else None

    summary = {
        "headline": "Baseline outperforms all RAG configurations; retrieval does not help",
        "baseline_accuracy": baseline_acc,
        "best_rag_config": best_rag,
        "best_rag_accuracy": overall_accuracy.get(best_rag, {}).get("mean_accuracy", 0) if best_rag else 0,
        "key_findings": [
            "Retrieval-augmented generation does not improve over bare baseline on this benchmark",
            "Embedding model choice (PubMedBERT vs MiniLM) is the single biggest performance lever",
            "From-scratch model fails via incoherence; baseline/RAG fail via confident fabrication",
            "500-token MiniLM chunks cause catastrophic degenerate output",
        ],
    }

    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "approaches": [
                {
                    "name": name,
                    "display": display[name],
                    "type": approach_types[name],
                }
                for name in approach_names
            ],
            "question_count": question_count,
        },
        "overall_accuracy": overall_accuracy,
        "per_category_accuracy": per_category,
        "fabrication_rates": fabrication_rates,
        "taxonomy_distribution": taxonomy_dist,
        "failure_decomposition": {
            name: failure_decomposition.get(name)
            for name in approach_names
        },
        "contrast_questions": contrast_data,
        "deep_dive_examples": deep_dive_data,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Orchestrate the full comparison pipeline."""
    args = parse_args()

    if args.verbose:
        print("Cross-Approach Comparison Script")
        print()

    # 1. Find project root
    project_root = find_project_root()
    output_dir = args.output_dir or (project_root / "rag" / "results")

    if args.verbose:
        print(f"  Project root: {project_root}")
        print(f"  Output dir:   {output_dir}")
        print()

    # 2. Discover approaches
    if args.verbose:
        print("Discovering approach result directories...")
    approaches = discover_approaches(
        project_root,
        verbose=args.verbose,
        from_scratch_dir=args.from_scratch_dir,
        baseline_dir=args.baseline_dir,
        rag_dir=args.rag_dir,
    )

    if len(approaches) < 2:
        print(
            f"ERROR: Need at least 2 approaches for comparison, found "
            f"{len(approaches)}.",
            file=sys.stderr,
        )
        return 1

    if len(approaches) < 6:
        print(
            f"WARNING: Expected 6 approaches, found {len(approaches)}. "
            f"Proceeding with available data.",
            file=sys.stderr,
        )

    approach_names = [a["name"] for a in approaches]

    if args.verbose:
        print(f"  Comparing {len(approaches)} approaches: "
              f"{[a['display'] for a in approaches]}")
        print()

    # 3. Load all data
    if args.verbose:
        print("Loading evaluation data...")
    all_data = {}
    for approach in approaches:
        if args.verbose:
            print(f"  Loading {approach['display']}...")
        all_data[approach["name"]] = load_approach_data(approach)

    # 4. Load benchmark questions
    if args.verbose:
        print("Loading benchmark questions...")
    questions = load_benchmark_questions(project_root, args.benchmark_path)
    question_count = len(questions)
    if args.verbose:
        print(f"  Loaded {question_count} questions")
        print()

    # 5. Compute all metrics
    if args.verbose:
        print("Computing metrics...")

    overall_accuracy = compute_overall_accuracy(all_data)
    per_category = compute_per_category_accuracy(all_data)
    fabrication_rates = compute_fabrication_rates(all_data)
    taxonomy_dist = compute_taxonomy_distribution(all_data)

    if args.verbose:
        print("  Overall accuracy computed")
        print("  Per-category accuracy computed")
        print("  Fabrication rates computed")
        print("  Taxonomy distribution computed")

    # 6. Compute failure decomposition for RAG approaches
    if args.verbose:
        print("  Computing failure decomposition for RAG approaches...")
    failure_decomposition = {}
    for approach in approaches:
        name = approach["name"]
        if approach["approach_type"] != "rag":
            failure_decomposition[name] = None
            continue

        responses_data = all_data[name]["responses"]
        if isinstance(responses_data, dict) and "responses" in responses_data:
            responses_list = responses_data["responses"]
        else:
            responses_list = responses_data

        scores_per_q = all_data[name]["scores"]["per_question"]
        decomp = compute_failure_decomposition(
            questions, responses_list, scores_per_q,
        )
        failure_decomposition[name] = decomp
        if args.verbose:
            print(
                f"    {approach['display']}: "
                f"{decomp['retrieval_failures']} retrieval / "
                f"{decomp['generation_failures']} generation failures "
                f"(of {decomp['total_wrong']} wrong)"
            )

    # 7. Select contrast and deep-dive examples
    if args.verbose:
        print()
        print("Selecting qualitative examples...")

    # Build taxonomy and scores lookup for selection
    all_taxonomy = {}
    all_scores = {}
    for name in approach_names:
        all_taxonomy[name] = all_data[name]["taxonomy"]
        all_scores[name] = all_data[name]["scores"]

    contrast_questions = select_contrast_questions(
        all_taxonomy, all_scores, approach_names, n=10,
    )
    if args.verbose:
        print(f"  Selected {len(contrast_questions)} contrast questions")

    deep_dive_examples = select_deep_dive_examples(
        contrast_questions, all_data, approach_names, approaches, n=4,
    )
    if args.verbose:
        print(f"  Selected {len(deep_dive_examples)} deep-dive examples")
        for dd in deep_dive_examples:
            print(f"    {dd['question_id']}: {dd.get('pattern', 'N/A')}")
        print()

    # 8. Generate Markdown report
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "comparison_report.md"
    if args.verbose:
        print(f"Writing Markdown report to {md_path}...")
    md_content = generate_markdown_report(
        approach_names,
        approaches,
        overall_accuracy,
        per_category,
        fabrication_rates,
        taxonomy_dist,
        failure_decomposition,
        contrast_questions,
        deep_dive_examples,
        all_data,
        questions,
        question_count,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 9. Build and write JSON output
    json_path = output_dir / "comparison_report.json"
    if args.verbose:
        print(f"Writing JSON output to {json_path}...")
    json_output = build_json_output(
        approach_names,
        approaches,
        overall_accuracy,
        per_category,
        fabrication_rates,
        taxonomy_dist,
        failure_decomposition,
        contrast_questions,
        deep_dive_examples,
        all_data,
        questions,
        question_count,
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # 10. Print summary
    if args.verbose:
        print()
        print("Done.")
        print(f"  Markdown: {md_path}")
        print(f"  JSON:     {json_path}")
        print()
        print("Summary:")
        for name in approach_names:
            acc = overall_accuracy[name]["mean_accuracy"]
            disp = next(a["display"] for a in approaches if a["name"] == name)
            print(f"  {disp:>16}: {acc:.4f} accuracy")

    return 0


if __name__ == "__main__":
    sys.exit(main())
