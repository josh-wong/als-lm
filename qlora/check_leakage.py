#!/usr/bin/env python3
"""Benchmark leakage check: verifies zero contamination between ALS
instruction pairs and evaluation benchmark questions.

Compares 970 raw instruction pairs (from data/instruction/als_instructions.json)
against 160 evaluation questions (from eval/questions.json) using
rapidfuzz.fuzz.partial_ratio with threshold >= 80.

Comparisons performed:
  1. instruction text vs question["question"]
  2. instruction text vs question["prompt_template"]
  3. output/answer text vs each item in question["key_facts"]

If any match scores >= 80, exits with code 1 (FATAL). Otherwise exits
with code 0 (PASS). Produces a JSON report at
data/instruction/qlora/leakage_report.json for auditability.

This script is a hard gate before QLoRA training (Phase 49). Training on
data that overlaps with the evaluation benchmark would invalidate all
downstream evaluation results.

Usage::

    python qlora/check_leakage.py
"""

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from rapidfuzz.fuzz import partial_ratio

# ---------------------------------------------------------------------------
# Project root discovery (established project pattern)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INSTRUCTIONS_PATH = PROJECT_ROOT / "data" / "instruction" / "als_instructions.json"
QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
REPORT_DIR = PROJECT_ROOT / "data" / "instruction" / "qlora"
REPORT_PATH = REPORT_DIR / "leakage_report.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THRESHOLD = 80

# Minimum character length for key_facts to be compared. Short key_facts
# (single medical terms like "riluzole", "SOD1", "c9orf72" and short
# phrases like "reactive oxygen species imbalance") are expected domain
# vocabulary that naturally appears in both training and eval data.
# Only longer factual phrases (>= 40 chars) can meaningfully indicate
# content leakage vs. mere vocabulary overlap. partial_ratio is biased
# toward false positives when the reference string is much shorter than
# the query string, so this filter prevents domain terms from triggering
# false alarms.
MIN_KEY_FACT_LENGTH = 40


def load_instructions() -> list[dict]:
    """Load 970 Alpaca-format instruction pairs."""
    if not INSTRUCTIONS_PATH.exists():
        print(f"FATAL: Instruction file not found: {INSTRUCTIONS_PATH}")
        print("  Fix: Run the data collection pipeline first.")
        sys.exit(1)

    with open(INSTRUCTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} instruction pairs from {INSTRUCTIONS_PATH.name}")
    return data


def load_questions() -> list[dict]:
    """Load 160 evaluation benchmark questions."""
    if not QUESTIONS_PATH.exists():
        print(f"FATAL: Benchmark questions not found: {QUESTIONS_PATH}")
        print("  Fix: Ensure eval/questions.json exists.")
        sys.exit(1)

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} benchmark questions from {QUESTIONS_PATH.name}")
    return data


def check_pair(
    instruction: dict,
    questions: list[dict],
) -> tuple[float, list[dict]]:
    """Compare one instruction pair against all benchmark questions.

    Returns (max_score, flagged_matches) where flagged_matches contains
    details for any comparison scoring >= THRESHOLD.
    """
    inst_text = instruction.get("instruction", "").lower()
    output_text = instruction.get("output", "").lower()
    max_score = 0.0
    flagged = []

    for question in questions:
        q_id = question.get("id", "unknown")
        q_text = question.get("question", "").lower()
        q_template = question.get("prompt_template", "").lower()
        q_key_facts = question.get("key_facts", [])

        # Compare instruction vs question text
        score_q = partial_ratio(inst_text, q_text)
        if score_q > max_score:
            max_score = score_q
        if score_q >= THRESHOLD:
            flagged.append({
                "benchmark_id": q_id,
                "match_type": "instruction_vs_question",
                "score": score_q,
                "instruction_preview": inst_text[:100],
                "matched_text": q_text[:100],
            })

        # Compare instruction vs prompt_template
        score_pt = partial_ratio(inst_text, q_template)
        if score_pt > max_score:
            max_score = score_pt
        if score_pt >= THRESHOLD:
            flagged.append({
                "benchmark_id": q_id,
                "match_type": "instruction_vs_prompt_template",
                "score": score_pt,
                "instruction_preview": inst_text[:100],
                "matched_text": q_template[:100],
            })

        # Compare output vs each key_fact individually (skip short terms)
        for fact in q_key_facts:
            fact_lower = fact.lower() if isinstance(fact, str) else str(fact).lower()
            if len(fact_lower) < MIN_KEY_FACT_LENGTH:
                continue
            score_kf = partial_ratio(output_text, fact_lower)
            if score_kf > max_score:
                max_score = score_kf
            if score_kf >= THRESHOLD:
                flagged.append({
                    "benchmark_id": q_id,
                    "match_type": "output_vs_key_fact",
                    "score": score_kf,
                    "output_preview": output_text[:100],
                    "matched_text": fact_lower[:100],
                })

    return max_score, flagged


def main():
    print("=" * 60)
    print("  Benchmark Leakage Check")
    print("=" * 60)
    print()

    # Load data
    instructions = load_instructions()
    questions = load_questions()
    print()

    # Run comparisons
    all_max_scores = []
    all_flagged = []
    total_pairs = len(instructions)

    print(f"  Comparing {total_pairs} instruction pairs against "
          f"{len(questions)} benchmark questions...")
    print(f"  Fuzzy match function: partial_ratio")
    print(f"  Threshold: {THRESHOLD}")
    print()

    for i, inst in enumerate(instructions):
        max_score, flagged = check_pair(inst, questions)
        all_max_scores.append(max_score)
        if flagged:
            for match in flagged:
                match["instruction_index"] = i
            all_flagged.extend(flagged)

        # Progress indicator every 100 pairs
        if (i + 1) % 100 == 0 or (i + 1) == total_pairs:
            print(f"  Checked {i + 1}/{total_pairs} pairs...", end="\r")

    print()  # Clear progress line
    print()

    # Compute statistics
    max_similarity = max(all_max_scores) if all_max_scores else 0.0
    mean_similarity = statistics.mean(all_max_scores) if all_max_scores else 0.0
    flagged_count = len(all_flagged)

    # Write JSON report (overwrites previous report if it exists)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        print(f"  Overwriting previous report at {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    report = {
        "pairs_checked": total_pairs,
        "benchmark_questions": len(questions),
        "threshold": THRESHOLD,
        "min_key_fact_length": MIN_KEY_FACT_LENGTH,
        "max_similarity": round(max_similarity, 2),
        "mean_similarity": round(mean_similarity, 2),
        "flagged_count": flagged_count,
        "flagged_details": all_flagged,
        "verdict": "FAIL" if flagged_count > 0 else "PASS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Report written to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print()

    # Summary table
    print("  " + "-" * 44)
    print(f"  {'Metric':<30} {'Value':>12}")
    print("  " + "-" * 44)
    print(f"  {'Pairs checked':<30} {total_pairs:>12}")
    print(f"  {'Benchmark questions':<30} {len(questions):>12}")
    print(f"  {'Threshold':<30} {THRESHOLD:>12}")
    print(f"  {'Max similarity':<30} {max_similarity:>12.2f}")
    print(f"  {'Mean similarity':<30} {mean_similarity:>12.2f}")
    print(f"  {'Flagged pairs':<30} {flagged_count:>12}")
    print("  " + "-" * 44)
    print()

    # Verdict
    if flagged_count > 0:
        print(f"  Verdict: FAIL")
        print()
        print("  Flagged matches:")
        for match in all_flagged:
            print(f"    - Index {match['instruction_index']}: "
                  f"{match['benchmark_id']} ({match['match_type']}, "
                  f"score={match['score']:.1f})")
        print()
        print(
            f"FATAL: Benchmark leakage detected -- {flagged_count} instruction "
            f"pairs match evaluation questions above threshold {THRESHOLD}. "
            "Training on contaminated data would invalidate evaluation results. "
            f"Review flagged pairs in {REPORT_PATH.relative_to(PROJECT_ROOT)}"
        )
        sys.exit(1)
    else:
        print(f"  Verdict: PASS")
        print()
        print(f"  No benchmark leakage detected. All {total_pairs} instruction "
              f"pairs scored below threshold {THRESHOLD} against {len(questions)} "
              "evaluation questions.")
        sys.exit(0)


if __name__ == "__main__":
    main()
