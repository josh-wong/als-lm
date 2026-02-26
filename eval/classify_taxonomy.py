#!/usr/bin/env python3
"""Classify benchmark responses into a failure taxonomy with severity labels.

Loads scored responses, fabrication results, benchmark questions, and raw
responses, then classifies each response into one of five failure modes using
rule-based logic. Supports manual overrides via a JSON overrides file that
are preserved across re-runs.

This is a research evaluation tool, not a medical information system.

Failure modes:
    - confident_fabrication: Fabricated entities asserted without hedging.
    - plausible_blending: Real facts mixed with incorrect details.
    - outdated_information: Temporal facts referenced incorrectly.
    - boundary_confusion: Wrong-domain information with hedging.
    - accurate_but_misleading: Correct facts framed without appropriate caveats.
    - accurate: Correct response (not a failure mode).
    - degenerate: Empty or garbage output.

Classification priority (first match wins):
    confident_fabrication > outdated_information > plausible_blending >
    boundary_confusion > accurate_but_misleading > fallback.

Usage examples::

    # Classify with default paths
    python eval/classify_taxonomy.py

    # Custom paths with overrides
    python eval/classify_taxonomy.py \\
        --scores eval/results/scores.json \\
        --fabrications eval/results/fabrications.json \\
        --overrides eval/taxonomy_overrides.json \\
        --output eval/results/taxonomy.json

    # Show help
    python eval/classify_taxonomy.py --help
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Auto-discover project root for default paths
try:
    from eval.utils import find_project_root, resolve_default_paths
    _PROJECT_ROOT = find_project_root()
    _DEFAULTS = resolve_default_paths(_PROJECT_ROOT)
except (ImportError, SystemExit):
    _PROJECT_ROOT = None
    _DEFAULTS = {}


# ---------------------------------------------------------------------------
# Failure mode definitions
# ---------------------------------------------------------------------------

FAILURE_MODES = [
    "confident_fabrication",
    "plausible_blending",
    "outdated_information",
    "boundary_confusion",
    "accurate_but_misleading",
    "accurate",
    "degenerate",
]

# Categories that test temporal/time-sensitive knowledge
TEMPORAL_CATEGORIES = {"clinical_trials", "temporal_accuracy"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for taxonomy classification."""
    parser = argparse.ArgumentParser(
        description="Classify benchmark responses into failure taxonomy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/classify_taxonomy.py\n"
            "  python eval/classify_taxonomy.py --overrides eval/taxonomy_overrides.json\n"
            "  python eval/classify_taxonomy.py --scores eval/results/scores.json\n"
        ),
    )
    parser.add_argument(
        "--scores",
        type=str,
        default="eval/results/scores.json",
        help="Path to scoring output JSON (default: eval/results/scores.json)",
    )
    parser.add_argument(
        "--fabrications",
        type=str,
        default="eval/results/fabrications.json",
        help="Path to fabrication output JSON (default: eval/results/fabrications.json)",
    )
    parser.add_argument(
        "--responses",
        type=str,
        default="eval/results/responses.json",
        help="Path to generated responses JSON (default: eval/results/responses.json)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(_DEFAULTS["benchmark"]) if "benchmark" in _DEFAULTS else "eval/questions.json",
        help="Path to benchmark questions JSON (default: eval/questions.json)",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default="eval/taxonomy_overrides.json",
        help="Path to manual overrides JSON (default: eval/taxonomy_overrides.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/taxonomy.json",
        help="Path for taxonomy output JSON (default: eval/results/taxonomy.json)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify_response(score_entry, flagged_count, category, difficulty,
                      tokens_generated):
    """Classify a single response into a failure mode with severity.

    Rules are evaluated in priority order. First match wins for the primary
    label. A secondary label is assigned if another rule also matches.

    Args:
        score_entry: Dict with accuracy_proportional and hedging_count from
            the scoring output.
        flagged_count: Number of flagged (potentially fabricated) entities.
        category: Question category string.
        difficulty: Question difficulty string.
        tokens_generated: Number of tokens in the model response.

    Returns:
        A dict with primary_mode, secondary_mode, and severity.
    """
    accuracy = score_entry["accuracy_proportional"]
    hedging_count = score_entry["hedging_count"]

    matches = []

    # Rule 1: confident_fabrication
    if flagged_count > 0 and hedging_count <= 1:
        matches.append(("confident_fabrication", "high"))

    # Rule 2: outdated_information (checked before plausible_blending)
    if category in TEMPORAL_CATEGORIES and accuracy < 0.5:
        matches.append(("outdated_information", "medium"))

    # Rule 3: plausible_blending
    if 0.2 <= accuracy < 0.5 and flagged_count == 0:
        severity = "high" if hedging_count == 0 else "medium"
        matches.append(("plausible_blending", severity))

    # Rule 4: boundary_confusion
    if accuracy < 0.3 and hedging_count >= 2 and flagged_count == 0:
        severity = "low" if hedging_count >= 3 else "medium"
        matches.append(("boundary_confusion", severity))

    # Rule 5: accurate_but_misleading
    if accuracy >= 0.5 and hedging_count == 0 and difficulty in ("hard", "medium"):
        severity = "medium" if difficulty == "hard" else "low"
        matches.append(("accurate_but_misleading", severity))

    # Determine primary and secondary from matches
    if matches:
        primary_mode, severity = matches[0]
        secondary_mode = matches[1][0] if len(matches) > 1 else None
        return {
            "primary_mode": primary_mode,
            "secondary_mode": secondary_mode,
            "severity": severity,
        }

    # Fallback classification
    if accuracy >= 0.5:
        return {
            "primary_mode": "accurate",
            "secondary_mode": None,
            "severity": "none",
        }

    if accuracy < 0.2 and tokens_generated < 10:
        return {
            "primary_mode": "degenerate",
            "secondary_mode": None,
            "severity": "low",
        }

    # Default fallback: plausible_blending
    return {
        "primary_mode": "plausible_blending",
        "secondary_mode": None,
        "severity": "medium",
    }


# ---------------------------------------------------------------------------
# Override handling
# ---------------------------------------------------------------------------

def load_overrides(overrides_path):
    """Load manual taxonomy overrides from a JSON file.

    The overrides file contains a list of objects with question_id,
    primary_mode, severity, and reason fields. If the file does not
    exist, returns an empty dict.

    Args:
        overrides_path: Path to the overrides JSON file.

    Returns:
        A dict mapping question_id to override entry.
    """
    if not os.path.isfile(overrides_path):
        return {}

    with open(overrides_path) as f:
        overrides_list = json.load(f)

    overrides = {}
    for entry in overrides_list:
        qid = entry.get("question_id")
        if qid:
            overrides[qid] = entry

    return overrides


def apply_override(classification, override_entry):
    """Apply a manual override to an automated classification.

    Args:
        classification: The automated classification dict for a question.
        override_entry: The override entry with primary_mode, severity,
            and reason.

    Returns:
        The classification dict with override applied.
    """
    classification["primary_mode"] = override_entry["primary_mode"]
    classification["severity"] = override_entry["severity"]
    classification["override"] = True
    classification["override_reason"] = override_entry.get("reason", "")
    return classification


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

def compute_distributions(per_question):
    """Compute failure mode and severity distributions.

    Args:
        per_question: List of per-question classification dicts.

    Returns:
        (mode_distribution, severity_distribution) dicts.
    """
    total = len(per_question)

    # Mode distribution
    mode_counts = {}
    for entry in per_question:
        mode = entry["primary_mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    mode_distribution = {}
    for mode in FAILURE_MODES:
        count = mode_counts.get(mode, 0)
        pct = round(count / total * 100, 1) if total > 0 else 0.0
        mode_distribution[mode] = {"count": count, "pct": pct}

    # Severity distribution
    severity_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for entry in per_question:
        sev = entry["severity"]
        if sev in severity_counts:
            severity_counts[sev] += 1

    return mode_distribution, severity_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for taxonomy classification."""
    args = parse_args()

    print("\n=== ALS-LM Failure Taxonomy Classification ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical"
          " information system.\n")

    # Load scores
    if not os.path.isfile(args.scores):
        print(f"ERROR: Scores file not found: {args.scores}")
        print("  Run eval/score_responses.py first to score model responses.")
        sys.exit(1)

    with open(args.scores) as f:
        scores_data = json.load(f)

    scores_by_id = {q["question_id"]: q for q in scores_data.get("per_question", [])}
    print(f"  Loaded {len(scores_by_id)} scored questions from {args.scores}")

    # Load fabrications
    if not os.path.isfile(args.fabrications):
        print(f"ERROR: Fabrications file not found: {args.fabrications}")
        print("  Run eval/detect_fabrications.py first.")
        sys.exit(1)

    with open(args.fabrications) as f:
        fabrications_data = json.load(f)

    fab_by_id = {q["question_id"]: q for q in fabrications_data.get("per_question", [])}
    print(f"  Loaded {len(fab_by_id)} fabrication results from {args.fabrications}")

    # Load responses (for tokens_generated)
    if not os.path.isfile(args.responses):
        print(f"ERROR: Responses file not found: {args.responses}")
        print("  Run eval/generate_responses.py first.")
        sys.exit(1)

    with open(args.responses) as f:
        responses_data = json.load(f)

    resp_by_id = {r["question_id"]: r for r in responses_data.get("responses", [])}
    print(f"  Loaded {len(resp_by_id)} responses from {args.responses}")

    # Load benchmark (for category metadata)
    if not os.path.isfile(args.benchmark):
        print(f"ERROR: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    with open(args.benchmark) as f:
        benchmark = json.load(f)

    bench_by_id = {q["id"]: q for q in benchmark}
    print(f"  Loaded {len(bench_by_id)} benchmark questions from {args.benchmark}")

    # Load overrides
    overrides = load_overrides(args.overrides)
    if overrides:
        print(f"  Loaded {len(overrides)} manual overrides from {args.overrides}")
    else:
        print(f"  No overrides file found at {args.overrides} (proceeding without)")

    # Classify each question
    per_question = []
    overrides_applied = 0
    total = len(scores_by_id)

    for i, (qid, score_entry) in enumerate(scores_by_id.items()):
        # Progress every 20 questions
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Classifying {i + 1}/{total}...")

        # Gather signals
        fab_entry = fab_by_id.get(qid, {"flagged_entities": []})
        flagged_count = len(fab_entry.get("flagged_entities", []))

        bench_entry = bench_by_id.get(qid, {})
        category = score_entry.get("category", bench_entry.get("category", ""))
        difficulty = score_entry.get("difficulty", bench_entry.get("difficulty", ""))

        resp_entry = resp_by_id.get(qid, {})
        tokens_generated = resp_entry.get("tokens_generated", 0)

        # Run classification
        result = classify_response(
            score_entry=score_entry,
            flagged_count=flagged_count,
            category=category,
            difficulty=difficulty,
            tokens_generated=tokens_generated,
        )

        classification = {
            "question_id": qid,
            "category": category,
            "difficulty": difficulty,
            "primary_mode": result["primary_mode"],
            "secondary_mode": result["secondary_mode"],
            "severity": result["severity"],
            "override": False,
            "signals": {
                "accuracy": score_entry["accuracy_proportional"],
                "hedging_count": score_entry["hedging_count"],
                "flagged_entities": flagged_count,
                "tokens_generated": tokens_generated,
            },
        }

        # Apply override if present
        if qid in overrides:
            classification = apply_override(classification, overrides[qid])
            overrides_applied += 1

        per_question.append(classification)

    print(f"  Classification complete: {len(per_question)} questions classified")
    if overrides_applied > 0:
        print(f"  Overrides applied: {overrides_applied}")

    # Compute distributions
    mode_distribution, severity_distribution = compute_distributions(per_question)

    # Build output
    output = {
        "metadata": {
            "scores_path": args.scores,
            "fabrications_path": args.fabrications,
            "responses_path": args.responses,
            "benchmark_path": args.benchmark,
            "overrides_path": args.overrides,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_classified": len(per_question),
            "overrides_applied": overrides_applied,
        },
        "distribution": mode_distribution,
        "severity_distribution": severity_distribution,
        "per_question": per_question,
    }

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n  === Taxonomy Distribution ===")
    for mode, stats in mode_distribution.items():
        if stats["count"] > 0:
            print(f"    {mode:30s} {stats['count']:4d} ({stats['pct']:.1f}%)")

    print(f"\n  === Severity Distribution ===")
    for sev, count in severity_distribution.items():
        print(f"    {sev:10s} {count:4d}")

    print(f"\n  Output saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
