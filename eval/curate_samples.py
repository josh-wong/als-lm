#!/usr/bin/env python3
"""Select and annotate qualitative samples for the ALS-LM evaluation report.

Curates the best, worst, and edge-case model responses from scoring data with
automated 2-3 sentence annotations explaining what the model did right or wrong.
Sample selection is fully reproducible from scoring data alone.

This is a research evaluation tool, not a medical information system.

Selection criteria:
    - Best: Top N responses by accuracy_proportional (descending).
    - Worst: Bottom N responses by accuracy_proportional (ascending),
      excluding degenerate responses (tokens_generated < 10).
    - Edge cases: N responses closest to 0.5 accuracy threshold,
      representing borderline knowledge fragmentation.

Usage examples::

    # Curate samples with default paths and counts
    python eval/curate_samples.py

    # Custom counts
    python eval/curate_samples.py --best-count 5 --worst-count 5 --edge-count 5

    # Custom paths
    python eval/curate_samples.py \\
        --scores eval/results/scores.json \\
        --taxonomy eval/results/taxonomy.json \\
        --output eval/results/samples.json

    # Show help
    python eval/curate_samples.py --help
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for sample curation."""
    parser = argparse.ArgumentParser(
        description="Curate qualitative samples for evaluation report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/curate_samples.py\n"
            "  python eval/curate_samples.py --best-count 5 --worst-count 5\n"
            "  python eval/curate_samples.py --scores eval/results/scores.json\n"
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
        "--taxonomy",
        type=str,
        default="eval/results/taxonomy.json",
        help="Path to taxonomy output JSON (default: eval/results/taxonomy.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/samples.json",
        help="Path for sample output JSON (default: eval/results/samples.json)",
    )
    parser.add_argument(
        "--best-count",
        type=int,
        default=10,
        help="Number of best samples to select (default: 10)",
    )
    parser.add_argument(
        "--worst-count",
        type=int,
        default=10,
        help="Number of worst samples to select (default: 10)",
    )
    parser.add_argument(
        "--edge-count",
        type=int,
        default=10,
        help="Number of edge case samples to select (default: 10)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Selection functions
# ---------------------------------------------------------------------------

def select_best(scores, count=10):
    """Select the highest-scoring responses.

    Sorts by accuracy_proportional descending and returns the top N.
    If fewer than N have accuracy > 0, returns all that scored above 0.

    Args:
        scores: List of per-question score dicts with accuracy_proportional.
        count: Maximum number of best samples to select.

    Returns:
        A list of score dicts for the best responses.
    """
    sorted_scores = sorted(scores, key=lambda x: x["accuracy_proportional"],
                           reverse=True)

    # Take top N, but if fewer than N have accuracy > 0, take all above 0
    above_zero = [s for s in sorted_scores if s["accuracy_proportional"] > 0]
    if len(above_zero) < count:
        return above_zero

    return sorted_scores[:count]


def select_worst(scores, count=10):
    """Select the lowest-scoring non-degenerate responses.

    Sorts by accuracy_proportional ascending, excluding degenerate responses
    (tokens_generated < 10) since those are uninteresting.

    Args:
        scores: List of per-question score dicts with accuracy_proportional
            and tokens_generated.
        count: Maximum number of worst samples to select.

    Returns:
        A list of score dicts for the worst responses.
    """
    # Filter out degenerate responses
    non_degenerate = [s for s in scores if s.get("tokens_generated", 0) >= 10]

    sorted_scores = sorted(non_degenerate,
                           key=lambda x: x["accuracy_proportional"])
    return sorted_scores[:count]


def select_edge_cases(scores, count=10):
    """Select responses closest to the 0.5 accuracy threshold.

    These borderline responses show where the model's knowledge fragments,
    neither clearly right nor clearly wrong.

    Args:
        scores: List of per-question score dicts with accuracy_proportional.
        count: Maximum number of edge case samples to select.

    Returns:
        A list of score dicts for the edge case responses.
    """
    sorted_scores = sorted(scores,
                           key=lambda x: abs(x["accuracy_proportional"] - 0.5))
    return sorted_scores[:count]


# ---------------------------------------------------------------------------
# Annotation generation
# ---------------------------------------------------------------------------

def _format_category(category):
    """Format a category string for human-readable display.

    Args:
        category: Raw category string (e.g., "drug_treatment").

    Returns:
        Formatted string (e.g., "drug treatment").
    """
    return category.replace("_", " ")


def generate_annotation(category, accuracy, key_facts_total, key_facts_found,
                        key_facts_missed, flagged_count, failure_mode,
                        sample_type):
    """Generate a 2-3 sentence annotation for a curated sample.

    Describes what the model did, specific fact hits/misses, and why this
    sample is notable for its category (best, worst, or edge case).

    Args:
        category: Question category string.
        accuracy: Accuracy score (0-1).
        key_facts_total: Total number of key facts.
        key_facts_found: List of found fact dicts with "fact" key.
        key_facts_missed: List of missed fact dicts with "fact" key.
        flagged_count: Number of flagged fabricated entities.
        failure_mode: Primary failure mode from taxonomy classification.
        sample_type: One of "best", "worst", or "edge".

    Returns:
        A string annotation of 2-3 sentences.
    """
    found_count = len(key_facts_found)
    missed_count = len(key_facts_missed)
    cat_display = _format_category(category)

    # Sentence 1: What the model did
    if accuracy >= 0.8:
        action = "correctly identified"
    elif accuracy >= 0.5:
        action = "partially addressed"
    elif accuracy > 0:
        action = "failed to adequately address"
    else:
        action = "failed to identify any of"

    sentence_1 = (f"The model {action} {found_count} of {key_facts_total} "
                  f"key facts about {cat_display}.")

    # Sentence 2: Specific hits/misses
    parts = []
    if key_facts_found:
        example_fact = key_facts_found[0]["fact"]
        if len(example_fact) > 60:
            example_fact = example_fact[:57] + "..."
        parts.append(f'It matched "{example_fact}"')

    if key_facts_missed:
        example_miss = key_facts_missed[0]["fact"]
        if len(example_miss) > 60:
            example_miss = example_miss[:57] + "..."
        if parts:
            parts.append(f'but missed "{example_miss}"')
        else:
            parts.append(f'It missed "{example_miss}"')

    sentence_2 = " ".join(parts) + "." if parts else ""

    if flagged_count > 0:
        sentence_2 += (f" It also produced {flagged_count} potentially "
                       f"fabricated {'entity' if flagged_count == 1 else 'entities'}.")

    # Sentence 3: Why notable
    if sample_type == "best":
        sentence_3 = (f"This represents the model's strongest performance "
                      f"in {cat_display}.")
    elif sample_type == "worst":
        mode_display = failure_mode.replace("_", " ")
        sentence_3 = (f"This illustrates {mode_display} where the model "
                      f"produced confidently wrong output about {cat_display}.")
    else:
        sentence_3 = (f"This borderline response shows fragmented knowledge "
                      f"where the model captured some facts about "
                      f"{cat_display} but missed critical details.")

    parts = [sentence_1]
    if sentence_2:
        parts.append(sentence_2)
    parts.append(sentence_3)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sample assembly
# ---------------------------------------------------------------------------

def build_sample_entry(score_entry, bench_by_id, resp_by_id, fab_by_id,
                       taxonomy_by_id, sample_type):
    """Build a complete sample entry with all context and annotation.

    Args:
        score_entry: Per-question scoring dict.
        bench_by_id: Benchmark questions keyed by ID.
        resp_by_id: Responses keyed by question ID.
        fab_by_id: Fabrication results keyed by question ID.
        taxonomy_by_id: Taxonomy classifications keyed by question ID.
        sample_type: One of "best", "worst", or "edge".

    Returns:
        A dict with all sample context and generated annotation.
    """
    qid = score_entry["question_id"]
    bench_entry = bench_by_id.get(qid, {})
    resp_entry = resp_by_id.get(qid, {})
    fab_entry = fab_by_id.get(qid, {"flagged_entities": []})
    tax_entry = taxonomy_by_id.get(qid, {"primary_mode": "unknown"})

    flagged_count = len(fab_entry.get("flagged_entities", []))

    annotation = generate_annotation(
        category=score_entry.get("category", ""),
        accuracy=score_entry["accuracy_proportional"],
        key_facts_total=score_entry.get("key_facts_total", 0),
        key_facts_found=score_entry.get("key_facts_found", []),
        key_facts_missed=score_entry.get("key_facts_missed", []),
        flagged_count=flagged_count,
        failure_mode=tax_entry.get("primary_mode", "unknown"),
        sample_type=sample_type,
    )

    return {
        "question_id": qid,
        "question": bench_entry.get("question", ""),
        "expected_answer": bench_entry.get("verified_answer", ""),
        "model_response": resp_entry.get("response", ""),
        "accuracy": score_entry["accuracy_proportional"],
        "failure_mode": tax_entry.get("primary_mode", "unknown"),
        "annotation": annotation,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for qualitative sample curation."""
    args = parse_args()

    print("\n=== ALS-LM Qualitative Sample Curation ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical"
          " information system.\n")

    # Load scores
    if not os.path.isfile(args.scores):
        print(f"ERROR: Scores file not found: {args.scores}")
        print("  Run eval/score_responses.py first.")
        sys.exit(1)

    with open(args.scores) as f:
        scores_data = json.load(f)

    per_question_scores = scores_data.get("per_question", [])
    print(f"  Loaded {len(per_question_scores)} scored questions from"
          f" {args.scores}")

    # Load fabrications
    if not os.path.isfile(args.fabrications):
        print(f"ERROR: Fabrications file not found: {args.fabrications}")
        sys.exit(1)

    with open(args.fabrications) as f:
        fabrications_data = json.load(f)

    fab_by_id = {q["question_id"]: q
                 for q in fabrications_data.get("per_question", [])}

    # Load responses
    if not os.path.isfile(args.responses):
        print(f"ERROR: Responses file not found: {args.responses}")
        sys.exit(1)

    with open(args.responses) as f:
        responses_data = json.load(f)

    resp_by_id = {r["question_id"]: r
                  for r in responses_data.get("responses", [])}

    # Load benchmark
    if not os.path.isfile(args.benchmark):
        print(f"ERROR: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    with open(args.benchmark) as f:
        benchmark = json.load(f)

    bench_by_id = {q["id"]: q for q in benchmark}

    # Load taxonomy
    if not os.path.isfile(args.taxonomy):
        print(f"ERROR: Taxonomy file not found: {args.taxonomy}")
        print("  Run eval/classify_taxonomy.py first.")
        sys.exit(1)

    with open(args.taxonomy) as f:
        taxonomy_data = json.load(f)

    taxonomy_by_id = {q["question_id"]: q
                      for q in taxonomy_data.get("per_question", [])}

    # Enrich score entries with tokens_generated for worst selection filtering
    enriched_scores = []
    for s in per_question_scores:
        entry = dict(s)
        resp = resp_by_id.get(s["question_id"], {})
        entry["tokens_generated"] = resp.get("tokens_generated", 0)
        enriched_scores.append(entry)

    # Select samples
    best_scores = select_best(enriched_scores, count=args.best_count)
    worst_scores = select_worst(enriched_scores, count=args.worst_count)
    edge_scores = select_edge_cases(enriched_scores, count=args.edge_count)

    print(f"  Selected {len(best_scores)} best, {len(worst_scores)} worst, "
          f"{len(edge_scores)} edge cases")

    # Build sample entries with annotations
    best_samples = [
        build_sample_entry(s, bench_by_id, resp_by_id, fab_by_id,
                           taxonomy_by_id, "best")
        for s in best_scores
    ]
    worst_samples = [
        build_sample_entry(s, bench_by_id, resp_by_id, fab_by_id,
                           taxonomy_by_id, "worst")
        for s in worst_scores
    ]
    edge_samples = [
        build_sample_entry(s, bench_by_id, resp_by_id, fab_by_id,
                           taxonomy_by_id, "edge")
        for s in edge_scores
    ]

    # Build output
    output = {
        "metadata": {
            "scores_path": args.scores,
            "fabrications_path": args.fabrications,
            "responses_path": args.responses,
            "benchmark_path": args.benchmark,
            "taxonomy_path": args.taxonomy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "selection_criteria": {
                "best": f"Top {args.best_count} by accuracy_proportional",
                "worst": (f"Bottom {args.worst_count} by accuracy_proportional"
                          " (excluding degenerate)"),
                "edge": (f"{args.edge_count} closest to 0.5 accuracy"
                         " threshold"),
            },
        },
        "best": best_samples,
        "worst": worst_samples,
        "edge_cases": edge_samples,
    }

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n  === Sample Curation Summary ===")

    if best_samples:
        best_accs = [s["accuracy"] for s in best_samples]
        print(f"  Best samples:  {len(best_samples)} "
              f"(accuracy {min(best_accs):.4f} - {max(best_accs):.4f})")

    if worst_samples:
        worst_accs = [s["accuracy"] for s in worst_samples]
        print(f"  Worst samples: {len(worst_samples)} "
              f"(accuracy {min(worst_accs):.4f} - {max(worst_accs):.4f})")

    if edge_samples:
        edge_accs = [s["accuracy"] for s in edge_samples]
        print(f"  Edge cases:    {len(edge_samples)} "
              f"(accuracy {min(edge_accs):.4f} - {max(edge_accs):.4f})")

    print(f"\n  Output saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
