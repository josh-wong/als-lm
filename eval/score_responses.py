#!/usr/bin/env python3
"""Score model responses against ALS benchmark key facts using fuzzy matching.

Loads generated responses and benchmark questions, then scores each response
by fuzzy-matching against key facts using a sliding window approach with
rapidfuzz. Computes per-question proportional and binary accuracy, aggregates
metrics by category, difficulty, and overall (mean and median), and detects
hedging language frequency.

This is a research evaluation tool, not a medical information system.

Scoring methodology:
    For each response, break the text into overlapping chunks (100 chars wide,
    50-char overlap). For each key fact in the question's key_facts list,
    compute rapidfuzz.fuzz.partial_ratio against every chunk. A key fact is
    "found" if any chunk scores >= threshold (default 80). Per-question
    accuracy is the proportion of key facts found. Binary pass if >= 50%.

Usage examples::

    # Score with default paths and threshold
    python eval/score_responses.py

    # Custom paths and threshold
    python eval/score_responses.py \\
        --responses eval/results/tiny_responses.json \\
        --benchmark eval/questions.json \\
        --output eval/results/tiny_scores.json \\
        --threshold 75

    # Show help
    python eval/score_responses.py --help
"""

import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so that `from eval.utils import ...`
# resolves correctly when running as `python eval/<script>.py`.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Auto-discover project root for default paths
try:
    from eval.utils import find_project_root, resolve_default_paths
    _PROJECT_ROOT = find_project_root()
    _DEFAULTS = resolve_default_paths(_PROJECT_ROOT)
except ImportError:
    import warnings
    warnings.warn(
        "Cannot import eval.utils. Ensure you're running from within "
        "the als-lm repository.",
        stacklevel=2,
    )
    _PROJECT_ROOT = None
    _DEFAULTS = {}
except SystemExit:
    _PROJECT_ROOT = None
    _DEFAULTS = {}

from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Hedging phrases — curated list for detecting uncertainty language
# ---------------------------------------------------------------------------

HEDGING_PHRASES = [
    "may", "might", "could", "possibly", "perhaps", "potentially",
    "it is thought that", "it is believed that", "some studies suggest",
    "it has been suggested", "it is possible that", "there is evidence that",
    "appears to", "seems to", "is thought to", "is believed to",
    "likely", "unlikely", "uncertain", "unclear",
    "in some cases", "in certain cases",
]

# Pre-compile word-boundary patterns for single-word hedging phrases to
# avoid false positives (e.g., "may" inside "mayonnaise"). Multi-word
# phrases are unlikely to produce false positives, so plain `in` works.
_HEDGING_PATTERNS = []
for phrase in HEDGING_PHRASES:
    if " " in phrase:
        # Multi-word phrase: simple substring match (lowered)
        _HEDGING_PATTERNS.append((phrase, None))
    else:
        # Single word: word-boundary regex
        _HEDGING_PATTERNS.append((phrase, re.compile(r"\b" + re.escape(phrase) + r"\b")))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for response scoring."""
    parser = argparse.ArgumentParser(
        description="Score model responses against ALS benchmark key facts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/score_responses.py\n"
            "  python eval/score_responses.py --threshold 75\n"
            "  python eval/score_responses.py --responses eval/results/tiny_responses.json\n"
        ),
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
        "--output",
        type=str,
        default="eval/results/scores.json",
        help="Path for scoring output JSON (default: eval/results/scores.json)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=80,
        help="Fuzzy match threshold 0-100 (default: 80)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sliding-window fuzzy matching
# ---------------------------------------------------------------------------

def build_chunks(text, chunk_size=100, overlap=50):
    """Break text into overlapping chunks for fuzzy matching.

    Args:
        text: The response text to chunk.
        chunk_size: Width of each chunk in characters.
        overlap: Overlap between consecutive chunks in characters.

    Returns:
        A list of lowercase string chunks. If the text is shorter than
        chunk_size, a single chunk containing the full text is returned.
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
        # Stop once we have covered the end of the text
        if start + chunk_size >= len(text):
            break
    return chunks


def score_key_fact(key_fact, chunks, threshold):
    """Check whether a key fact is found in any chunk above the threshold.

    Args:
        key_fact: The key fact string to search for.
        chunks: List of lowercase text chunks from the response.
        threshold: Minimum fuzzy score (0-100) for a match.

    Returns:
        (found, best_score) where found is True if best_score >= threshold.
    """
    key_fact_lower = key_fact.lower()
    best_score = 0

    for chunk in chunks:
        score = fuzz.partial_ratio(key_fact_lower, chunk)
        if score > best_score:
            best_score = score
        # Early exit if perfect match found
        if best_score == 100:
            break

    return best_score >= threshold, best_score


# ---------------------------------------------------------------------------
# Hedging detection
# ---------------------------------------------------------------------------

def detect_hedging(response_text):
    """Count hedging phrase occurrences in a response.

    Args:
        response_text: The model response text.

    Returns:
        (total_count, phrases_found) where phrases_found is a list of
        hedging phrases that appeared at least once.
    """
    text_lower = response_text.lower()
    total_count = 0
    phrases_found = []

    for phrase, pattern in _HEDGING_PATTERNS:
        if pattern is not None:
            # Single-word: use regex for word boundaries
            matches = pattern.findall(text_lower)
            count = len(matches)
        else:
            # Multi-word: count non-overlapping substring occurrences
            count = text_lower.count(phrase)

        if count > 0:
            total_count += count
            phrases_found.append(phrase)

    return total_count, phrases_found


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------

def score_question(response_entry, benchmark_entry, threshold):
    """Score a single question's response against its key facts.

    Args:
        response_entry: Dict with "response" text and question metadata.
        benchmark_entry: Dict with "key_facts" list from the benchmark.
        threshold: Fuzzy match threshold.

    Returns:
        A dict with scoring results for this question.
    """
    response_text = response_entry.get("response", "")
    key_facts = benchmark_entry.get("key_facts", [])

    # Build chunks from the response
    chunks = build_chunks(response_text)

    # Score each key fact
    found_facts = []
    missed_facts = []
    for fact in key_facts:
        matched, best_score = score_key_fact(fact, chunks, threshold)
        entry = {"fact": fact, "best_score": best_score}
        if matched:
            found_facts.append(entry)
        else:
            missed_facts.append(entry)

    # Proportional and binary accuracy
    total_facts = len(key_facts)
    if total_facts > 0:
        accuracy_proportional = len(found_facts) / total_facts
    else:
        accuracy_proportional = 0.0
    accuracy_binary = 1 if accuracy_proportional >= 0.5 else 0

    # Hedging detection
    hedging_count, hedging_phrases_found = detect_hedging(response_text)

    return {
        "question_id": response_entry["question_id"],
        "category": response_entry["category"],
        "difficulty": response_entry["difficulty"],
        "is_trap": response_entry["is_trap"],
        "accuracy_proportional": round(accuracy_proportional, 4),
        "accuracy_binary": accuracy_binary,
        "key_facts_total": total_facts,
        "key_facts_found": found_facts,
        "key_facts_missed": missed_facts,
        "hedging_count": hedging_count,
        "hedging_phrases_found": hedging_phrases_found,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_aggregate(scored_questions):
    """Compute aggregate metrics from per-question scores.

    Computes overall, per-category, per-difficulty, and trap-question
    aggregate metrics including mean accuracy, median accuracy, binary
    pass rate, and hedging counts.

    Args:
        scored_questions: List of per-question scoring dicts.

    Returns:
        A dict with "overall", "by_category", "by_difficulty", and
        "trap_questions" sub-dicts.
    """
    if not scored_questions:
        return {
            "overall": {
                "mean_accuracy": 0.0,
                "median_accuracy": 0.0,
                "binary_pass_rate": 0.0,
                "total_hedging_instances": 0,
            },
            "by_category": {},
            "by_difficulty": {},
            "trap_questions": {"count": 0, "mean_accuracy": 0.0, "binary_pass_rate": 0.0},
        }

    # Helper to compute stats for a group of questions
    def group_stats(questions):
        accuracies = [q["accuracy_proportional"] for q in questions]
        binaries = [q["accuracy_binary"] for q in questions]
        return {
            "mean_accuracy": round(statistics.mean(accuracies), 4) if accuracies else 0.0,
            "median_accuracy": round(statistics.median(accuracies), 4) if accuracies else 0.0,
            "binary_pass_rate": round(statistics.mean(binaries), 4) if binaries else 0.0,
            "count": len(questions),
        }

    # Overall
    all_accuracies = [q["accuracy_proportional"] for q in scored_questions]
    all_binaries = [q["accuracy_binary"] for q in scored_questions]
    total_hedging = sum(q["hedging_count"] for q in scored_questions)

    overall = {
        "mean_accuracy": round(statistics.mean(all_accuracies), 4),
        "median_accuracy": round(statistics.median(all_accuracies), 4),
        "binary_pass_rate": round(statistics.mean(all_binaries), 4),
        "total_hedging_instances": total_hedging,
    }

    # By category
    categories = {}
    for q in scored_questions:
        cat = q["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q)
    by_category = {cat: group_stats(qs) for cat, qs in sorted(categories.items())}

    # By difficulty
    difficulties = {}
    for q in scored_questions:
        diff = q["difficulty"]
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(q)
    by_difficulty = {diff: group_stats(qs) for diff, qs in sorted(difficulties.items())}

    # Trap questions
    trap_qs = [q for q in scored_questions if q["is_trap"]]
    if trap_qs:
        trap_accuracies = [q["accuracy_proportional"] for q in trap_qs]
        trap_binaries = [q["accuracy_binary"] for q in trap_qs]
        trap_questions = {
            "count": len(trap_qs),
            "mean_accuracy": round(statistics.mean(trap_accuracies), 4),
            "binary_pass_rate": round(statistics.mean(trap_binaries), 4),
        }
    else:
        trap_questions = {"count": 0, "mean_accuracy": 0.0, "binary_pass_rate": 0.0}

    return {
        "overall": overall,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "trap_questions": trap_questions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for response scoring."""
    args = parse_args()

    print("\n=== ALS-LM Response Scoring ===\n")
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

    # Load benchmark
    if not os.path.isfile(args.benchmark):
        print(f"ERROR: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    with open(args.benchmark) as f:
        benchmark = json.load(f)

    # Build lookup by question ID
    benchmark_by_id = {q["id"]: q for q in benchmark}
    print(f"  Loaded {len(benchmark)} benchmark questions from {args.benchmark}")
    print(f"  Fuzzy match threshold: {args.threshold}")

    # Score each response
    scored_questions = []
    skipped = 0
    total = len(responses)

    for i, resp in enumerate(responses):
        qid = resp["question_id"]

        # Progress output every 20 questions
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Scoring {i + 1}/{total}...")

        if qid not in benchmark_by_id:
            print(f"  WARNING: Question {qid} not found in benchmark, skipping")
            skipped += 1
            continue

        result = score_question(resp, benchmark_by_id[qid], args.threshold)
        scored_questions.append(result)

    print(f"  Scoring complete: {len(scored_questions)} scored, {skipped} skipped")

    # Compute aggregates
    aggregate = compute_aggregate(scored_questions)

    # Build output
    output = {
        "metadata": {
            "responses_path": args.responses,
            "benchmark_path": args.benchmark,
            "threshold": args.threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_questions": len(benchmark),
            "total_scored": len(scored_questions),
        },
        "aggregate": aggregate,
        "per_question": scored_questions,
    }

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    overall = aggregate["overall"]
    print(f"\n  === Scoring Summary ===")
    print(f"  Overall mean accuracy:   {overall['mean_accuracy']:.4f}")
    print(f"  Overall median accuracy: {overall['median_accuracy']:.4f}")
    print(f"  Binary pass rate:        {overall['binary_pass_rate']:.4f}")
    print(f"  Total hedging instances: {overall['total_hedging_instances']}")

    trap = aggregate["trap_questions"]
    if trap["count"] > 0:
        print(f"\n  Trap questions ({trap['count']}):")
        print(f"    Mean accuracy:  {trap['mean_accuracy']:.4f}")
        print(f"    Pass rate:      {trap['binary_pass_rate']:.4f}")

    print(f"\n  Output saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
