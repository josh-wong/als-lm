#!/usr/bin/env python3
"""Qualitative spot-check of SFT model via Ollama API.

Sends 8 ALS questions (one per knowledge category) in Alpaca instruction
format to the SFT model served through Ollama, checks each response for
coherence (non-empty, >20 chars, no repetition, no token salad), and reports
an overall PASS/FAIL verdict.

Pass threshold: 7/8+ coherent responses (per SFT verification spec).

This is a research verification tool, not a medical information system.

Usage:
    python scripts/verify_sft.py --model als-lm-1b-sft:f16
    python scripts/verify_sft.py --model als-lm-1b-sft:f16 --ollama-url http://localhost:11434
    python scripts/verify_sft.py --model als-lm-1b-sft:f16 --output results/sft/verify_results.json
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Project root discovery (standalone script pattern)
# ---------------------------------------------------------------------------

_project_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Default paths relative to project root
_DEFAULT_QUESTIONS = str(_project_root / "eval" / "questions.json")
_DEFAULT_OUTPUT = str(_project_root / "results" / "sft" / "verify_results.json")

# The 8 ALS knowledge categories in the benchmark
_CATEGORIES = [
    "clinical_trials",
    "diagnostic_criteria",
    "disease_mechanisms",
    "drug_treatment",
    "epidemiology",
    "gene_mutation",
    "patient_care",
    "temporal_accuracy",
]

# Coherence pass threshold (ratio of coherent responses)
_PASS_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def wrap_alpaca(question: str) -> str:
    """Wrap a question in Alpaca instruction format for the SFT model.

    Returns a string with the Alpaca template ready to send to Ollama.
    The model's Modelfile TEMPLATE should match this format.
    """
    return f"### Instruction:\n{question}\n\n### Response:\n"


def select_spot_check_questions(
    questions_path: str = _DEFAULT_QUESTIONS,
    seed: int = 42,
) -> list[dict]:
    """Select one question per ALS category for spot-checking.

    Loads the benchmark questions and picks one from each of the 8
    categories using a fixed random seed for deterministic selection.

    Args:
        questions_path: Path to eval/questions.json.
        seed: Random seed for reproducible selection.

    Returns:
        List of 8 question dicts (one per category), sorted by category.
    """
    with open(questions_path) as f:
        questions = json.load(f)

    rng = random.Random(seed)
    selected = []
    for category in sorted(set(q["category"] for q in questions)):
        category_questions = [q for q in questions if q["category"] == category]
        selected.append(rng.choice(category_questions))

    return selected


def check_coherence(text) -> bool:
    """Binary coherence check for model output.

    Uses the same heuristics as eval/generate_responses.py::is_coherent()
    but with the SFT-specific threshold of >20 characters (instead of >10).

    A response is incoherent if any of these conditions hold:

    - None, empty, or whitespace-only
    - Shorter than 21 characters
    - Contains a word repeated 6+ times consecutively
    - Contains punctuation-separated token repetition
    - Contains any 3-gram repeated 4+ times
    - Contains concatenated substring repeated 5+ times (e.g. "TheTheThe")
    - More than 80% non-alphanumeric characters (token salad)

    Args:
        text: The generated response text.

    Returns:
        True if the text appears coherent, False otherwise.
    """
    if text is None or not text or not text.strip():
        return False

    stripped = text.strip()

    # SFT threshold: >20 chars (stricter than eval's >10)
    if len(stripped) <= 20:
        return False

    # Consecutive word repetition (same word 6+ times in a row)
    if re.search(r"(\b\w+\b)(\s+\1){5,}", stripped, re.IGNORECASE):
        return False

    # Punctuation-separated token repetition
    cleaned = re.sub(r"[,;\-]+", " ", stripped)
    cleaned = re.sub(r"\s+", " ", cleaned)
    if re.search(r"(\b\w+\b)(\s+\1){5,}", cleaned, re.IGNORECASE):
        return False

    # 3-gram repetition (any trigram appearing 4+ times)
    words = stripped.split()
    if len(words) >= 6:
        trigram_counts: dict[str, int] = {}
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i + 3]).lower()
            trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1
            if trigram_counts[trigram] >= 4:
                return False

    # Concatenated token repetition (e.g. "TheTheTheThe..." without spaces)
    # Detects short non-whitespace substrings (2-10 chars) repeated 5+ times
    if re.search(r"(\S{2,10})\1{4,}", stripped):
        return False

    # Token salad (>80% non-alphanumeric)
    alnum_count = sum(1 for ch in stripped if ch.isalnum())
    if len(stripped) > 0 and alnum_count / len(stripped) < 0.2:
        return False

    return True


def format_verdict(coherent_count: int, total_count: int) -> str:
    """Determine overall PASS/FAIL verdict.

    The threshold is 70% coherent responses (7/10 scaled to any total).

    Args:
        coherent_count: Number of coherent responses.
        total_count: Total number of responses.

    Returns:
        "PASS" if coherent_count/total_count >= 0.7, else "FAIL".
    """
    if total_count == 0:
        return "FAIL"
    ratio = coherent_count / total_count
    return "PASS" if ratio >= _PASS_THRESHOLD else "FAIL"


def format_results(results: list[dict]) -> list[dict]:
    """Format results for display, truncating long responses.

    Each result dict gets a response_preview field with the first 100
    characters of the response (or the full response if shorter).

    Args:
        results: List of dicts with question, category, response, coherent.

    Returns:
        List of dicts with question, category, response_preview, coherent.
    """
    formatted = []
    for r in results:
        response = r["response"]
        if len(response) > 100:
            preview = response[:100] + "..."
        else:
            preview = response
        formatted.append({
            "question": r["question"],
            "category": r["category"],
            "response_preview": preview,
            "coherent": r["coherent"],
        })
    return formatted


def generate_response(
    ollama_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Generate a single response via the Ollama HTTP API.

    Posts to /api/generate with streaming disabled. Retries up to 3 times
    on connection errors, timeouts, or server errors (HTTP 5xx) with a
    5-second delay between attempts.

    Args:
        ollama_url: Base URL of the Ollama server.
        model_name: Ollama model tag.
        prompt: The prompt text to send.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).

    Returns:
        The response text, or an error placeholder string.
    """
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries} after error: {exc}")
                time.sleep(5)
            else:
                return f"[ollama error: {exc}]"
        except requests.HTTPError as exc:
            if resp.status_code >= 500 and attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries} after HTTP {resp.status_code}")
                time.sleep(5)
            else:
                return f"[ollama error: HTTP {resp.status_code} - {exc}]"
        except Exception as exc:
            return f"[ollama error: {exc}]"

    return "[ollama error: max retries exceeded]"


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qualitative spot-check of SFT model via Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/verify_sft.py --model als-lm-1b-sft:f16\n"
            "  python scripts/verify_sft.py --model als-lm-1b-sft:f16 "
            "--ollama-url http://localhost:11434\n"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name (e.g., 'als-lm-1b-sft:f16')",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {_DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main():
    """Run the SFT qualitative spot-check."""
    args = parse_args()

    print("\n=== SFT Qualitative Verification ===\n")
    print("  NOTE: This is a research verification tool, not a medical system.\n")
    print(f"  Model: {args.model}")
    print(f"  Ollama URL: {args.ollama_url}")

    # Load and select questions
    questions = select_spot_check_questions(_DEFAULT_QUESTIONS)
    print(f"  Questions selected: {len(questions)} (one per category)\n")

    # Run spot-check
    results = []
    for i, q in enumerate(questions):
        prompt = wrap_alpaca(q["question"])
        print(f"  [{i + 1}/{len(questions)}] {q['category']}: {q['question'][:60]}...")

        response = generate_response(args.ollama_url, args.model, prompt)
        coherent = check_coherence(response)
        status = "PASS" if coherent else "FAIL"

        preview = response[:100] + "..." if len(response) > 100 else response
        print(f"    Response: {preview}")
        print(f"    Coherent: {status}\n")

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "prompt": prompt,
            "response": response,
            "coherent": coherent,
        })

    # Compute verdict
    coherent_count = sum(1 for r in results if r["coherent"])
    total = len(results)
    verdict = format_verdict(coherent_count, total)

    print(f"  === Verdict: {verdict} ({coherent_count}/{total} coherent) ===\n")

    # Save results
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "metadata": {
            "model": args.model,
            "ollama_url": args.ollama_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "questions_path": _DEFAULT_QUESTIONS,
            "seed": 42,
        },
        "results": results,
        "summary": {
            "coherent_count": coherent_count,
            "total_count": total,
            "verdict": verdict,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"  Results saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
