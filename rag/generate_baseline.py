#!/usr/bin/env python3
"""Generate no-retrieval baseline responses and run the evaluation pipeline.

Queries Llama 3.1 8B (or another Ollama model) via /api/chat with only the
system prompt and benchmark question -- no retrieved context. This establishes
what the model knows from parametric knowledge alone. The evaluation pipeline
(stages 2-6) runs automatically after generation, producing scores,
fabrication detection, taxonomy classification, qualitative samples, and a
Markdown report in a single output directory.

The baseline results serve as the attribution boundary for RAG evaluation:
anything the RAG pipeline (Phase 16) gets right that this baseline gets wrong
is directly attributable to retrieval augmentation.

This is a research evaluation tool, not a medical information system.

Usage examples::

    # Run full baseline with defaults
    python rag/generate_baseline.py

    # Specify a different Ollama model tag
    python rag/generate_baseline.py --ollama-model llama3.1:latest

    # Generate responses only (skip eval stages 2-6)
    python rag/generate_baseline.py --skip-eval

    # Resume an interrupted run
    python rag/generate_baseline.py --resume

    # Custom system prompt for experimentation
    python rag/generate_baseline.py --system-prompt "You are an ALS expert."
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on sys.path so that imports from eval/ resolve
# correctly when running as `python rag/generate_baseline.py`.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eval.generate_responses import is_coherent
from eval.utils import find_project_root, resolve_default_paths, relativize_path

import requests

# Project root and default paths
PROJECT_ROOT = find_project_root()
DEFAULTS = resolve_default_paths(PROJECT_ROOT)

# Default system prompt: minimal factual instruction without domain knowledge
# injection. This exact prompt will be reused in the Phase 16 RAG pipeline so
# the ONLY variable between baseline and RAG is whether retrieved chunks are
# injected into the user message.
DEFAULT_SYSTEM_PROMPT = (
    "Answer the following question about ALS (amyotrophic lateral sclerosis) "
    "based on your knowledge. Be concise and factual."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for baseline generation and evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate no-retrieval baseline responses from an Ollama model "
            "and run the full evaluation pipeline (stages 2-6)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python rag/generate_baseline.py\n"
            "  python rag/generate_baseline.py --ollama-model llama3.1:latest\n"
            "  python rag/generate_baseline.py --resume --skip-eval\n"
        ),
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model tag (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help=(
            "System prompt for the chat API. Default is a minimal factual "
            "instruction. Override for prompt template experimentation (RAG-04)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response (default: 512)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300, conservative for cold-start)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULTS["benchmark"]),
        help=f"Path to benchmark questions JSON (default: {DEFAULTS['benchmark']})",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=str(DEFAULTS["registry"]),
        help=f"Path to entity registry JSON (default: {DEFAULTS['registry']})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "rag" / "results" / "baseline"),
        help="Output directory for all artifacts (default: rag/results/baseline)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed questions (resume interrupted run)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Generate responses only, skip evaluation stages 2-6",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate responses even if responses.json already exists",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_ollama_running(ollama_url, timeout=10):
    """Verify the Ollama server is reachable.

    Args:
        ollama_url: Base URL of the Ollama server.
        timeout: Connection timeout in seconds.

    Returns:
        List of available model names, or exits on failure.
    """
    tags_url = f"{ollama_url.rstrip('/')}/api/tags"
    try:
        resp = requests.get(tags_url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return models
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to Ollama at {ollama_url}")
        print("  Is Ollama running? Start it with: ollama serve")
        sys.exit(1)
    except requests.Timeout:
        print(f"ERROR: Ollama server at {ollama_url} timed out")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Unexpected error checking Ollama: {exc}")
        sys.exit(1)


def check_model_available(ollama_url, model_name, available_models):
    """Verify the requested model is pulled in Ollama.

    Args:
        ollama_url: Base URL of the Ollama server.
        model_name: Requested model tag.
        available_models: List of available model names from /api/tags.
    """
    if model_name in available_models:
        return

    # Check without tag suffix (e.g., "llama3.1:8b" matches "llama3.1:8b")
    # Also try matching base name
    base_name = model_name.split(":")[0]
    for m in available_models:
        if m == model_name or m.startswith(f"{base_name}:"):
            return

    print(f"ERROR: Model '{model_name}' not found in Ollama")
    print(f"  Available models: {', '.join(available_models) if available_models else '(none)'}")
    print(f"  Pull it with: ollama pull {model_name}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Response generation via /api/chat
# ---------------------------------------------------------------------------

def generate_chat_response(ollama_url, model_name, system_prompt, user_message,
                           max_tokens, timeout):
    """Generate a single response via the Ollama /api/chat endpoint.

    Uses the chat completions API (not /api/generate) so the model's native
    instruction template is applied. Retries up to 3 times on connection
    errors, timeouts, or 5xx HTTP errors with a 5-second delay.

    Args:
        ollama_url: Base URL of the Ollama server.
        model_name: Ollama model tag.
        system_prompt: System message content.
        user_message: User message content (the benchmark question).
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.

    Returns:
        (response_text, tokens_generated) on success, or
        ("[chat error: <details>]", 0) on permanent failure.
    """
    url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
        },
    }

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("message", {}).get("content", "")
            tokens_generated = data.get("eval_count", 0)
            if tokens_generated == 0 and response_text:
                # Fallback: estimate from word count
                tokens_generated = len(response_text.split())
            return response_text, tokens_generated
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries} after error: {exc}")
                time.sleep(5)
            else:
                return f"[chat error: {exc}]", 0
        except requests.HTTPError as exc:
            if resp.status_code >= 500 and attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries} after HTTP {resp.status_code}")
                time.sleep(5)
            else:
                return f"[chat error: HTTP {resp.status_code} - {exc}]", 0
        except Exception as exc:
            return f"[chat error: {exc}]", 0

    return "[chat error: max retries exceeded]", 0


# ---------------------------------------------------------------------------
# Incremental save
# ---------------------------------------------------------------------------

def save_responses(output_path, responses, metadata):
    """Write responses and metadata to a JSON file.

    Args:
        output_path: Path to the output JSON file.
        responses: List of response dicts.
        metadata: Metadata dict.
    """
    output = {
        "metadata": metadata,
        "responses": responses,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# Eval stage orchestration
# ---------------------------------------------------------------------------

EVAL_STAGES = [
    {
        "name": "score",
        "display": "Scoring responses",
        "script": "eval/score_responses.py",
        "build_args": lambda paths: [
            "--responses", paths["responses"],
            "--benchmark", paths["benchmark"],
            "--output", paths["scores"],
        ],
    },
    {
        "name": "fabrications",
        "display": "Detecting fabrications",
        "script": "eval/detect_fabrications.py",
        "build_args": lambda paths: [
            "--responses", paths["responses"],
            "--registry", paths["registry"],
            "--output", paths["fabrications"],
        ],
    },
    {
        "name": "taxonomy",
        "display": "Classifying taxonomy",
        "script": "eval/classify_taxonomy.py",
        "build_args": lambda paths: [
            "--scores", paths["scores"],
            "--fabrications", paths["fabrications"],
            "--responses", paths["responses"],
            "--benchmark", paths["benchmark"],
            "--output", paths["taxonomy"],
        ],
    },
    {
        "name": "samples",
        "display": "Curating samples",
        "script": "eval/curate_samples.py",
        "build_args": lambda paths: [
            "--scores", paths["scores"],
            "--fabrications", paths["fabrications"],
            "--responses", paths["responses"],
            "--benchmark", paths["benchmark"],
            "--taxonomy", paths["taxonomy"],
            "--output", paths["samples"],
        ],
    },
    {
        "name": "report",
        "display": "Generating report",
        "script": "eval/generate_report.py",
        "build_args": lambda paths: [
            "--scores", paths["scores"],
            "--fabrications", paths["fabrications"],
            "--taxonomy", paths["taxonomy"],
            "--samples", paths["samples"],
            "--responses", paths["responses"],
            "--output", paths["report"],
        ],
    },
]


def run_eval_stages(output_dir, benchmark_path, registry_path):
    """Invoke evaluation stages 2-6 via subprocess.

    Each stage is a separate Python script invoked with explicit file path
    arguments. This avoids argparse/sys.argv conflicts from direct imports
    and provides subprocess isolation for memory management.

    Args:
        output_dir: Directory containing responses.json and for all output.
        benchmark_path: Absolute path to benchmark questions JSON.
        registry_path: Absolute path to entity registry JSON.

    Returns:
        True if all stages succeeded, False otherwise.
    """
    # Build file path map for all stages
    paths = {
        "responses": os.path.join(output_dir, "responses.json"),
        "benchmark": benchmark_path,
        "registry": registry_path,
        "scores": os.path.join(output_dir, "scores.json"),
        "fabrications": os.path.join(output_dir, "fabrications.json"),
        "taxonomy": os.path.join(output_dir, "taxonomy.json"),
        "samples": os.path.join(output_dir, "samples.json"),
        "report": os.path.join(output_dir, "hallucination_eval_baseline.md"),
    }

    total = len(EVAL_STAGES)
    pipeline_start = time.time()

    for i, stage in enumerate(EVAL_STAGES, 1):
        stage_name = stage["name"]
        script_path = str(PROJECT_ROOT / stage["script"])
        args = stage["build_args"](paths)

        print(f"  [{i}/{total}] {stage['display']}...", end="", flush=True)
        t0 = time.time()

        result = subprocess.run(
            [sys.executable, script_path] + args,
            capture_output=True,
            text=True,
        )

        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f" FAILED ({elapsed:.1f}s)")
            print(f"\n  ERROR in stage '{stage_name}':")
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    print(f"    {line}")
            else:
                print("    (no stderr output)")
            if result.stdout:
                print(f"\n  Stage stdout (last 10 lines):")
                for line in result.stdout.strip().split("\n")[-10:]:
                    print(f"    {line}")
            return False

        print(f" done ({elapsed:.1f}s)")

    pipeline_elapsed = time.time() - pipeline_start
    print(f"\n  Evaluation pipeline complete ({pipeline_elapsed:.1f}s total)")
    print(f"  Report: {paths['report']}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for baseline generation and evaluation."""
    args = parse_args()

    print("\n=== ALS Baseline Evaluation (No Retrieval) ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical"
          " information system.\n")

    # Resolve paths
    output_dir = os.path.abspath(args.output_dir)
    responses_path = os.path.join(output_dir, "responses.json")
    benchmark_path = os.path.abspath(args.benchmark)
    registry_path = os.path.abspath(args.registry)

    # --- Pre-flight checks ---------------------------------------------------

    # 1. Verify Ollama is running and model is available
    print("  Pre-flight checks:")
    available_models = check_ollama_running(args.ollama_url)
    print(f"    Ollama server: OK ({len(available_models)} models available)")

    check_model_available(args.ollama_url, args.ollama_model, available_models)
    print(f"    Model '{args.ollama_model}': OK")

    # 2. Verify benchmark file exists
    if not os.path.isfile(benchmark_path):
        print(f"\n  ERROR: Benchmark file not found: {benchmark_path}")
        sys.exit(1)
    print(f"    Benchmark: {benchmark_path}")

    # 3. Verify registry file exists
    if not os.path.isfile(registry_path):
        print(f"\n  ERROR: Entity registry not found: {registry_path}")
        print("  Run eval/build_entity_registry.py first.")
        sys.exit(1)
    print(f"    Registry: {registry_path}")

    # 4. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"    Output dir: {output_dir}")

    # --- Check for existing responses ----------------------------------------

    existing_responses = {}
    if os.path.isfile(responses_path) and not args.force:
        if not args.resume:
            print(f"\n  Responses already exist at {responses_path}")
            print("  Use --resume to continue, --force to regenerate, "
                  "or --skip-eval to skip generation.")
            # If skip-eval is not set, jump straight to eval
            if not args.skip_eval:
                print("\n  Skipping generation, running evaluation stages...\n")
                success = run_eval_stages(output_dir, benchmark_path,
                                          registry_path)
                if not success:
                    sys.exit(1)
                return
            return

    # --- Load and resume -----------------------------------------------------

    if args.resume and os.path.isfile(responses_path):
        try:
            with open(responses_path) as f:
                data = json.load(f)
            existing_responses = {
                r["question_id"]: r for r in data.get("responses", [])
            }
            print(f"\n  Resume: {len(existing_responses)} existing responses loaded")
        except (json.JSONDecodeError, KeyError, OSError):
            print("  WARNING: Could not parse existing responses, starting fresh")
            existing_responses = {}

    # --- Load benchmark questions --------------------------------------------

    with open(benchmark_path) as f:
        questions = json.load(f)
    total_questions = len(questions)
    print(f"\n  Benchmark: {total_questions} questions loaded")

    # Filter for resume
    if existing_responses:
        questions_to_run = [
            q for q in questions if q["id"] not in existing_responses
        ]
        print(f"  Skipping {total_questions - len(questions_to_run)} "
              f"already-completed questions")
    else:
        questions_to_run = questions

    if not questions_to_run:
        print("  All questions already completed.")
    else:
        # --- Generation settings -------------------------------------------------

        print(f"\n  Model: {args.ollama_model}")
        print(f"  System prompt: {args.system_prompt[:80]}...")
        print(f"  Max tokens: {args.max_tokens}")
        print(f"  Temperature: 0.0 (greedy decoding)")
        print(f"  Timeout: {args.timeout}s per request")
        print(f"  Questions to generate: {len(questions_to_run)}")
        print()

        # --- Generation loop -----------------------------------------------------

        responses = list(existing_responses.values())
        generation_start = time.time()

        for i, question in enumerate(questions_to_run):
            qid = question["id"]
            category = question["category"]
            prompt_text = question["prompt_template"]

            response_text, tokens_generated = generate_chat_response(
                args.ollama_url,
                args.ollama_model,
                args.system_prompt,
                prompt_text,
                args.max_tokens,
                args.timeout,
            )

            entry = {
                "question_id": qid,
                "category": category,
                "difficulty": question["difficulty"],
                "is_trap": question["is_trap"],
                "prompt": prompt_text,
                "response": response_text,
                "tokens_generated": tokens_generated,
                "is_coherent": is_coherent(response_text),
            }
            responses.append(entry)

            print(f"  [{i + 1}/{len(questions_to_run)}] {qid} ({category}) "
                  f"... {tokens_generated} tokens")

            # Incremental save every 10 questions for crash recovery
            if (i + 1) % 10 == 0:
                metadata = _build_metadata(args, total_questions, len(responses))
                save_responses(responses_path, responses, metadata)

        generation_elapsed = time.time() - generation_start

        # --- Final save ----------------------------------------------------------

        metadata = _build_metadata(args, total_questions, len(responses))
        save_responses(responses_path, responses, metadata)

        # --- Generation summary --------------------------------------------------

        total_tokens = sum(r["tokens_generated"] for r in responses)
        error_count = sum(
            1 for r in responses
            if r["response"].startswith("[chat error:")
        )
        incoherent_count = sum(
            1 for r in responses if not r.get("is_coherent", True)
        )

        print(f"\n  === Generation Complete ===")
        if questions_to_run:
            print(f"  Time: {generation_elapsed:.1f}s "
                  f"({generation_elapsed / len(questions_to_run):.2f}s per question)")
        print(f"  Total responses: {len(responses)}")
        print(f"  Total tokens: {total_tokens:,}")
        if responses:
            print(f"  Average tokens/response: "
                  f"{total_tokens / len(responses):.1f}")
        print(f"  Errors: {error_count}")
        print(f"  Incoherent: {incoherent_count}")
        print(f"  Saved to: {responses_path}")

    # --- Run evaluation stages -----------------------------------------------

    if args.skip_eval:
        print("\n  --skip-eval: Skipping evaluation stages 2-6")
        return

    print("\n  Running evaluation stages 2-6...\n")
    success = run_eval_stages(output_dir, benchmark_path, registry_path)
    if not success:
        print("\n  ERROR: Evaluation pipeline failed")
        sys.exit(1)

    print("\n  === Baseline Evaluation Complete ===")
    print(f"  Results directory: {output_dir}")
    print()


def _build_metadata(args, total_questions, completed_questions):
    """Build the metadata block for responses.json.

    Args:
        args: Parsed command-line arguments.
        total_questions: Total number of benchmark questions.
        completed_questions: Number of completed responses.

    Returns:
        Metadata dict for JSON serialization.
    """
    return {
        "inference_mode": "ollama",
        "ollama_model": args.ollama_model,
        "ollama_url": args.ollama_url,
        "system_prompt": args.system_prompt,
        "generation_params": {
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
        },
        "benchmark_path": relativize_path(str(args.benchmark)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": total_questions,
        "completed_questions": completed_questions,
        "eval_type": "baseline_no_retrieval",
    }


if __name__ == "__main__":
    main()
