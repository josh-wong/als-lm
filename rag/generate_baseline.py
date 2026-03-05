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
import sys
import time
from datetime import datetime, timezone

# Ensure the project root is on sys.path so that imports from eval/ resolve
# correctly when running as `python rag/generate_baseline.py`.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eval.generate_responses import is_coherent
from eval.utils import find_project_root, resolve_default_paths, relativize_path
from rag.ollama_utils import (
    DEFAULT_SYSTEM_PROMPT,
    check_ollama_running,
    check_model_available,
    generate_chat_response,
    save_responses,
    run_eval_stages,
)

# Project root and default paths
PROJECT_ROOT = find_project_root()
DEFAULTS = resolve_default_paths(PROJECT_ROOT)

# Re-export for backwards compatibility with --system-prompt default
# The canonical definition lives in rag.ollama_utils.


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
                                          registry_path, PROJECT_ROOT)
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
    success = run_eval_stages(output_dir, benchmark_path, registry_path, PROJECT_ROOT)
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
