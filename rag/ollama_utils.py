"""Shared Ollama utilities for baseline and RAG evaluation scripts.

Provides pre-flight checks, chat response generation with retry logic,
response serialization, and eval stage orchestration. Both
generate_baseline.py and generate_rag.py import from this module to
avoid duplicating ~200 lines of identical logic.

This is a research evaluation tool, not a medical information system.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

# Default system prompt: minimal factual instruction without domain knowledge
# injection. Shared between baseline and RAG scripts so the ONLY variable
# between them is whether retrieved chunks are injected into the user message.
DEFAULT_SYSTEM_PROMPT = (
    "Answer the following question about ALS (amyotrophic lateral sclerosis) "
    "based on your knowledge. Be concise and factual."
)


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

    # Only match base_name:latest as a fallback (e.g. "llama3.1" matches
    # "llama3.1:latest" but NOT "llama3.1:70b")
    base_name = model_name.split(":")[0]
    if f"{base_name}:latest" in available_models:
        return

    print(f"ERROR: Model '{model_name}' not found in Ollama")
    print(f"  Available models: {', '.join(available_models) if available_models else '(none)'}")
    print(f"  Pull it with: ollama pull {model_name}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Response generation via /api/chat
# ---------------------------------------------------------------------------

def generate_chat_response(ollama_url, model_name, system_prompt, user_message,
                           max_tokens, timeout, num_ctx=None):
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
        num_ctx: Context window size for Ollama. If None, uses Ollama's
            default (2048). Set to a larger value (e.g. 8192) when injecting
            retrieved context to prevent silent truncation.

    Returns:
        (response_text, tokens_generated) on success, or
        ("[chat error: <details>]", 0) on permanent failure.
    """
    url = f"{ollama_url.rstrip('/')}/api/chat"
    options = {
        "temperature": 0.0,
        "num_predict": max_tokens,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": options,
    }

    max_retries = 3
    resp = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            response_text = data.get("message", {}).get("content", "")
            tokens_generated = data.get("eval_count", 0)
            if tokens_generated == 0 and response_text:
                # Fallback: estimate from word count with BPE multiplier
                tokens_generated = int(len(response_text.split()) * 1.3)
            return response_text, tokens_generated
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries} after error: {exc}")
                time.sleep(5)
            else:
                return f"[chat error: {exc}]", 0
        except requests.HTTPError as exc:
            if resp is not None and resp.status_code >= 500 and attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries} after HTTP {resp.status_code}")
                time.sleep(5)
            else:
                return f"[chat error: HTTP {resp.status_code} - {exc}]", 0
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
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


def run_eval_stages(output_dir, benchmark_path, registry_path,
                    project_root, report_suffix="baseline"):
    """Invoke evaluation stages 2-6 via subprocess.

    Each stage is a separate Python script invoked with explicit file path
    arguments. This avoids argparse/sys.argv conflicts from direct imports
    and provides subprocess isolation for memory management.

    Args:
        output_dir: Directory containing responses.json and for all output.
        benchmark_path: Absolute path to benchmark questions JSON.
        registry_path: Absolute path to entity registry JSON.
        project_root: Project root path for resolving eval scripts.
        report_suffix: Suffix for the report filename
            (e.g. "baseline" -> hallucination_eval_baseline.md).

    Returns:
        True if all stages succeeded, False otherwise.
    """
    paths = {
        "responses": os.path.join(output_dir, "responses.json"),
        "benchmark": benchmark_path,
        "registry": registry_path,
        "scores": os.path.join(output_dir, "scores.json"),
        "fabrications": os.path.join(output_dir, "fabrications.json"),
        "taxonomy": os.path.join(output_dir, "taxonomy.json"),
        "samples": os.path.join(output_dir, "samples.json"),
        "report": os.path.join(output_dir, f"hallucination_eval_{report_suffix}.md"),
    }

    total = len(EVAL_STAGES)
    pipeline_start = time.time()

    for i, stage in enumerate(EVAL_STAGES, 1):
        stage_name = stage["name"]
        script_path = str(Path(project_root) / stage["script"])
        args = stage["build_args"](paths)

        print(f"  [{i}/{total}] {stage['display']}...", end="", flush=True)
        t0 = time.time()

        try:
            result = subprocess.run(
                [sys.executable, script_path] + args,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            print(f" TIMED OUT (>1h)")
            print(f"\n  ERROR: Stage '{stage_name}' exceeded 1-hour timeout")
            return False

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
