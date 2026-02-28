#!/usr/bin/env python3
"""Generate model responses to ALS benchmark questions for evaluation.

Supports two inference modes: (1) loading a PyTorch checkpoint directly, or
(2) querying an Ollama server hosting GGUF models. Includes a coherence
pre-filter to flag degenerate output (empty, repetitive, token salad) before
expensive downstream scoring stages, and resume support to pick up from the
last completed question if a run is interrupted.

This is a research evaluation tool, not a medical information system.

Usage examples::

    # Using a checkpoint directory (contains best.pt)
    python eval/generate_responses.py --checkpoint checkpoints/tiny_20260225/best

    # Using a direct .pt file
    python eval/generate_responses.py --checkpoint checkpoints/tiny_20260225/best/best.pt

    # Using an Ollama model
    python eval/generate_responses.py --ollama-model als-lm-500m:q8_0

    # Ollama with custom URL and resume support
    python eval/generate_responses.py \\
        --ollama-model als-lm-500m:f16 \\
        --ollama-url http://localhost:11434 \\
        --resume

    # Custom output path and max tokens
    python eval/generate_responses.py \\
        --checkpoint checkpoints/tiny_20260225/best \\
        --output eval/results/tiny_responses.json \\
        --max-tokens 256

    # Force CPU inference
    python eval/generate_responses.py \\
        --checkpoint checkpoints/tiny_20260225/best \\
        --device cpu
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# Ensure the project root is on sys.path so that `from model.model import ...`
# resolves correctly when running as `python eval/generate_responses.py`.
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

# Checkpoint-mode imports are deferred until needed so that Ollama-only runs
# do not require PyTorch, tokenizers, or the model package to be installed.
_CHECKPOINT_IMPORTS_LOADED = False


def _ensure_checkpoint_imports():
    """Lazily import PyTorch, tokenizers, and the GPT model."""
    global _CHECKPOINT_IMPORTS_LOADED
    if _CHECKPOINT_IMPORTS_LOADED:
        return
    global torch, Tokenizer, GPT, GPTConfig
    import torch as _torch
    from tokenizers import Tokenizer as _Tokenizer
    from model.model import GPT as _GPT, GPTConfig as _GPTConfig
    torch = _torch
    Tokenizer = _Tokenizer
    GPT = _GPT
    GPTConfig = _GPTConfig
    _CHECKPOINT_IMPORTS_LOADED = True


def parse_args():
    """Parse command-line arguments for response generation."""
    parser = argparse.ArgumentParser(
        description="Generate model responses to ALS benchmark questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/generate_responses.py --checkpoint checkpoints/tiny/best\n"
            "  python eval/generate_responses.py --ollama-model als-lm-500m:q8_0\n"
            "  python eval/generate_responses.py --ollama-model als-lm-500m:f16 --resume\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (containing best.pt) or direct .pt file",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Ollama model name (e.g., 'als-lm-500m:q8_0'). Uses Ollama API instead of checkpoint.",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0 for deterministic results)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed question if output file exists",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(_DEFAULTS["benchmark"]) if "benchmark" in _DEFAULTS else "eval/questions.json",
        help="Path to benchmark questions file (default: eval/questions.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results/responses.json",
        help="Path for output file (default: eval/results/responses.json)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference: cuda or cpu (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate: exactly one of --checkpoint or --ollama-model must be provided
    if args.checkpoint and args.ollama_model:
        parser.error("Specify either --checkpoint or --ollama-model, not both.")
    if not args.checkpoint and not args.ollama_model:
        parser.error("One of --checkpoint or --ollama-model is required.")

    return args


def resolve_checkpoint_path(checkpoint_arg):
    """Resolve the checkpoint argument to an actual .pt file path.

    Handles two formats:
    1. A directory containing best.pt (from save_best_checkpoint_atomic)
    2. A direct .pt file path

    Returns the resolved path or exits with an error message.
    """
    if os.path.isdir(checkpoint_arg):
        pt_path = os.path.join(checkpoint_arg, "best.pt")
        if os.path.isfile(pt_path):
            return pt_path
        # Try to find any .pt file in the directory
        pt_files = [f for f in os.listdir(checkpoint_arg) if f.endswith(".pt")]
        if pt_files:
            return os.path.join(checkpoint_arg, pt_files[0])
        print(f"ERROR: No .pt file found in checkpoint directory: {checkpoint_arg}")
        print("  Expected best.pt or another .pt file in the directory.")
        sys.exit(1)
    elif os.path.isfile(checkpoint_arg):
        return checkpoint_arg
    else:
        print(f"ERROR: Checkpoint not found: {checkpoint_arg}")
        print("  Provide a directory containing best.pt or a direct .pt file path.")
        sys.exit(1)


def load_model_from_checkpoint(pt_path, device):
    """Load a GPT model from a .pt checkpoint file.

    The checkpoint contains:
        {"model": state_dict, "config": model_config_dict, "step": ..., "val_loss": ...}

    Reconstructs GPTConfig from the checkpoint config dict, instantiates the
    model, loads the state dict, and sets eval mode with dropout disabled.

    Returns (model, model_config_dict).
    """
    _ensure_checkpoint_imports()
    print(f"  Loading checkpoint: {pt_path}")
    checkpoint = torch.load(pt_path, map_location=device, weights_only=False)

    config_dict = checkpoint["config"]
    print(f"  Model config: n_layer={config_dict['n_layer']}, "
          f"n_embd={config_dict['n_embd']}, "
          f"n_head={config_dict['n_head']}, "
          f"vocab_size={config_dict['vocab_size']}")

    # Reconstruct GPTConfig, forcing dropout to 0 for inference
    config = GPTConfig(
        block_size=config_dict["block_size"],
        vocab_size=config_dict["vocab_size"],
        n_layer=config_dict["n_layer"],
        n_head=config_dict["n_head"],
        n_embd=config_dict["n_embd"],
        dropout=0.0,
        bias=config_dict["bias"],
    )

    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    step = checkpoint.get("step", "unknown")
    val_loss = checkpoint.get("val_loss", "unknown")
    print(f"  Checkpoint step: {step}, val_loss: {val_loss}")
    print(f"  Parameters: {model.get_num_params():,}")

    return model, config_dict


def load_tokenizer(tokenizer_path=None):
    """Load the canonical ALS tokenizer.

    If no path is provided, resolves the default location using
    ``_PROJECT_ROOT`` so the script works from any working directory.

    Returns the Tokenizer instance or exits with an error message.
    """
    _ensure_checkpoint_imports()
    if tokenizer_path is None:
        if _PROJECT_ROOT is not None:
            tokenizer_path = str(_PROJECT_ROOT / "tokenizer" / "als_tokenizer.json")
        else:
            tokenizer_path = "tokenizer/als_tokenizer.json"

    if not os.path.isfile(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("  Have you trained the tokenizer? The canonical tokenizer must "
              "exist at tokenizer/als_tokenizer.json")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"  Tokenizer loaded: {tokenizer.get_vocab_size():,} tokens")
    return tokenizer


def generate_greedy(model, input_ids, max_new_tokens, eos_token_id, device):
    """Generate tokens using greedy decoding (temperature=0).

    Replicates the generation pattern from model/train.py's generate_sample()
    with these differences:
    - Always greedy (argmax of logits)
    - Stops on <|endoftext|> token
    - Returns only the new tokens (excluding prompt)
    - Tracks token count

    Args:
        model: GPT model in eval mode.
        input_ids: List of integer token IDs for the prompt.
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id: Token ID for <|endoftext|> (stop token), or None.
        device: Torch device.

    Returns:
        (new_token_ids, tokens_generated) where new_token_ids is a list of
        generated token IDs (excluding prompt) and tokens_generated is the count.
    """
    _ensure_checkpoint_imports()
    with torch.no_grad():
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        block_size = model.config.block_size
        new_tokens = []

        for _ in range(max_new_tokens):
            # Truncate to block_size if sequence exceeds context window
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            # Take logits at the last position
            logits = logits[:, -1, :]
            # Greedy decoding: argmax
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = idx_next.item()

            # Stop on <|endoftext|>
            if eos_token_id is not None and token_id == eos_token_id:
                break

            new_tokens.append(token_id)
            idx = torch.cat([idx, idx_next], dim=1)

    return new_tokens, len(new_tokens)


def is_coherent(text):
    """Binary coherence pre-filter for model output.

    Flags degenerate responses that should be classified as failures without
    expensive downstream scoring. A response is incoherent if any of these
    conditions hold:

    - Empty or whitespace-only
    - Shorter than 10 characters
    - Contains a word repeated more than 5 times consecutively
    - Contains punctuation-separated token repetition (e.g. "tau, tau, tau")
    - Contains any 3-gram repeated 4+ times
    - More than 80% non-alphanumeric characters (token salad)

    Args:
        text: The generated response text.

    Returns:
        True if the text appears coherent, False otherwise.
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()

    if len(stripped) < 10:
        return False

    # Check for consecutive word repetition (same word 6+ times in a row)
    if re.search(r"(\b\w+\b)(\s+\1){5,}", stripped, re.IGNORECASE):
        return False

    # Check for punctuation-separated token repetition. Strip commas,
    # semicolons, and hyphens then re-check for consecutive words. This
    # catches patterns like "TDP-43, TDP-43, TDP-43" and "tau, tau, tau".
    cleaned = re.sub(r"[,;\-]+", " ", stripped)
    cleaned = re.sub(r"\s+", " ", cleaned)
    if re.search(r"(\b\w+\b)(\s+\1){5,}", cleaned, re.IGNORECASE):
        return False

    # Check for n-gram repetition. Build a frequency table of 3-word
    # sliding windows. If any 3-gram appears 4+ times, the output is
    # degenerate. This catches phrase-level loops like "protein-binding
    # protein-binding" and "and FTD, and FTD, and FTD".
    words = stripped.split()
    if len(words) >= 6:
        trigram_counts = {}
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i + 3]).lower()
            trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1
            if trigram_counts[trigram] >= 4:
                return False

    # Check for token salad (>80% non-alphanumeric)
    alnum_count = sum(1 for ch in stripped if ch.isalnum())
    if len(stripped) > 0 and alnum_count / len(stripped) < 0.2:
        return False

    return True


def generate_ollama_response(ollama_url, model_name, prompt, max_tokens,
                             temperature):
    """Generate a single response via the Ollama HTTP API.

    Posts to ``/api/generate`` with streaming disabled. Retries up to 3 times
    on connection errors or timeouts with a 5-second delay between attempts.

    Args:
        ollama_url: Base URL of the Ollama server (e.g. ``http://localhost:11434``).
        model_name: Ollama model tag (e.g. ``als-lm-500m:q8_0``).
        prompt: The prompt text to send.
        max_tokens: Maximum tokens to generate (``num_predict``).
        temperature: Sampling temperature (0.0 for greedy).

    Returns:
        ``(response_text, tokens_generated)`` on success, or
        ``("[ollama error: <details>]", 0)`` on permanent failure.
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
            data = resp.json()
            response_text = data.get("response", "")
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
                return f"[ollama error: {exc}]", 0
        except requests.HTTPError as exc:
            return f"[ollama error: HTTP {resp.status_code} - {exc}]", 0
        except Exception as exc:
            return f"[ollama error: {exc}]", 0

    return "[ollama error: max retries exceeded]", 0


def generate_responses(model, tokenizer, questions, max_tokens, device):
    """Run inference on all benchmark questions.

    For each question, encodes the prompt_template field, generates a response
    using greedy decoding, and collects structured output.

    Args:
        model: GPT model in eval mode.
        tokenizer: Tokenizer instance.
        questions: List of question dicts from questions.json.
        max_tokens: Maximum tokens to generate per response.
        device: Torch device.

    Returns:
        List of response dicts with question metadata and generated text.
    """
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_token_id is not None:
        print(f"  EOS token ID: {eos_token_id}")
    else:
        print("  WARNING: <|endoftext|> not found in tokenizer vocabulary. "
              "Generation will rely on max_tokens limit only.")

    responses = []
    total = len(questions)

    for i, question in enumerate(questions):
        qid = question["id"]
        prompt_text = question["prompt_template"]

        print(f"  Generating response {i + 1}/{total} ({qid})...")

        try:
            # Encode prompt
            input_ids = tokenizer.encode(prompt_text).ids

            # Generate response
            new_tokens, tokens_generated = generate_greedy(
                model, input_ids, max_tokens, eos_token_id, device
            )

            # Decode only the generated tokens (not the prompt)
            if new_tokens:
                response_text = tokenizer.decode(new_tokens)
            else:
                response_text = ""

        except Exception as e:
            print(f"  WARNING: Generation error for {qid}: {e}")
            response_text = f"[generation error: {e}]"
            tokens_generated = 0

        responses.append({
            "question_id": qid,
            "category": question["category"],
            "difficulty": question["difficulty"],
            "is_trap": question["is_trap"],
            "prompt": prompt_text,
            "response": response_text,
            "tokens_generated": tokens_generated,
            "is_coherent": is_coherent(response_text),
        })

    return responses


def generate_responses_ollama(ollama_url, model_name, questions, max_tokens,
                              temperature):
    """Run inference on all benchmark questions via the Ollama API.

    For each question, sends the ``prompt_template`` to the Ollama server and
    collects the structured response including a coherence check.

    Args:
        ollama_url: Base URL of the Ollama server.
        model_name: Ollama model tag.
        questions: List of question dicts from questions.json.
        max_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature.

    Returns:
        List of response dicts with question metadata and generated text.
    """
    responses = []
    total = len(questions)

    for i, question in enumerate(questions):
        qid = question["id"]
        prompt_text = question["prompt_template"]

        print(f"  Generating response {i + 1}/{total} ({qid})...")

        response_text, tokens_generated = generate_ollama_response(
            ollama_url, model_name, prompt_text, max_tokens, temperature,
        )

        responses.append({
            "question_id": qid,
            "category": question["category"],
            "difficulty": question["difficulty"],
            "is_trap": question["is_trap"],
            "prompt": prompt_text,
            "response": response_text,
            "tokens_generated": tokens_generated,
            "is_coherent": is_coherent(response_text),
        })

    return responses


def _load_existing_responses(output_path):
    """Load previously generated responses from *output_path*.

    Returns a dict mapping ``question_id`` -> response dict, or an empty dict
    if the file does not exist or cannot be parsed.
    """
    if not os.path.isfile(output_path):
        return {}
    try:
        with open(output_path) as fh:
            data = json.load(fh)
        return {r["question_id"]: r for r in data.get("responses", [])}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def main():
    """Main entry point for response generation."""
    args = parse_args()

    print("\n=== ALS-LM Response Generation ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical information system.\n")

    # Determine inference mode
    use_ollama = args.ollama_model is not None

    # ---- Resume logic -------------------------------------------------------
    existing_responses = {}
    if args.resume:
        existing_responses = _load_existing_responses(args.output)
        if existing_responses:
            print(f"  Resume mode: {len(existing_responses)} existing responses loaded")

    # ---- Load benchmark questions --------------------------------------------
    if not os.path.isfile(args.benchmark):
        print(f"ERROR: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    with open(args.benchmark) as f:
        questions = json.load(f)
    print(f"  Benchmark loaded: {len(questions)} questions from {args.benchmark}")

    # Filter questions for resume
    if existing_responses:
        questions_to_run = [
            q for q in questions if q["id"] not in existing_responses
        ]
        print(f"  Skipping {len(questions) - len(questions_to_run)} already-completed questions")
    else:
        questions_to_run = questions

    # ---- Generate responses --------------------------------------------------
    if use_ollama:
        # Ollama mode
        print(f"  Inference mode: Ollama")
        print(f"  Ollama model: {args.ollama_model}")
        print(f"  Ollama URL: {args.ollama_url}")
        print(f"\n  Starting inference (max_tokens={args.max_tokens}, "
              f"temperature={args.temperature})...\n")

        t0 = time.time()
        new_responses = generate_responses_ollama(
            args.ollama_url, args.ollama_model, questions_to_run,
            args.max_tokens, args.temperature,
        )
        elapsed = time.time() - t0

        # Build metadata for Ollama mode
        metadata = {
            "inference_mode": "ollama",
            "ollama_model": args.ollama_model,
            "ollama_url": args.ollama_url,
            "model_name": args.ollama_model,
            "generation_params": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
            "benchmark_path": args.benchmark,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_questions": len(questions),
        }
    else:
        # Checkpoint mode
        _ensure_checkpoint_imports()

        # Resolve device
        if args.device is not None:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        # Load checkpoint
        pt_path = resolve_checkpoint_path(args.checkpoint)
        model, model_config_dict = load_model_from_checkpoint(pt_path, device)

        # Load tokenizer
        tokenizer = load_tokenizer()

        print(f"\n  Starting inference (max_tokens={args.max_tokens}, "
              f"temperature=0.0)...\n")

        t0 = time.time()
        new_responses = generate_responses(
            model, tokenizer, questions_to_run, args.max_tokens, device,
        )
        elapsed = time.time() - t0

        # Build metadata for checkpoint mode
        metadata = {
            "inference_mode": "checkpoint",
            "checkpoint_path": os.path.abspath(pt_path),
            "model_config": model_config_dict,
            "generation_params": {
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
            },
            "benchmark_path": args.benchmark,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_questions": len(questions),
            "device": str(device),
        }

    # ---- Merge with existing responses (resume) -----------------------------
    if existing_responses:
        merged = list(existing_responses.values()) + new_responses
    else:
        merged = new_responses

    # Build output structure
    output = {
        "metadata": metadata,
        "responses": merged,
    }

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write output
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # ---- Summary statistics --------------------------------------------------
    all_responses = merged
    total_tokens = sum(r["tokens_generated"] for r in all_responses)
    empty_count = sum(1 for r in all_responses if r["tokens_generated"] == 0)
    error_count = sum(
        1 for r in all_responses
        if r["response"].startswith("[generation error:")
        or r["response"].startswith("[ollama error:")
    )
    incoherent_count = sum(
        1 for r in all_responses if not r.get("is_coherent", True)
    )

    print(f"\n  === Generation Complete ===")
    if questions_to_run:
        print(f"  Total time: {elapsed:.1f}s "
              f"({elapsed / len(questions_to_run):.2f}s per question)")
    print(f"  New responses generated: {len(new_responses)}")
    print(f"  Total responses (including resumed): {len(all_responses)}")
    print(f"  Total tokens generated: {total_tokens:,}")
    if all_responses:
        print(f"  Average tokens per response: "
              f"{total_tokens / len(all_responses):.1f}")
    print(f"  Empty responses: {empty_count}")
    print(f"  Incoherent responses: {incoherent_count}")
    print(f"  Generation errors: {error_count}")
    print(f"  Output saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
