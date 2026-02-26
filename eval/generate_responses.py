#!/usr/bin/env python3
"""Generate model responses to ALS benchmark questions for evaluation.

Loads a model checkpoint, runs greedy inference on every benchmark question,
and saves structured JSON output for downstream scoring and fabrication
detection. The script is checkpoint-agnostic and works with tiny, medium, or
500M models. It is robust to degenerate outputs (gibberish, repetition, empty
output) from undertrained models.

This is a research evaluation tool, not a medical information system.

Usage examples::

    # Using a checkpoint directory (contains best.pt)
    python eval/generate_responses.py --checkpoint checkpoints/tiny_20260225/best

    # Using a direct .pt file
    python eval/generate_responses.py --checkpoint checkpoints/tiny_20260225/best/best.pt

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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
except (ImportError, SystemExit):
    _PROJECT_ROOT = None
    _DEFAULTS = {}

import torch
from tokenizers import Tokenizer

from model.model import GPT, GPTConfig


def parse_args():
    """Parse command-line arguments for response generation."""
    parser = argparse.ArgumentParser(
        description="Generate model responses to ALS benchmark questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/generate_responses.py --checkpoint checkpoints/tiny/best\n"
            "  python eval/generate_responses.py --checkpoint best.pt --device cpu\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (containing best.pt) or direct .pt file",
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
    return parser.parse_args()


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


def load_tokenizer(tokenizer_path="tokenizer/als_tokenizer.json"):
    """Load the canonical ALS tokenizer.

    Returns the Tokenizer instance or exits with an error message.
    """
    if not os.path.isfile(tokenizer_path):
        print(f"ERROR: Tokenizer not found: {tokenizer_path}")
        print("  The canonical tokenizer must exist at tokenizer/als_tokenizer.json")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"  Tokenizer loaded: {tokenizer.get_vocab_size():,} tokens")
    return tokenizer


@torch.no_grad()
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
        })

    return responses


def main():
    """Main entry point for response generation."""
    args = parse_args()

    print("\n=== ALS-LM Response Generation ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical information system.\n")

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

    # Load benchmark questions
    if not os.path.isfile(args.benchmark):
        print(f"ERROR: Benchmark file not found: {args.benchmark}")
        sys.exit(1)

    with open(args.benchmark) as f:
        questions = json.load(f)
    print(f"  Benchmark loaded: {len(questions)} questions from {args.benchmark}")

    # Generate responses
    print(f"\n  Starting inference (max_tokens={args.max_tokens}, temperature=0.0)...\n")
    t0 = time.time()
    responses = generate_responses(model, tokenizer, questions, args.max_tokens, device)
    elapsed = time.time() - t0

    # Build output structure
    output = {
        "metadata": {
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
        },
        "responses": responses,
    }

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write output
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Summary statistics
    total_tokens = sum(r["tokens_generated"] for r in responses)
    empty_count = sum(1 for r in responses if r["tokens_generated"] == 0)
    error_count = sum(1 for r in responses if r["response"].startswith("[generation error:"))

    print(f"\n  === Generation Complete ===")
    print(f"  Total time: {elapsed:.1f}s ({elapsed / len(questions):.2f}s per question)")
    print(f"  Total tokens generated: {total_tokens:,}")
    print(f"  Average tokens per response: {total_tokens / len(questions):.1f}")
    print(f"  Empty responses: {empty_count}")
    print(f"  Generation errors: {error_count}")
    print(f"  Output saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
