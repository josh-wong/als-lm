"""Convert PyTorch GPT checkpoints to HuggingFace GPT2LMHeadModel format.

This script handles the weight transposition required when converting from
a model that uses ``nn.Linear`` layers (standard PyTorch, as used in this
project and nanoGPT) to HuggingFace's ``GPT2LMHeadModel`` which internally
uses ``Conv1D`` layers.

The key difference is in weight matrix storage:

- ``nn.Linear`` stores weights as ``(out_features, in_features)``
- ``Conv1D`` stores weights as ``(in_features, out_features)``

So the conversion must transpose exactly four weight matrix types:

1. ``attn.c_attn.weight`` -- attention QKV combined projection
2. ``attn.c_proj.weight`` -- attention output projection
3. ``mlp.c_fc.weight`` -- MLP up-projection
4. ``mlp.c_proj.weight`` -- MLP down-projection

All other weights (biases, LayerNorm, embeddings) are copied directly.
The ``lm_head.weight`` is NOT included in the converted state dict because
HuggingFace ``GPT2LMHeadModel`` ties it automatically to
``transformer.wte.weight``.

Usage as CLI::

    # Convert a checkpoint
    python export/convert_to_hf.py --checkpoint model.pt --output-dir export/hf_model/

    # Run self-test (no real checkpoint needed)
    python export/convert_to_hf.py --self-test

Usage as importable module::

    from export.convert_to_hf import convert_checkpoint_to_hf, validate_conversion

    hf_model = convert_checkpoint_to_hf("model.pt", "export/hf_model/")
"""

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from transformers import GPT2Config, GPT2LMHeadModel

# Add project root to path for model imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.model import GPT, GPTConfig  # noqa: E402

# Weight matrix suffixes that require transposition (nn.Linear -> Conv1D).
# These are the four projection layers in each transformer block where
# nn.Linear stores weights as (out, in) but Conv1D expects (in, out).
TRANSPOSE_SUFFIXES = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]


def convert_checkpoint_to_hf(
    checkpoint_path: str,
    output_dir: str,
    tokenizer_dir: Optional[str] = None,
) -> GPT2LMHeadModel:
    """Convert a PyTorch GPT checkpoint to HuggingFace GPT2LMHeadModel.

    Loads a checkpoint containing a model state dict and GPTConfig, creates
    a matching HuggingFace GPT2LMHeadModel, maps all weights with appropriate
    transposition, validates weight tying, and saves the result.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint file. The checkpoint
            must contain at minimum ``'model'`` (state dict) and ``'config'``
            (GPTConfig instance or dict with matching fields).
        output_dir: Directory where the HuggingFace model will be saved.
            Created if it does not exist.
        tokenizer_dir: Optional path to a HuggingFace tokenizer directory.
            If provided, tokenizer files are copied into ``output_dir`` to
            create a complete HuggingFace model directory.

    Returns:
        The converted ``GPT2LMHeadModel`` instance (in eval mode).

    Raises:
        AssertionError: If weight shapes do not match after transposition,
            or if weight tying is broken after loading.
        KeyError: If the checkpoint is missing required keys.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config, handling both GPTConfig dataclass and plain dict
    raw_config = ckpt["config"]
    if isinstance(raw_config, GPTConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        config = GPTConfig(**raw_config)
    else:
        raise TypeError(
            f"Checkpoint 'config' must be GPTConfig or dict, got {type(raw_config)}"
        )

    model_state = ckpt["model"]

    # Create matching HuggingFace GPT2Config
    hf_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.block_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        activation_function="gelu",
        resid_pdrop=config.dropout,
        embd_pdrop=config.dropout,
        attn_pdrop=config.dropout,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=0,
    )

    # Create HuggingFace model and get its state dict as a reference
    print("Creating HuggingFace GPT2LMHeadModel...")
    hf_model = GPT2LMHeadModel(hf_config)
    hf_sd = hf_model.state_dict()

    # Map weights from our model to HuggingFace format
    transposed_count = 0
    copied_count = 0
    skipped_count = 0

    for key in model_state:
        # Skip lm_head.weight -- HuggingFace ties it to transformer.wte.weight
        # automatically via _tied_weights_keys. Including it separately would
        # break the weight tie.
        if key == "lm_head.weight":
            skipped_count += 1
            continue

        if key not in hf_sd:
            print(f"  WARNING: Key '{key}' not found in HuggingFace model, skipping")
            skipped_count += 1
            continue

        needs_transpose = any(key.endswith(suffix) for suffix in TRANSPOSE_SUFFIXES)

        if needs_transpose:
            # Transpose: nn.Linear (out, in) -> Conv1D (in, out)
            our_shape = model_state[key].shape
            hf_shape = hf_sd[key].shape
            assert our_shape[0] == hf_shape[1] and our_shape[1] == hf_shape[0], (
                f"Shape mismatch for {key}: ours {our_shape} vs HF {hf_shape} "
                f"(expected transposed shapes to match)"
            )
            hf_sd[key] = model_state[key].t()
            transposed_count += 1
        else:
            # Direct copy (biases, LayerNorm, embeddings)
            assert model_state[key].shape == hf_sd[key].shape, (
                f"Shape mismatch for {key}: ours {model_state[key].shape} "
                f"vs HF {hf_sd[key].shape}"
            )
            hf_sd[key] = model_state[key]
            copied_count += 1

    print(
        f"  Mapped {transposed_count} transposed + {copied_count} copied weights "
        f"({skipped_count} skipped)"
    )

    # Load the mapped state dict
    hf_model.load_state_dict(hf_sd)

    # Verify weight tying is intact
    assert hf_model.lm_head.weight is hf_model.transformer.wte.weight, (
        "Weight tying broken after loading state dict. "
        "lm_head.weight should be the same tensor as transformer.wte.weight"
    )
    print("  Weight tying verified")

    # Save the HuggingFace model
    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir)
    print(f"  Model saved to {output_dir}")

    # Copy tokenizer files if provided
    if tokenizer_dir is not None:
        tokenizer_path = Path(tokenizer_dir)
        if tokenizer_path.is_dir():
            for f in tokenizer_path.iterdir():
                if f.is_file():
                    shutil.copy2(f, output_dir)
            print(f"  Tokenizer files copied from {tokenizer_dir}")
        else:
            print(f"  WARNING: Tokenizer directory '{tokenizer_dir}' not found")

    return hf_model


def validate_conversion(
    our_model: GPT,
    hf_model: GPT2LMHeadModel,
    vocab_size: int,
    device: str = "cpu",
) -> dict:
    """Validate that our model and the HuggingFace model produce matching logits.

    Uses random token IDs as input (with a fixed seed for reproducibility) to
    compare the forward pass outputs of both models. This approach is
    independent of any tokenizer, testing only the model conversion fidelity.

    Args:
        our_model: The original PyTorch GPT model.
        hf_model: The converted HuggingFace GPT2LMHeadModel.
        vocab_size: Vocabulary size for generating random input tokens.
        device: Device to run validation on (default: "cpu").

    Returns:
        A dict with validation results containing:

        - ``logits_match`` (bool): Whether logits are within atol=1e-5
        - ``max_diff`` (float): Maximum absolute difference between logits
        - ``atol`` (float): The tolerance used (1e-5)
        - ``input_length`` (int): Length of the test input sequence
        - ``next_token_match`` (bool): Whether the predicted next token is
          identical from both models
    """
    our_model.eval()
    hf_model.eval()
    our_model.to(device)
    hf_model.to(device)

    # Fixed seed for reproducible test input (random tokens, not real text)
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, 64), device=device)

    with torch.no_grad():
        our_logits, _ = our_model(input_ids)
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits

    # Compare logits
    logits_match = torch.allclose(our_logits, hf_logits, atol=1e-5)
    max_diff = (our_logits - hf_logits).abs().max().item()

    # Compare predicted next token (argmax of last position logits)
    our_next = torch.argmax(our_logits[0, -1, :]).item()
    hf_next = torch.argmax(hf_logits[0, -1, :]).item()
    next_token_match = our_next == hf_next

    return {
        "logits_match": logits_match,
        "max_diff": max_diff,
        "atol": 1e-5,
        "input_length": 64,
        "next_token_match": next_token_match,
    }


def run_self_test() -> bool:
    """Run a self-contained conversion test without a real checkpoint.

    Creates a tiny model with random weights, saves a mock checkpoint,
    converts it to HuggingFace format, validates logits match, and cleans
    up temporary files.

    Returns:
        True if the self-test passes, False otherwise.
    """
    print("Running self-test...")
    print("=" * 60)

    # Create a tiny model
    config = GPTConfig(
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        dropout=0.0,
        bias=True,
    )
    our_model = GPT(config)
    print(f"  Created tiny model: {our_model.get_num_params():,} parameters")

    # Save a mock checkpoint
    tmpdir = tempfile.mkdtemp(prefix="als_lm_selftest_")
    checkpoint_path = os.path.join(tmpdir, "model.pt")
    output_dir = os.path.join(tmpdir, "hf_model")

    checkpoint = {
        "model": our_model.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  Saved mock checkpoint to {checkpoint_path}")

    try:
        # Convert to HuggingFace
        hf_model = convert_checkpoint_to_hf(checkpoint_path, output_dir)

        # Validate conversion
        results = validate_conversion(our_model, hf_model, config.vocab_size)

        print("\n" + "=" * 60)
        print("Validation results:")
        print(f"  Logits match (atol=1e-5): {results['logits_match']}")
        print(f"  Max absolute difference:  {results['max_diff']:.2e}")
        print(f"  Next token match:         {results['next_token_match']}")
        print(f"  Input length:             {results['input_length']} tokens")

        if results["logits_match"] and results["next_token_match"]:
            print("\nSelf-test PASSED")
            return True
        else:
            print("\nSelf-test FAILED")
            if not results["logits_match"]:
                print(f"  Logits differ by {results['max_diff']:.2e} (atol=1e-5)")
            if not results["next_token_match"]:
                print("  Predicted next tokens differ")
            return False

    finally:
        # Clean up temporary files
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"  Cleaned up temporary files")


def main() -> None:
    """CLI entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch GPT checkpoint to HuggingFace GPT2LMHeadModel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python export/convert_to_hf.py --self-test\n"
            "  python export/convert_to_hf.py --checkpoint model.pt --output-dir export/hf_model/\n"
            "  python export/convert_to_hf.py --checkpoint model.pt --output-dir export/hf_model/ "
            "--tokenizer-dir tokenizer/hf_tokenizer/\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the PyTorch checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the HuggingFace model",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="Optional path to HuggingFace tokenizer directory to include in output",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip logit comparison after conversion (useful for large models)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for validation (default: cpu)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test with a tiny model and exit",
    )

    args = parser.parse_args()

    if args.self_test:
        success = run_self_test()
        sys.exit(0 if success else 1)

    if not args.checkpoint or not args.output_dir:
        parser.error("--checkpoint and --output-dir are required (or use --self-test)")

    # Convert checkpoint
    hf_model = convert_checkpoint_to_hf(
        args.checkpoint, args.output_dir, args.tokenizer_dir
    )

    if args.skip_validation:
        print("Validation skipped (--skip-validation)")
        sys.exit(0)

    # Load our model for validation
    print("\nLoading original model for validation...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_config = ckpt["config"]
    if isinstance(raw_config, GPTConfig):
        config = raw_config
    elif isinstance(raw_config, dict):
        config = GPTConfig(**raw_config)
    else:
        raise TypeError(f"Unexpected config type: {type(raw_config)}")

    our_model = GPT(config)
    our_model.load_state_dict(ckpt["model"])

    # Validate
    results = validate_conversion(
        our_model, hf_model, config.vocab_size, device=args.device
    )

    print("\nValidation results:")
    print(f"  Logits match (atol=1e-5): {results['logits_match']}")
    print(f"  Max absolute difference:  {results['max_diff']:.2e}")
    print(f"  Next token match:         {results['next_token_match']}")

    if results["logits_match"]:
        print("\nConversion PASSED")
        sys.exit(0)
    else:
        print("\nConversion FAILED: logits do not match within tolerance")
        sys.exit(1)


if __name__ == "__main__":
    main()
