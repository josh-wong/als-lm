"""Download GPT-2 large pretrained weights and save as a custom checkpoint.

Downloads GPT-2 large (774M parameters) from HuggingFace, transposes Conv1D
weight matrices to nn.Linear format for compatibility with this project's
custom GPT implementation, and saves the result as a checkpoint file.

After saving, a dual self-test verifies correctness:

1. **Logit matching:** Compares forward pass logits between the custom model
   loaded with transposed weights and the original HuggingFace model. The
   outputs must match within atol=1e-5.

2. **ALS validation loss:** Runs a forward pass on the ALS validation corpus
   (tokenized with GPT-2 tokenizer). The loss must be below 5.0, confirming
   the model has pretrained knowledge rather than random-init behavior (~10.8).

This script is the correctness gate for Phase 24 fine-tuning. If either test
fails, the weights were loaded incorrectly and fine-tuning would produce
meaningless results.

Usage::

    # Standard run (download, save, validate)
    python scripts/load_gpt2_weights.py

    # Skip validation (just download and save)
    python scripts/load_gpt2_weights.py --skip-validation

    # Force re-download even if init.pt exists
    python scripts/load_gpt2_weights.py --force

    # Custom output directory
    python scripts/load_gpt2_weights.py --output-dir checkpoints/custom/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path for model imports (same pattern as export/convert_to_hf.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import GPT, GPTConfig  # noqa: E402

# Weight matrix suffixes that require transposition (Conv1D -> nn.Linear).
# HuggingFace Conv1D stores weights as (in_features, out_features) while
# nn.Linear stores as (out_features, in_features). Transposing converts
# from HF format to our format.
TRANSPOSE_SUFFIXES = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]

DEFAULT_OUTPUT_DIR = "checkpoints/gpt2large_init"
CHECKPOINT_FILENAME = "init.pt"


def download_and_transpose(output_dir: str, force: bool = False) -> tuple:
    """Download GPT-2 large, transpose weights, and save checkpoint.

    Args:
        output_dir: Directory to save the checkpoint.
        force: If True, overwrite existing checkpoint.

    Returns:
        Tuple of (our_model, hf_model, output_path) for use in validation.
    """
    output_path = os.path.join(output_dir, CHECKPOINT_FILENAME)

    # Idempotent check
    if os.path.exists(output_path) and not force:
        print(f"init.pt already exists at {output_path}. Use --force to re-download.")
        return None, None, output_path

    if force and os.path.exists(output_path):
        print(f"Overwriting existing init.pt at {output_path}")

    # Step 1: Download GPT-2 large from HuggingFace
    print("Downloading GPT-2 large from HuggingFace...")
    print("  (HuggingFace handles caching internally, subsequent runs are fast)")
    try:
        from transformers import GPT2LMHeadModel
        hf_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    except Exception as e:
        print(f"ERROR: Failed to download GPT-2 large: {e}")
        sys.exit(1)

    hf_sd = hf_model.state_dict()
    print(f"  Downloaded {len(hf_sd)} weight tensors from HuggingFace")

    # Step 2: Instantiate custom model with gpt2-large config
    print("Instantiating custom model with gpt2-large config...")
    our_model = GPT.from_config("gpt2-large", vocab_size=50257)
    our_sd = our_model.state_dict()
    print(f"  Custom model has {len(our_sd)} weight tensors")
    print(f"  Parameters: {our_model.get_num_params():,}")

    # Step 3: Transpose Conv1D weights and build new state dict
    print("Transposing Conv1D weights to nn.Linear format...")
    new_sd = {}
    transposed_count = 0
    copied_count = 0
    skipped_count = 0

    for key in hf_sd:
        # Skip lm_head.weight -- tied to transformer.wte.weight
        if key == "lm_head.weight":
            skipped_count += 1
            continue

        # Check if this key exists in our model
        if key not in our_sd:
            print(f"  WARNING: HF key '{key}' not found in custom model, skipping")
            skipped_count += 1
            continue

        needs_transpose = any(key.endswith(suffix) for suffix in TRANSPOSE_SUFFIXES)

        if needs_transpose:
            # Transpose: Conv1D (in, out) -> nn.Linear (out, in)
            hf_shape = hf_sd[key].shape
            our_shape = our_sd[key].shape
            if not (hf_shape[0] == our_shape[1] and hf_shape[1] == our_shape[0]):
                raise ValueError(
                    f"Shape mismatch for {key}: HF {hf_shape} vs ours {our_shape} "
                    f"(expected transposed shapes to match)"
                )
            new_sd[key] = hf_sd[key].t()
            transposed_count += 1
        else:
            # Direct copy (biases, LayerNorm, embeddings)
            if hf_sd[key].shape != our_sd[key].shape:
                raise ValueError(
                    f"Shape mismatch for {key}: HF {hf_sd[key].shape} vs ours {our_sd[key].shape}"
                )
            new_sd[key] = hf_sd[key]
            copied_count += 1

    print(f"  Transposed: {transposed_count}, Copied: {copied_count}, Skipped: {skipped_count}")

    # Verify all keys in our model are covered
    missing_keys = set(our_sd.keys()) - set(new_sd.keys())
    # lm_head.weight is tied to wte.weight, so it's expected to be "missing"
    # from the HF state dict but present in ours via weight tying
    expected_missing = {"lm_head.weight"}
    unexpected_missing = missing_keys - expected_missing
    if unexpected_missing:
        print(f"  ERROR: Missing keys in mapped state dict: {unexpected_missing}")
        sys.exit(1)

    # Tie lm_head.weight to wte.weight so strict=True succeeds
    new_sd["lm_head.weight"] = new_sd["transformer.wte.weight"]
    our_model.load_state_dict(new_sd, strict=True)

    # Verify weights actually loaded (spot-check a few keys)
    for key in list(new_sd.keys())[:3]:
        if not torch.equal(our_model.state_dict()[key], new_sd[key]):
            raise ValueError(f"Weight verification failed for {key}")
    print("  Weight loading verified (strict load + spot check)")

    # Step 4: Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model": our_model.state_dict(),
        "config": our_model.config,
    }
    torch.save(checkpoint, output_path)
    size_bytes = os.path.getsize(output_path)
    size_gb = size_bytes / (1024**3)
    print(f"  Saved checkpoint to {output_path} ({size_gb:.2f} GB)")

    return our_model, hf_model, output_path


def run_logit_test(our_model, hf_model) -> tuple:
    """Test A: Compare logits between custom and HF models.

    Returns:
        Tuple of (passed, max_diff).
    """
    print("\nTest A: Logit matching against HuggingFace reference...")
    our_model.eval()
    hf_model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(0, 50257, (1, 64))

    with torch.no_grad():
        our_logits, _ = our_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    match = torch.allclose(our_logits, hf_logits, atol=1e-5)
    max_diff = (our_logits - hf_logits).abs().max().item()

    status = "PASS" if match else "FAIL"
    print(f"  Logit match (atol=1e-5): {status}")
    print(f"  Max absolute difference: {max_diff:.2e}")

    return match, max_diff


def run_loss_test(our_model) -> tuple:
    """Test B: Measure loss on ALS validation data.

    Returns:
        Tuple of (passed, val_loss) or (None, None) if val data missing.
    """
    val_path = os.path.join(
        Path(__file__).resolve().parent.parent,
        "data", "tokenized_gpt2", "val.bin",
    )

    print("\nTest B: Loss on ALS validation data...")

    if not os.path.exists(val_path):
        print(f"  WARNING: {val_path} not found, skipping Test B")
        print("  (Run scripts/prepare_gpt2_data.py first to create tokenized data)")
        return None, None

    our_model.eval()

    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    n_tokens = min(10240, len(val_data) - 1)
    x = torch.from_numpy(val_data[:n_tokens].astype(np.int64)).unsqueeze(0)
    y = torch.from_numpy(val_data[1:n_tokens + 1].astype(np.int64)).unsqueeze(0)

    # Use block_size chunks to stay within positional embedding range
    block_size = our_model.config.block_size
    with torch.no_grad():
        _, loss = our_model(x[:, :block_size], targets=y[:, :block_size])

    val_loss = loss.item()
    passed = val_loss < 5.0

    status = "PASS" if passed else "FAIL"
    print(f"  ALS validation loss: {val_loss:.4f}")
    print(f"  Loss gate (<5.0): {status}")

    return passed, val_loss


def run_dual_self_test(our_model, hf_model) -> bool:
    """Run both validation tests and print overall verdict.

    Args:
        our_model: Custom GPT model loaded with transposed weights.
        hf_model: Original HuggingFace GPT2LMHeadModel.

    Returns:
        True if both tests pass (or Test B is skipped), False otherwise.
    """
    logit_passed, max_diff = run_logit_test(our_model, hf_model)
    loss_passed, val_loss = run_loss_test(our_model)

    # Print summary block
    print("\n" + "=" * 56)
    print("SELF-TEST RESULTS")
    print("=" * 56)

    logit_status = "PASS" if logit_passed else "FAIL"
    diff_str = f"{max_diff:.2e}" if max_diff is not None else "N/A"
    print(f"Logit matching:      {logit_status} (max_diff={diff_str})")

    if loss_passed is None:
        print("ALS validation loss: SKIPPED (val.bin not found)")
    else:
        loss_status = "PASS" if loss_passed else "FAIL"
        loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        print(f"ALS validation loss: {loss_str} ({loss_status})")

    print("=" * 56)

    # Overall verdict: logit test is critical, loss test is informational
    # if val.bin is missing
    overall_pass = logit_passed and (loss_passed is None or loss_passed)

    if overall_pass:
        print("OVERALL: PASS -- init.pt is ready for Phase 24 fine-tuning")
    else:
        print("OVERALL: FAIL")
        if not logit_passed:
            print("  -> Logit mismatch: check use_post_ln=False and gelu_approximate='tanh'")
            print("     in gpt2-large config. Verify TRANSPOSE_SUFFIXES matches all Conv1D layers.")
        if loss_passed is not None and not loss_passed:
            print("  -> Loss too high: weights may not be pretrained. Check transposition.")
            print("     Random init loss is ~10.8. Pretrained should be ~3.0-3.5.")

    print("=" * 56)

    return overall_pass


def main() -> None:
    """CLI entry point for the weight loading script."""
    parser = argparse.ArgumentParser(
        description=(
            "Download GPT-2 large pretrained weights, transpose Conv1D to "
            "nn.Linear format, save as init.pt, and run dual self-test."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/load_gpt2_weights.py\n"
            "  python scripts/load_gpt2_weights.py --force\n"
            "  python scripts/load_gpt2_weights.py --skip-validation\n"
            "  python scripts/load_gpt2_weights.py --output-dir checkpoints/custom/\n"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing init.pt instead of skipping",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the dual self-test after saving",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for checkpoint (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    print("=" * 56)
    print("GPT-2 Large Weight Loader")
    print("=" * 56)
    print(f"  Output directory: {args.output_dir}")
    print(f"  Force re-download: {args.force}")
    print(f"  Skip validation: {args.skip_validation}")
    print()

    our_model, hf_model, output_path = download_and_transpose(
        args.output_dir, force=args.force
    )

    # If idempotent skip happened, we're done
    if our_model is None:
        sys.exit(0)

    if args.skip_validation:
        print("\nValidation skipped (--skip-validation)")
        print(f"Checkpoint saved to {output_path}")
        sys.exit(0)

    # Run dual self-test
    overall_pass = run_dual_self_test(our_model, hf_model)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
