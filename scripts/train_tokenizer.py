"""Train byte-level BPE tokenizers on the ALS corpus.

Trains GPT-2-style byte-level BPE tokenizers at multiple vocabulary sizes
using the HuggingFace tokenizers low-level API. Each tokenizer uses NFC
Unicode normalization (preserving Greek letters and superscripts per Phase 2
decision), ByteLevel pre-tokenization, and <|endoftext|> as the sole special
token.

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --corpus data/processed/train.txt --vocab-sizes 16384,32768,50257
    python scripts/train_tokenizer.py --output-dir tokenizer/ --vocab-sizes 32768
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)


def build_tokenizer() -> Tokenizer:
    """Assemble a GPT-2-style byte-level BPE tokenizer (untrained).

    Components follow the exact GPT-2 architecture with the addition
    of NFC normalization to preserve Greek letters and superscripts
    used in medical text.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def vocab_size_label(size: int) -> str:
    """Convert a vocab size integer to a human-readable label.

    Examples: 16384 -> '16k', 32768 -> '32k', 50257 -> '50k'
    """
    if size <= 16384:
        return "16k"
    elif size <= 32768:
        return "32k"
    else:
        return "50k"


def train_single_tokenizer(
    corpus_path: str, output_path: str, vocab_size: int
) -> dict:
    """Train a single BPE tokenizer and verify it.

    Args:
        corpus_path: Path to the training text file (train.txt only).
        output_path: Path to save the tokenizer JSON.
        vocab_size: Target vocabulary size.

    Returns:
        Dictionary with training statistics.
    """
    tokenizer = build_tokenizer()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"\nTraining tokenizer with vocab_size={vocab_size}...")
    start = time.time()
    tokenizer.train([corpus_path], trainer=trainer)
    duration = time.time() - start
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  Trained in {duration:.2f}s, actual vocab size: {actual_vocab}")
    if actual_vocab < vocab_size:
        print(
            f"  Note: Actual vocab ({actual_vocab}) < target ({vocab_size}). "
            f"Corpus exhausted all possible merges at min_frequency=2. "
            f"This is normal for small corpora; a larger corpus will reach "
            f"the target."
        )

    # Save tokenizer
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    print(f"  Saved to {output_path}")

    # Run verification tests
    stats = {
        "target_vocab_size": vocab_size,
        "actual_vocab_size": actual_vocab,
        "training_time_seconds": round(duration, 2),
        "output_path": output_path,
        "round_trip_tests": [],
        "special_token_test": None,
    }

    # Round-trip test strings
    # Note: decode() strips special tokens by default. For text containing
    # <|endoftext|>, we use skip_special_tokens=False to verify full
    # round-trip fidelity including the separator.
    test_strings = [
        ("The quick brown fox jumps over the lazy dog.", False),
        ("Riluzole inhibits glutamate release in ALS patients.", False),
        (
            "First document.<|endoftext|>Second document about SOD1 mutations.",
            True,
        ),
    ]

    all_passed = True
    for test_text, has_special in test_strings:
        encoded = tokenizer.encode(test_text)
        # For text with special tokens, preserve them by disabling
        # skip_special_tokens (which is True by default in decode)
        decoded = tokenizer.decode(
            encoded.ids,
            skip_special_tokens=not has_special,
        )
        passed = decoded == test_text
        if not passed:
            all_passed = False
            print(f"  FAILED round-trip: {test_text!r} -> {decoded!r}")
        stats["round_trip_tests"].append(
            {
                "input": test_text,
                "output": decoded,
                "passed": passed,
                "n_tokens": len(encoded.ids),
            }
        )

    if all_passed:
        print(f"  Round-trip: PASSED (all {len(test_strings)} tests)")
    else:
        print(f"  Round-trip: FAILED")

    # Special token test: <|endoftext|> should encode as exactly 1 token
    eot_encoded = tokenizer.encode("<|endoftext|>")
    eot_count = len(eot_encoded.ids)
    eot_passed = eot_count == 1
    stats["special_token_test"] = {
        "token": "<|endoftext|>",
        "n_tokens": eot_count,
        "token_id": eot_encoded.ids[0] if eot_passed else None,
        "passed": eot_passed,
    }

    if eot_passed:
        print(
            f"  <|endoftext|>: PASSED (single token, ID={eot_encoded.ids[0]})"
        )
    else:
        print(
            f"  <|endoftext|>: FAILED (encoded as {eot_count} tokens)"
        )

    stats["all_passed"] = all_passed and eot_passed
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Train byte-level BPE tokenizers on the ALS corpus"
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/train.txt",
        help="Path to training text file (default: data/processed/train.txt)",
    )
    parser.add_argument(
        "--vocab-sizes",
        default="16384,32768,50257",
        help="Comma-separated vocab sizes (default: 16384,32768,50257)",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizer/",
        help="Output directory for tokenizer JSON files (default: tokenizer/)",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    corpus_size_mb = corpus_path.stat().st_size / (1024 * 1024)
    print(f"Corpus: {corpus_path} ({corpus_size_mb:.2f} MB)")

    vocab_sizes = [int(s.strip()) for s in args.vocab_sizes.split(",")]
    print(f"Vocab sizes to train: {vocab_sizes}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for vocab_size in vocab_sizes:
        label = vocab_size_label(vocab_size)
        output_path = str(output_dir / f"als_tokenizer_{label}.json")
        stats = train_single_tokenizer(
            str(corpus_path), output_path, vocab_size
        )
        all_stats.append(stats)

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(
        f"{'Vocab Size':>12} {'Actual':>8} {'Time (s)':>10} {'Round-trip':>12} {'EOT':>6}"
    )
    print("-" * 60)
    for s in all_stats:
        rt = "PASS" if all(t["passed"] for t in s["round_trip_tests"]) else "FAIL"
        eot = "PASS" if s["special_token_test"]["passed"] else "FAIL"
        print(
            f"{s['target_vocab_size']:>12} {s['actual_vocab_size']:>8} "
            f"{s['training_time_seconds']:>10.2f} {rt:>12} {eot:>6}"
        )

    # Write summary JSON
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nFull summary written to {summary_path}")

    # Check for any failures
    all_passed = all(s["all_passed"] for s in all_stats)
    if not all_passed:
        print("\nWARNING: Some tests failed!", file=sys.stderr)
        sys.exit(1)

    print("\nAll tokenizers trained and verified successfully.")


if __name__ == "__main__":
    main()
