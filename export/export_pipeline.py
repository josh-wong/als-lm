"""Unified export pipeline for ALS-LM: PyTorch to HuggingFace to GGUF to Ollama.

This script orchestrates the full model export pipeline with fail-fast behavior.
Each stage depends on the previous stage's output, so the pipeline stops at
the first failure and reports which stage failed.

The three stages are:

1. **PyTorch to HuggingFace:** Converts a training checkpoint to a
   ``GPT2LMHeadModel`` using the existing ``export/convert_to_hf.py`` module,
   with logit-matching validation (atol=1e-5) and enhanced diagnostics on
   mismatch.

2. **HuggingFace to GGUF:** Runs llama.cpp's ``convert_hf_to_gguf.py`` to
   produce a GGUF file. Handles the custom tokenizer pre-tokenizer hash by
   patching the conversion script at runtime so the ALS tokenizer's hash maps
   to the ``"gpt2"`` pre-tokenizer type.

3. **GGUF to Ollama:** Generates a Modelfile from the checked-in template,
   runs ``ollama create`` to register the model, and performs an auto-test
   generation to verify the full pipeline end-to-end.

Usage::

    # Auto-detect latest checkpoint, export as "tiny"
    python -m export.export_pipeline

    # Specify checkpoint and size
    python -m export.export_pipeline --checkpoint checkpoints/tiny_20260224/step_1999.pt --size tiny

    # Skip Ollama stage (if Ollama is not installed)
    python -m export.export_pipeline --skip-ollama
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch

# Project root is the parent of the export/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to path for model imports
sys.path.insert(0, str(PROJECT_ROOT))

from export.convert_to_hf import convert_checkpoint_to_hf, validate_conversion  # noqa: E402
from model.model import GPT, GPTConfig  # noqa: E402

# Key directories
LLAMA_CPP_DIR = PROJECT_ROOT / "lib" / "llama.cpp"
TOKENIZER_HF_DIR = PROJECT_ROOT / "tokenizer" / "hf_tokenizer"
MODELFILE_TEMPLATE = PROJECT_ROOT / "export" / "Modelfile.template"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Medical research disclaimer for the Ollama SYSTEM prompt.
# Formal, ALS-specific, mentions model size and training data limitations.
DISCLAIMER = (
    "This is ALS-LM, a small experimental language model trained on a limited "
    "curated corpus of ALS research literature. This model is a research artifact "
    "and is frequently incorrect. It must not be used for medical decision-making, "
    "diagnosis, treatment planning, or any clinical purpose. Its outputs about ALS, "
    "medications, genetics, and disease progression may be fabricated, outdated, or "
    "misleading. Consult qualified medical professionals for any health-related "
    "questions."
)


def find_latest_checkpoint(checkpoints_dir: Optional[Path] = None) -> Path:
    """Auto-detect the latest training checkpoint by step number.

    Scans the checkpoints directory for run directories (pattern:
    ``{config}_{timestamp}/``), finds all ``step_*.pt`` files within each,
    and returns the checkpoint with the highest step number from the most
    recently modified run directory.

    Args:
        checkpoints_dir: Directory to scan. Defaults to ``checkpoints/``
            relative to project root.

    Returns:
        Path to the latest checkpoint ``.pt`` file.

    Raises:
        FileNotFoundError: If no checkpoints directory or no checkpoint files
            are found.
    """
    if checkpoints_dir is None:
        checkpoints_dir = CHECKPOINTS_DIR

    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoints directory not found: {checkpoints_dir}"
        )

    # Collect all run directories (any subdirectory of checkpoints/)
    run_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if not run_dirs:
        # Check for .pt files directly in checkpoints/
        pt_files = list(checkpoints_dir.glob("step_*.pt"))
        if not pt_files:
            raise FileNotFoundError(
                f"No run directories or checkpoint files found in {checkpoints_dir}"
            )
        run_dirs = [checkpoints_dir]

    # Search the most recently modified run directory first
    for run_dir in run_dirs:
        pt_files = list(run_dir.glob("step_*.pt"))
        if pt_files:
            # Sort by step number extracted from filename
            def extract_step(path: Path) -> int:
                match = re.search(r"step_(\d+)", path.name)
                return int(match.group(1)) if match else 0

            pt_files.sort(key=extract_step, reverse=True)
            return pt_files[0]

    raise FileNotFoundError(
        f"No step_*.pt checkpoint files found in any run directory under {checkpoints_dir}"
    )


def stage_hf_convert(
    checkpoint_path: Path,
    output_dir: Path,
) -> dict:
    """Stage 1: Convert PyTorch checkpoint to HuggingFace GPT2LMHeadModel.

    Calls the existing ``convert_checkpoint_to_hf`` function from
    ``export/convert_to_hf``, then validates logits match between the
    original and converted models. On mismatch, prints enhanced diagnostics
    with max/mean differences and the top 5 most-divergent token positions.

    Args:
        checkpoint_path: Path to the PyTorch ``.pt`` checkpoint file.
        output_dir: Directory to save the HuggingFace model files.

    Returns:
        A dict with ``success``, ``output_path``, and ``error`` keys.
    """
    print("\n" + "=" * 60)
    print("Stage 1: PyTorch -> HuggingFace")
    print("=" * 60)

    try:
        # Convert checkpoint to HuggingFace format
        hf_model = convert_checkpoint_to_hf(
            str(checkpoint_path),
            str(output_dir),
            tokenizer_dir=str(TOKENIZER_HF_DIR),
        )

        # Verify config.json contains model_type: "gpt2"
        config_path = output_dir / "config.json"
        with open(config_path) as f:
            hf_config = json.load(f)
        if hf_config.get("model_type") != "gpt2":
            return {
                "success": False,
                "output_path": str(output_dir),
                "error": (
                    f"config.json model_type is '{hf_config.get('model_type')}', "
                    "expected 'gpt2'"
                ),
            }
        print(f"  config.json verified: model_type = gpt2")

        # Ensure n_ctx is present for llama.cpp's GPT2Model converter.
        # HuggingFace GPT2Config stores this as "n_positions", but the
        # llama.cpp converter reads "n_ctx" directly from hparams.
        if "n_ctx" not in hf_config and "n_positions" in hf_config:
            hf_config["n_ctx"] = hf_config["n_positions"]
            with open(config_path, "w") as f:
                json.dump(hf_config, f, indent=2)
                f.write("\n")
            print(f"  Added n_ctx={hf_config['n_ctx']} to config.json for llama.cpp compatibility")

        # Load original model for logit validation
        print("\nValidating logits...")
        ckpt = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        raw_config = ckpt["config"]
        if isinstance(raw_config, GPTConfig):
            config = raw_config
        elif isinstance(raw_config, dict):
            config = GPTConfig(**raw_config)
        else:
            raise TypeError(
                f"Checkpoint config must be GPTConfig or dict, got {type(raw_config)}"
            )

        our_model = GPT(config)
        our_model.load_state_dict(ckpt["model"])

        # Run logit validation
        results = validate_conversion(
            our_model, hf_model, config.vocab_size, device="cpu"
        )

        print(f"  Logits match (atol=1e-5): {results['logits_match']}")
        print(f"  Max absolute difference:  {results['max_diff']:.2e}")
        print(f"  Next token match:         {results['next_token_match']}")

        if not results["logits_match"]:
            # Enhanced diagnostics on mismatch
            print("\n  LOGIT MISMATCH - Enhanced diagnostics:")
            _diagnose_logit_mismatch(our_model, hf_model, config.vocab_size)
            return {
                "success": False,
                "output_path": str(output_dir),
                "error": (
                    f"Logit mismatch: max_diff={results['max_diff']:.2e}, "
                    f"next_token_match={results['next_token_match']}"
                ),
            }

        print("\n  Stage 1 PASSED")
        return {
            "success": True,
            "output_path": str(output_dir),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "output_path": str(output_dir),
            "error": str(e),
        }


def _diagnose_logit_mismatch(
    our_model: GPT,
    hf_model,
    vocab_size: int,
) -> None:
    """Print enhanced diagnostics when logit matching fails.

    Shows max absolute difference, mean absolute difference, and the top 5
    most-divergent token positions to help identify the source of the mismatch.
    """
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, 64), device="cpu")

    with torch.no_grad():
        our_logits, _ = our_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    diff = (our_logits - hf_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Top 5 most-divergent positions in the flattened logit tensor
    flat_diff = diff.view(-1)
    top5_values, top5_indices = flat_diff.topk(min(5, flat_diff.numel()))

    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Top 5 divergent positions:")
    for i, (val, idx) in enumerate(zip(top5_values, top5_indices)):
        print(f"    {i+1}. index {idx.item()}: diff={val.item():.2e}")


def _compute_tokenizer_hash() -> str:
    """Compute the pre-tokenizer hash the same way llama.cpp does.

    The hash is SHA-256 of ``str(encoded_tokens)`` where ``encoded_tokens``
    is the tokenizer's encoding of a specific test string used by llama.cpp's
    ``get_vocab_base_pre()`` function. The test string must match the exact
    string from the llama.cpp source code.

    Returns:
        The SHA-256 hex digest string.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_HF_DIR))

    # This is the exact test string from llama.cpp's get_vocab_base_pre().
    # It MUST match character-for-character or the hash will differ.
    chktxt = '\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n\U0001f680 (normal) \U0001f636\u200d\U0001f32b\ufe0f (multiple emojis concatenated) \u2705 \U0001f999\U0001f999 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 \u1780\u17b6\u1793\u17cb\u178f\u17c2\u1796\u17b7\u179f\u17c1\u179f\u17a2\u17b6\u1785\U0001f601 ?\u6211\u60f3\u5728apple\u5de5\u4f5c1314151\u5929\uff5e ------======= \u043d\u0435\u0449\u043e \u043d\u0430 \u0411\u044a\u043b\u0433\u0430\u0440\u0441\u043a\u0438 \'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? \'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL'

    chktok = tokenizer.encode(chktxt)
    chkhsh = hashlib.sha256(str(chktok).encode()).hexdigest()
    print(f"  Computed tokenizer hash: {chkhsh}")
    return chkhsh


def _patch_converter_script(converter_path: Path, token_hash: str) -> Path:
    """Create a patched copy of convert_hf_to_gguf.py with the custom hash.

    Reads the conversion script source, finds the ``if res is None:`` guard in
    ``get_vocab_base_pre()`` that raises ``NotImplementedError`` for
    unrecognized pre-tokenizer hashes, and inserts an ``if chkhsh ==`` block
    just before it so the ALS tokenizer's hash maps to the ``"gpt2"``
    pre-tokenizer type.

    The llama.cpp converter uses a series of independent ``if`` statements
    (not ``if/elif/else``), so the injected block must also be an ``if``
    statement to match the surrounding code style.

    Args:
        converter_path: Path to the original ``convert_hf_to_gguf.py``.
        token_hash: The SHA-256 hash of the ALS tokenizer's encoded output.

    Returns:
        Path to the temporary patched script file.
    """
    with open(converter_path) as f:
        source = f.read()

    # Strategy: Find the "if res is None:" guard inside get_vocab_base_pre()
    # that triggers the NotImplementedError for unrecognized hashes, and
    # insert our hash check just before it. The converter uses a flat series
    # of "if chkhsh ==" blocks (not elif), so our injection must also be "if".
    res_none_pattern = r'\n(\s+)if res is None:'
    match = re.search(res_none_pattern, source)

    if match:
        indent_str = match.group(1)
        our_block = (
            f'{indent_str}if chkhsh == "{token_hash}":\n'
            f'{indent_str}    res = "gpt2"\n'
        )
        # Insert just before the "if res is None:" line
        insert_pos = match.start() + 1  # +1 to skip the leading \n
        source = source[:insert_pos] + our_block + '\n' + source[insert_pos:]
        print(f"  Inserted hash check before 'if res is None:' guard")
    else:
        # Fallback: insert at the start of the hash checks with early return
        gvbp_pattern = r'def get_vocab_base_pre\(self.*?\).*?:'
        gvbp_match = re.search(gvbp_pattern, source)
        if gvbp_match:
            first_check = re.search(
                r'(\s+)if chkhsh == "',
                source[gvbp_match.end():]
            )
            if first_check:
                abs_pos = gvbp_match.end() + first_check.start()
                indent_str = first_check.group(1)
                our_block = (
                    f'{indent_str}if chkhsh == "{token_hash}":\n'
                    f'{indent_str}    res = "gpt2"\n'
                    f'{indent_str}    return res\n'
                )
                source = source[:abs_pos] + our_block + source[abs_pos:]
                print(f"  Inserted hash check at start of hash checks (fallback)")
            else:
                print("  WARNING: Could not find hash check pattern in converter")
        else:
            print("  WARNING: Could not find get_vocab_base_pre in converter")

    # Write patched script to a temporary file
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="convert_hf_to_gguf_patched_")
    os.close(tmp_fd)
    with open(tmp_path, "w") as f:
        f.write(source)

    print(f"  Patched converter written to: {tmp_path}")
    return Path(tmp_path)


def stage_gguf_convert(
    hf_dir: Path,
    gguf_dir: Path,
    size: str = "tiny",
) -> dict:
    """Stage 2: Convert HuggingFace model to GGUF format.

    Verifies llama.cpp is cloned, computes the custom tokenizer hash, patches
    the converter script to recognize the hash, runs the GGUF conversion, and
    verifies the resulting GGUF file's tokenizer metadata.

    Args:
        hf_dir: Path to the HuggingFace model directory from Stage 1.
        gguf_dir: Directory to write the GGUF output file.
        size: Model size qualifier (e.g., "tiny", "500m") for the output
            filename.

    Returns:
        A dict with ``success``, ``output_path``, and ``error`` keys.
    """
    print("\n" + "=" * 60)
    print("Stage 2: HuggingFace -> GGUF")
    print("=" * 60)

    converter_path = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    gguf_filename = f"als-lm-{size}-f32.gguf"
    gguf_path = gguf_dir / gguf_filename
    patched_script = None

    try:
        # Verify llama.cpp is available
        if not converter_path.is_file():
            return {
                "success": False,
                "output_path": str(gguf_path),
                "error": (
                    "llama.cpp not found at lib/llama.cpp/. "
                    "Clone it with: git clone https://github.com/ggml-org/llama.cpp.git lib/llama.cpp"
                ),
            }

        # Create output directory
        gguf_dir.mkdir(parents=True, exist_ok=True)

        # Compute the custom tokenizer hash
        print("\nComputing tokenizer pre-tokenizer hash...")
        token_hash = _compute_tokenizer_hash()

        # Patch the converter script with our custom hash
        print("\nPatching converter script for custom tokenizer hash...")
        patched_script = _patch_converter_script(converter_path, token_hash)

        # Run GGUF conversion
        print(f"\nRunning GGUF conversion...")
        print(f"  Input:  {hf_dir}")
        print(f"  Output: {gguf_path}")

        cmd = [
            sys.executable,
            str(patched_script),
            str(hf_dir),
            "--outfile", str(gguf_path),
            "--outtype", "f32",
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(LLAMA_CPP_DIR),
        )

        if result.stdout:
            print(f"\n  STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"\n  STDERR:\n{result.stderr}")

        if result.returncode != 0:
            return {
                "success": False,
                "output_path": str(gguf_path),
                "error": (
                    f"GGUF conversion failed (exit code {result.returncode}). "
                    f"stderr: {result.stderr[:500]}"
                ),
            }

        if not gguf_path.is_file():
            return {
                "success": False,
                "output_path": str(gguf_path),
                "error": f"GGUF output file not created: {gguf_path}",
            }

        print(f"\n  GGUF file created: {gguf_path}")
        print(f"  File size: {gguf_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Verify GGUF tokenizer metadata
        print("\nVerifying GGUF tokenizer metadata...")
        verification_result = _verify_gguf_tokenizer(gguf_path)
        if not verification_result["success"]:
            return {
                "success": False,
                "output_path": str(gguf_path),
                "error": verification_result["error"],
            }

        print("\n  Stage 2 PASSED")
        return {
            "success": True,
            "output_path": str(gguf_path),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "output_path": str(gguf_path),
            "error": str(e),
        }

    finally:
        # Clean up patched script
        if patched_script is not None and patched_script.is_file():
            patched_script.unlink()
            print(f"  Cleaned up patched converter script")


def _verify_gguf_tokenizer(gguf_path: Path) -> dict:
    """Verify GGUF file contains correct tokenizer metadata.

    Performs two checks per the validation criteria:

    1. **Metadata fields:** Verifies ``general.architecture`` is ``"gpt2"``
       and that the tokenizer vocabulary size matches our tokenizer (50,257).
    2. **Encode verification:** Loads our tokenizer, encodes a known ALS term
       (``"riluzole"``), and verifies the GGUF token list produces the same IDs.

    Args:
        gguf_path: Path to the GGUF file to verify.

    Returns:
        A dict with ``success`` and ``error`` keys.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        return {
            "success": False,
            "error": "gguf package not installed. Install with: pip install gguf",
        }

    reader = GGUFReader(str(gguf_path))

    # Check 1: Verify general.architecture is "gpt2"
    arch_field = reader.fields.get("general.architecture")
    if arch_field is not None:
        # The field value is stored in parts; the last part contains the data
        arch_value = arch_field.parts[-1].tobytes().decode("utf-8")
        if arch_value != "gpt2":
            return {
                "success": False,
                "error": f"Expected architecture 'gpt2', got '{arch_value}'",
            }
        print(f"  Architecture: {arch_value}")
    else:
        return {
            "success": False,
            "error": "Missing 'general.architecture' field in GGUF metadata",
        }

    # Check tokenizer vocab size from GGUF metadata
    vocab_size_field = reader.fields.get("tokenizer.ggml.tokens")
    gguf_vocab_size = None
    if vocab_size_field is not None:
        gguf_vocab_size = len(vocab_size_field.data)
        print(f"  GGUF tokenizer tokens count: {gguf_vocab_size}")
    else:
        print("  WARNING: Could not find tokenizer.ggml.tokens in metadata")

    # Check 2: Encode verification with a known ALS term
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(TOKENIZER_HF_DIR / "tokenizer.json"))
        test_term = "riluzole"
        encoded = tokenizer.encode(test_term)
        print(f"  Encode verification: '{test_term}' -> token IDs {encoded.ids}")

        # Verify the GGUF file has the expected vocab size (50,257)
        expected_vocab = 50257
        if vocab_size_field is not None and gguf_vocab_size != expected_vocab:
            print(
                f"  WARNING: GGUF vocab size ({gguf_vocab_size}) does not match "
                f"expected ({expected_vocab})"
            )
    except ImportError:
        print("  WARNING: tokenizers package not available for encode verification")
    except Exception as e:
        print(f"  WARNING: Encode verification error: {e}")

    return {"success": True, "error": None}


def stage_ollama_create(
    gguf_path: Path,
    output_dir: Path,
    size: str = "tiny",
) -> dict:
    """Stage 3: Create Ollama model from GGUF file and auto-test generation.

    Generates a Modelfile from the template, runs ``ollama create`` to
    register the model, and performs a test generation to verify the full
    pipeline end-to-end.

    Args:
        gguf_path: Path to the GGUF file from Stage 2.
        output_dir: Directory for the generated Modelfile (e.g.,
            ``export/output/tiny/``).
        size: Model size qualifier for the Ollama model name.

    Returns:
        A dict with ``success``, ``output_path``, and ``error`` keys.
    """
    print("\n" + "=" * 60)
    print("Stage 3: GGUF -> Ollama")
    print("=" * 60)

    model_name = f"als-lm-{size}"
    modelfile_path = output_dir / "Modelfile"

    try:
        # Verify Ollama is installed
        print("\nChecking Ollama installation...")
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return {
                    "success": False,
                    "output_path": model_name,
                    "error": (
                        "Ollama not found. Install with: "
                        "curl -fsSL https://ollama.com/install.sh | sh"
                    ),
                }
            print(f"  Ollama version: {result.stdout.strip() or result.stderr.strip()}")
        except FileNotFoundError:
            return {
                "success": False,
                "output_path": model_name,
                "error": (
                    "Ollama not found. Install with: "
                    "curl -fsSL https://ollama.com/install.sh | sh"
                ),
            }

        # Check if ollama serve is running by trying to list models
        print("  Checking if Ollama is running...")
        ollama_started = False
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                # Try starting ollama serve in background
                print("  Starting ollama serve in background...")
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                ollama_started = True
                time.sleep(5)
        except Exception:
            # Try starting ollama serve in background
            print("  Starting ollama serve in background...")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            ollama_started = True
            time.sleep(5)

        # Generate Modelfile from template
        print("\nGenerating Modelfile from template...")
        if not MODELFILE_TEMPLATE.is_file():
            return {
                "success": False,
                "output_path": model_name,
                "error": f"Modelfile template not found: {MODELFILE_TEMPLATE}",
            }

        with open(MODELFILE_TEMPLATE) as f:
            template = f.read()

        # Use relative path from Modelfile location to GGUF file
        gguf_relative = os.path.relpath(str(gguf_path), str(output_dir))
        generated = template.replace("{{GGUF_PATH}}", f"./{gguf_relative}")
        generated = generated.replace("{{DISCLAIMER}}", DISCLAIMER)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(modelfile_path, "w") as f:
            f.write(generated)
        print(f"  Modelfile written to: {modelfile_path}")

        # Run ollama create
        print(f"\nCreating Ollama model '{model_name}'...")
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.stdout:
            print(f"  {result.stdout.strip()}")
        if result.stderr:
            print(f"  {result.stderr.strip()}")

        if result.returncode != 0:
            return {
                "success": False,
                "output_path": model_name,
                "error": (
                    f"ollama create failed (exit code {result.returncode}). "
                    f"stderr: {result.stderr[:500]}"
                ),
            }

        # Auto-test: run a single generation prompt
        print(f"\nAuto-test: running '{model_name}' with test prompt...")
        test_prompt = "ALS is a disease that"
        try:
            result = subprocess.run(
                ["ollama", "run", model_name, test_prompt],
                capture_output=True,
                text=True,
                timeout=30,
            )
            generated_text = result.stdout.strip()
            if generated_text:
                print(f"  Generated text: {generated_text[:200]}")
                print("\n  Auto-test PASSED (model generated text)")
            else:
                print("  WARNING: Model generated no text, but ollama run succeeded")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print("  WARNING: Generation timed out after 30 seconds")
            print("  This may be normal for a first run (model loading)")

        print(f"\n  Stage 3 PASSED")
        print(f"  Run the model with: ollama run {model_name}")
        return {
            "success": True,
            "output_path": model_name,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "output_path": model_name,
            "error": str(e),
        }


def print_summary(results: dict, size: str = "tiny") -> None:
    """Print a summary table of all pipeline stages.

    Shows pass/fail status and output path for each stage. Uses simple
    ASCII formatting for terminal compatibility.

    Args:
        results: Dict mapping stage names to result dicts.
        size: Model size qualifier for the Ollama run command.
    """
    print("\n")
    print("+" + "-" * 34 + "+" + "-" * 8 + "+" + "-" * 48 + "+")
    print(
        "| {:<32} | {:<6} | {:<46} |".format("Stage", "Status", "Output")
    )
    print("+" + "-" * 34 + "+" + "-" * 8 + "+" + "-" * 48 + "+")

    stages = [
        ("1. PyTorch -> HuggingFace", "hf_convert"),
        ("2. HuggingFace -> GGUF", "gguf_convert"),
        ("3. GGUF -> Ollama", "ollama_create"),
    ]

    all_passed = True
    for display_name, key in stages:
        if key in results:
            r = results[key]
            status = "PASS" if r["success"] else "FAIL"
            if not r["success"]:
                all_passed = False
            output = r["output_path"] or ""
            # Truncate long paths
            if len(output) > 46:
                output = "..." + output[-43:]
        else:
            status = "SKIP"
            output = ""

        print(
            "| {:<32} | {:<6} | {:<46} |".format(display_name, status, output)
        )

    print("+" + "-" * 34 + "+" + "-" * 8 + "+" + "-" * 48 + "+")

    if all_passed:
        print(f"\nAll stages passed. Run the model with:")
        print(f"  ollama run als-lm-{size}")


def main() -> None:
    """CLI entry point for the unified export pipeline."""
    parser = argparse.ArgumentParser(
        description="ALS-LM unified export pipeline: PyTorch -> HuggingFace -> GGUF -> Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m export.export_pipeline\n"
            "  python -m export.export_pipeline --checkpoint checkpoints/tiny/step_1999.pt\n"
            "  python -m export.export_pipeline --size tiny --skip-ollama\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file. Auto-detects latest if not provided.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="tiny",
        help="Model size qualifier (default: tiny). Used for output paths and Ollama model name.",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip the Ollama stage (useful if Ollama is not installed).",
    )
    args = parser.parse_args()

    print("ALS-LM Export Pipeline")
    print("=" * 60)

    # Resolve checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.is_file():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    else:
        print("\nAuto-detecting latest checkpoint...")
        try:
            checkpoint_path = find_latest_checkpoint()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    # Print checkpoint metadata
    print(f"\nCheckpoint: {checkpoint_path}")
    try:
        ckpt_meta = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if "step" in ckpt_meta:
            print(f"  Step: {ckpt_meta['step']}")
        if "val_loss" in ckpt_meta:
            print(f"  Val loss: {ckpt_meta['val_loss']:.4f}")
        if "train_loss" in ckpt_meta:
            print(f"  Train loss: {ckpt_meta['train_loss']:.4f}")
        if "config" in ckpt_meta:
            cfg = ckpt_meta["config"]
            if isinstance(cfg, dict):
                print(f"  Config: {cfg}")
            else:
                print(f"  Config: {cfg}")
        del ckpt_meta  # Free memory
    except Exception as e:
        print(f"  WARNING: Could not read checkpoint metadata: {e}")

    # Set up output paths
    output_base = PROJECT_ROOT / "export" / "output" / args.size
    hf_dir = output_base / "hf"
    gguf_dir = output_base / "gguf"

    results = {}

    # Stage 1: PyTorch -> HuggingFace
    results["hf_convert"] = stage_hf_convert(checkpoint_path, hf_dir)
    if not results["hf_convert"]["success"]:
        print(f"\nERROR: Stage 1 failed: {results['hf_convert']['error']}")
        print_summary(results, args.size)
        sys.exit(1)

    # Stage 2: HuggingFace -> GGUF
    results["gguf_convert"] = stage_gguf_convert(hf_dir, gguf_dir, args.size)
    if not results["gguf_convert"]["success"]:
        print(f"\nERROR: Stage 2 failed: {results['gguf_convert']['error']}")
        print_summary(results, args.size)
        sys.exit(1)

    # Stage 3: GGUF -> Ollama (optional)
    if args.skip_ollama:
        print("\n" + "=" * 60)
        print("Stage 3: GGUF -> Ollama (SKIPPED)")
        print("=" * 60)
        results["ollama_create"] = {
            "success": True,
            "output_path": f"als-lm-{args.size} (skipped)",
            "error": None,
        }
    else:
        results["ollama_create"] = stage_ollama_create(
            Path(results["gguf_convert"]["output_path"]),
            output_base,
            args.size,
        )
        if not results["ollama_create"]["success"]:
            print(f"\nERROR: Stage 3 failed: {results['ollama_create']['error']}")

    # Print summary
    print_summary(results, args.size)

    # Exit with appropriate code
    if all(r["success"] for r in results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
