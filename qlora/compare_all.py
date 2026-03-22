#!/usr/bin/env python3
"""Compare hallucination evaluation results across all 6 model variants.

Standalone script that reads Q8_0 evaluation results for:
  1. From-scratch 500M model
  2. From-scratch 1B base model
  3. GPT-2 large fine-tuned model
  4. From-scratch 1B instruction-tuned model
  5. Llama 3.2 1B Instruct (base ablation)
  6. Llama 3.2 1B QLoRA (domain-adapted)

Produces a Markdown report with grouped bar chart, JSON artifact, and
analysis organized by approach family (from-scratch, pretrained fine-tune,
pretrained instruct).

This is a research analysis tool, not a medical information system.

Usage examples::

    # Default: write reports to reports/
    python qlora/compare_all.py

    # Custom output directory
    python qlora/compare_all.py --output-dir /tmp/reports

    # Verbose output
    python qlora/compare_all.py --verbose
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (stubs -- will be populated in GREEN phase)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

_MODEL_SUBDIRS = {}  # Empty stub -- tests expect 6 entries

_MODEL_ORDER = []  # Empty stub -- tests expect 6 entries

_MODEL_LABELS = {}  # Empty stub -- tests expect 6 entries

_FAMILY_MAP = {}  # Empty stub -- tests expect 6 entries


# ---------------------------------------------------------------------------
# Functions (stubs -- will be implemented in GREEN phase)
# ---------------------------------------------------------------------------

def load_model_data(model_dir: Path) -> dict:
    """Load all JSON result files for a single model."""
    raise NotImplementedError("Stub -- implement in GREEN phase")


def compute_capability_gap(all_data: dict) -> dict:
    """Compute the perceived capability gap for each model."""
    raise NotImplementedError("Stub -- implement in GREEN phase")


def compute_ablation_delta(qlora_data: dict, baseline_data: dict) -> dict:
    """Compute delta between QLoRA and ablation baseline."""
    raise NotImplementedError("Stub -- implement in GREEN phase")
