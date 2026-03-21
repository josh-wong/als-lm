#!/usr/bin/env python3
"""Compare hallucination evaluation results across four models.

Standalone script that reads Q8_0 evaluation results for:
  1. From-scratch 500M model
  2. Fine-tuned GPT-2 large model
  3. From-scratch 1B base model
  4. From-scratch 1B instruction-tuned model

Produces a dual-format comparison report (Markdown + JSON) analyzing how
model architecture, scale, pre-training, and instruction tuning affect
accuracy, hallucination behavior, failure mode distribution, and the
perceived capability gap (coherence% minus accuracy%).

This is a research analysis tool, not a medical information system.

Usage examples::

    # Default: write reports to reports/
    python scripts/compare_models.py

    # Custom output directory
    python scripts/compare_models.py --output-dir /tmp/reports

    # Verbose output
    python scripts/compare_models.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Default result directory names (Q8_0 representative).
# The 1B base subdir contains a timestamp from the training run; override
# with --model-dir scratch_1b_base=<new_subdir> if the base model is retrained.
_DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"
_MODEL_SUBDIRS = {
    "scratch_500m": "als-lm-500m_q8_0",
    "gpt2_large_finetune": "als-lm-gpt2-large_q8_0",
    "scratch_1b_base": "1B_20260317_205331",
    "scratch_1b_instruct": "als-lm-1b-instruct_q8_0",
}
_MODEL_LABELS = {
    "scratch_500m": ("ALS-LM 500M (from-scratch)", "500M"),
    "gpt2_large_finetune": ("GPT-2 large (fine-tuned)", "GPT-2 large"),
    "scratch_1b_base": ("ALS-LM 1B (from-scratch base)", "1B base"),
    "scratch_1b_instruct": ("ALS-LM 1B (instruction-tuned)", "1B instruct"),
}

# Canonical display order for the 4 models
_MODEL_ORDER = [
    "scratch_500m",
    "gpt2_large_finetune",
    "scratch_1b_base",
    "scratch_1b_instruct",
]


def _build_model_configs(results_dir: Path) -> list:
    """Build MODEL_CONFIGS list from a results directory."""
    return [
        {
            "key": key,
            "label": _MODEL_LABELS[key][0],
            "short_label": _MODEL_LABELS[key][1],
            "path": results_dir / subdir,
        }
        for key, subdir in _MODEL_SUBDIRS.items()
    ]

# All taxonomy modes in display order
TAXONOMY_MODES = [
    "confident_fabrication",
    "plausible_blending",
    "outdated_information",
    "boundary_confusion",
    "accurate_but_misleading",
    "accurate",
    "degenerate",
]

# All 8 evaluation categories
CATEGORIES = [
    "clinical_trials",
    "diagnostic_criteria",
    "disease_mechanisms",
    "drug_treatment",
    "epidemiology",
    "gene_mutation",
    "patient_care",
    "temporal_accuracy",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_data(model_dir: Path) -> dict:
    """Load all JSON result files for a single model."""
    data = {}
    for filename in ["scores.json", "fabrications.json", "taxonomy.json", "responses.json"]:
        filepath = model_dir / filename
        if not filepath.is_file():
            print(f"ERROR: Missing file: {filepath}", file=sys.stderr)
            sys.exit(1)
        with open(filepath, "r", encoding="utf-8") as f:
            data[filename.replace(".json", "")] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(data: dict, keys: list, model_key: str):
    """Navigate nested dict keys, raising KeyError with model context on failure."""
    current = data
    for k in keys:
        try:
            current = current[k]
        except (KeyError, TypeError):
            path = " -> ".join(str(p) for p in keys)
            raise KeyError(
                f"Missing key path [{path}] in {model_key} data (failed at '{k}')"
            )
    return current


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_accuracy_comparison(all_data: dict) -> dict:
    """Extract mean_accuracy, binary_pass_rate, hedging for each model."""
    result = {}
    for key, data in all_data.items():
        overall = _safe_get(data, ["scores", "aggregate", "overall"], key)
        result[key] = {
            "mean_accuracy": overall["mean_accuracy"],
            "binary_pass_rate": overall.get("binary_pass_rate", 0),
            "hedging": overall.get("total_hedging_instances", 0),
        }
    return result


def compute_per_category_accuracy(all_data: dict) -> dict:
    """Extract per-category accuracy for each model."""
    result = {}
    for cat in CATEGORIES:
        result[cat] = {}
        for key, data in all_data.items():
            by_cat = _safe_get(data, ["scores", "aggregate", "by_category"], key)
            cat_data = by_cat.get(cat, {})
            result[cat][key] = cat_data.get("mean_accuracy", 0.0)
    return result


def compute_taxonomy_comparison(all_data: dict) -> dict:
    """Extract taxonomy distribution counts and percentages for each model."""
    result = {}
    for key, data in all_data.items():
        dist = _safe_get(data, ["taxonomy", "distribution"], key)
        result[key] = {}
        for mode in TAXONOMY_MODES:
            mode_data = dist.get(mode, {"count": 0, "pct": 0.0})
            result[key][mode] = {
                "count": mode_data["count"],
                "pct": mode_data["pct"],
            }
    return result


def compute_degenerate_rates(all_data: dict) -> dict:
    """Derive non-degenerate count and rate from taxonomy distribution."""
    result = {}
    for key, data in all_data.items():
        dist = _safe_get(data, ["taxonomy", "distribution"], key)
        total = sum(mode_data["count"] for mode_data in dist.values())
        degenerate = dist.get("degenerate", {}).get("count", 0)
        non_degenerate = total - degenerate
        result[key] = {
            "total": total,
            "degenerate": degenerate,
            "non_degenerate": non_degenerate,
            "non_degenerate_rate": non_degenerate / total if total > 0 else 0.0,
        }
    return result


def compute_fabrication_comparison(all_data: dict) -> dict:
    """Compute fabrication rates overall and among non-degenerate responses only.

    The per-question fabrication data is filtered using the is_coherent field
    from responses.json to compute the non-degenerate-only fabrication rate.
    """
    result = {}
    for key, data in all_data.items():
        summary = _safe_get(data, ["fabrications", "summary"], key)

        # Build is_coherent lookup from responses.json
        responses_list = _safe_get(data, ["responses", "responses"], key)
        coherent_lookup = {
            r["question_id"]: r.get("is_coherent", True)
            for r in responses_list
        }

        # Per-question fabrication data filtered to coherent responses only
        coherent_extracted = 0
        coherent_flagged = 0
        per_question = _safe_get(data, ["fabrications", "per_question"], key)
        for pq in per_question:
            if coherent_lookup.get(pq["question_id"], True):
                coherent_extracted += len(pq["entities_extracted"])
                coherent_flagged += len(pq["flagged_entities"])

        coherent_count = sum(1 for v in coherent_lookup.values() if v)

        result[key] = {
            "overall_extracted": summary["total_entities_extracted"],
            "overall_flagged": summary["total_flagged"],
            "overall_rate": summary["flagged_rate"],
            "non_degenerate_extracted": coherent_extracted,
            "non_degenerate_flagged": coherent_flagged,
            "non_degenerate_rate": (
                coherent_flagged / coherent_extracted
                if coherent_extracted > 0 else 0.0
            ),
            "non_degenerate_count": coherent_count,
        }
    return result


def compute_capability_gap(all_data: dict) -> dict:
    """Compute the perceived capability gap for each model.

    The capability gap is defined as coherence% minus accuracy%. A model that
    produces coherent-sounding output but scores poorly on factual accuracy
    has a high gap, indicating ethical risk (users may trust plausible-sounding
    but incorrect answers). A model that is entirely degenerate has gap = 0%
    because 0% - 0% = 0% -- the ethical risk did not materialize because the
    output is obviously broken.

    Parameters
    ----------
    all_data : dict
        Mapping of model key to loaded evaluation data (scores, responses, etc.).

    Returns
    -------
    dict
        Mapping of model key to ``{coherence_pct, accuracy_pct, gap_pct}``.
    """
    result = {}
    for key, data in all_data.items():
        responses_list = _safe_get(data, ["responses", "responses"], key)
        total = len(responses_list)
        coherent_count = sum(
            1 for r in responses_list if r.get("is_coherent", True)
        )
        coherence_pct = (coherent_count / total * 100) if total > 0 else 0.0
        accuracy_pct = (
            _safe_get(data, ["scores", "aggregate", "overall", "mean_accuracy"], key)
            * 100
        )
        gap = coherence_pct - accuracy_pct
        result[key] = {
            "coherence_pct": round(coherence_pct, 2),
            "accuracy_pct": round(accuracy_pct, 2),
            "gap_pct": round(gap, 2),
        }
    return result


def select_qualitative_samples(all_data: dict) -> list:
    """Select 3-5 diverse question pairs showing different comparison patterns.

    Selection strategy for 4 models:
    (a) One where the instruct model is degenerate but 500M or 1B base is coherent
    (b) One where GPT-2 large is coherent with different taxonomy from others
    (c) One where all models are degenerate
    (d) One where 500M is coherent but GPT-2 large is degenerate (original pattern)

    Falls back to 2-model patterns when 4-model data is not all available.
    """
    model_keys = [k for k in _MODEL_ORDER if k in all_data]

    # Build per-model response and taxonomy lookups
    resp_lookup = {}
    tax_lookup = {}
    for mk in model_keys:
        resp_lookup[mk] = {
            r["question_id"]: r for r in all_data[mk]["responses"]["responses"]
        }
        tax_lookup[mk] = {
            e["question_id"]: e["primary_mode"]
            for e in all_data[mk]["taxonomy"]["per_question"]
        }

    # Find question IDs common to all models
    if not model_keys:
        return []
    common_qids = set(resp_lookup[model_keys[0]].keys())
    for mk in model_keys[1:]:
        common_qids &= set(resp_lookup[mk].keys())
    all_qids = sorted(common_qids)

    samples = []

    # Helper to build a sample entry
    def _build_sample(pattern: str, qid: str) -> dict:
        first_mk = model_keys[0]
        entry = {
            "pattern": pattern,
            "question_id": qid,
            "category": resp_lookup[first_mk][qid]["category"],
            "prompt": resp_lookup[first_mk][qid]["prompt"],
        }
        for mk in model_keys:
            short = _MODEL_LABELS.get(mk, (mk, mk))[1]
            entry[f"{mk}_response"] = resp_lookup[mk][qid]["response"][:300]
            entry[f"{mk}_taxonomy"] = tax_lookup[mk].get(qid, "unknown")
            entry[f"{mk}_label"] = short
        return entry

    # (a) Instruct model degenerate while 500M or 1B base is coherent
    if "scratch_1b_instruct" in model_keys:
        for qid in all_qids:
            instruct_coherent = resp_lookup["scratch_1b_instruct"][qid].get("is_coherent", True)
            if instruct_coherent:
                continue
            # Check if any other model is coherent for this question
            for mk in model_keys:
                if mk == "scratch_1b_instruct":
                    continue
                if resp_lookup[mk][qid].get("is_coherent", True):
                    samples.append(_build_sample("instruct_degenerate_others_coherent", qid))
                    break
            if samples:
                break

    # (b) GPT-2 large coherent with different taxonomy from 500M
    if "gpt2_large_finetune" in model_keys and "scratch_500m" in model_keys:
        for qid in all_qids:
            if qid in {s["question_id"] for s in samples}:
                continue
            f_coh = resp_lookup["gpt2_large_finetune"][qid].get("is_coherent", True)
            s_coh = resp_lookup["scratch_500m"][qid].get("is_coherent", True)
            if f_coh and s_coh:
                s_mode = tax_lookup["scratch_500m"].get(qid, "unknown")
                f_mode = tax_lookup["gpt2_large_finetune"].get(qid, "unknown")
                if s_mode != f_mode:
                    samples.append(_build_sample("both_coherent_different_taxonomy", qid))
                    break

    # (c) All models degenerate
    for qid in all_qids:
        if qid in {s["question_id"] for s in samples}:
            continue
        all_degenerate = all(
            not resp_lookup[mk][qid].get("is_coherent", True)
            for mk in model_keys
        )
        if all_degenerate:
            samples.append(_build_sample("all_degenerate", qid))
            break

    # (d) 500M coherent, GPT-2 large degenerate
    if "scratch_500m" in model_keys and "gpt2_large_finetune" in model_keys:
        seen_categories = {s.get("category") for s in samples}
        for qid in all_qids:
            if qid in {s["question_id"] for s in samples}:
                continue
            cat = resp_lookup["scratch_500m"][qid]["category"]
            if cat in seen_categories:
                continue
            s_coh = resp_lookup["scratch_500m"][qid].get("is_coherent", True)
            f_coh = resp_lookup["gpt2_large_finetune"][qid].get("is_coherent", True)
            if s_coh and not f_coh:
                samples.append(_build_sample("scratch_coherent_finetune_degenerate", qid))
                seen_categories.add(cat)
                if len(samples) >= 5:
                    break

    return samples


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(
    accuracy: dict,
    per_category: dict,
    taxonomy: dict,
    degenerate: dict,
    fabrication: dict,
    samples: list,
    model_configs: list | None = None,
    capability_gap: dict | None = None,
) -> str:
    """Build the full Markdown comparison report for N models."""
    lines = []

    def add(text: str = "") -> None:
        lines.append(text)

    _configs = model_configs or []
    configs = {c["key"]: c for c in _configs}
    model_keys = [c["key"] for c in _configs]

    # Section 1: Title
    n_models = len(model_keys)
    add(f"# Model comparison report: {n_models}-model cross-comparison")
    add()

    # Section 2: TL;DR
    add(
        f"Cross-comparison of {n_models} model variants evaluated on the "
        "160-question ALS hallucination benchmark using Q8_0 quantization "
        "as the representative level. The table below summarizes each "
        "model's accuracy and degenerate output rate."
    )
    add()

    # Compact summary table
    add(f"| {'Model':<35} | {'Accuracy':>10} | {'Non-degenerate':>16} | {'Degenerate':>12} |")
    add(f"| {'-' * 35} | {'-' * 10} | {'-' * 16} | {'-' * 12} |")
    for mk in model_keys:
        label = configs[mk]["short_label"]
        acc = accuracy[mk]["mean_accuracy"] * 100
        nd_rate = degenerate[mk]["non_degenerate_rate"] * 100
        d_count = degenerate[mk]["degenerate"]
        total = degenerate[mk]["total"]
        add(
            f"| {label:<35} | {acc:>9.2f}% | {nd_rate:>15.1f}% | {d_count:>4}/{total:<6} |"
        )
    add()

    # Key finding paragraph
    gpt2_acc = accuracy.get("gpt2_large_finetune", {}).get("mean_accuracy", 0) * 100
    s500_acc = accuracy.get("scratch_500m", {}).get("mean_accuracy", 0) * 100
    instruct_deg = degenerate.get("scratch_1b_instruct", {}).get("degenerate", 0)
    instruct_total = degenerate.get("scratch_1b_instruct", {}).get("total", 0)
    base_1b_deg = degenerate.get("scratch_1b_base", {}).get("non_degenerate_rate", 0) * 100

    add(
        f"The fine-tuned GPT-2 large model remains the highest-accuracy variant "
        f"at {gpt2_acc:.2f}%, while both 1B models (base and instruction-tuned) "
        f"achieve 0.00% accuracy. The instruction-tuned model produces "
        f"{instruct_deg}/{instruct_total} degenerate responses (100%), "
        f"worse than the 1B base model's {base_1b_deg:.1f}% non-degenerate rate. "
        f"Scaling from 500M to 1B parameters without pre-trained knowledge does "
        f"not improve accuracy. Instruction tuning the from-scratch 1B model "
        f"with ~970 pairs causes complete output collapse rather than "
        f"surfacing internalized knowledge."
    )
    add()

    # Section 3: Overall accuracy comparison
    add("## Overall accuracy comparison")
    add()
    add(
        "Mean accuracy, binary pass rate, and hedging instances for each "
        "model, evaluated on the 160-question ALS hallucination benchmark."
    )
    add()

    # Dynamic column headers
    col_labels = [configs[mk]["short_label"] for mk in model_keys]
    col_widths = [max(len(lbl), 14) for lbl in col_labels]

    header = f"| {'Metric':<25} |"
    separator = f"| {'-' * 25} |"
    for lbl, w in zip(col_labels, col_widths):
        header += f" {lbl:>{w}} |"
        separator += f" {'-' * w} |"
    add(header)
    add(separator)

    # Mean accuracy row
    row = f"| {'Mean accuracy':<25} |"
    for mk, w in zip(model_keys, col_widths):
        val = accuracy[mk]["mean_accuracy"] * 100
        row += f" {val:>{w - 1}.2f}% |"
    add(row)

    # Binary pass rate row
    row = f"| {'Binary pass rate':<25} |"
    for mk, w in zip(model_keys, col_widths):
        val = accuracy[mk]["binary_pass_rate"] * 100
        row += f" {val:>{w - 1}.2f}% |"
    add(row)

    # Hedging instances row
    row = f"| {'Hedging instances':<25} |"
    for mk, w in zip(model_keys, col_widths):
        val = accuracy[mk]["hedging"]
        row += f" {val:>{w}} |"
    add(row)
    add()

    # Section 4: Per-category accuracy
    add("## Per-category accuracy")
    add()
    add(
        "Mean accuracy broken down by the 8 evaluation categories. All "
        "from-scratch models (500M, 1B base, 1B instruct) score near zero "
        "across all categories, while GPT-2 large shows modest variation."
    )
    add()

    header = f"| {'Category':<25} |"
    separator = f"| {'-' * 25} |"
    for lbl, w in zip(col_labels, col_widths):
        header += f" {lbl:>{w}} |"
        separator += f" {'-' * w} |"
    add(header)
    add(separator)

    for cat in CATEGORIES:
        display = cat.replace("_", " ").title()
        row = f"| {display:<25} |"
        for mk, w in zip(model_keys, col_widths):
            val = per_category[cat].get(mk, 0.0) * 100
            row += f" {val:>{w - 1}.2f}% |"
        add(row)
    add()

    # Section 5: Taxonomy distribution
    add("## Taxonomy distribution")
    add()
    add(
        "Distribution of failure modes across the 7 taxonomy categories "
        f"for all {n_models} models. The 1B instruct model shows 100% "
        "degenerate output, while the 1B base model produces a mix of "
        "degenerate and non-degenerate failure modes similar to the 500M "
        "model's pattern."
    )
    add()

    header = f"| {'Failure mode':<28} |"
    separator = f"| {'-' * 28} |"
    for lbl in col_labels:
        w = max(len(lbl), 8)
        header += f" {lbl + ' (n)':>{w + 4}} | {lbl + ' (%)':>{w + 4}} |"
        separator += f" {'-' * (w + 4)} | {'-' * (w + 4)} |"
    add(header)
    add(separator)

    for mode in TAXONOMY_MODES:
        display = mode.replace("_", " ").title()
        row = f"| {display:<28} |"
        for mk in model_keys:
            lbl = configs[mk]["short_label"]
            w = max(len(lbl), 8)
            mode_data = taxonomy[mk].get(mode, {"count": 0, "pct": 0.0})
            row += f" {mode_data['count']:>{w + 4}} | {mode_data['pct']:>{w + 3}.1f}% |"
        add(row)
    add()

    # Section 6: Degenerate output analysis
    add("## Degenerate output analysis")
    add()

    add(f"| {'Model':<35} | {'Non-degenerate':>16} | {'Degenerate':>12} | {'Non-deg rate':>14} |")
    add(f"| {'-' * 35} | {'-' * 16} | {'-' * 12} | {'-' * 14} |")
    for mk in model_keys:
        label = configs[mk]["short_label"]
        d = degenerate[mk]
        add(
            f"| {label:<35} | {d['non_degenerate']:>16} | {d['degenerate']:>12} | {d['non_degenerate_rate'] * 100:>13.1f}% |"
        )
    add()

    add(
        "The 1B instruction-tuned model produces 160/160 degenerate responses, "
        "worse than any other model. The instruction tuning process caused the "
        "model to collapse into repeating the most common English token "
        "('TheTheThe...'). By contrast, the 1B base model without SFT produces "
        f"{degenerate.get('scratch_1b_base', {}).get('non_degenerate', 0)} "
        "non-degenerate responses, demonstrating that the SFT process itself "
        "degraded the model's output diversity rather than improving it."
    )
    add()

    # Section 7: Fabrication rate comparison
    add("## Fabrication rate comparison")
    add()
    add(
        "Entity-level fabrication analysis. Models with 0 non-degenerate "
        "responses extract 0 entities from coherent output, so their "
        "non-degenerate fabrication rate is 0.00% by definition."
    )
    add()

    add(f"| {'Metric':<40} |")
    header_suffix = ""
    sep_suffix = ""
    for mk in model_keys:
        lbl = configs[mk]["short_label"]
        w = max(len(lbl), 14)
        header_suffix += f" {lbl:>{w}} |"
        sep_suffix += f" {'-' * w} |"
    add(f"| {'Metric':<40} |{header_suffix}")
    add(f"| {'-' * 40} |{sep_suffix}")

    metrics = [
        ("Total entities extracted", lambda mk: f"{fabrication[mk]['overall_extracted']}"),
        ("Total entities flagged", lambda mk: f"{fabrication[mk]['overall_flagged']}"),
        ("Overall fabrication rate", lambda mk: f"{fabrication[mk]['overall_rate'] * 100:.2f}%"),
        ("Non-degenerate responses", lambda mk: f"{fabrication[mk]['non_degenerate_count']}"),
        ("Entities from non-degenerate", lambda mk: f"{fabrication[mk]['non_degenerate_extracted']}"),
        ("Flagged from non-degenerate", lambda mk: f"{fabrication[mk]['non_degenerate_flagged']}"),
        ("Fabrication rate (non-deg)", lambda mk: f"{fabrication[mk]['non_degenerate_rate'] * 100:.2f}%"),
    ]
    for metric_name, fmt_fn in metrics:
        row = f"| {metric_name:<40} |"
        for mk in model_keys:
            lbl = configs[mk]["short_label"]
            w = max(len(lbl), 14)
            val_str = fmt_fn(mk)
            row += f" {val_str:>{w}} |"
        add(row)
    add()

    # Section 8: Qualitative sample pairs
    add("## Qualitative sample pairs")
    add()
    add(
        f"Selected examples comparing how all {n_models} models respond to the "
        "same question. These examples illustrate patterns including "
        "instruct-model degenerate output alongside other models' responses, "
        "and cases where all models are degenerate. "
        "Response text is truncated to 300 characters."
    )
    add()

    pattern_labels = {
        "both_coherent_different_taxonomy": "Both coherent, different taxonomy",
        "both_degenerate": "Both degenerate",
        "all_degenerate": "All models degenerate",
        "scratch_coherent_finetune_degenerate": "500M coherent, GPT-2 large degenerate",
        "instruct_degenerate_others_coherent": "Instruct degenerate, others coherent",
    }

    for i, sample in enumerate(samples, 1):
        pattern = pattern_labels.get(sample["pattern"], sample["pattern"].replace("_", " "))
        add(f"### Example {i}: {sample['question_id']} ({pattern})")
        add()
        add(f"**Category:** {sample['category'].replace('_', ' ')}")
        add()
        add(f"**Prompt:** {sample['prompt']}")
        add()

        for mk in model_keys:
            resp_key = f"{mk}_response"
            tax_key = f"{mk}_taxonomy"
            label_key = f"{mk}_label"
            if resp_key not in sample:
                continue
            short_label = sample.get(label_key, mk)
            tax_mode = sample.get(tax_key, "unknown").replace("_", " ")
            resp_clean = " ".join(sample[resp_key].split())[:300]
            add(f"**{short_label} response** (taxonomy: {tax_mode}):")
            add()
            add(f"> {resp_clean}")
            add()

    # Section 9: Instruction-following limitation
    add("## Instruction-following limitation")
    add()
    add(
        "Three of the four models in this comparison lack instruction-tuning "
        "on a pre-trained base: the 500M from-scratch, the 1B base, and the "
        "GPT-2 large (fine-tuned on ALS text but not on Q&A pairs). The fourth "
        "model (1B instruct) received supervised fine-tuning on ~970 ALS Q&A "
        "pairs, but the from-scratch base model lacked sufficient internalized "
        "knowledge for SFT to surface. The result was complete output collapse "
        "rather than structured question answering."
    )
    add()
    f_deg_rate = degenerate.get("gpt2_large_finetune", {}).get("non_degenerate_rate", 0)
    f_nondeg_n = degenerate.get("gpt2_large_finetune", {}).get("non_degenerate", 0)
    add(
        f"The GPT-2 large model's {(1 - f_deg_rate) * 100:.1f}% degenerate rate "
        f"reflects the instruction-following limitation of a completion model. "
        f"The 1B instruct model's 100% degenerate rate shows that SFT on a "
        f"from-scratch model with insufficient data makes instruction following "
        f"worse, not better. The v1 research paper (Section 9.2) recommended "
        f"'a more modern instruction-capable base model' as the next step, and "
        f"this SFT failure validates that recommendation."
    )
    add()

    # Section 10: Perceived capability gap
    if capability_gap:
        add("## Perceived capability gap")
        add()
        add(
            "The perceived capability gap measures the difference between how "
            "coherent a model's output appears (coherence%) and how factually "
            "accurate it is (accuracy%). A high gap indicates ethical risk: users "
            "may trust plausible-sounding but incorrect answers. A gap of 0% for "
            "degenerate models means the ethical risk did not materialize because "
            "the output is obviously broken."
        )
        add()
        add(f"| {'Model':<35} | {'Coherence':>12} | {'Accuracy':>10} | {'Gap':>8} |")
        add(f"| {'-' * 35} | {'-' * 12} | {'-' * 10} | {'-' * 8} |")
        for mk in model_keys:
            label = configs[mk]["short_label"]
            cg = capability_gap.get(mk, {})
            coh = cg.get("coherence_pct", 0.0)
            acc = cg.get("accuracy_pct", 0.0)
            gap = cg.get("gap_pct", 0.0)
            add(f"| {label:<35} | {coh:>11.1f}% | {acc:>9.2f}% | {gap:>7.1f}% |")
        add()
        add(
            "The 1B instruct and 1B base models (0% and near-0% accuracy) both "
            "show near-zero or zero capability gap. The 500M model has the highest "
            "gap due to producing many coherent-sounding but inaccurate responses. "
            "The GPT-2 large model's gap is limited by its high degenerate rate. "
            "For models that are fully degenerate, the gap is 0% -- the ethical "
            "concern of confident-but-wrong output does not apply when output is "
            "obviously broken."
        )
        add()

    # Section 11: Caveats and limitations
    add("## Caveats and limitations")
    add()
    add(
        "**General knowledge confound.** The GPT-2 large model was "
        "pretrained on WebText, which includes diverse English text that may "
        "contain biomedical content. When the fine-tuned model produces a "
        "correct answer, we cannot determine whether the knowledge comes from "
        "(a) the ALS-specific fine-tuning on 143M tokens, or (b) general "
        "biomedical knowledge retained from WebText pretraining."
    )
    add()
    add(
        f"**Limited coherent sample size.** With only {f_nondeg_n} "
        f"non-degenerate responses from the GPT-2 large model, per-response "
        f"metrics are computed over an extremely small sample."
    )
    add()
    add(
        "**Checkpoint inference for 1B base.** The 1B base model was "
        "evaluated via direct checkpoint inference rather than through Ollama "
        "GGUF quantization. While this does not affect accuracy scoring, it "
        "means the 1B base results use a different inference path than the "
        "other three models."
    )
    add()
    add(
        "**Single quantization level.** This comparison uses Q8_0 as the "
        "representative quantization level, based on cross-quantization "
        "analysis showing that quantization does not meaningfully affect "
        "evaluation results."
    )
    add()

    # Section 12: Summary
    add("## Summary")
    add()
    add(
        f"The {n_models}-model comparison tells a consistent story: "
        "pre-trained knowledge (GPT-2 large) helps far more than model scale "
        "(500M to 1B) or instruction tuning on a from-scratch model. "
        f"The GPT-2 large model achieves {gpt2_acc:.2f}% accuracy despite "
        f"producing mostly degenerate output. Both 1B variants (base and "
        f"instruction-tuned) achieve 0.00% accuracy, demonstrating that "
        f"doubling model size without pre-trained knowledge provides no "
        f"benefit. The instruction-tuned model's complete output collapse "
        f"(100% degenerate) is the strongest evidence that SFT on a "
        f"from-scratch model with ~970 instruction pairs and ~153M training "
        f"tokens is insufficient -- the model lacks the foundational language "
        f"understanding needed for instruction following."
    )
    add()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json_output(
    accuracy: dict,
    per_category: dict,
    taxonomy: dict,
    degenerate: dict,
    fabrication: dict,
    samples: list,
    model_configs: list | None = None,
    capability_gap: dict | None = None,
) -> dict:
    """Build the structured JSON comparison output.

    The three metric groups for the figure (accuracy, non_degenerate_rate,
    fabrication_rate_non_degenerate) are directly extractable without
    recomputation. The capability_gap field contains coherence-minus-accuracy
    for each model.
    """
    _configs = model_configs or []
    result = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models_compared": [c["key"] for c in _configs],
            "model_labels": {c["key"]: c["label"] for c in _configs},
            "model_short_labels": {c["key"]: c["short_label"] for c in _configs},
            "quantization": "Q8_0",
        },
        "accuracy": accuracy,
        "per_category_accuracy": per_category,
        "taxonomy_distribution": taxonomy,
        "non_degenerate_rate": {
            key: degenerate[key]["non_degenerate_rate"]
            for key in degenerate
        },
        "degenerate_details": degenerate,
        "fabrication": fabrication,
        "fabrication_rate_non_degenerate": {
            key: fabrication[key]["non_degenerate_rate"]
            for key in fabrication
        },
        "capability_gap": capability_gap or {},
        "qualitative_samples": samples,
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hallucination evaluation results across four model variants: "
            "from-scratch 500M, fine-tuned GPT-2 large, from-scratch 1B base, "
            "and instruction-tuned 1B."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write comparison outputs (default: reports/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help="Directory containing per-model evaluation results (default: eval/results/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages to stdout",
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        metavar="KEY=SUBDIR",
        help=(
            "Override a model result subdirectory, e.g. "
            "--model-dir scratch_1b_base=1B_20260401_120000. "
            f"Valid keys: {', '.join(sorted(_MODEL_SUBDIRS.keys()))}"
        ),
    )
    args = parser.parse_args()

    # Apply model-dir overrides
    for override in args.model_dir:
        if "=" not in override:
            parser.error(f"--model-dir must be KEY=SUBDIR, got: {override}")
        key, subdir = override.split("=", 1)
        if key not in _MODEL_SUBDIRS:
            parser.error(
                f"Unknown model key '{key}'. "
                f"Valid keys: {', '.join(sorted(_MODEL_SUBDIRS.keys()))}"
            )
        _MODEL_SUBDIRS[key] = subdir

    model_configs = _build_model_configs(args.results_dir)

    if args.verbose:
        print("Model Comparison Script")
        print(f"  Output dir: {args.output_dir}")
        print(f"  Results dir: {args.results_dir}")
        print()

    # Load data for both models
    all_data = {}
    for config in model_configs:
        if args.verbose:
            print(f"Loading {config['label']}...")
        if not config["path"].is_dir():
            print(
                f"ERROR: Results directory not found: {config['path']}",
                file=sys.stderr,
            )
            return 1
        all_data[config["key"]] = load_model_data(config["path"])

    # Compute all analyses
    if args.verbose:
        print()
        print("Computing comparisons...")

    accuracy = compute_accuracy_comparison(all_data)
    per_category = compute_per_category_accuracy(all_data)
    taxonomy = compute_taxonomy_comparison(all_data)
    degenerate = compute_degenerate_rates(all_data)
    fabrication = compute_fabrication_comparison(all_data)
    capability_gap = compute_capability_gap(all_data)
    samples = select_qualitative_samples(all_data)

    if args.verbose:
        for config in model_configs:
            mk = config["key"]
            print(f"  {config['short_label']} accuracy: {accuracy[mk]['mean_accuracy']:.4f}")
            print(f"  {config['short_label']} non-degenerate: {degenerate[mk]['non_degenerate']}/{degenerate[mk]['total']}")
            cg = capability_gap[mk]
            print(f"  {config['short_label']} capability gap: {cg['coherence_pct']:.1f}% - {cg['accuracy_pct']:.1f}% = {cg['gap_pct']:.1f}%")
        print(f"  Qualitative samples: {len(samples)}")
        print()

    # Generate outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    md_path = args.output_dir / "model_comparison_report.md"
    if args.verbose:
        print(f"Writing Markdown report to {md_path}...")
    md_content = generate_markdown_report(
        accuracy, per_category, taxonomy, degenerate, fabrication, samples,
        model_configs=model_configs,
        capability_gap=capability_gap,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # JSON output
    json_path = args.output_dir / "model_comparison_report.json"
    if args.verbose:
        print(f"Writing JSON output to {json_path}...")
    json_output = build_json_output(
        accuracy, per_category, taxonomy, degenerate, fabrication, samples,
        model_configs=model_configs,
        capability_gap=capability_gap,
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    if args.verbose:
        print()
        print("Done.")
        print(f"  Markdown: {md_path}")
        print(f"  JSON:     {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
