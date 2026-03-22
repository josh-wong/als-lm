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

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"

_MODEL_SUBDIRS = {
    "scratch_500m":        "als-lm-500m_q8_0",
    "scratch_1b_base":     "1B_20260317_205331",
    "gpt2_large":          "als-lm-gpt2-large_q8_0",
    "scratch_1b_instruct": "als-lm-1b-instruct_q8_0",
    "llama32_base":        "als-lm-llama32-base",
    "llama32_qlora":       "alslm-1b_q8_0",
}

_MODEL_ORDER = [
    "scratch_500m", "scratch_1b_base",
    "gpt2_large", "scratch_1b_instruct",
    "llama32_base", "llama32_qlora",
]

_MODEL_LABELS = {
    "scratch_500m":        ("ALS-LM 500M (from-scratch)", "500M"),
    "scratch_1b_base":     ("ALS-LM 1B (from-scratch base)", "1B base"),
    "gpt2_large":          ("GPT-2 large (fine-tuned)", "GPT-2 large"),
    "scratch_1b_instruct": ("ALS-LM 1B (instruction-tuned)", "1B SFT"),
    "llama32_base":        ("Llama 3.2 1B Instruct (base)", "Llama base"),
    "llama32_qlora":       ("Llama 3.2 1B QLoRA", "Llama QLoRA"),
}

_FAMILY_MAP = {
    "scratch_500m": "from_scratch",
    "scratch_1b_base": "from_scratch",
    "gpt2_large": "pretrained_finetune",
    "scratch_1b_instruct": "pretrained_finetune",
    "llama32_base": "pretrained_instruct",
    "llama32_qlora": "pretrained_instruct",
}

FAMILY_COLORS = {
    "from_scratch": "#1f77b4",
    "pretrained_finetune": "#ff7f0e",
    "pretrained_instruct": "#2ca02c",
}

FAMILY_DISPLAY = {
    "from_scratch": "From-scratch",
    "pretrained_finetune": "Pre-trained fine-tune",
    "pretrained_instruct": "Pre-trained instruct",
}

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


def _build_model_configs(results_dir: Path) -> list:
    """Build MODEL_CONFIGS list from a results directory."""
    return [
        {
            "key": key,
            "label": _MODEL_LABELS[key][0],
            "short_label": _MODEL_LABELS[key][1],
            "path": results_dir / _MODEL_SUBDIRS[key],
            "family": _FAMILY_MAP[key],
        }
        for key in _MODEL_ORDER
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_data(model_dir: Path) -> dict:
    """Load all JSON result files for a single model.

    Parameters
    ----------
    model_dir : Path
        Directory containing scores.json, fabrications.json, taxonomy.json,
        and responses.json.

    Returns
    -------
    dict
        Mapping of file stem to loaded JSON content.
    """
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


def compute_capability_gap(all_data: dict) -> dict:
    """Compute the perceived capability gap for each model.

    The capability gap is defined as coherence% minus accuracy%. A model that
    produces coherent-sounding output but scores poorly on factual accuracy
    has a high gap, indicating ethical risk (users may trust plausible-sounding
    but incorrect answers).

    Parameters
    ----------
    all_data : dict
        Mapping of model key to loaded evaluation data.

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


def compute_ablation_delta(qlora_data: dict, baseline_data: dict) -> dict:
    """Compute deltas between QLoRA and ablation baseline.

    Returns accuracy_delta (QLoRA - baseline mean_accuracy),
    fabrication_delta (QLoRA - baseline fabrication rate),
    and coherence_delta (QLoRA coherence% - baseline coherence%).

    Parameters
    ----------
    qlora_data : dict
        Loaded evaluation data for the QLoRA model.
    baseline_data : dict
        Loaded evaluation data for the ablation baseline.

    Returns
    -------
    dict
        ``{accuracy_delta, fabrication_delta, coherence_delta}``.
    """
    qlora_acc = qlora_data["scores"]["aggregate"]["overall"]["mean_accuracy"]
    base_acc = baseline_data["scores"]["aggregate"]["overall"]["mean_accuracy"]

    qlora_fab = qlora_data["fabrications"]["summary"]["flagged_rate"]
    base_fab = baseline_data["fabrications"]["summary"]["flagged_rate"]

    qlora_responses = qlora_data["responses"]["responses"]
    base_responses = baseline_data["responses"]["responses"]

    qlora_total = len(qlora_responses)
    qlora_coherent = sum(1 for r in qlora_responses if r.get("is_coherent", True))
    qlora_coherence = (qlora_coherent / qlora_total * 100) if qlora_total > 0 else 0.0

    base_total = len(base_responses)
    base_coherent = sum(1 for r in base_responses if r.get("is_coherent", True))
    base_coherence = (base_coherent / base_total * 100) if base_total > 0 else 0.0

    return {
        "accuracy_delta": round(qlora_acc - base_acc, 4),
        "fabrication_delta": round(qlora_fab - base_fab, 4),
        "coherence_delta": round(qlora_coherence - base_coherence, 2),
    }


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


def compute_fabrication_comparison(all_data: dict) -> dict:
    """Compute fabrication rates overall and among non-degenerate responses.

    The per-question fabrication data is filtered using the is_coherent field
    from responses.json to compute the non-degenerate-only fabrication rate.
    """
    result = {}
    for key, data in all_data.items():
        summary = _safe_get(data, ["fabrications", "summary"], key)

        responses_list = _safe_get(data, ["responses", "responses"], key)
        coherent_lookup = {
            r["question_id"]: r.get("is_coherent", True)
            for r in responses_list
        }

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


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

def plot_6model_comparison(data: dict, output_path: Path) -> None:
    """Generate grouped bar chart: 3 metrics x 6 models, color-coded by family.

    Metrics: Accuracy (%), Coherence (%), Capability Gap (%).
    Models are grouped into 3 approach families with 2 models each.

    Parameters
    ----------
    data : dict
        Must contain ``accuracy``, ``capability_gap``, and ``model_configs`` keys.
    output_path : Path
        Where to save the PNG figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    model_configs = data["model_configs"]
    accuracy = data["accuracy"]
    capability_gap = data["capability_gap"]

    model_keys = [c["key"] for c in model_configs]
    n_models = len(model_keys)

    metrics = ["Accuracy", "Coherence", "Capability\ngap"]
    n_metrics = len(metrics)

    def _shade(hex_color: str, factor: float) -> str:
        """Lighten (factor>1) or darken (factor<1) a hex color."""
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        if factor > 1:
            r = int(r + (255 - r) * (factor - 1))
            g = int(g + (255 - g) * (factor - 1))
            b = int(b + (255 - b) * (factor - 1))
        else:
            r, g, b = int(r * factor), int(g * factor), int(b * factor)
        return f"#{min(r,255):02x}{min(g,255):02x}{min(b,255):02x}"

    # Track position within each family to alternate dark/light shades
    _family_seen: dict[str, int] = {}

    # Build values: [model_index][metric_index]
    values = []
    colors = []
    for mk in model_keys:
        acc = accuracy[mk]["mean_accuracy"] * 100
        cg = capability_gap.get(mk, {})
        coh = cg.get("coherence_pct", 0.0)
        gap = cg.get("gap_pct", 0.0)
        values.append([acc, coh, gap])
        family = _FAMILY_MAP.get(mk, "from_scratch")
        base_color = FAMILY_COLORS.get(family, "#999999")
        idx = _family_seen.get(family, 0)
        _family_seen[family] = idx + 1
        # First model in family: darker shade, second: lighter shade
        shade_factor = 0.7 if idx == 0 else 1.4
        colors.append(_shade(base_color, shade_factor))

    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(n_metrics)
    total_width = 0.75
    bar_width = total_width / n_models

    for i, mk in enumerate(model_keys):
        offset = (i - (n_models - 1) / 2) * bar_width
        short_label = _MODEL_LABELS[mk][1]
        bars = ax.bar(
            x + offset, values[i], bar_width,
            color=colors[i], label=short_label,
            edgecolor="white", linewidth=0.5,
        )
        for bar in bars:
            height = bar.get_height()
            label_y = max(height, 0) + 0.5
            ax.text(
                bar.get_x() + bar.get_width() / 2, label_y,
                f"{height:.1f}%",
                ha="center", va="bottom", fontsize=6, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_title("6-model cross-comparison: accuracy, coherence, and capability gap",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(alpha=0.3, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Single legend: each model with its unique shade, grouped by family
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), fontsize=8,
              title="Models (shading: dark = first in family, light = second)",
              ncol=3, frameon=True, title_fontsize=9)

    fig.subplots_adjust(bottom=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(
    accuracy: dict,
    taxonomy: dict,
    degenerate: dict,
    fabrication: dict,
    capability_gap: dict,
    ablation_delta: dict | None,
    model_configs: list,
) -> str:
    """Build the full Markdown comparison report for 6 models.

    Parameters
    ----------
    accuracy : dict
        Per-model accuracy metrics from compute_accuracy_comparison.
    taxonomy : dict
        Per-model taxonomy distribution from compute_taxonomy_comparison.
    degenerate : dict
        Per-model degenerate rates from compute_degenerate_rates.
    fabrication : dict
        Per-model fabrication metrics from compute_fabrication_comparison.
    capability_gap : dict
        Per-model capability gap from compute_capability_gap.
    ablation_delta : dict or None
        Delta between QLoRA and baseline from compute_ablation_delta.
    model_configs : list
        Model configuration list from _build_model_configs.

    Returns
    -------
    str
        Complete Markdown report content.
    """
    lines = []

    def add(text: str = "") -> None:
        lines.append(text)

    configs = {c["key"]: c for c in model_configs}
    model_keys = [c["key"] for c in model_configs]
    n_models = len(model_keys)

    # Title
    add(f"# QLoRA comparison report: {n_models}-model cross-comparison")
    add()

    # Summary paragraph
    add(
        f"Cross-comparison of {n_models} model variants evaluated on the "
        "160-question ALS hallucination benchmark using Q8_0 quantization "
        "as the representative level. Models are grouped by approach family: "
        "from-scratch training, pre-trained fine-tuning, and pre-trained "
        "instruction tuning."
    )
    add()

    # Summary table
    delta_header = " Delta vs base |" if ablation_delta else ""
    add(f"| {'Model':<40} | {'Family':<22} | {'Accuracy':>10} | {'Fab. rate':>10} | {'Coherence':>11} | {'Cap. gap':>10} |{delta_header}")
    sep = f"| {'-' * 40} | {'-' * 22} | {'-' * 10} | {'-' * 10} | {'-' * 11} | {'-' * 10} |"
    if ablation_delta:
        sep += f" {'-' * 13} |"
    add(sep)

    for mk in model_keys:
        label = configs[mk]["short_label"]
        family = FAMILY_DISPLAY.get(_FAMILY_MAP.get(mk, ""), "")
        acc = accuracy[mk]["mean_accuracy"] * 100
        fab = fabrication.get(mk, {}).get("overall_rate", 0.0) * 100
        cg = capability_gap.get(mk, {})
        coh = cg.get("coherence_pct", 0.0)
        gap = cg.get("gap_pct", 0.0)

        delta_col = ""
        if ablation_delta:
            if mk == "llama32_qlora":
                delta_val = ablation_delta["accuracy_delta"] * 100
                delta_col = f" {delta_val:>+12.1f}% |"
            else:
                delta_col = f" {'--':>13} |"

        add(
            f"| {label:<40} | {family:<22} | {acc:>9.2f}% | {fab:>9.1f}% | {coh:>10.1f}% | {gap:>9.1f}% |{delta_col}"
        )
    add()

    # Capability gap section
    add("## Perceived capability gap")
    add()
    add(
        "The perceived capability gap measures the difference between how "
        "coherent a model's output appears (coherence%) and how factually "
        "accurate it is (accuracy%). A high gap indicates ethical risk: users "
        "may trust plausible-sounding but incorrect answers."
    )
    add()
    add(f"| {'Model':<40} | {'Coherence':>12} | {'Accuracy':>10} | {'Gap':>8} |")
    add(f"| {'-' * 40} | {'-' * 12} | {'-' * 10} | {'-' * 8} |")
    for mk in model_keys:
        label = configs[mk]["short_label"]
        cg = capability_gap.get(mk, {})
        coh = cg.get("coherence_pct", 0.0)
        acc_pct = cg.get("accuracy_pct", 0.0)
        gap = cg.get("gap_pct", 0.0)
        add(f"| {label:<40} | {coh:>11.1f}% | {acc_pct:>9.2f}% | {gap:>7.1f}% |")
    add()

    # Ablation delta section
    if ablation_delta:
        add("## QLoRA ablation delta")
        add()
        add(
            "Delta between the Llama 3.2 QLoRA model and the unmodified "
            "Llama 3.2 1B Instruct baseline. Positive accuracy delta and "
            "negative fabrication delta indicate improvement from domain "
            "adaptation."
        )
        add()
        add(f"| {'Metric':<30} | {'Value':>12} |")
        add(f"| {'-' * 30} | {'-' * 12} |")
        add(f"| {'Accuracy delta':<30} | {ablation_delta['accuracy_delta'] * 100:>+11.2f}% |")
        add(f"| {'Fabrication rate delta':<30} | {ablation_delta['fabrication_delta'] * 100:>+11.2f}% |")
        add(f"| {'Coherence delta':<30} | {ablation_delta['coherence_delta']:>+11.1f}% |")
        add()

    # Failure taxonomy distribution
    add("## Failure taxonomy distribution")
    add()
    add(
        f"Distribution of failure modes across the {len(TAXONOMY_MODES)} "
        f"taxonomy categories for all {n_models} models."
    )
    add()

    col_labels = [configs[mk]["short_label"] for mk in model_keys]
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

    # Implications section
    add("## Implications")
    add()
    add(
        "The 6-model comparison reveals a consistent narrative about knowledge "
        "sources in domain-specific language models. From-scratch models "
        "(500M and 1B) internalize minimal factual knowledge from the ALS "
        "corpus despite training on 143M tokens. Pre-trained models (GPT-2 "
        "large and the 1B SFT variant) show that pre-existing knowledge from "
        "large-scale pretraining provides a measurable but limited advantage."
    )
    add()
    add(
        "The Llama 3.2 comparison pair (base ablation vs QLoRA) provides the "
        "clearest test of domain adaptation: starting from an instruction-capable "
        "model with strong general knowledge, QLoRA fine-tuning on ALS-specific "
        "data can shift the model's behavior toward the target domain. The "
        "ablation delta quantifies this shift directly."
    )
    add()
    add(
        "The SFT failure (1B instruction-tuned model producing 100% degenerate "
        "output) remains the strongest evidence that instruction tuning cannot "
        "create knowledge that was never internalized during pretraining. For "
        "detailed analysis, see reports/sft_failure_analysis.md."
    )
    add()

    # Caveats
    add("## Caveats and limitations")
    add()
    add(
        "**Single quantization level.** This comparison uses Q8_0 as the "
        "representative quantization level, based on cross-quantization "
        "analysis showing that quantization does not meaningfully affect "
        "evaluation results."
    )
    add()
    add(
        "**General knowledge confound.** Pre-trained models (GPT-2 large, "
        "Llama 3.2) carry general biomedical knowledge from their pretraining "
        "corpora. When these models answer correctly, we cannot fully separate "
        "domain-specific fine-tuning effects from retained general knowledge."
    )
    add()
    add(
        "**Benchmark scope.** The 160-question ALS hallucination benchmark "
        "covers 8 categories but cannot exhaustively test all aspects of ALS "
        "knowledge. Results reflect performance on this specific benchmark."
    )
    add()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json_output(
    accuracy: dict,
    taxonomy: dict,
    degenerate: dict,
    fabrication: dict,
    capability_gap: dict,
    ablation_delta: dict | None,
    model_configs: list,
) -> dict:
    """Build the structured JSON comparison output.

    Parameters
    ----------
    accuracy : dict
        Per-model accuracy metrics.
    taxonomy : dict
        Per-model taxonomy distribution.
    degenerate : dict
        Per-model degenerate rates.
    fabrication : dict
        Per-model fabrication metrics.
    capability_gap : dict
        Per-model capability gap.
    ablation_delta : dict or None
        QLoRA vs baseline deltas.
    model_configs : list
        Model configuration list.

    Returns
    -------
    dict
        Complete JSON-serializable output.
    """
    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models_compared": [c["key"] for c in model_configs],
            "model_labels": {c["key"]: c["label"] for c in model_configs},
            "model_short_labels": {c["key"]: c["short_label"] for c in model_configs},
            "model_families": {c["key"]: c["family"] for c in model_configs},
            "quantization": "Q8_0",
        },
        "accuracy": accuracy,
        "taxonomy_distribution": taxonomy,
        "degenerate_details": degenerate,
        "non_degenerate_rate": {
            key: degenerate[key]["non_degenerate_rate"]
            for key in degenerate
        },
        "fabrication": fabrication,
        "fabrication_rate_non_degenerate": {
            key: fabrication[key]["non_degenerate_rate"]
            for key in fabrication
        },
        "capability_gap": capability_gap,
        "ablation_delta": ablation_delta,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point for the 6-model comparison script."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare hallucination evaluation results across 6 model variants: "
            "from-scratch 500M, from-scratch 1B base, GPT-2 large fine-tuned, "
            "1B instruction-tuned, Llama 3.2 base ablation, Llama 3.2 QLoRA."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write comparison report outputs (default: reports/)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help="Directory containing per-model evaluation results (default: eval/results/)",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Directory to write chart PNG (default: docs/figures/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages to stdout",
    )
    args = parser.parse_args()

    model_configs = _build_model_configs(args.results_dir)

    if args.verbose:
        print("6-Model Comparison Script")
        print(f"  Output dir:  {args.output_dir}")
        print(f"  Results dir: {args.results_dir}")
        print(f"  Figure dir:  {args.figure_dir}")
        print()

    # Load data for all models
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
    taxonomy = compute_taxonomy_comparison(all_data)
    degenerate = compute_degenerate_rates(all_data)
    fabrication = compute_fabrication_comparison(all_data)
    capability_gap = compute_capability_gap(all_data)

    # Compute ablation delta (QLoRA vs baseline)
    ablation_delta = None
    if "llama32_qlora" in all_data and "llama32_base" in all_data:
        ablation_delta = compute_ablation_delta(
            all_data["llama32_qlora"], all_data["llama32_base"]
        )

    if args.verbose:
        for config in model_configs:
            mk = config["key"]
            print(f"  {config['short_label']} accuracy: {accuracy[mk]['mean_accuracy']:.4f}")
            cg = capability_gap[mk]
            print(f"  {config['short_label']} capability gap: {cg['coherence_pct']:.1f}% - {cg['accuracy_pct']:.1f}% = {cg['gap_pct']:.1f}%")
        if ablation_delta:
            print(f"  Ablation delta (accuracy): {ablation_delta['accuracy_delta']:+.4f}")
        print()

    # Generate outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    md_path = args.output_dir / "qlora_comparison_report.md"
    if args.verbose:
        print(f"Writing Markdown report to {md_path}...")
    md_content = generate_markdown_report(
        accuracy, taxonomy, degenerate, fabrication,
        capability_gap, ablation_delta, model_configs,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # JSON output
    json_path = args.output_dir / "qlora_comparison_report.json"
    if args.verbose:
        print(f"Writing JSON output to {json_path}...")
    json_output = build_json_output(
        accuracy, taxonomy, degenerate, fabrication,
        capability_gap, ablation_delta, model_configs,
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Grouped bar chart
    chart_path = args.figure_dir / "qlora_comparison.png"
    if args.verbose:
        print(f"Writing chart to {chart_path}...")
    chart_data = {
        "model_configs": model_configs,
        "accuracy": accuracy,
        "capability_gap": capability_gap,
    }
    plot_6model_comparison(chart_data, chart_path)

    if args.verbose:
        print()
        print("Done.")
        print(f"  Markdown: {md_path}")
        print(f"  JSON:     {json_path}")
        print(f"  Chart:    {chart_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
