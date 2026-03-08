#!/usr/bin/env python3
"""Compare hallucination evaluation results across two models.

Standalone script that reads Q8_0 evaluation results for the from-scratch
500M model and the fine-tuned GPT-2 large model, then produces a dual-format
comparison report (Markdown + JSON) analyzing how model architecture and
training approach affect accuracy, hallucination behavior, and failure mode
distribution.

This is a research analysis tool, not a medical information system.

Usage examples::

    # Default: write reports to reports/
    python scripts/compare_models.py

    # Custom output directory
    python scripts/compare_models.py --output-dir /tmp/reports

    # Verbose output
    python scripts/compare_models.py --verbose
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Hardcoded model paths per user decision (Q8_0 representative)
MODEL_CONFIGS = [
    {
        "key": "scratch_500m",
        "label": "ALS-LM 500M (from-scratch)",
        "short_label": "500M",
        "path": REPO_ROOT / "eval/results/als-lm-500m_q8_0",
    },
    {
        "key": "gpt2_large_finetune",
        "label": "GPT-2 large (fine-tuned)",
        "short_label": "GPT-2 large",
        "path": REPO_ROOT / "eval/results/als-lm-gpt2-large_q8_0",
    },
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
# Analysis functions
# ---------------------------------------------------------------------------

def compute_accuracy_comparison(all_data: dict) -> dict:
    """Extract mean_accuracy, binary_pass_rate, hedging for each model."""
    result = {}
    for key, data in all_data.items():
        overall = data["scores"]["aggregate"]["overall"]
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
            cat_data = data["scores"]["aggregate"]["by_category"].get(cat, {})
            result[cat][key] = cat_data.get("mean_accuracy", 0.0)
    return result


def compute_taxonomy_comparison(all_data: dict) -> dict:
    """Extract taxonomy distribution counts and percentages for each model."""
    result = {}
    for key, data in all_data.items():
        dist = data["taxonomy"]["distribution"]
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
        dist = data["taxonomy"]["distribution"]
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
        summary = data["fabrications"]["summary"]

        # Build is_coherent lookup from responses.json
        coherent_lookup = {
            r["question_id"]: r.get("is_coherent", True)
            for r in data["responses"]["responses"]
        }

        # Per-question fabrication data filtered to coherent responses only
        coherent_extracted = 0
        coherent_flagged = 0
        for pq in data["fabrications"]["per_question"]:
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


def select_qualitative_samples(all_data: dict) -> list:
    """Select 3-5 diverse question pairs showing different comparison patterns.

    Selection strategy:
    (a) One where GPT-2 large gives a coherent response (rare, showing different
        behavior from 500M)
    (b) One where both models produce degenerate output
    (c) One where 500M is coherent but GPT-2 large is degenerate
    """
    scratch_data = all_data["scratch_500m"]
    finetune_data = all_data["gpt2_large_finetune"]

    # Build lookups
    scratch_resp = {
        r["question_id"]: r for r in scratch_data["responses"]["responses"]
    }
    finetune_resp = {
        r["question_id"]: r for r in finetune_data["responses"]["responses"]
    }
    scratch_tax = {
        e["question_id"]: e["primary_mode"]
        for e in scratch_data["taxonomy"]["per_question"]
    }
    finetune_tax = {
        e["question_id"]: e["primary_mode"]
        for e in finetune_data["taxonomy"]["per_question"]
    }

    samples = []
    all_qids = sorted(set(scratch_resp.keys()) & set(finetune_resp.keys()))

    # (a) GPT-2 large coherent, different taxonomy from 500M
    for qid in all_qids:
        if finetune_resp[qid].get("is_coherent", True) and scratch_resp[qid].get("is_coherent", True):
            s_mode = scratch_tax.get(qid, "unknown")
            f_mode = finetune_tax.get(qid, "unknown")
            if s_mode != f_mode:
                samples.append({
                    "pattern": "both_coherent_different_taxonomy",
                    "question_id": qid,
                    "category": scratch_resp[qid]["category"],
                    "prompt": scratch_resp[qid]["prompt"],
                    "scratch_response": scratch_resp[qid]["response"][:300],
                    "finetune_response": finetune_resp[qid]["response"][:300],
                    "scratch_taxonomy": s_mode,
                    "finetune_taxonomy": f_mode,
                })
                break

    # (b) Both degenerate
    for qid in all_qids:
        if (not scratch_resp[qid].get("is_coherent", True) and
                not finetune_resp[qid].get("is_coherent", True)):
            samples.append({
                "pattern": "both_degenerate",
                "question_id": qid,
                "category": scratch_resp[qid]["category"],
                "prompt": scratch_resp[qid]["prompt"],
                "scratch_response": scratch_resp[qid]["response"][:300],
                "finetune_response": finetune_resp[qid]["response"][:300],
                "scratch_taxonomy": scratch_tax.get(qid, "unknown"),
                "finetune_taxonomy": finetune_tax.get(qid, "unknown"),
            })
            break

    # (c) 500M coherent, GPT-2 large degenerate — pick from diverse categories
    seen_categories = set()
    for qid in all_qids:
        cat = scratch_resp[qid]["category"]
        if cat in seen_categories:
            continue
        if (scratch_resp[qid].get("is_coherent", True) and
                not finetune_resp[qid].get("is_coherent", True)):
            samples.append({
                "pattern": "scratch_coherent_finetune_degenerate",
                "question_id": qid,
                "category": cat,
                "prompt": scratch_resp[qid]["prompt"],
                "scratch_response": scratch_resp[qid]["response"][:300],
                "finetune_response": finetune_resp[qid]["response"][:300],
                "scratch_taxonomy": scratch_tax.get(qid, "unknown"),
                "finetune_taxonomy": finetune_tax.get(qid, "unknown"),
            })
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
) -> str:
    """Build the full Markdown comparison report."""
    lines = []

    def add(text: str = "") -> None:
        lines.append(text)

    configs = {c["key"]: c for c in MODEL_CONFIGS}

    # Section 1: Title
    add("# Model comparison report: from-scratch vs. fine-tuned")
    add()

    # Section 2: TL;DR
    s_acc = accuracy["scratch_500m"]["mean_accuracy"] * 100
    f_acc = accuracy["gpt2_large_finetune"]["mean_accuracy"] * 100
    s_nondeg = degenerate["scratch_500m"]["non_degenerate_rate"] * 100
    f_nondeg = degenerate["gpt2_large_finetune"]["non_degenerate_rate"] * 100
    improvement = f_acc / s_acc if s_acc > 0 else 0

    add(
        f"The fine-tuned GPT-2 large model (774M parameters) achieves "
        f"{f_acc:.2f}% mean accuracy compared to {s_acc:.2f}% for the "
        f"from-scratch 500M model, a {improvement:.0f}x relative improvement. "
        f"However, the GPT-2 large model produces {f_nondeg:.1f}% "
        f"non-degenerate responses compared to {s_nondeg:.1f}% for the "
        f"from-scratch model, meaning {100 - f_nondeg:.1f}% of its output "
        f"is repetitive or incoherent. This reflects the fundamental "
        f"instruction-following limitation of a completion model that has not "
        f"undergone RLHF or chat fine-tuning. Even with 774M pretrained "
        f"parameters and general English knowledge from WebText, fine-tuning "
        f"on 143M ALS tokens produces a model that is still 97% away from "
        f"useful accuracy, reinforcing the data deficit thesis."
    )
    add()

    # Section 3: Overall accuracy comparison
    add("## Overall accuracy comparison")
    add()
    add(
        "Mean accuracy, binary pass rate, and hedging instances for each "
        "model, evaluated on the 160-question ALS hallucination benchmark "
        "using Q8_0 quantization as the representative level."
    )
    add()
    add(
        f"| {'Metric':<25} | {'500M (from-scratch)':>22} | {'GPT-2 large (fine-tuned)':>26} |"
    )
    add(
        f"| {'-' * 25} | {'-' * 22} | {'-' * 26} |"
    )
    s = accuracy["scratch_500m"]
    f = accuracy["gpt2_large_finetune"]
    add(
        f"| {'Mean accuracy':<25} | {s['mean_accuracy'] * 100:>21.2f}% | {f['mean_accuracy'] * 100:>25.2f}% |"
    )
    add(
        f"| {'Binary pass rate':<25} | {s['binary_pass_rate'] * 100:>21.2f}% | {f['binary_pass_rate'] * 100:>25.2f}% |"
    )
    add(
        f"| {'Hedging instances':<25} | {s['hedging']:>22} | {f['hedging']:>26} |"
    )
    add()

    # Section 4: Per-category accuracy
    add("## Per-category accuracy")
    add()
    add(
        "Mean accuracy broken down by the 8 evaluation categories. "
        "The GPT-2 large model shows its strongest performance in the "
        "gene_mutation category (13.75%), while the 500M model's only "
        "non-zero category is patient_care (1.67%)."
    )
    add()
    add(
        f"| {'Category':<25} | {'500M (from-scratch)':>22} | {'GPT-2 large (fine-tuned)':>26} |"
    )
    add(
        f"| {'-' * 25} | {'-' * 22} | {'-' * 26} |"
    )
    for cat in CATEGORIES:
        display = cat.replace("_", " ").title()
        s_val = per_category[cat]["scratch_500m"] * 100
        f_val = per_category[cat]["gpt2_large_finetune"] * 100
        add(
            f"| {display:<25} | {s_val:>21.2f}% | {f_val:>25.2f}% |"
        )
    add()

    # Section 5: Taxonomy distribution
    add("## Taxonomy distribution")
    add()
    add(
        "Distribution of failure modes across the 7 taxonomy categories. "
        "The most striking difference is in degenerate output: the 500M "
        "model produces 52 degenerate responses (32.5%) while the GPT-2 "
        "large model produces 156 (97.5%). The 500M model exhibits more "
        "diverse failure modes, including confident fabrication (33.1%), "
        "plausible blending (23.8%), and outdated information (10.6%)."
    )
    add()
    add(
        f"| {'Failure mode':<28} | {'500M (n)':>10} | {'500M (%)':>10} | {'GPT-2 (n)':>10} | {'GPT-2 (%)':>10} |"
    )
    add(
        f"| {'-' * 28} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} |"
    )
    for mode in TAXONOMY_MODES:
        display = mode.replace("_", " ").title()
        s_data = taxonomy["scratch_500m"][mode]
        f_data = taxonomy["gpt2_large_finetune"][mode]
        add(
            f"| {display:<28} | {s_data['count']:>10} | {s_data['pct']:>9.1f}% | {f_data['count']:>10} | {f_data['pct']:>9.1f}% |"
        )
    add()

    # Section 6: Degenerate output analysis
    add("## Degenerate output analysis")
    add()
    s_deg = degenerate["scratch_500m"]
    f_deg = degenerate["gpt2_large_finetune"]
    deg_ratio = (
        f"{s_deg['non_degenerate_rate'] / f_deg['non_degenerate_rate']:.0f}x"
        if f_deg["non_degenerate_rate"] > 0
        else "large"
    )
    add(
        f"The 500M from-scratch model produces {s_deg['non_degenerate']} "
        f"non-degenerate responses out of {s_deg['total']} "
        f"({s_deg['non_degenerate_rate'] * 100:.1f}%), while the GPT-2 large "
        f"fine-tuned model produces only {f_deg['non_degenerate']} "
        f"non-degenerate responses ({f_deg['non_degenerate_rate'] * 100:.1f}%). "
        f"This {deg_ratio} "
        f"difference in coherent output is the defining characteristic of "
        f"the comparison."
    )
    add()
    add(
        "The GPT-2 large model is a completion model trained on WebText "
        "without reinforcement learning from human feedback (RLHF) or "
        "instruction tuning. When given a question prompt, it tends to "
        "generate text that continues the question's topic rather than "
        "answering it, often falling into repetitive loops. The 500M "
        "from-scratch model, while also lacking instruction tuning, was "
        "trained exclusively on ALS text and produces more diverse (though "
        "largely incorrect) responses. The CLI demo uses keyword filtering "
        "as a practical workaround to surface coherent responses, but this "
        "does not address the underlying instruction-following limitation."
    )
    add()

    # Section 7: Fabrication rate comparison
    add("## Fabrication rate comparison")
    add()
    s_fab = fabrication["scratch_500m"]
    f_fab = fabrication["gpt2_large_finetune"]
    add(
        "Entity-level fabrication analysis, both overall and filtered to "
        "non-degenerate responses only. The overall rate includes entities "
        "extracted from all responses, while the non-degenerate rate filters "
        "to coherent responses before computing fabrication."
    )
    add()
    add(
        f"| {'Metric':<40} | {'500M (from-scratch)':>22} | {'GPT-2 large (fine-tuned)':>26} |"
    )
    add(
        f"| {'-' * 40} | {'-' * 22} | {'-' * 26} |"
    )
    add(
        f"| {'Total entities extracted':<40} | {s_fab['overall_extracted']:>22} | {f_fab['overall_extracted']:>26} |"
    )
    add(
        f"| {'Total entities flagged':<40} | {s_fab['overall_flagged']:>22} | {f_fab['overall_flagged']:>26} |"
    )
    add(
        f"| {'Overall fabrication rate':<40} | {s_fab['overall_rate'] * 100:>21.2f}% | {f_fab['overall_rate'] * 100:>25.2f}% |"
    )
    add(
        f"| {'Non-degenerate responses':<40} | {s_fab['non_degenerate_count']:>22} | {f_fab['non_degenerate_count']:>26} |"
    )
    add(
        f"| {'Entities from non-degenerate':<40} | {s_fab['non_degenerate_extracted']:>22} | {f_fab['non_degenerate_extracted']:>26} |"
    )
    add(
        f"| {'Flagged from non-degenerate':<40} | {s_fab['non_degenerate_flagged']:>22} | {f_fab['non_degenerate_flagged']:>26} |"
    )
    add(
        f"| {'Fabrication rate (non-degenerate)':<40} | {s_fab['non_degenerate_rate'] * 100:>21.2f}% | {f_fab['non_degenerate_rate'] * 100:>25.2f}% |"
    )
    add()
    fab_ratio = (
        f"{f_fab['overall_extracted'] / s_fab['overall_extracted']:.1f}x difference"
        if s_fab["overall_extracted"] > 0
        else "significant difference"
    )
    add(
        f"The GPT-2 large model extracts {f_fab['overall_extracted']} entities "
        f"compared to {s_fab['overall_extracted']} for the 500M model. This "
        f"{fab_ratio} "
        f"reflects the larger model's tendency to generate more text per "
        f"response, including entity-like strings in degenerate output. When "
        f"filtered to non-degenerate responses only, the GPT-2 large model's "
        f"4 coherent responses still produce {f_fab['non_degenerate_extracted']} "
        f"entities with a {f_fab['non_degenerate_rate'] * 100:.2f}% fabrication "
        f"rate, comparable to the 500M model's "
        f"{s_fab['non_degenerate_rate'] * 100:.2f}%."
    )
    add()

    # Section 8: Qualitative sample pairs
    add("## Qualitative sample pairs")
    add()
    add(
        "Selected examples comparing how both models respond to the same "
        "question. These examples are chosen to illustrate three patterns: "
        "(a) both models coherent with different failure taxonomies, "
        "(b) both models degenerate, and "
        "(c) 500M coherent while GPT-2 large is degenerate. "
        "Response text is truncated to 300 characters."
    )
    add()

    pattern_labels = {
        "both_coherent_different_taxonomy": "Both coherent, different taxonomy",
        "both_degenerate": "Both degenerate",
        "scratch_coherent_finetune_degenerate": "500M coherent, GPT-2 large degenerate",
    }

    for i, sample in enumerate(samples, 1):
        pattern = pattern_labels.get(sample["pattern"], sample["pattern"])
        add(f"### Example {i}: {sample['question_id']} ({pattern})")
        add()
        add(f"**Category:** {sample['category'].replace('_', ' ')}")
        add()
        add(f"**Prompt:** {sample['prompt']}")
        add()
        add(
            f"**500M response** (taxonomy: "
            f"{sample['scratch_taxonomy'].replace('_', ' ')}):"
        )
        add()
        # Clean whitespace for display
        scratch_clean = " ".join(sample["scratch_response"].split())[:300]
        add(f"> {scratch_clean}")
        add()
        add(
            f"**GPT-2 large response** (taxonomy: "
            f"{sample['finetune_taxonomy'].replace('_', ' ')}):"
        )
        add()
        finetune_clean = " ".join(sample["finetune_response"].split())[:300]
        add(f"> {finetune_clean}")
        add()

    # Section 9: Instruction-following limitation
    add("## Instruction-following limitation")
    add()
    add(
        "GPT-2 is a causal language model trained to predict the next token "
        "in a sequence. Unlike instruction-tuned models (e.g., those trained "
        "with RLHF or supervised fine-tuning on Q&A pairs), GPT-2 has no "
        "mechanism to distinguish a question from a text to be continued. "
        "When given a question prompt, it treats it as the beginning of a "
        "document and generates a plausible continuation, which typically "
        "means more text in the same style rather than an answer."
    )
    add()
    add(
        "This explains the 97.5% degenerate output rate: the model is not "
        "\"failing\" to answer questions so much as performing a different "
        "task (text completion) than the one being evaluated (question "
        "answering). The 4 coherent responses likely result from prompts "
        "that happen to align with patterns in the training data where "
        "question-like text is followed by answer-like text."
    )
    add()
    add(
        "The CLI demo addresses this with a keyword filter that detects and "
        "re-prompts on degenerate output, but this is a practical workaround "
        "rather than a solution. Instruction tuning the fine-tuned model on "
        "ALS Q&A pairs would be the principled approach but is beyond the "
        "scope of this project."
    )
    add()

    # Section 10: Caveats and limitations
    add("## Caveats and limitations")
    add()
    add(
        "**General knowledge confound.** The GPT-2 large model was "
        "pretrained on WebText, which includes diverse English text that may "
        "contain biomedical content. When the fine-tuned model produces a "
        "correct answer, we cannot determine whether the knowledge comes from "
        "(a) the ALS-specific fine-tuning on 143M tokens, or (b) general "
        "biomedical knowledge retained from WebText pretraining. This is an "
        "inherent limitation of the fine-tuning approach that would require "
        "ablation studies (e.g., comparing the fine-tuned model to the base "
        "GPT-2 large without ALS fine-tuning) to resolve. Such ablation is "
        "beyond the scope of this project."
    )
    add()
    add(
        "**Limited coherent sample size.** With only 4 non-degenerate "
        "responses from the GPT-2 large model, all per-response metrics "
        "(accuracy, fabrication rate among non-degenerate) are computed over "
        "an extremely small sample. These numbers should be interpreted as "
        "indicative rather than statistically robust."
    )
    add()
    add(
        "**Single quantization level.** This comparison uses Q8_0 as the "
        "representative quantization level for both models, based on "
        "cross-quantization analysis showing that quantization does not "
        "meaningfully affect evaluation results (151/160 taxonomy agreements "
        "across F16, Q8_0, Q4_K_M for each model)."
    )
    add()

    # Section 11: Summary
    add("## Summary")
    add()
    add(
        "The comparison between the from-scratch 500M model and the "
        "fine-tuned GPT-2 large model reinforces the data deficit thesis "
        "that is the central finding of this project. Even with 774M "
        "pretrained parameters and general English knowledge from WebText "
        "pretraining, fine-tuning on 143M ALS tokens only improves mean "
        f"accuracy from {s_acc:.2f}% to {f_acc:.2f}% -- a "
        f"{improvement:.0f}x relative improvement that still leaves the model "
        f"97% away from useful accuracy. The fine-tuned model trades diverse "
        f"failure modes (fabrication, blending, outdated information) for "
        f"overwhelming degenerate output ({100 - f_nondeg:.1f}%), reflecting "
        f"the instruction-following limitation of a base completion model. "
        f"Both models demonstrate that small-scale domain-specific training "
        f"is insufficient for reliable medical question answering, whether "
        f"starting from scratch or building on a pretrained foundation."
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
) -> dict:
    """Build the structured JSON comparison output.

    The three metric groups for the figure (accuracy, non_degenerate_rate,
    fabrication_rate_non_degenerate) are directly extractable without
    recomputation.
    """
    return {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models_compared": [c["key"] for c in MODEL_CONFIGS],
            "model_labels": {c["key"]: c["label"] for c in MODEL_CONFIGS},
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
        "qualitative_samples": samples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare hallucination evaluation results between the from-scratch "
            "500M model and the fine-tuned GPT-2 large model."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write comparison outputs (default: reports/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages to stdout",
    )
    args = parser.parse_args()

    if args.verbose:
        print("Model Comparison Script")
        print(f"  Output dir: {args.output_dir}")
        print()

    # Load data for both models
    all_data = {}
    for config in MODEL_CONFIGS:
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
    samples = select_qualitative_samples(all_data)

    if args.verbose:
        print(f"  500M accuracy: {accuracy['scratch_500m']['mean_accuracy']:.4f}")
        print(f"  GPT-2 accuracy: {accuracy['gpt2_large_finetune']['mean_accuracy']:.4f}")
        print(f"  500M non-degenerate: {degenerate['scratch_500m']['non_degenerate']}/{degenerate['scratch_500m']['total']}")
        print(f"  GPT-2 non-degenerate: {degenerate['gpt2_large_finetune']['non_degenerate']}/{degenerate['gpt2_large_finetune']['total']}")
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
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # JSON output
    json_path = args.output_dir / "model_comparison_report.json"
    if args.verbose:
        print(f"Writing JSON output to {json_path}...")
    json_output = build_json_output(
        accuracy, per_category, taxonomy, degenerate, fabrication, samples,
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
