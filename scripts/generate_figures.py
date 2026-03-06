#!/usr/bin/env python3
"""
Figure generation script for ALS-LM documentation.

Reads evaluation and RAG comparison data to produce publication-quality
figures, and consolidates all visual assets into docs/figures/ for embedding
in the project paper, README, and model card.

Generated figures (FIG-01 through FIG-03):
  - accuracy_comparison.png: Grouped bar chart of accuracy across 6 approaches
  - failure_taxonomy.png: Stacked bar chart of 7 failure categories
  - retrieval_decomposition.png: Stacked bar chart of retrieval vs generation
    failures for the 4 RAG configurations

Copied figures (FIG-04):
  - train_val_loss.png, perplexity_gap.png, lr_schedule.png from training
    analysis

Usage:
    python scripts/generate_figures.py [--report PATH] [--output-dir DIR]
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend — save only, no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

FIGURE_SIZE = (10, 6)
DPI = 300
GRID_ALPHA = 0.3

# Approach colors (tab10 palette — one per evaluation approach)
APPROACH_COLORS = {
    "als_lm": "#1f77b4",            # Blue
    "baseline": "#d62728",           # Red
    "rag_500_minilm": "#ff7f0e",     # Orange
    "rag_200_minilm": "#2ca02c",     # Green
    "rag_500_pubmedbert": "#9467bd", # Purple
    "rag_200_pubmedbert": "#8c564b", # Brown
}

# Display names for plot labels
DISPLAY_NAMES = {
    "als_lm": "ALS-LM (Q8_0)",
    "baseline": "Baseline (Llama 3.1 8B)",
    "rag_500_minilm": "RAG 500-MiniLM",
    "rag_200_minilm": "RAG 200-MiniLM",
    "rag_500_pubmedbert": "RAG 500-PubMedBERT",
    "rag_200_pubmedbert": "RAG 200-PubMedBERT",
}

# Left-to-right ordering: from-scratch, baseline ceiling, then RAG configs
APPROACH_ORDER = [
    "als_lm",
    "baseline",
    "rag_500_minilm",
    "rag_200_minilm",
    "rag_500_pubmedbert",
    "rag_200_pubmedbert",
]

# Taxonomy category ordering and labels
TAXONOMY_KEYS = [
    "confident_fabrication",
    "plausible_blending",
    "outdated_information",
    "boundary_confusion",
    "accurate_but_misleading",
    "accurate",
    "degenerate",
]

TAXONOMY_LABELS = [
    "Confident Fabrication",
    "Plausible Blending",
    "Outdated Information",
    "Boundary Confusion",
    "Accurate but Misleading",
    "Accurate",
    "Degenerate",
]

# Taxonomy colors: semantically meaningful (distinct from approach colors)
# Reds/oranges = severe failures, yellows = moderate, green = accurate,
# gray = degenerate
TAXONOMY_COLORS = {
    "confident_fabrication": "#c0392b",   # Dark red
    "plausible_blending": "#e74c3c",      # Red
    "outdated_information": "#e67e22",    # Orange
    "boundary_confusion": "#f39c12",      # Amber
    "accurate_but_misleading": "#f1c40f", # Yellow
    "accurate": "#27ae60",                # Green
    "degenerate": "#95a5a6",              # Gray
}

# RAG-only configs for failure decomposition chart
RAG_CONFIGS = [
    "rag_500_minilm",
    "rag_200_minilm",
    "rag_500_pubmedbert",
    "rag_200_pubmedbert",
]

# Training figures to copy (FIG-04)
TRAINING_FIGURES = [
    "train_val_loss.png",
    "perplexity_gap.png",
    "lr_schedule.png",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent academic styling to a matplotlib axes object."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=GRID_ALPHA, linestyle="-", linewidth=0.5)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_comparison_data(report_path: Path) -> dict:
    """Load the RAG comparison report JSON."""
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FIG-01: Accuracy comparison
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(data: dict, output_path: Path) -> None:
    """Generate grouped bar chart comparing accuracy across all approaches."""
    overall = data["overall_accuracy"]

    # Extract accuracy values in display order
    accuracies = [overall[key]["mean_accuracy"] * 100 for key in APPROACH_ORDER]
    colors = [APPROACH_COLORS[key] for key in APPROACH_ORDER]
    labels = [DISPLAY_NAMES[key] for key in APPROACH_ORDER]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    x = np.arange(len(APPROACH_ORDER))
    bars = ax.bar(x, accuracies, color=colors, width=0.6, edgecolor="white",
                  linewidth=0.5)

    # Annotate each bar with its percentage value
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 22)
    setup_axes(ax, "Accuracy Comparison Across Approaches",
               "", "Mean Accuracy (%)")

    # Inset axes for ALS-LM detail (near-zero value needs zoom)
    # Place in the empty headroom above all bars (max bar is ~14.3%, ylim is 22)
    ax_inset = inset_axes(ax, width="25%", height="30%",
                          bbox_to_anchor=(0.7, 0.68, 0.28, 0.35),
                          bbox_transform=ax.transAxes, loc="center")
    als_lm_acc = accuracies[0]
    ax_inset.bar([0], [als_lm_acc], color=APPROACH_COLORS["als_lm"],
                 width=0.4, edgecolor="white", linewidth=0.5)
    ax_inset.set_ylim(0, 2)
    ax_inset.set_xlim(-0.5, 0.5)
    ax_inset.set_xticks([0])
    ax_inset.set_xticklabels(["ALS-LM"], fontsize=7)
    ax_inset.set_ylabel("Accuracy (%)", fontsize=7)
    ax_inset.tick_params(labelsize=7)
    ax_inset.set_title("ALS-LM Detail (0-2%)", fontsize=8, fontweight="bold")
    ax_inset.text(0, als_lm_acc + 0.05, f"{als_lm_acc:.1f}%",
                  ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)

    # Skip tight_layout (incompatible with inset_axes); bbox_inches="tight"
    # in savefig handles margins instead.
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-02: Failure taxonomy distribution
# ---------------------------------------------------------------------------

def plot_failure_taxonomy(data: dict, output_path: Path) -> None:
    """Generate stacked bar chart of failure taxonomy across all approaches."""
    taxonomy = data["taxonomy_distribution"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(APPROACH_ORDER))
    bar_width = 0.6
    bottom = np.zeros(len(APPROACH_ORDER))

    for cat_key, cat_label in zip(TAXONOMY_KEYS, TAXONOMY_LABELS):
        counts = [taxonomy[appr][cat_key]["count"] for appr in APPROACH_ORDER]
        ax.bar(x, counts, bar_width, bottom=bottom,
               color=TAXONOMY_COLORS[cat_key], label=cat_label,
               edgecolor="white", linewidth=0.3)
        bottom += np.array(counts)

    labels = [DISPLAY_NAMES[key] for key in APPROACH_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    setup_axes(ax, "Failure Taxonomy Distribution by Approach",
               "", "Number of Responses (out of 160)")

    ax.legend(ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=8, framealpha=0.9, borderaxespad=0)

    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-03: Retrieval failure decomposition
# ---------------------------------------------------------------------------

def plot_retrieval_decomposition(data: dict, output_path: Path) -> None:
    """Generate stacked bar chart of retrieval vs generation failures."""
    decomp = data["failure_decomposition"]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    labels = [DISPLAY_NAMES[key] for key in RAG_CONFIGS]
    x = np.arange(len(RAG_CONFIGS))
    bar_width = 0.5

    retrieval_counts = []
    generation_counts = []
    retrieval_pcts = []
    generation_pcts = []

    for key in RAG_CONFIGS:
        d = decomp[key]
        retrieval_counts.append(d["retrieval_failures"])
        generation_counts.append(d["generation_failures"])
        retrieval_pcts.append(d["retrieval_pct"] * 100)
        generation_pcts.append(d["generation_pct"] * 100)

    # Bottom segment: retrieval failures (red)
    bars_ret = ax.bar(x, retrieval_counts, bar_width,
                      color="#d62728", label="Retrieval Failures",
                      edgecolor="white", linewidth=0.5)

    # Top segment: generation failures (blue)
    bars_gen = ax.bar(x, generation_counts, bar_width,
                      bottom=retrieval_counts,
                      color="#1f77b4", label="Generation Failures",
                      edgecolor="white", linewidth=0.5)

    # Annotate each segment with percentage (skip if < 5% of total)
    for i in range(len(RAG_CONFIGS)):
        total = retrieval_counts[i] + generation_counts[i]

        # Retrieval segment annotation
        if retrieval_pcts[i] >= 5:
            ax.text(
                x[i], retrieval_counts[i] / 2,
                f"{retrieval_pcts[i]:.0f}%",
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white",
            )

        # Generation segment annotation
        if generation_pcts[i] >= 5:
            ax.text(
                x[i], retrieval_counts[i] + generation_counts[i] / 2,
                f"{generation_pcts[i]:.0f}%",
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    setup_axes(ax, "Retrieval vs Generation Failures by RAG Configuration",
               "", "Number of Wrong Answers")

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-04: Copy training figures
# ---------------------------------------------------------------------------

def copy_training_figures(output_dir: Path) -> list[Path]:
    """Copy existing training analysis PNGs to the output directory."""
    source_dir = Path("reports/production_training_analysis")
    copied = []

    for filename in TRAINING_FIGURES:
        src = source_dir / filename
        dst = output_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(dst)
            print(f"  Copied: {filename}")
        else:
            print(f"  WARNING: Source file not found: {src}")

    return copied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all documentation figures."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for ALS-LM documentation"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("rag/results/comparison_report.json"),
        help="Path to RAG comparison report JSON (default: rag/results/comparison_report.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Output directory for generated figures (default: docs/figures)",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load comparison data
    print("Loading comparison data...")
    data = load_comparison_data(args.report)

    # Generate FIG-01: Accuracy comparison
    print("Generating FIG-01: Accuracy comparison...")
    plot_accuracy_comparison(data, args.output_dir / "accuracy_comparison.png")
    print("  Saved: accuracy_comparison.png")

    # Generate FIG-02: Failure taxonomy
    print("Generating FIG-02: Failure taxonomy...")
    plot_failure_taxonomy(data, args.output_dir / "failure_taxonomy.png")
    print("  Saved: failure_taxonomy.png")

    # Generate FIG-03: Retrieval decomposition
    print("Generating FIG-03: Retrieval decomposition...")
    plot_retrieval_decomposition(
        data, args.output_dir / "retrieval_decomposition.png"
    )
    print("  Saved: retrieval_decomposition.png")

    # Copy FIG-04: Training figures
    print("Copying FIG-04: Training analysis figures...")
    copied = copy_training_figures(args.output_dir)

    # Summary
    total = 3 + len(copied)
    print(f"\nComplete: {total} figures in {args.output_dir}/")


if __name__ == "__main__":
    main()
