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

Diagram figures (FIG-05 through FIG-07):
  - pipeline_diagram.png: End-to-end data-to-evaluation pipeline flow
  - model_architecture.png: GPT-2 Pre-LN transformer block diagram
  - eval_framework.png: 6-stage evaluation pipeline flow

Usage:
    python scripts/generate_figures.py [--report PATH] [--output-dir DIR]
                                       [--diagrams-only]
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend — save only, no display
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
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
# Diagram helpers
# ---------------------------------------------------------------------------

# Diagram figure sizes (wider than data charts for flow layouts)
DIAGRAM_SIZE_WIDE = (14, 6)
DIAGRAM_SIZE_TALL = (10, 14)


def _draw_box(
    ax,
    cx: float,
    cy: float,
    width: float,
    height: float,
    text: str,
    fill: str,
    border: str,
    fontsize: float = 9,
    bold: bool = True,
    annotation: str | None = None,
    ann_offset: tuple[float, float] = (0, -0.04),
) -> None:
    """Draw a rounded box with centered text and optional annotation."""
    box = mpatches.FancyBboxPatch(
        (cx - width / 2, cy - height / 2),
        width,
        height,
        boxstyle="round,pad=0.01",
        facecolor=fill,
        edgecolor=border,
        linewidth=1.5,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        cx, cy, text, ha="center", va="center",
        fontsize=fontsize, fontweight=weight, color=border,
        wrap=True,
    )
    if annotation:
        ax.text(
            cx + ann_offset[0], cy + ann_offset[1], annotation,
            ha="center", va="top", fontsize=7, color="#555555",
            style="italic",
        )


def _draw_arrow(
    ax,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    color: str = "#444444",
    label: str | None = None,
    label_offset: tuple[float, float] = (0, 0.02),
) -> None:
    """Draw an arrow between two points with optional label."""
    arrow = mpatches.FancyArrowPatch(
        (x_start, y_start),
        (x_end, y_end),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color=color,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)
    if label:
        mid_x = (x_start + x_end) / 2 + label_offset[0]
        mid_y = (y_start + y_end) / 2 + label_offset[1]
        ax.text(
            mid_x, mid_y, label,
            ha="center", va="bottom", fontsize=7,
            color="#555555", style="italic",
        )


# ---------------------------------------------------------------------------
# FIG-05: Pipeline diagram
# ---------------------------------------------------------------------------

def create_pipeline_diagram(output_path: Path) -> None:
    """Generate end-to-end data-to-evaluation pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=DIAGRAM_SIZE_WIDE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor("white")

    fill = "#dbeafe"
    border = "#1e40af"
    bw = 0.115  # box width
    bh = 0.11   # box height

    # Row 1 (top): Data Collection, Cleaning, Tokenizer, Model Training
    row1_y = 0.72
    row1_boxes = [
        ("Data\nCollection", 0.23, "PubMed, ClinicalTrials,\nPatient Narratives"),
        ("Cleaning &\nDedup", 0.41, "19,164 docs"),
        ("Tokenizer\nTraining", 0.59, "BPE, 50,257 tokens"),
        ("Model\nTraining", 0.77, "DeepSpeed ZeRO-2\n516M params"),
    ]

    for label, cx, ann in row1_boxes:
        _draw_box(ax, cx, row1_y, bw, bh, label, fill, border, annotation=ann,
                  ann_offset=(0, -bh / 2 - 0.02))

    # Arrows for row 1 (with data annotation on first arrow)
    arrow_y1 = row1_y
    for i in range(len(row1_boxes) - 1):
        x_start = row1_boxes[i][1] + bw / 2 + 0.005
        x_end = row1_boxes[i + 1][1] - bw / 2 - 0.005
        label = "142.9M tokens" if i == 0 else None
        _draw_arrow(ax, x_start, arrow_y1, x_end, arrow_y1,
                    color=border, label=label)

    # Connector from row 1 to row 2: Model Training -> GGUF Export
    corner_x = row1_boxes[3][1] + bw / 2 + 0.02
    row2_y = 0.35
    # Vertical connector down from last row1 box
    _draw_arrow(
        ax, row1_boxes[3][1], row1_y - bh / 2 - 0.005,
        row1_boxes[3][1], row2_y + bh / 2 + 0.005,
        color=border,
    )

    # Row 2 (bottom): GGUF Export, Evaluation, RAG Comparison
    row2_boxes = [
        ("GGUF\nExport", 0.77, "Ollama-compatible"),
        ("Hallucination\nEvaluation", 0.59, "160 questions\n6-stage pipeline"),
        ("RAG\nComparison", 0.41, "ChromaDB\n4 configurations"),
    ]

    for label, cx, ann in row2_boxes:
        _draw_box(ax, cx, row2_y, bw, bh, label, fill, border, annotation=ann,
                  ann_offset=(0, -bh / 2 - 0.02))

    # Arrows for row 2 (right to left)
    for i in range(len(row2_boxes) - 1):
        x_start = row2_boxes[i][1] - bw / 2 - 0.005
        x_end = row2_boxes[i + 1][1] + bw / 2 + 0.005
        _draw_arrow(ax, x_start, row2_y, x_end, row2_y, color=border)

    # Final output box
    _draw_box(
        ax, 0.23, row2_y, bw, bh,
        "Research\nPaper", "#e0e7ff", border,
        annotation="research-paper.md + figures",
        ann_offset=(0, -bh / 2 - 0.02),
    )
    _draw_arrow(
        ax, row2_boxes[2][1] - bw / 2 - 0.005, row2_y,
        0.23 + bw / 2 + 0.005, row2_y,
        color=border,
    )

    # Title
    ax.text(
        0.5, 0.95, "ALS-LM: End-to-End Pipeline",
        ha="center", va="top", fontsize=14, fontweight="bold", color="#1e293b",
    )

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-06: Model architecture diagram
# ---------------------------------------------------------------------------

def create_model_architecture_diagram(output_path: Path) -> None:
    """Generate GPT-2 Pre-LN transformer block diagram (vertical, bottom to top)."""
    fig, ax = plt.subplots(figsize=DIAGRAM_SIZE_TALL)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor("white")

    fill = "#dcfce7"
    border = "#166534"
    bw = 0.30   # box width
    bh = 0.045  # box height
    cx = 0.50   # center x for all main boxes

    # Stack from bottom to top
    # y-positions for each layer (bottom to top)
    layers = []

    # Input Embedding + Positional Embedding
    y_input = 0.08
    _draw_box(ax, 0.35, y_input, 0.22, bh, "Token\nEmbedding", fill, border,
              fontsize=9, annotation="vocab=50,257, d=1,280")
    _draw_box(ax, 0.65, y_input, 0.22, bh, "Positional\nEmbedding", fill, border,
              fontsize=9, annotation="ctx=1,024")

    # Sum + Dropout
    y_sum = 0.155
    _draw_box(ax, cx, y_sum, bw, 0.03, "Sum + Dropout", "#f0fdf4", border, fontsize=8)
    # Arrows from embeddings to sum
    _draw_arrow(ax, 0.35, y_input + bh / 2 + 0.003, 0.42, y_sum - 0.015 - 0.003,
                color=border)
    _draw_arrow(ax, 0.65, y_input + bh / 2 + 0.003, 0.58, y_sum - 0.015 - 0.003,
                color=border)

    # Transformer block (expanded)
    block_bottom = 0.22
    block_top = 0.76

    # Block border (dashed)
    block_rect = mpatches.FancyBboxPatch(
        (cx - 0.20, block_bottom), 0.40, block_top - block_bottom,
        boxstyle="round,pad=0.01",
        facecolor="#f0fdf4", edgecolor=border,
        linewidth=1.5, linestyle="--",
    )
    ax.add_patch(block_rect)
    ax.text(
        cx + 0.22, block_top - 0.01,
        "x24 Transformer Blocks",
        ha="left", va="top", fontsize=10, fontweight="bold",
        color=border, style="italic",
    )

    # Inside the block: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    inner_bw = 0.28
    inner_bh = 0.042
    spacing = 0.09

    inner_y = block_bottom + 0.04
    inner_layers = [
        ("LayerNorm 1", None),
        ("Causal Self-Attention", "16 heads, d_head=80"),
        ("Residual Add", None),
        ("LayerNorm 2", None),
        ("MLP (GELU)", "d_ff=5,120 (4x expansion)"),
        ("Residual Add", None),
    ]

    inner_positions = []
    for i, (label, ann) in enumerate(inner_layers):
        y = inner_y + i * spacing
        inner_positions.append(y)
        is_residual = "Residual" in label
        box_fill = "#bbf7d0" if is_residual else fill
        _draw_box(
            ax, cx, y, inner_bw, inner_bh, label, box_fill, border,
            fontsize=8, bold=not is_residual, annotation=ann,
        )

    # Arrows between inner layers
    for i in range(len(inner_positions) - 1):
        _draw_arrow(
            ax, cx, inner_positions[i] + inner_bh / 2 + 0.003,
            cx, inner_positions[i + 1] - inner_bh / 2 - 0.003,
            color=border,
        )

    # Residual connections (curved arrows on the side)
    # First residual: from before LN1 to Residual Add 1
    res1_x = cx + inner_bw / 2 + 0.03
    ax.annotate(
        "", xy=(res1_x - 0.01, inner_positions[2]),
        xytext=(res1_x - 0.01, inner_positions[0]),
        arrowprops=dict(
            arrowstyle="-|>", color="#22c55e", linewidth=1.0,
            connectionstyle="arc3,rad=-0.4",
        ),
    )
    ax.text(res1_x + 0.04, (inner_positions[0] + inner_positions[2]) / 2,
            "residual", fontsize=6, color="#22c55e", rotation=90,
            ha="left", va="center")

    # Second residual: from before LN2 to Residual Add 2
    ax.annotate(
        "", xy=(res1_x - 0.01, inner_positions[5]),
        xytext=(res1_x - 0.01, inner_positions[3]),
        arrowprops=dict(
            arrowstyle="-|>", color="#22c55e", linewidth=1.0,
            connectionstyle="arc3,rad=-0.4",
        ),
    )
    ax.text(res1_x + 0.04, (inner_positions[3] + inner_positions[5]) / 2,
            "residual", fontsize=6, color="#22c55e", rotation=90,
            ha="left", va="center")

    # Arrow from sum to block
    _draw_arrow(ax, cx, y_sum + 0.015 + 0.003, cx, inner_positions[0] - inner_bh / 2 - 0.003,
                color=border)

    # Final LayerNorm
    y_ln_f = block_top + 0.04
    _draw_box(ax, cx, y_ln_f, bw, 0.035, "Final LayerNorm", fill, border, fontsize=9)
    _draw_arrow(ax, cx, inner_positions[-1] + inner_bh / 2 + 0.003,
                cx, y_ln_f - 0.0175 - 0.003, color=border)

    # Linear (weight-tied)
    y_linear = y_ln_f + 0.065
    _draw_box(ax, cx, y_linear, bw, 0.035, "Linear (weight-tied)", fill, border,
              fontsize=9, annotation="shares weights with token embedding")
    _draw_arrow(ax, cx, y_ln_f + 0.0175 + 0.003, cx, y_linear - 0.0175 - 0.003,
                color=border)

    # Output logits
    y_output = y_linear + 0.065
    _draw_box(ax, cx, y_output, bw, 0.035, "Output Logits", "#bbf7d0", border,
              fontsize=9, annotation="(B, T, 50,257)")
    _draw_arrow(ax, cx, y_linear + 0.0175 + 0.003, cx, y_output - 0.0175 - 0.003,
                color=border)

    # Title
    ax.text(
        0.5, 0.98, "ALS-LM: GPT-2 Pre-LN Architecture (516M params)",
        ha="center", va="top", fontsize=14, fontweight="bold", color="#1e293b",
    )

    # Dimension annotation box (right side)
    dim_text = (
        "d_model = 1,280\n"
        "n_heads = 16\n"
        "d_head = 80\n"
        "d_ff = 5,120\n"
        "n_layers = 24\n"
        "vocab = 50,257\n"
        "ctx = 1,024"
    )
    ax.text(
        0.88, 0.50, dim_text,
        ha="center", va="center", fontsize=8,
        color=border, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0fdf4",
                  edgecolor=border, linewidth=1.0),
    )

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FIG-07: Evaluation framework diagram
# ---------------------------------------------------------------------------

def create_eval_framework_diagram(output_path: Path) -> None:
    """Generate 6-stage evaluation pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=DIAGRAM_SIZE_WIDE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor("white")

    fill = "#fef3c7"
    border = "#92400e"
    bw = 0.115
    bh = 0.22

    # 6 stages in a horizontal row (with stage numbers)
    stages = [
        ("1", "Response\nGeneration", "Ollama API\nbatch querying"),
        ("2", "Scoring", "Fuzzy matching\nkey fact extraction"),
        ("3", "Fabrication\nDetection", "Entity registry\ncross-referencing"),
        ("4", "Failure\nTaxonomy", "5 failure modes\nclassification"),
        ("5", "Stratified\nSampling", "Category-balanced\nsubset selection"),
        ("6", "Report\nGeneration", "Markdown reports\nfigure generation"),
    ]

    # Spread stages evenly across the figure
    n = len(stages)
    x_start = 0.13
    x_end = 0.87
    x_positions = [x_start + i * (x_end - x_start) / (n - 1) for i in range(n)]
    stage_y = 0.50

    for i, (num, label, ann) in enumerate(stages):
        cx = x_positions[i]
        # Stage number circle
        circle = mpatches.Circle(
            (cx, stage_y + bh / 2 + 0.04), 0.022,
            facecolor=border, edgecolor=border, linewidth=1.0,
        )
        ax.add_patch(circle)
        ax.text(cx, stage_y + bh / 2 + 0.04, num,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")

        # Stage box
        _draw_box(ax, cx, stage_y, bw, bh, label, fill, border,
                  fontsize=9, annotation=ann,
                  ann_offset=(0, -bh / 2 - 0.02))

    # Arrows between stages
    for i in range(n - 1):
        x_s = x_positions[i] + bw / 2 + 0.005
        x_e = x_positions[i + 1] - bw / 2 - 0.005
        _draw_arrow(ax, x_s, stage_y, x_e, stage_y, color=border)

    # Input label (left of stage 1)
    ax.text(
        x_positions[0] - bw / 2 - 0.06, stage_y,
        "160\nquestions",
        ha="center", va="center", fontsize=8, fontweight="bold",
        color=border,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbeb",
                  edgecolor=border, linewidth=1.0, linestyle="--"),
    )
    _draw_arrow(
        ax, x_positions[0] - bw / 2 - 0.03, stage_y,
        x_positions[0] - bw / 2 - 0.005, stage_y,
        color=border,
    )

    # Output label (right of stage 6)
    ax.text(
        x_positions[-1] + bw / 2 + 0.06, stage_y,
        "Eval\nreport",
        ha="center", va="center", fontsize=8, fontweight="bold",
        color=border,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbeb",
                  edgecolor=border, linewidth=1.0, linestyle="--"),
    )
    _draw_arrow(
        ax, x_positions[-1] + bw / 2 + 0.005, stage_y,
        x_positions[-1] + bw / 2 + 0.03, stage_y,
        color=border,
    )

    # Title
    ax.text(
        0.5, 0.93, "ALS-LM: Hallucination Evaluation Framework",
        ha="center", va="top", fontsize=14, fontweight="bold", color="#1e293b",
    )

    # Subtitle with approach count
    ax.text(
        0.5, 0.88,
        "Applied to 6 approaches: ALS-LM (from-scratch), Llama 3.1 8B (baseline), 4 RAG configurations",
        ha="center", va="top", fontsize=9, color="#555555",
    )

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _generate_diagrams(output_dir: Path) -> int:
    """Generate all architecture/pipeline diagram PNGs. Returns count."""
    print("Generating FIG-05: Pipeline diagram...")
    create_pipeline_diagram(output_dir / "pipeline_diagram.png")
    print("  Saved: pipeline_diagram.png")

    print("Generating FIG-06: Model architecture diagram...")
    create_model_architecture_diagram(output_dir / "model_architecture.png")
    print("  Saved: model_architecture.png")

    print("Generating FIG-07: Evaluation framework diagram...")
    create_eval_framework_diagram(output_dir / "eval_framework.png")
    print("  Saved: eval_framework.png")

    return 3


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
    parser.add_argument(
        "--diagrams-only",
        action="store_true",
        help="Generate only the architecture/pipeline diagrams (FIG-05 through FIG-07)",
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.diagrams_only:
        count = _generate_diagrams(args.output_dir)
        print(f"\nComplete: {count} diagram figures in {args.output_dir}/")
        return

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

    # Generate FIG-05 through FIG-07: Diagrams
    diagram_count = _generate_diagrams(args.output_dir)

    # Summary
    total = 3 + len(copied) + diagram_count
    print(f"\nComplete: {total} figures in {args.output_dir}/")


if __name__ == "__main__":
    main()
