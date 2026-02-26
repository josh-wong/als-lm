#!/usr/bin/env python3
"""Generate a self-contained Markdown evaluation report from JSON artifacts.

Reads scored responses, fabrication results, taxonomy classifications, and
curated qualitative samples to produce a Markdown document suitable for
inclusion in the eventual white paper. The report includes accuracy tables,
failure taxonomy distribution, fabrication analysis, hedging summary, and
annotated qualitative samples.

This is a research evaluation tool, not a medical information system.

Usage examples::

    # Generate report with default paths
    python eval/generate_report.py --output reports/hallucination_eval_tiny.md

    # Custom input paths
    python eval/generate_report.py \\
        --scores eval/results/scores.json \\
        --fabrications eval/results/fabrications.json \\
        --taxonomy eval/results/taxonomy.json \\
        --samples eval/results/samples.json \\
        --responses eval/results/responses.json \\
        --output reports/hallucination_eval_500m.md
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown evaluation report from JSON artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval/generate_report.py --output reports/report.md\n"
            "  python eval/generate_report.py --scores eval/results/scores.json\n"
        ),
    )
    parser.add_argument(
        "--scores",
        type=str,
        default="eval/results/scores.json",
        help="Path to scoring output JSON (default: eval/results/scores.json)",
    )
    parser.add_argument(
        "--fabrications",
        type=str,
        default="eval/results/fabrications.json",
        help="Path to fabrication output JSON (default: eval/results/fabrications.json)",
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default="eval/results/taxonomy.json",
        help="Path to taxonomy output JSON (default: eval/results/taxonomy.json)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="eval/results/samples.json",
        help="Path to curated samples JSON (default: eval/results/samples.json)",
    )
    parser.add_argument(
        "--responses",
        type=str,
        default="eval/results/responses.json",
        help="Path to generated responses JSON (default: eval/results/responses.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output Markdown file",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path, description):
    """Load a JSON file with error handling.

    Args:
        path: Path to the JSON file.
        description: Human-readable description for error messages.

    Returns:
        The parsed JSON data.
    """
    if not os.path.isfile(path):
        print(f"ERROR: {description} not found: {path}")
        sys.exit(1)

    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Report section builders
# ---------------------------------------------------------------------------

def build_header(responses_data):
    """Build the report title and metadata section.

    Args:
        responses_data: The loaded responses.json data with metadata.

    Returns:
        A Markdown string for the header section.
    """
    metadata = responses_data.get("metadata", {})
    checkpoint_path = metadata.get("checkpoint_path", "unknown")
    model_config = metadata.get("model_config", {})
    gen_params = metadata.get("generation_params", {})
    total_questions = metadata.get("total_questions", 0)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# ALS-LM hallucination evaluation report",
        "",
        "This report presents the results of the ALS-LM hallucination evaluation"
        " framework, which systematically measures factual accuracy, fabrication"
        " tendencies, and failure modes of a domain-specific language model trained"
        " on ALS research literature.",
        "",
        "## Metadata",
        "",
        f"- **Report generated:** {timestamp}",
        f"- **Checkpoint:** `{checkpoint_path}`",
    ]

    if model_config:
        config_parts = []
        if "n_layer" in model_config:
            config_parts.append(f"n_layer={model_config['n_layer']}")
        if "n_embd" in model_config:
            config_parts.append(f"n_embd={model_config['n_embd']}")
        if "n_head" in model_config:
            config_parts.append(f"n_head={model_config['n_head']}")
        if "vocab_size" in model_config:
            config_parts.append(f"vocab_size={model_config['vocab_size']}")
        if config_parts:
            lines.append(f"- **Model config:** {', '.join(config_parts)}")

    lines.append(f"- **Generation:** max_tokens={gen_params.get('max_tokens', 'N/A')},"
                 f" temperature={gen_params.get('temperature', 'N/A')}")
    lines.append(f"- **Benchmark questions:** {total_questions}")
    lines.append("")
    lines.append("> **Disclaimer:** This model is a research artifact exploring"
                 " what a purpose-built language model can learn about ALS. It"
                 " should never be used for medical decision-making. The"
                 " hallucination evaluation framework exists to quantify the"
                 " model's unreliability, not to demonstrate its usefulness as"
                 " an information source.")
    lines.append("")

    return "\n".join(lines)


def build_methodology():
    """Build the methodology section explaining each evaluation stage.

    Returns:
        A Markdown string for the methodology section.
    """
    lines = [
        "## Methodology",
        "",
        "The evaluation pipeline consists of six stages, each producing"
        " structured JSON artifacts consumed by subsequent stages.",
        "",
        "### Response generation",
        "",
        "Each benchmark question is presented to the model as a prompt, and"
        " the model generates a response using greedy decoding (temperature=0)"
        " for full reproducibility. Responses are generated one at a time with"
        " per-question error isolation to handle degenerate outputs gracefully.",
        "",
        "### Accuracy scoring",
        "",
        "Responses are scored against curated key facts using a sliding-window"
        " fuzzy matching approach. The response text is broken into overlapping"
        " 100-character chunks with 50-character overlap, and each key fact is"
        " matched against all chunks using rapidfuzz partial_ratio. A key fact"
        " is considered found if any chunk scores at or above the threshold"
        " (default 80). Per-question accuracy is the proportion of key facts"
        " found, with binary pass at 50% or above.",
        "",
        "### Fabrication detection",
        "",
        "Drug names, gene names, and clinical trial NCT IDs are extracted from"
        " each response using heuristic patterns (capitalization, pharmaceutical"
        " suffixes, uppercase+digit gene patterns, NCT regex). Each extracted"
        " entity is checked against a training-corpus registry built from the"
        " same data the model was trained on. Entities not found in the registry"
        " are flagged as potentially fabricated. Drug and gene matching uses"
        " fuzzy matching (threshold 85) while trial IDs use exact matching.",
        "",
        "### Taxonomy classification",
        "",
        "Each response is classified into one of five failure modes using"
        " rule-based logic that combines accuracy scores, fabrication flags,"
        " hedging counts, and question metadata. Rules are evaluated in priority"
        " order (confident_fabrication > outdated_information > plausible_blending"
        " > boundary_confusion > accurate_but_misleading). Responses that pass"
        " the accuracy threshold are labeled accurate, while very short outputs"
        " are labeled degenerate.",
        "",
        "### Sample curation",
        "",
        "The best (highest accuracy), worst (lowest accuracy, excluding"
        " degenerate), and edge case (closest to 0.5 threshold) responses are"
        " selected as qualitative samples. Each sample receives an automated"
        " 2-3 sentence annotation describing what the model got right or wrong,"
        " referencing specific key facts by name.",
        "",
    ]
    return "\n".join(lines)


def build_accuracy_table(aggregate):
    """Build the accuracy tables section from aggregate scoring data.

    Args:
        aggregate: The aggregate dict from scores.json containing overall,
            by_category, by_difficulty, and trap_questions sub-dicts.

    Returns:
        A Markdown string for the accuracy tables section.
    """
    overall = aggregate.get("overall", {})
    by_category = aggregate.get("by_category", {})
    by_difficulty = aggregate.get("by_difficulty", {})
    trap = aggregate.get("trap_questions", {})

    lines = [
        "## Accuracy results",
        "",
        "Aggregate accuracy metrics computed from key-fact fuzzy matching"
        " across all benchmark questions.",
        "",
        "### Overall summary",
        "",
    ]

    # Overall summary table
    total_q = sum(v.get("count", 0) for v in by_category.values()) if by_category else 0
    if total_q == 0:
        total_q = trap.get("count", 0)

    lines.extend([
        "| Metric           | Value  |",
        "| ---------------- | ------ |",
        f"| Mean accuracy    | {overall.get('mean_accuracy', 0):.4f} |",
        f"| Median accuracy  | {overall.get('median_accuracy', 0):.4f} |",
        f"| Binary pass rate | {overall.get('binary_pass_rate', 0):.4f} |",
        f"| Total questions  | {total_q:6d} |",
        "",
    ])

    # By category table
    if by_category:
        lines.extend([
            "### By category",
            "",
            "Accuracy broken down by benchmark question category.",
            "",
            "| Category                 | Count | Mean   | Median | Pass rate |",
            "| ------------------------ | ----- | ------ | ------ | --------- |",
        ])
        for cat, stats in sorted(by_category.items()):
            cat_display = cat.replace("_", " ")
            lines.append(
                f"| {cat_display:24s} | {stats.get('count', 0):5d} "
                f"| {stats.get('mean_accuracy', 0):.4f} "
                f"| {stats.get('median_accuracy', 0):.4f} "
                f"| {stats.get('binary_pass_rate', 0):.4f}    |"
            )
        lines.append("")

    # By difficulty table
    if by_difficulty:
        lines.extend([
            "### By difficulty",
            "",
            "Accuracy broken down by question difficulty level.",
            "",
            "| Difficulty | Count | Mean   | Median | Pass rate |",
            "| ---------- | ----- | ------ | ------ | --------- |",
        ])
        for diff, stats in sorted(by_difficulty.items()):
            lines.append(
                f"| {diff:10s} | {stats.get('count', 0):5d} "
                f"| {stats.get('mean_accuracy', 0):.4f} "
                f"| {stats.get('median_accuracy', 0):.4f} "
                f"| {stats.get('binary_pass_rate', 0):.4f}    |"
            )
        lines.append("")

    # Trap questions
    if trap.get("count", 0) > 0:
        lines.extend([
            "### Trap question performance",
            "",
            "Trap questions contain fabricated entities or misleading premises"
            " to test the model's tendency to agree with incorrect information.",
            "",
            "| Metric           | Value  |",
            "| ---------------- | ------ |",
            f"| Count            | {trap['count']:6d} |",
            f"| Mean accuracy    | {trap.get('mean_accuracy', 0):.4f} |",
            f"| Binary pass rate | {trap.get('binary_pass_rate', 0):.4f} |",
            "",
        ])

    return "\n".join(lines)


def build_taxonomy_section(taxonomy_data):
    """Build the failure taxonomy distribution section.

    Args:
        taxonomy_data: The loaded taxonomy.json data with distribution
            and severity_distribution.

    Returns:
        A Markdown string for the taxonomy section.
    """
    distribution = taxonomy_data.get("distribution", {})
    severity_dist = taxonomy_data.get("severity_distribution", {})
    per_question = taxonomy_data.get("per_question", [])

    lines = [
        "## Failure taxonomy distribution",
        "",
        "Each response is classified into one of five failure modes (plus"
        " accurate and degenerate categories) using rule-based logic.",
        "",
    ]

    # ASCII bar chart
    if distribution:
        max_count = max(
            (v.get("count", 0) for v in distribution.values()), default=0
        )
        bar_scale = 40  # max bar width in characters

        lines.append("```")
        for mode, stats in distribution.items():
            count = stats.get("count", 0)
            if max_count > 0:
                bar_len = int((count / max_count) * bar_scale)
            else:
                bar_len = 0
            bar = "#" * bar_len
            lines.append(f"  {mode:28s} {bar} ({count})")
        lines.append("```")
        lines.append("")

    # Distribution table with severity breakdown
    if distribution:
        # Compute per-mode severity counts
        mode_severity = {}
        for entry in per_question:
            mode = entry.get("primary_mode", "unknown")
            sev = entry.get("severity", "none")
            if mode not in mode_severity:
                mode_severity[mode] = {"high": 0, "medium": 0, "low": 0}
            if sev in mode_severity[mode]:
                mode_severity[mode][sev] += 1

        lines.extend([
            "| Failure mode             | Count | Pct    | High | Medium | Low |",
            "| ------------------------ | ----- | ------ | ---- | ------ | --- |",
        ])

        for mode, stats in distribution.items():
            count = stats.get("count", 0)
            pct = stats.get("pct", 0.0)
            sev = mode_severity.get(mode, {"high": 0, "medium": 0, "low": 0})
            mode_display = mode.replace("_", " ")
            lines.append(
                f"| {mode_display:24s} | {count:5d} | {pct:5.1f}% "
                f"| {sev['high']:4d} | {sev['medium']:6d} | {sev['low']:3d} |"
            )
        lines.append("")

    # Interpretation
    if distribution:
        # Find dominant mode (excluding accurate and degenerate)
        failure_modes = {
            k: v for k, v in distribution.items()
            if k not in ("accurate", "degenerate") and v.get("count", 0) > 0
        }
        if failure_modes:
            dominant = max(failure_modes, key=lambda k: failure_modes[k]["count"])
            dominant_display = dominant.replace("_", " ")
            dominant_count = failure_modes[dominant]["count"]
            total = sum(v.get("count", 0) for v in distribution.values())
            dominant_pct = (dominant_count / total * 100) if total > 0 else 0

            lines.extend([
                f"The dominant failure mode is **{dominant_display}**,"
                f" accounting for {dominant_count} responses"
                f" ({dominant_pct:.1f}% of total). ",
            ])

            # Count total failures vs accurate
            accurate_count = distribution.get("accurate", {}).get("count", 0)
            degenerate_count = distribution.get("degenerate", {}).get("count", 0)
            failure_count = total - accurate_count - degenerate_count
            if total > 0:
                lines.append(
                    f"Overall, {failure_count} of {total} responses"
                    f" ({failure_count / total * 100:.1f}%) exhibit a"
                    f" classified failure mode, while {accurate_count} are"
                    f" accurate and {degenerate_count} are degenerate."
                )
            lines.append("")

    # Severity summary
    if severity_dist:
        total_sev = sum(severity_dist.values())
        lines.extend([
            "Severity distribution across all responses:",
            "",
        ])
        for sev in ["high", "medium", "low", "none"]:
            count = severity_dist.get(sev, 0)
            pct = (count / total_sev * 100) if total_sev > 0 else 0
            lines.append(f"- **{sev.capitalize()}:** {count} ({pct:.1f}%)")
        lines.append("")

    return "\n".join(lines)


def build_fabrication_section(fabrications_data):
    """Build the fabrication analysis section.

    Args:
        fabrications_data: The loaded fabrications.json data with summary
            and all_flagged fields.

    Returns:
        A Markdown string for the fabrication section.
    """
    summary = fabrications_data.get("summary", {})
    all_flagged = fabrications_data.get("all_flagged", [])
    by_type = summary.get("by_type", {})

    total_extracted = summary.get("total_entities_extracted", 0)
    total_flagged = summary.get("total_flagged", 0)
    flagged_rate = summary.get("flagged_rate", 0.0)

    lines = [
        "## Fabrication analysis",
        "",
        "Entities (drug names, gene names, clinical trial IDs) extracted from"
        " model responses are checked against a training-corpus registry."
        " Entities not found in the registry are flagged as potentially"
        " fabricated.",
        "",
        "### Summary",
        "",
        f"- **Total entities extracted:** {total_extracted}",
        f"- **Total flagged:** {total_flagged}",
        f"- **Flagged rate:** {flagged_rate:.4f}",
        "",
    ]

    # By type table
    if by_type:
        lines.extend([
            "### By entity type",
            "",
            "Breakdown of entity extraction and flagging by type.",
            "",
            "| Entity type | Extracted | Flagged | Flagged rate |",
            "| ----------- | --------- | ------- | ------------ |",
        ])
        for type_name, stats in sorted(by_type.items()):
            extracted = stats.get("extracted", 0)
            flagged = stats.get("flagged", 0)
            rate = stats.get("rate", 0.0)
            lines.append(
                f"| {type_name:11s} | {extracted:9d} | {flagged:7d} | {rate:.4f}       |"
            )
        lines.append("")

    # Top flagged entities
    if all_flagged:
        # Count occurrences of each entity text
        entity_counter = Counter()
        entity_type_map = {}
        entity_context_map = {}
        for entry in all_flagged:
            text = entry.get("text", "")
            entity_counter[text] += 1
            entity_type_map[text] = entry.get("type", "unknown")
            # Keep the first context seen
            if text not in entity_context_map:
                context = entry.get("context", "")
                if len(context) > 80:
                    context = context[:77] + "..."
                entity_context_map[text] = context

        top_flagged = entity_counter.most_common(10)

        lines.extend([
            "### Top flagged entities",
            "",
            "The most frequently flagged entities across all responses.",
            "",
            "| Entity           | Type   | Occurrences | Context                          |",
            "| ---------------- | ------ | ----------- | -------------------------------- |",
        ])
        for entity_text, count in top_flagged:
            etype = entity_type_map.get(entity_text, "unknown")
            context = entity_context_map.get(entity_text, "")
            # Truncate entity text and context for table display
            entity_display = entity_text[:16] if len(entity_text) > 16 else entity_text
            context_display = context[:32] if len(context) > 32 else context
            lines.append(
                f"| {entity_display:16s} | {etype:6s} "
                f"| {count:11d} | {context_display:32s} |"
            )
        lines.append("")

    return "\n".join(lines)


def build_hedging_section(scores_data):
    """Build the hedging behavior summary section.

    Args:
        scores_data: The loaded scores.json data with per_question entries
            containing hedging_count and hedging_phrases_found fields.

    Returns:
        A Markdown string for the hedging section.
    """
    per_question = scores_data.get("per_question", [])
    overall = scores_data.get("aggregate", {}).get("overall", {})

    total_hedging = overall.get("total_hedging_instances", 0)
    total_questions = len(per_question) if per_question else 1
    avg_hedging = total_hedging / total_questions if total_questions > 0 else 0

    # Count phrase frequencies across all questions
    phrase_counter = Counter()
    for entry in per_question:
        for phrase in entry.get("hedging_phrases_found", []):
            phrase_counter[phrase] += 1

    lines = [
        "## Hedging behavior summary",
        "",
        "Hedging language (words and phrases indicating uncertainty such as"
        " \"may\", \"possibly\", \"it is thought that\") is detected in model"
        " responses to assess the model's confidence calibration.",
        "",
        f"- **Total hedging instances:** {total_hedging}",
        f"- **Average hedging per response:** {avg_hedging:.2f}",
        "",
    ]

    if phrase_counter:
        top_phrases = phrase_counter.most_common(10)
        lines.extend([
            "Most common hedging phrases:",
            "",
        ])
        for phrase, count in top_phrases:
            lines.append(f"- \"{phrase}\" ({count} occurrences)")
        lines.append("")

    return "\n".join(lines)


def build_samples_section(samples_data):
    """Build the qualitative samples section with three subsections.

    Args:
        samples_data: The loaded samples.json data with best, worst,
            and edge_cases lists.

    Returns:
        A Markdown string for the qualitative samples section.
    """
    best = samples_data.get("best", [])
    worst = samples_data.get("worst", [])
    edge = samples_data.get("edge_cases", [])

    lines = [
        "## Qualitative samples",
        "",
        "Representative model responses selected by accuracy score."
        " Best responses show the model's strongest performance, worst"
        " responses illustrate systematic failures, and edge cases reveal"
        " where the model's knowledge fragments at the pass/fail boundary.",
        "",
    ]

    def render_sample_group(group_name, samples):
        """Render a group of samples as Markdown."""
        group_lines = [
            f"### {group_name}",
            "",
        ]
        if not samples:
            group_lines.extend([
                f"No {group_name.lower()} available.",
                "",
            ])
            return group_lines

        group_lines.append(
            f"{len(samples)} samples selected for this category."
        )
        group_lines.append("")

        for sample in samples:
            qid = sample.get("question_id", "unknown")
            question = sample.get("question", "")
            expected = sample.get("expected_answer", "")
            response = sample.get("model_response", "")
            accuracy = sample.get("accuracy", 0.0)
            failure_mode = sample.get("failure_mode", "unknown")
            annotation = sample.get("annotation", "")

            # Abbreviate question for heading
            q_abbrev = question[:60] + "..." if len(question) > 60 else question

            # Truncate response to 500 chars
            if len(response) > 500:
                response_display = response[:497] + "..."
            else:
                response_display = response

            # Determine severity display from failure mode
            mode_display = failure_mode.replace("_", " ")

            # Count key facts if available (from accuracy)
            # We don't have the exact counts here, so use accuracy as proxy
            group_lines.extend([
                f"#### {qid}: {q_abbrev}",
                "",
                f"**Question:** {question}",
                "",
                f"> {response_display}",
                "",
                f"**Expected answer:** {expected}",
                f"**Accuracy:** {accuracy:.2f}",
                f"**Failure mode:** {mode_display}",
                "",
                annotation,
                "",
            ])

        return group_lines

    lines.extend(render_sample_group("Best responses", best))
    lines.extend(render_sample_group("Worst responses", worst))
    lines.extend(render_sample_group("Edge cases", edge))

    return "\n".join(lines)


def build_footer():
    """Build the report footer with disclaimer and generation metadata.

    Returns:
        A Markdown string for the footer section.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "---",
        "",
        "## Disclaimer",
        "",
        "This report is a research artifact produced by the ALS-LM"
        " hallucination evaluation framework. The model evaluated in this"
        " report is not a medical tool and should never be used for medical"
        " decision-making. The evaluation framework exists to quantify the"
        " model's unreliability and characterize its failure modes for"
        " research purposes.",
        "",
        f"*Generated: {timestamp}*",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def generate_report(scores_data, fabrications_data, taxonomy_data,
                    samples_data, responses_data):
    """Assemble the complete Markdown report from all data sources.

    Args:
        scores_data: Loaded scores.json.
        fabrications_data: Loaded fabrications.json.
        taxonomy_data: Loaded taxonomy.json.
        samples_data: Loaded samples.json.
        responses_data: Loaded responses.json.

    Returns:
        A complete Markdown report string.
    """
    aggregate = scores_data.get("aggregate", {})

    sections = [
        build_header(responses_data),
        build_methodology(),
        build_accuracy_table(aggregate),
        build_taxonomy_section(taxonomy_data),
        build_fabrication_section(fabrications_data),
        build_hedging_section(scores_data),
        build_samples_section(samples_data),
        build_footer(),
    ]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for report generation."""
    args = parse_args()

    print("\n=== ALS-LM Report Generation ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical"
          " information system.\n")

    # Load all data sources
    scores_data = load_json(args.scores, "Scores file")
    fabrications_data = load_json(args.fabrications, "Fabrications file")
    taxonomy_data = load_json(args.taxonomy, "Taxonomy file")
    samples_data = load_json(args.samples, "Samples file")
    responses_data = load_json(args.responses, "Responses file")

    print(f"  Loaded all evaluation artifacts")

    # Generate report
    report_md = generate_report(
        scores_data, fabrications_data, taxonomy_data,
        samples_data, responses_data,
    )

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        f.write(report_md)

    print(f"  Report generated: {args.output}")

    # Report stats
    line_count = report_md.count("\n")
    print(f"  Report length: {line_count} lines, {len(report_md):,} characters")
    print()


if __name__ == "__main__":
    main()
