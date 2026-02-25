#!/usr/bin/env python3
"""Validate benchmark questions.json and entity registry JSON files.

Usage:
    python eval/validate_benchmark.py --benchmark eval/questions.json
    python eval/validate_benchmark.py --entity-registry eval/entity_registry.json
    python eval/validate_benchmark.py --all

Exit code 0 if all checks pass, exit code 1 if any check fails.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

VALID_CATEGORIES = {
    "drug_treatment",
    "gene_mutation",
    "diagnostic_criteria",
    "clinical_trials",
    "disease_mechanisms",
    "temporal_accuracy",
    "epidemiology",
    "patient_care",
}

CATEGORY_PREFIXES = {
    "drug_treatment": "DRUG",
    "gene_mutation": "GENE",
    "diagnostic_criteria": "DIAG",
    "clinical_trials": "TRIAL",
    "disease_mechanisms": "MECH",
    "temporal_accuracy": "TEMP",
    "epidemiology": "EPI",
    "patient_care": "CARE",
}

PREFIX_TO_CATEGORY = {v: k for k, v in CATEGORY_PREFIXES.items()}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}

VALID_SOURCE_TYPES = {"pubmed", "nct", "guideline", "textbook", "trap", "review"}

ENTITY_TYPES = {"drugs", "genes", "proteins", "trials", "institutions"}

REQUIRED_QUESTION_FIELDS = {
    "id",
    "category",
    "difficulty",
    "question",
    "prompt_template",
    "verified_answer",
    "key_facts",
    "source",
    "is_trap",
}

REQUIRED_SOURCE_FIELDS = {"type", "id"}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

ID_PATTERN = re.compile(r"^([A-Z]+)-(\d{3})$")


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""
    return " ".join(text.lower().strip().split())


# --------------------------------------------------------------------------- #
# Benchmark validation
# --------------------------------------------------------------------------- #


def validate_benchmark(path: str) -> list[str]:
    """Validate benchmark questions.json and return list of error strings."""
    errors: list[str] = []
    filepath = Path(path)

    if not filepath.exists():
        errors.append(f"File not found: {path}")
        return errors

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON: {exc}")
        return errors

    if not isinstance(data, list):
        errors.append("Top-level structure must be a JSON array")
        return errors

    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}  # normalized -> original id
    category_counts: Counter = Counter()
    difficulty_counts: Counter = Counter()
    trap_count = 0
    total = len(data)

    for idx, entry in enumerate(data):
        entry_label = f"Entry [{idx}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label}: Must be a JSON object")
            continue

        entry_id = entry.get("id", f"<missing-id-{idx}>")
        entry_label = f"Entry [{idx}] (id={entry_id})"

        # --- Required fields ---
        missing = REQUIRED_QUESTION_FIELDS - set(entry.keys())
        if missing:
            errors.append(f"{entry_label}: Missing fields: {sorted(missing)}")

        # --- ID format ---
        qid = entry.get("id")
        if isinstance(qid, str):
            match = ID_PATTERN.match(qid)
            if not match:
                errors.append(
                    f"{entry_label}: ID '{qid}' does not match pattern "
                    "CATEGORY_PREFIX-NNN (e.g., DRUG-001)"
                )
            else:
                prefix = match.group(1)
                if prefix not in PREFIX_TO_CATEGORY:
                    errors.append(
                        f"{entry_label}: Unknown ID prefix '{prefix}'. "
                        f"Valid: {sorted(PREFIX_TO_CATEGORY.keys())}"
                    )
                else:
                    # Check prefix matches category
                    cat = entry.get("category")
                    if cat and cat in CATEGORY_PREFIXES:
                        expected_prefix = CATEGORY_PREFIXES[cat]
                        if prefix != expected_prefix:
                            errors.append(
                                f"{entry_label}: ID prefix '{prefix}' does not "
                                f"match category '{cat}' (expected '{expected_prefix}')"
                            )

            # Duplicate ID check
            if qid in seen_ids:
                errors.append(f"{entry_label}: Duplicate ID '{qid}'")
            seen_ids.add(qid)

        # --- Category ---
        cat = entry.get("category")
        if isinstance(cat, str):
            if cat not in VALID_CATEGORIES:
                errors.append(
                    f"{entry_label}: Invalid category '{cat}'. "
                    f"Valid: {sorted(VALID_CATEGORIES)}"
                )
            else:
                category_counts[cat] += 1

        # --- Difficulty ---
        diff = entry.get("difficulty")
        if isinstance(diff, str):
            if diff not in VALID_DIFFICULTIES:
                errors.append(
                    f"{entry_label}: Invalid difficulty '{diff}'. "
                    f"Valid: {sorted(VALID_DIFFICULTIES)}"
                )
            else:
                difficulty_counts[diff] += 1

        # --- Question text ---
        question = entry.get("question")
        if isinstance(question, str) and question.strip():
            norm = normalize_text(question)
            if norm in seen_questions:
                errors.append(
                    f"{entry_label}: Duplicate question text "
                    f"(same as {seen_questions[norm]})"
                )
            seen_questions[norm] = entry_id
        elif "question" in entry:
            errors.append(f"{entry_label}: 'question' must be a non-empty string")

        # --- Prompt template ---
        pt = entry.get("prompt_template")
        if "prompt_template" in entry and (not isinstance(pt, str) or not pt.strip()):
            errors.append(
                f"{entry_label}: 'prompt_template' must be a non-empty string"
            )

        # --- Verified answer ---
        va = entry.get("verified_answer")
        if "verified_answer" in entry and (not isinstance(va, str) or not va.strip()):
            errors.append(
                f"{entry_label}: 'verified_answer' must be a non-empty string"
            )

        # --- Key facts ---
        kf = entry.get("key_facts")
        if isinstance(kf, list):
            if len(kf) < 3 or len(kf) > 5:
                errors.append(
                    f"{entry_label}: 'key_facts' must have 3-5 items, "
                    f"found {len(kf)}"
                )
            for i, fact in enumerate(kf):
                if not isinstance(fact, str) or not fact.strip():
                    errors.append(
                        f"{entry_label}: key_facts[{i}] must be a non-empty string"
                    )
        elif "key_facts" in entry:
            errors.append(f"{entry_label}: 'key_facts' must be a list")

        # --- Source ---
        source = entry.get("source")
        if isinstance(source, dict):
            missing_src = REQUIRED_SOURCE_FIELDS - set(source.keys())
            if missing_src:
                errors.append(
                    f"{entry_label}: source missing fields: {sorted(missing_src)}"
                )
            src_type = source.get("type")
            if isinstance(src_type, str) and src_type not in VALID_SOURCE_TYPES:
                errors.append(
                    f"{entry_label}: Invalid source type '{src_type}'. "
                    f"Valid: {sorted(VALID_SOURCE_TYPES)}"
                )
            # corpus_ref required for non-trap questions
            is_trap = entry.get("is_trap", False)
            if not is_trap and source.get("corpus_ref") is None:
                errors.append(
                    f"{entry_label}: 'corpus_ref' required in source "
                    "for non-trap questions"
                )
        elif "source" in entry:
            errors.append(f"{entry_label}: 'source' must be a JSON object")

        # --- is_trap ---
        is_trap_val = entry.get("is_trap")
        if "is_trap" in entry and not isinstance(is_trap_val, bool):
            errors.append(f"{entry_label}: 'is_trap' must be a boolean")
        if isinstance(is_trap_val, bool) and is_trap_val:
            trap_count += 1

    # --- Distribution report ---
    print("=" * 60)
    print("BENCHMARK VALIDATION REPORT")
    print("=" * 60)
    print(f"\nTotal questions: {total}")
    print(f"Validation errors: {len(errors)}")

    print("\nCategory distribution:")
    for cat in sorted(VALID_CATEGORIES):
        count = category_counts.get(cat, 0)
        print(f"  {cat:<25s} {count:>3d}")

    print("\nDifficulty distribution:")
    for diff in sorted(VALID_DIFFICULTIES):
        count = difficulty_counts.get(diff, 0)
        print(f"  {diff:<10s} {count:>3d}")

    trap_pct = (trap_count / total * 100) if total > 0 else 0
    print(f"\nTrap questions: {trap_count}/{total} ({trap_pct:.1f}%)")

    if trap_pct < 10 or trap_pct > 15:
        print(f"  WARNING: Trap percentage outside 10-15% target range")

    return errors


# --------------------------------------------------------------------------- #
# Entity registry validation
# --------------------------------------------------------------------------- #


def validate_entity_registry(path: str) -> list[str]:
    """Validate entity registry JSON and return list of error strings."""
    errors: list[str] = []
    filepath = Path(path)

    if not filepath.exists():
        errors.append(f"File not found: {path}")
        return errors

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON: {exc}")
        return errors

    if not isinstance(data, dict):
        errors.append("Top-level structure must be a JSON object")
        return errors

    # --- Metadata section ---
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("Missing or invalid 'metadata' section")
    else:
        for field in ("source", "generated_at", "entity_counts"):
            if field not in metadata:
                errors.append(f"metadata missing field: '{field}'")
        entity_counts = metadata.get("entity_counts", {})
        if isinstance(entity_counts, dict):
            for etype in ENTITY_TYPES:
                if etype not in entity_counts:
                    errors.append(
                        f"metadata.entity_counts missing count for '{etype}'"
                    )

    # --- Entity types ---
    missing_types = ENTITY_TYPES - set(data.keys())
    if missing_types:
        errors.append(f"Missing entity types: {sorted(missing_types)}")

    total_entities = 0
    for etype in ENTITY_TYPES:
        entries = data.get(etype)
        if entries is None:
            continue  # Already reported above

        if not isinstance(entries, list):
            errors.append(f"'{etype}' must be a JSON array")
            continue

        if len(entries) == 0:
            errors.append(f"'{etype}' must not be empty")
            continue

        seen_canonical: set[str] = set()
        for idx, entry in enumerate(entries):
            label = f"{etype}[{idx}]"
            if not isinstance(entry, dict):
                errors.append(f"{label}: Must be a JSON object")
                continue

            # Canonical name
            canonical = entry.get("canonical")
            if not isinstance(canonical, str) or not canonical.strip():
                errors.append(f"{label}: 'canonical' must be a non-empty string")
            else:
                norm = canonical.lower().strip()
                if norm in seen_canonical:
                    errors.append(
                        f"{label}: Duplicate canonical name '{canonical}'"
                    )
                seen_canonical.add(norm)

            # Aliases
            aliases = entry.get("aliases")
            if not isinstance(aliases, list):
                errors.append(f"{label}: 'aliases' must be a list")
            else:
                for ai, alias in enumerate(aliases):
                    if not isinstance(alias, str):
                        errors.append(
                            f"{label}: aliases[{ai}] must be a string"
                        )

            # Frequency
            freq = entry.get("frequency")
            if not isinstance(freq, int) or freq < 1:
                errors.append(
                    f"{label}: 'frequency' must be an integer >= 1, "
                    f"got {freq!r}"
                )

        total_entities += len(entries)

    # --- Report ---
    print("=" * 60)
    print("ENTITY REGISTRY VALIDATION REPORT")
    print("=" * 60)
    print(f"\nValidation errors: {len(errors)}")
    print(f"Total entities: {total_entities}")
    print("\nEntity type counts:")
    for etype in sorted(ENTITY_TYPES):
        entries = data.get(etype, [])
        count = len(entries) if isinstance(entries, list) else 0
        print(f"  {etype:<15s} {count:>4d}")

    return errors


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ALS-LM benchmark and entity registry files."
    )
    parser.add_argument(
        "--benchmark",
        metavar="PATH",
        help="Path to benchmark questions.json to validate",
    )
    parser.add_argument(
        "--entity-registry",
        metavar="PATH",
        help="Path to entity registry JSON to validate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate both (default paths: eval/questions.json, "
        "eval/entity_registry.json)",
    )

    args = parser.parse_args()

    # Default to --all if no flags given
    if not args.benchmark and not args.entity_registry and not args.all:
        args.all = True

    all_errors: list[str] = []

    if args.benchmark:
        errors = validate_benchmark(args.benchmark)
        all_errors.extend(errors)
    elif args.all:
        errors = validate_benchmark("eval/questions.json")
        all_errors.extend(errors)

    if args.entity_registry:
        errors = validate_entity_registry(args.entity_registry)
        all_errors.extend(errors)
    elif args.all:
        errors = validate_entity_registry("eval/entity_registry.json")
        all_errors.extend(errors)

    if all_errors:
        print("\n" + "=" * 60)
        print(f"ERRORS ({len(all_errors)}):")
        print("=" * 60)
        for err in all_errors:
            print(f"  ERROR: {err}")
        sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("ALL CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
