#!/usr/bin/env python3
"""Generate Alpaca-format ALS instruction pairs from corpus passages via Ollama.

Extracts passages from the ALS training corpus by category, generates Q&A pairs
using Llama 3.1 8B through local Ollama with structured JSON output, and writes
the results to an Alpaca-format JSON file.

The pipeline is fully automated and runs end-to-end with a single command.
Incremental save and --resume support allow recovery from interruptions during
the multi-hour Ollama generation run.

Usage::

    python data/instruction/generate.py
    python data/instruction/generate.py --target-per-category 100
    python data/instruction/generate.py --resume

Output:

    data/instruction/als_instructions.json  - Alpaca-format Q&A pairs
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Optional

import requests

# Ensure project root is on sys.path
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Category keywords (8 ALS knowledge domains, 10+ keywords each)
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "diagnosis": [
        "diagnosis", "diagnosed", "diagnostic", "el escorial",
        "electromyography", "emg", "nerve conduction", "clinical criteria",
        "differential diagnosis", "biomarker", "mri", "lumbar puncture",
    ],
    "symptoms": [
        "symptom", "weakness", "fasciculation", "dysarthria", "dysphagia",
        "spasticity", "muscle atrophy", "bulbar", "limb onset",
        "respiratory", "progression", "cramping", "hyperreflexia",
    ],
    "genetics": [
        "gene", "mutation", "sod1", "c9orf72", "fus", "tardbp", "tdp-43",
        "familial", "hereditary", "genetic", "repeat expansion",
        "chromosome", "autosomal",
    ],
    "treatment": [
        "treatment", "therapy", "riluzole", "edaravone", "drug",
        "clinical trial", "therapeutic", "neuroprotective",
        "antisense oligonucleotide", "tofersen", "pharmacological",
    ],
    "clinical_trials": [
        "clinical trial", "randomized", "placebo", "phase ii",
        "phase iii", "endpoint", "alsfrs", "enrollment",
        "efficacy", "nct", "double-blind", "multicenter",
    ],
    "epidemiology": [
        "incidence", "prevalence", "epidemiol", "risk factor",
        "mortality", "survival", "age of onset", "population",
        "male female ratio", "geographic", "demographic",
    ],
    "pathophysiology": [
        "pathophysiol", "mechanism", "excitotoxicity", "glutamate",
        "oxidative stress", "protein aggregation", "motor neuron",
        "neurodegeneration", "mitochondr", "autophagy", "apoptosis",
    ],
    "patient_care": [
        "care", "ventilation", "nutrition", "gastrostomy", "palliative",
        "multidisciplinary", "quality of life", "caregiver",
        "rehabilitation", "non-invasive ventilation", "hospice",
    ],
}


# ---------------------------------------------------------------------------
# Alpaca structured output schema for Ollama
# ---------------------------------------------------------------------------

ALPACA_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
        "input": {"type": "string"},
        "output": {"type": "string"},
    },
    "required": ["instruction", "input", "output"],
}


# ---------------------------------------------------------------------------
# Corpus loading and passage extraction
# ---------------------------------------------------------------------------

def load_corpus(corpus_path: str) -> list[str]:
    """Load the ALS training corpus and split into documents.

    Reads the full corpus file and splits on the ``<|endoftext|>`` delimiter
    to produce a list of individual document strings.

    Args:
        corpus_path: Path to the training corpus text file.

    Returns:
        List of non-empty document strings.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    docs = text.split("<|endoftext|>")
    docs = [d.strip() for d in docs if d.strip()]
    return docs


def _score_document(doc: str, keywords: list[str]) -> int:
    """Score a document by counting keyword occurrences.

    Args:
        doc: Document text (lowercased for matching).
        keywords: List of keyword strings to search for.

    Returns:
        Total number of keyword hits found in the document.
    """
    doc_lower = doc.lower()
    score = 0
    for kw in keywords:
        score += doc_lower.count(kw.lower())
    return score


def _is_reference_heavy(text: str) -> bool:
    """Check if a text passage is dominated by references.

    A passage is considered reference-heavy if it has more than 3 "et al."
    occurrences or more than 5 numbered citation patterns like [1], [2].

    Args:
        text: Passage text to check.

    Returns:
        True if the passage is reference-heavy.
    """
    et_al_count = text.lower().count("et al.")
    citation_count = len(re.findall(r"\[\d+\]", text))
    return et_al_count > 3 or citation_count > 5


def extract_passages(
    docs: list[str],
    category: str,
    seed: int = 42,
    target_passages: int = 350,
    min_chars: int = 200,
    min_words: int = 50,
) -> list[dict]:
    """Extract category-relevant passages from corpus documents.

    Scores documents by keyword hit count for the given category, then
    extracts 300-500 character passages centered on keyword hits with
    100-char overlap. Filters out passages that are too short or are
    dominated by references.

    Args:
        docs: List of document strings from the corpus.
        category: Category name (key in CATEGORY_KEYWORDS).
        seed: Random seed for reproducible sampling.
        target_passages: Target number of passages to extract.
        min_chars: Minimum passage length in characters.
        min_words: Minimum passage length in words.

    Returns:
        List of dicts with keys: text, doc_index, category.
    """
    rng = random.Random(seed)
    keywords = CATEGORY_KEYWORDS[category]

    # Score all documents for this category
    scored = []
    for i, doc in enumerate(docs):
        score = _score_document(doc, keywords)
        if score > 0:
            scored.append((i, doc, score))

    # Sort by score descending and take top documents
    scored.sort(key=lambda x: x[2], reverse=True)
    top_docs = scored[:max(len(scored), 1)]

    # Extract passages from top-scoring documents
    passages = []
    chunk_size = 400  # Target 300-500 chars
    overlap = 100

    for doc_idx, doc, _score in top_docs:
        if len(passages) >= target_passages:
            break

        doc_text = doc.strip()
        if len(doc_text) < min_chars:
            continue

        # For short documents, use the whole text as one passage
        if len(doc_text) <= chunk_size + overlap:
            if (len(doc_text) >= min_chars
                    and len(doc_text.split()) >= min_words
                    and not _is_reference_heavy(doc_text)):
                passages.append({
                    "text": doc_text,
                    "doc_index": doc_idx,
                    "category": category,
                })
            continue

        # Extract overlapping chunks from longer documents
        pos = 0
        while pos < len(doc_text) and len(passages) < target_passages:
            end = min(pos + chunk_size, len(doc_text))
            chunk = doc_text[pos:end]

            # Adjust to avoid splitting mid-word
            if end < len(doc_text):
                last_space = chunk.rfind(" ")
                if last_space > chunk_size // 2:
                    chunk = chunk[:last_space]

            if (len(chunk) >= min_chars
                    and len(chunk.split()) >= min_words
                    and not _is_reference_heavy(chunk)):
                # Check that the chunk contains at least one keyword
                chunk_lower = chunk.lower()
                has_keyword = any(
                    kw.lower() in chunk_lower for kw in keywords
                )
                if has_keyword:
                    passages.append({
                        "text": chunk,
                        "doc_index": doc_idx,
                        "category": category,
                    })

            pos += chunk_size - overlap

    # Shuffle and limit to target
    rng.shuffle(passages)
    return passages[:target_passages]


# ---------------------------------------------------------------------------
# Question type selection and input field logic
# ---------------------------------------------------------------------------

def select_question_type() -> str:
    """Select a question type using the target distribution.

    Distribution: ~40% factual, ~30% explanation, ~15% comparison,
    ~15% listing. Uses the current state of the random module.

    Returns:
        One of: "factual", "explanation", "comparison", "listing".
    """
    r = random.random()
    if r < 0.40:
        return "factual"
    elif r < 0.70:
        return "explanation"
    elif r < 0.85:
        return "comparison"
    else:
        return "listing"


def decide_input_field(passage: str) -> str:
    """Decide whether to populate the input field for a Q&A pair.

    Returns an empty string ~80% of the time and a passage excerpt
    ~20% of the time to provide additional context for the question.

    Args:
        passage: Source passage text.

    Returns:
        Empty string or a relevant excerpt from the passage.
    """
    if random.random() < 0.80:
        return ""

    # Extract a meaningful excerpt (first sentence or up to 150 chars)
    sentences = re.split(r"(?<=[.!?])\s+", passage)
    if sentences:
        excerpt = sentences[0]
        if len(excerpt) > 150:
            excerpt = excerpt[:147] + "..."
        return excerpt
    return passage[:150]


# ---------------------------------------------------------------------------
# Prompt building for Ollama Q&A generation
# ---------------------------------------------------------------------------

# Question type directive templates
_QUESTION_TYPE_DIRECTIVES = {
    "factual": (
        "Ask a factual recall question starting with 'What is...', "
        "'What are...', or 'Which...' about the key facts in the passage."
    ),
    "explanation": (
        "Ask an explanation question starting with 'Explain how...', "
        "'Describe the mechanism...', or 'How does...' about a process "
        "or mechanism described in the passage."
    ),
    "comparison": (
        "Ask a comparison question starting with 'How does X differ "
        "from Y?' or 'Compare...' about two concepts mentioned in the "
        "passage."
    ),
    "listing": (
        "Ask a listing question starting with 'What are the "
        "symptoms/types/features of...' or 'List the...' about items "
        "enumerated or described in the passage."
    ),
}


def build_generation_prompt(
    passage: str,
    category: str,
    question_type: str,
) -> str:
    """Build the prompt for Ollama Q&A pair generation.

    Constructs a prompt that instructs the model to generate an Alpaca-format
    Q&A pair based on a specific corpus passage, category, and question type.

    Args:
        passage: Source passage text from the ALS corpus.
        category: ALS knowledge category name.
        question_type: One of "factual", "explanation", "comparison", "listing".

    Returns:
        Formatted prompt string for Ollama.
    """
    directive = _QUESTION_TYPE_DIRECTIVES.get(question_type, _QUESTION_TYPE_DIRECTIVES["factual"])
    category_display = category.replace("_", " ")

    prompt = (
        f"You are an expert medical researcher specializing in ALS "
        f"(amyotrophic lateral sclerosis). Based on the following passage "
        f"from the {category_display} domain, generate a question-answer "
        f"pair.\n\n"
        f"PASSAGE:\n{passage}\n\n"
        f"INSTRUCTIONS:\n"
        f"- {directive}\n"
        f"- Write a concise answer in 2-4 sentences using clinical and "
        f"academic tone.\n"
        f"- Base your answer ONLY on information from the passage above.\n"
        f"- Do NOT include citations, references, or source attributions.\n"
        f"- Do NOT include medical disclaimers or safety warnings.\n"
        f"- The 'instruction' field should contain the question.\n"
        f"- The 'input' field should be an empty string.\n"
        f"- The 'output' field should contain the answer.\n\n"
        f"Generate the question-answer pair as JSON with 'instruction', "
        f"'input', and 'output' fields."
    )
    return prompt


# ---------------------------------------------------------------------------
# Pair validation
# ---------------------------------------------------------------------------

def validate_pair_format(pair: dict) -> bool:
    """Validate that an Alpaca pair has the required format.

    Checks that the pair has instruction, input, and output keys, that
    instruction and output are non-empty, and that the output is between
    10 and 200 words.

    Args:
        pair: Dictionary to validate.

    Returns:
        True if the pair is valid, False otherwise.
    """
    required_keys = {"instruction", "input", "output"}
    if not required_keys.issubset(pair.keys()):
        return False

    instruction = pair.get("instruction", "")
    output = pair.get("output", "")

    if not isinstance(instruction, str) or not instruction.strip():
        return False
    if not isinstance(output, str) or not output.strip():
        return False

    word_count = len(output.split())
    if word_count < 10 or word_count > 200:
        return False

    return True


# ---------------------------------------------------------------------------
# Progress save and resume
# ---------------------------------------------------------------------------

def save_progress(data: dict, filepath: str) -> None:
    """Save generation progress to a JSON file.

    Writes to a temporary file first and then renames to avoid data
    corruption if the process is interrupted during writing.

    Args:
        data: Progress data dictionary.
        filepath: Path to the progress file.
    """
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, filepath)


def load_progress(filepath: str) -> Optional[dict]:
    """Load generation progress from a JSON file.

    Args:
        filepath: Path to the progress file.

    Returns:
        Progress data dictionary, or None if the file does not exist.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ollama API interaction
# ---------------------------------------------------------------------------

def check_model_available(ollama_url: str, model_name: str) -> bool:
    """Check if the specified model is available in Ollama.

    Queries the Ollama /api/tags endpoint to verify the model is pulled.

    Args:
        ollama_url: Base URL of the Ollama server.
        model_name: Ollama model tag to check.

    Returns:
        True if the model is available.

    Raises:
        SystemExit: If the model is not available or Ollama is unreachable.
    """
    try:
        resp = requests.get(
            f"{ollama_url.rstrip('/')}/api/tags",
            timeout=10,
        )
        resp.raise_for_status()
        models = resp.json().get("models", [])
        available = [m.get("name", "") for m in models]
        # Check both exact match and prefix match (e.g. "llama3.1:8b" in "llama3.1:8b-instruct-q8_0")
        for name in available:
            if model_name in name or name.startswith(model_name.split(":")[0]):
                return True
        print(f"ERROR: Model '{model_name}' not found in Ollama.")
        print(f"Available models: {available}")
        print(f"Run: ollama pull {model_name}")
        sys.exit(1)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to Ollama at {ollama_url}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)


def generate_qa_pair(
    passage: str,
    category: str,
    question_type: str,
    input_field: str,
    model: str,
    ollama_url: str,
    max_retries: int = 3,
) -> Optional[dict]:
    """Generate one Alpaca Q&A pair from a corpus passage via Ollama.

    Calls the Ollama /api/generate endpoint with structured JSON output
    using the ALPACA_SCHEMA. Retries on connection errors and server errors.

    Args:
        passage: Source passage text.
        category: ALS knowledge category.
        question_type: Type of question to generate.
        input_field: Pre-decided input field value (empty or excerpt).
        model: Ollama model tag.
        ollama_url: Base URL of the Ollama server.
        max_retries: Maximum retry attempts.

    Returns:
        Validated Alpaca pair dict with metadata, or None on failure.
    """
    prompt = build_generation_prompt(passage, category, question_type)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": ALPACA_SCHEMA,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        },
    }

    url = f"{ollama_url.rstrip('/')}/api/generate"
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code >= 500 and attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries} after HTTP {resp.status_code}")
                time.sleep(5)
                continue
            resp.raise_for_status()

            result = json.loads(resp.json()["response"])

            # Override input field with pre-decided value
            result["input"] = input_field

            # Validate the pair
            if not validate_pair_format(result):
                return None

            # Add metadata
            result["metadata"] = {
                "category": category,
                "question_type": question_type,
                "source_doc_index": -1,  # Set by caller
                "passage_preview": passage[:80],
            }
            return result

        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt < max_retries:
                print(f"    Retry {attempt}/{max_retries} after error: {exc}")
                time.sleep(5)
            else:
                print(f"    Failed after {max_retries} retries: {exc}")
                return None
        except (json.JSONDecodeError, KeyError, requests.HTTPError) as exc:
            print(f"    Generation error: {exc}")
            return None

    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Run the instruction dataset generation pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate Alpaca-format ALS instruction pairs from corpus "
            "passages via Ollama"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model tag (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/instruction/als_instructions.json",
        help="Output JSON file path (default: data/instruction/als_instructions.json)",
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=250,
        help="Target number of pairs per category (default: 250)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress file if available",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    corpus_path = os.path.join(_project_root, "data", "processed", "train.txt")
    output_path = (
        os.path.join(_project_root, args.output)
        if not os.path.isabs(args.output)
        else args.output
    )
    progress_path = output_path.replace(".json", "_progress.json")

    random.seed(args.seed)

    print("ALS Instruction Dataset Generation")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Ollama URL:         {args.ollama_url}")
    print(f"  Output:             {output_path}")
    print(f"  Target/category:    {args.target_per_category}")
    print(f"  Seed:               {args.seed}")
    print(f"  Resume:             {args.resume}")
    print()

    # Check model availability
    check_model_available(args.ollama_url, args.model)
    print(f"Model '{args.model}' is available.")
    print()

    # Load corpus
    print(f"Loading corpus from {corpus_path}...")
    start_load = time.time()
    docs = load_corpus(corpus_path)
    load_time = time.time() - start_load
    print(f"Loaded {len(docs)} documents in {load_time:.1f}s")
    print()

    # Load or initialize progress
    all_pairs = []
    completed_categories = set()
    if args.resume:
        progress = load_progress(progress_path)
        if progress:
            all_pairs = progress.get("pairs", [])
            completed_categories = set(progress.get("completed_categories", []))
            print(f"Resumed: {len(all_pairs)} pairs, "
                  f"{len(completed_categories)} categories complete")
        else:
            print("No progress file found, starting fresh.")
    print()

    # Generate pairs for each category
    start_gen = time.time()
    rejected_count = 0
    categories = sorted(CATEGORY_KEYWORDS.keys())

    for category in categories:
        if category in completed_categories:
            print(f"[{category}] Already complete, skipping.")
            continue

        print(f"[{category}] Extracting passages...")
        passages = extract_passages(
            docs, category, seed=args.seed,
            target_passages=args.target_per_category + 100,
        )
        print(f"[{category}] Found {len(passages)} passages")

        category_pairs = 0
        category_rejected = 0

        for i, passage_info in enumerate(passages):
            if category_pairs >= args.target_per_category:
                break

            question_type = select_question_type()
            input_field = decide_input_field(passage_info["text"])

            pair = generate_qa_pair(
                passage=passage_info["text"],
                category=category,
                question_type=question_type,
                input_field=input_field,
                model=args.model,
                ollama_url=args.ollama_url,
            )

            if pair is not None:
                pair["metadata"]["source_doc_index"] = passage_info["doc_index"]
                all_pairs.append(pair)
                category_pairs += 1
            else:
                category_rejected += 1

            # Print progress
            total = len(all_pairs)
            target_total = args.target_per_category * len(categories)
            print(
                f"\r  [{category}] {category_pairs}/{args.target_per_category} "
                f"pairs generated ({100*category_pairs/args.target_per_category:.1f}%) "
                f"| Total: {total}/{target_total}",
                end="", flush=True,
            )

            # Save progress every 50 pairs
            if total % 50 == 0 and total > 0:
                save_progress(
                    {
                        "pairs": all_pairs,
                        "completed_categories": list(completed_categories),
                        "current_category": category,
                        "current_index": i,
                    },
                    progress_path,
                )

        print()
        rejected_count += category_rejected
        completed_categories.add(category)

        # Save after each category
        save_progress(
            {
                "pairs": all_pairs,
                "completed_categories": list(completed_categories),
                "current_category": category,
                "current_index": len(passages),
            },
            progress_path,
        )

    gen_time = time.time() - start_gen

    # Sort by category for organized output
    all_pairs.sort(key=lambda p: p.get("metadata", {}).get("category", ""))

    # Write final output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    # Clean up progress file
    if os.path.exists(progress_path):
        os.remove(progress_path)

    # Print summary
    print()
    print("=" * 60)
    print("Generation Summary")
    print("=" * 60)
    print(f"  Total pairs:    {len(all_pairs)}")
    print(f"  Rejected:       {rejected_count}")
    print(f"  Elapsed time:   {gen_time:.1f}s ({gen_time/60:.1f}m)")
    print()

    # Per-category counts
    cat_counts = {}
    for pair in all_pairs:
        cat = pair.get("metadata", {}).get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("Per-category breakdown:")
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]}")
    print()

    # Question type counts
    qt_counts = {}
    for pair in all_pairs:
        qt = pair.get("metadata", {}).get("question_type", "unknown")
        qt_counts[qt] = qt_counts.get(qt, 0) + 1

    print("Question type breakdown:")
    for qt in sorted(qt_counts):
        print(f"  {qt}: {qt_counts[qt]}")
    print()

    print(f"Output written to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
