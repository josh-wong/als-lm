#!/usr/bin/env python3
"""Generate RAG-augmented responses and run the evaluation pipeline.

Sweeps all 4 ChromaDB collection configurations (2 chunk sizes x 2 embedding
models), retrieves top-k chunks per benchmark question, injects context into
the Ollama chat prompt, generates responses, runs the existing 6-stage eval
pipeline, and appends retrieval analysis (hit rate + context window
utilization) to each config's report.

This is the core implementation of the RAG comparison pipeline. It composes
existing components (ChromaDB querying from Phase 14, Ollama chat generation
and eval orchestration from Phase 15, fuzzy matching from the eval pipeline)
into a single sweep script that produces self-contained evaluation results
per configuration.

The baseline results from rag/generate_baseline.py serve as the attribution
boundary: anything the RAG pipeline gets right that the baseline gets wrong is
directly attributable to retrieval augmentation.

This is a research evaluation tool, not a medical information system.

Usage examples::

    # Run full sweep with defaults (4 configs, context_only template)
    python rag/generate_rag.py

    # Use the combined-knowledge template across all configs
    python rag/generate_rag.py --template combined

    # Sweep both templates across all configs (8 configs total)
    python rag/generate_rag.py --all-templates

    # Resume an interrupted sweep
    python rag/generate_rag.py --resume

    # Run a single config only
    python rag/generate_rag.py --configs 500_minilm

    # Generate responses only (skip eval stages 2-6)
    python rag/generate_rag.py --skip-eval
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

# Ensure the project root is on sys.path so that imports from eval/ and rag/
# resolve correctly when running as `python rag/generate_rag.py`.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eval.generate_responses import is_coherent
from eval.utils import relativize_path
from eval.utils import find_project_root, resolve_default_paths
from rag.chunker import count_tokens
from rag.ollama_utils import (
    check_ollama_running,
    check_model_available,
    generate_chat_response,
    save_responses,
    run_eval_stages,
)

try:
    from rapidfuzz import fuzz
except ImportError:
    print("ERROR: rapidfuzz is required. Install with: pip install rapidfuzz")
    sys.exit(1)

# Project root and default paths
PROJECT_ROOT = find_project_root()
DEFAULTS = resolve_default_paths(PROJECT_ROOT)

# Default system prompt: identical to baseline (Phase 15) so the ONLY variable
# between baseline and RAG is whether retrieved chunks are injected into the
# user message.
DEFAULT_SYSTEM_PROMPT = (
    "Answer the following question about ALS (amyotrophic lateral sclerosis) "
    "based on your knowledge. Be concise and factual."
)

# ---------------------------------------------------------------------------
# Collection and template configurations
# ---------------------------------------------------------------------------

COLLECTION_CONFIGS = [
    {"chunk_size": 500, "embedding": "minilm"},
    {"chunk_size": 200, "embedding": "minilm"},
    {"chunk_size": 500, "embedding": "pubmedbert"},
    {"chunk_size": 200, "embedding": "pubmedbert"},
]

PROMPT_TEMPLATES = {
    "context_only": (
        "Context:\n"
        "{context_block}\n\n"
        "Using only the context above, answer the following question. "
        "If the answer is not found in the context, say "
        '"Not found in context."\n'
        "Question: {question}"
    ),
    "combined": (
        "Context:\n"
        "{context_block}\n\n"
        "Using the context above and your own knowledge, answer the "
        "following question.\n"
        "Question: {question}"
    ),
}

# Embedding model names matching rag/index_corpus.py. Query-time functions
# use device="cpu" to avoid GPU memory contention with Ollama.
EMBEDDING_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "pubmedbert": "NeuML/pubmedbert-base-embeddings",
}


def _get_query_embedding_function(embedding_name):
    """Get a CPU-only embedding function for query-time retrieval.

    Uses the same models as index_corpus.py but forces device="cpu" to
    avoid GPU memory contention with Ollama. The sentence-transformers
    import is deferred to avoid loading PyTorch at module import time.
    """
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )

    model_name = EMBEDDING_MODELS[embedding_name]
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments for RAG generation and evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate RAG-augmented responses from an Ollama model using "
            "ChromaDB-indexed ALS corpus chunks, and run the full evaluation "
            "pipeline (stages 2-6) per collection configuration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python rag/generate_rag.py\n"
            "  python rag/generate_rag.py --template combined\n"
            "  python rag/generate_rag.py --all-templates\n"
            "  python rag/generate_rag.py --resume\n"
            "  python rag/generate_rag.py --configs 500_minilm\n"
        ),
    )
    parser.add_argument(
        "--ollama-model",
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
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the chat API (default: same as baseline)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response (default: 512)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULTS["benchmark"]),
        help=f"Path to benchmark questions JSON (default: {DEFAULTS['benchmark']})",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=str(DEFAULTS["registry"]),
        help=f"Path to entity registry JSON (default: {DEFAULTS['registry']})",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=str(PROJECT_ROOT / "rag" / "results"),
        help="Parent directory for per-config output subdirs (default: rag/results)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="context_only",
        help=(
            "Prompt template name or custom template string "
            "(default: context_only, built-in: context_only, combined)"
        ),
    )
    parser.add_argument(
        "--all-templates",
        action="store_true",
        help="Sweep both templates across all configs (produces 8 configs)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=8192,
        help="Ollama context window size for augmented prompts (default: 8192)",
    )
    parser.add_argument(
        "--chroma-db",
        type=str,
        default=os.environ.get("ALS_CHROMADB_PATH", str(PROJECT_ROOT / "data" / "chromadb")),
        help="ChromaDB persistent storage path (env: ALS_CHROMADB_PATH, default: data/chromadb)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed configs and resume interrupted ones",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Generate responses only, skip evaluation stages 2-6",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help=(
            "Comma-separated config filter (e.g., '500_minilm,200_pubmedbert'). "
            "Only matching configs will be run."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Retrieval and prompt construction
# ---------------------------------------------------------------------------

def retrieve_and_build_prompt(collection, question_text, template_name, top_k):
    """Retrieve chunks from ChromaDB and build the augmented prompt.

    Queries the collection for the top-k most similar chunks to the question,
    formats them into a labeled context block, and applies the selected
    prompt template.

    Args:
        collection: ChromaDB collection to query.
        question_text: The benchmark question text.
        template_name: Key in PROMPT_TEMPLATES or a custom template string.
        top_k: Number of chunks to retrieve.

    Returns:
        (user_message, retrieval_info) where user_message is the formatted
        prompt and retrieval_info is a dict with chunks, distances, metadatas,
        and token counts.
    """
    results = collection.query(
        query_texts=[question_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    # Build labeled context block
    context_lines = []
    for i, doc in enumerate(documents, 1):
        context_lines.append(f"[{i}] {doc}")
    context_block = "\n".join(context_lines)

    # Compute token counts for the context and full prompt
    context_token_count = count_tokens(context_block)

    # Apply template
    if template_name in PROMPT_TEMPLATES:
        template = PROMPT_TEMPLATES[template_name]
    else:
        # Custom template string -- must contain {context_block} and {question}
        template = template_name

    user_message = template.format(
        context_block=context_block,
        question=question_text,
    )

    augmented_prompt_token_count = count_tokens(user_message)

    retrieval_info = {
        "chunks": documents,
        "distances": [float(d) for d in distances],
        "metadatas": metadatas,
        "context_token_count": context_token_count,
        "augmented_prompt_token_count": augmented_prompt_token_count,
    }

    return user_message, retrieval_info


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

def build_config_list(args):
    """Build the list of configurations to sweep.

    Each config dict contains chunk_size, embedding, template, config_name,
    and output_dir. The config_name determines the subdirectory name under
    output_base.

    When --all-templates is set, crosses COLLECTION_CONFIGS with both
    templates (8 configs). Otherwise, uses the single --template value
    (4 configs).
    """
    if args.all_templates:
        templates = list(PROMPT_TEMPLATES.keys())
    else:
        templates = [args.template]

    configs = []
    for cc in COLLECTION_CONFIGS:
        for tmpl in templates:
            # Config naming: rag_{chunk_size}_{embedding} for default template,
            # rag_{chunk_size}_{embedding}_{template} for non-default
            config_key = f"{cc['chunk_size']}_{cc['embedding']}"
            if tmpl != "context_only" or args.all_templates:
                config_name = f"rag_{config_key}_{tmpl}"
            else:
                config_name = f"rag_{config_key}"

            configs.append({
                "chunk_size": cc["chunk_size"],
                "embedding": cc["embedding"],
                "template": tmpl,
                "config_name": config_name,
                "config_key": config_key,
                "output_dir": os.path.join(args.output_base, config_name),
            })

    # Apply --configs filter if specified
    if args.configs:
        filter_keys = [k.strip() for k in args.configs.split(",")]
        configs = [
            c for c in configs
            if c["config_key"] in filter_keys or c["config_name"] in filter_keys
        ]

    return configs


def is_config_complete(output_dir, total_questions):
    """Check if a config's responses.json exists and has all questions."""
    responses_path = os.path.join(output_dir, "responses.json")
    if not os.path.isfile(responses_path):
        return False
    try:
        with open(responses_path) as f:
            data = json.load(f)
        return len(data.get("responses", [])) >= total_questions
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def load_partial_responses(output_dir):
    """Load partial responses from an interrupted config for resume.

    Returns a dict mapping question_id to response entry.
    """
    responses_path = os.path.join(output_dir, "responses.json")
    if not os.path.isfile(responses_path):
        return {}
    try:
        with open(responses_path) as f:
            data = json.load(f)
        return {r["question_id"]: r for r in data.get("responses", [])}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Retrieval analysis
# ---------------------------------------------------------------------------

def compute_hit_rate(questions, responses, threshold=80):
    """Compute retrieval hit rate using fuzzy matching of key_facts.

    For each question, checks whether each key_fact appears in the
    concatenated chunk text using rapidfuzz partial_ratio. A key_fact is
    considered "found" if any chunk has a partial_ratio score >= threshold.

    Returns (hit_rate, found_count, total_facts, per_question_hits).
    """
    total_facts = 0
    found_count = 0
    per_question_hits = []

    response_map = {r["question_id"]: r for r in responses}

    for question in questions:
        qid = question["id"]
        key_facts = question.get("key_facts", [])
        response = response_map.get(qid)

        if not response or "retrieval" not in response or not key_facts:
            per_question_hits.append({
                "question_id": qid,
                "category": question["category"],
                "total_facts": len(key_facts),
                "found_facts": 0,
                "hit_rate": 0.0,
            })
            total_facts += len(key_facts)
            continue

        chunks = response["retrieval"]["chunks"]
        concatenated = " ".join(chunks).lower()
        q_found = 0

        for fact in key_facts:
            score = fuzz.partial_ratio(fact.lower(), concatenated)
            if score >= threshold:
                q_found += 1

        total_facts += len(key_facts)
        found_count += q_found

        q_hit_rate = q_found / len(key_facts) if key_facts else 0.0
        per_question_hits.append({
            "question_id": qid,
            "category": question["category"],
            "total_facts": len(key_facts),
            "found_facts": q_found,
            "hit_rate": q_hit_rate,
        })

    overall_hit_rate = found_count / total_facts if total_facts > 0 else 0.0
    return overall_hit_rate, found_count, total_facts, per_question_hits


def compute_context_stats(responses):
    """Compute context window utilization statistics across all responses.

    Reads the augmented_prompt_token_count from each response's retrieval
    field to compute mean, max, min, and median token counts.
    """
    token_counts = []
    for r in responses:
        retrieval = r.get("retrieval", {})
        count = retrieval.get("augmented_prompt_token_count", 0)
        if count > 0:
            token_counts.append(count)

    if not token_counts:
        return {
            "mean_tokens": 0,
            "max_tokens": 0,
            "min_tokens": 0,
            "median_tokens": 0,
            "total_responses": 0,
        }

    return {
        "mean_tokens": statistics.mean(token_counts),
        "max_tokens": max(token_counts),
        "min_tokens": min(token_counts),
        "median_tokens": statistics.median(token_counts),
        "total_responses": len(token_counts),
    }


def append_retrieval_analysis(report_path, questions, responses):
    """Append a Retrieval Analysis section to the existing Markdown report.

    Computes hit rate and context window utilization, then appends formatted
    analysis to the eval report including per-category breakdown and
    retrieval-vs-generation failure correlation.
    """
    if not os.path.isfile(report_path):
        print(f"  WARNING: Report file not found: {report_path}")
        return

    hit_rate, found, total, per_question_hits = compute_hit_rate(
        questions, responses
    )
    ctx_stats = compute_context_stats(responses)

    # Per-category hit rate breakdown
    category_stats = {}
    for hit in per_question_hits:
        cat = hit["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total_facts": 0, "found_facts": 0}
        category_stats[cat]["total_facts"] += hit["total_facts"]
        category_stats[cat]["found_facts"] += hit["found_facts"]

    # Load scores.json for retrieval-vs-generation correlation.
    # The scores schema uses per_question with accuracy_binary (0/1).
    scores_path = os.path.join(os.path.dirname(report_path), "scores.json")
    category_accuracy = {}
    if os.path.isfile(scores_path):
        try:
            with open(scores_path) as f:
                scores_data = json.load(f)
            for entry in scores_data.get("per_question", []):
                cat = entry.get("category", "unknown")
                if cat not in category_accuracy:
                    category_accuracy[cat] = {"correct": 0, "total": 0}
                category_accuracy[cat]["total"] += 1
                if entry.get("accuracy_binary", 0) == 1:
                    category_accuracy[cat]["correct"] += 1
        except (json.JSONDecodeError, KeyError, OSError):
            pass

    # Build the retrieval analysis section
    lines = [
        "",
        "## Retrieval analysis",
        "",
        "Retrieval diagnostics measure how well the ChromaDB index surfaces "
        "relevant information for each benchmark question.",
        "",
        "### Hit rate",
        "",
        f"Key facts found in retrieved chunks: **{found}/{total}** "
        f"(**{hit_rate:.1%}**)",
        "",
        "Matching uses rapidfuzz partial_ratio with threshold >= 80.",
        "",
        "### Per-category retrieval breakdown",
        "",
        "| Category | Facts found | Total facts | Hit rate |",
        "| -------- | ----------: | ----------: | -------: |",
    ]

    for cat in sorted(category_stats.keys()):
        stats = category_stats[cat]
        cat_hr = (
            stats["found_facts"] / stats["total_facts"]
            if stats["total_facts"] > 0
            else 0.0
        )
        lines.append(
            f"| {cat} | {stats['found_facts']} | "
            f"{stats['total_facts']} | {cat_hr:.1%} |"
        )

    lines.extend([
        "",
        "### Context window utilization",
        "",
        f"Augmented prompt token counts across {ctx_stats['total_responses']} "
        f"responses:",
        "",
        f"- **Mean:** {ctx_stats['mean_tokens']:.0f} tokens",
        f"- **Median:** {ctx_stats['median_tokens']:.0f} tokens",
        f"- **Min:** {ctx_stats['min_tokens']} tokens",
        f"- **Max:** {ctx_stats['max_tokens']} tokens",
    ])

    # Retrieval vs generation correlation
    if category_accuracy and category_stats:
        lines.extend([
            "",
            "### Retrieval vs generation correlation",
            "",
            "Categories where retrieval hit rate diverges from answer accuracy "
            "indicate generation failures (high retrieval, low accuracy) or "
            "parametric knowledge contributions (low retrieval, high accuracy).",
            "",
            "| Category | Hit rate | Accuracy | Pattern |",
            "| -------- | -------: | -------: | ------- |",
        ])

        for cat in sorted(category_stats.keys()):
            c_stats = category_stats[cat]
            cat_hr = (
                c_stats["found_facts"] / c_stats["total_facts"]
                if c_stats["total_facts"] > 0
                else 0.0
            )
            acc_info = category_accuracy.get(cat, {"correct": 0, "total": 0})
            cat_acc = (
                acc_info["correct"] / acc_info["total"]
                if acc_info["total"] > 0
                else 0.0
            )

            if cat_hr >= 0.5 and cat_acc < 0.3:
                pattern = "Generation failure"
            elif cat_hr < 0.3 and cat_acc >= 0.5:
                pattern = "Parametric knowledge"
            elif cat_hr >= 0.5 and cat_acc >= 0.5:
                pattern = "Retrieval success"
            elif cat_hr < 0.3 and cat_acc < 0.3:
                pattern = "Retrieval failure"
            else:
                pattern = "Mixed"

            lines.append(
                f"| {cat} | {cat_hr:.1%} | {cat_acc:.1%} | {pattern} |"
            )

    lines.append("")

    # Append to report
    with open(report_path, "a") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _build_metadata(args, config, total_questions, completed_questions,
                    collection_name):
    """Build the metadata block for responses.json."""
    return {
        "inference_mode": "ollama",
        "ollama_model": args.ollama_model,
        "ollama_url": args.ollama_url,
        "system_prompt": args.system_prompt,
        "generation_params": {
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "num_ctx": args.num_ctx,
        },
        "rag_params": {
            "collection_name": collection_name,
            "chunk_size": config["chunk_size"],
            "embedding": config["embedding"],
            "template": config["template"],
            "top_k": args.top_k,
        },
        "benchmark_path": relativize_path(str(args.benchmark)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": total_questions,
        "completed_questions": completed_questions,
        "eval_type": "rag",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for RAG generation and evaluation sweep."""
    args = parse_args()

    print("\n=== ALS RAG Evaluation Pipeline ===\n")
    print("  NOTE: This is a research evaluation tool, not a medical"
          " information system.\n")

    # Resolve paths
    benchmark_path = os.path.abspath(args.benchmark)
    registry_path = os.path.abspath(args.registry)
    chroma_db_path = os.path.abspath(args.chroma_db)

    # --- Pre-flight checks ---------------------------------------------------

    print("  Pre-flight checks:")
    available_models = check_ollama_running(args.ollama_url)
    print(f"    Ollama server: OK ({len(available_models)} models available)")

    check_model_available(args.ollama_url, args.ollama_model, available_models)
    print(f"    Model '{args.ollama_model}': OK")

    if not os.path.isfile(benchmark_path):
        print(f"\n  ERROR: Benchmark file not found: {benchmark_path}")
        sys.exit(1)
    print(f"    Benchmark: {benchmark_path}")

    if not os.path.isfile(registry_path):
        print(f"\n  ERROR: Entity registry not found: {registry_path}")
        print("  Run eval/build_entity_registry.py first.")
        sys.exit(1)
    print(f"    Registry: {registry_path}")

    # Verify ChromaDB path exists
    import chromadb

    if not os.path.isdir(chroma_db_path):
        print(f"\n  ERROR: ChromaDB directory not found: {chroma_db_path}")
        print("  Run rag/index_corpus.py first to index the corpus.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=chroma_db_path)
    existing_collections = []
    for c in client.list_collections():
        name = c.name if hasattr(c, "name") else c
        existing_collections.append(name)
    print(f"    ChromaDB: {len(existing_collections)} collections at {chroma_db_path}")

    # --- Load benchmark questions --------------------------------------------

    with open(benchmark_path) as f:
        questions = json.load(f)
    total_questions = len(questions)
    print(f"\n  Benchmark: {total_questions} questions loaded")

    # --- Build config list ---------------------------------------------------

    configs = build_config_list(args)
    if not configs:
        print("\n  ERROR: No configs matched the filter. Check --configs value.")
        sys.exit(1)

    # Verify required collections exist
    for config in configs:
        col_name = f"als_{config['chunk_size']}_{config['embedding']}"
        if col_name not in existing_collections:
            print(f"\n  ERROR: Collection '{col_name}' not found in ChromaDB")
            print(f"  Available: {existing_collections}")
            print(f"  Run: python rag/index_corpus.py --chunk-size "
                  f"{config['chunk_size']} --embedding {config['embedding']}")
            sys.exit(1)

    print(f"  Configs to sweep: {len(configs)}")
    for config in configs:
        print(f"    - {config['config_name']} (template: {config['template']})")
    print()

    # --- Sweep loop ----------------------------------------------------------

    sweep_start = time.time()
    completed_configs = []
    skipped_configs = []

    # Cache embedding functions to avoid reloading for same embedding model
    embedding_cache = {}

    for config_idx, config in enumerate(configs, 1):
        config_name = config["config_name"]
        output_dir = config["output_dir"]
        collection_name = f"als_{config['chunk_size']}_{config['embedding']}"
        responses_path = os.path.join(output_dir, "responses.json")

        print(f"{'=' * 60}")
        print(f"  Config [{config_idx}/{len(configs)}]: {config_name}")
        print(f"  Collection: {collection_name}")
        print(f"  Template: {config['template']}")
        print(f"  Output: {output_dir}")
        print(f"{'=' * 60}")

        # 1. Skip if complete (resume mode)
        if args.resume and is_config_complete(output_dir, total_questions):
            print(f"  SKIP: Config already complete ({total_questions} responses)")

            # Still run eval + retrieval analysis if not skipping eval
            if not args.skip_eval:
                report_path = os.path.join(
                    output_dir, f"hallucination_eval_{config_name}.md"
                )
                if not os.path.isfile(report_path):
                    print("\n  Running evaluation stages...\n")
                    success = run_eval_stages(
                        output_dir, benchmark_path, registry_path,
                        PROJECT_ROOT, report_suffix=config_name
                    )
                    if success:
                        with open(responses_path) as f:
                            saved_responses = json.load(f)["responses"]
                        append_retrieval_analysis(
                            report_path, questions, saved_responses
                        )

            skipped_configs.append(config_name)
            print()
            continue

        # 2. Open ChromaDB collection with CPU embedding function
        embedding_name = config["embedding"]
        if embedding_name not in embedding_cache:
            print(f"  Loading {embedding_name} embedding model (CPU)...")
            embedding_cache[embedding_name] = _get_query_embedding_function(
                embedding_name
            )
        ef = embedding_cache[embedding_name]

        collection = client.get_collection(
            collection_name, embedding_function=ef
        )
        chunk_count = collection.count()
        print(f"  Collection loaded: {chunk_count:,} chunks")

        # 3. Load partial responses for resume
        existing_responses = {}
        if args.resume:
            existing_responses = load_partial_responses(output_dir)
            if existing_responses:
                print(f"  Resume: {len(existing_responses)} existing responses")

        # 4. Generate responses for remaining questions
        os.makedirs(output_dir, exist_ok=True)

        questions_to_run = [
            q for q in questions if q["id"] not in existing_responses
        ]

        if not questions_to_run:
            print("  All questions already completed.")
        else:
            print(f"\n  Generating {len(questions_to_run)} responses "
                  f"(top-k={args.top_k}, num_ctx={args.num_ctx})...\n")

            responses = list(existing_responses.values())
            gen_start = time.time()

            for i, question in enumerate(questions_to_run):
                qid = question["id"]
                category = question["category"]
                prompt_text = question["prompt_template"]

                # Retrieve and build augmented prompt
                user_message, retrieval_info = retrieve_and_build_prompt(
                    collection, prompt_text, config["template"], args.top_k
                )

                # Generate response
                response_text, tokens_generated = generate_chat_response(
                    args.ollama_url,
                    args.ollama_model,
                    args.system_prompt,
                    user_message,
                    args.max_tokens,
                    args.timeout,
                    args.num_ctx,
                )

                entry = {
                    "question_id": qid,
                    "category": category,
                    "difficulty": question["difficulty"],
                    "is_trap": question["is_trap"],
                    "prompt": user_message,
                    "response": response_text,
                    "tokens_generated": tokens_generated,
                    "is_coherent": is_coherent(response_text),
                    "retrieval": {
                        "chunks": retrieval_info["chunks"],
                        "distances": retrieval_info["distances"],
                        "metadatas": retrieval_info["metadatas"],
                        "context_token_count": retrieval_info["context_token_count"],
                        "augmented_prompt_token_count": retrieval_info[
                            "augmented_prompt_token_count"
                        ],
                    },
                }
                responses.append(entry)

                print(f"  [{i + 1}/{len(questions_to_run)}] {qid} ({category}) "
                      f"... {tokens_generated} tokens "
                      f"(ctx: {retrieval_info['context_token_count']} tok)")

                # Incremental save every 10 questions
                if (i + 1) % 10 == 0:
                    metadata = _build_metadata(
                        args, config, total_questions, len(responses),
                        collection_name,
                    )
                    save_responses(responses_path, responses, metadata)

            gen_elapsed = time.time() - gen_start

            # Final save
            metadata = _build_metadata(
                args, config, total_questions, len(responses), collection_name
            )
            save_responses(responses_path, responses, metadata)

            # Generation summary
            total_tokens = sum(r["tokens_generated"] for r in responses)
            error_count = sum(
                1 for r in responses
                if r["response"].startswith("[chat error:")
            )
            incoherent_count = sum(
                1 for r in responses if not r.get("is_coherent", True)
            )

            print(f"\n  Generation complete ({gen_elapsed:.1f}s)")
            print(f"  Responses: {len(responses)}")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Errors: {error_count}, Incoherent: {incoherent_count}")

        # 5. Run eval stages 2-6
        if not args.skip_eval:
            print("\n  Running evaluation stages...\n")
            success = run_eval_stages(
                output_dir, benchmark_path, registry_path,
                PROJECT_ROOT, report_suffix=config_name
            )

            # 6. Append retrieval analysis
            if success:
                report_path = os.path.join(
                    output_dir, f"hallucination_eval_{config_name}.md"
                )
                with open(responses_path) as f:
                    resp_data = json.load(f)
                append_retrieval_analysis(
                    report_path, questions, resp_data["responses"]
                )
                print(f"  Retrieval analysis appended to report")
        else:
            print("\n  --skip-eval: Skipping evaluation stages 2-6")

        completed_configs.append(config_name)
        print()

    # --- Sweep summary -------------------------------------------------------

    sweep_elapsed = time.time() - sweep_start

    print("=" * 60)
    print("  SWEEP COMPLETE")
    print("=" * 60)
    print(f"  Total time: {sweep_elapsed:.1f}s ({sweep_elapsed / 60:.1f} min)")
    print(f"  Configs completed: {len(completed_configs)}")
    if skipped_configs:
        print(f"  Configs skipped (already complete): {len(skipped_configs)}")
    print()

    # Print accuracy and hit rate summary table if eval was run
    if not args.skip_eval:
        print("  Summary across configs:")
        print()
        print(f"  {'Config':<35} {'Accuracy':>10} {'Hit Rate':>10} "
              f"{'Mean Ctx Tokens':>16}")
        print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 16}")

        all_config_names = completed_configs + skipped_configs
        for cname in all_config_names:
            cdir = os.path.join(args.output_base, cname)
            scores_path = os.path.join(cdir, "scores.json")
            responses_path = os.path.join(cdir, "responses.json")

            accuracy_str = "N/A"
            hit_rate_str = "N/A"
            ctx_str = "N/A"

            if os.path.isfile(scores_path):
                try:
                    with open(scores_path) as f:
                        scores = json.load(f)
                    # scores.json uses per_question with accuracy_binary
                    score_list = scores.get("per_question", [])
                    if score_list:
                        correct = sum(
                            1 for s in score_list
                            if s.get("accuracy_binary", 0) == 1
                        )
                        accuracy_str = f"{correct}/{len(score_list)} " \
                                       f"({correct / len(score_list):.1%})"
                except (json.JSONDecodeError, KeyError, OSError):
                    pass

            if os.path.isfile(responses_path):
                try:
                    with open(responses_path) as f:
                        resp_data = json.load(f)
                    resps = resp_data.get("responses", [])

                    hr, found, total_f, _ = compute_hit_rate(questions, resps)
                    hit_rate_str = f"{found}/{total_f} ({hr:.1%})"

                    ctx_s = compute_context_stats(resps)
                    if ctx_s["total_responses"] > 0:
                        ctx_str = f"{ctx_s['mean_tokens']:.0f}"
                except (json.JSONDecodeError, KeyError, OSError):
                    pass

            print(f"  {cname:<35} {accuracy_str:>10} {hit_rate_str:>10} "
                  f"{ctx_str:>16}")

    print()
    print("  === RAG Evaluation Complete ===")
    print()


if __name__ == "__main__":
    main()
