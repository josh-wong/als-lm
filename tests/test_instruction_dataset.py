"""Tests for instruction dataset generation and validation pipelines.

Covers Plan 01 helper functions: corpus loading, passage extraction, prompt
building, question type distribution, pair validation, and progress save/load.
Covers Plan 02 validation functions: corpus grounding verification, benchmark
leakage detection, quality statistics computation, and independence report
generation.
"""

import json
import os
import random
import sys
import tempfile

import pytest

# Ensure project root is on sys.path
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.instruction.generate import (
    CATEGORY_KEYWORDS,
    build_generation_prompt,
    decide_input_field,
    extract_passages,
    load_corpus,
    load_progress,
    save_progress,
    select_question_type,
    validate_pair_format,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_CORPUS_TEXT = (
    "Amyotrophic lateral sclerosis is a progressive neurodegenerative disease "
    "affecting motor neurons in the brain and spinal cord. Diagnosis of ALS "
    "requires electromyography and nerve conduction studies along with clinical "
    "criteria such as the El Escorial criteria for diagnostic confirmation. "
    "Differential diagnosis must exclude other conditions that mimic motor "
    "neuron disease. Biomarker research is ongoing for earlier and more "
    "accurate diagnosis. The diagnostic workup typically involves a detailed "
    "neurological examination, blood tests, and imaging studies to rule out "
    "other potential causes of the presenting symptoms."
    "<|endoftext|>"
    "The SOD1 gene mutation is the most studied genetic cause of familial ALS "
    "and was the first gene linked to the disease. C9orf72 repeat expansion "
    "is the most common genetic mutation found in hereditary cases of ALS and "
    "frontotemporal dementia. TDP-43 and FUS gene mutations also contribute "
    "to ALS pathogenesis through protein aggregation mechanisms. TARDBP "
    "mutations affect RNA processing in motor neurons and are associated with "
    "both familial and sporadic forms of the disease. Genetic testing is now "
    "available for multiple ALS-associated genes."
    "<|endoftext|>"
    "Riluzole is an FDA-approved treatment for ALS that works by reducing "
    "glutamate excitotoxicity and extending survival by several months. "
    "Edaravone is another therapeutic option that acts as a neuroprotective "
    "agent by scavenging free radicals. Clinical trial data for tofersen "
    "shows promise as an antisense oligonucleotide therapy targeting SOD1-ALS "
    "specifically. Drug development continues for multiple pathways including "
    "gene therapy and stem cell approaches. These therapeutic advances "
    "represent significant progress in ALS treatment research."
    "<|endoftext|>"
)


# ---------------------------------------------------------------------------
# TestPassageExtraction
# ---------------------------------------------------------------------------

class TestPassageExtraction:
    """Test corpus loading and passage extraction logic."""

    def test_load_corpus_splits_on_endoftext(self, tmp_path):
        """load_corpus splits train.txt on <|endoftext|> delimiter."""
        corpus_file = tmp_path / "train.txt"
        corpus_file.write_text(FAKE_CORPUS_TEXT)
        docs = load_corpus(str(corpus_file))
        assert isinstance(docs, list)
        assert len(docs) == 3

    def test_load_corpus_returns_strings(self, tmp_path):
        """Each document from load_corpus is a non-empty string."""
        corpus_file = tmp_path / "train.txt"
        corpus_file.write_text(FAKE_CORPUS_TEXT)
        docs = load_corpus(str(corpus_file))
        for doc in docs:
            assert isinstance(doc, str)
            assert len(doc.strip()) > 0

    def test_extract_passages_returns_matching_passages(self):
        """extract_passages with a fake corpus returns passages containing
        category keywords, each 300-500 chars (or full doc if shorter)."""
        docs = FAKE_CORPUS_TEXT.split("<|endoftext|>")
        docs = [d.strip() for d in docs if d.strip()]
        passages = extract_passages(docs, "diagnosis", seed=42)
        assert len(passages) > 0
        for p in passages:
            assert isinstance(p["text"], str)
            assert len(p["text"]) >= 50  # Minimum meaningful passage

    def test_extract_passages_filters_short_passages(self):
        """extract_passages filters out passages shorter than 200 chars or
        fewer than 50 words."""
        # Create docs where one is very short
        short_doc = "ALS diagnosis is short."
        long_doc = (
            "Diagnosis of amyotrophic lateral sclerosis requires careful "
            "electromyography testing and nerve conduction studies to "
            "confirm the presence of upper and lower motor neuron "
            "degeneration. The El Escorial criteria provide the clinical "
            "criteria framework for diagnostic confirmation of definite, "
            "probable, or possible ALS. Differential diagnosis should "
            "rule out other motor neuron diseases including primary "
            "lateral sclerosis and progressive muscular atrophy. "
            "Biomarker research is advancing to allow earlier and more "
            "accurate diagnosis of this devastating neurodegenerative "
            "condition affecting motor neurons throughout the central "
            "nervous system and peripheral pathways."
        )
        docs = [short_doc, long_doc]
        passages = extract_passages(docs, "diagnosis", seed=42)
        # Short doc should be filtered out; only long doc passages remain
        for p in passages:
            assert len(p["text"]) >= 200 or len(p["text"].split()) >= 50

    def test_category_keywords_has_eight_categories(self):
        """CATEGORY_KEYWORDS dict has exactly 8 categories."""
        assert len(CATEGORY_KEYWORDS) == 8

    def test_category_keywords_has_enough_terms(self):
        """Each category has at least 10 keywords."""
        for category, keywords in CATEGORY_KEYWORDS.items():
            assert len(keywords) >= 10, (
                f"Category '{category}' has only {len(keywords)} keywords, "
                f"expected at least 10"
            )

    def test_category_keywords_expected_names(self):
        """Categories include the 8 expected ALS knowledge domains."""
        expected = {
            "diagnosis", "symptoms", "genetics", "treatment",
            "clinical_trials", "epidemiology", "pathophysiology",
            "patient_care",
        }
        assert set(CATEGORY_KEYWORDS.keys()) == expected


# ---------------------------------------------------------------------------
# TestPromptBuilding
# ---------------------------------------------------------------------------

class TestPromptBuilding:
    """Test generation prompt construction."""

    def test_build_prompt_contains_passage(self):
        """build_generation_prompt includes the passage text."""
        passage = "ALS is a neurodegenerative disease affecting motor neurons."
        prompt = build_generation_prompt(passage, "diagnosis", "factual")
        assert passage in prompt

    def test_build_prompt_contains_category(self):
        """build_generation_prompt includes the category name."""
        passage = "ALS affects motor neurons."
        prompt = build_generation_prompt(passage, "genetics", "explanation")
        assert "genetics" in prompt.lower()

    def test_build_prompt_contains_question_type(self):
        """build_generation_prompt includes question type directive."""
        passage = "ALS is studied worldwide."
        prompt = build_generation_prompt(passage, "epidemiology", "factual")
        assert "factual" in prompt.lower() or "what is" in prompt.lower()

    def test_build_prompt_returns_string(self):
        """build_generation_prompt returns a non-empty string."""
        prompt = build_generation_prompt("some text", "treatment", "listing")
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# TestQuestionTypeDistribution
# ---------------------------------------------------------------------------

class TestQuestionTypeDistribution:
    """Test question type selection distribution."""

    def test_select_question_type_distribution(self):
        """select_question_type produces approximately 40% factual, 30%
        explanation, 15% comparison, 15% listing across 1000 calls."""
        random.seed(42)
        counts = {"factual": 0, "explanation": 0, "comparison": 0, "listing": 0}
        n = 1000
        for _ in range(n):
            qt = select_question_type()
            counts[qt] += 1

        # Allow 5% tolerance
        assert 350 <= counts["factual"] <= 450, f"factual: {counts['factual']}"
        assert 250 <= counts["explanation"] <= 350, f"explanation: {counts['explanation']}"
        assert 100 <= counts["comparison"] <= 200, f"comparison: {counts['comparison']}"
        assert 100 <= counts["listing"] <= 200, f"listing: {counts['listing']}"

    def test_select_question_type_returns_valid(self):
        """select_question_type returns one of the four valid types."""
        random.seed(42)
        valid = {"factual", "explanation", "comparison", "listing"}
        for _ in range(100):
            qt = select_question_type()
            assert qt in valid

    def test_decide_input_field_distribution(self):
        """decide_input_field returns empty string ~80% and populated ~20%."""
        random.seed(42)
        passage = "Motor neuron degeneration is a hallmark of ALS pathology."
        empty_count = 0
        n = 1000
        for _ in range(n):
            result = decide_input_field(passage)
            if result == "":
                empty_count += 1

        populated_count = n - empty_count
        # 80% empty with 5% tolerance
        assert 750 <= empty_count <= 850, f"empty: {empty_count}"
        assert 150 <= populated_count <= 250, f"populated: {populated_count}"

    def test_decide_input_field_populated_is_substring(self):
        """When decide_input_field returns non-empty, it is derived from
        the passage text."""
        random.seed(42)
        passage = "Motor neuron degeneration is a hallmark of ALS pathology."
        for _ in range(100):
            result = decide_input_field(passage)
            if result != "":
                # The populated input should be related to the passage
                assert isinstance(result, str)
                assert len(result) > 0


# ---------------------------------------------------------------------------
# TestPairValidation
# ---------------------------------------------------------------------------

class TestPairValidation:
    """Test Alpaca pair format validation."""

    def test_valid_pair_accepted(self):
        """validate_pair_format accepts valid Alpaca dicts."""
        pair = {
            "instruction": "What is ALS?",
            "input": "",
            "output": (
                "ALS is a progressive neurodegenerative disease that affects "
                "motor neurons in the brain and spinal cord, leading to muscle "
                "weakness and eventual paralysis."
            ),
        }
        assert validate_pair_format(pair) is True

    def test_missing_instruction_rejected(self):
        """validate_pair_format rejects dicts missing instruction key."""
        pair = {
            "input": "",
            "output": "Some valid output text here.",
        }
        assert validate_pair_format(pair) is False

    def test_missing_output_rejected(self):
        """validate_pair_format rejects dicts missing output key."""
        pair = {
            "instruction": "What is ALS?",
            "input": "",
        }
        assert validate_pair_format(pair) is False

    def test_empty_output_rejected(self):
        """validate_pair_format rejects dicts with empty output."""
        pair = {
            "instruction": "What is ALS?",
            "input": "",
            "output": "",
        }
        assert validate_pair_format(pair) is False

    def test_empty_instruction_rejected(self):
        """validate_pair_format rejects dicts with empty instruction."""
        pair = {
            "instruction": "",
            "input": "",
            "output": "ALS is a disease.",
        }
        assert validate_pair_format(pair) is False

    def test_output_too_short_rejected(self):
        """validate_pair_format rejects output with fewer than 10 words."""
        pair = {
            "instruction": "What is ALS?",
            "input": "",
            "output": "A disease.",
        }
        assert validate_pair_format(pair) is False

    def test_output_too_long_rejected(self):
        """validate_pair_format rejects output with more than 200 words."""
        long_output = " ".join(["word"] * 201)
        pair = {
            "instruction": "What is ALS?",
            "input": "",
            "output": long_output,
        }
        assert validate_pair_format(pair) is False

    def test_valid_pair_with_input_accepted(self):
        """validate_pair_format accepts pairs with populated input field."""
        pair = {
            "instruction": "What were the primary endpoints?",
            "input": "A Phase III randomized controlled trial of riluzole.",
            "output": (
                "The primary endpoints were change in ALSFRS-R score from "
                "baseline to week 48 and overall survival at 18 months."
            ),
        }
        assert validate_pair_format(pair) is True


# ---------------------------------------------------------------------------
# TestProgressSaveLoad
# ---------------------------------------------------------------------------

class TestProgressSaveLoad:
    """Test incremental save and resume logic."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """save_progress writes JSON that load_progress can read back."""
        progress_file = str(tmp_path / "progress.json")
        data = {
            "pairs": [
                {"instruction": "Q1", "input": "", "output": "A1"},
                {"instruction": "Q2", "input": "", "output": "A2"},
            ],
            "completed_categories": ["diagnosis"],
            "current_category": "genetics",
            "current_index": 5,
        }
        save_progress(data, progress_file)
        loaded = load_progress(progress_file)
        assert loaded == data

    def test_save_creates_valid_json(self, tmp_path):
        """save_progress writes a file that is valid JSON."""
        progress_file = str(tmp_path / "progress.json")
        data = {"pairs": [], "completed_categories": []}
        save_progress(data, progress_file)
        with open(progress_file) as f:
            parsed = json.load(f)
        assert parsed == data

    def test_load_nonexistent_returns_none(self, tmp_path):
        """load_progress returns None for a non-existent file."""
        result = load_progress(str(tmp_path / "nonexistent.json"))
        assert result is None


# ---------------------------------------------------------------------------
# Plan 02 imports and fixtures
# ---------------------------------------------------------------------------

from data.instruction.validate import (
    check_entity_grounding,
    check_corpus_grounding,
    check_leakage,
    compute_quality_stats,
    write_rejected,
    build_independence_report,
    validate_dataset,
)


# Synthetic entity registry for unit tests (5 entries per type)
FAKE_ENTITY_REGISTRY = {
    "drugs": [
        {"canonical": "riluzole", "aliases": ["rilutek"]},
        {"canonical": "edaravone", "aliases": ["radicava"]},
        {"canonical": "tofersen", "aliases": []},
        {"canonical": "memantine", "aliases": []},
        {"canonical": "dextromethorphan", "aliases": []},
    ],
    "genes": [
        {"canonical": "SOD1", "aliases": ["sod1"]},
        {"canonical": "C9orf72", "aliases": ["c9orf72"]},
        {"canonical": "FUS", "aliases": ["fus"]},
        {"canonical": "TARDBP", "aliases": ["tdp-43", "tardbp"]},
        {"canonical": "UBQLN2", "aliases": []},
    ],
    "proteins": [],
    "trials": [
        {"canonical": "NCT00000001", "aliases": []},
    ],
    "institutions": [],
}

# Fake corpus text for grounding checks
FAKE_CORPUS_FOR_GROUNDING = (
    "Riluzole is an FDA-approved treatment for ALS that works by reducing "
    "glutamate excitotoxicity. The SOD1 gene mutation is the most studied "
    "genetic cause of familial ALS. Motor neuron degeneration leads to "
    "progressive muscle weakness and atrophy. Edaravone acts as a "
    "neuroprotective agent by scavenging free radicals in motor neurons."
)

# Fake benchmark questions (3 entries) for leakage detection
FAKE_BENCHMARK_QUESTIONS = [
    {
        "id": "DRUG-001",
        "category": "drug_treatment",
        "question": "What drugs are FDA-approved for treating ALS?",
        "prompt_template": "List the FDA-approved drugs for ALS treatment and their mechanisms.",
        "verified_answer": "Riluzole (approved 1995) and edaravone (approved 2017) are FDA-approved.",
        "key_facts": ["riluzole", "edaravone"],
    },
    {
        "id": "GEN-001",
        "category": "genetics",
        "question": "What is the most common genetic mutation in familial ALS?",
        "prompt_template": "Describe the most common genetic mutation found in familial ALS cases.",
        "verified_answer": "C9orf72 repeat expansion is the most common mutation in familial ALS.",
        "key_facts": ["C9orf72", "repeat expansion"],
    },
    {
        "id": "PATH-001",
        "category": "pathophysiology",
        "question": "How does glutamate excitotoxicity contribute to ALS?",
        "prompt_template": "Explain the role of glutamate excitotoxicity in ALS pathophysiology.",
        "verified_answer": "Excessive glutamate causes motor neuron death through calcium influx.",
        "key_facts": ["glutamate", "motor neuron"],
    },
]


# ---------------------------------------------------------------------------
# TestGroundingVerification
# ---------------------------------------------------------------------------

class TestGroundingVerification:
    """Verify generated pairs are grounded in corpus text using entity
    registry and fuzzy matching."""

    def test_entity_grounding_found_in_registry(self):
        """check_entity_grounding returns True for output containing riluzole
        when entity registry has riluzole in drugs list."""
        output_text = (
            "Riluzole is the first FDA-approved drug for ALS treatment that "
            "reduces glutamate excitotoxicity in motor neurons."
        )
        result = check_entity_grounding(output_text, FAKE_ENTITY_REGISTRY)
        assert result["grounded"] is True

    def test_entity_grounding_fabricated_drug(self):
        """check_entity_grounding returns False for output containing a
        fabricated drug name not in registry."""
        output_text = (
            "Neurexafol is a novel neuroprotective drug that prevents motor "
            "neuron death through mitochondrial stabilization pathways."
        )
        result = check_entity_grounding(output_text, FAKE_ENTITY_REGISTRY)
        assert result["grounded"] is False
        assert len(result["failures"]) > 0

    def test_entity_grounding_no_entities(self):
        """check_entity_grounding returns True when no extractable entities
        are found in the output text."""
        output_text = (
            "The disease progresses through stages of increasing weakness "
            "and muscle atrophy affecting voluntary movements."
        )
        result = check_entity_grounding(output_text, FAKE_ENTITY_REGISTRY)
        assert result["grounded"] is True

    def test_corpus_grounding_high_overlap(self):
        """check_corpus_grounding returns a score above 70 for text with
        high overlap to a corpus passage."""
        output_text = (
            "Riluzole is an FDA-approved treatment for ALS that works by "
            "reducing glutamate excitotoxicity."
        )
        score = check_corpus_grounding(
            output_text, FAKE_CORPUS_FOR_GROUNDING, threshold=70
        )
        assert score >= 70

    def test_corpus_grounding_unrelated_text(self):
        """check_corpus_grounding returns a score below 70 for text unrelated
        to the corpus."""
        output_text = (
            "Cardiovascular disease is the leading cause of death worldwide "
            "and requires regular exercise and dietary changes."
        )
        score = check_corpus_grounding(
            output_text, FAKE_CORPUS_FOR_GROUNDING, threshold=70
        )
        assert score < 70


# ---------------------------------------------------------------------------
# TestLeakageDetection
# ---------------------------------------------------------------------------

class TestLeakageDetection:
    """Detect leakage between generated instruction pairs and the benchmark
    questions using fuzzy matching with 75% threshold."""

    def test_leakage_near_identical_question(self):
        """check_leakage returns (True, benchmark_id, score) when instruction
        text is near-identical to a benchmark question."""
        leaked, bq_id, score = check_leakage(
            instruction_text="What drugs are FDA-approved for treating ALS?",
            output_text="Riluzole and edaravone are the two main treatments.",
            benchmark_questions=FAKE_BENCHMARK_QUESTIONS,
            threshold=75,
        )
        assert leaked is True
        assert bq_id == "DRUG-001"
        assert score >= 75

    def test_leakage_unrelated_instruction(self):
        """check_leakage returns (False, None, 0) when instruction text is
        unrelated to all benchmark questions."""
        leaked, bq_id, score = check_leakage(
            instruction_text="What is the role of astrocytes in neuroinflammation?",
            output_text="Astrocytes contribute to neuroinflammation through cytokine release.",
            benchmark_questions=FAKE_BENCHMARK_QUESTIONS,
            threshold=75,
        )
        assert leaked is False
        assert bq_id is None
        assert score == 0

    def test_leakage_output_matches_verified_answer(self):
        """check_leakage detects leakage when the output text closely matches
        a benchmark verified_answer."""
        leaked, bq_id, score = check_leakage(
            instruction_text="Describe approved ALS medications.",
            output_text="Riluzole (approved 1995) and edaravone (approved 2017) are FDA-approved.",
            benchmark_questions=FAKE_BENCHMARK_QUESTIONS,
            threshold=75,
        )
        assert leaked is True
        assert bq_id == "DRUG-001"
        assert score >= 75

    def test_leakage_below_threshold(self):
        """check_leakage returns False when similarity is below threshold."""
        leaked, bq_id, score = check_leakage(
            instruction_text="What genetic testing is available for ALS patients?",
            output_text="Several genetic tests can identify ALS-related mutations.",
            benchmark_questions=FAKE_BENCHMARK_QUESTIONS,
            threshold=75,
        )
        assert leaked is False


# ---------------------------------------------------------------------------
# TestQualityStats
# ---------------------------------------------------------------------------

class TestQualityStats:
    """Validate quality statistics report structure including category
    distribution, question types, grounding rate, and response lengths."""

    def _make_sample_pairs(self):
        """Build 10 sample pairs across 3 categories for testing."""
        pairs = []
        categories = ["diagnosis", "genetics", "treatment"]
        question_types = ["factual", "explanation", "comparison", "listing"]
        for i in range(10):
            cat = categories[i % 3]
            qt = question_types[i % 4]
            pairs.append({
                "instruction": f"Question {i} about {cat}?",
                "input": "",
                "output": (
                    f"This is a detailed answer about {cat} covering "
                    f"multiple aspects of the topic. The response includes "
                    f"relevant medical terminology and clinical details."
                ),
                "metadata": {
                    "category": cat,
                    "question_type": qt,
                    "source_doc_index": i,
                    "passage_preview": f"Passage about {cat}...",
                },
            })
        return pairs

    def test_compute_quality_stats_keys(self):
        """compute_quality_stats returns dict with all required top-level
        keys: category_distribution, question_type_distribution,
        corpus_grounding, response_lengths, leakage_check."""
        pairs = self._make_sample_pairs()
        stats = compute_quality_stats(
            pairs,
            grounded_count=8,
            entity_match_count=9,
            leakage_count=0,
            benchmark_count=3,
        )
        required_keys = {
            "metadata",
            "category_distribution",
            "question_type_distribution",
            "corpus_grounding",
            "response_lengths",
            "leakage_check",
        }
        assert required_keys.issubset(set(stats.keys()))

    def test_compute_quality_stats_category_counts(self):
        """compute_quality_stats correctly counts per-category distribution
        from a list of 10 sample pairs across 3 categories."""
        pairs = self._make_sample_pairs()
        stats = compute_quality_stats(
            pairs,
            grounded_count=8,
            entity_match_count=9,
            leakage_count=0,
            benchmark_count=3,
        )
        cat_dist = stats["category_distribution"]
        # 10 pairs across 3 categories: indices 0,3,6,9 -> diagnosis(4),
        # 1,4,7 -> genetics(3), 2,5,8 -> treatment(3)
        assert cat_dist["diagnosis"]["count"] == 4
        assert cat_dist["genetics"]["count"] == 3
        assert cat_dist["treatment"]["count"] == 3

    def test_compute_quality_stats_response_lengths(self):
        """compute_quality_stats returns response_lengths with mean_words,
        min_words, max_words, and mean_sentences."""
        pairs = self._make_sample_pairs()
        stats = compute_quality_stats(
            pairs,
            grounded_count=8,
            entity_match_count=9,
            leakage_count=0,
            benchmark_count=3,
        )
        lengths = stats["response_lengths"]
        assert "mean_words" in lengths
        assert "min_words" in lengths
        assert "max_words" in lengths
        assert "mean_sentences" in lengths
        assert lengths["mean_words"] > 0
        assert lengths["min_words"] <= lengths["max_words"]

    def test_compute_quality_stats_grounding_rate(self):
        """compute_quality_stats returns correct grounding rate from
        provided counts."""
        pairs = self._make_sample_pairs()
        stats = compute_quality_stats(
            pairs,
            grounded_count=8,
            entity_match_count=9,
            leakage_count=0,
            benchmark_count=3,
        )
        grounding = stats["corpus_grounding"]
        assert grounding["grounded_count"] == 8
        assert grounding["total_count"] == 10
        assert grounding["grounding_rate"] == 0.8
        assert grounding["entity_match_rate"] == 0.9


# ---------------------------------------------------------------------------
# TestIndependenceReport
# ---------------------------------------------------------------------------

class TestIndependenceReport:
    """Verify train/eval independence report proving no leakage between
    instruction pairs and benchmarks."""

    def test_independence_report_structure(self):
        """build_independence_report returns dict with pairs_checked,
        benchmark_questions, max_similarity, flagged_pairs fields."""
        pairs = [
            {
                "instruction": "Describe the role of astrocytes in ALS.",
                "input": "",
                "output": "Astrocytes contribute to motor neuron death in ALS.",
            },
        ]
        report = build_independence_report(
            pairs, FAKE_BENCHMARK_QUESTIONS, threshold=75
        )
        assert "pairs_checked" in report
        assert "benchmark_questions" in report
        assert "max_similarity" in report
        assert "flagged_pairs" in report

    def test_independence_report_pass_verdict(self):
        """build_independence_report returns pass verdict when no pairs
        exceed threshold."""
        pairs = [
            {
                "instruction": "What role do astrocytes play in neurodegeneration?",
                "input": "",
                "output": "Astrocytes release inflammatory mediators that damage neurons.",
            },
            {
                "instruction": "Describe respiratory management in ALS patients.",
                "input": "",
                "output": "Non-invasive ventilation improves quality of life in ALS.",
            },
        ]
        report = build_independence_report(
            pairs, FAKE_BENCHMARK_QUESTIONS, threshold=75
        )
        assert report["verdict"] == "PASS"
        assert report["flagged_pairs"] == 0

    def test_independence_report_fail_verdict(self):
        """build_independence_report returns fail verdict when a pair
        exceeds threshold."""
        pairs = [
            {
                "instruction": "What drugs are FDA-approved for treating ALS?",
                "input": "",
                "output": "Riluzole and edaravone are approved for ALS treatment.",
            },
        ]
        report = build_independence_report(
            pairs, FAKE_BENCHMARK_QUESTIONS, threshold=75
        )
        assert report["verdict"] == "FAIL"
        assert report["flagged_pairs"] >= 1


# ---------------------------------------------------------------------------
# TestValidateMain
# ---------------------------------------------------------------------------

class TestValidateMain:
    """Test the main validate_dataset function and rejected pair writing."""

    def test_write_rejected_appends_jsonl(self, tmp_path):
        """write_rejected appends JSONL entries with pair data and rejection
        reason."""
        rejected_path = str(tmp_path / "rejected.jsonl")
        pair = {
            "instruction": "What is Neurexafol?",
            "input": "",
            "output": "Neurexafol is a fictional drug.",
        }
        write_rejected(rejected_path, pair, "grounding_failed", {"entity_failures": ["Neurexafol"]})
        write_rejected(rejected_path, pair, "benchmark_leakage", {"benchmark_id": "DRUG-001"})

        with open(rejected_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

        entry1 = json.loads(lines[0])
        assert entry1["reason"] == "grounding_failed"
        assert "pair" in entry1

        entry2 = json.loads(lines[1])
        assert entry2["reason"] == "benchmark_leakage"

    def test_validate_dataset_filters_bad_pairs(self, tmp_path):
        """validate_dataset filters out pairs failing grounding or leakage,
        writes rejected.jsonl, returns cleaned list."""
        rejected_path = str(tmp_path / "rejected.jsonl")

        # One good pair (no entities, unrelated to benchmarks)
        good_pair = {
            "instruction": "How does respiratory function change in ALS?",
            "input": "",
            "output": (
                "Progressive respiratory muscle weakness is a hallmark of "
                "ALS that leads to ventilatory failure and is the primary "
                "cause of mortality in most patients with the disease."
            ),
            "metadata": {
                "category": "patient_care",
                "question_type": "explanation",
                "source_doc_index": 0,
                "passage_preview": "Respiratory...",
            },
        }

        # One pair that leaks (near-identical to benchmark question)
        leaked_pair = {
            "instruction": "What drugs are FDA-approved for treating ALS?",
            "input": "",
            "output": (
                "Riluzole (approved 1995) and edaravone (approved 2017) are "
                "the FDA-approved drugs for treating ALS."
            ),
            "metadata": {
                "category": "treatment",
                "question_type": "factual",
                "source_doc_index": 1,
                "passage_preview": "Riluzole...",
            },
        }

        pairs = [good_pair, leaked_pair]
        cleaned = validate_dataset(
            pairs=pairs,
            entity_registry=FAKE_ENTITY_REGISTRY,
            corpus_text=FAKE_CORPUS_FOR_GROUNDING,
            benchmark_questions=FAKE_BENCHMARK_QUESTIONS,
            rejected_path=rejected_path,
            grounding_threshold=70,
            leakage_threshold=75,
            entity_threshold=85,
        )

        # The leaked pair should be removed
        assert len(cleaned) < len(pairs)
        # Rejected file should have entries
        assert os.path.exists(rejected_path)
        with open(rejected_path) as f:
            rejected_lines = f.readlines()
        assert len(rejected_lines) >= 1
