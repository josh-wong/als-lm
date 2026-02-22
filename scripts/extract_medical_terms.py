"""Extract and curate medical terms from the ALS corpus.

Builds a categorized list of ~200 medical terms by combining frequency-based
auto-extraction from the corpus with a curated set of key ALS-specific terms.
Terms are classified into five categories: abbreviations, disease names, drug
names, clinical terms, and gene names.

Usage:
    python scripts/extract_medical_terms.py
    python scripts/extract_medical_terms.py --corpus data/processed/train.txt --output reports/medical_terms.json
    python scripts/extract_medical_terms.py --min-frequency 3
"""

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path


# Curated ALS-specific terms that may be rare but are medically important.
# These supplement the frequency-based extraction to ensure coverage of
# key domain vocabulary regardless of corpus composition.
CURATED_TERMS = {
    "drug": [
        "riluzole",
        "edaravone",
        "tofersen",
        "sodium phenylbutyrate",
        "taurursodiol",
        "radicava",
        "nuedexta",
        "dextromethorphan",
        "quinidine",
        "baclofen",
        "tizanidine",
        "dantrolene",
        "botulinum toxin",
        "rilutek",
        "AMX0035",
    ],
    "abbreviation": [
        "ALS",
        "FTD",
        "EMG",
        "FVC",
        "UMN",
        "LMN",
        "MND",
        "ALSFRS-R",
        "PLS",
        "PMA",
        "PBP",
        "NIV",
        "BiPAP",
        "PEG",
        "CSF",
        "DTI",
        "PET",
        "NfL",
        "ASO",
        "AAV",
        "iPSC",
        "MSC",
        "SBMA",
        "HSP",
        "PBA",
        "sALS",
        "fALS",
        "bvFTD",
        "AAC",
        "SNIP",
        "SVC",
        "pNfH",
        "RNAi",
    ],
    "gene": [
        "SOD1",
        "C9orf72",
        "TARDBP",
        "FUS",
        "TBK1",
        "OPTN",
        "NEK1",
        "ATXN2",
        "GluR2",
        "TDP-43",
    ],
    "disease": [
        "amyotrophic lateral sclerosis",
        "frontotemporal dementia",
        "primary lateral sclerosis",
        "progressive muscular atrophy",
        "progressive bulbar palsy",
        "motor neuron disease",
        "Kennedy disease",
        "hereditary spastic paraplegia",
        "spinocerebellar ataxia",
        "multifocal motor neuropathy",
    ],
    "clinical": [
        "fasciculation",
        "denervation",
        "reinnervation",
        "electromyography",
        "spirometry",
        "dysphagia",
        "dysarthria",
        "spasticity",
        "atrophy",
        "hyperreflexia",
        "clonus",
        "sialorrhea",
        "tracheostomy",
        "gastrostomy",
        "excitotoxicity",
        "neurodegeneration",
        "neuroinflammation",
        "proteostasis",
        "autophagy",
        "apoptosis",
        "El Escorial criteria",
        "Awaji criteria",
        "Gold Coast criteria",
        "Babinski sign",
        "bulbar onset",
        "limb onset",
        "respiratory failure",
        "diaphragmatic pacing",
        "percutaneous endoscopic gastrostomy",
        "non-invasive ventilation",
    ],
}

# Drug name suffixes for auto-detection
DRUG_SUFFIXES = (
    "ole", "one", "ine", "mab", "nib", "ase", "sol",
    "diol", "avone", "ersen", "pril", "artan", "olol",
    "azine", "azole", "cycline", "mycin", "oxacin",
)

# Gene name patterns (alphanumeric, typically uppercase with numbers)
GENE_PATTERN = re.compile(
    r"^[A-Z][A-Z0-9]{1,}(?:orf[0-9]+)?(?:-[A-Z0-9]+)?$"
)

# Abbreviation pattern (2-8 uppercase characters, may include numbers/hyphens)
ABBREV_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,7}(?:-[A-Z0-9]+)?$")


def extract_word_frequencies(corpus_path: str) -> Counter:
    """Count word frequencies from the corpus using whitespace splitting.

    Strips punctuation from word boundaries but preserves internal
    hyphens and special characters (important for terms like TDP-43,
    ALSFRS-R, C9orf72).
    """
    word_counts = Counter()
    strip_chars = string.punctuation.replace("-", "").replace("(", "").replace(")", "")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            # Skip the endoftext separator
            line = line.replace("<|endoftext|>", " ")
            words = line.split()
            for word in words:
                # Strip leading/trailing punctuation but preserve internal
                cleaned = word.strip(strip_chars)
                if cleaned and len(cleaned) >= 2:
                    word_counts[cleaned] += 1

    return word_counts


def classify_term(term: str, frequency: int) -> str:
    """Classify a term into a category based on pattern heuristics.

    Returns one of: abbreviation, gene, drug, clinical, disease, or None
    if the term does not match any medical pattern.
    """
    # Check gene pattern first (more specific than abbreviation)
    if GENE_PATTERN.match(term) and any(c.isdigit() for c in term):
        return "gene"

    # Check abbreviation pattern
    if ABBREV_PATTERN.match(term) and len(term) <= 8:
        return "abbreviation"

    # Check drug suffixes
    lower = term.lower()
    if any(lower.endswith(suffix) for suffix in DRUG_SUFFIXES):
        return "drug"

    # Multi-word terms are typically disease names or clinical terms
    if " " in term:
        return "disease"

    # Remaining medical-looking terms
    if lower.endswith(("tion", "sis", "phy", "gia", "ia", "ity", "osis")):
        return "clinical"

    return None


def extract_terms_from_corpus(
    corpus_path: str, min_frequency: int = 5
) -> list[dict]:
    """Auto-extract medical terms from corpus by frequency and pattern.

    Identifies potential medical terms using frequency thresholds and
    pattern-based classification heuristics, then deduplicates against
    the curated term list.
    """
    word_counts = extract_word_frequencies(corpus_path)
    extracted = []
    seen_terms = set()

    for word, count in word_counts.most_common():
        if count < min_frequency:
            break

        category = classify_term(word, count)
        if category is not None:
            term_lower = word.lower()
            if term_lower not in seen_terms:
                extracted.append({
                    "term": word,
                    "category": category,
                    "corpus_frequency": count,
                    "source": "auto-extracted",
                })
                seen_terms.add(term_lower)

    return extracted


def build_term_list(
    corpus_path: str, min_frequency: int = 5
) -> list[dict]:
    """Build the complete medical term list combining auto-extraction and curation.

    The curated list takes priority for category assignment when a term
    appears in both the auto-extracted and curated sets. Terms are
    deduplicated by lowercase comparison.
    """
    # Start with frequency-based extraction
    auto_terms = extract_terms_from_corpus(corpus_path, min_frequency)

    # Build a lookup of auto-extracted terms for frequency data
    auto_freq = {}
    for t in auto_terms:
        auto_freq[t["term"].lower()] = t["corpus_frequency"]

    # Get word frequencies for curated terms that may not be auto-extracted
    word_counts = extract_word_frequencies(corpus_path)
    word_counts_lower = {}
    for word, count in word_counts.items():
        key = word.lower()
        if key not in word_counts_lower or count > word_counts_lower[key]:
            word_counts_lower[key] = count

    # Merge curated terms (they take category priority)
    seen = set()
    final_terms = []

    # Add curated terms first (they define authoritative categories)
    for category, terms in CURATED_TERMS.items():
        for term in terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                # Look up corpus frequency
                freq = word_counts_lower.get(term_lower, 0)
                # For multi-word terms, try partial matching
                if freq == 0 and " " in term:
                    words = term.lower().split()
                    # Use frequency of the least common word as proxy
                    word_freqs = [
                        word_counts_lower.get(w, 0) for w in words
                    ]
                    freq = min(word_freqs) if word_freqs else 0
                final_terms.append({
                    "term": term,
                    "category": category,
                    "corpus_frequency": freq,
                    "source": "curated",
                })

    # Add auto-extracted terms not already in curated set
    for t in auto_terms:
        term_lower = t["term"].lower()
        if term_lower not in seen:
            seen.add(term_lower)
            final_terms.append(t)

    # Sort by category then by frequency (descending)
    final_terms.sort(key=lambda x: (x["category"], -x["corpus_frequency"]))

    return final_terms


def main():
    parser = argparse.ArgumentParser(
        description="Extract and curate medical terms from the ALS corpus"
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/train.txt",
        help="Path to training text file (default: data/processed/train.txt)",
    )
    parser.add_argument(
        "--output",
        default="reports/medical_terms.json",
        help="Output path for term list JSON (default: reports/medical_terms.json)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=5,
        help="Minimum corpus frequency for auto-extraction (default: 5)",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting medical terms from: {corpus_path}")
    print(f"Minimum frequency threshold: {args.min_frequency}")

    terms = build_term_list(str(corpus_path), args.min_frequency)

    # Summary by category
    categories = {}
    for t in terms:
        cat = t["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nExtracted {len(terms)} terms across {len(categories)} categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(terms, f, indent=2, ensure_ascii=False)

    print(f"\nTerm list saved to: {output_path}")

    # Print sample terms
    print("\nSample terms:")
    for t in terms[:10]:
        print(f"  {t['term']:30s}  {t['category']:15s}  freq={t['corpus_frequency']}")


if __name__ == "__main__":
    main()
