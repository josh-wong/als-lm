"""Build an entity registry from the ALS training corpus.

Extracts drug names, gene names, protein names, clinical trial NCT numbers,
and institution names from the training corpus. The registry serves as the
ground-truth reference for fabrication detection: model-generated entities
are checked against this registry to identify hallucinated names.

Entity extraction replicates and extends the patterns established in
scripts/extract_medical_terms.py, with the additional requirement of
canonical/alias grouping and five distinct entity types.

Usage:
    python eval/build_entity_registry.py
    python eval/build_entity_registry.py --corpus data/processed/train.txt --output eval/entity_registry.json
    python eval/build_entity_registry.py --min-frequency 1
"""

import argparse
import json
import re
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Seed data (replicated from scripts/extract_medical_terms.py CURATED_TERMS
# plus additional ALS-domain seeds). The eval package is self-contained.
# ---------------------------------------------------------------------------

DRUG_SEEDS = {
    # canonical -> list of known aliases (case-insensitive matching)
    "riluzole": ["rilutek"],
    "edaravone": ["radicava"],
    "tofersen": [],
    "sodium phenylbutyrate": [],
    "taurursodiol": [],
    "nuedexta": [],
    "dextromethorphan": [],
    "quinidine": [],
    "baclofen": [],
    "tizanidine": [],
    "dantrolene": [],
    "botulinum toxin": [],
    "AMX0035": ["relyvrio"],
    "memantine": [],
    "lithium": [],
    "celecoxib": [],
    "minocycline": [],
    "ceftriaxone": [],
    "rasagiline": [],
    "tamoxifen": [],
    "methylcobalamin": [],
    "masitinib": [],
    "tirasemtiv": [],
    "levosimendan": [],
    "pimozide": [],
    "arimoclomol": [],
    "Cu(II)ATSM": ["CuATSM"],
}

GENE_SEEDS = [
    "SOD1", "C9orf72", "TARDBP", "FUS", "TBK1", "OPTN", "NEK1", "ATXN2",
    "ANG", "VAPB", "VCP", "UBQLN2", "PFN1", "HNRNPA1", "HNRNPA2B1",
    "SQSTM1", "CHMP2B", "DCTN1", "SETX", "ALS2", "SPG11", "FIG4",
    "SIGMAR1", "ERBB4", "MATR3", "CHCHD10", "CCNF", "KIF5A", "GLT8D1",
    "NEFH", "PRPH", "TUBA4A", "GRN", "EWSR1",
]

PROTEIN_SEEDS = {
    # canonical -> aliases
    "TDP-43": ["TDP43", "TARDBP protein"],
    "FUS protein": ["FUS"],
    "SOD1 protein": ["superoxide dismutase 1", "Cu/Zn superoxide dismutase"],
    "ubiquitin": [],
    "neurofilament": ["NfL", "NfH", "pNfH", "neurofilament light",
                      "neurofilament heavy"],
    "dynactin": ["DCTN1 protein"],
    "profilin": ["PFN1 protein"],
    "VAPB protein": [],
    "annexin": [],
    "caspase": [],
    "p62": ["SQSTM1 protein", "sequestosome"],
    "optineurin": ["OPTN protein"],
    "C9orf72 protein": ["C9orf72 DPR"],
    "ataxin-2": ["ATXN2 protein"],
    "matrin-3": ["MATR3 protein"],
    "CHCHD10 protein": [],
    "cyclophilin A": [],
}

INSTITUTION_PATTERNS = [
    # Each pattern captures the full institution name.
    # Patterns require a leading capitalized word to avoid fragments.
    r"\bUniversity of [A-Z][\w\s\-]{2,40}(?:Medical (?:Center|School))?",
    r"\b[A-Z][\w\s\-]{2,30} University(?:\s+(?:Medical (?:Center|School)|Hospital))?",
    r"\b[A-Z][\w\s\-]{2,30} Institute(?:\s+(?:of|for) [A-Z][\w\s\-]{2,30})?",
    r"\bNational Institute(?:s)? of [A-Z][\w\s\-]{2,40}",
    r"\b[A-Z][\w\s\-]{2,30} Hospital",
    r"\b[A-Z][\w\s\-]{2,30} Medical Center",
    r"\b[A-Z][\w\s\-]{2,30} Foundation(?:\s+(?:of|for) [A-Z][\w\s\-]{2,30})?",
    r"\b[A-Z][\w\s\-]{2,30} Clinic(?:al Center)?",
]

# Drug name suffixes for auto-detection (from extract_medical_terms.py)
DRUG_SUFFIXES = (
    "ole", "one", "ine", "mab", "nib", "ase", "sol",
    "diol", "avone", "ersen", "pril", "artan", "olol",
    "azine", "azole", "cycline", "mycin", "oxacin",
    "tinib", "zumab", "ximab", "umab", "stat",
)

# Common English words and scientific terms that happen to match drug suffixes
# but are not drugs. Extensive list to reduce false positives.
DRUG_SUFFIX_STOPWORDS = {
    # -one words
    "the", "one", "done", "gone", "none", "bone", "zone", "tone", "alone",
    "stone", "phone", "ozone", "clone", "prone", "throne", "someone",
    "everyone", "anyone", "hormone", "acetone", "ketone", "histone",
    "milestone", "backbone", "osterone", "cortisone", "cyclone",
    "component", "ezone",
    # -ole words
    "role", "whole", "pole", "hole", "sole", "mole", "stole", "console",
    "parole", "casserole", "control", "protocol", "aerosol",
    # -ine words
    "line", "mine", "fine", "wine", "pine", "vine", "dine", "nine",
    "shine", "combine", "define", "decline", "online", "baseline",
    "outline", "discipline", "examine", "determine", "imagine", "machine",
    "routine", "magazine", "marine", "medicine", "vaccine", "engine",
    "genuine", "feminine", "masculine", "divine", "doctrine",
    "famine", "jasmine", "trampoline", "gasoline", "eine", "oline",
    "seine", "sunshine", "pipeline", "guideline", "lifetime", "sometime",
    "Palestine", "antine", "figurine", "landmine", "pristine", "ravine",
    "canine", "bovine", "equine", "swine", "uterine", "intestine",
    "hemoglobine", "creatinine",
    # -ase words
    "phase", "base", "case", "release", "increase", "disease", "please",
    "purchase", "decrease", "database", "erase", "chase",
    # -ide words
    "provide", "inside", "outside", "beside", "guide", "pride", "wide",
    "slide", "hide", "side", "ride", "decide", "divide", "include",
    "peptide", "nucleotide", "oxide", "dioxide", "chloride",
    "fluoride", "sulfide", "bromide", "iodide", "hydroxide", "cyanide",
    "amide", "lipide", "worldwide", "override", "aside", "stride",
    # -ile/-ile words
    "april", "mobile", "smile", "while", "file", "profile", "tile",
    "mile", "style", "meanwhile", "hostile", "missile", "futile",
    "percentile", "quartile", "fertile", "volatile", "textile",
    # -sol words
    "aerosol",
    # -stat words
    "thermostat", "photostat",
    # Biological terms that are not drugs
    "cytokine", "chemokine", "lymphokine", "monokine", "adipokine",
    "myokine", "exerkine", "interleukin",
    "kinase", "phosphatase", "protease", "lipase", "nuclease",
    "polymerase", "synthetase", "transferase", "oxidase", "reductase",
    "dismutase", "ligase", "helicase", "endonuclease", "exonuclease",
    "peptidase", "esterase", "isomerase", "hydrolase", "lyase",
    "telomerase", "reverse",
    "colchicine", "glycine", "alanine", "leucine", "isoleucine",
    "valine", "proline", "serine", "threonine", "cysteine", "tyrosine",
    "histidine", "tryptophan", "asparagine", "glutamine", "lysine",
    "arginine", "methionine", "phenylalanine",
    "adenine", "guanine", "thymine", "cytosine", "uracil",
    "purine", "pyrimidine",
    "creatine", "carnitine", "choline", "betaine",
    "caffeine", "nicotine", "cocaine", "morphine", "codeine",
    "dopamine", "serotonine", "epinephrine", "norepinephrine",
    "acetylcholine", "glutamate", "aspartate",
    "polyglutamine", "prion", "misfolding",
    "benzothiazole", "thioflavin", "fluorescein",
    "amine", "imine", "indole", "pyrrole",
    # Scientific/biological terms caught by suffix matching
    "murine", "saline", "synthase", "chaperone", "cytosol",
    "dehydrogenase", "adenosine", "endocrine", "peroxidase",
    "luciferase", "gtpase", "atpase", "catalase", "paracrine",
    "lipofectamine", "deacetylase", "autocrine", "migraine",
    "undergone", "refine", "overexpression", "coenzyme",
    "calmoduline", "transaminase", "aminotransferase",
    "glucuronidase", "galactosidase", "arginase", "nitrilase",
    "ubiquitinase", "demethylase", "methyltransferase",
    "acetyltransferase", "phospholipase", "thioredoxine",
    "glucosidase", "mannosidase", "fucosidase", "sialidase",
    "neuraminidase", "collagenase", "gelatinase", "elastase",
    "cathepsine", "calpaine", "aldolase", "enolase",
    "glutamate", "aspartate",
    "serotonine", "melatonine", "taurine",
    "cytochrome", "flavone", "isoflavone", "chalcone",
    "lactone", "quinone", "semiquinone",
    "terpenone", "sterone", "androne",
    "glutathione", "caspase", "proteasome",
    "exosome", "ribosome", "chromosome", "lysosome",
    "endosome", "autophagosome", "phagosome", "peroxisome",
    "mitochondrione", "nucleosome",
    "acridine", "flavine", "riboflavine",
    "supernatine", "crystalline",
    "opaline", "tramadole",
}

# Gene name pattern (uppercase letters with numbers, optional orf notation)
GENE_PATTERN = re.compile(
    r"^[A-Z][A-Z0-9]{1,}(?:orf[0-9]+)?(?:-[A-Z0-9]+)?$"
)

# NCT trial pattern
NCT_PATTERN = re.compile(r"NCT\d{8}")

# Protein-matching patterns
PROTEIN_PATTERN_SUFFIXES = ("ase", "in")
PROTEIN_COMMON_ENGLISH = {
    "main", "again", "certain", "mountain", "captain", "contain", "obtain",
    "maintain", "explain", "remain", "brain", "train", "pain", "gain",
    "chain", "rain", "plain", "strain", "domain", "fountain", "curtain",
    "bargain", "villain", "complain", "sustain", "retain", "attain",
    "entertain", "refrain", "constrain", "restrain", "detain", "pertain",
    "campaign", "origin", "within", "begin", "margin", "latin", "satin",
    "cabin", "robin", "basin", "resin", "toxin", "plugin", "coin",
    "join", "thin", "skin", "spin", "twin", "pin", "win", "bin", "sin",
    "fin", "tin", "kin",
}


def extract_word_frequencies(corpus_path: str) -> Counter:
    """Count word frequencies from the corpus using whitespace splitting.

    Strips punctuation from word boundaries but preserves internal hyphens
    and special characters (important for terms like TDP-43, C9orf72).
    Reads line-by-line to handle large files without excessive memory use.
    """
    word_counts: Counter = Counter()
    strip_chars = string.punctuation.replace("-", "").replace("(", "").replace(")", "")

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.replace("<|endoftext|>", " ")
            for word in line.split():
                cleaned = word.strip(strip_chars)
                if cleaned and len(cleaned) >= 2:
                    word_counts[cleaned] += 1

    return word_counts


def extract_drugs(word_counts: Counter) -> list[dict]:
    """Extract drug entities from word frequencies and seed data.

    Combines curated drug seeds with suffix-based auto-detection.
    Groups canonical names with known aliases.
    """
    entities = []
    seen_lower: set[str] = set()

    # Build a case-insensitive lookup from word_counts
    lower_counts: dict[str, int] = defaultdict(int)
    for word, count in word_counts.items():
        lower_counts[word.lower()] += count

    # Process curated drug seeds first
    for canonical, aliases in DRUG_SEEDS.items():
        canonical_lower = canonical.lower()
        if canonical_lower in seen_lower:
            continue
        seen_lower.add(canonical_lower)

        freq = lower_counts.get(canonical_lower, 0)
        # For multi-word terms, check if any word appears
        if freq == 0 and " " in canonical:
            words = canonical_lower.split()
            word_freqs = [lower_counts.get(w, 0) for w in words]
            freq = min(word_freqs) if word_freqs else 0

        # Find alias frequencies
        valid_aliases = []
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower != canonical_lower:
                seen_lower.add(alias_lower)
                alias_freq = lower_counts.get(alias_lower, 0)
                if alias_freq > 0:
                    valid_aliases.append(alias)
                    freq = max(freq, alias_freq)
                else:
                    valid_aliases.append(alias)

        if freq >= 1:
            entities.append({
                "canonical": canonical,
                "aliases": valid_aliases,
                "frequency": freq,
            })

    # Auto-detect additional drugs by suffix matching.
    # Require minimum 6 chars and minimum frequency of 2 to reduce noise
    # from random words that happen to end in drug suffixes.
    for word, count in word_counts.most_common():
        word_lower = word.lower()
        if word_lower in seen_lower:
            continue
        if word_lower in DRUG_SUFFIX_STOPWORDS:
            continue
        if len(word) < 6:
            continue
        # Skip words with digits (likely identifiers, not drug names)
        if any(c.isdigit() for c in word):
            continue
        # Skip ALL_CAPS words (abbreviations, not drug names)
        if word.isupper():
            continue

        if any(word_lower.endswith(suffix) for suffix in DRUG_SUFFIXES):
            if count >= 2:
                seen_lower.add(word_lower)
                entities.append({
                    "canonical": word_lower,
                    "aliases": [],
                    "frequency": count,
                })

    return entities


def extract_genes(word_counts: Counter) -> list[dict]:
    """Extract gene entities from word frequencies and seed data.

    Uses curated gene seeds plus regex pattern matching for additional
    gene names. Gene names are typically uppercase with numbers.
    """
    entities = []
    seen_lower: set[str] = set()

    lower_counts: dict[str, int] = defaultdict(int)
    for word, count in word_counts.items():
        lower_counts[word.lower()] += count

    # Also build an exact-case lookup for gene names
    exact_counts: dict[str, int] = {}
    for word, count in word_counts.items():
        exact_counts[word] = count

    # Process gene seeds
    for gene in GENE_SEEDS:
        gene_lower = gene.lower()
        if gene_lower in seen_lower:
            continue
        seen_lower.add(gene_lower)

        # Try exact case first, then case-insensitive
        freq = exact_counts.get(gene, 0)
        if freq == 0:
            freq = lower_counts.get(gene_lower, 0)

        if freq >= 1:
            entities.append({
                "canonical": gene,
                "aliases": [],
                "frequency": freq,
            })

    # Auto-detect additional gene-like tokens from corpus.
    # Gene names are typically 2-10 uppercase chars with at least one digit
    # (e.g., BRCA1, TP53, IL6, APOE4). We require:
    #   - Matches GENE_PATTERN (uppercase + digits)
    #   - Has at least one digit (to distinguish from abbreviations)
    #   - Length 2-10 characters (gene names are short)
    #   - Not a common non-gene abbreviation
    #   - Minimum frequency of 3 (filters out noise/typos)
    for word, count in word_counts.most_common():
        if word.lower() in seen_lower:
            continue
        if len(word) < 2 or len(word) > 10:
            continue
        if count < 3:
            break  # word_counts.most_common() is sorted desc, so stop early

        if GENE_PATTERN.match(word):
            has_digit = any(c.isdigit() for c in word)
            has_letter = any(c.isalpha() for c in word)

            # Must have both letters and digits to look like a gene
            if not (has_digit and has_letter):
                continue

            # Exclude common non-gene abbreviations and identifiers
            non_gene_abbrevs = {
                "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
                "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS",
                "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO",
                "DID", "GET", "HIM", "LET", "SAY", "SHE", "TOO", "USE",
                "MAY", "RNA", "DNA", "BMI", "FDA", "USA", "MRI", "ICU",
                "HIV", "WHO", "ALS", "FTD", "EMG", "FVC", "UMN", "LMN",
                "MND", "PLS", "PMA", "PBP", "NIV", "PEG", "CSF", "DTI",
                "PET", "ASO", "AAV", "MSC", "SBMA", "HSP", "PBA",
                "AAC", "CNS", "BBB", "EEG", "PCR", "SNP", "QOL",
                "RCT", "IRB", "NIH", "DOI", "PDF", "URL", "PMC", "PPI",
                "AUC", "ROC", "SEM", "IQR", "SD1", "SD2", "IC50",
                "EC50", "LD50", "ED50", "HR1", "HR2", "CI95",
            }
            if word in non_gene_abbrevs:
                continue

            # Exclude figure/table references (e.g., S1, S2, S3...)
            if re.match(r"^[A-Z]\d+$", word) and len(word) <= 3:
                continue

            seen_lower.add(word.lower())
            entities.append({
                "canonical": word,
                "aliases": [],
                "frequency": count,
            })

    return entities


def extract_proteins(word_counts: Counter) -> list[dict]:
    """Extract protein entities from word frequencies and seed data.

    Matches protein-specific patterns: words ending in -ase or -in that
    are biological (not common English), plus curated seed proteins.
    """
    entities = []
    seen_lower: set[str] = set()

    lower_counts: dict[str, int] = defaultdict(int)
    for word, count in word_counts.items():
        lower_counts[word.lower()] += count

    # Process curated protein seeds
    for canonical, aliases in PROTEIN_SEEDS.items():
        canonical_lower = canonical.lower()
        if canonical_lower in seen_lower:
            continue
        seen_lower.add(canonical_lower)

        freq = lower_counts.get(canonical_lower, 0)
        # Try the base form without " protein" suffix
        if freq == 0 and canonical_lower.endswith(" protein"):
            base = canonical_lower.replace(" protein", "")
            freq = lower_counts.get(base, 0)

        # Check alias frequencies
        valid_aliases = []
        for alias in aliases:
            alias_lower = alias.lower()
            seen_lower.add(alias_lower)
            alias_freq = lower_counts.get(alias_lower, 0)
            if alias_freq > 0:
                freq = max(freq, alias_freq)
            valid_aliases.append(alias)

        if freq >= 1:
            entities.append({
                "canonical": canonical,
                "aliases": valid_aliases,
                "frequency": freq,
            })

    # Auto-detect protein-like terms from corpus
    for word, count in word_counts.most_common():
        word_lower = word.lower()
        if word_lower in seen_lower:
            continue
        if len(word) < 4:
            continue

        # Match protein name patterns
        is_protein = False

        # Words ending in -ase (enzymes are proteins)
        if word_lower.endswith("ase") and word_lower not in DRUG_SUFFIX_STOPWORDS:
            if word_lower not in {"base", "case", "phase", "release", "increase",
                                   "decrease", "disease", "please", "purchase",
                                   "database", "erase", "chase"}:
                is_protein = True

        # Words ending in -in that are likely proteins (not common English)
        if word_lower.endswith("in") and not is_protein:
            if word_lower not in PROTEIN_COMMON_ENGLISH:
                if word_lower not in DRUG_SUFFIX_STOPWORDS:
                    # Additional check: likely a protein if it contains
                    # scientific-looking substrings
                    protein_indicators = (
                        "globin", "lectin", "ferrin", "actin", "myosin",
                        "tubulin", "keratin", "collagen", "elastin",
                        "fibrin", "albumin", "globulin", "casein",
                        "pepsin", "trypsin", "insulin", "leptin",
                        "avidin", "claudin", "connexin", "catenin",
                        "cadherin", "integrin", "selectin", "laminin",
                        "tenascin", "fibronectin", "vitronectin",
                    )
                    if any(word_lower.endswith(ind) or word_lower == ind
                           for ind in protein_indicators):
                        is_protein = True

        if is_protein and count >= 1:
            seen_lower.add(word_lower)
            entities.append({
                "canonical": word_lower,
                "aliases": [],
                "frequency": count,
            })

    return entities


def extract_trials(corpus_path: str) -> list[dict]:
    """Extract clinical trial NCT numbers from the corpus.

    Reads the corpus line-by-line and matches NCT\\d{8} patterns.
    Each NCT number is a unique entity with no aliasing.
    """
    nct_counts: Counter = Counter()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            matches = NCT_PATTERN.findall(line)
            for nct in matches:
                nct_counts[nct] += 1

    entities = []
    for nct, count in nct_counts.most_common():
        entities.append({
            "canonical": nct,
            "aliases": [],
            "frequency": count,
        })

    return entities


def extract_institutions(corpus_path: str) -> list[dict]:
    """Extract institution names from the corpus.

    Uses regex patterns to match university, institute, hospital, and
    foundation names. Deduplicates by normalized form (lowercased, trimmed).
    """
    institution_counts: Counter = Counter()

    # Compile patterns
    compiled_patterns = [re.compile(p) for p in INSTITUTION_PATTERNS]

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line_clean = line.replace("<|endoftext|>", " ")
            for pattern in compiled_patterns:
                matches = pattern.findall(line_clean)
                for match in matches:
                    # Normalize: strip whitespace, collapse internal spaces
                    normalized = " ".join(match.split()).strip()
                    if len(normalized) < 10:
                        continue
                    # Skip matches that are too generic or fragments
                    skip_patterns = {
                        "the hospital", "a hospital", "the institute",
                        "a foundation", "the foundation", "the university",
                        "s hospital", "this hospital", "our hospital",
                        "general hospital", "the clinic", "the medical center",
                    }
                    if normalized.lower() in skip_patterns:
                        continue
                    # Must start with a capital letter (proper noun)
                    if not normalized[0].isupper():
                        continue
                    institution_counts[normalized] += 1

    # Deduplicate by case-insensitive matching, keeping the most frequent form
    dedup: dict[str, tuple[str, int]] = {}
    for name, count in institution_counts.items():
        key = name.lower()
        if key not in dedup or count > dedup[key][1]:
            dedup[key] = (name, count)
        else:
            # Accumulate counts for the same institution
            existing_name, existing_count = dedup[key]
            dedup[key] = (existing_name, existing_count + count)

    entities = []
    for key in sorted(dedup.keys()):
        name, count = dedup[key]
        if count >= 1:
            entities.append({
                "canonical": name,
                "aliases": [],
                "frequency": count,
            })

    # Sort by frequency descending
    entities.sort(key=lambda x: -x["frequency"])

    return entities


def build_entity_registry(corpus_path: str, min_frequency: int = 1) -> dict:
    """Build the complete entity registry from the training corpus.

    Extracts all five entity types (drugs, genes, proteins, trials,
    institutions) and assembles them into a structured JSON object with
    metadata and per-type entity lists.
    """
    print(f"Reading corpus: {corpus_path}")
    corpus_size_mb = Path(corpus_path).stat().st_size / (1024 * 1024)
    print(f"Corpus size: {corpus_size_mb:.1f} MB")
    print()

    # Step 1: Extract word frequencies (single pass for drugs, genes, proteins)
    print("Extracting word frequencies...")
    word_counts = extract_word_frequencies(corpus_path)
    print(f"  Unique tokens: {len(word_counts):,}")
    print(f"  Total tokens: {sum(word_counts.values()):,}")
    print()

    # Step 2: Extract each entity type
    print("Extracting drugs...")
    drugs = extract_drugs(word_counts)
    print(f"  Found {len(drugs)} drug entities")

    print("Extracting genes...")
    genes = extract_genes(word_counts)
    print(f"  Found {len(genes)} gene entities")

    print("Extracting proteins...")
    proteins = extract_proteins(word_counts)
    print(f"  Found {len(proteins)} protein entities")

    print("Extracting clinical trial NCT numbers...")
    trials = extract_trials(corpus_path)
    print(f"  Found {len(trials)} trial entities")

    print("Extracting institutions...")
    institutions = extract_institutions(corpus_path)
    print(f"  Found {len(institutions)} institution entities")
    print()

    # Step 3: Apply minimum frequency filter
    if min_frequency > 1:
        drugs = [e for e in drugs if e["frequency"] >= min_frequency]
        genes = [e for e in genes if e["frequency"] >= min_frequency]
        proteins = [e for e in proteins if e["frequency"] >= min_frequency]
        trials = [e for e in trials if e["frequency"] >= min_frequency]
        institutions = [e for e in institutions if e["frequency"] >= min_frequency]

    # Step 4: Sort each type by frequency descending
    drugs.sort(key=lambda x: -x["frequency"])
    genes.sort(key=lambda x: -x["frequency"])
    proteins.sort(key=lambda x: -x["frequency"])
    trials.sort(key=lambda x: -x["frequency"])
    institutions.sort(key=lambda x: -x["frequency"])

    # Step 5: Assemble registry
    registry = {
        "metadata": {
            "source": str(corpus_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "min_frequency": min_frequency,
            "entity_counts": {
                "drugs": len(drugs),
                "genes": len(genes),
                "proteins": len(proteins),
                "trials": len(trials),
                "institutions": len(institutions),
            },
        },
        "drugs": drugs,
        "genes": genes,
        "proteins": proteins,
        "trials": trials,
        "institutions": institutions,
    }

    return registry


def print_summary(registry: dict) -> None:
    """Print summary statistics for the entity registry."""
    meta = registry["metadata"]
    counts = meta["entity_counts"]
    total = sum(counts.values())

    print("=" * 60)
    print("Entity Registry Summary")
    print("=" * 60)
    print(f"Source: {meta['source']}")
    print(f"Generated: {meta['generated_at']}")
    print(f"Min frequency: {meta['min_frequency']}")
    print(f"Total entities: {total}")
    print()
    print("Counts by type:")
    for entity_type, count in counts.items():
        print(f"  {entity_type:15s} {count:5d}")
    print()

    # Show top 5 from each type
    for entity_type in ["drugs", "genes", "proteins", "trials", "institutions"]:
        entities = registry[entity_type]
        print(f"Top {entity_type} (by frequency):")
        for e in entities[:5]:
            aliases_str = f" (aliases: {', '.join(e['aliases'])})" if e["aliases"] else ""
            print(f"  {e['canonical']:30s}  freq={e['frequency']}{aliases_str}")
        if len(entities) > 5:
            print(f"  ... and {len(entities) - 5} more")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Build entity registry from the ALS training corpus"
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/train.txt",
        help="Path to training text file (default: data/processed/train.txt)",
    )
    parser.add_argument(
        "--output",
        default="eval/entity_registry.json",
        help="Output path for entity registry JSON (default: eval/entity_registry.json)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum corpus frequency to include an entity (default: 1)",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    registry = build_entity_registry(str(corpus_path), args.min_frequency)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"Entity registry saved to: {output_path}")
    print()
    print_summary(registry)


if __name__ == "__main__":
    main()
