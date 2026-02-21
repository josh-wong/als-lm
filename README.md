# ALS-LM: A Domain-Specific Language Model for ALS Knowledge

This repository contains ALS-LM, a domain-specific language model focused on ALS knowledge.

> [!NOTE]
> 
> This project explores the intersection of NLP, medical informatics, and responsible AI development. It is an independent personal project and is not affiliated with any employer or medical institution.
> 
> *If you find an issue with this project's approach to medical content or data ethics, please open an issue. Constructive feedback on the ethical framework is especially welcome.*

## Overview

ALS-LM is a domain-specific language model (500M–1B parameters) trained from scratch on a curated corpus of amyotrophic lateral sclerosis (ALS) research, clinical trial data, and educational content. The project explores what a purpose-built, domain-specific model can learn about a complex medical topic and, just as importantly, where it fails.

This is not a medical tool. It is a machine learning research and education project that uses ALS as its knowledge domain.

## Motivation

The motivation for this project is to understand how focused models learn and represent medical knowledge, particularly in a complex and high-stakes domain like ALS.

Most people interact with large language models as black boxes. This project asks a more fundamental question: **if you train a focused model on a specific body of medical knowledge, what does it actually learn?**

ALS was chosen as the target domain for several reasons:

- **Rich public data.** Tens of thousands of open-access research papers exist on PubMed Central, ClinicalTrials.gov provides structured trial data, and major medical institutions publish extensive educational materials.
- **Well-defined boundaries.** ALS is a specific diagnosis with a clear body of associated research, making it possible to define a meaningful corpus at a manageable scale.
- **Meaningful subject matter.** ALS remains a disease with limited treatment options and active research frontiers. Building tools—even experimental ones—in this space is worth doing carefully and thoughtfully.

## Project goals

The project goals are outlined below to clarify the scope and objectives.

1. **Demonstrate end-to-end language model development.** From data collection and curation through tokenizer training, model architecture, training, and evaluation.
2. **Explore domain-specific model behavior.** Evaluate how a purpose-built model internalizes structured medical knowledge—what it retains accurately, what it confuses, and where it hallucinates.
3. **Compare architectural approaches.** Benchmark the from-scratch model against retrieval-augmented generation (RAG) by using the same corpus, with a focus on accuracy, safety, and failure modes in a sensitive domain.
4. **Document the process transparently.** Every decision, tradeoff, and mistake is recorded. The journey is as valuable as the result.

## ⚠️ Important disclaimers

Please read the following disclaimers carefully before using or referencing ALS-LM.

### This is not a medical resource

ALS-LM is a machine learning research project. It is **not** a diagnostic tool, treatment guide, or substitute for professional medical advice. The model will generate text that sounds authoritative but may be factually incorrect, outdated, or misleading.

**If you or someone you know is affected by ALS, please consult qualified healthcare providers and trusted resources such as:**

- [ALS Association](https://www.als.org/)
- [Mayo Clinic – ALS Overview](https://www.mayoclinic.org/diseases-conditions/amyotrophic-lateral-sclerosis/symptoms-causes/syc-20354022)
- [NIH National Institute of Neurological Disorders and Stroke](https://www.ninds.nih.gov/health-information/disorders/amyotrophic-lateral-sclerosis-als)

### On hallucinations and medical safety

Hallucinations are a known risk in language models, especially in medical contexts, and a model of this size **will hallucinate**. It will occasionally generate plausible-sounding statements that are medically inaccurate. In a general-knowledge context, this is a known limitation. In a medical context, it is a potential harm.

This project takes that risk seriously by:

- Clearly labeling all outputs as generated and unverified.
- Evaluating the model specifically on hallucination rates for known medical facts.
- Comparing hallucination behavior against RAG-based approaches to ask: **which architecture is safer for sensitive domains?**
- Never framing this model as a usable medical information system.

## Ethical framework

The ethical framework guides data collection, model training, and publication.

### Data ethics

All training data is sourced from publicly available, appropriately licensed material:

| Source                       | Type                     | License/access                 |
|------------------------------|--------------------------|--------------------------------|
| PubMed Central               | Research papers          | Open Access subset             |
| ClinicalTrials.gov           | Clinical trial records   | Public domain (US government)  |
| ALS Association, MDA         | Educational content      | Public web content             |
| NIH, WHO, FDA                | Medical reference        | Public domain (US government)  |
| Published patient narratives | Blog posts, public talks | Publicly shared by individuals |

**What this project does not use:**

- Private medical records or patient data of any kind
- Content from private support groups or closed communities
- Data scraped from forums or communities where participants had a reasonable expectation of privacy
- Any data subject to HIPAA or equivalent protections

### Patient narrative policy

When including patient perspectives, this project only uses content that individuals have intentionally published for public audiences—blog posts, published essays, public talks, and official testimonial pages. Even for public content, the project documents the source and rationale for inclusion.

The project does not attempt to simulate or generate patient voices. The model may learn patterns from patient narratives, but any demo or evaluation interface makes it clear that outputs are machine-generated.

### Responsible disclosure

If this model is found to generate content that could cause specific, identifiable harm (e.g., dangerous treatment misinformation), the relevant outputs and analysis will be documented in the evaluation section and, if necessary, flagged to relevant communities.

## Hallucination evaluation

A core focus of this project is understanding and measuring how a domain-specific, from-scratch model handles factual medical knowledge. Rather than treating hallucinations as an unfortunate side effect, this project treats them as a primary research question: **how do from-scratch, domain-specific models fail in sensitive domains, and how do those failures compare to retrieval-augmented approaches?**

### Evaluation benchmark

The project includes a hand-curated benchmark of 100–200 factual questions about ALS with verified correct answers, covering:

- **Drug and treatment knowledge.** Can the model correctly name FDA-approved treatments? Does it invent nonexistent drugs?
- **Gene and mutation associations.** Does it accurately link genes like SOD1 and C9orf72 to ALS? Does it fabricate gene names or associations?
- **Diagnostic criteria.** Can it describe the El Escorial criteria or distinguish ALS from similar conditions like PLS or Kennedy's disease?
- **Clinical trial literacy.** Does it accurately reflect the status of known trials, or does it generate fictional results?
- **Temporal accuracy.** Does it confuse historical and current treatment landscapes (e.g., citing treatments that failed trials as if they were approved)?

### Failure taxonomy

Model outputs are categorized by failure mode:

- **Confident fabrication.** Stating false information with no hedging.
- **Plausible blending.** Combining real facts into a false composite (e.g., attributing the wrong mechanism to a real drug).
- **Outdated information.** Generating accurate-at-some-point information that is no longer current.
- **Boundary confusion.** Answering questions outside the ALS domain by pulling from loosely related training patterns.
- **Accurate but misleading.** Technically correct statements presented without critical context.

### Comparative analysis

The same benchmark is run against:

1. The from-scratch ALS-LM model.
2. A RAG pipeline that uses the same corpus with a pretrained model.
3. (Optional) A fine-tuned 7B open-source model as an upper baseline.

The comparison focuses not just on accuracy rates but on **failure severity**—a model that says "I don't know" is safer than one that confidently fabricates a drug name, even if both score the same on a simple accuracy metric.

## Project documents

| Document                                                          | Description                                               |
|-------------------------------------------------------------------|-----------------------------------------------------------|
| [White paper](docs/white-paper.md)                                | Research motivation, approach, and expected contributions |
| [Product requirements document](docs/product-requirements-doc.md) | Scope, requirements, and success criteria                 |
| [Design document](docs/design-doc.md)                             | Technical architecture and implementation plan            |

## Repository structure

The repository structure is organized for clarity and reproducibility.

```
als-lm/
├── README.md
├── docs/
│   ├── white-paper.md
│   ├── product-requirements-doc.md
│   └── design-doc.md
├── data/
│   ├── scrapers/            # Data collection scripts
│   ├── processing/          # Cleaning and normalization pipeline
│   ├── sources.md           # Full source inventory and licensing
│   └── stats.md             # Corpus statistics and analysis
├── tokenizer/               # Custom tokenizer training and analysis
├── model/                   # Model architecture and training code
├── export/                  # GGUF conversion and Ollama Modelfile
├── cli/                     # CLI demo for interactive querying
├── eval/                    # Evaluation scripts, benchmarks, results
│   ├── benchmark/           # Curated factual Q&A benchmark
│   └── failure-taxonomy/    # Hallucination categorization and analysis
├── samples/                 # Curated output examples (good and bad)
├── comparison/              # RAG comparison implementation and analysis
├── notebooks/               # Exploratory analysis and experiments
└── blog/                    # Project writeup and lessons learned
```

## Technical summary

> Full details in the [design document](docs/design-doc.md).

- **Model size:** 500M–1B parameters (determined by corpus size and training experiments)
- **Architecture:** Decoder-only transformer (GPT-2 style via nanoGPT)
- **Training data:** ~100MB curated ALS corpus from public sources
- **Tokenizer:** Custom BPE trained on the corpus, optimized for medical vocabulary
- **Hardware:** NVIDIA RTX 3060 (12GB VRAM), 64GB system RAM, Intel i5-12400
- **Comparison:** RAG pipeline that uses the same corpus with a pretrained model as baseline
- **Export:** GGUF format for Ollama compatibility (Q4_K_M, Q5_K_M, Q8_0 quantization)
- **Interface:** CLI demo with interactive querying and benchmark runner

## Current status

📋 **Phase: Planning and documentation**

- [x] Project README and ethical framework
- [x] White paper
- [x] Product requirements document
- [x] Design document
- [ ] Data collection pipeline
- [ ] Data cleaning and processing
- [ ] Tokenizer training
- [ ] Model training
- [ ] GGUF export and Ollama integration
- [ ] CLI demo
- [ ] Evaluation
- [ ] Hallucination benchmark creation
- [ ] RAG comparison
- [ ] Final writeup

## License

Code in this repository is released under the MIT License. Training data sources retain their original licenses as documented in `data/sources.md`. Model weights, if published, will include the disclaimers outlined in this README.
