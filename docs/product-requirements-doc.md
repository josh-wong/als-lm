# ALS-LM: Product requirements document

**Author:** [josh-wong](https://github.com/josh-wong)
**Date:** February 21, 2026
**Status:** Approved

---

## 1. Purpose

This document defines the scope, requirements, and success criteria for ALS-LM, which is a domain-specific language model trained from scratch on ALS (amyotrophic lateral sclerosis) knowledge. It serves as the bridge between the project's research goals (outlined in the white paper) and its technical implementation (detailed in the design document).

## 2. Problem statement

Domain-specific knowledge systems face a fundamental architectural choice: internalize knowledge into model weights or retrieve it at query time from external sources. For sensitive medical domains, this choice has direct implications for accuracy, safety, and trustworthiness. There is limited publicly available research comparing these approaches at reproducible scales by using consumer-grade hardware.

ALS-LM exists to explore this question concretely. The "product" is not a medical tool; rather, it is a documented, reproducible experiment that produces a working model, a hallucination evaluation framework, and a comparative analysis against RAG-based approaches.

## 3. Target users

This project serves three audiences:

**Primary: Portfolio reviewers (hiring managers, technical leads).** People evaluating the author's ML engineering, data curation, and responsible AI skills. They need to see clear thinking, professional documentation, and honest analysis of results.

**Secondary: ML practitioners and students.** People interested in training models from scratch on domain-specific data, particularly in medical or other sensitive domains. They benefit from the reproducible methodology and documented lessons learned.

**Tertiary: Medical AI researchers.** People studying hallucinations and safety in medical language models. They may find value in the failure taxonomy and evaluation benchmark.

## 4. Scope

Scope defines what is included and excluded from the project.

### 4.1 In scope

- Data collection pipeline for ALS-specific content from public sources
- Data cleaning, deduplication, and processing pipeline
- Custom BPE tokenizer trained on the ALS corpus
- Decoder-only transformer model (500M–1B parameters) trained from scratch
- Hallucination evaluation benchmark (160 factual Q&A pairs)
- Failure taxonomy and categorization of model outputs
- RAG comparison pipeline that uses the same corpus with a pretrained model
- Model export to GGUF format for Ollama compatibility
- Simple CLI demo for querying the model
- Project documentation (README, white paper, product requirements document, design document)

### 4.2 Out of scope

- Any production-facing medical information system
- Fine-tuning of existing pretrained models beyond the controlled comparison added in v0.8.0 (GPT-2 large 774M fine-tuned on the ALS corpus to measure the effect of pretrained knowledge)
- Web-based or graphical user interfaces
- Mobile applications
- Multi-language support (the model trains on English-language content; Japanese-language ALS content is excluded to keep scope manageable)
- Real-time data ingestion or model updating
- HIPAA compliance or handling of private medical data
- API hosting or cloud deployment

## 5. Functional requirements

Functional requirements specify the features and behaviors expected from ALS-LM.

### FR-1: Data collection pipeline

The data collection pipeline gathers ALS-specific content from multiple public sources into a standardized format.

| ID     | Requirement                                                                                    | Priority    |
|--------|------------------------------------------------------------------------------------------------|-------------|
| FR-1.1 | Scrape open-access ALS papers from PubMed Central via the PMC API                              | Must have   |
| FR-1.2 | Collect ALS clinical trial records from ClinicalTrials.gov API                                 | Must have   |
| FR-1.3 | Scrape educational content from ALS Association, MDA, NIH, and similar public sources          | Must have   |
| FR-1.4 | Collect published patient narratives from public blogs and talks                               | Should have |
| FR-1.5 | Collect supplementary neuroscience background material                                         | Should have |
| FR-1.6 | Log all sources with URL, access date, license, and ethical justification in `data/sources.md` | Must have   |

### FR-2: Data processing pipeline

The processing pipeline cleans, deduplicates, and normalizes raw data into a training-ready corpus.

| ID     | Requirement                                                                       | Priority  |
|--------|-----------------------------------------------------------------------------------|-----------|
| FR-2.1 | Strip HTML, boilerplate, and non-content elements from scraped data               | Must have |
| FR-2.2 | Deduplicate content at the document and paragraph level                           | Must have |
| FR-2.3 | Remove volatile content (schedules, contact info, "currently" statements)         | Must have |
| FR-2.4 | Normalize text encoding and formatting                                            | Must have |
| FR-2.5 | Generate corpus statistics (total size, source distribution, vocabulary analysis) | Must have |
| FR-2.6 | Create 90/10 train/validation split                                               | Must have |

### FR-3: Tokenizer

The tokenizer must efficiently encode medical terminology specific to the ALS domain.

| ID     | Requirement                                                                                                            | Priority    |
|--------|------------------------------------------------------------------------------------------------------------------------|-------------|
| FR-3.1 | Train a custom BPE tokenizer on the processed corpus                                                                   | Must have   |
| FR-3.2 | Vocabulary size configurable (final: 50,257 tokens after experimental validation)                                       | Must have   |
| FR-3.3 | Validate that key medical terms (drug names, gene names, diagnostic terms) are represented as single or minimal tokens | Must have   |
| FR-3.4 | Document tokenizer analysis and vocabulary coverage                                                                    | Should have |

### FR-4: Model training

Model training must support transformer architectures at 500M–1B scale within consumer hardware constraints.

| ID     | Requirement                                                                       | Priority    |
|--------|-----------------------------------------------------------------------------------|-------------|
| FR-4.1 | Implement a decoder-only transformer architecture (GPT-2 style)                   | Must have   |
| FR-4.2 | Support model sizes from 500M to 1B parameters                                    | Must have   |
| FR-4.3 | Train by using DeepSpeed ZeRO with CPU offloading to accommodate VRAM constraints | Must have   |
| FR-4.4 | Implement gradient checkpointing for memory efficiency                            | Must have   |
| FR-4.5 | Log training metrics (loss, perplexity, learning rate) at regular intervals       | Must have   |
| FR-4.6 | Support checkpoint saving and resumption for long training runs                   | Must have   |
| FR-4.7 | Generate training loss curves and convergence visualizations                      | Should have |

### FR-5: Model export and Ollama integration

The export pipeline converts trained models to GGUF format for local inference via Ollama.

| ID     | Requirement                                                              | Priority  |
|--------|--------------------------------------------------------------------------|-----------|
| FR-5.1 | Export trained model to GGUF format                                      | Must have |
| FR-5.2 | Support quantization during export (Q4_K_M, Q8_0, F16 produced)           | Must have |
| FR-5.3 | Create an Ollama Modelfile with appropriate parameters and system prompt | Must have |
| FR-5.4 | Model loads and runs in Ollama via `ollama run als-lm`                   | Must have |
| FR-5.5 | Document Ollama setup and usage instructions                             | Must have |

### FR-6: CLI demo

The CLI demo provides an interactive interface for querying the model with appropriate safety disclaimers.

| ID     | Requirement                                                                        | Priority     |
|--------|------------------------------------------------------------------------------------|--------------|
| FR-6.1 | Provide a Python-based CLI for querying the model interactively                    | Must have    |
| FR-6.2 | Display a disclaimer on launch stating this is not a medical resource              | Must have    |
| FR-6.3 | Support configurable generation parameters (temperature, top-k, top-p, max tokens) | Should have  |
| FR-6.4 | Support both direct model inference and Ollama as backends                         | Should have  |
| FR-6.5 | Log all queries and outputs for evaluation purposes                                | Should have  |
| FR-6.6 | Include a `/benchmark` command that runs the hallucination evaluation suite        | Nice to have |

### FR-7: Hallucination evaluation

The evaluation framework measures factual accuracy and categorizes failure modes against a curated benchmark.

| ID     | Requirement                                                                                                                                               | Priority    |
|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| FR-7.1 | Create a benchmark of 160 factual ALS questions with verified answers                                                                                     | Must have   |
| FR-7.2 | Implement automated scoring against the benchmark                                                                                                         | Must have   |
| FR-7.3 | Categorize failures by using the failure taxonomy (confident fabrication, plausible blending, outdated information, boundary confusion, accurate but misleading) | Must have   |
| FR-7.4 | Run the same benchmark against a RAG baseline                                                                                                             | Must have   |
| FR-7.5 | Generate a comparison report with accuracy rates and failure severity analysis                                                                            | Must have   |
| FR-7.6 | Include qualitative sample outputs (both good and bad) in the evaluation report                                                                           | Should have |

### FR-8: RAG comparison baseline

The RAG baseline enables direct comparison between internalized knowledge and retrieval-augmented approaches.

| ID     | Requirement                                                                       | Priority  |
|--------|-----------------------------------------------------------------------------------|-----------|
| FR-8.1 | Build a RAG pipeline that uses the same ALS corpus as the document store          | Must have |
| FR-8.2 | Use a pretrained open-source model (e.g., Llama 3.1 8B or Mistral 7B) as the base | Must have |
| FR-8.3 | Implement vector-based retrieval by using an embedding model and vector store     | Must have |
| FR-8.4 | Evaluate against the same hallucination benchmark for direct comparison           | Must have |

## 6. Non-functional requirements

Non-functional requirements address hardware, reproducibility, and documentation.

### NFR-1: Hardware compatibility

All training and inference must run on consumer-grade hardware without requiring cloud resources.

| ID      | Requirement                                                                   | Priority    |
|---------|-------------------------------------------------------------------------------|-------------|
| NFR-1.1 | All training runs on a single RTX 3060 (12GB VRAM) with 64GB system RAM       | Must have   |
| NFR-1.2 | Training completes within 2 weeks for the largest model configuration         | Should have |
| NFR-1.3 | Inference runs locally on the same hardware via Ollama without noticeable lag | Must have   |

### NFR-2: Reproducibility

The project must be reproducible by third parties with equivalent hardware and API access.

| ID      | Requirement                                                                        | Priority    |
|---------|------------------------------------------------------------------------------------|-------------|
| NFR-2.1 | All data collection scripts are runnable by a third party with the same API access | Must have   |
| NFR-2.2 | Random seeds are fixed and documented for all training runs                        | Must have   |
| NFR-2.3 | A `requirements.txt` or equivalent locks all dependency versions                   | Must have   |
| NFR-2.4 | Training can be reproduced on any machine with at least 12GB VRAM and 32GB RAM     | Should have |

### NFR-3: Documentation

Documentation must be sufficient for a third party to understand, set up, and run the project.

| ID      | Requirement                                                                        | Priority  |
|---------|------------------------------------------------------------------------------------|-----------|
| NFR-3.1 | README provides setup instructions sufficient for a third party to run the project | Must have |
| NFR-3.2 | All data sources are inventoried with license and access information               | Must have |
| NFR-3.3 | Ethical considerations are documented and visible                                  | Must have |

## 7. Success criteria

The project is considered successful if the following are achieved:

**Minimum success (all must be met):**

1. A trained model of at least 500M parameters that generates coherent, ALS-related text when prompted.
2. The model runs in Ollama on the target hardware.
3. The hallucination benchmark is complete with at least 100 questions.
4. The RAG comparison is implemented and produces a quantitative comparison.
5. All project documentation (README, white paper, product requirements document, design document) is complete.

**Target success (demonstrates strong portfolio value):**

6. The model scores above 40% accuracy on the factual benchmark (acknowledging this is a high bar for a from-scratch model at this scale).
7. The failure taxonomy reveals measurable differences in failure profiles between the from-scratch model and RAG baseline.
8. The project includes a published blog post summarizing findings.
9. Training is reproducible by a third party following the documentation.

**Stretch goals:**

10. A fine-tuned model is included as a controlled comparison (achieved: GPT-2 large 774M fine-tuned on ALS corpus).
11. The hallucination benchmark is reusable for other ALS-related models.
12. The CLI demo includes the `/benchmark` command for on-demand evaluation.

## 8. Milestones

The project is divided into six phases, from planning through final documentation.

| Phase             | Milestone                                                                    | Estimated duration |
|-------------------|------------------------------------------------------------------------------|--------------------|
| 1 – Planning      | README, white paper, product requirements document, design document complete | 1–2 weeks          |
| 2 – Data          | Collection scripts written and run, corpus assembled and cleaned             | 2–3 weeks          |
| 3 – Tokenizer     | Custom tokenizer trained and validated                                       | 2–3 days           |
| 4 – Training      | Model trained, checkpointed, and initial evaluation complete                 | 2–4 weeks          |
| 5 – Export        | GGUF export, Ollama integration, CLI demo functional                         | 3–5 days           |
| 6 – Evaluation    | Benchmark created, evaluation run, failure taxonomy applied                  | 1–2 weeks          |
| 7 – Comparison    | RAG baseline built and benchmarked                                           | 1–2 weeks          |
| 8 – Documentation | Final writeup, blog post, repo polish                                        | 1 week             |

**Total estimated timeline: 8–12 weeks**

## 9. Risks and mitigations

The following risks were identified during planning with corresponding mitigation strategies.

| Risk                                             | Likelihood | Impact | Mitigation                                                                                                                                                       |
|--------------------------------------------------|------------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Training exceeds VRAM even with offloading       | Medium     | High   | Start with 500M params, scale up only if stable. Have a fallback config ready.                                                                                   |
| Insufficient corpus size for meaningful learning | Low        | High   | PubMed alone has more than enough content. Risk is low given source availability.                                                                                |
| Model produces harmful medical misinformation    | High       | Medium | Hallucination evaluation framework, clear disclaimers, responsible publication guidelines. Impact is medium because the model is not deployed as a medical tool. |
| Training takes longer than expected              | Medium     | Medium | Checkpoint frequently. Treat partial results as valid for analysis. A model that didn't fully converge is still interesting to evaluate.                         |
| PubMed or ClinicalTrials.gov API changes         | Low        | Medium | Cache all downloaded data locally. Document API versions.                                                                                                        |
| Scope creep into fine-tuning or larger models    | Medium     | Low    | PRD clearly defines scope. Fine-tuning was added as a controlled comparison (GPT-2 large 774M) rather than a production objective.                               |

## 10. Dependencies

The project relies on the following external tools and services, each with an identified fallback.

| Dependency             | Purpose                         | Fallback                                            |
|------------------------|---------------------------------|-----------------------------------------------------|
| PubMed Central API     | Research paper collection       | Manual download of open-access papers               |
| ClinicalTrials.gov API | Clinical trial data             | Bulk download available from site                   |
| nanoGPT                | Training framework              | Hugging Face Transformers with custom training loop |
| DeepSpeed              | Memory-efficient training       | Gradient accumulation + smaller batch sizes         |
| llama.cpp              | GGUF conversion                 | ctransformers or manual conversion                  |
| Ollama                 | Local model serving             | llama.cpp CLI directly                              |
| FAISS or ChromaDB      | Vector store for RAG comparison | Simple TF-IDF retrieval as baseline                 |
