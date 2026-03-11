# ALS-LM-2: Product requirements document

**Author:** [josh-wong](https://github.com/josh-wong)
**Date:** March 11, 2026
**Last revised:** March 11, 2026
**Status:** Draft

---

## 1. Purpose

This document defines the scope, requirements, and success criteria for ALS-LM-2, which investigates whether instruction tuning, data quality improvements, and parameter scaling can address the failure modes identified in [ALS-LM-1](v1-research-paper.md). It serves as the bridge between the project's research vision (outlined in the [v2 white paper](v2-white-paper.md)) and its technical implementation (detailed in the forthcoming design document).

The [v1 product requirements document](v1-product-requirements-doc.md) scoped the original experiment: training a domain-specific model from scratch on ALS research, evaluating it against a hallucination benchmark, and comparing it with retrieval-augmented generation. This v2 PRD extends that scope with three coordinated interventions targeting the specific failure modes that v1 revealed, while maintaining the project's identity as a research and education artifact rather than a medical tool.

## 2. Problem statement

ALS-LM-1 demonstrated a fundamental disconnect between language-modeling competence and factual accuracy. The from-scratch 516M-parameter model achieved a Well-fit training classification (validation loss [5.4956](v1-research-paper.md), relative gap +0.42%) while attaining only [0.21% mean factual accuracy](v1-research-paper.md) on a 160-question domain-specific benchmark. Training on [143M tokens](v1-research-paper.md) at 0.25 tokens per parameter placed the model 80 times below the Chinchilla-optimal ratio, establishing data deficit as the dominant bottleneck for knowledge acquisition.

Two additional experiments isolated orthogonal failure dimensions. Fine-tuning GPT-2 large ([774M parameters](v1-research-paper.md)) on the same ALS corpus improved accuracy 15-fold to [3.12%](v1-research-paper.md) but produced degenerate output for [97.5%](v1-research-paper.md) of evaluation questions, revealing that completion-based architectures without instruction-following alignment cannot express their knowledge through structured question answering. A retrieval-augmented generation comparison established a [14.3% no-retrieval baseline](v1-research-paper.md) that no RAG configuration exceeded, with the best configuration (PubMedBERT embeddings, 500-token chunks) achieving [13.8%](v1-research-paper.md), identifying retrieval quality rather than generation capability as the primary bottleneck.

These findings define the v2 research agenda. ALS-LM-2 investigates three coordinated hypotheses: (1) expanding and improving the training corpus addresses the data deficit, (2) scaling from 516M to 1B parameters provides additional capacity for encoding factual relationships, and (3) supervised fine-tuning on instruction-response pairs resolves the degenerate output problem. The product is not a medical tool; it is a documented, reproducible investigation into whether these three interventions, applied together, produce measurably better results than any single approach achieved in v1.

## 3. Target users

This project serves the same three audiences identified in the [v1 product requirements document](v1-product-requirements-doc.md), with the extended investigation adding depth to each audience's engagement.

**Primary: Portfolio reviewers (hiring managers, technical leads).** People evaluating the author's ML engineering, data curation, and responsible AI skills. The v2 investigation demonstrates the ability to iterate on research findings, apply instruction tuning techniques, and conduct rigorous cross-version evaluation. They need to see clear thinking, professional documentation, and honest analysis of whether interventions produced the expected improvements.

**Secondary: ML practitioners and students.** People interested in training domain-specific models, particularly in medical or other sensitive domains. The v2 investigation extends the reproducible methodology with instruction dataset creation, supervised fine-tuning, and cross-version comparison techniques. They benefit from documented lessons about what instruction tuning can and cannot accomplish at consumer-accessible scales.

**Tertiary: Medical AI researchers.** People studying hallucinations and safety in medical language models. The v2 investigation provides additional data points on the relationship between instruction tuning and perceived capability, extending the failure taxonomy if new failure modes emerge. They may find value in the cross-approach comparison spanning from-scratch training, pretrained fine-tuning, instruction tuning, and retrieval-augmented generation.

## 4. Scope

This section defines what ALS-LM-2 includes and excludes, beginning with context from the v1 investigation that motivates the v2 scope decisions.

### 4.1 v1.0.0 context

ALS-LM-1 built a complete pipeline from data collection through evaluation: scrapers for three public sources (PubMed Central, ClinicalTrials.gov, and educational content) produced a [143M-token](v1-research-paper.md) corpus, a custom BPE tokenizer encoded medical terminology, and a [516M-parameter](v1-research-paper.md) decoder-only transformer was trained from scratch using DeepSpeed ZeRO Stage 2 on consumer hardware (RTX 3060, 12GB VRAM). Training completed in [4 hours and 27 minutes](v1-research-paper.md) using [6.37 GB peak VRAM](v1-research-paper.md). The model was exported to GGUF format for Ollama compatibility, and a [160-question](v1-research-paper.md) hallucination benchmark with a 5-mode failure taxonomy evaluated factual accuracy. A RAG comparison using ChromaDB with dual embedding models completed the investigation.

The quantitative results established clear baselines for v2. The from-scratch model achieved [0.21% mean accuracy](v1-research-paper.md) despite healthy training metrics, confirming that language-modeling competence does not imply knowledge acquisition at [0.25 tokens per parameter](v1-research-paper.md) ([80x below Chinchilla-optimal](v1-research-paper.md)). A controlled fine-tuning comparison using GPT-2 large ([774M parameters](v1-research-paper.md)) improved accuracy to [3.12%](v1-research-paper.md) but produced degenerate output for [97.5%](v1-research-paper.md) of questions (156 out of 160), isolating the instruction-following limitation as orthogonal to data deficit. The best RAG configuration (PubMedBERT 500-token) achieved [13.8%](v1-research-paper.md) accuracy, falling short of the [14.3% no-retrieval baseline](v1-research-paper.md). These results motivate the three-hypothesis approach described in the [v2 white paper](v2-white-paper.md).

### 4.2 In scope

The following items define the boundaries of the ALS-LM-2 investigation.

- Corpus expansion and quality improvement targeting the data deficit identified in v1
- Text normalization and cleaning to address punctuation artifacts and whitespace issues from PDF extraction
- Re-tokenization of the expanded and cleaned corpus with updated vocabulary statistics
- Training a 1B-parameter model on the improved corpus with resource monitoring and logging
- Creation of an instruction Q&A dataset with factual validation against the ALS corpus
- Supervised fine-tuning (SFT) of the 1B model on the instruction dataset
- Export of the instruction-tuned model to GGUF format for Ollama compatibility
- Adaptation of the hallucination evaluation framework for instruction-formatted prompts
- Re-running the hallucination benchmark on the instruction-tuned model
- RAG re-comparison using the best v1 configuration only (PubMedBERT 500-token)
- Updated documentation (research paper, model card, README)
- Release packaging (tag, GitHub release)

### 4.3 Out of scope

The following items are excluded from the ALS-LM-2 investigation.

- **3B+ parameter models:** The 3B configuration may be explored if the 1B training run demonstrates sufficient hardware headroom (less than 10 GB peak VRAM, completion within 24 hours), but it is not a committed deliverable. See Section 7.5 for the conditional success criterion.
- **DPO/RLHF alignment:** Hardware constraints and project scope limit the investigation to supervised fine-tuning only. Reinforcement learning from human feedback and direct preference optimization are excluded.
- **Full RAG re-run (all 4 configurations):** Only the best-performing v1 configuration (PubMedBERT 500-token) is re-run. The v1 investigation already established the full RAG comparison across all configurations.
- **Web-based or graphical user interfaces:** The evaluation interface remains a command-line demo, consistent with v1 scope.
- **API hosting or cloud deployment:** All training and inference run locally on consumer hardware.
- **Production medical tool:** This remains a research project, not a deployable medical information system. All interfaces carry disclaimers.
- **Non-English content:** The model trains on English-language content only, consistent with v1 scope.
- **HIPAA compliance or private medical data:** All training data comes from publicly available sources. No private patient data is used.
- **Mobile applications:** No mobile deployment is planned.
- **Real-time data ingestion:** No live data feeds or continuous model updating.

## 5. Functional requirements

Functional requirements are organized into five groups aligned to the v2 milestones. Each group opens with a traceability paragraph linking the requirements to the relevant [white paper](v2-white-paper.md) hypothesis and [v1 findings](v1-research-paper.md).

### FR-1: Corpus expansion (v1.2.0)

FR-1 addresses [Hypothesis 1 (data quality)](v2-white-paper.md), targeting the [80x Chinchilla deficit](v1-research-paper.md) identified in ALS-LM-1 where the from-scratch model trained on just [0.25 tokens per parameter](v1-research-paper.md). Expanding and improving the training corpus aims to increase the tokens-per-parameter ratio and improve the effective information density of training data. This group extends the data pipeline work from [v1 FR-1 (data collection)](v1-product-requirements-doc.md) and [v1 FR-2 (data processing)](v1-product-requirements-doc.md).

| ID     | Requirement                                                                                                | Priority    |
|--------|------------------------------------------------------------------------------------------------------------|-------------|
| FR-1.1 | Fix text normalization issues including punctuation artifacts and whitespace inconsistencies from PDF extraction | Must have   |
| FR-1.2 | Identify and evaluate new ALS-relevant sources beyond the three categories used in v1                       | Must have   |
| FR-1.3 | Implement scrapers for approved new sources following the v1 data collection patterns                       | Must have   |
| FR-1.4 | Validate corpus quality through statistical comparison with the v1 corpus (size, source distribution, deduplication rate) | Must have   |
| FR-1.5 | Re-train the BPE tokenizer on the expanded and cleaned corpus with updated vocabulary statistics            | Must have   |
| FR-1.6 | Document all new sources with URL, access date, license, and ethical justification                          | Must have   |

### FR-2: 1B model training (v1.3.0)

FR-2 addresses [Hypothesis 2 (parameter scaling)](v2-white-paper.md), investigating whether scaling from [516M to 1B parameters](v1-research-paper.md) provides additional capacity for encoding factual relationships. The ALS-LM-1 from-scratch model used [6.37 GB peak VRAM](v1-research-paper.md) during training, suggesting headroom for a larger model on the same hardware. This group extends [v1 FR-4 (model training)](v1-product-requirements-doc.md).

| ID     | Requirement                                                                                                | Priority    |
|--------|------------------------------------------------------------------------------------------------------------|-------------|
| FR-2.1 | Train a 1B-parameter decoder-only transformer on the expanded corpus within consumer hardware constraints   | Must have   |
| FR-2.2 | Log resource usage (peak VRAM, system RAM, training time per epoch) throughout the training run              | Must have   |
| FR-2.3 | Track and report training metrics (loss, perplexity, learning rate) at regular intervals                    | Must have   |
| FR-2.4 | Implement checkpoint saving and resumption consistent with v1 practices                                     | Must have   |
| FR-2.5 | Produce a resource usage report that informs the go/no-go decision on a potential 3B model configuration    | Should have |

### FR-3: Instruction tuning (v1.4.0)

FR-3 addresses [Hypothesis 3 (instruction tuning)](v2-white-paper.md), targeting the [97.5% degenerate output rate](v1-research-paper.md) observed in the ALS-LM-1 fine-tuned GPT-2 large model. Supervised fine-tuning on instruction-response pairs aims to teach the model to respond to structured questions rather than producing completion-style text. This is a new capability for ALS-LM with no corresponding v1 FR group.

| ID     | Requirement                                                                                                | Priority    |
|--------|------------------------------------------------------------------------------------------------------------|-------------|
| FR-3.1 | Create an instruction Q&A dataset covering key ALS knowledge categories from the evaluation benchmark       | Must have   |
| FR-3.2 | Validate instruction dataset entries for factual accuracy against the ALS corpus and authoritative sources   | Must have   |
| FR-3.3 | Perform supervised fine-tuning of the 1B base model on the instruction dataset                               | Must have   |
| FR-3.4 | Verify that the instruction-tuned model produces structured Q&A responses rather than degenerate output      | Must have   |
| FR-3.5 | Export the instruction-tuned model to GGUF format with Ollama compatibility                                  | Must have   |
| FR-3.6 | Create an Ollama Modelfile with appropriate parameters and system prompt for the instruction-tuned model      | Should have |

### FR-4: Evaluation and comparison (v1.5.0)

FR-4 addresses all three hypotheses by measuring the combined effect of data quality, parameter scaling, and instruction tuning against [ALS-LM-1 baselines](v1-research-paper.md). The evaluation framework must preserve cross-version comparability while accommodating instruction-formatted prompts. This group extends [v1 FR-7 (hallucination evaluation)](v1-product-requirements-doc.md) and [v1 FR-8 (RAG comparison baseline)](v1-product-requirements-doc.md).

| ID     | Requirement                                                                                                | Priority    |
|--------|------------------------------------------------------------------------------------------------------------|-------------|
| FR-4.1 | Adapt the evaluation harness to support instruction-formatted prompts while preserving the same questions and key facts | Must have   |
| FR-4.2 | Run the hallucination benchmark on the instruction-tuned model and report results in the same format as v1  | Must have   |
| FR-4.3 | Re-run the best RAG configuration (PubMedBERT 500-token) against the instruction-tuned model               | Must have   |
| FR-4.4 | Produce a cross-version comparison spanning from-scratch, fine-tuned, instruction-tuned, and RAG results    | Must have   |
| FR-4.5 | Extend the 5-mode failure taxonomy if instruction-tuned model outputs reveal failure modes not captured by existing categories | Should have |
| FR-4.6 | Measure the gap between perceived capability and actual accuracy for the instruction-tuned model             | Should have |

### FR-5: Release packaging (v2.0.0)

FR-5 covers the release deliverables for ALS-LM-2, ensuring that all documentation, model artifacts, and project metadata are updated and published. This group extends [v1 FR-5 (model export and Ollama integration)](v1-product-requirements-doc.md) and [v1 FR-6 (CLI demo)](v1-product-requirements-doc.md).

| ID     | Requirement                                                                                                | Priority    |
|--------|------------------------------------------------------------------------------------------------------------|-------------|
| FR-5.1 | Write an updated research paper documenting v2 methodology, findings, and cross-version analysis            | Must have   |
| FR-5.2 | Update the model card for the 1B instruction-tuned model with known failure modes and limitations            | Must have   |
| FR-5.3 | Create a GitHub release with version tag, release notes, and updated README                                  | Must have   |
| FR-5.4 | Update project documentation to reflect the full v1-to-v2 investigation arc                                  | Must have   |

## 6. Non-functional requirements

Non-functional requirements address hardware, reproducibility, documentation, and v2-specific quality constraints. The first three NFR groups carry forward from the [v1 product requirements document](v1-product-requirements-doc.md) with v2 extensions.

### NFR-1: Hardware compatibility

All training and inference must run on consumer-grade hardware without requiring cloud resources.

| ID      | Requirement                                                                                           | Priority    |
|---------|-------------------------------------------------------------------------------------------------------|-------------|
| NFR-1.1 | All training (base model and SFT) runs on a single RTX 3060 (12GB VRAM) with 64GB system RAM          | Must have   |
| NFR-1.2 | 1B model training completes within a reasonable timeframe given consumer hardware constraints           | Should have |
| NFR-1.3 | Inference for the instruction-tuned model runs locally via Ollama without noticeable lag                | Must have   |
| NFR-1.4 | Resource usage (peak VRAM, system RAM, training time) is logged for all training runs                   | Must have   |

### NFR-2: Reproducibility

The project must be reproducible by third parties with equivalent hardware and API access.

| ID      | Requirement                                                                                           | Priority    |
|---------|-------------------------------------------------------------------------------------------------------|-------------|
| NFR-2.1 | All data collection and processing scripts are runnable by a third party with the same API access      | Must have   |
| NFR-2.2 | Random seeds are fixed and documented for all training runs including SFT                              | Must have   |
| NFR-2.3 | A `requirements.txt` or equivalent locks all dependency versions including any new v2 dependencies     | Must have   |
| NFR-2.4 | Instruction dataset creation methodology is documented sufficiently for independent replication         | Must have   |

### NFR-3: Documentation

Documentation must be sufficient for a third party to understand, set up, and run the complete v1-to-v2 investigation.

| ID      | Requirement                                                                                           | Priority    |
|---------|-------------------------------------------------------------------------------------------------------|-------------|
| NFR-3.1 | README provides setup instructions covering both v1 and v2 components                                  | Must have   |
| NFR-3.2 | All data sources (v1 and new v2 sources) are inventoried with license and access information            | Must have   |
| NFR-3.3 | Ethical considerations including the instruction-tuning perceived capability risk are documented         | Must have   |

### NFR-4: Instruction dataset quality

The instruction Q&A dataset must meet quality standards that ensure evaluation validity.

| ID      | Requirement                                                                                           | Priority    |
|---------|-------------------------------------------------------------------------------------------------------|-------------|
| NFR-4.1 | All instruction dataset entries are validated for factual accuracy against authoritative sources        | Must have   |
| NFR-4.2 | The instruction dataset covers the same knowledge categories as the evaluation benchmark                | Should have |

### NFR-5: Cross-version evaluation compatibility

Evaluation results must be comparable across model versions to support the cross-approach analysis.

| ID      | Requirement                                                                                           | Priority    |
|---------|-------------------------------------------------------------------------------------------------------|-------------|
| NFR-5.1 | The evaluation benchmark preserves the same 160 questions and key facts used in v1                     | Must have   |
| NFR-5.2 | Results from all model versions (from-scratch, fine-tuned, instruction-tuned, RAG) can be presented in a single comparison table | Must have   |

## 7. Success criteria

Success criteria are defined in tiers anchored to [ALS-LM-1 baselines](v1-research-paper.md), consistent with the tiered framework established in the [v2 white paper](v2-white-paper.md). The minimum tier represents the threshold below which the investigation would be considered unsuccessful. The target tier represents the outcome that would validate the combined hypothesis. The stretch tier represents an aspirational outcome.

### 7.1 Accuracy

Accuracy is measured using the same [160-question ALS benchmark](v1-research-paper.md) and proportional key-fact fuzzy matching score used in ALS-LM-1. All accuracy values are reported at the Q8_0 quantization level.

| Tier    | Criterion                                                     | Rationale                                                                                               |
|---------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Minimum | Exceed fine-tuned GPT-2 large accuracy (3.12%)                | Demonstrates that the combined approach outperforms pretrained fine-tuning alone                        |
| Target  | Approach RAG no-retrieval baseline accuracy (14.3%)           | Demonstrates that a trained model can compete with parametric knowledge of a general 8B model           |
| Stretch | Exceed RAG no-retrieval baseline accuracy (14.3%)             | Demonstrates that domain-specific training with instruction tuning surpasses general pretrained knowledge |

### 7.2 Coherence

Coherence is measured as the percentage of non-degenerate responses on the evaluation benchmark. This metric directly addresses the [97.5% degenerate output rate](v1-research-paper.md) observed in the ALS-LM-1 fine-tuned model, compared to the [67.5% coherent rate](v1-research-paper.md) (108 out of 160 questions) from the from-scratch model.

| Tier    | Criterion                      | Rationale                                                                                |
|---------|--------------------------------|------------------------------------------------------------------------------------------|
| Minimum | >50% coherent responses        | Demonstrates that instruction tuning substantially resolves the degenerate output problem |
| Target  | >80% coherent responses        | Approaches the from-scratch model's 67.5% coherence while maintaining higher accuracy    |
| Stretch | >90% coherent responses        | Demonstrates that instruction tuning effectively eliminates degenerate output            |

### 7.3 RAG re-comparison

The ALS-LM-1 RAG comparison established that no RAG configuration exceeded the [14.3% no-retrieval baseline](v1-research-paper.md). Re-running the best configuration against the instruction-tuned model provides a controlled measure of improvement.

| Tier    | Criterion                                                                    | Rationale                                                                            |
|---------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Minimum | Instruction-tuned model evaluated against the same RAG baseline              | Ensures comparability with ALS-LM-1 results                                         |
| Target  | Model approaches or exceeds best RAG config (PubMedBERT 500-token, 13.8%)   | Demonstrates that training-based knowledge can match retrieval-augmented approaches |

### 7.4 Deliverable success criteria

The following criteria address project completeness independently of model performance. These are not tiered because they are binary requirements.

- The updated research paper is complete and documents v2 methodology, findings, and cross-version analysis with honest evaluation of results regardless of accuracy outcomes.
- The model card is updated for the 1B instruction-tuned model with documented failure modes, limitations, and the perceived capability risk.
- The evaluation benchmark is re-run on the instruction-tuned model with results presented alongside v1 results in a single comparison.
- All v2 milestones (v1.2.0 through v2.0.0) are shipped with corresponding documentation.

### 7.5 3B model feasibility (conditional)

If the 1B training run demonstrates sufficient hardware headroom, a 3B model may be feasible on the same consumer hardware. This is a conditional stretch criterion, not a committed deliverable. The ALS-LM-1 from-scratch model ([516M parameters](v1-research-paper.md)) used [6.37 GB peak VRAM](v1-research-paper.md) and trained in [4 hours and 27 minutes](v1-research-paper.md), providing a reference point for extrapolation.

| Condition                                                                    | Action                                                                       |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| 1B training uses less than 10 GB peak VRAM and completes within 24 hours     | 3B training may be attempted with ZeRO Stage 3 and aggressive CPU offloading |
| 1B training exceeds 10 GB peak VRAM or requires more than 24 hours           | 3B training is deferred; 1B model proceeds to instruction tuning             |

## 8. Milestones

The v2 investigation is divided into five milestones executed sequentially. Each milestone depends on the prior milestone's completion, as the pipeline is cumulative: improved data feeds into model training, which feeds into instruction tuning, which feeds into evaluation.

| Milestone | Description                                                                             | Estimated duration |
|-----------|-----------------------------------------------------------------------------------------|--------------------|
| v1.2.0    | Corpus expansion and data quality improvement, re-tokenization on expanded corpus       | 2–3 weeks          |
| v1.3.0    | 1B model training with resource monitoring, go/no-go data for 3B assessment             | 2–4 weeks          |
| v1.4.0    | Instruction dataset creation, supervised fine-tuning, GGUF export of instruction-tuned model | 2–3 weeks          |
| v1.5.0    | Evaluation harness adaptation, benchmark re-run, RAG re-comparison, cross-version analysis  | 1–2 weeks          |
| v2.0.0    | Updated research paper, model card, release packaging, README and documentation updates     | 1–2 weeks          |

**Total estimated timeline: 8–14 weeks**

## 9. Risks and mitigations

The following risks were identified during planning, combining risks that carry forward from [v1](v1-product-requirements-doc.md) with new risks specific to the v2 investigation.

| Risk                                                                     | Likelihood | Impact | Mitigation                                                                                                                                                      |
|--------------------------------------------------------------------------|------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1B training exceeds VRAM even with ZeRO Stage 2 CPU offloading          | Low        | High   | v1 516M model used 6.37 GB peak VRAM, leaving headroom. ZeRO Stage 3 is available as fallback.                                                                 |
| Training takes longer than expected on consumer hardware                 | Medium     | Medium | Checkpoint frequently. Treat partial results as valid for analysis. Adjust batch size or gradient accumulation if needed.                                       |
| Model produces harmful medical misinformation                            | High       | Medium | Hallucination evaluation framework, clear disclaimers, responsible publication guidelines. Impact is medium because the model is not deployed as a medical tool. |
| Instruction dataset contains factual errors                              | Medium     | High   | Validate all entries against the ALS corpus and authoritative sources. Cross-reference with evaluation benchmark key facts.                                     |
| SFT overfitting on small instruction dataset                             | Medium     | Medium | Monitor validation loss during SFT. Use early stopping if overfitting is detected. Keep instruction dataset diverse across knowledge categories.                |
| Instruction-tuned model appears more capable than it is                  | High       | Medium | Measure the gap between perceived capability and actual accuracy. Document this gap prominently in the model card. Display accuracy metrics alongside output.   |
| Cross-version evaluation results not directly comparable                 | Low        | Medium | Preserve the same 160 questions and key facts. Use the same scoring methodology. Report all results at Q8_0 quantization level.                                |

## 10. Dependencies

The project relies on the following external tools and services. Continuing v1 dependencies are listed alongside new v2-specific additions.

| Dependency                    | Purpose                                            | Version/notes                                                                       |
|-------------------------------|----------------------------------------------------|-------------------------------------------------------------------------------------|
| PubMed Central API            | Research paper collection (v1, extended in v2)     | Continuing from v1; expanded source queries for v2                                  |
| ClinicalTrials.gov API        | Clinical trial data (v1, continuing)               | Continuing from v1                                                                  |
| nanoGPT                       | Base training framework (v1, continuing)           | Continuing from v1; may require modifications for 1B scale                          |
| DeepSpeed                     | Memory-efficient training (v1, continuing)         | ZeRO Stage 2 primary; Stage 3 as fallback for 1B or potential 3B                    |
| PyTorch                       | ML framework (v1, continuing)                      | Continuing from v1; version locked in requirements.txt                              |
| Hugging Face tokenizers       | BPE tokenizer training (v1, re-run in v2)          | Continuing from v1; re-training on expanded corpus                                  |
| llama.cpp                     | GGUF conversion (v1, continuing)                   | Continuing from v1; export for instruction-tuned model                              |
| Ollama                        | Local model serving (v1, continuing)               | Continuing from v1; updated Modelfile for instruction-tuned model                   |
| ChromaDB                      | Vector store for RAG re-comparison (v1, continuing) | Continuing from v1; re-run with best config only                                    |
| Instruction dataset tools     | Q&A dataset creation for SFT (NEW)                 | Specific tooling determined in design doc                                           |
| SFT framework                 | Supervised fine-tuning pipeline (NEW)               | Specific framework determined in design doc                                         |
| Evaluation harness (updated)  | Instruction-formatted prompt evaluation (UPDATED)  | Adapted from v1 harness; preserves cross-version comparability                      |
