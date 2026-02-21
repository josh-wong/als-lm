# ALS-LM: Exploring Domain-Specific Language Models for Sensitive Medical Knowledge

**Author:** [josh-wong](https://github.com/josh-wong)
**Date:** February 21, 2026
**Status:** Approved
**Version:** 1.0

---

## Abstract

Large language models have demonstrated impressive general-purpose capabilities, but their behavior in specialized medical domains raises significant questions about accuracy, safety, and trustworthiness. This paper presents ALS-LM, a project that trains a domain-specific language model (500M–1B parameters) from scratch on a curated corpus of amyotrophic lateral sclerosis (ALS) research and educational content. The project's primary contribution is not the model itself but a structured investigation into how domain-specific, from-scratch models internalize medical knowledge, where they fail, and how their failure modes compare to retrieval-augmented generation (RAG) approaches that use the same corpus. By treating hallucination measurement as a first-class research objective, the project aims to contribute practical insights into the question of which architectures are most appropriate for sensitive domain knowledge.

## 1. Introduction

The proliferation of large language models (LLMs) has created a tension in medical informatics. On one hand, these models can synthesize and surface medical knowledge in accessible ways. On the other hand, their tendency to hallucinate—generating plausible but factually incorrect statements—poses a particular risk in medical contexts where inaccurate information can cause real harm.

Most research in this space focuses on large-scale models (7B parameters and above) that are fine-tuned on medical corpora. Less attention has been given to a more fundamental question: what happens when a purpose-built model is trained from scratch on a tightly scoped medical domain? What does it actually learn, what does it fail to learn, and how do those failures differ from retrieval-based approaches?

ALS-LM addresses this gap by training a 500M–1B parameter transformer model on approximately 100MB of curated ALS content, such as research papers, clinical trial data, educational materials, and published patient narratives. The project then benchmarks this model against a RAG pipeline that uses the same corpus, evaluating not just accuracy but the nature and severity of each approach's failures.

## 2. Background

This section reviews relevant background and prior work.

### 2.1 Domain-specific language models in medicine

The medical AI landscape has seen several notable efforts to build domain-specific models. Models like PubMedBERT, BioGPT, and Med-PaLM have demonstrated that domain-specific training or fine-tuning can improve medical question-answering performance. However, these efforts typically operate at scales (billions of parameters, massive compute budgets) that are inaccessible to independent researchers.

Less explored is the lower end of the scale: what can a model with fewer than 1B parameters, trained on a focused corpus with consumer-grade hardware, actually accomplish in a medical domain? The answer has practical implications for resource-constrained deployments and for understanding the relationship between model capacity, domain scope, and factual reliability.

### 2.2 The hallucination problem in medical contexts

Hallucinations are a critical issue for medical language models, with potential risks for users.

Hallucinations in LLMs are well-documented, but their severity varies by domain. In creative writing or casual conversation, hallucinations may be acceptable or even desirable. In medical contexts, they are potentially dangerous. A model that confidently states an incorrect drug dosage or invents a nonexistent gene association creates risk that goes beyond simple inaccuracy.

Existing approaches to mitigating hallucinations include retrieval-augmented generation (grounding outputs in source documents), reinforcement learning from human feedback (RLHF), and constrained decoding. This project focuses on the first of these as a comparison baseline, asking whether retrieval grounding offers a measurable safety advantage over internalized knowledge for medical content.

### 2.3 Why ALS?

ALS was selected as the target domain for several practical and methodological reasons:

- **Corpus availability.** ALS research is well-represented in open-access literature. PubMed Central contains thousands of open-access papers, ClinicalTrials.gov provides structured trial data in the public domain, and major health organizations publish extensive educational materials.
- **Domain boundaries.** ALS is a specific neurological diagnosis with a well-defined body of associated research, genetics, and clinical practice. This makes it possible to construct a corpus that is both comprehensive and bounded, unlike broader domains like "oncology" or "cardiology" where the scope would be difficult to manage at this scale.
- **Active research frontiers.** ALS treatment options remain limited, and the research landscape is active and evolving. This creates natural opportunities to evaluate temporal accuracy. For example, does the model distinguish between approved treatments and experimental ones?
- **Structured knowledge.** ALS knowledge includes clear factual relationships (gene-mutation-phenotype associations, drug-mechanism-trial outcome chains) that lend themselves to benchmark evaluation.
- **Personal connection.** The father of the author of this white paper suffered from ALS, which provided firsthand insight into the impact of the disease and motivated the choice of ALS as the project's domain.

## 3. Approach

The approach section outlines data collection, model design, and evaluation.

### 3.1 Data collection and curation

Data collection and curation ensure the corpus is relevant, clean, and ethically sourced. The training corpus targets approximately 100MB of clean, deduplicated text drawn from the following source categories:

- **Biomedical research literature (40–50% of corpus).** Open-access papers from PubMed Central, prioritizing review articles and meta-analyses for their knowledge density. Papers are filtered by license (CC-BY, CC-BY-NC, or public domain) and relevance (ALS as primary subject).
- **Clinical trial data (15–20%).** Full-text descriptions from ClinicalTrials.gov, including trial designs, eligibility criteria, intervention descriptions, and outcome summaries. This data is public domain as a US government resource.
- **Educational and institutional content (15–20%).** Published materials from the ALS Association, Muscular Dystrophy Association, Mayo Clinic, NIH, and similar organizations. This content tends to be more accessible in language than research papers, providing the model with exposure to both technical and patient-facing registers.
- **Published patient narratives (10–15%).** Blog posts, public essays, and transcripts from public talks by individuals who have chosen to share their ALS experiences publicly. This category is handled with particular care (see Section 5, Ethical Considerations).
- **Supplementary scientific context (5–10%).** Background material on motor neuron biology, neurodegeneration mechanisms, and genetics relevant to ALS, providing the model with foundational context for understanding the primary ALS literature.

A key curation principle is **temporal stability**. Content that changes frequently (clinic schedules, support group meeting times, evolving treatment protocols presented as current fact) is excluded or normalized. The goal is a corpus of durable knowledge rather than a snapshot of current events.

### 3.2 Tokenizer

A custom byte-pair encoding (BPE) tokenizer is trained on the corpus by using the Hugging Face `tokenizers` library. The primary motivation is efficient handling of medical vocabulary. Terms like "neurodegeneration," "fasciculation," "riluzole," and "superoxide dismutase" should be represented as single or few tokens rather than fragmented into subword pieces by a general-purpose tokenizer. Vocabulary size will be determined experimentally in the 8K–32K range, balancing coverage against model capacity.

### 3.3 Model architecture

The model follows a standard decoder-only transformer architecture (GPT-2 style), implemented via Andrej Karpathy's nanoGPT as a starting point. Target model size is 500M–1B parameters, with the final size determined by training stability and hardware constraints.

Training is conducted on a consumer-grade setup: NVIDIA RTX 3060 (12GB VRAM), 64GB system RAM, and Intel i5-12400. To accommodate model sizes that exceed available VRAM, the project employs DeepSpeed ZeRO Stage 2/3 with CPU offloading and gradient checkpointing. The large system RAM (64GB) is a significant asset for CPU offloading, making 1B-parameter training feasible at the cost of increased training time.

### 3.4 Evaluation framework

Evaluation is structured around a curated benchmark of 100–200 factual questions with verified correct answers spanning drug knowledge, gene associations, diagnostic criteria, clinical trial literacy, and temporal accuracy.

Model outputs are categorized by using a failure taxonomy (see Section 4) that distinguishes between types of errors by both kind and severity. The same benchmark is applied to a RAG pipeline that uses the same corpus with a pretrained base model, enabling direct comparison of failure modes.

## 4. Failure taxonomy

A central contribution of this project is a structured taxonomy for categorizing model failures in medical domains. The taxonomy distinguishes between the following failure modes:

- **Confident fabrication.** The model states false information with no hedging or uncertainty markers. Example: inventing a drug name or claiming a fictional clinical trial showed positive results. Severity: high.
- **Plausible blending.** The model combines real facts into a false composite. Example: correctly naming a real drug but attributing the wrong mechanism of action, or associating a real gene with the wrong disease variant. Severity: high, because the output contains enough real information to appear credible.
- **Outdated information.** The model generates information that was accurate at some point but is no longer current. Example: describing a drug as being in active trials when the trial has since concluded or failed. Severity: moderate, and particularly interesting for temporal analysis.
- **Boundary confusion.** The model generates content outside the ALS domain by drawing on loosely related patterns. Example: when asked about ALS genetics, producing content about Huntington's disease genetics instead. Severity: moderate, and useful for understanding domain scope limitations.
- **Accurate but misleading.** The model produces technically correct statements that lack critical context. Example: correctly stating that a drug reduced motor neuron loss in mice without noting that it failed in human trials. Severity: moderate to high, depending on context.

This taxonomy is applied consistently to both the from-scratch model and the RAG baseline, enabling comparison not just of accuracy rates but of failure profiles.

## 5. Ethical considerations

Ethical considerations are central to the project's design and publication.

### 5.1 Medical information risk

This project does not produce a medical tool. All project documentation, demo interfaces, and published outputs carry clear disclaimers that the model is a research artifact, not a source of medical advice. The hallucination evaluation framework exists specifically to quantify the model's unreliability, not to demonstrate its usefulness as an information source.

### 5.2 Patient data and privacy

The project uses no private medical data of any kind. Patient narratives included in the training corpus are limited to content that individuals have intentionally published for public audiences. The project does not scrape private forums, support groups, or any content protected by medical privacy regulations (HIPAA or equivalent). The [data sources document](data/sources.md) provides a full inventory of all sources with licensing and ethical justification.

### 5.3 Responsible publication

If the trained model weights are published, they will be accompanied by the full disclaimer framework and a model card documenting known failure modes and limitations. The project will not publish the model in any format that encourages or facilitates its use as a medical information system.

### 5.4 Patient narrative representation

The model may learn linguistic patterns from patient narratives in its training data. Any demo or evaluation interface makes clear that all outputs are machine-generated and do not represent the voices or experiences of real patients. The project does not attempt to generate simulated patient narratives as a use case.

## 6. Expected contributions

This project aims to contribute to the understanding of domain-specific language models in three ways:

- **Practical insights on domain-specific model behavior in medical contexts.** Most published work on medical language models operates at scales beyond independent replication. This project provides a documented case study at a reproducible scale, including detailed analysis of what a sub-1B model can and cannot learn about a specific medical domain.
- **A structured framework for evaluating medical hallucinations.** The failure taxonomy and benchmark methodology are designed to be reusable. Other researchers working on domain-specific medical models can adopt the evaluation framework for their own domains.
- **An empirical comparison of architectural approaches for sensitive knowledge.** The from-scratch versus RAG comparison, with its emphasis on failure severity rather than simple accuracy, offers practical guidance for practitioners deciding how to deploy medical knowledge systems.

## 7. Limitations

This project has several known limitations that should be acknowledged:

- **Model scale.** A 500M–1B parameter model is orders of magnitude smaller than state-of-the-art medical language models. The project's findings about model behavior may not generalize to larger scales.
- **Corpus scope.** Approximately 100MB of training data is modest by current standards. The model's knowledge will have gaps even within the ALS domain.
- **Single-domain focus.** ALS was chosen for practical reasons, but findings may not transfer to medical domains with different characteristics (e.g., higher ambiguity, broader scope, less structured knowledge).
- **Hardware constraints.** Consumer-grade GPU training with CPU offloading may introduce training dynamics (slower convergence, different batch size constraints) that would not be present at larger scales.
- **Evaluation subjectivity.** Some failure taxonomy categories (particularly "accurate but misleading") require subjective judgment. The project will document inter-rater agreement where applicable.

## 8. Conclusion

ALS-LM is an intentionally focused project that asks a big question: what is the right way to build AI systems that handle sensitive medical knowledge? By training a model from scratch, evaluating its failures rigorously, and comparing it against retrieval-augmented alternatives, the project aims to produce insights that are useful beyond its specific domain and scale.

The project prioritizes transparency, reproducibility, and ethical responsibility. Every decision—from data sourcing to model publication—is documented and justified. The goal is not to build a better medical chatbot but to understand the tradeoffs involved in doing so, and to share that understanding openly.
