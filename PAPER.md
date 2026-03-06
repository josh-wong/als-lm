# ALS-LM: Investigating Domain-Specific Language Model Training on a Narrow Medical Corpus

## Abstract

Domain-specific language models such as BioBERT, BioGPT, and GatorTron have demonstrated that pretraining on biomedical corpora improves downstream clinical and scientific tasks, benefiting from billions of tokens of broad biomedical literature. We investigate the opposite extreme: what can a purpose-built decoder-only transformer (516M parameters, GPT-2 architecture with Pre-LN) learn from just 143M tokens of curated amyotrophic lateral sclerosis (ALS) research? At 0.26 tokens per parameter, this represents a 77x deficit relative to the Chinchilla-optimal ratio.

We construct a reproducible pipeline spanning data collection from three public sources (19,164 documents), BPE tokenizer training with domain-specific vocabulary (50/100 top medical terms as single tokens), and model training using DeepSpeed ZeRO Stage 2 on consumer hardware (NVIDIA RTX 3060, 12GB VRAM). Training for 3 epochs (11,760 steps, 4h 32m) achieves a Well-fit classification with validation loss 5.4956 (relative gap +0.42%), yet the model attains only 0.52% mean factual accuracy on our 160-question ALS benchmark, demonstrating the gap between language modeling competence and factual knowledge.

We further evaluate retrieval-augmented generation (RAG) using ChromaDB with dual embedding models. Domain-specific embeddings (PubMedBERT) outperform general-purpose embeddings (MiniLM) by 2.1x on mean accuracy (12.7% vs 5.9%), but even the best RAG configuration (13.8%) does not exceed the no-retrieval baseline (14.3%), revealing retrieval quality as the primary bottleneck. We contribute an end-to-end open-source pipeline, a hallucination evaluation framework with a 5-mode failure taxonomy and entity-based fabrication detection (~48K entities), and honest documentation of negative results that illuminate data requirements for domain-specific model training.

## 1. Introduction

The past five years have seen an accelerating trend toward domain-specific language models in biomedicine. BioBERT ([Lee et al., 2020](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)) demonstrated that continued pretraining of BERT on PubMed abstracts and full-text articles improves biomedical named entity recognition by 0.62 F1 points over the general-domain baseline. BioGPT ([Luo et al., 2022](https://arxiv.org/abs/2210.10341)) achieved state-of-the-art results on PubMedQA and biomedical relation extraction using a GPT-2-style decoder trained on 15 million PubMed abstracts. GatorTron ([Yang et al., 2022](https://arxiv.org/abs/2203.03540)), at 8.9 billion parameters trained on over 90 billion words of clinical and biomedical text, pushed the boundaries of clinical natural language inference. These models share a common thread: massive training corpora spanning broad biomedical domains, ranging from billions to tens of billions of tokens.

We ask a different question. What happens when we dramatically reduce both the breadth and volume of training data, focusing on a single disease domain with a corpus orders of magnitude smaller than what these models consume? Specifically, we train a 516M-parameter decoder-only transformer from scratch on 143M tokens of curated ALS research literature. At 0.26 tokens per parameter, this places our model 77 times below the Chinchilla-optimal ratio of approximately 20 tokens per parameter ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). We are not merely data-limited; we are operating in a regime where conventional scaling laws predict that the model cannot possibly acquire meaningful factual knowledge.

This is by design. Our research question is not whether a small model trained on narrow data can compete with larger systems. It cannot, and our results confirm this decisively. Instead, we investigate what such a model does learn, how it fails, and what the failure modes reveal about the relationship between training data volume, language modeling loss, and factual accuracy. The answer turns out to be more nuanced than simple failure: our model achieves a Well-fit classification with a validation loss relative gap of just +0.42%, indicating that it has learned generalizable statistical patterns of ALS research language, yet it attains only 0.52% mean factual accuracy on a domain-specific benchmark. This disconnect between language modeling competence and factual knowledge acquisition is itself an informative finding.

We emphasize that this project is a machine learning research and education artifact, not a medical information tool. The model should never be used for clinical decision-making, patient education, or any application where factual accuracy matters. The hallucination evaluation framework we develop exists to quantify the model's unreliability, not to demonstrate its utility. Any outputs from the model or from the RAG comparison system should be treated as experimental results, not as medical information.

Our investigation makes four contributions:

1. **A complete, open-source pipeline from data collection to evaluation.** We provide scripts for scraping PubMed Central, ClinicalTrials.gov, and educational sources; data cleaning with MinHash deduplication; BPE tokenizer training with medical term validation; model training with DeepSpeed ZeRO on consumer hardware; GGUF export for local inference via Ollama; and automated evaluation. Every stage is scripted and reproducible with fixed random seeds and locked dependencies.

2. **A hallucination evaluation framework with a 5-mode failure taxonomy.** We design a 160-question benchmark spanning 8 ALS knowledge categories (20 questions each) with key-fact-based fuzzy matching scoring. We implement entity-based fabrication detection using a registry of approximately 48,000 known entities (drugs, genes, proteins, institutions) and classify failures into five modes: confident fabrication, plausible blending, outdated information, boundary confusion, and accurate but misleading. This taxonomy enables structured analysis of how and why models fail, not merely that they fail.

3. **A RAG comparison revealing retrieval as the bottleneck.** We evaluate four RAG configurations using ChromaDB with two embedding models (MiniLM and PubMedBERT) at two chunk sizes (200 and 500 tokens) against a no-retrieval Llama 3.1 8B baseline. The results demonstrate that embedding model choice is the single most impactful variable (PubMedBERT outperforms MiniLM by 2.1x), but that even the best RAG configuration does not exceed the no-retrieval baseline, pointing to retrieval quality rather than generation capability as the primary limitation.

4. **Honest documentation of a negative result.** The machine learning literature suffers from a well-known publication bias toward positive results. We contribute a detailed analysis of a project that achieves excellent language modeling metrics but near-zero factual accuracy, providing an empirical data point at the extreme low end of the data scaling curve. The dual narrative of rigorous pipeline engineering alongside transparent negative results is itself a contribution to the field's understanding of data requirements for domain-specific models.

The remainder of this paper is organized as follows. Section 2 surveys related work across domain-specific language models, hallucination evaluation, medical RAG, and data scaling laws. Section 3 details our methodology, covering the data pipeline, tokenizer, model architecture, and training procedure. Section 4 describes our evaluation framework, including benchmark design, scoring methodology, fabrication detection, and failure taxonomy. Section 5 presents training results and hallucination evaluation findings across three quantization levels. Section 6 reports the RAG comparison experiment and failure decomposition analysis. Section 7 discusses the implications of our findings, with particular attention to the data deficit, embedding model impact, and the loss-accuracy gap. Section 8 concludes with a summary and directions for future work.

## 2. Related Work

<!-- Content to be written in Task 2 -->

## 3. Methodology

### 3.1 Data pipeline

<!-- Content to be written in subsequent plans -->

### 3.2 Tokenizer

<!-- Content to be written in subsequent plans -->

### 3.3 Model architecture

<!-- Content to be written in subsequent plans -->

### 3.4 Training

<!-- Content to be written in subsequent plans -->

## 4. Evaluation framework

### 4.1 Benchmark design

<!-- Content to be written in subsequent plans -->

### 4.2 Scoring

<!-- Content to be written in subsequent plans -->

### 4.3 Fabrication detection

<!-- Content to be written in subsequent plans -->

### 4.4 Failure taxonomy

<!-- Content to be written in subsequent plans -->

## 5. Results

### 5.1 Training results

<!-- Content to be written in subsequent plans -->

### 5.2 Hallucination evaluation

<!-- Content to be written in subsequent plans -->

### 5.3 Quantization impact

<!-- Content to be written in subsequent plans -->

## 6. RAG comparison

### 6.1 Experimental setup

<!-- Content to be written in subsequent plans -->

### 6.2 Baseline vs RAG results

<!-- Content to be written in subsequent plans -->

### 6.3 Failure decomposition

<!-- Content to be written in subsequent plans -->

## 7. Discussion

### 7.1 Data deficit and scaling laws

<!-- Content to be written in subsequent plans -->

### 7.2 Embedding model impact

<!-- Content to be written in subsequent plans -->

### 7.3 Loss-accuracy gap

<!-- Content to be written in subsequent plans -->

### 7.4 Implications

<!-- Content to be written in subsequent plans -->

## 8. Conclusion

### 8.1 Summary

<!-- Content to be written in subsequent plans -->

### 8.2 Future work

<!-- Content to be written in subsequent plans -->

## References

<!-- Content to be written in Task 3 -->
