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

Our work sits at the intersection of four research threads: domain-specific language model training, hallucination evaluation, retrieval-augmented generation for medical text, and neural scaling laws. We survey each below, positioning our contributions relative to the existing literature.

### 2.1 Domain-specific language models

The dominant paradigm in biomedical NLP has been to adapt general-purpose pretrained models to the biomedical domain through continued pretraining on domain-specific corpora. BioBERT ([Lee et al., 2020](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)) pioneered this approach by continuing BERT pretraining on PubMed abstracts and PMC full-text articles (approximately 18 billion words), achieving improvements of 0.62 F1 on biomedical NER, 2.80 F1 on relation extraction, and 12.24 F1 on question answering over the general-domain BERT baseline. SciBERT ([Beltagy et al., 2019](https://aclanthology.org/D19-1371.pdf)) took a complementary approach, training a BERT-base model from scratch on 3.1 billion tokens from Semantic Scholar, finding that a custom scientific vocabulary (scivocab) improves performance on scientific information extraction tasks over the default BERT vocabulary.

The shift from encoder-only to decoder-only architectures brought BioGPT ([Luo et al., 2022](https://arxiv.org/abs/2210.10341)), a 347M-parameter GPT-2-style model trained on 15 million PubMed abstracts. BioGPT achieved state-of-the-art results on PubMedQA and biomedical relation extraction, demonstrating that generative pretraining can capture biomedical knowledge effectively. BioMedLM ([Bolton et al., 2024](https://arxiv.org/html/2403.18421v1)) scaled this approach to 2.7 billion parameters trained on PubMed abstracts and full-text articles, showing competitive performance with much larger general-domain models on medical question answering. At the largest scale, GatorTron ([Yang et al., 2022](https://arxiv.org/abs/2203.03540)) trained an 8.9-billion-parameter model on over 90 billion words combining clinical notes, PubMed articles, and Wikipedia, achieving the best results on clinical NLI benchmarks.

Table 1 summarizes the key characteristics of these models alongside ALS-LM.

**Table 1.** Domain-specific biomedical language models. ALS-LM operates at 77x below the Chinchilla-optimal data ratio, with a corpus 20-35x smaller than the nearest comparable decoder model (BioGPT).

| Model    | Parameters | Training Data                      | Architecture    | Year | Key Result                                       |
|----------|------------|------------------------------------|-----------------|------|--------------------------------------------------|
| BioBERT  | 110M       | PubMed abstracts + PMC (18B words) | BERT-base       | 2020 | +0.62 F1 on biomedical NER over BERT             |
| SciBERT  | 110M       | Semantic Scholar (3.1B tokens)     | BERT-base       | 2019 | Custom scivocab improves over BERT on SciIE      |
| BioGPT   | 347M       | 15M PubMed abstracts               | GPT-2 medium    | 2022 | State-of-art on PubMedQA, relation extraction    |
| GatorTron | 8.9B      | 90B+ words (82B clinical)          | BERT-like       | 2022 | Best clinical NLI on MedNLI                      |
| BioMedLM | 2.7B       | PubMed abstracts + full text       | GPT-2 style     | 2022 | Competitive with larger models on medical QA     |
| ALS-LM   | 516M       | 143M tokens (ALS only)             | GPT-2 (Pre-LN)  | 2026 | 0.52% accuracy; demonstrates data deficit impact |

The scale difference is immediately visible. Even BioBERT, the smallest model in the table, trained on a corpus approximately 125 times larger than ours (18 billion words vs 143 million tokens). BioGPT, the closest architectural comparison at 347M parameters, used approximately 15 million abstracts that we estimate contain 3-5 billion tokens, placing it 20-35x above our data volume. Unlike these models, which benefit from broad biomedical coverage across many diseases and subdomains, our work deliberately investigates the data-starved regime: a single-disease corpus where the available literature is fundamentally insufficient for the model size. Our results confirm that this data deficit is the dominant factor in the model's near-zero factual accuracy, even as it achieves healthy language modeling loss.

### 2.2 Hallucination evaluation

Evaluating factual accuracy and hallucination in language models has become an active research area as model capabilities have scaled. TruthfulQA ([Lin et al., 2022](https://arxiv.org/abs/2109.07958)) introduced a benchmark of 817 adversarial questions targeting common misconceptions, finding that larger models are often less truthful than smaller ones because they more effectively learn the statistical patterns of human misconceptions in their training data. FActScore ([Min et al., 2023](https://arxiv.org/abs/2305.14251)) took a fine-grained approach, decomposing generated biographies into atomic facts and scoring each against a reference corpus, enabling precision measurement at the individual claim level rather than the response level.

SelfCheckGPT ([Manakul et al., 2023](https://arxiv.org/abs/2303.08896)) addressed the reference-free setting by leveraging the observation that factual claims tend to be consistent across multiple stochastic samples while hallucinated claims vary, enabling hallucination detection without ground truth. In the medical domain, Med-HALT ([Umapathi et al., 2023](https://arxiv.org/abs/2307.15343)) tested large language models on reasoning and memory tasks derived from medical licensing exams, establishing structured categories of medical hallucination. MedHallu ([Chen et al., 2025](https://arxiv.org/abs/2502.14302)) extended this with a controlled hallucination generation pipeline producing 10,000 medical question-answer pairs across multiple hallucination types.

Table 2 situates our evaluation approach relative to these benchmarks.

**Table 2.** Hallucination evaluation approaches. Our benchmark is the only domain-specific evaluation combining curated questions, entity-based fabrication detection, and multi-mode failure taxonomy on a narrow medical subdomain.

| Benchmark/Tool | Approach                      | Year | Scale           | Key Innovation                                       |
|----------------|-------------------------------|------|-----------------|------------------------------------------------------|
| TruthfulQA     | 817 adversarial questions     | 2022 | General domain  | Targets common misconceptions                        |
| FActScore      | Fine-grained factual scoring  | 2023 | Biography       | Per-atomic-fact precision                            |
| SelfCheckGPT   | Self-consistency checking     | 2023 | General         | No reference needed; uses stochastic sampling        |
| Med-HALT       | Medical hallucination test    | 2023 | Medical         | Reasoning + memory tests from medical exams          |
| MedHallu       | 10K medical QA pairs          | 2025 | Medical         | Controlled hallucination generation pipeline         |
| ALS-LM Eval   | 160 curated ALS questions     | 2026 | ALS-specific    | 5-mode taxonomy + entity-based fabrication detection |

Our evaluation framework differs from these approaches in three respects. First, our benchmark is domain-specific rather than general medical or general knowledge: 160 questions curated across 8 categories of ALS knowledge (clinical trials, diagnostic criteria, disease mechanisms, drug treatment, epidemiology, gene mutations, patient care, and temporal accuracy), each with expert-defined key facts for scoring. Second, we implement entity-based fabrication detection using a registry of approximately 48,000 known entities (6,469 drugs, 20,173 genes, 8,075 proteins, and 13,354 institutions), allowing us to flag not just incorrect answers but specifically fabricated entities that do not exist in the medical literature. Third, our 5-mode failure taxonomy (confident fabrication, plausible blending, outdated information, boundary confusion, and accurate but misleading) enables structured analysis of failure mechanisms rather than binary correct/incorrect classification. This taxonomy proved essential for understanding the qualitative differences between our from-scratch model's failures (dominated by degenerate output and repetitive loops) and the baseline model's failures (dominated by confident fabrication of plausible-sounding but incorrect medical claims).

### 2.3 RAG for medical and scientific text

Retrieval-augmented generation has emerged as a strategy for grounding language model outputs in verified external knowledge. In the biomedical domain, BioASQ ([Tsatsaronis et al., 2015](https://www.nature.com/articles/s41597-023-02068-4)) established a long-running challenge for biomedical question answering with retrieval, providing a standardized benchmark that has driven progress since 2013. More recently, MedRAG and the MIRAGE benchmark ([Xiong et al., 2024](https://arxiv.org/abs/2402.13178)) evaluated RAG across multiple medical corpora, finding that retrieval-augmented approaches can improve accuracy by up to 18% over chain-of-thought baselines, though performance varies substantially depending on corpus selection and retrieval configuration.

Our RAG comparison tests a different experimental condition from most published work. Existing RAG benchmarks typically augment models that already possess substantial parametric medical knowledge with retrieval, measuring the incremental gain from adding context. In our setup, the from-scratch ALS-LM has near-zero factual accuracy (0.52%), creating a scenario where almost all correct answers must come from retrieved context rather than parametric knowledge. We compare four RAG configurations (two embedding models at two chunk sizes) against a no-retrieval Llama 3.1 8B baseline that has strong parametric knowledge from general pretraining.

Our findings diverge from the optimistic RAG literature. Even the best RAG configuration (500-token chunks with PubMedBERT embeddings, 13.8% accuracy) does not exceed the no-retrieval baseline (14.3%). The failure decomposition reveals that retrieval failures account for 52-89% of wrong answers depending on configuration, with the embedding model as the dominant variable: PubMedBERT-based retrieval averages 12.7% accuracy compared to 5.9% for MiniLM, a 2.1x improvement. This suggests that naive chunk-based retrieval with general-purpose embeddings is insufficient for domain-specific medical question answering, and that retrieval quality, not generation capability, is the primary bottleneck in this setting.

### 2.4 Data scaling laws

The relationship between training data volume and model performance is well-established through empirical scaling laws. [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) demonstrated that language model loss follows smooth power-law relationships with model size, dataset size, and compute budget, enabling predictions of model performance from training configuration. The Chinchilla study ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) refined these findings, establishing that compute-optimal training requires approximately 20 tokens per parameter, suggesting that models and data should be scaled in roughly equal proportion.

These scaling laws provide a quantitative framework for understanding our results. ALS-LM trains at 0.26 tokens per parameter (128.5 million training tokens for a 500-million-parameter model), placing us at 77 times below the Chinchilla-optimal ratio. If the power-law relationships hold in this extreme regime, the model should achieve far worse loss than a compute-optimally trained model of the same size, and the gap between language modeling competence and downstream task performance should widen dramatically.

In practice, we observe a more nuanced picture. The model achieves a Well-fit classification with validation loss 5.4956 and a relative gap of +0.42% between training and validation loss, suggesting that it has effectively learned the statistical distribution of its training corpus without significant overfitting. Yet factual accuracy on our benchmark is 0.52%, effectively zero. This disconnect suggests that the scaling laws governing loss may have different implications for different types of downstream capability: the model can learn to produce text that statistically resembles ALS research (low perplexity) without internalizing the factual content of that research (near-zero accuracy). Our work provides an empirical data point at the extreme low end of the data scaling curve, complementing the large-scale studies that established these relationships.

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
