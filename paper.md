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

The relationship between training data volume and model performance is well-established through empirical scaling laws. [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) demonstrated that language model loss follows smooth power-law relationships with model size, dataset size, and compute budget, enabling predictions of model performance from training configuration. The Chinchilla study ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) refined these findings, establishing that compute-optimal training requires approximately 20 tokens per parameter, suggesting that models and data should be scaled in roughly equal proportion.

These scaling laws provide a quantitative framework for understanding our results. ALS-LM trains at 0.26 tokens per parameter (128.5 million training tokens for a 500-million-parameter model), placing us at 77 times below the Chinchilla-optimal ratio. If the power-law relationships hold in this extreme regime, the model should achieve far worse loss than a compute-optimally trained model of the same size, and the gap between language modeling competence and downstream task performance should widen dramatically.

In practice, we observe a more nuanced picture. The model achieves a Well-fit classification with validation loss 5.4956 and a relative gap of +0.42% between training and validation loss, suggesting that it has effectively learned the statistical distribution of its training corpus without significant overfitting. Yet factual accuracy on our benchmark is 0.52%, effectively zero. This disconnect suggests that the scaling laws governing loss may have different implications for different types of downstream capability: the model can learn to produce text that statistically resembles ALS research (low perplexity) without internalizing the factual content of that research (near-zero accuracy). Our work provides an empirical data point at the extreme low end of the data scaling curve, complementing the large-scale studies that established these relationships.

## 3. Methodology

Our methodology spans four stages: corpus construction, tokenizer training, model architecture design, and training. We describe each stage below with concrete numbers at every decision point, since reproducibility requires not just stating what we did but explaining why we made each choice. The complete pipeline is implemented in Python with all random seeds fixed and dependencies locked, enabling third-party replication on comparable hardware.

### 3.1 Data pipeline

Figure 1 shows the end-to-end data pipeline from source collection through tokenized training files.

![Data pipeline showing the flow from data collection through cleaning, deduplication, tokenization, training, export, evaluation, and RAG comparison](docs/figures/pipeline_diagram.png)

**Data sources.** We collected documents from three publicly available sources: PubMed Central open-access ALS research articles, ClinicalTrials.gov ALS clinical trial records, and publicly published patient and educational narratives. We deliberately restricted the corpus to a single disease domain to investigate what a model can learn from a narrow medical literature base. This yielded 19,164 total documents across the three source categories.

**Scraping.** We implemented source-specific scrapers that respect API rate limits: NCBI E-utilities allows 3 requests per second without an API key and 10 with a key, so we configured appropriate delays with exponential backoff retry logic for transient failures. All data comes from public, appropriately licensed sources. We excluded private medical records, HIPAA-protected data, and content from private support groups. Patient narratives are limited to content that individuals intentionally published for public audiences.

**Cleaning.** We applied an 11-step source-aware cleaning pipeline (implemented in `data/processing/clean.py`) that handles PubMed papers differently from patient narratives and clinical trial records. The steps, applied in order, are: (1) strip residual HTML/XML markup using BeautifulSoup, (2) strip non-content sections from scientific papers (references, acknowledgments, funding, tables, figure captions, author affiliations), (3) strip in-text citations (both numbered and author-year formats), (4) remove volatile content (URLs, emails, phone numbers, temporal qualifiers, copyright notices, calls-to-action, license text), (5) strip clinical trial status lines for trial documents, (6) re-scrub personally identifiable information from patient narratives using Presidio (a second pass after the initial scraping-stage scrub), (7) fix encoding errors with ftfy and normalize Unicode to NFC form (we chose NFC over NFKC because NFKC destructively normalizes Greek letters, superscripts, and math symbols that appear frequently in medical text), (8) normalize whitespace while preserving paragraph structure, (9) apply an English language safety net heuristic, (10) normalize medical abbreviations to canonical forms (ALS, SOD1, TDP-43, C9orf72, and 8 other domain-specific terms), and (11) embed the document title as a header. Documents shorter than 100 words after cleaning were rejected.

**Deduplication.** We implemented a two-level deduplication approach. At the document level, we used MinHash Locality-Sensitive Hashing (LSH) with 128 permutations and a Jaccard similarity threshold of 0.85, using word-level 5-gram shingles. We chose word 5-grams over character shingles because medical text shares high vocabulary overlap (many documents use the same technical terms), and word n-grams reduce false positive duplicate matches in this setting. At the paragraph level, we applied SHA-256 hashing on paragraphs of at least 50 words, removing exact-duplicate text blocks across the surviving corpus. Documents that lost all qualifying paragraphs were rejected. After the train/validation split, we ran a cross-set leakage check using MinHash LSH with a lower Jaccard threshold of 0.7 to catch near-duplicate documents that might have been placed in both sets.

**Source caps.** To prevent any single source type from dominating the corpus distribution, we enforced post-deduplication caps of 10% for patient narratives and 15% for Wikipedia-sourced content.

**Train/validation split.** We split the cleaned, deduplicated corpus into training and validation sets using a 90/10 ratio, stratified by source category, with a fixed random seed of 42. This stratification ensures that both sets contain proportional representation from PubMed Central, ClinicalTrials.gov, and educational sources.

**Final corpus size.** The tokenized corpus contains 142,939,320 total tokens: 128,495,047 training tokens and 14,444,273 validation tokens. At 516M model parameters, this yields 0.26 tokens per parameter, placing us 77 times below the Chinchilla-optimal ratio of approximately 20 tokens per parameter ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)).

### 3.2 Tokenizer

We trained a byte-level BPE tokenizer on the cleaned ALS corpus using the Hugging Face `tokenizers` library. We chose BPE over WordPiece or Unigram because BPE's greedy merge strategy tends to create whole-word tokens for frequent domain terms, which is exactly the behavior we wanted for medical vocabulary.

**Vocabulary size.** We set the vocabulary size to 50,257, matching GPT-2's vocabulary size. We chose this value for compatibility with the GPT-2 architecture and to avoid introducing an additional variable when comparing our model's behavior to GPT-2 baselines. The tokenizer uses a single special token (`<|endoftext|>`) for document boundaries.

**Pre-processing.** Input text is normalized to NFC Unicode form before tokenization (consistent with the cleaning pipeline), and we use a byte-level pre-tokenizer that splits on whitespace boundaries while preserving the ability to encode any Unicode character as a sequence of byte tokens.

**Medical term coverage.** We validated the tokenizer's domain coverage by checking the top 100 ALS-specific medical terms. Of these, 50 are tokenized as single tokens (e.g., "riluzole", "fasciculations", "edaravone"), meaning the model can process these terms without fragmentation. The remaining terms are split into 2-3 subword units, which is typical for longer compound terms. We compared our domain-trained tokenizer against GPT-2's general-purpose tokenizer (via tiktoken) and confirmed that the ALS tokenizer produces fewer fragments on domain-specific terminology, as documented in the tokenizer comparison report (`reports/comparison_report.md`).

**Output format.** The tokenizer encodes the train/validation split into nanoGPT-compatible uint16 binary files (`train.bin` and `val.bin`) with an accompanying `meta.pkl` metadata file containing the vocabulary size. This format enables memory-mapped data loading during training, allowing random batch sampling without loading the entire corpus into memory.

### 3.3 Model architecture

We use a GPT-2-style decoder-only transformer with Pre-LN (layer normalization before attention and MLP sublayers, not after). Figure 2 shows the architecture.

![Model architecture showing the GPT-2 Pre-LN transformer with input embeddings, 24 repeated transformer blocks, and output projection](docs/figures/model_architecture.png)

**Configuration.** The production model uses 24 transformer layers, 16 attention heads, an embedding dimension of 1,280, and an MLP inner dimension of 5,120 (4x expansion). The context length is 1,024 tokens. With a vocabulary size of 50,257, the total parameter count is approximately 516 million. We also maintain two smaller configurations for pipeline validation: a "tiny" model (~9M parameters: 6 layers, 6 heads, 192 embedding dimension) for rapid iteration, and a "medium" model (~111M parameters: 12 layers, 12 heads, 768 embedding dimension) matching GPT-2 Small dimensions.

**Pre-LN.** We chose Pre-LN over the original Post-LN transformer design because Pre-LN provides more stable gradients in deeper networks, tolerates higher learning rates without divergence, and converges faster in wall-time ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)). In the Pre-LN formulation, each transformer block applies layer normalization before the attention and MLP sublayers, then adds the sublayer output as a residual: `x = x + attn(ln(x))` and `x = x + mlp(ln(x))`. A final layer norm after the last transformer block manages the slightly growing hidden state variance that Pre-LN produces across layers.

**Weight tying.** The token embedding matrix and the language model head share the same weight tensor, following the standard GPT-2 convention ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)). This reduces the effective parameter count and ensures that input and output token representations live in the same vector space.

**Attention.** We use causal self-attention with learned positional embeddings (up to 1,024 positions). The implementation uses PyTorch's `F.scaled_dot_product_attention` with `is_causal=True`, which dispatches to FlashAttention on Ampere GPUs (the RTX 3060 is SM 8.6), providing O(N) memory usage and fused CUDA kernels for the attention computation.

**Initialization.** All Linear and Embedding weights are initialized from N(0, 0.02). Residual projection layers (`c_proj` in both attention and MLP) use a scaled initialization of N(0, 0.02 / sqrt(2 * n_layer)) to prevent the residual stream variance from growing with depth, following the GPT-2 initialization scheme.

### 3.4 Training

**Hardware.** All training runs on a single consumer-grade machine: an NVIDIA RTX 3060 with 12GB VRAM, 64GB system RAM, and an Intel i5-12400 processor, running under WSL2 on Windows. We chose to train on consumer hardware deliberately, both as a constraint that tests the feasibility of domain-specific model training outside of institutional compute clusters and as a reproducibility requirement (the hardware is widely available and affordable).

**Memory strategy.** A 516M-parameter model in fp32 requires approximately 2GB for weights alone. With optimizer states (Adam maintains two additional copies per parameter), gradients, and activations, the total memory requirement far exceeds the 12GB VRAM available on the RTX 3060. We address this using DeepSpeed ZeRO Stage 2 with CPU offloading ([Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)). ZeRO Stage 2 partitions optimizer states and gradients across data-parallel ranks (in our single-GPU case, this means offloading to CPU RAM), while keeping the model parameters on GPU for fast forward and backward passes. We additionally enable gradient checkpointing, which recomputes activations during the backward pass instead of storing them, reducing GPU memory usage by approximately 30-40% at the cost of roughly 20-30% slower training. Combined with fp16 mixed-precision training, these techniques allow the 516M model to train within 12GB VRAM.

**Hyperparameters.** We use the Adam optimizer with learning rate 3e-4, betas (0.9, 0.95), epsilon 1e-8, and weight decay 0.1. The learning rate follows a cosine annealing schedule with 500 warmup steps, decaying to zero over the full training run. We use a micro batch size of 4 with 8 gradient accumulation steps, yielding an effective batch size of 32 sequences (32,768 tokens per step at context length 1,024). Gradient clipping is applied at 1.0 to prevent training instability. Dropout is set to 0.1 on attention weights, residual connections, and embedding outputs.

Figure 3 shows the learning rate schedule over the full training run.

![Learning rate schedule showing cosine annealing with 500-step warmup over 11,760 total steps](docs/figures/lr_schedule.png)

**Training duration.** We train for 3 epochs over the 128.5M training tokens, which corresponds to 11,760 training steps. The complete training run took 4 hours and 32 minutes of wall-clock time. We chose 3 epochs based on monitoring the validation loss during training: the model maintains a Well-fit classification throughout, with the validation-training loss gap remaining below 1% across all checkpoints.

**Checkpointing.** We save DeepSpeed checkpoints every 1,000 steps with a retention policy of the last 3 regular checkpoints plus the best checkpoint by validation loss. The best checkpoint additionally includes a raw `.pt` state dict export for downstream conversion to Hugging Face and GGUF formats.

**Training results.** The model converges to a final training loss of 5.4727 and validation loss of 5.4956, representing a relative gap of +0.42%. This yields a Well-fit classification: the model has learned generalizable statistical patterns from the training corpus without significant overfitting. Training loss decreased from 11.1484 to 5.4727 over the full run, a 50.6% reduction, while validation loss tracked closely from 7.7284 to 5.4956. We observe mild validation loss divergence starting around step 11,000, suggesting that additional epochs would risk overfitting, but the magnitude is small enough that the 3-epoch training budget is appropriate for this corpus size.

**Reproducibility.** All random seeds are fixed (seed=42), dependency versions are locked in `requirements.txt`, and the training script supports dry-run mode for configuration validation before committing GPU time. The DeepSpeed configuration, model architecture, and all hyperparameters are logged as the first entry in a structured JSONL training log for full provenance.

## 4. Evaluation framework

### 4.1 Benchmark design

We designed a 6-stage evaluation pipeline to assess the model's factual accuracy, detect fabricated entities, classify failure modes, and enable structured comparison across model variants and retrieval configurations. Figure 4 shows the complete evaluation flow.

![Evaluation framework showing the 6-stage pipeline from response generation through scoring, fabrication detection, taxonomy classification, stratified sampling, and report generation](docs/figures/eval_framework.png)

The evaluation begins with a curated benchmark of 160 ALS-specific questions distributed across 8 knowledge categories, with 20 questions per category: Disease Mechanisms, Genetics, Clinical Features, Diagnosis, Treatment, Epidemiology, Research Methods, and Patient Care. We chose these categories to span the full breadth of ALS knowledge, from molecular biology (gene mutations, protein pathology) through clinical practice (diagnostic criteria, treatment options) to population-level data (incidence rates, risk factors).

Each question includes the question text, an expected answer, a list of independently verifiable key facts (for partial credit scoring), a category label, and a difficulty level. We manually curated all 160 questions rather than auto-generating them from the training corpus, because auto-generated questions risk testing the model's ability to memorize specific passages rather than its factual understanding. Manual curation allowed us to ensure clinical accuracy, control difficulty distribution, and include questions that require synthesis across multiple sources (e.g., "What is the relationship between TDP-43 pathology and C9orf72 repeat expansions?").

The design rationale is that binary correct/incorrect scoring obscures important distinctions between types of failure. A model that produces coherent but fabricated medical claims fails differently from one that produces repetitive gibberish, and understanding these failure modes is essential for characterizing what the model has actually learned. The 6-stage pipeline provides this granularity.

### 4.2 Scoring

The scoring stage evaluates each model response against the benchmark's key facts using fuzzy string matching via the `rapidfuzz` library. We chose fuzzy matching over exact string matching because models rarely reproduce expected answers verbatim, even when they contain the correct information expressed in different words or word order.

**Key fact extraction.** Each expected answer in the benchmark is decomposed into independently verifiable factual claims. For example, an expected answer about riluzole might contain three key facts: "riluzole is the first FDA-approved treatment for ALS," "it works by reducing glutamate excitotoxicity," and "it extends survival by approximately 2-3 months." This decomposition enables partial credit: a response that mentions the mechanism but not the survival benefit receives proportional credit rather than a binary pass or fail.

**Matching methodology.** For each response, we break the text into overlapping chunks (100 characters wide, with 50-character overlap). For each key fact, we compute the `partial_ratio` score from rapidfuzz against every chunk. A key fact is considered "found" if any chunk scores at or above the threshold of 80. Per-question accuracy is the proportion of key facts found (mean accuracy), and a question-level binary pass requires at least 50% of key facts to be matched.

**Coherence pre-filtering.** Before scoring, each response passes through a coherence pre-filter that flags degenerate output. A response is classified as incoherent (and excluded from substantive scoring) if it meets any of four conditions: it is empty or shorter than 10 characters, it contains a word repeated 6 or more times consecutively, it contains any 3-gram repeated 4 or more times (catching phrase-level repetition loops), or more than 80% of its characters are non-alphanumeric (token salad). This pre-filter is necessary because the from-scratch ALS-LM frequently produces degenerate output, and passing such output through the full scoring pipeline would waste computation and distort aggregate metrics.

**Aggregation.** We compute per-category accuracy (mean and median across the 20 questions in each category), overall accuracy (mean across all 160 questions), and binary pass rate (percentage of questions where at least 50% of key facts were matched). We also track hedging language frequency as a qualitative signal of model confidence calibration.

### 4.3 Fabrication detection

The fabrication detection stage identifies entities in model responses that do not appear in the training corpus. Unlike reference-based hallucination detection (which checks whether claims are supported by cited sources), our approach is entity-based: we maintain a registry of known entities extracted from the training data and flag any entity in a model response that is absent from this registry. We chose this approach because our from-scratch model has no concept of citations or references, making reference-based methods inapplicable. Entity fabrication, where the model generates plausible-sounding drug names, gene symbols, or institution names that do not exist, is the primary hallucination signal in this setting.

**Entity registry.** The registry contains approximately 48,000 known entities extracted from the training corpus across four categories: 6,469 drug names, 20,173 gene names, 8,075 protein names, and 13,354 institution names. The registry is built by scanning the cleaned corpus for entities matching known patterns and cross-referencing against established databases.

**Detection methodology.** We extract entity candidates from each model response using three approaches. NCT clinical trial identifiers are extracted via regex (NCT followed by 8 digits) and checked by exact match against registry entries. Drug name candidates are extracted by identifying capitalized words and words ending with known pharmaceutical suffixes (e.g., -mab, -nib, -zole, -pril), then fuzzy-matched against the drug registry using rapidfuzz with a threshold of 85. Gene name candidates are extracted via an uppercase-letter-and-digit pattern (2-10 characters, must contain at least one letter), then fuzzy-matched against the gene registry.

**Fabrication rate.** We report the fabrication rate as the proportion of responses containing at least one entity not found in the registry. We note that flagged entities may include false positives: real entities that happen to be absent from our training corpus. For this reason, we designed the fabrication detection as a screening tool that identifies candidates for manual review, not as an automated ground truth classifier.

### 4.4 Failure taxonomy

The taxonomy stage classifies each response into one of five failure modes (plus two non-failure categories) using rule-based logic that combines scoring results, fabrication flags, and text analysis. The taxonomy is designed to cover the full spectrum of model failure, from complete incoherence at one extreme to near-correct responses at the other.

**Failure modes.** We define five failure modes, listed in classification priority order (first match wins):

- **Confident fabrication:** The response contains fabricated entities (flagged by the fabrication detection stage) asserted without hedging language. This is the most dangerous failure mode because the model presents false information with apparent confidence.
- **Outdated information:** The response references temporal facts incorrectly, particularly for questions in time-sensitive categories such as clinical trials and treatment approvals.
- **Plausible blending:** The response mixes real facts with incorrect details, producing output that is partially correct but misleadingly wrong on specific claims. This mode is detected when the response has partial key fact matches combined with fabrication flags.
- **Boundary confusion:** The response provides information from a wrong domain or related-but-incorrect medical context, typically accompanied by hedging language that suggests the model is uncertain.
- **Accurate but misleading:** The response contains factually correct information but frames it without appropriate caveats, potentially leading to incorrect conclusions.

Two additional categories handle edge cases: **accurate** (correct response, not a failure) and **degenerate** (empty or incoherent output that fails the coherence pre-filter). Degenerate responses receive low severity because, while useless, they are obviously wrong and unlikely to mislead a reader.

**Classification logic.** The taxonomy classifier processes each response through the priority chain. If a response fails the coherence check, it is immediately classified as degenerate. Otherwise, the classifier examines fabrication results, scoring metrics, hedging language presence, and category-specific temporal indicators to assign the primary failure mode.

**Stratified sampling.** To enable manual review without examining all 160 responses, we implement proportional stratified sampling across categories and failure modes. This selects representative responses from each stratum (the worst-scoring, best-scoring, and closest-to-threshold responses per category) for qualitative analysis, ensuring that the manual review sample reflects the full distribution of model behavior.

**Design rationale.** The five failure modes were chosen to capture the qualitative differences we observed between the from-scratch model's failures and the baseline model's failures during development. The from-scratch ALS-LM predominantly produces degenerate output (repetitive loops, token salad) because it has not learned enough factual content to generate coherent medical claims. In contrast, the Llama 3.1 8B baseline predominantly produces confident fabrication because it has sufficient language modeling capability to generate fluent medical text but lacks the specific domain knowledge to make it accurate. This distinction, invisible to binary accuracy metrics, is exactly what the taxonomy is designed to capture.

## 5. Results

This section presents the quantitative findings from training and evaluating ALS-LM. We begin with training convergence analysis, proceed to hallucination evaluation across three quantization levels, and conclude with a brief assessment of quantization impact. All numbers in this section are transcribed verbatim from our automated analysis reports; no values have been rounded or re-calculated.

### 5.1 Training results

Figure 5 shows the training and validation loss curves over the full 3-epoch training run.

![Training and validation loss curves over 11,760 steps showing convergence with minimal overfitting](docs/figures/train_val_loss.png)

Training ran for 3 epochs (11,760 steps) over 4 hours and 32 minutes of wall-clock time. The final training loss was 5.4727 and the final validation loss was 5.4956, yielding a relative gap of +0.42%. Our automated overfitting analysis classifies this as Well-fit: the model has learned generalizable statistical patterns from the training corpus without significant memorization.

Training loss decreased from 11.1484 to 5.4727 over the full run, a 50.6% reduction. Validation loss tracked the training loss closely throughout, falling from 7.7284 to 5.4956. Every checkpoint across all three epochs received a Well-fit classification, with the relative gap between training and validation loss remaining below 1% at all 24 validation checkpoints.

Figure 6 shows the train and validation perplexity trajectories, illustrating the perplexity gap over training.

![Train and validation perplexity curves showing divergence over training](docs/figures/perplexity_gap.png)

Train perplexity decreased from 2275.25 to 238.09 while validation perplexity decreased from 2272.05 to 243.63. We observe mild validation loss divergence starting around step 11,000, where validation loss increased for two consecutive checkpoints while training loss continued to decrease. This is a classic early overfitting signal, though the magnitude is small: the final perplexity gap of 5.54 (243.63 minus 238.09) represents a 2.3% relative difference, confirming that 3 epochs is an appropriate training budget for this corpus size.

These results present a productive tension. The Well-fit classification and low relative gap (+0.42%) indicate that the model has successfully learned the statistical distribution of ALS research language. It can produce text whose token-level statistics closely match the training corpus. However, as Section 3.4 established, learning to model the distribution of medical text is not the same as internalizing the factual content of that text. To assess whether this language modeling competence translates to factual accuracy, we turn to our hallucination evaluation framework.

### 5.2 Hallucination evaluation

We evaluated the exported ALS-LM model across three GGUF quantization levels: full precision (F16), 8-bit integer (Q8_0), and 4-bit mixed (Q4_K_M). Each configuration was evaluated against the same 160-question ALS benchmark using identical generation parameters (temperature=0, max_tokens=512).

Table 3 summarizes the aggregate results across all three quantization levels.

**Table 3.** Hallucination evaluation results across three quantization levels. Mean accuracy uses the proportional key-fact fuzzy matching score (0-1). Binary pass rate counts questions where at least 50% of key facts were matched. Fabrication rate is the proportion of extracted entities not found in the training corpus registry.

| Model            | Mean Accuracy | Binary Pass | Fabrication Rate | Coherent Responses |
|------------------|---------------|-------------|------------------|--------------------|
| ALS-LM (F16)     |        0.0036 |      0.0%   |           0.6522 |   110/160 (68.8%)  |
| ALS-LM (Q8_0)    |        0.0021 |      0.0%   |           0.6641 |   108/160 (67.5%)  |
| ALS-LM (Q4_K_M)  |        0.0052 |      0.0%   |           0.6620 |   116/160 (72.5%)  |

The results are unambiguous: across all three quantization levels, the model achieves near-zero factual accuracy with a 0.0% binary pass rate. Not a single response out of 480 total (160 questions times 3 quantization levels) passed the 50% key fact threshold. Mean accuracy ranges from 0.0021 (Q8_0) to 0.0052 (Q4_K_M), representing the occasional accidental match of a single key fact fragment rather than genuine knowledge.

The coherent response rates reveal a significant proportion of degenerate output. Across the three quantization levels, 27.5% to 32.5% of responses were classified as degenerate by the coherence pre-filter (empty, repetitive loops, or token salad). The remaining responses, while passing the coherence threshold, consisted primarily of grammatically plausible but factually empty text: phrases like "we investigated the role of the disease progression" and "the most common genetic mutations in the disease" repeated with minor variations.

Figure 7 shows the failure taxonomy distribution across the evaluated model.

![Failure taxonomy distribution showing the proportions of confident fabrication, plausible blending, outdated information, and degenerate responses](docs/figures/failure_taxonomy.png)

The failure mode distribution for ALS-LM (using the Q4_K_M results as representative, since all three levels produce equivalent patterns) reveals three dominant categories: confident fabrication at 54 responses (33.8%), degenerate output at 44 responses (27.5%), and plausible blending at 43 responses (26.9%). Outdated information accounts for 19 responses (11.9%). Boundary confusion and accurate but misleading each account for 0 responses, and no responses were classified as accurate.

Two patterns deserve emphasis. First, the model never hedges. Across 160 responses, the Q4_K_M evaluation detected zero hedging instances (the F16 evaluation also detected zero, while Q8_0 detected exactly one instance of "likely"). The model does not produce uncertainty markers because it has not learned to distinguish what it knows from what it does not. Second, the fabrication rate is remarkably consistent across quantization levels (0.6522 to 0.6641), indicating that approximately 65% of all entities extracted from model responses do not appear in the training corpus registry. The model generates gene-like strings (e.g., "RNA-43-43-"), disease abbreviations from related but incorrect domains (e.g., frequent references to "AD" for Alzheimer's disease and "tau pathology" in response to ALS-specific questions), and repetitive protein binding constructs that do not correspond to real molecular biology.

This result confirms the data deficit hypothesis. The model has learned ALS-adjacent language patterns, enough to produce text that superficially resembles research writing, but has not internalized the factual relationships between entities, mechanisms, and clinical findings. The gap between the Well-fit training classification and the 0.0% binary pass rate is itself the central empirical finding of this work.

### 5.3 Quantization impact

Quantization has no meaningful impact on ALS-LM's evaluation results. Table 3 shows that mean accuracy varies between 0.0021 and 0.0052 across the three quantization levels, with all three achieving identical 0.0% binary pass rates and fabrication rates within 1.2 percentage points of each other (0.6522 to 0.6641).

This null result is expected. Quantization degrades model performance by introducing rounding errors in weight representations, but these errors are only detectable when the model has a measurable signal to degrade. At near-zero accuracy, the base signal is too weak for quantization artifacts to manifest. The practical implication is that Q4_K_M (which requires approximately 4x less storage than F16) can be used for all downstream evaluation and inference with no loss of information, since there is no information to lose.

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

The following references are cited in this paper. Each entry includes a hyperlink to the primary publication or preprint.

- [Beltagy et al., 2019](https://aclanthology.org/D19-1371.pdf) — "SciBERT: A Pretrained Language Model for Scientific Text." EMNLP 2019.
- [Bolton et al., 2024](https://arxiv.org/html/2403.18421v1) — "BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text." arXiv 2024.
- [Chen et al., 2025](https://arxiv.org/abs/2502.14302) — "MedHallu: A Comprehensive Benchmark for Detecting LLM Hallucinations in Medical Contexts." arXiv 2025.
- [Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556) — "Training Compute-Optimal Large Language Models." NeurIPS 2022.
- [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) — "Scaling Laws for Neural Language Models." arXiv 2020.
- [Lee et al., 2020](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) — "BioBERT: A Pre-trained Biomedical Language Representation Model for Biomedical Text Mining." Bioinformatics, 36(4), 2020.
- [Lin et al., 2022](https://arxiv.org/abs/2109.07958) — "TruthfulQA: Measuring How Models Mimic Human Falsehoods." ACL 2022.
- [Luo et al., 2022](https://arxiv.org/abs/2210.10341) — "BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining." Briefings in Bioinformatics, 2022.
- [Manakul et al., 2023](https://arxiv.org/abs/2303.08896) — "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models." EMNLP 2023.
- [Min et al., 2023](https://arxiv.org/abs/2305.14251) — "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation." EMNLP 2023.
- [Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — "Language Models are Unsupervised Multitask Learners." OpenAI Technical Report, 2019.
- [Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054) — "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC 2020.
- [Tsatsaronis et al., 2015](https://www.nature.com/articles/s41597-023-02068-4) — "An Overview of the BioASQ Large-Scale Biomedical Semantic Indexing and Question Answering Competition." BMC Bioinformatics, 2015.
- [Umapathi et al., 2023](https://arxiv.org/abs/2307.15343) — "Med-HALT: Medical Domain Hallucination Test for Large Language Models." CoNLL 2023.
- [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) — "Attention Is All You Need." NeurIPS 2017.
- [Xiong et al., 2020](https://arxiv.org/abs/2002.04745) — "On Layer Normalization in the Transformer Architecture." ICML 2020.
- [Xiong et al., 2024](https://arxiv.org/abs/2402.13178) — "Benchmarking Large Language Models in Retrieval-Augmented Generation." ACL Findings 2024.
- [Yang et al., 2022](https://arxiv.org/abs/2203.03540) — "GatorTron: A Large Language Model for Electronic Health Records." Nature NPJ Digital Medicine, 2022.

