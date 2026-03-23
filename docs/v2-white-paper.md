# ALS-LM-2: Investigating instruction tuning and parameter scaling for domain-specific medical knowledge

**Author:** [josh-wong](https://github.com/josh-wong)
**Date:** March 10, 2026
**Last revised:** March 11, 2026
**Status:** Approved

---

## Abstract

ALS-LM-1 trained a 516M-parameter decoder-only transformer from scratch on 143M tokens of curated amyotrophic lateral sclerosis (ALS) research and evaluated a fine-tuned GPT-2 large (774M parameters) on the same corpus. The from-scratch model achieved 0.21% mean factual accuracy despite a Well-fit training classification, confirming that language-modeling competence does not imply knowledge acquisition at 80x below the Chinchilla-optimal data ratio. The fine-tuned model improved accuracy 15-fold to 3.12% but produced degenerate output for 97.5% of evaluation questions, revealing instruction-following as a limitation orthogonal to factual knowledge. A retrieval-augmented generation (RAG) comparison by using Llama 3.1 8B established a 14.3% no-retrieval baseline that no RAG configuration exceeded, identifying retrieval quality as the primary bottleneck.

ALS-LM-2 investigates three hypotheses derived from these findings: (1) expanding and improving the training corpus addresses the data deficit that limited knowledge acquisition, (2) scaling from 516M to 1B parameters provides additional capacity for encoding factual relationships, and (3) supervised fine-tuning on instruction-response pairs resolves the degenerate output problem that prevented the fine-tuned model from expressing its knowledge. This white paper defines tiered success criteria anchored to ALS-LM-1 baselines, with minimum targets of exceeding 3.12% accuracy and 50% response coherence, and stretch targets of exceeding the RAG baseline. The investigation aims to determine whether these three interventions, applied together, can produce a domain-specific model whose factual accuracy approaches or exceeds retrieval-augmented alternatives.

## 1. Motivation

ALS-LM-1 produced three findings that, taken together, define the research agenda for version 2. First, the from-scratch 516M model demonstrated that a purpose-built transformer can learn the statistical distribution of a medical corpus well enough to achieve healthy training metrics (validation loss 5.4956, relative gap +0.42%) while retaining essentially zero factual knowledge (0.21% mean accuracy on a 160-question domain-specific benchmark). This disconnect between language-modeling loss and factual accuracy established the data deficit as the dominant bottleneck: at 0.25 tokens per parameter, the model operated 80 times below the ratio at which scaling laws predict meaningful knowledge acquisition.

Second, the fine-tuned GPT-2 large experiment isolated the role of pretrained knowledge. By holding the training corpus constant and introducing a model with pretrained weights from broad web text, the experiment showed that general knowledge provides a measurable accuracy improvement (0.21% to 3.12%, a 15-fold gain) but does not resolve the underlying data limitation. More notably, the fine-tuned model produced degenerate output for 97.5% of evaluation questions, compared to 67.5% coherent responses from the from-scratch model. GPT-2 is a completion-based architecture without instruction-following alignment, and this architectural mismatch manifested as repetitive or incoherent output when the model was evaluated against structured questions. The data deficit and the instruction-following limitation are orthogonal dimensions of failure.

Third, the RAG comparison revealed that retrieval-augmented generation does not automatically solve the knowledge problem. The best RAG configuration (PubMedBERT embeddings with 500-token chunks) achieved 13.8% mean accuracy, falling short of the 14.3% no-retrieval baseline. Retrieval failures accounted for 52-89% of wrong answers across configurations, pointing to retrieval quality rather than generation capability as the bottleneck. Domain-specific embeddings (PubMedBERT) outperformed general-purpose embeddings (MiniLM) by 2.1x, but even this improvement was insufficient to surpass what the base model already knew.

These findings raise a focused question: if the from-scratch model's failures stem primarily from insufficient data and the fine-tuned model's failures stem from an architectural inability to follow instructions, can a model that addresses both limitations simultaneously produce meaningfully better results? ALS-LM-2 investigates this question through three coordinated hypotheses targeting data quality, model capacity, and instruction-following capability.

## 2. Background

This section reviews the research landscape relevant to ALS-LM-2, drawing on prior work surveyed in the [ALS-LM-1 research paper](v1-research-paper.md) and extending it to the instruction tuning paradigm.

### 2.1 Domain-specific language models in medicine

The dominant approach in biomedical natural language processing has been to adapt general-purpose pretrained models to the biomedical domain through continued pretraining on domain-specific corpora. PubMedBERT demonstrated that training from scratch on PubMed abstracts outperforms continued pretraining of general-domain BERT on biomedical NER and relation extraction tasks. BioGPT, a 347M-parameter GPT-2-style decoder trained on 15 million PubMed abstracts, achieved state-of-the-art results on PubMedQA and biomedical relation extraction. At larger scales, BioMedLM (2.7B parameters) showed competitive performance with much larger general-domain models on medical question answering, and GatorTron (8.9B parameters, trained on over 90 billion words of clinical and biomedical text) established the best results on clinical natural language inference benchmarks.

These models share a common characteristic: large training corpora spanning broad biomedical domains, ranging from billions to tens of billions of tokens. ALS-LM-1 deliberately investigated the opposite extreme, training on 143M tokens restricted to a single disease domain, and confirmed that this data volume is insufficient for factual knowledge acquisition at the 516M parameter scale.

A parallel trend in medical NLP is the application of instruction tuning to biomedical models. Supervised fine-tuning on instruction-response pairs has shown substantial gains in general-domain models' ability to follow structured prompts and produce coherent answers, and recent work has begun applying this approach to medical models. ALS-LM-2 builds on this trend by investigating whether instruction tuning can resolve the degenerate output pattern observed in the ALS-LM-1 fine-tuned model.

### 2.2 The hallucination problem in medical contexts

Hallucinations in language models are well-documented across domains, but their severity is particularly concerning in medicine, where confident fabrication of drug names, gene associations, or treatment protocols creates risk beyond simple inaccuracy. ALS-LM-1 contributed a 5-mode failure taxonomy for categorizing medical hallucinations: confident fabrication (stating false information with no hedging), plausible blending (combining real facts into false composites), outdated information (generating claims that were once accurate but are no longer current), boundary confusion (producing content outside the target domain), and accurate but misleading (stating technically correct information without critical context).

This taxonomy proved essential for understanding the qualitative differences between model architectures. The from-scratch model's failures distributed across confident fabrication (33.1%), degenerate output (32.5%), and plausible blending (23.8%), while the fine-tuned model's failures concentrated almost entirely in degenerate output (97.5%). ALS-LM-2 aims to extend this taxonomy if instruction-tuned models produce failure modes not captured by the existing five categories.

### 2.3 Why ALS?

The selection of ALS as the target domain, established in the [ALS-LM-1 white paper](v1-white-paper.md), was motivated by several factors. ALS research is well-represented in open-access literature across PubMed Central, ClinicalTrials.gov, and major health organizations, providing sufficient material for corpus construction. The disease represents a well-defined body of research with clear domain boundaries, making it possible to construct a corpus that is both comprehensive and bounded. ALS knowledge includes structured factual relationships (gene-mutation-phenotype associations, drug-mechanism-trial outcome chains) that lend themselves to benchmark evaluation. Active research frontiers in ALS treatment create opportunities to evaluate temporal accuracy. The author's personal connection to the disease, having a father who suffered from ALS, provided firsthand insight into the impact of the disease and motivated the choice of domain.

These reasons remain applicable to ALS-LM-2. The same corpus sources are available for expansion, the domain boundaries remain well-defined, and the 160-question evaluation benchmark provides a validated instrument for measuring improvement.

## 3. ALS-LM-1 findings

This section consolidates the key quantitative results and failure patterns from [ALS-LM-1](v1-research-paper.md) that motivate the ALS-LM-2 research agenda. All numbers are reported at the Q8_0 quantization level, established as the representative level in the ALS-LM-1 research paper.

The following table summarizes the aggregate performance of the three approaches evaluated in ALS-LM-1.

| Approach                              | Mean accuracy | Binary pass | Coherent responses  | Fabrication rate |
|---------------------------------------|---------------|-------------|---------------------|------------------|
| ALS-LM 516M (from-scratch)            |         0.21% |        0.0% | 108/160 (67.5%)     |            66.4% |
| GPT-2 large 774M (fine-tuned on ALS)  |         3.12% |       1.87% | 4/160 (2.5%)        |            77.0% |
| Llama 3.1 8B (no-retrieval baseline)  |        14.3%  |      13.8%  | N/A                 |            87.2% |
| Best RAG (500-PubMedBERT)             |        13.8%  |      10.6%  | N/A                 |            80.3% |

Three findings from these results are central to the ALS-LM-2 approach:

- **Data deficit.** The from-scratch model trained on 143M tokens at 0.25 tokens per parameter, placing it 80 times below the Chinchilla-optimal ratio of approximately 20 tokens per parameter. Despite achieving a Well-fit training classification with validation loss 5.4956 and a relative gap of just +0.42%, the model attained 0.21% mean accuracy with a 0.0% binary pass rate on the 160-question ALS benchmark. Training completed in 4 hours and 27 minutes over 3 epochs. The model learned the statistical distribution of ALS research language well enough to produce superficially plausible text but did not internalize the factual relationships between entities, mechanisms, and clinical findings. The gap between language-modeling competence and factual knowledge acquisition is itself a key empirical finding.
- **Loss-accuracy gap.** The fine-tuned GPT-2 large demonstrated that pretrained knowledge partially compensates for the data deficit. With 774M parameters and general knowledge from web-text pretraining, fine-tuning on the same ALS corpus for 2 epochs (approximately 16 hours, validation loss 2.37) yielded a 15-fold accuracy improvement from 0.21% to 3.12%. However, this still leaves the model at 97% below useful accuracy thresholds. The pretrained knowledge provides a measurable advantage, but the magnitude confirms that training data volume, not the absence of general knowledge, is the dominant bottleneck.
- **Instruction-following limitation.** The most striking result from the fine-tuned model comparison was the degenerate output dominance. The fine-tuned GPT-2 large produced degenerate (repetitive or incoherent) output for 97.5% of evaluation questions (156 out of 160), compared to 32.5% for the from-scratch model (52 out of 160). GPT-2 is a completion-based architecture trained on next-token prediction without instruction-following alignment. When evaluated against the structured Q&A format of the benchmark, it generated text that continued from the prompt rather than answering it. The from-scratch model exhibited diverse failure modes across confident fabrication (33.1%), degenerate output (32.5%), and plausible blending (23.8%). The fine-tuned model concentrated almost all failures in the degenerate category, with only 2 instances of confident fabrication and 2 of plausible blending. This result demonstrates that data deficit and instruction-following capability are orthogonal dimensions of model failure.

## 4. Approach

ALS-LM-2 proposes three hypotheses, each addressing a specific ALS-LM-1 failure mode. The hypotheses are presented as equal pillars of the investigation; no single hypothesis is expected to be sufficient on its own.

### 4.1 Hypothesis 1: Data quality

The ALS-LM-1 from-scratch model trained on a corpus that placed it 80 times below the Chinchilla-optimal data ratio. At 0.25 tokens per parameter, the model had insufficient exposure to internalize the factual relationships present in ALS research. The data deficit hypothesis proposes that expanding the training corpus and improving data quality moves the model closer to the Chinchilla-optimal ratio, increasing the likelihood that the model acquires factual knowledge rather than merely learning statistical patterns of medical language.

Corpus expansion involves broadening the range of sources beyond the three categories used in ALS-LM-1 (PubMed Central, ClinicalTrials.gov, and educational content). Clinical practice guidelines, systematic reviews from organizations such as the World Health Organization, and additional peer-reviewed literature offer structured factual content with high knowledge density. Data quality improvements address artifacts identified during ALS-LM-1 development, including punctuation and whitespace inconsistencies from PDF extraction that reduce the effective information density of training tokens.

The expected outcome is a higher tokens-per-parameter ratio and cleaner training signal, providing the model with more opportunities to learn factual associations from each training token.

### 4.2 Hypothesis 2: Parameter scaling

The ALS-LM-1 from-scratch model used 516M parameters, and even the fine-tuned GPT-2 large at 774M parameters achieved only 3.12% accuracy. While the data deficit is the dominant bottleneck, model capacity may also play a role: a larger model has more parameters available to encode the factual associations present in the training data. The parameter scaling hypothesis proposes that increasing model size to approximately 1B parameters provides additional capacity for knowledge retention.

Scaling to 1B parameters on the same consumer hardware (RTX 3060, 12GB VRAM) is feasible through DeepSpeed ZeRO Stage 2 with CPU offloading, the same memory management strategy used in ALS-LM-1. The ALS-LM-1 from-scratch model used 6.37 GB peak VRAM during training, leaving headroom for a larger model. The combination of more data (Hypothesis 1) and more parameters creates a regime where the tokens-per-parameter ratio and model capacity are both improved relative to ALS-LM-1.

The expected outcome is a model with greater capacity to encode factual relationships, particularly when combined with the improved data quality from Hypothesis 1.

### 4.3 Hypothesis 3: Instruction tuning

The ALS-LM-1 fine-tuned GPT-2 large produced degenerate output for 97.5% of evaluation questions because it is a completion-based architecture without instruction-following alignment. The instruction tuning hypothesis proposes that supervised fine-tuning (SFT) on instruction-response pairs teaches the model to respond to structured questions in a Q&A format, enabling it to surface whatever factual knowledge it acquires from improved data and larger capacity.

Instruction tuning addresses a limitation that is orthogonal to data quality and model capacity. A model that has internalized factual knowledge but cannot follow instructions will fail to demonstrate that knowledge on a structured benchmark. Conversely, a model with strong instruction-following capability but no factual knowledge will produce coherent but empty responses. Both dimensions must be addressed for the model to demonstrate meaningful accuracy improvement.

The expected outcome is a model that produces coherent, structured responses to questions rather than degenerate repetitive output, enabling the evaluation framework to measure the model's actual factual knowledge.

The implementation uses quantized low-rank adaptation (QLoRA) on Llama 3.2 1B Instruct rather than full SFT on the from-scratch model. This approach leverages the instruct model's existing language coherence and instruction-following capabilities while adapting it to the ALS domain through the curated instruction dataset. A natural concern is that the base model's pretrained ALS knowledge could contaminate results, attributing improvements to pretraining rather than the ALS corpus. An ablation baseline evaluation of the unmodified Llama 3.2 1B Instruct model on the full 160-question benchmark addresses this concern: the model achieved only 10.3% mean accuracy with 70.6% degenerate outputs, demonstrating that its pretrained ALS knowledge is negligible. This validates a clean division of labor where the base model contributes instruction-following capability and the corpus contributes domain knowledge.

### 4.4 Hypothesis interactions

The three hypotheses are interdependent in ways that affect experimental design and interpretation. Instruction tuning (Hypothesis 3) depends on the other two hypotheses for its effectiveness: if the base model has not acquired sufficient factual knowledge through improved data quality and larger capacity, instruction tuning will produce a model that responds coherently but inaccurately. Conversely, improved data quality and parameter scaling (Hypotheses 1 and 2) cannot be fully evaluated without instruction tuning, because a completion-based model may produce degenerate output that masks whatever knowledge it has acquired.

This interdependence means the hypotheses are best tested in combination rather than in isolation. The ALS-LM-1 experiments already provide partial controls: the from-scratch model tests limited data with moderate capacity and no instruction tuning; the fine-tuned model tests limited data with larger capacity and no instruction tuning. ALS-LM-2 adds improved data, 1B-scale capacity, and instruction tuning simultaneously, then evaluates the combined effect against ALS-LM-1 baselines.

The risk of testing all three hypotheses together is that attribution becomes difficult: if accuracy improves, the individual contribution of each hypothesis cannot be cleanly separated. This is an accepted tradeoff. The primary research question is whether the combination produces a meaningful improvement, not the precise contribution of each factor. The ALS-LM-1 baselines provide sufficient reference points for qualitative analysis of which failure modes are resolved.

The following table summarizes the mapping from each hypothesis to the ALS-LM-1 failure mode it addresses.

| Hypothesis           | ALS-LM-1 failure mode addressed                        | Proposed response                                                          |
|----------------------|--------------------------------------------------------|----------------------------------------------------------------------------|
| Data quality         | 80x below Chinchilla-optimal data ratio                | Expand corpus and improve cleaning to increase tokens-per-parameter ratio  |
| Parameter scaling    | 516M model may lack capacity for knowledge retention   | Scale to 1B parameters on same hardware via DeepSpeed                      |
| Instruction tuning   | 97.5% degenerate output from completion-based model    | SFT on instruction-response pairs for Q&A capability                       |

## 5. Success criteria

Success criteria are defined in tiers anchored to ALS-LM-1 baselines. The minimum tier represents the threshold below which the investigation would be considered unsuccessful. The target tier represents the outcome that would validate the combined hypothesis. The stretch tier represents an aspirational outcome that would suggest the approach merits further scaling.

### 5.1 Accuracy

Accuracy is measured by using the same 160-question ALS benchmark and proportional key-fact fuzzy matching score used in ALS-LM-1.

| Tier    | Criterion                                                    | Rationale                                                                                                 |
|---------|--------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Minimum | Exceed fine-tuned GPT-2 large accuracy (3.12%)               | Demonstrates that the combined approach outperforms pretrained fine-tuning alone                          |
| Target  | Approach RAG no-retrieval baseline accuracy (14.3%)          | Demonstrates that a trained model can compete with parametric knowledge of a general 8B model             |
| Stretch | Exceed RAG no-retrieval baseline accuracy (14.3%)            | Demonstrates that domain-specific training with instruction tuning surpasses general pretrained knowledge |

### 5.2 Coherence

Coherence is measured as the percentage of non-degenerate responses on the evaluation benchmark. This metric directly addresses the 97.5% degenerate output rate observed in the ALS-LM-1 fine-tuned model.

| Tier    | Criterion                     | Rationale                                                                                 |
|---------|-------------------------------|-------------------------------------------------------------------------------------------|
| Minimum | >50% coherent responses       | Demonstrates that instruction tuning substantially resolves the degenerate output problem |
| Target  | >80% coherent responses       | Approaches the from-scratch model's 67.5% coherence while maintaining higher accuracy     |
| Stretch | >90% coherent responses       | Demonstrates that instruction tuning effectively eliminates degenerate output             |

### 5.3 RAG re-comparison

The ALS-LM-1 RAG comparison established that no RAG configuration exceeded the no-retrieval baseline. Re-running the same comparison against the instruction-tuned model provides a controlled measure of improvement.

| Tier    | Criterion                                                           | Rationale                                                                           |
|---------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Minimum | Instruction-tuned model evaluated against same RAG baseline         | Ensures comparability with ALS-LM-1 results                                         |
| Target  | Model approaches or exceeds best RAG config (500-PubMedBERT, 13.8%) | Demonstrates that training-based knowledge can match retrieval-augmented approaches |

### 5.4 Evaluation framework

The ALS-LM-1 evaluation framework must be adapted for instruction-tuned model output. Success criteria for the framework itself ensure that comparison across model versions remains valid.

- The 160-question benchmark is adapted for instruction-formatted prompts while preserving the same questions and key facts for cross-version comparability.
- Cross-model comparison works end-to-end: from-scratch, fine-tuned, instruction-tuned, and RAG results can be presented in a single comparison table.
- The 5-mode failure taxonomy is extended if instruction-tuned models produce failure modes not captured by the existing categories.

### 5.5 3B model feasibility

If the 1B training run demonstrates sufficient hardware headroom, a 3B model may be feasible on the same consumer hardware. This is a conditional objective, not a primary goal. If 1B training uses less than 10 GB peak VRAM and completes within 24 hours, 3B training may be feasible with ZeRO Stage 3 and aggressive CPU offloading. The ALS-LM-1 from-scratch model (516M) used 6.37 GB peak VRAM and trained in 4 hours and 27 minutes, providing a reference point for extrapolation.

## 6. Failure taxonomy

ALS-LM-1 contributed a 5-mode failure taxonomy for categorizing how domain-specific language models fail in medical contexts. The taxonomy distinguishes failures by both kind and severity.

- **Confident fabrication.** The model states false information with no hedging or uncertainty markers, such as inventing a drug name or claiming a fictional clinical trial showed positive results. This was the most common failure mode in the from-scratch model at 33.1% of responses. Severity: high.
- **Plausible blending.** The model combines real facts into a false composite, such as correctly naming a real drug but attributing the wrong mechanism of action, or associating a real gene with the wrong disease variant. This accounted for 23.8% of from-scratch model responses. Severity: high, because the output contains enough real information to appear credible.
- **Outdated information.** The model generates information that was accurate at some point but is no longer current, such as describing a drug as being in active trials when the trial has since concluded or failed. This accounted for 10.6% of from-scratch model responses. Severity: moderate.
- **Boundary confusion.** The model generates content outside the ALS domain by drawing on loosely related patterns, such as producing content about Huntington's disease genetics when asked about ALS genetics. Severity: moderate.
- **Accurate but misleading.** The model produces technically correct statements that lack critical context, such as correctly stating that a drug reduced motor neuron loss in mice without noting that it failed in human trials. Severity: moderate to high, depending on context.

ALS-LM-2 aims to apply this taxonomy consistently to the instruction-tuned model. If instruction tuning produces failure modes not captured by these five categories, the taxonomy will be extended. One anticipated possibility is a new category of failures where the model produces coherent, well-structured responses that are factually empty, answering the question format correctly but providing no substantive information. Whether this constitutes a distinct failure mode or a variant of confident fabrication is a question the evaluation will need to address.

## 7. Ethical considerations

The ethical framework established in the [ALS-LM-1 white paper](v1-white-paper.md) carries forward in full to ALS-LM-2.

### 7.1 Medical information risk

ALS-LM-2 does not produce a medical tool. All project documentation, demo interfaces, and published outputs carry clear disclaimers that the model is a research artifact, not a source of medical advice. The hallucination evaluation framework exists to quantify the model's unreliability, not to demonstrate its usefulness as an information source. This disclaimer applies regardless of the model's accuracy on the evaluation benchmark.

### 7.2 Patient data and privacy

The project uses no private medical data. Patient narratives included in the training corpus are limited to content that individuals have intentionally published for public audiences. The project does not scrape private forums, support groups, or any content protected by medical privacy regulations. All data sources are documented with licensing and ethical justification.

### 7.3 Responsible publication

If model weights are published, they are accompanied by the full disclaimer framework and a model card documenting known failure modes and limitations. The project does not publish the model in any format that encourages its use as a medical information system.

### 7.4 Instruction tuning and perceived capability

Instruction tuning introduces a risk not present in ALS-LM-1. A model that produces coherent, well-structured responses to medical questions may appear more capable and trustworthy than a model that produces degenerate output, even if its factual accuracy remains low. The ALS-LM-1 fine-tuned model's 97.5% degenerate output rate, while a failure of instruction-following, served as a natural barrier to misuse: users quickly recognized the output as unreliable. An instruction-tuned model that responds coherently but inaccurately may not trigger the same skepticism.

This risk requires additional mitigation beyond the ALS-LM-1 disclaimer framework. The ALS-LM-2 evaluation must explicitly measure the gap between perceived and actual capability, and the model card must document this gap prominently. Any demo interface must display accuracy metrics alongside model output, ensuring that coherent responses are not mistaken for accurate ones.

## 8. Expected contributions

ALS-LM-2 aims to contribute to the understanding of domain-specific language models in four ways:

- **Instruction tuning investigation on a domain-specific corpus.** Most published work on instruction tuning operates on general-domain or broad biomedical models at scales exceeding independent replication. This investigation applies instruction tuning to a narrowly scoped medical corpus at a reproducible scale, documenting what instruction tuning can and cannot accomplish when both the corpus and the model are small by current standards.
- **Extended evaluation framework.** The ALS-LM-1 hallucination evaluation framework (160-question benchmark, 5-mode failure taxonomy, entity-based fabrication detection) is adapted for instruction-formatted outputs. This adaptation, including any taxonomy extensions required by new failure modes, contributes a reusable methodology for evaluating instruction-tuned domain-specific models.
- **Data quality impact analysis.** By improving the training corpus between ALS-LM-1 and ALS-LM-2, the investigation provides empirical evidence on the relationship between corpus quality and factual accuracy in the data-starved regime. The ALS-LM-1 baseline (0.21% accuracy at 0.25 tokens per parameter) establishes a controlled reference point for measuring the impact of corpus expansion and cleaning.
- **Cross-approach comparison.** ALS-LM-2 enables comparison across three training approaches applied to the same domain: from-scratch pretraining (ALS-LM-1, 516M), pretrained fine-tuning (ALS-LM-1, 774M GPT-2 large), and instruction-tuned training (ALS-LM-2, 1B). Combined with the RAG comparison, this provides a four-way analysis of how different architectural approaches handle domain-specific medical knowledge, with emphasis on failure modes and severity rather than accuracy alone.

## 9. Limitations

This section documents the known limitations of the ALS-LM-2 investigation.

### 9.1 Research limitations

Several constraints bound the scope and generalizability of the investigation:

- **Hardware constraints.** All training runs on consumer-grade hardware (NVIDIA RTX 3060, 12GB VRAM, 64GB RAM, Intel i5-12400). This restricts model size, training duration, and batch size relative to what would be possible with datacenter-grade resources. Training dynamics at this scale (CPU offloading latency, smaller effective batch sizes) may differ from larger-scale training.
- **Corpus size.** Even with corpus expansion, the training data remains modest by current standards. The model's knowledge will have gaps within the ALS domain, and the tokens-per-parameter ratio, while improved, may still fall below what is needed for reliable factual knowledge acquisition.
- **Single-domain focus.** ALS was chosen for practical reasons, but findings may not transfer to medical domains with different characteristics, such as higher ambiguity, broader scope, or less structured knowledge.
- **Attribution difficulty.** Testing all three hypotheses simultaneously means that if accuracy improves, the individual contribution of data quality, parameter scaling, and instruction tuning cannot be cleanly separated. The investigation prioritizes measuring the combined effect over isolating individual contributions.

### 9.2 Scope boundaries

The ALS-LM-2 investigation is scoped to supervised fine-tuning (SFT) on instruction-response pairs. Reinforcement learning from human feedback (RLHF), direct preference optimization (DPO), and other alignment techniques are out of scope. The evaluation interface remains a command-line demo; no web interface or API deployment is planned. All training data and evaluation are in English only.

If the 1B training run demonstrates sufficient hardware headroom (less than 10 GB peak VRAM, completion within 24 hours), a 3B model configuration may be explored as a secondary objective by using ZeRO Stage 3 with aggressive CPU offloading. This is a conditional extension, not a planned deliverable.

## 10. Conclusion

ALS-LM-2 proposes a focused investigation into whether three coordinated interventions can address the failure modes identified in ALS-LM-1. The data quality hypothesis targets the 80x Chinchilla deficit that limited knowledge acquisition. The parameter scaling hypothesis provides additional model capacity for encoding factual relationships. The instruction tuning hypothesis resolves the 97.5% degenerate output rate that prevented the fine-tuned model from expressing whatever knowledge it had acquired.

The investigation defines success in terms of measurable improvement over ALS-LM-1 baselines, with tiered criteria that range from exceeding the fine-tuned model's 3.12% accuracy (minimum) to exceeding the RAG no-retrieval baseline of 14.3% (stretch). The emphasis on honest evaluation carries forward from ALS-LM-1: the primary contribution is not a performant medical model but a documented investigation into what domain-specific training, instruction tuning, and parameter scaling can and cannot accomplish at consumer-accessible scales.

Whether the model achieves its target accuracy or falls short, the results will extend the empirical record on domain-specific language model training. The hallucination evaluation framework, with its failure taxonomy and cross-approach comparison methodology, provides a reusable contribution regardless of the model's absolute performance. The investigation prioritizes transparency and reproducibility, documenting what works, what fails, and why.
