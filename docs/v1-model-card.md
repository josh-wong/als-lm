# ALS-LM model card

> [!CAUTION]
>
> This project evaluated six model variants across three approach families. None is suitable for medical use. The best-performing variant (unmodified Qwen 2.5 1.5B Instruct) achieves only 10.31% accuracy on the ALS benchmark. The QLoRA domain-adapted variant achieves 7.24%. The from-scratch 500M model achieves 0.21%, the from-scratch 1B base achieves 0.00%, the instruction-tuned 1B SFT produces 100% degenerate output, and the fine-tuned GPT-2 large achieves 3.12% with 97.5% degenerate output. No model should ever be used for clinical decision-making, patient education, or any application where factual accuracy matters.
>
> For reliable ALS information, please consult the [ALS Association](https://www.als.org/), [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/amyotrophic-lateral-sclerosis/symptoms-causes/syc-20354022), or [NIH NINDS](https://www.ninds.nih.gov/health-information/disorders/amyotrophic-lateral-sclerosis-als).

## Model details

ALS-LM is a 516M-parameter decoder-only transformer trained from scratch on 143M tokens of curated amyotrophic lateral sclerosis (ALS) research. We built the model to investigate what a purpose-built language model can learn from a narrow medical corpus, how it fails, and how those failure modes compare to retrieval-augmented generation (RAG) approaches. The near-zero factual accuracy we observed is a central research finding, not a shortcoming—it demonstrates the data deficit threshold below which domain-specific models cannot internalize factual knowledge. We subsequently extended the investigation to six model variants spanning from-scratch training, pretrained fine-tuning, and pretrained domain adaptation, with full results documented in the [research paper](v2-research-paper.md).

We use a GPT-2-style architecture with Pre-LN (layer normalization before attention) for training stability. As a controlled comparison experiment, we also fine-tuned GPT-2 large (774M parameters) on the same ALS corpus; see the [Model variants](#model-variants) section below for details.

| Parameter           | Value               |
|---------------------|---------------------|
| Parameters          | 516M                |
| Layers              | 24                  |
| Attention heads     | 16                  |
| Embedding dimension | 1,280               |
| Context length      | 1,024 tokens        |
| Vocabulary          | 50,257 (custom BPE) |
| Normalization       | Pre-LN (LayerNorm)  |

We trained the model with PyTorch and DeepSpeed ZeRO Stage 2 with CPU offloading in FP16 mixed precision on a single NVIDIA RTX 3060 (12GB VRAM) with 64GB system RAM.

**Model sources:**

- **Repository:** [als-lm on GitHub](https://github.com/josh-wong/als-lm)
- **Research paper:** [Full methodology and analysis](v2-research-paper.md)
- **Interactive demo:** [CLI chat interface](../demo/cli.py)

## Model variants

As a controlled comparison experiment, we fine-tuned OpenAI's GPT-2 large (774M parameters) on the same ALS corpus to test whether pretrained general knowledge could overcome the data deficit hypothesis observed with the from-scratch model. See [Section 7 of the research paper](v2-research-paper.md) for the full methodology and analysis.

### ALS-LM 1B base (1.02B)

The 1B variant scales the from-scratch architecture to test whether increased model capacity improves factual accuracy.

| Parameter           | Value                                           |
|---------------------|-------------------------------------------------|
| Parameters          | 1.02B                                           |
| Layers              | 24                                              |
| Attention heads     | 32                                              |
| Embedding dimension | 2,048                                           |
| Context length      | 1,024 tokens                                    |
| Vocabulary          | 50,257 (custom BPE)                             |
| Normalization       | Pre-LN (LayerNorm)                              |
| Training data       | 142M tokens (ALS corpus)                        |
| Training time       | ~21 hours, 3 epochs                             |
| Memory strategy     | DeepSpeed ZeRO Stage 2/3, CPU offloading        |

The 1B model achieves 0.00% accuracy on the ALS benchmark compared to 0.21% for the 500M model, demonstrating that scaling parameters without additional data reduces rather than improves accuracy. With 65.0% degenerate output and 100.0% fabrication rate across non-degenerate responses, the model confirms that model capacity is not the binding constraint.

### Fine-tuned GPT-2 large (774M)

The fine-tuned variant uses OpenAI's pretrained GPT-2 large weights as a starting point.

| Parameter           | Value                                        |
|---------------------|----------------------------------------------|
| Base model          | GPT-2 large (OpenAI, pretrained on WebText)  |
| Parameters          | 774M                                         |
| Layers              | 36                                           |
| Attention heads     | 20                                           |
| Embedding dimension | 1,280                                        |
| Context length      | 1,024 tokens                                 |
| Normalization       | Post-LN (LayerNorm)                          |
| Fine-tuning data    | 146M tokens (ALS corpus, GPT-2 tokenizer)    |
| Training time       | ~16 hours, 2 epochs                          |
| Optimizer           | AdamW (lr=2e-5, cosine decay)                |
| Memory strategy     | DeepSpeed ZeRO Stage 2, CPU offloading       |

See the [6-model comparison table](#evaluation-results) for the GPT-2 large evaluation results alongside all other variants.

### Domain adaptation experiments

Two additional models were evaluated to test instruction tuning and pretrained domain adaptation. These used external base models rather than the ALS-LM architecture.

**1B SFT (instruction-tuned).** The from-scratch 1B base model was instruction-tuned with supervised fine-tuning (SFT) on 970 Alpaca-format ALS question-answer pairs using completion-only loss masking. The result was catastrophic: 160/160 responses were degenerate (100%), producing no evaluable factual content. This demonstrates that instruction tuning cannot surface knowledge that was never internalized during pretraining. See [Section 8 of the research paper](v2-research-paper.md) for full methodology and root cause analysis.

**Qwen 2.5 1.5B Instruct (unmodified baseline).** The unmodified Qwen 2.5 1.5B Instruct model was evaluated without any ALS-specific training to establish the pretrained knowledge baseline. It achieves 10.31% accuracy — the highest of any model evaluated — with 29.4% coherent non-degenerate output and 87.6% fabrication rate. This confirms that a model's pretrained parametric knowledge determines its ALS performance floor.

**Qwen 2.5 1.5B QLoRA (domain-adapted).** Qwen 2.5 1.5B Instruct was fine-tuned with QLoRA (4-bit NF4 quantization, LoRA rank 16, alpha 32) on the same 970 ALS instruction pairs. Accuracy degraded from 10.31% to 7.24% while coherence increased from 29.4% to 50.0%, widening the perceived capability gap from 19.1% to 42.8%. The model sounds more knowledgeable while being less accurate. See [Section 9 of the research paper](v2-research-paper.md) and the [6-model comparison report](../reports/qlora_comparison_report.md) for the full QLoRA ablation analysis.

## Intended use

We designed ALS-LM for two audiences, neither of which involves end users seeking medical information.

- **Research use.** The model and its evaluation framework serve as a case study in domain-specific training failure modes. Researchers can use the project to study data scaling thresholds for factual knowledge acquisition, the disconnect between language-modeling loss and downstream factual accuracy, hallucination evaluation methodology (key-fact matching, entity-based fabrication detection, failure taxonomy), and from-scratch training versus RAG as competing approaches to domain specialization.
- **Educational use.** ML practitioners and students can use the project as a teaching resource for understanding data requirements for domain-specific models (we trained on 80x fewer tokens than Chinchilla-optimal), training pipelines on consumer hardware with DeepSpeed, the gap between overfitting metrics and factual competence, and end-to-end reproducible ML workflows (scraping through evaluation).

The data collection, processing, training, and evaluation pipeline we built is domain-agnostic. Practitioners can adapt it for other specialized corpora beyond ALS research.

### Out-of-scope use

We explicitly prohibit the following uses of ALS-LM:

- Clinical diagnosis or treatment recommendations
- Patient education or counseling
- Drug or treatment information lookup
- Any application requiring factual accuracy about ALS or any other medical topic
- Generating text presented as medical information to any audience

Neither model variant—the from-scratch model at 0.0% and the fine-tuned model at 1.87% binary pass rate—achieves accuracy sufficient for any information-retrieval purpose.

## Training data

We trained the model on 19,164 documents (143M tokens) from three publicly available sources.

| Source              | Type                    | License/access                |
|---------------------|-------------------------|-------------------------------|
| PubMed Central      | Research papers         | Open Access subset            |
| ClinicalTrials.gov  | Clinical trial records  | Public domain (US government) |
| Educational sources | Patient/medical content | Public web content            |

All data comes from publicly available sources. We did not use private medical records, content from private support groups, or any data subject to HIPAA or equivalent protections. For the full data collection and processing methodology, including our 11-step cleaning pipeline and deduplication approach, see [Section 3.1 of the research paper](v1-research-paper.md#31-data-pipeline).

## Training procedure

We trained for 3 epochs (11,760 steps) over 4 hours and 27 minutes.

| Hyperparameter       | Value                                        |
|----------------------|----------------------------------------------|
| Learning rate        | 3e-4                                         |
| LR schedule          | Cosine decay with 500-step linear warmup     |
| Effective batch size | 32 (4 micro-batch x 8 gradient accumulation) |
| Weight decay         | 0.1                                          |
| Optimizer            | AdamW (beta1=0.9, beta2=0.95)                |
| Precision            | FP16 mixed precision                         |
| Memory strategy      | DeepSpeed ZeRO Stage 2, CPU offloading       |

Training converged to a final validation loss of 5.4956 with a loss relative gap of +0.42%, which we classify as Well-fit. Despite this healthy training dynamic, the model achieves near-zero factual accuracy—a central finding of the research that we analyze in detail in the [research paper](v1-research-paper.md).

## Evaluation results

The following table compares all six model variants at the Q8_0 quantization level.

**Table 1.** 6-model comparison on the 160-question ALS benchmark (Q8_0). All approaches fall short of the RAG baseline accuracy (14.3%).

| Model                                       | Approach             | Accuracy | Degenerate rate | Fabrication rate |
|---------------------------------------------|----------------------|:--------:|:---------------:|:----------------:|
| ALS-LM 500M (from-scratch)                  | From-scratch         |    0.21% |           32.5% |            66.4% |
| ALS-LM 1B base (from-scratch)               | From-scratch         |    0.00% |           65.0% |           100.0% |
| GPT-2 large 774M (fine-tuned)               | Pretrained fine-tune |    3.12% |           97.5% |            77.0% |
| ALS-LM 1B SFT (instruction-tuned)           | From-scratch SFT     |    0.00% |          100.0% |             0.0% |
| Qwen 2.5 1.5B Instruct (unmodified)         | Pretrained instruct  |   10.31% |           70.6% |            87.6% |
| Qwen 2.5 1.5B QLoRA (domain-adapted)        | Pretrained QLoRA     |    7.24% |           50.0% |            81.0% |

The unmodified Qwen 2.5 1.5B Instruct model achieves the highest accuracy (10.31%) without any ALS-specific training, confirming that pretrained parametric knowledge determines the performance floor. QLoRA domain adaptation on 970 ALS instruction pairs reduced accuracy by 3.07 percentage points while increasing coherence, widening the perceived capability gap. The 1B SFT model's complete output collapse (100% degenerate) demonstrates that instruction tuning cannot create knowledge absent from the base model. See the [research paper](v2-research-paper.md) for the full analysis.

### From-scratch quantization comparison

We evaluated all three GGUF quantization levels of the from-scratch 500M model against the 160-question ALS factual benchmark by using key-fact fuzzy matching, entity-based fabrication detection (~48K entities), and a 5-mode failure taxonomy.

| Model            | Mean accuracy | Binary pass | Fabrication rate | Coherent responses |
|------------------|---------------|-------------|------------------|--------------------|
| ALS-LM (F16)    |        0.0036 |       0.0%  |           65.2%  |  110/160 (68.8%)   |
| ALS-LM (Q8_0)   |        0.0021 |       0.0%  |           66.4%  |  108/160 (67.5%)   |
| ALS-LM (Q4_K_M) |        0.0052 |       0.0%  |           66.2%  |  116/160 (72.5%)   |

All three quantization levels achieve 0.0% binary pass rate. The dominant failure modes are confident fabrication (33.1%), degenerate output (32.5%), and plausible blending (23.8%). Quantization level has no meaningful effect on evaluation quality, suggesting the accuracy ceiling is determined by training data volume rather than inference precision.

We also conducted a RAG comparison experiment by using four configurations (two embedding models at two chunk sizes) with ChromaDB, benchmarked against a no-retrieval Llama 3.1 8B baseline. The best RAG configuration (500-token chunks with PubMedBERT embeddings) achieved 13.8% mean accuracy but did not exceed the no-retrieval baseline at 14.3%, revealing retrieval quality as the primary bottleneck rather than generation capability. For the full RAG methodology and failure decomposition analysis, see [Section 6 of the research paper](v2-research-paper.md).

## Bias, risks, and limitations

### Medical safety

ALS-LM has demonstrated a near-complete inability to produce factually accurate medical content. Across 480 evaluations (160 questions x 3 quantization levels), the model achieves 0.0% binary pass rate, fabricates medical entities at a 66.4% rate, and produces degenerate output (repetitive or incoherent text) 32.5% of the time. Anyone who encounters this model should understand it as a research artifact demonstrating failure modes, not a functional information source.

The fine-tuned GPT-2 large model exhibits a distinct failure profile. While the from-scratch model fails primarily through fabrication—inventing plausible but false medical content with high confidence—the fine-tuned model fails through degeneration, with 97.5% of responses producing repetitive or incoherent output. Both failure profiles make the models unsuitable for any medical use, but for different reasons: the from-scratch model is dangerously confident in wrong answers, while the fine-tuned model mostly fails to produce coherent responses at all.

### Technical limitations

We trained on only 143M tokens, which is 80x below the Chinchilla-optimal ratio of ~20 tokens per parameter for a 516M model. This severe data deficit is the primary explanation for the near-zero factual accuracy despite healthy training loss convergence. The fine-tuning experiment confirmed the data deficit as the primary accuracy bottleneck, with pretrained knowledge yielding a 15x improvement, while revealing instruction-following as a separate limitation not addressed by domain-specific training alone. Additional limitations include single-domain training (ALS literature only, with no general English pretraining), a single model size (516M parameters, with no scaling experiments), and a 1,024-token context window that constrains the complexity of questions the model can address.

### Recommendations

We recommend that anyone encountering ALS-LM treat it exclusively as a research artifact. The evaluation results quantify the model's unreliability; they do not suggest any scenario in which the model produces trustworthy output. For actual ALS information, consult qualified healthcare providers and trusted resources.

- [ALS Association](https://www.als.org/)
- [Mayo Clinic – ALS overview](https://www.mayoclinic.org/diseases-conditions/amyotrophic-lateral-sclerosis/symptoms-causes/syc-20354022)
- [NIH National Institute of Neurological Disorders and Stroke](https://www.ninds.nih.gov/health-information/disorders/amyotrophic-lateral-sclerosis-als)

## Ethical considerations

- **Data sourcing.** All training data comes from publicly available, appropriately licensed sources. We did not use HIPAA-protected data, private medical records, or content from private support groups or communities where participants had a reasonable expectation of privacy.
- **Patient narrative policy.** We only included patient perspectives from content that individuals intentionally published for public audiences. The model's outputs are machine-generated text and should never be interpreted as patient experiences or testimonials.
- **Research transparency.** We document negative results transparently as research findings. The 0.0% factual accuracy, 66% fabrication rate, and data deficit analysis are presented as the project's primary contributions, not as shortcomings to minimize. We believe honest reporting of negative results advances the field more than selective reporting of positive metrics.

## Citation

If you use ALS-LM in your research, please cite the repository:

```
ALS-LM: A domain-specific language model for ALS knowledge
https://github.com/josh-wong/als-lm
```

For detailed methodology, evaluation results, and analysis, see the [research paper](v1-research-paper.md).
