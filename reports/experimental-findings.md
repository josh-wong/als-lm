# Experimental findings for ALS-LM-2

This document synthesizes the experimental results from ALS-LM-1 (v1.0.0 through v1.5.0) into actionable findings for ALS-LM-2 planning. It references existing reports for detailed data and focuses on conclusions that should shape v2 architecture, training, and evaluation decisions.

## 6-model comparison summary

Six model variants were evaluated on the same 160-question ALS hallucination benchmark at Q8_0 quantization. The models span three approach families: from-scratch training, fine-tuning, and pre-trained instruction tuning.

| Model                          | Approach                | Accuracy | Coherence | Capability gap | Fabrication rate |
|--------------------------------|-------------------------|----------|-----------|----------------|------------------|
| ALS-LM 500M (from-scratch)    | From-scratch            |    0.21% |     67.5% |          67.3% |            66.4% |
| ALS-LM 1B (from-scratch base) | From-scratch            |    0.00% |     35.0% |          35.0% |           100.0% |
| GPT-2 large (fine-tuned)       | Fine-tuned              |    3.12% |      2.5% |          -0.6% |            77.0% |
| ALS-LM 1B (instruction-tuned) | Fine-tuned              |    0.00% |      0.0% |           0.0% |            0.0%* |
| Qwen 2.5 1.5B Instruct (base) | Pre-trained instruct    |   10.31% |     29.4% |          19.1% |            87.6% |
| Qwen 2.5 1.5B QLoRA           | Pre-trained instruct    |    7.24% |     50.0% |          42.8% |            81.0% |

\*The 1B instruction-tuned model's 0.0% fabrication rate reflects that all 160 outputs were degenerate, so no entities could be checked.

The original plan targeted Llama 3.2 1B Instruct, but gated access was not available, so Qwen 2.5 1.5B Instruct was substituted. Some result directories and JSON keys still use the `llama32_*` names.

Full data: [6-model comparison report](qlora_comparison_report.md) and [qlora_comparison_report.json](qlora_comparison_report.json)

## Key findings

### 1. Knowledge source is the primary factor

The unmodified Qwen 2.5 1.5B Instruct model achieved the highest accuracy (10.31%) without any ALS-specific training. Its pre-trained parametric knowledge from large-scale general and biomedical corpora outperformed every from-scratch model regardless of size or post-training. This is consistent with pre-trained knowledge mattering more than model size or fine-tuning approach for domain-specific tasks with limited training data, though a single model per family does not establish it.

### 2. Domain fine-tuning with limited data is a double-edged sword

QLoRA domain adaptation on 970 ALS instruction pairs shifted the model's behavior toward ALS-specific patterns, increasing coherence from 29.4% to 50.0%. However, it decreased factual accuracy from 10.31% to 7.24% (a -3.07% delta). The model became more willing to engage with ALS topics but partially overwrote general biomedical knowledge in the process. This accuracy-coherence tradeoff suggests that 970 instruction pairs may be below the threshold needed for domain adaptation to improve factual performance.

### 3. SFT did not surface knowledge in a from-scratch model

Instruction tuning the from-scratch 1B model produced 160/160 degenerate responses (0% accuracy, 0% coherence). The model collapsed to repeating "TheTheThe..." for every input. SFT is designed to format existing knowledge into structured responses, and this result is consistent with the base model having no extractable knowledge for it to format, though we did not test this directly. This rejected Hypothesis 3 from the v2 white paper. Full analysis: [SFT failure analysis](sft_failure_analysis.md)

### 4. From-scratch training on about 143M tokens did not yield factual knowledge

Both from-scratch models (500M and 1B) achieved near-zero accuracy despite training on the full ALS corpus. They learned statistical patterns in ALS text (producing coherent-sounding domain language) but did not internalize extractable medical facts. Scaling from 500M to 1B parameters did not help, which is consistent with data being the bottleneck rather than model capacity. We did not vary data volume, so this remains a hypothesis.

### 5. Capability gap is the primary safety concern

The 500M from-scratch model has the highest capability gap at 67.3% — it produces coherent-sounding ALS text that is almost never factually accurate. The QLoRA model has a 42.8% gap. Any model deployed for domain-specific Q&A must be evaluated for this gap, as coherent but wrong medical information is more dangerous than obviously broken output.

## Implications for v2 planning

### Training data

The 970 instruction pairs used for SFT and QLoRA appear insufficient, though dataset size was not varied. V2 should target an order of magnitude more domain-specific training data (5,000-50,000 instruction pairs) to test whether more data lets domain adaptation improve rather than degrade factual accuracy. The data collection pipeline from v1 (PubMed Central, ClinicalTrials.gov, educational sources) is proven and can be extended.

### Base model selection

Pre-trained instruction-capable models appear to be the more viable starting point for domain-specific Q&A with limited data, based on these results. V2 should build on a pre-trained base rather than training from scratch, and should select a model with strong existing biomedical knowledge.

### Evaluation framework

The 160-question hallucination benchmark, 6-stage evaluation pipeline, and capability gap metric are validated and reusable for v2. The benchmark should be expanded to cover more ALS knowledge areas and include harder questions that better discriminate between models in the 10-30% accuracy range.

### Quantization stability

Cross-quantization analysis of the QLoRA model showed no meaningful accuracy degradation across F16, Q8_0, and Q4_K_M levels. V2 can safely use Q8_0 or Q4_K_M for development and evaluation without worrying about quantization artifacts.

## Hypothesis status from v2 white paper

| Hypothesis | Statement                                                                                                      | Status   | Evidence                            |
|------------|----------------------------------------------------------------------------------------------------------------|----------|-------------------------------------|
| 1          | Higher-quality, deduplicated training data improves model accuracy                                             | Partial  | Corpus expanded; effect not isolated |
| 2          | Scaling from 500M to 1B+ parameters improves domain knowledge                                                  | Not supported | 1B base: 0.00% vs 500M: 0.21% |
| 3          | Instruction tuning surfaces internalized ALS knowledge in Q&A format                                           | Not supported | SFT: 160/160 degenerate; QLoRA: -3 pts |

## Source reports

These reports contain the detailed data behind the findings above.

- [6-model comparison report](qlora_comparison_report.md) — Cross-model evaluation with accuracy, coherence, capability gap, and failure taxonomy across all 6 variants
- [6-model comparison data](qlora_comparison_report.json) — Machine-readable metrics for all models
- [SFT failure analysis](sft_failure_analysis.md) — Root cause analysis of instruction tuning collapse on the from-scratch 1B model
- [4-model comparison report](model_comparison_report.md) — Earlier comparison of from-scratch and GPT-2 fine-tuned models
- [Ablation baseline report](eval/alslm-1b-base/) — Evaluation of unmodified Qwen 2.5 1.5B Instruct as pre-fine-tuning baseline

---

*This document synthesizes research findings from ALS-LM-1 for planning purposes. The models described here are research artifacts and should never be used for medical decision-making.*
