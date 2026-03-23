# QLoRA comparison report: 6-model cross-comparison

Cross-comparison of 6 model variants evaluated on the 160-question ALS hallucination benchmark using Q8_0 quantization as the representative level. Models are grouped by approach family: from-scratch training, pre-trained fine-tuning, and pre-trained instruction tuning.

| Model                                    | Family                 |   Accuracy |  Fab. rate |   Coherence |   Cap. gap | Delta vs base |
| ---------------------------------------- | ---------------------- | ---------- | ---------- | ----------- | ---------- | ------------- |
| 500M                                     | From-scratch           |      0.21% |      66.4% |       67.5% |      67.3% |            -- |
| 1B base                                  | From-scratch           |      0.00% |     100.0% |       35.0% |      35.0% |            -- |
| GPT-2 large                              | Pre-trained fine-tune  |      3.12% |      77.0% |        2.5% |      -0.6% |            -- |
| 1B SFT                                   | Pre-trained fine-tune  |      0.00% |       0.0% |        0.0% |       0.0% |            -- |
| Llama base                               | Pre-trained instruct   |     10.31% |      87.6% |       29.4% |      19.1% |            -- |
| Llama QLoRA                              | Pre-trained instruct   |      7.24% |      81.0% |       50.0% |      42.8% |         -3.1% |

## Perceived capability gap

The perceived capability gap measures the difference between how coherent a model's output appears (coherence%) and how factually accurate it is (accuracy%). A high gap indicates ethical risk: users may trust plausible-sounding but incorrect answers.

| Model                                    |    Coherence |   Accuracy |      Gap |
| ---------------------------------------- | ------------ | ---------- | -------- |
| 500M                                     |        67.5% |      0.21% |    67.3% |
| 1B base                                  |        35.0% |      0.00% |    35.0% |
| GPT-2 large                              |         2.5% |      3.12% |    -0.6% |
| 1B SFT                                   |         0.0% |      0.00% |     0.0% |
| Llama base                               |        29.4% |     10.31% |    19.1% |
| Llama QLoRA                              |        50.0% |      7.24% |    42.8% |

## QLoRA ablation delta

Delta between the Llama 3.2 QLoRA model and the unmodified Llama 3.2 1B Instruct baseline. Positive accuracy delta and negative fabrication delta indicate improvement from domain adaptation.

| Metric                         |        Value |
| ------------------------------ | ------------ |
| Accuracy delta                 |       -3.07% |
| Fabrication rate delta         |       -6.64% |
| Coherence delta                |       +20.6% |

## Failure taxonomy distribution

Distribution of failure modes across the 7 taxonomy categories for all 6 models.

| Failure mode                 |     500M (n) |     500M (%) |  1B base (n) |  1B base (%) | GPT-2 large (n) | GPT-2 large (%) |   1B SFT (n) |   1B SFT (%) | Llama base (n) | Llama base (%) | Llama QLoRA (n) | Llama QLoRA (%) |
| ---------------------------- | ------------ | ------------ | ------------ | ------------ | --------------- | --------------- | ------------ | ------------ | -------------- | -------------- | --------------- | --------------- |
| Confident Fabrication        |           53 |        33.1% |           19 |        11.9% |               2 |            1.2% |            0 |         0.0% |             28 |          17.5% |              51 |           31.9% |
| Plausible Blending           |           38 |        23.8% |           29 |        18.1% |               2 |            1.2% |            0 |         0.0% |             17 |          10.6% |              22 |           13.8% |
| Outdated Information         |           17 |        10.6% |            8 |         5.0% |               0 |            0.0% |            0 |         0.0% |              1 |           0.6% |               3 |            1.9% |
| Boundary Confusion           |            0 |         0.0% |            0 |         0.0% |               0 |            0.0% |            0 |         0.0% |              0 |           0.0% |               1 |            0.6% |
| Accurate But Misleading      |            0 |         0.0% |            0 |         0.0% |               0 |            0.0% |            0 |         0.0% |              0 |           0.0% |               0 |            0.0% |
| Accurate                     |            0 |         0.0% |            0 |         0.0% |               0 |            0.0% |            0 |         0.0% |              1 |           0.6% |               3 |            1.9% |
| Degenerate                   |           52 |        32.5% |          104 |        65.0% |             156 |           97.5% |          160 |       100.0% |            113 |          70.6% |              80 |           50.0% |

## Implications

The 6-model comparison reveals a consistent narrative about knowledge sources in domain-specific language models. From-scratch models (500M and 1B) internalize minimal factual knowledge from the ALS corpus despite training on 143M tokens. Pre-trained models (GPT-2 large and the 1B SFT variant) show that pre-existing knowledge from large-scale pretraining provides a measurable but limited advantage.

The Llama 3.2 comparison pair (base ablation vs QLoRA) provides the clearest test of domain adaptation: starting from an instruction-capable model with strong general knowledge, QLoRA fine-tuning on ALS-specific data can shift the model's behavior toward the target domain. The ablation delta quantifies this shift directly.

The SFT failure (1B instruction-tuned model producing 100% degenerate output) remains the strongest evidence that instruction tuning cannot create knowledge that was never internalized during pretraining. For detailed analysis, see reports/sft_failure_analysis.md.

## Caveats and limitations

**Single quantization level.** This comparison uses Q8_0 as the representative quantization level, based on cross-quantization analysis showing that quantization does not meaningfully affect evaluation results.

**General knowledge confound.** Pre-trained models (GPT-2 large, Llama 3.2) carry general biomedical knowledge from their pretraining corpora. When these models answer correctly, we cannot fully separate domain-specific fine-tuning effects from retained general knowledge.

**Benchmark scope.** The 160-question ALS hallucination benchmark covers 8 categories but cannot exhaustively test all aspects of ALS knowledge. Results reflect performance on this specific benchmark.
