# QLoRA comparison report: 6-model cross-comparison

Cross-comparison of 6 model variants evaluated on the 160-question ALS hallucination benchmark using Q8_0 quantization as the representative level. Models are grouped by approach family: from-scratch training, fine-tuning, and pre-trained instruction tuning.

| Model                                    | Family                 |   Accuracy |  Fab. rate |   Coherence |   Cap. gap | Delta vs base |
| ---------------------------------------- | ---------------------- | ---------- | ---------- | ----------- | ---------- | ------------- |
| 500M                                     | From-scratch           |      0.21% |      66.4% |       67.5% |      67.3% |            -- |
| 1B base                                  | From-scratch           |      0.00% |     100.0% |       35.0% |      35.0% |            -- |
| GPT-2 large                              | Fine-tuned             |      3.12% |      77.0% |        2.5% |      -0.6% |            -- |
| 1B SFT                                   | Fine-tuned             |      0.00% |       0.0% |        0.0% |       0.0% |            -- |
| Qwen base                                | Pre-trained instruct   |     10.31% |      87.6% |       29.4% |      19.1% |            -- |
| Qwen QLoRA                               | Pre-trained instruct   |      7.24% |      81.0% |       50.0% |      42.8% |         -3.1% |

**Notes.** The 1B SFT model was trained from scratch and then instruction-tuned. The Fine-tuned family groups it with GPT-2 large (which starts from pretrained weights) by technique, not origin. Its 0.0% fabrication rate reflects that all 160 outputs were degenerate, so no entities were extracted. Qwen base and Qwen QLoRA are the unmodified Qwen 2.5 1.5B Instruct model and its QLoRA-adapted variant; the original plan targeted Llama 3.2 1B Instruct, which was not accessible.

## Perceived capability gap

The perceived capability gap measures the difference between how coherent a model's output appears (coherence%) and how factually accurate it is (accuracy%). A high gap indicates ethical risk: users may trust plausible-sounding but incorrect answers.

| Model                                    |    Coherence |   Accuracy |      Gap |
| ---------------------------------------- | ------------ | ---------- | -------- |
| 500M                                     |        67.5% |      0.21% |    67.3% |
| 1B base                                  |        35.0% |      0.00% |    35.0% |
| GPT-2 large                              |         2.5% |      3.12% |    -0.6% |
| 1B SFT                                   |         0.0% |      0.00% |     0.0% |
| Qwen base                                |        29.4% |     10.31% |    19.1% |
| Qwen QLoRA                               |        50.0% |      7.24% |    42.8% |

## QLoRA ablation delta

Delta between the Qwen 2.5 1.5B QLoRA model and the unmodified Qwen 2.5 1.5B Instruct baseline. Positive accuracy delta and negative fabrication delta indicate improvement from domain adaptation.

| Metric                         |        Value |
| ------------------------------ | ------------ |
| Accuracy delta                 |       -3.07% |
| Fabrication rate delta         |       -6.64% |
| Coherence delta                |       +20.6% |

## Failure taxonomy distribution

Distribution of failure modes across the 7 taxonomy categories for all 6 models.

| Failure mode                 |     500M (n) |     500M (%) |  1B base (n) |  1B base (%) | GPT-2 large (n) | GPT-2 large (%) |   1B SFT (n) |   1B SFT (%) |  Qwen base (n) |  Qwen base (%) |  Qwen QLoRA (n) |  Qwen QLoRA (%) |
| ---------------------------- | ------------ | ------------ | ------------ | ------------ | --------------- | --------------- | ------------ | ------------ | -------------- | -------------- | --------------- | --------------- |
| Confident Fabrication        |           53 |        33.1% |           19 |        11.9% |               2 |            1.2% |            0 |         0.0% |             28 |          17.5% |              51 |           31.9% |
| Plausible Blending           |           38 |        23.8% |           29 |        18.1% |               2 |            1.2% |            0 |         0.0% |             17 |          10.6% |              22 |           13.8% |
| Outdated Information         |           17 |        10.6% |            8 |         5.0% |               0 |            0.0% |            0 |         0.0% |              1 |           0.6% |               3 |            1.9% |
| Boundary Confusion           |            0 |         0.0% |            0 |         0.0% |               0 |            0.0% |            0 |         0.0% |              0 |           0.0% |               1 |            0.6% |
| Accurate But Misleading      |            0 |         0.0% |            0 |         0.0% |               0 |            0.0% |            0 |         0.0% |              0 |           0.0% |               0 |            0.0% |
| Accurate                     |            0 |         0.0% |            0 |         0.0% |               0 |            0.0% |            0 |         0.0% |              1 |           0.6% |               3 |            1.9% |
| Degenerate                   |           52 |        32.5% |          104 |        65.0% |             156 |           97.5% |          160 |       100.0% |            113 |          70.6% |              80 |           50.0% |

## Implications

The 6-model comparison reveals a consistent narrative about knowledge sources in domain-specific language models. From-scratch models (500M and 1B) internalize minimal factual knowledge from the ALS corpus despite training on roughly 143-153M tokens. GPT-2 large, which starts from pretrained weights, shows that pre-existing knowledge from large-scale pretraining provides a measurable but limited advantage. The 1B SFT variant is different: it was trained from scratch and then instruction-tuned, not initialized from a pretrained model.

The Qwen 2.5 1.5B comparison pair (unmodified vs QLoRA) provides the clearest test of domain adaptation: starting from an instruction-capable model with strong general knowledge, QLoRA fine-tuning on 970 ALS instruction pairs raised the share of non-degenerate responses but lowered mean key-fact accuracy. The ablation delta quantifies this shift directly. The pair was added after the instruction-tuned 1B checkpoint produced degenerate output on all 160 questions; the original plan targeted Llama 3.2 1B Instruct, which was not accessible, so Qwen 2.5 1.5B Instruct was used instead.

The SFT failure (1B instruction-tuned model producing 100% degenerate output) is consistent with instruction tuning being unable to surface knowledge that the from-scratch base model did not internalize. The model's 0.0% fabrication rate reflects that no entities could be extracted from degenerate output, not that it avoided fabrication. For detailed analysis, see reports/sft_failure_analysis.md.

## Caveats and limitations

**Single quantization level.** This comparison uses Q8_0 as the representative quantization level, based on cross-quantization analysis showing that quantization does not meaningfully affect evaluation results.

**General knowledge confound.** Pre-trained models (GPT-2 large, Qwen 2.5 1.5B) carry general biomedical knowledge from their pretraining corpora. When these models answer correctly, we cannot fully separate domain-specific fine-tuning effects from retained general knowledge.

**Benchmark scope.** The 160-question ALS hallucination benchmark covers 8 categories but cannot exhaustively test all aspects of ALS knowledge. Results reflect performance on this specific benchmark.
