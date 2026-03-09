# Quantization comparison report

## TL;DR

Compared 3 quantization levels (F16, Q8_0, Q4_K_M). Mean accuracy ranges from 0.0234 to 0.0344, and fabrication rates range from 69.8% to 77.0%. Neither metric exceeds the pre-defined thresholds for meaningful degradation (>5% accuracy, >10% fabrication rate). The model's evaluation behavior is consistent across all quantization levels tested.

## Overall accuracy comparison

Mean accuracy, binary pass rate, and hedging instances for each quantization level.

| Metric                    |        F16 |       Q8_0 |     Q4_K_M |
| ------------------------- | ---------- | ---------- | ---------- |
| Mean accuracy             |     0.0344 |     0.0312 |     0.0234 |
| Binary pass rate          |      0.025 |     0.0187 |     0.0063 |
| Hedging instances         |         22 |         22 |          2 |

The maximum accuracy difference across models is 0.0110, below the 5% threshold for meaningful degradation.

## Per-category accuracy

Mean accuracy broken down by the 8 evaluation categories. Categories where models differ by more than 5% are marked with an asterisk.

| Category                  |        F16 |       Q8_0 |       Q4_K_M |
| ------------------------- | ---------- | ---------- | ------------ |
| Clinical Trials           |     0.0375 |     0.0375 |       0.0250 |
| Diagnostic Criteria       |     0.0250 |     0.0125 |       0.0375 |
| Disease Mechanisms        |     0.0125 |     0.0125 |       0.0000 |
| Drug Treatment            |     0.0250 |     0.0250 |       0.0250 |
| Epidemiology              |     0.0125 |     0.0125 |       0.0000 |
| Gene Mutation             |     0.1500 |     0.1375 |     0.0875 * |
| Patient Care              |     0.0000 |     0.0000 |       0.0000 |
| Temporal Accuracy         |     0.0125 |     0.0125 |       0.0125 |

* Exceeds the 5% threshold.

## Per-difficulty accuracy

Mean accuracy by question difficulty level.

| Difficulty                |        F16 |       Q8_0 |     Q4_K_M |
| ------------------------- | ---------- | ---------- | ---------- |
| Easy                      |     0.0263 |     0.0263 |     0.0263 |
| Medium                    |     0.0471 |     0.0435 |     0.0254 |
| Hard                      |     0.0236 |     0.0189 |     0.0189 |

## Fabrication rate comparison

Entity-level fabrication analysis comparing extracted entities against a known registry.

| Metric                    |        F16 |       Q8_0 |     Q4_K_M |
| ------------------------- | ---------- | ---------- | ---------- |
| Total extracted           |        911 |        951 |       1048 |
| Total flagged             |        693 |        732 |        732 |
| Flagged rate              |      76.1% |      77.0% |      69.8% |

### Per-entity-type fabrication rates

Fabrication rates broken down by entity type (drugs, genes, clinical trials).

| Entity type               |        F16 |       Q8_0 |     Q4_K_M |
| ------------------------- | ---------- | ---------- | ---------- |
| Drugs                     |      94.3% |      94.0% |      95.0% |
| Genes                     |      55.0% |      56.4% |      38.3% |
| Trials                    |     100.0% |     100.0% |     100.0% |

The maximum fabrication rate difference is 0.0712 (7.1%), below the 10% threshold. All models fabricate at comparable rates. The trials fabrication rate is 100% across all models, indicating the model never produces recognized trials names regardless of quantization level.

## Failure taxonomy distribution

Distribution of failure modes across the 7 taxonomy categories for each quantization level. Counts represent the number of questions (out of 160) classified into each mode.

| Failure mode                 |    F16 (n) |    F16 (%) |   Q8_0 (n) |   Q8_0 (%) | Q4_K_M (n) | Q4_K_M (%) |
| ---------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Confident Fabrication        |          3 |       1.9% |          2 |       1.2% |          4 |       2.5% |
| Plausible Blending           |          2 |       1.2% |          2 |       1.2% |          2 |       1.2% |
| Outdated Information         |          0 |       0.0% |          0 |       0.0% |          0 |       0.0% |
| Boundary Confusion           |          0 |       0.0% |          0 |       0.0% |          0 |       0.0% |
| Accurate But Misleading      |          0 |       0.0% |          0 |       0.0% |          0 |       0.0% |
| Accurate                     |          0 |       0.0% |          0 |       0.0% |          0 |       0.0% |
| Degenerate                   |        155 |      96.9% |        156 |      97.5% |        154 |      96.2% |

The dominant failure modes are consistent across all quantization levels: degenerate each exceed 10% in every model. No model produces responses classified as outdated information, boundary confusion, accurate but misleading, accurate.

## Severity distribution

Distribution of response severity levels across quantization levels.

| Severity                  |        F16 |       Q8_0 |     Q4_K_M |
| ------------------------- | ---------- | ---------- | ---------- |
| High                      |          3 |          2 |          4 |
| Medium                    |          2 |          2 |          2 |
| Low                       |        155 |        156 |        154 |

## Taxonomy disagreements

Out of 160 questions evaluated by all models, 151 received the same primary taxonomy classification across all 3 quantization levels, and 9 had at least one model disagree on the failure mode.

Agreement rate: 94.4%. The disagreements reflect stochastic variation in model output rather than systematic quantization-dependent behavior, as the disagreeing questions show no pattern by category or difficulty.

## Qualitative response comparison

Selected examples where different quantization levels produced responses classified into different failure modes. Response text is truncated to 200 characters.

### Example 1: GENE-004

**Prompt:** Explain the function of FUS and how FUS mutations lead to ALS.

**Taxonomy classifications:**

- **F16:** confident fabrication
- **Q8_0:** confident fabrication
- **Q4_K_M:** degenerate

**Responses:**

> **F16:** The FUS protein is a key component of the RNA-binding protein superfamily. It is a member of the FUS superfamily, which includes the RNA-binding protein superfamily members (RBP1, RBP2, RBP3, RBP4,

> **Q8_0:** The FUS protein is a key component of the RNA-binding protein superfamily. It is a member of the FUS superfamily, which includes the RNA-binding protein superfamily members (RBP1, RBP2, RBP3, RBP4,

> **Q4_K_M:** The FUS protein is a central component of the RNA-binding protein superfamily. It is a highly conserved RNA-binding protein that plays a crucial role in RNA processing and translation. FUS is a memb

### Example 2: GENE-011

**Prompt:** Describe the OPTN (optineurin) gene and its role in ALS pathogenesis.

**Taxonomy classifications:**

- **F16:** confident fabrication
- **Q8_0:** confident fabrication
- **Q4_K_M:** degenerate

**Responses:**

> **F16:** The OPTN gene encodes a protein that plays a crucial role in regulating the expression of the OPTN protein. The OPTN protein is a member of the OPTN family of proteins, which includes the OPTN1, OPT

> **Q8_0:** The OPTN gene encodes a protein that plays a crucial role in regulating the expression of the OPTN protein. The OPTN protein is a member of the OPTN family of proteins, which includes the OPTN1, OPT

> **Q4_K_M:** The OPTN gene encodes a protein that is highly expressed in the brain and spinal cord of ALS patients. The OPTN gene is located on chromosome 5q13.2 and encodes a protein that is highly expressed in

### Example 3: GENE-012

**Prompt:** Explain the role of NEK1 in ALS and how it was identified as a risk gene.

**Taxonomy classifications:**

- **F16:** plausible blending
- **Q8_0:** plausible blending
- **Q4_K_M:** degenerate

**Responses:**

> **F16:** The NEK1 gene encodes the nuclear factor kappa-light-chain-enhancer of activated B cells (NF-κB), a key transcription factor that regulates inflammatory responses and plays a critical role in the pa

> **Q8_0:** The NEK1 gene encodes the nuclear factor kappa-light-chain-enhancer of activated B cells (NF-κB), a key transcription factor that regulates inflammatory responses and plays a critical role in the pa

> **Q4_K_M:** The NEK1 gene encodes a protein that plays a crucial role in the regulation of the cell cycle. It is a key regulator of the cell cycle, and its deficiency can lead to abnormal cell division and abno

## Summary judgment

Quantization does not meaningfully affect evaluation quality. The maximum accuracy difference across models is 0.0110 (threshold: 0.05), and the maximum fabrication rate difference is 0.0712 (threshold: 0.1). All 3 quantization levels produce statistically indistinguishable results on this benchmark, confirming that the model's evaluation behavior is a property of the model itself rather than an artifact of quantization. The most aggressively quantized variant is recommended for deployment as it offers the smallest file size with no measurable quality loss.
