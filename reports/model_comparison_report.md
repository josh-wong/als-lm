# Model comparison report: 4-model cross-comparison

Cross-comparison of 4 model variants evaluated on the 160-question ALS hallucination benchmark using Q8_0 quantization as the representative level. The table below summarizes each model's accuracy and degenerate output rate.

| Model                               |   Accuracy |   Non-degenerate |   Degenerate |
| ----------------------------------- | ---------- | ---------------- | ------------ |
| 500M                                |      0.21% |            67.5% |   52/160    |
| GPT-2 large                         |      3.12% |             2.5% |  156/160    |
| 1B base                             |      0.00% |            35.0% |  104/160    |
| 1B instruct                         |      0.00% |             0.0% |  160/160    |

The fine-tuned GPT-2 large model remains the highest-accuracy variant at 3.12%, while both 1B models (base and instruction-tuned) achieve 0.00% accuracy. The instruction-tuned model produces 160/160 degenerate responses (100%), worse than the 1B base model's 35.0% non-degenerate rate. Scaling from 500M to 1B parameters without pre-trained knowledge does not improve accuracy. Instruction tuning the from-scratch 1B model with ~970 pairs causes complete output collapse rather than surfacing internalized knowledge.

## Overall accuracy comparison

Mean accuracy, binary pass rate, and hedging instances for each model, evaluated on the 160-question ALS hallucination benchmark.

| Metric                    |           500M |    GPT-2 large |        1B base |    1B instruct |
| ------------------------- | -------------- | -------------- | -------------- | -------------- |
| Mean accuracy             |          0.21% |          3.12% |          0.00% |          0.00% |
| Binary pass rate          |          0.00% |          1.87% |          0.00% |          0.00% |
| Hedging instances         |              1 |             22 |              0 |              0 |

## Per-category accuracy

Mean accuracy broken down by the 8 evaluation categories. All from-scratch models (500M, 1B base, 1B instruct) score near zero across all categories, while GPT-2 large shows modest variation.

| Category                  |           500M |    GPT-2 large |        1B base |    1B instruct |
| ------------------------- | -------------- | -------------- | -------------- | -------------- |
| Clinical Trials           |          0.00% |          3.75% |          0.00% |          0.00% |
| Diagnostic Criteria       |          0.00% |          1.25% |          0.00% |          0.00% |
| Disease Mechanisms        |          0.00% |          1.25% |          0.00% |          0.00% |
| Drug Treatment            |          0.00% |          2.50% |          0.00% |          0.00% |
| Epidemiology              |          0.00% |          1.25% |          0.00% |          0.00% |
| Gene Mutation             |          0.00% |         13.75% |          0.00% |          0.00% |
| Patient Care              |          1.67% |          0.00% |          0.00% |          0.00% |
| Temporal Accuracy         |          0.00% |          1.25% |          0.00% |          0.00% |

## Taxonomy distribution

Distribution of failure modes across the 7 taxonomy categories for all 4 models. The 1B instruct model shows 100% degenerate output, while the 1B base model produces a mix of degenerate and non-degenerate failure modes similar to the 500M model's pattern.

| Failure mode                 |     500M (n) |     500M (%) | GPT-2 large (n) | GPT-2 large (%) |  1B base (n) |  1B base (%) | 1B instruct (n) | 1B instruct (%) |
| ---------------------------- | ------------ | ------------ | --------------- | --------------- | ------------ | ------------ | --------------- | --------------- |
| Confident Fabrication        |           53 |        33.1% |               2 |            1.2% |           19 |        11.9% |               0 |            0.0% |
| Plausible Blending           |           38 |        23.8% |               2 |            1.2% |           29 |        18.1% |               0 |            0.0% |
| Outdated Information         |           17 |        10.6% |               0 |            0.0% |            8 |         5.0% |               0 |            0.0% |
| Boundary Confusion           |            0 |         0.0% |               0 |            0.0% |            0 |         0.0% |               0 |            0.0% |
| Accurate But Misleading      |            0 |         0.0% |               0 |            0.0% |            0 |         0.0% |               0 |            0.0% |
| Accurate                     |            0 |         0.0% |               0 |            0.0% |            0 |         0.0% |               0 |            0.0% |
| Degenerate                   |           52 |        32.5% |             156 |           97.5% |          104 |        65.0% |             160 |          100.0% |

## Degenerate output analysis

| Model                               |   Non-degenerate |   Degenerate |   Non-deg rate |
| ----------------------------------- | ---------------- | ------------ | -------------- |
| 500M                                |              108 |           52 |          67.5% |
| GPT-2 large                         |                4 |          156 |           2.5% |
| 1B base                             |               56 |          104 |          35.0% |
| 1B instruct                         |                0 |          160 |           0.0% |

The 1B instruction-tuned model produces 160/160 degenerate responses, worse than any other model. The instruction tuning process caused the model to collapse into repeating the most common English token ('TheTheThe...'). By contrast, the 1B base model without SFT produces 56 non-degenerate responses, demonstrating that the SFT process itself degraded the model's output diversity rather than improving it.

## Fabrication rate comparison

Entity-level fabrication analysis. Models with 0 non-degenerate responses extract 0 entities from coherent output, so their non-degenerate fabrication rate is 0.00% by definition.

| Metric                                   |
| Metric                                   |           500M |    GPT-2 large |        1B base |    1B instruct |
| ---------------------------------------- | -------------- | -------------- | -------------- | -------------- |
| Total entities extracted                 |            128 |            951 |             73 |              0 |
| Total entities flagged                   |             85 |            732 |             73 |              0 |
| Overall fabrication rate                 |         66.41% |         76.97% |        100.00% |          0.00% |
| Non-degenerate responses                 |            108 |              4 |             56 |              0 |
| Entities from non-degenerate             |            128 |            241 |             21 |              0 |
| Flagged from non-degenerate              |             85 |            158 |             21 |              0 |
| Fabrication rate (non-deg)               |         66.41% |         65.56% |        100.00% |          0.00% |

## Qualitative sample pairs

Selected examples comparing how all 4 models respond to the same question. These examples illustrate patterns including instruct-model degenerate output alongside other models' responses, and cases where all models are degenerate. Response text is truncated to 300 characters.

### Example 1: CARE-001 (Instruct degenerate, others coherent)

**Category:** patient care

**Prompt:** Describe the criteria and guidelines for starting non-invasive ventilation (NIV) in ALS patients.

**500M response** (taxonomy: confident fabrication):

> The study, we have a study, we found that the study, we used to evaluate the need for the importance of the future research. The study, we conducted a comprehensive studies, we identified a comprehensive and clinical trials, we have been developed a comprehensive understanding of ALS. We ha

**GPT-2 large response** (taxonomy: degenerate):

> A. The criteria for starting NIV in ALS patients are as follows: 1. The patient is able to communicate and understand the instructions given by the healthcare professional. 2. The patient has a stable respiratory rate (RPR) of at least 60% of predicted RPR. 3. The patient has a normal vital cap

**1B base response** (taxonomy: degenerate):

> D3- toir3- toil3- toil3- toil3- toil/aintt deathulitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp hitamp

**1B instruct response** (taxonomy: degenerate):

> TheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheThe

### Example 2: DRUG-010 (Both coherent, different taxonomy)

**Category:** drug treatment

**Prompt:** Describe masitinib and explain how it is proposed to help in ALS treatment.

**500M response** (taxonomy: confident fabrication):

> The study, we have a recent study, we found that the first study, we identified a significant differences in the disease progression of AD pathology. The most common tau pathology in AD patients with AD, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau pathology

**GPT-2 large response** (taxonomy: plausible blending):

> The authors declare no conflicts of interest.

**1B base response** (taxonomy: degenerate):

> Fr four effort Star/ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

**1B instruct response** (taxonomy: degenerate):

> TheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheThe

### Example 3: CARE-002 (All models degenerate)

**Category:** patient care

**Prompt:** Describe the indications and timing for PEG tube placement in ALS patients.

**500M response** (taxonomy: degenerate):

> , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,

**GPT-2 large response** (taxonomy: degenerate):

> A. The patient was diagnosed with ALS at the age of 35 years. B. The patient was treated with riluzole for 3 months. C. The patient was treated with edaravone for 3 months. D. The patient was treated with riluzole for 3 months. E. The patient was treated with edaravone for 3 months. F. The pa

**1B base response** (taxonomy: degenerate):

> Fr four effort Star- to Star- to Star- to Star- to Star- to Star- to Star- to Star- to Star/ - - - - - - - - - - - - - - - - -

**1B instruct response** (taxonomy: degenerate):

> TheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheThe

### Example 4: DIAG-005 (500M coherent, GPT-2 large degenerate)

**Category:** diagnostic criteria

**Prompt:** Describe the electromyography (EMG) findings typically seen in ALS patients.

**500M response** (taxonomy: confident fabrication):

> , we investigated the study of the study. We have a significant differences between the disease progression and AD. The results in the most common genetic changes in AD pathology in AD pathology in AD pathology in AD pathology in AD pathology in AD pathology. , we identified a recent study, we

**GPT-2 large response** (taxonomy: degenerate):

> The ALSFRS-R is a validated tool for assessing the functional status of ALS patients. The ALSFRS-R is a self-report scale that consists of 10 items, each rated on a 5-point Likert scale. The total score ranges from 0 to 48, with higher scores indicating greater functional impairment. The ALSFRS-R

**1B base response** (taxonomy: plausible blending):

> 3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.3.

**1B instruct response** (taxonomy: degenerate):

> TheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheThe

### Example 5: EPI-002 (500M coherent, GPT-2 large degenerate)

**Category:** epidemiology

**Prompt:** State the average age at which ALS symptoms typically begin.

**500M response** (taxonomy: plausible blending):

> The study, we used a single-based approach for the model. The data were performed using the data from the data. \usepackage{upgreek} \usepackage{mathrsfs} \usepackage{amssymb} \usepackage{amsbsy} \usepackage{amsfonts} \usepackage{wasysym} \usepackage{upgreek} \usepackage{upgreek

**GPT-2 large response** (taxonomy: degenerate):

> The ALSFRS-R is a validated tool for assessing the severity of ALS. It is a self-report scale that consists of 10 items, each rated on a 5-point Likert scale. The total score ranges from 0 to 48, with higher scores indicating greater severity of the disease. The ALSFRS-R is a validated tool for a

**1B base response** (taxonomy: plausible blending):

> D.D.D.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.3.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.3.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.3.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.3.2.2.2.2.2.2.2.2.2.2.2.

**1B instruct response** (taxonomy: degenerate):

> TheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheTheThe

## Instruction-following limitation

Three of the four models in this comparison lack instruction-tuning on a pre-trained base: the 500M from-scratch, the 1B base, and the GPT-2 large (fine-tuned on ALS text but not on Q&A pairs). The fourth model (1B instruct) received supervised fine-tuning on ~970 ALS Q&A pairs, but the from-scratch base model lacked sufficient internalized knowledge for SFT to surface. The result was complete output collapse rather than structured question answering.

The GPT-2 large model's 97.5% degenerate rate reflects the instruction-following limitation of a completion model. The 1B instruct model's 100% degenerate rate shows that SFT on a from-scratch model with insufficient data makes instruction following worse, not better. The v1 research paper (Section 9.2) recommended 'a more modern instruction-capable base model' as the next step, and this SFT failure validates that recommendation.

## Perceived capability gap

The perceived capability gap measures the difference between how coherent a model's output appears (coherence%) and how factually accurate it is (accuracy%). A high gap indicates ethical risk: users may trust plausible-sounding but incorrect answers. A gap of 0% for degenerate models means the ethical risk did not materialize because the output is obviously broken.

| Model                               |    Coherence |   Accuracy |      Gap |
| ----------------------------------- | ------------ | ---------- | -------- |
| 500M                                |        67.5% |      0.21% |    67.3% |
| GPT-2 large                         |         2.5% |      3.12% |    -0.6% |
| 1B base                             |        35.0% |      0.00% |    35.0% |
| 1B instruct                         |         0.0% |      0.00% |     0.0% |

The 1B instruct and 1B base models (0% and near-0% accuracy) both show near-zero or zero capability gap. The 500M model has the highest gap due to producing many coherent-sounding but inaccurate responses. The GPT-2 large model's gap is limited by its high degenerate rate. For models that are fully degenerate, the gap is 0% -- the ethical concern of confident-but-wrong output does not apply when output is obviously broken.

## Caveats and limitations

**General knowledge confound.** The GPT-2 large model was pretrained on WebText, which includes diverse English text that may contain biomedical content. When the fine-tuned model produces a correct answer, we cannot determine whether the knowledge comes from (a) the ALS-specific fine-tuning on 143M tokens, or (b) general biomedical knowledge retained from WebText pretraining.

**Limited coherent sample size.** With only 4 non-degenerate responses from the GPT-2 large model, per-response metrics are computed over an extremely small sample.

**Checkpoint inference for 1B base.** The 1B base model was evaluated via direct checkpoint inference rather than through Ollama GGUF quantization. While this does not affect accuracy scoring, it means the 1B base results use a different inference path than the other three models.

**Single quantization level.** This comparison uses Q8_0 as the representative quantization level, based on cross-quantization analysis showing that quantization does not meaningfully affect evaluation results.

## Summary

The 4-model comparison tells a consistent story: pre-trained knowledge (GPT-2 large) helps far more than model scale (500M to 1B) or instruction tuning on a from-scratch model. The GPT-2 large model achieves 3.12% accuracy despite producing mostly degenerate output. Both 1B variants (base and instruction-tuned) achieve 0.00% accuracy, demonstrating that doubling model size without pre-trained knowledge provides no benefit. The instruction-tuned model's complete output collapse (100% degenerate) is the strongest evidence that SFT on a from-scratch model with ~970 instruction pairs and ~153M training tokens is insufficient -- the model lacks the foundational language understanding needed for instruction following.
