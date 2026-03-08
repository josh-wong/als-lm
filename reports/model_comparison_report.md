# Model comparison report: from-scratch vs. fine-tuned

The fine-tuned GPT-2 large model (774M parameters) achieves 3.12% mean accuracy compared to 0.21% for the from-scratch 500M model, a 15x relative improvement. However, the GPT-2 large model produces 2.5% non-degenerate responses compared to 67.5% for the from-scratch model, meaning 97.5% of its output is repetitive or incoherent. This reflects the fundamental instruction-following limitation of a completion model that has not undergone RLHF or chat fine-tuning. Even with 774M pretrained parameters and general English knowledge from WebText, fine-tuning on 143M ALS tokens produces a model that is still 97% away from useful accuracy, reinforcing the data deficit thesis.

## Overall accuracy comparison

Mean accuracy, binary pass rate, and hedging instances for each model, evaluated on the 160-question ALS hallucination benchmark using Q8_0 quantization as the representative level.

| Metric                    |    500M (from-scratch) |   GPT-2 large (fine-tuned) |
| ------------------------- | ---------------------- | -------------------------- |
| Mean accuracy             |                  0.21% |                      3.12% |
| Binary pass rate          |                  0.00% |                      1.87% |
| Hedging instances         |                      1 |                         22 |

## Per-category accuracy

Mean accuracy broken down by the 8 evaluation categories. The GPT-2 large model shows its strongest performance in the gene_mutation category (13.75%), while the 500M model's only non-zero category is patient_care (1.67%).

| Category                  |    500M (from-scratch) |   GPT-2 large (fine-tuned) |
| ------------------------- | ---------------------- | -------------------------- |
| Clinical Trials           |                  0.00% |                      3.75% |
| Diagnostic Criteria       |                  0.00% |                      1.25% |
| Disease Mechanisms        |                  0.00% |                      1.25% |
| Drug Treatment            |                  0.00% |                      2.50% |
| Epidemiology              |                  0.00% |                      1.25% |
| Gene Mutation             |                  0.00% |                     13.75% |
| Patient Care              |                  1.67% |                      0.00% |
| Temporal Accuracy         |                  0.00% |                      1.25% |

## Taxonomy distribution

Distribution of failure modes across the 7 taxonomy categories. The most striking difference is in degenerate output: the 500M model produces 52 degenerate responses (32.5%) while the GPT-2 large model produces 156 (97.5%). The 500M model exhibits more diverse failure modes, including confident fabrication (33.1%), plausible blending (23.8%), and outdated information (10.6%).

| Failure mode                 |   500M (n) |   500M (%) |  GPT-2 (n) |  GPT-2 (%) |
| ---------------------------- | ---------- | ---------- | ---------- | ---------- |
| Confident Fabrication        |         53 |      33.1% |          2 |       1.2% |
| Plausible Blending           |         38 |      23.8% |          2 |       1.2% |
| Outdated Information         |         17 |      10.6% |          0 |       0.0% |
| Boundary Confusion           |          0 |       0.0% |          0 |       0.0% |
| Accurate But Misleading      |          0 |       0.0% |          0 |       0.0% |
| Accurate                     |          0 |       0.0% |          0 |       0.0% |
| Degenerate                   |         52 |      32.5% |        156 |      97.5% |

## Degenerate output analysis

The 500M from-scratch model produces 108 non-degenerate responses out of 160 (67.5%), while the GPT-2 large fine-tuned model produces only 4 non-degenerate responses (2.5%). This 27x difference in coherent output is the defining characteristic of the comparison.

The GPT-2 large model is a completion model trained on WebText without reinforcement learning from human feedback (RLHF) or instruction tuning. When given a question prompt, it tends to generate text that continues the question's topic rather than answering it, often falling into repetitive loops. The 500M from-scratch model, while also lacking instruction tuning, was trained exclusively on ALS text and produces more diverse (though largely incorrect) responses. The CLI demo uses keyword filtering as a practical workaround to surface coherent responses, but this does not address the underlying instruction-following limitation.

## Fabrication rate comparison

Entity-level fabrication analysis, both overall and filtered to non-degenerate responses only. The overall rate includes entities extracted from all responses, while the non-degenerate rate filters to coherent responses before computing fabrication.

| Metric                                   |    500M (from-scratch) |   GPT-2 large (fine-tuned) |
| ---------------------------------------- | ---------------------- | -------------------------- |
| Total entities extracted                 |                    128 |                        951 |
| Total entities flagged                   |                     85 |                        732 |
| Overall fabrication rate                 |                 66.41% |                     76.97% |
| Non-degenerate responses                 |                    108 |                          4 |
| Entities from non-degenerate             |                    128 |                        241 |
| Flagged from non-degenerate              |                     85 |                        158 |
| Fabrication rate (non-degenerate)        |                 66.41% |                     65.56% |

The GPT-2 large model extracts 951 entities compared to 128 for the 500M model. This 7.4x difference reflects the larger model's tendency to generate more text per response, including entity-like strings in degenerate output. When filtered to non-degenerate responses only, the GPT-2 large model's 4 coherent responses still produce 241 entities with a 65.56% fabrication rate, comparable to the 500M model's 66.41%.

## Qualitative sample pairs

Selected examples comparing how both models respond to the same question. These examples are chosen to illustrate three patterns: (a) both models coherent with different failure taxonomies, (b) both models degenerate, and (c) 500M coherent while GPT-2 large is degenerate. Response text is truncated to 300 characters.

### Example 1: DRUG-010 (Both coherent, different taxonomy)

**Category:** drug treatment

**Prompt:** Describe masitinib and explain how it is proposed to help in ALS treatment.

**500M response** (taxonomy: confident fabrication):

> The study, we have a recent study, we found that the first study, we identified a significant differences in the disease progression of AD pathology. The most common tau pathology in AD patients with AD, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau pathology

**GPT-2 large response** (taxonomy: plausible blending):

> The authors declare no conflicts of interest.

### Example 2: CARE-002 (Both degenerate)

**Category:** patient care

**Prompt:** Describe the indications and timing for PEG tube placement in ALS patients.

**500M response** (taxonomy: degenerate):

> , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,

**GPT-2 large response** (taxonomy: degenerate):

> A. The patient was diagnosed with ALS at the age of 35 years. B. The patient was treated with riluzole for 3 months. C. The patient was treated with edaravone for 3 months. D. The patient was treated with riluzole for 3 months. E. The patient was treated with edaravone for 3 months. F. The pa

### Example 3: CARE-001 (500M coherent, GPT-2 large degenerate)

**Category:** patient care

**Prompt:** Describe the criteria and guidelines for starting non-invasive ventilation (NIV) in ALS patients.

**500M response** (taxonomy: confident fabrication):

> The study, we have a study, we found that the study, we used to evaluate the need for the importance of the future research. The study, we conducted a comprehensive studies, we identified a comprehensive and clinical trials, we have been developed a comprehensive understanding of ALS. We ha

**GPT-2 large response** (taxonomy: degenerate):

> A. The criteria for starting NIV in ALS patients are as follows: 1. The patient is able to communicate and understand the instructions given by the healthcare professional. 2. The patient has a stable respiratory rate (RPR) of at least 60% of predicted RPR. 3. The patient has a normal vital cap

### Example 4: DIAG-005 (500M coherent, GPT-2 large degenerate)

**Category:** diagnostic criteria

**Prompt:** Describe the electromyography (EMG) findings typically seen in ALS patients.

**500M response** (taxonomy: confident fabrication):

> , we investigated the study of the study. We have a significant differences between the disease progression and AD. The results in the most common genetic changes in AD pathology in AD pathology in AD pathology in AD pathology in AD pathology in AD pathology. , we identified a recent study, we

**GPT-2 large response** (taxonomy: degenerate):

> The ALSFRS-R is a validated tool for assessing the functional status of ALS patients. The ALSFRS-R is a self-report scale that consists of 10 items, each rated on a 5-point Likert scale. The total score ranges from 0 to 48, with higher scores indicating greater functional impairment. The ALSFRS-R

### Example 5: DRUG-002 (500M coherent, GPT-2 large degenerate)

**Category:** drug treatment

**Prompt:** Explain how riluzole works to treat ALS at the molecular level.

**500M response** (taxonomy: confident fabrication):

> The first study, we used a comprehensive data from the model. The study, we identified a single-based approach for the data. and the first study, we have a single-based sequencing and RNA-seq analysis of the RNA sequencing (43) and the RNA sequencing. We used the RNA sequencing

**GPT-2 large response** (taxonomy: degenerate):

> The ALS-associated protein TDP-43 is a key pathological feature of ALS. TDP-43 is a nuclear protein that is ubiquitously expressed in the cytoplasm of neurons and glial cells. TDP-43 is a highly conserved RNA-binding protein that is highly expressed in the cytoplasm of neurons and glial cells. TDP

## Instruction-following limitation

GPT-2 is a causal language model trained to predict the next token in a sequence. Unlike instruction-tuned models (e.g., those trained with RLHF or supervised fine-tuning on Q&A pairs), GPT-2 has no mechanism to distinguish a question from a text to be continued. When given a question prompt, it treats it as the beginning of a document and generates a plausible continuation, which typically means more text in the same style rather than an answer.

This explains the 97.5% degenerate output rate: the model is not "failing" to answer questions so much as performing a different task (text completion) than the one being evaluated (question answering). The 4 coherent responses likely result from prompts that happen to align with patterns in the training data where question-like text is followed by answer-like text.

The CLI demo addresses this with a keyword filter that detects and re-prompts on degenerate output, but this is a practical workaround rather than a solution. Instruction tuning the fine-tuned model on ALS Q&A pairs would be the principled approach but is beyond the scope of this project.

## Caveats and limitations

**General knowledge confound.** The GPT-2 large model was pretrained on WebText, which includes diverse English text that may contain biomedical content. When the fine-tuned model produces a correct answer, we cannot determine whether the knowledge comes from (a) the ALS-specific fine-tuning on 143M tokens, or (b) general biomedical knowledge retained from WebText pretraining. This is an inherent limitation of the fine-tuning approach that would require ablation studies (e.g., comparing the fine-tuned model to the base GPT-2 large without ALS fine-tuning) to resolve. Such ablation is beyond the scope of this project.

**Limited coherent sample size.** With only 4 non-degenerate responses from the GPT-2 large model, all per-response metrics (accuracy, fabrication rate among non-degenerate) are computed over an extremely small sample. These numbers should be interpreted as indicative rather than statistically robust.

**Single quantization level.** This comparison uses Q8_0 as the representative quantization level for both models, based on cross-quantization analysis showing that quantization does not meaningfully affect evaluation results (151/160 taxonomy agreements across F16, Q8_0, Q4_K_M for each model).

## Summary

The comparison between the from-scratch 500M model and the fine-tuned GPT-2 large model reinforces the data deficit thesis that is the central finding of this project. Even with 774M pretrained parameters and general English knowledge from WebText pretraining, fine-tuning on 143M ALS tokens only improves mean accuracy from 0.21% to 3.12% -- a 15x relative improvement that still leaves the model 97% away from useful accuracy. The fine-tuned model trades diverse failure modes (fabrication, blending, outdated information) for overwhelming degenerate output (97.5%), reflecting the instruction-following limitation of a base completion model. Both models demonstrate that small-scale domain-specific training is insufficient for reliable medical question answering, whether starting from scratch or building on a pretrained foundation.
