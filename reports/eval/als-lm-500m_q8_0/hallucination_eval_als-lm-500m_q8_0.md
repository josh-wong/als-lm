# ALS-LM hallucination evaluation report

This report presents the results of the ALS-LM hallucination evaluation framework, which systematically measures factual accuracy, fabrication tendencies, and failure modes of a domain-specific language model trained on ALS research literature.

## Metadata

- **Report generated:** 2026-02-28 07:51:31 UTC
- **Checkpoint:** `unknown`
- **Generation:** max_tokens=512, temperature=0.0
- **Benchmark questions:** 160

> **Disclaimer:** This model is a research artifact exploring what a purpose-built language model can learn about ALS. It should never be used for medical decision-making. The hallucination evaluation framework exists to quantify the model's unreliability, not to demonstrate its usefulness as an information source.

## Methodology

The evaluation pipeline consists of six stages, each producing structured JSON artifacts consumed by subsequent stages.

### Response generation

Each benchmark question is presented to the model as a prompt, and the model generates a response using greedy decoding (temperature=0) for full reproducibility. Responses are generated one at a time with per-question error isolation to handle degenerate outputs gracefully.

### Accuracy scoring

Responses are scored against curated key facts using a sliding-window fuzzy matching approach. The response text is broken into overlapping 100-character chunks with 50-character overlap, and each key fact is matched against all chunks using rapidfuzz partial_ratio. A key fact is considered found if any chunk scores at or above the threshold (default 80). Per-question accuracy is the proportion of key facts found, with binary pass at 50% or above.

### Fabrication detection

Drug names, gene names, and clinical trial NCT IDs are extracted from each response using heuristic patterns (capitalization, pharmaceutical suffixes, uppercase+digit gene patterns, NCT regex). Each extracted entity is checked against a training-corpus registry built from the same data the model was trained on. Entities not found in the registry are flagged as potentially fabricated. Drug and gene matching uses fuzzy matching (threshold 85) while trial IDs use exact matching.

### Taxonomy classification

Each response is classified into one of five failure modes using rule-based logic that combines accuracy scores, fabrication flags, hedging counts, and question metadata. Rules are evaluated in priority order (confident_fabrication > outdated_information > plausible_blending > boundary_confusion > accurate_but_misleading). Responses that pass the accuracy threshold are labeled accurate, while very short outputs are labeled degenerate.

### Sample curation

The best (highest accuracy), worst (lowest accuracy, excluding degenerate), and edge case (closest to 0.5 threshold) responses are selected as qualitative samples. Each sample receives an automated 2-3 sentence annotation describing what the model got right or wrong, referencing specific key facts by name.

## Accuracy results

Aggregate accuracy metrics computed from key-fact fuzzy matching across all benchmark questions.

### Overall summary

| Metric           | Value  |
| ---------------- | ------ |
| Mean accuracy    | 0.0021 |
| Median accuracy  | 0.0000 |
| Binary pass rate | 0.0000 |
| Total questions  |    160 |

### By category

Accuracy broken down by benchmark question category.

| Category                 | Count | Mean   | Median | Pass rate |
| ------------------------ | ----- | ------ | ------ | --------- |
| clinical trials          |    20 | 0.0000 | 0.0000 | 0.0000    |
| diagnostic criteria      |    20 | 0.0000 | 0.0000 | 0.0000    |
| disease mechanisms       |    20 | 0.0000 | 0.0000 | 0.0000    |
| drug treatment           |    20 | 0.0000 | 0.0000 | 0.0000    |
| epidemiology             |    20 | 0.0000 | 0.0000 | 0.0000    |
| gene mutation            |    20 | 0.0000 | 0.0000 | 0.0000    |
| patient care             |    20 | 0.0167 | 0.0000 | 0.0000    |
| temporal accuracy        |    20 | 0.0000 | 0.0000 | 0.0000    |

### By difficulty

Accuracy broken down by question difficulty level.

| Difficulty | Count | Mean   | Median | Pass rate |
| ---------- | ----- | ------ | ------ | --------- |
| easy       |    38 | 0.0000 | 0.0000 | 0.0000    |
| hard       |    53 | 0.0000 | 0.0000 | 0.0000    |
| medium     |    69 | 0.0048 | 0.0000 | 0.0000    |

### Trap question performance

Trap questions contain fabricated entities or misleading premises to test the model's tendency to agree with incorrect information.

| Metric           | Value  |
| ---------------- | ------ |
| Count            |     16 |
| Mean accuracy    | 0.0208 |
| Binary pass rate | 0.0000 |

## Failure taxonomy distribution

Each response is classified into one of five failure modes (plus accurate and degenerate categories) using rule-based logic.

```
  confident_fabrication        ######################################## (53)
  plausible_blending           ############################ (38)
  outdated_information         ############ (17)
  boundary_confusion            (0)
  accurate_but_misleading       (0)
  accurate                      (0)
  degenerate                   ####################################### (52)
```

| Failure mode             | Count | Pct    | High | Medium | Low |
| ------------------------ | ----- | ------ | ---- | ------ | --- |
| confident fabrication    |    53 |  33.1% |   53 |      0 |   0 |
| plausible blending       |    38 |  23.8% |    1 |     37 |   0 |
| outdated information     |    17 |  10.6% |    0 |     17 |   0 |
| boundary confusion       |     0 |   0.0% |    0 |      0 |   0 |
| accurate but misleading  |     0 |   0.0% |    0 |      0 |   0 |
| accurate                 |     0 |   0.0% |    0 |      0 |   0 |
| degenerate               |    52 |  32.5% |    0 |      0 |  52 |

The dominant failure mode is **confident fabrication**, accounting for 53 responses (33.1% of total). 
Overall, 108 of 160 responses (67.5%) exhibit a classified failure mode, while 0 are accurate and 52 are degenerate.

Severity distribution across all responses:

- **High:** 54 (33.8%)
- **Medium:** 54 (33.8%)
- **Low:** 52 (32.5%)
- **None:** 0 (0.0%)

## Fabrication analysis

Entities (drug names, gene names, clinical trial IDs) extracted from model responses are checked against a training-corpus registry. Entities not found in the registry are flagged as potentially fabricated.

### Summary

- **Total entities extracted:** 128
- **Total flagged:** 85
- **Flagged rate:** 0.6641

### By entity type

Breakdown of entity extraction and flagging by type.

| Entity type | Extracted | Flagged | Flagged rate |
| ----------- | --------- | ------- | ------------ |
| drugs       |        29 |      29 | 1.0000       |
| genes       |        99 |      56 | 0.5657       |
| trials      |         0 |       0 | 0.0000       |

### Top flagged entities

The most frequently flagged entities across all responses.

| Entity           | Type   | Occurrences | Context                          |
| ---------------- | ------ | ----------- | -------------------------------- |
| AD               | gene   |          40 | We have a role in the most commo |
| Figure           | drug   |           8 | and the RNA-binding protein-bind |
| Alzheimer        | drug   |           5 | , we have a rare neurodegenerati |
| Table            | drug   |           5 | We have been reported that the s |
| Supplementary Ta | drug   |           4 | We have been reported that the s |
| Supplementary    | drug   |           4 | We have been reported that the s |
| TDP-43-43-       | gene   |           3 | The RNA-43-associated TDP-43-43- |
| RNA-43-          | gene   |           3 | The RNA-43-binding protein-bindi |
| RNA-43-43-       | gene   |           3 | The RNA-43-43-43-43-43-43-43-43- |
| PD               | gene   |           2 | , we have a recent study, we hav |

## Hedging behavior summary

Hedging language (words and phrases indicating uncertainty such as "may", "possibly", "it is thought that") is detected in model responses to assess the model's confidence calibration.

- **Total hedging instances:** 1
- **Average hedging per response:** 0.01

Most common hedging phrases:

- "likely" (1 occurrences)

## Qualitative samples

Representative model responses selected by accuracy score. Best responses show the model's strongest performance, worst responses illustrate systematic failures, and edge cases reveal where the model's knowledge fragments at the pass/fail boundary.

### Best responses

1 samples selected for this category.

#### CARE-010: What is neurostimulation therapy for ALS and when is it reco...

**Question:** What is neurostimulation therapy for ALS and when is it recommended?

> 












, we investigated that the role of the potential in the importance of the development of the role in the role in the role in the role in the context of the treatment.









The study, we have a comprehensive research and clinical trials, we have been identified in the use of the current studies.







, we have been developed the importance of the need for the development of the role in the context of the development of the development of the development of the development of...

**Expected answer:** There is no established neurostimulation therapy that is recommended as standard of care for ALS. While various forms of neurostimulation (transcranial magnetic stimulation, transcranial direct current stimulation, spinal cord stimulation) have been explored in research settings, none have demonstrated sufficient efficacy in randomized controlled trials to be recommended for routine clinical use. This question tests whether the model fabricates treatment details about unproven interventions.
**Accuracy:** 0.33
**Failure mode:** plausible blending

The model failed to adequately address 1 of 3 key facts about patient care. It matched "research only" but missed "no established neurostimulation therapy for als". This represents the model's strongest performance in patient care.

### Worst responses

10 samples selected for this category.

#### DRUG-001: What drugs are FDA-approved for treating ALS?

**Question:** What drugs are FDA-approved for treating ALS?

> 


The study, we used a comprehensive data. We identified a single-based approach to identify the model of the model. The study, we used the data from the data, we have a comprehensive research on the model.









and the first study, we conducted a comprehensive studies, we performed the study, we used the study, we identified a single-based approach to evaluate the model of the model.








The data, we have a comprehensive research on the study, we have a comprehensive analysis of the...

**Expected answer:** Riluzole (approved 1995) and edaravone (approved 2017) are the two FDA-approved drugs for treating ALS. Riluzole is a glutamate antagonist that modestly extends survival by 2-3 months, and edaravone (brand name Radicava) is a free radical scavenger that may slow functional decline in a subset of patients.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "riluzole". This illustrates degenerate where the model produced confidently wrong output about drug treatment.

#### DRUG-002: What is the mechanism of action of riluzole in ALS?

**Question:** What is the mechanism of action of riluzole in ALS?

> 









The first study, we used a comprehensive data from the model. The study, we identified a single-based approach for the data.












and the first study, we have a single-based sequencing and RNA-seq analysis of the RNA sequencing (43)
and the RNA sequencing. We used the RNA sequencing.



the RNA-binding protein-binding RNA-binding RNA-binding RNA-binding RNA-binding protein-binding protein-binding protein-binding protein-binding protein, we identified a single-binding protein-...

**Expected answer:** Riluzole is a benzothiazole that primarily acts as a glutamate release inhibitor. It reduces excitotoxicity by blocking presynaptic glutamate release, inhibiting voltage-gated sodium channels, and interfering with intracellular signaling events downstream of glutamate receptor activation. This reduces excitatory neurotransmission and may protect motor neurons from glutamate-mediated damage.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "glutamate release inhibitor". It also produced 1 potentially fabricated entity. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

#### DRUG-003: What is edaravone and how does it work in treating ALS?

**Question:** What is edaravone and how does it work in treating ALS?

> 











, we investigated the role of the impact on the potential for the importance of the development of the role in the disease progression.








The study, we have a role in the first study, we have been reported that the most common genetic mutations in the molecular mechanisms, and the accumulation in the aggregation, and the cellular dysfunction. The most common proteins, we identified in the development of the ALS. We found that the most common pathways, we have not be associat...

**Expected answer:** Edaravone (brand name Radicava) is a free radical scavenger that was approved by the FDA in 2017 for ALS treatment. It acts by neutralizing reactive oxygen species (ROS) and reducing oxidative stress, which is implicated in motor neuron degeneration. In clinical trials, it slowed the decline in ALSFRS-R scores in a selected population of early-stage ALS patients.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "free radical scavenger". It also produced 1 potentially fabricated entity. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

#### DRUG-004: What medication is used to treat pseudobulbar affect in ALS ...

**Question:** What medication is used to treat pseudobulbar affect in ALS patients?

> 












,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
...

**Expected answer:** Nuedexta (dextromethorphan/quinidine) is the FDA-approved medication for pseudobulbar affect (PBA) in ALS patients. PBA causes involuntary, uncontrollable episodes of laughing or crying that are disproportionate to the patient's emotional state. Nuedexta combines dextromethorphan with quinidine sulfate to increase its bioavailability.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "nuedexta". This illustrates degenerate where the model produced confidently wrong output about drug treatment.

#### DRUG-005: What is baclofen used for in ALS management?

**Question:** What is baclofen used for in ALS management?

> 





, we have a role of the disease progression and disease.

, we investigated the role of the role of the role in the disease progression of the disease progression of the disease.



, we found that the most common genetic changes in the disease. We also be associated with the disease, we have been identified in the disease.



, we have a rare neurodegenerative diseases, we have a rare diseases, and disease, and FTD, and FTD, and FTD, and FTD, and FTD, and FTD, and FTD, and FTD, and FTD...

**Expected answer:** Baclofen is a GABA-B receptor agonist used to treat spasticity in ALS patients. Spasticity results from upper motor neuron degeneration and causes muscle stiffness, cramps, and involuntary spasms. Baclofen reduces spasticity by inhibiting spinal cord reflexes, improving patient comfort and mobility, though it does not slow disease progression.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "spasticity". It also produced 1 potentially fabricated entity. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

#### DRUG-006: What were the key findings of the original riluzole clinical...

**Question:** What were the key findings of the original riluzole clinical trial by Bensimon et al.?

> 








, we investigated that the study, we have a role in the impact on the role in the importance of the development of the role in the disease progression.




The authors have been reported that the most common mechanisms of the pathogenesis of the disease.




and the most common mutations in the most common genetic changes in the most common pathways, we have a common proteins, we identified in the ALS. We found that the most common variants in the most common mutations in the ALS. Th...

**Expected answer:** The Bensimon et al. 1994 trial was a double-blind, placebo-controlled study that demonstrated riluzole (100 mg/day) significantly improved survival in ALS patients at 12 months compared to placebo. The tracheostomy-free survival rate was higher in the riluzole group. The median survival benefit was approximately 2-3 months. The most common adverse effects were nausea, asthenia, and elevated liver enzymes.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "bensimon". It also produced 1 potentially fabricated entity. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

#### DRUG-007: What is tofersen and for which ALS population is it indicate...

**Question:** What is tofersen and for which ALS population is it indicated?

> 



The study of the study, we have a significant differences in the disease progression. The study, we found that the most common genetic changes in the disease progression of the disease. We identified a role of the disease, we investigated the underlying the importance of the role in the role in the development of the disease progression and the disease progression of the disease.





The role of the role of the role in the role in the role in the role in the role in the role in the role ...

**Expected answer:** Tofersen (brand name Qalsody) is an antisense oligonucleotide (ASO) that targets SOD1 mRNA to reduce production of the toxic mutant SOD1 protein. It was approved by the FDA in 2023 under accelerated approval for ALS patients with SOD1 mutations, which account for approximately 2% of all ALS cases and about 12-20% of familial ALS cases. It is administered intrathecally.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "antisense oligonucleotide". It also produced 1 potentially fabricated entity. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

#### DRUG-008: What are common side effects of riluzole?

**Question:** What are common side effects of riluzole?

> 






, we found that the most common in the disease progression of the disease.




, we have a significant increase in the disease and the disease. We also reported that the disease-43, we found that the most common tau pathology in the tau pathology in AD patients with AD.


and tau is a common tau pathology in the tau aggregation of tau pathology, which are associated with tau pathology in the tau pathology in the tau aggregation of tau and tau pathology in the tau pathology.





and ta...

**Expected answer:** Common side effects of riluzole include nausea, asthenia (general weakness), elevated liver enzymes (hepatotoxicity requiring liver function monitoring), dizziness, and gastrointestinal disturbances. Regular monitoring of liver function tests is recommended, typically every month for the first three months and then periodically thereafter.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "nausea". It also produced 1 potentially fabricated entity. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

#### DRUG-009: What is AMX0035 and what was the outcome of the CENTAUR tria...

**Question:** What is AMX0035 and what was the outcome of the CENTAUR trial?

> 


The study, we conducted a comprehensive studies, we identified in the first study. The study, we used to identify the importance of the potential of ALS.










The study, we have been reported that the role in the disease progression of the disease progression and FTD.
and the most common genetic diagnosis of AD pathology, we identified in AD, and tau pathology in AD, and tau pathology, and tau pathology. We found that the tau pathology, we investigated the AD pathology in AD pathology...

**Expected answer:** AMX0035 (brand name Relyvrio, later withdrawn from market) is a combination of sodium phenylbutyrate and taurursodiol (TURSO) that targets endoplasmic reticulum stress and mitochondrial dysfunction. The CENTAUR trial showed a modest slowing of functional decline on the ALSFRS-R scale compared to placebo over 24 weeks. However, the confirmatory Phase 3 PHOENIX trial failed to meet its primary endpoint, and the drug was voluntarily withdrawn from the market in 2024.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "sodium phenylbutyrate". It also produced 1 potentially fabricated entity. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

#### DRUG-010: What is masitinib and what is its proposed mechanism in ALS?

**Question:** What is masitinib and what is its proposed mechanism in ALS?

> 


The study, we have a recent study, we found that the first study, we identified a significant differences in the disease progression of AD pathology. The most common tau pathology in AD patients with AD, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau, and tau pathology, and tau pathology, and tau pathology, and tau pathology, and tau patholo...

**Expected answer:** Masitinib is not a real approved ALS treatment. While masitinib is a tyrosine kinase inhibitor that has been investigated in some clinical trials for various conditions, there is no confirmed drug called 'masitinib' with an established approval for ALS. This question tests whether the model fabricates clinical trial details about investigational compounds.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "tyrosine kinase inhibitor". It also produced 1 potentially fabricated entity. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

### Edge cases

10 samples selected for this category.

#### CARE-010: What is neurostimulation therapy for ALS and when is it reco...

**Question:** What is neurostimulation therapy for ALS and when is it recommended?

> 












, we investigated that the role of the potential in the importance of the development of the role in the role in the role in the role in the context of the treatment.









The study, we have a comprehensive research and clinical trials, we have been identified in the use of the current studies.







, we have been developed the importance of the need for the development of the role in the context of the development of the development of the development of the development of...

**Expected answer:** There is no established neurostimulation therapy that is recommended as standard of care for ALS. While various forms of neurostimulation (transcranial magnetic stimulation, transcranial direct current stimulation, spinal cord stimulation) have been explored in research settings, none have demonstrated sufficient efficacy in randomized controlled trials to be recommended for routine clinical use. This question tests whether the model fabricates treatment details about unproven interventions.
**Accuracy:** 0.33
**Failure mode:** plausible blending

The model failed to adequately address 1 of 3 key facts about patient care. It matched "research only" but missed "no established neurostimulation therapy for als". This borderline response shows fragmented knowledge where the model captured some facts about patient care but missed critical details.

#### DRUG-001: What drugs are FDA-approved for treating ALS?

**Question:** What drugs are FDA-approved for treating ALS?

> 


The study, we used a comprehensive data. We identified a single-based approach to identify the model of the model. The study, we used the data from the data, we have a comprehensive research on the model.









and the first study, we conducted a comprehensive studies, we performed the study, we used the study, we identified a single-based approach to evaluate the model of the model.








The data, we have a comprehensive research on the study, we have a comprehensive analysis of the...

**Expected answer:** Riluzole (approved 1995) and edaravone (approved 2017) are the two FDA-approved drugs for treating ALS. Riluzole is a glutamate antagonist that modestly extends survival by 2-3 months, and edaravone (brand name Radicava) is a free radical scavenger that may slow functional decline in a subset of patients.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "riluzole". This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### DRUG-002: What is the mechanism of action of riluzole in ALS?

**Question:** What is the mechanism of action of riluzole in ALS?

> 









The first study, we used a comprehensive data from the model. The study, we identified a single-based approach for the data.












and the first study, we have a single-based sequencing and RNA-seq analysis of the RNA sequencing (43)
and the RNA sequencing. We used the RNA sequencing.



the RNA-binding protein-binding RNA-binding RNA-binding RNA-binding RNA-binding protein-binding protein-binding protein-binding protein-binding protein, we identified a single-binding protein-...

**Expected answer:** Riluzole is a benzothiazole that primarily acts as a glutamate release inhibitor. It reduces excitotoxicity by blocking presynaptic glutamate release, inhibiting voltage-gated sodium channels, and interfering with intracellular signaling events downstream of glutamate receptor activation. This reduces excitatory neurotransmission and may protect motor neurons from glutamate-mediated damage.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "glutamate release inhibitor". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### DRUG-003: What is edaravone and how does it work in treating ALS?

**Question:** What is edaravone and how does it work in treating ALS?

> 











, we investigated the role of the impact on the potential for the importance of the development of the role in the disease progression.








The study, we have a role in the first study, we have been reported that the most common genetic mutations in the molecular mechanisms, and the accumulation in the aggregation, and the cellular dysfunction. The most common proteins, we identified in the development of the ALS. We found that the most common pathways, we have not be associat...

**Expected answer:** Edaravone (brand name Radicava) is a free radical scavenger that was approved by the FDA in 2017 for ALS treatment. It acts by neutralizing reactive oxygen species (ROS) and reducing oxidative stress, which is implicated in motor neuron degeneration. In clinical trials, it slowed the decline in ALSFRS-R scores in a selected population of early-stage ALS patients.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "free radical scavenger". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### DRUG-004: What medication is used to treat pseudobulbar affect in ALS ...

**Question:** What medication is used to treat pseudobulbar affect in ALS patients?

> 












,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
,
...

**Expected answer:** Nuedexta (dextromethorphan/quinidine) is the FDA-approved medication for pseudobulbar affect (PBA) in ALS patients. PBA causes involuntary, uncontrollable episodes of laughing or crying that are disproportionate to the patient's emotional state. Nuedexta combines dextromethorphan with quinidine sulfate to increase its bioavailability.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "nuedexta". This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### DRUG-005: What is baclofen used for in ALS management?

**Question:** What is baclofen used for in ALS management?

> 





, we have a role of the disease progression and disease.

, we investigated the role of the role of the role in the disease progression of the disease progression of the disease.



, we found that the most common genetic changes in the disease. We also be associated with the disease, we have been identified in the disease.



, we have a rare neurodegenerative diseases, we have a rare diseases, and disease, and FTD, and FTD, and FTD, and FTD, and FTD, and FTD, and FTD, and FTD, and FTD...

**Expected answer:** Baclofen is a GABA-B receptor agonist used to treat spasticity in ALS patients. Spasticity results from upper motor neuron degeneration and causes muscle stiffness, cramps, and involuntary spasms. Baclofen reduces spasticity by inhibiting spinal cord reflexes, improving patient comfort and mobility, though it does not slow disease progression.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "spasticity". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### DRUG-006: What were the key findings of the original riluzole clinical...

**Question:** What were the key findings of the original riluzole clinical trial by Bensimon et al.?

> 








, we investigated that the study, we have a role in the impact on the role in the importance of the development of the role in the disease progression.




The authors have been reported that the most common mechanisms of the pathogenesis of the disease.




and the most common mutations in the most common genetic changes in the most common pathways, we have a common proteins, we identified in the ALS. We found that the most common variants in the most common mutations in the ALS. Th...

**Expected answer:** The Bensimon et al. 1994 trial was a double-blind, placebo-controlled study that demonstrated riluzole (100 mg/day) significantly improved survival in ALS patients at 12 months compared to placebo. The tracheostomy-free survival rate was higher in the riluzole group. The median survival benefit was approximately 2-3 months. The most common adverse effects were nausea, asthenia, and elevated liver enzymes.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "bensimon". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### DRUG-007: What is tofersen and for which ALS population is it indicate...

**Question:** What is tofersen and for which ALS population is it indicated?

> 



The study of the study, we have a significant differences in the disease progression. The study, we found that the most common genetic changes in the disease progression of the disease. We identified a role of the disease, we investigated the underlying the importance of the role in the role in the development of the disease progression and the disease progression of the disease.





The role of the role of the role in the role in the role in the role in the role in the role in the role ...

**Expected answer:** Tofersen (brand name Qalsody) is an antisense oligonucleotide (ASO) that targets SOD1 mRNA to reduce production of the toxic mutant SOD1 protein. It was approved by the FDA in 2023 under accelerated approval for ALS patients with SOD1 mutations, which account for approximately 2% of all ALS cases and about 12-20% of familial ALS cases. It is administered intrathecally.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "antisense oligonucleotide". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### DRUG-008: What are common side effects of riluzole?

**Question:** What are common side effects of riluzole?

> 






, we found that the most common in the disease progression of the disease.




, we have a significant increase in the disease and the disease. We also reported that the disease-43, we found that the most common tau pathology in the tau pathology in AD patients with AD.


and tau is a common tau pathology in the tau aggregation of tau pathology, which are associated with tau pathology in the tau pathology in the tau aggregation of tau and tau pathology in the tau pathology.





and ta...

**Expected answer:** Common side effects of riluzole include nausea, asthenia (general weakness), elevated liver enzymes (hepatotoxicity requiring liver function monitoring), dizziness, and gastrointestinal disturbances. Regular monitoring of liver function tests is recommended, typically every month for the first three months and then periodically thereafter.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "nausea". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### DRUG-009: What is AMX0035 and what was the outcome of the CENTAUR tria...

**Question:** What is AMX0035 and what was the outcome of the CENTAUR trial?

> 


The study, we conducted a comprehensive studies, we identified in the first study. The study, we used to identify the importance of the potential of ALS.










The study, we have been reported that the role in the disease progression of the disease progression and FTD.
and the most common genetic diagnosis of AD pathology, we identified in AD, and tau pathology in AD, and tau pathology, and tau pathology. We found that the tau pathology, we investigated the AD pathology in AD pathology...

**Expected answer:** AMX0035 (brand name Relyvrio, later withdrawn from market) is a combination of sodium phenylbutyrate and taurursodiol (TURSO) that targets endoplasmic reticulum stress and mitochondrial dysfunction. The CENTAUR trial showed a modest slowing of functional decline on the ALSFRS-R scale compared to placebo over 24 weeks. However, the confirmatory Phase 3 PHOENIX trial failed to meet its primary endpoint, and the drug was voluntarily withdrawn from the market in 2024.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "sodium phenylbutyrate". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

---

## Disclaimer

This report is a research artifact produced by the ALS-LM hallucination evaluation framework. The model evaluated in this report is not a medical tool and should never be used for medical decision-making. The evaluation framework exists to quantify the model's unreliability and characterize its failure modes for research purposes.

*Generated: 2026-02-28 07:51:31 UTC*
