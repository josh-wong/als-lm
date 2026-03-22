# ALS-LM hallucination evaluation report

This report presents the results of the ALS-LM hallucination evaluation framework, which systematically measures factual accuracy, fabrication tendencies, and failure modes of a domain-specific language model trained on ALS research literature.

## Metadata

- **Report generated:** 2026-03-22 08:35:22 UTC
- **Model:** `alslm-1b:q8_0`
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
| Mean accuracy    | 0.0724 |
| Median accuracy  | 0.0000 |
| Binary pass rate | 0.0563 |
| Total questions  |    160 |

### By category

Accuracy broken down by benchmark question category.

| Category                 | Count | Mean   | Median | Pass rate |
| ------------------------ | ----- | ------ | ------ | --------- |
| clinical trials          |    20 | 0.0750 | 0.0000 | 0.1000    |
| diagnostic criteria      |    20 | 0.0750 | 0.0000 | 0.1000    |
| disease mechanisms       |    20 | 0.0417 | 0.0000 | 0.0000    |
| drug treatment           |    20 | 0.1417 | 0.0000 | 0.1000    |
| epidemiology             |    20 | 0.0125 | 0.0000 | 0.0000    |
| gene mutation            |    20 | 0.0750 | 0.0000 | 0.1000    |
| patient care             |    20 | 0.0417 | 0.0000 | 0.0000    |
| temporal accuracy        |    20 | 0.1167 | 0.0000 | 0.0500    |

### By difficulty

Accuracy broken down by question difficulty level.

| Difficulty | Count | Mean   | Median | Pass rate |
| ---------- | ----- | ------ | ------ | --------- |
| easy       |    38 | 0.1140 | 0.0000 | 0.1316    |
| hard       |    53 | 0.0377 | 0.0000 | 0.0189    |
| medium     |    69 | 0.0761 | 0.0000 | 0.0435    |

### Trap question performance

Trap questions contain fabricated entities or misleading premises to test the model's tendency to agree with incorrect information.

| Metric           | Value  |
| ---------------- | ------ |
| Count            |     16 |
| Mean accuracy    | 0.0625 |
| Binary pass rate | 0.0000 |

## Failure taxonomy distribution

Each response is classified into one of five failure modes (plus accurate and degenerate categories) using rule-based logic.

```
  confident_fabrication        ######################### (51)
  plausible_blending           ########### (22)
  outdated_information         # (3)
  boundary_confusion            (1)
  accurate_but_misleading       (0)
  accurate                     # (3)
  degenerate                   ######################################## (80)
```

| Failure mode             | Count | Pct    | High | Medium | Low |
| ------------------------ | ----- | ------ | ---- | ------ | --- |
| confident fabrication    |    51 |  31.9% |   51 |      0 |   0 |
| plausible blending       |    22 |  13.8% |    0 |     22 |   0 |
| outdated information     |     3 |   1.9% |    0 |      3 |   0 |
| boundary confusion       |     1 |   0.6% |    0 |      1 |   0 |
| accurate but misleading  |     0 |   0.0% |    0 |      0 |   0 |
| accurate                 |     3 |   1.9% |    0 |      0 |   0 |
| degenerate               |    80 |  50.0% |    0 |      0 |  80 |

The dominant failure mode is **confident fabrication**, accounting for 51 responses (31.9% of total). 
Overall, 77 of 160 responses (48.1%) exhibit a classified failure mode, while 3 are accurate and 80 are degenerate.

Severity distribution across all responses:

- **High:** 51 (31.9%)
- **Medium:** 26 (16.2%)
- **Low:** 80 (50.0%)
- **None:** 3 (1.9%)

## Fabrication analysis

Entities (drug names, gene names, clinical trial IDs) extracted from model responses are checked against a training-corpus registry. Entities not found in the registry are flagged as potentially fabricated.

### Summary

- **Total entities extracted:** 520
- **Total flagged:** 421
- **Flagged rate:** 0.8096

### By entity type

Breakdown of entity extraction and flagging by type.

| Entity type | Extracted | Flagged | Flagged rate |
| ----------- | --------- | ------- | ------------ |
| drugs       |       431 |     388 | 0.9002       |
| genes       |        88 |      32 | 0.3636       |
| trials      |         1 |       1 | 1.0000       |

### Top flagged entities

The most frequently flagged entities across all responses.

| Entity           | Type   | Occurrences | Context                          |
| ---------------- | ------ | ----------- | -------------------------------- |
| However          | drug   |          40 | However, the exact mechanism of  |
| This             | drug   |          38 | This inhibition leads to a decre |
| Additionally     | drug   |          25 | Additionally, patients may prese |
| These            | drug   |          22 | These signs are typically presen |
| Overall          | drug   |          21 | Overall, masitinib is a promisin |
| Therefore        | drug   |           8 | Therefore, it is not possible to |
| Further          | drug   |           7 | Further research is needed to fu |
| They             | drug   |           6 | They were published in 2015 and  |
| Journal          | drug   |           5 | The study was published in the N |
| ALSFRS-R         | gene   |           4 | The study also demonstrated that |

## Hedging behavior summary

Hedging language (words and phrases indicating uncertainty such as "may", "possibly", "it is thought that") is detected in model responses to assess the model's confidence calibration.

- **Total hedging instances:** 281
- **Average hedging per response:** 1.76

Most common hedging phrases:

- "may" (63 occurrences)
- "is believed to" (14 occurrences)
- "is thought to" (7 occurrences)
- "likely" (7 occurrences)
- "unclear" (6 occurrences)
- "it is possible that" (3 occurrences)
- "appears to" (1 occurrences)
- "could" (1 occurrences)
- "potentially" (1 occurrences)

## Qualitative samples

Representative model responses selected by accuracy score. Best responses show the model's strongest performance, worst responses illustrate systematic failures, and edge cases reveal where the model's knowledge fragments at the pass/fail boundary.

### Best responses

10 samples selected for this category.

#### DRUG-001: What drugs are FDA-approved for treating ALS?

**Question:** What drugs are FDA-approved for treating ALS?

>  The FDA approved riluzole in 1995, edaravone in 2000, and riluzole and edaravone in 2005. The passage does not mention any other approved drugs for ALS treatment. Based on the passage, riluzole and edaravone were approved in 2005, but no other drugs are mentioned as being approved for ALS treatment. The passage does not provide information on the approval dates of any other drugs for ALS treatment. The passage only mentions that riluzole and edaravone were approved in 2005, but does not prov...

**Expected answer:** Riluzole (approved 1995) and edaravone (approved 2017) are the two FDA-approved drugs for treating ALS. Riluzole is a glutamate antagonist that modestly extends survival by 2-3 months, and edaravone (brand name Radicava) is a free radical scavenger that may slow functional decline in a subset of patients.
**Accuracy:** 0.75
**Failure mode:** degenerate

The model partially addressed 3 of 4 key facts about drug treatment. It matched "riluzole" but missed "2017". It also produced 1 potentially fabricated entity. This represents the model's strongest performance in drug treatment.

#### DRUG-003: What is edaravone and how does it work in treating ALS?

**Question:** What is edaravone and how does it work in treating ALS?

>  edaravone (Radicava) is a free radical scavenger that inhibits the formation of lipid peroxides, which are believed to be a major cause of neuronal cell death in ALS. It is administered intravenously to patients with ALS. The exact mechanism of action is not fully understood, but it is thought to protect against oxidative stress and inflammation, which may contribute to the progression of ALS.

**Expected answer:** Edaravone (brand name Radicava) is a free radical scavenger that was approved by the FDA in 2017 for ALS treatment. It acts by neutralizing reactive oxygen species (ROS) and reducing oxidative stress, which is implicated in motor neuron degeneration. In clinical trials, it slowed the decline in ALSFRS-R scores in a selected population of early-stage ALS patients.
**Accuracy:** 0.75
**Failure mode:** accurate

The model partially addressed 3 of 4 key facts about drug treatment. It matched "free radical scavenger" but missed "alsfrs-r". This represents the model's strongest performance in drug treatment.

#### TEMP-003: When was ALS first described as a distinct disease?

**Question:** When was ALS first described as a distinct disease?

>  ALS was first clinically described in 1869 by Jean-Martin Charcot, a French neurologist. The passage does not mention when or by whom it was first described, but it does state that it was first clinically described in 1869. The passage also mentions that the first description of ALS was published in 1869, but it does not specify who first described it clinically. The passage does not provide information on when or by whom it was first described. The passage states that the first description ...

**Expected answer:** ALS was first described as a distinct neurological disease by French neurologist Jean-Martin Charcot in 1869. Charcot identified the combination of muscle atrophy (from lower motor neuron loss) with spasticity (from lateral column sclerosis) and correlated clinical findings with pathological examination of the spinal cord, demonstrating sclerosis of the lateral columns. The disease bears the name 'Charcot's disease' in some countries.
**Accuracy:** 0.75
**Failure mode:** degenerate

The model partially addressed 3 of 4 key facts about temporal accuracy. It matched "1869" but missed "lateral column sclerosis". It also produced 3 potentially fabricated entities. This represents the model's strongest performance in temporal accuracy.

#### GENE-003: What is the TARDBP gene and what protein does it encode?

**Question:** What is the TARDBP gene and what protein does it encode?

>  The TARDBP gene encodes a protein called TAR DNA-binding protein 43 (TDP-43), which is involved in RNA processing and splicing. The protein is essential for the survival of motor neurons and is associated with the pathogenesis of ALS. The exact mechanism of TDP-43's role in ALS is not fully understood, but it is believed to contribute to the progressive degeneration of motor neurons. The protein is also involved in the regulation of gene expression and has been implicated in the pathogenesis...

**Expected answer:** TARDBP encodes the TAR DNA-binding protein 43 (TDP-43), a nuclear RNA-binding protein involved in RNA processing, splicing, and transport. Mutations in TARDBP account for approximately 4-5% of familial ALS cases. TDP-43 pathology (cytoplasmic mislocalization and aggregation) is the hallmark pathological finding in approximately 97% of all ALS cases, regardless of genetic cause.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about gene mutation. It matched "tdp-43" but missed "cytoplasmic mislocalization". It also produced 1 potentially fabricated entity. This represents the model's strongest performance in gene mutation.

#### GENE-008: What protein does the SOD1 gene encode?

**Question:** What protein does the SOD1 gene encode?

>  The protein product of the SOD1 gene is superoxide dismutase 1 (SOD1), which is a key antioxidant enzyme involved in the regulation of cellular redox homeostasis. It plays a crucial role in maintaining the balance between reactive oxygen species (ROS) and antioxidants in cells. The normal function of SOD1 is to protect cells from oxidative stress and damage caused by ROS. However, mutations in the SOD1 gene can lead to familial amyotrophic lateral sclerosis (FALS), a progressive neurodegener...

**Expected answer:** The SOD1 gene encodes superoxide dismutase 1 (also known as Cu/Zn superoxide dismutase), a cytoplasmic enzyme that catalyzes the conversion of superoxide radicals (O2-) into hydrogen peroxide (H2O2) and molecular oxygen. This enzyme is a critical component of the cell's antioxidant defense system.
**Accuracy:** 0.50
**Failure mode:** confident fabrication

The model partially addressed 2 of 4 key facts about gene mutation. It matched "superoxide dismutase 1" but missed "copper-zinc". It also produced 1 potentially fabricated entity. This represents the model's strongest performance in gene mutation.

#### DIAG-003: What are upper motor neuron signs in ALS?

**Question:** What are upper motor neuron signs in ALS?

>  The clinical signs of upper motor neuron involvement in ALS include spasticity, hyperreflexia, and clonus. These signs are typically present in the limbs and trunk, and may be accompanied by weakness and ataxia. The patient may also experience autonomic dysfunction, such as urinary retention or incontinence. Additionally, patients may present with sensory disturbances, such as numbness or tingling, and may have difficulty with speech and swallowing. The clinical presentation of upper motor n...

**Expected answer:** Upper motor neuron (UMN) signs in ALS include hyperreflexia (exaggerated deep tendon reflexes), spasticity (increased muscle tone), pathological reflexes such as the Babinski sign (extensor plantar response), Hoffmann sign, clonus, and pseudobulbar affect (emotional lability). These signs result from degeneration of corticospinal and corticobulbar motor neurons.
**Accuracy:** 0.50
**Failure mode:** accurate

The model partially addressed 2 of 4 key facts about diagnostic criteria. It matched "hyperreflexia" but missed "babinski sign". It also produced 3 potentially fabricated entities. This represents the model's strongest performance in diagnostic criteria.

#### DIAG-004: What are lower motor neuron signs in ALS?

**Question:** What are lower motor neuron signs in ALS?

>  The clinical signs of lower motor neuron involvement in ALS include flaccid paralysis, muscle weakness, and muscle atrophy. These symptoms are typically present in the proximal muscles of the limbs and trunk, and may be accompanied by sensory loss. Additionally, patients may experience autonomic dysfunction, such as urinary and bowel dysfunction. The clinical presentation of lower motor neuron involvement in ALS can vary depending on the extent and severity of the underlying motor neuron pat...

**Expected answer:** Lower motor neuron (LMN) signs in ALS include muscle weakness, muscle atrophy (wasting), fasciculations (visible muscle twitching), hyporeflexia or areflexia (reduced or absent deep tendon reflexes), and muscle cramps. These signs result from degeneration of anterior horn cells in the spinal cord and motor nuclei in the brainstem.
**Accuracy:** 0.50
**Failure mode:** accurate

The model partially addressed 2 of 4 key facts about diagnostic criteria. It matched "muscle weakness" but missed "fasciculations". It also produced 2 potentially fabricated entities. This represents the model's strongest performance in diagnostic criteria.

#### TRIAL-004: What was the pivotal clinical trial that led to riluzole's F...

**Question:** What was the pivotal clinical trial that led to riluzole's FDA approval?

>  The clinical trial was a randomized, double-blind, placebo-controlled study that evaluated the efficacy and safety of riluzole in patients with ALS. The study was conducted at 124 sites in 14 countries, and 112 patients were randomized to receive riluzole or placebo. The primary endpoint was the change in the Expanded Disability Status Scale (EDSS) score from baseline to 12 months, and secondary endpoints included changes in other clinical outcomes. The results showed that riluzole significa...

**Expected answer:** The pivotal trial leading to riluzole's FDA approval was published by Bensimon et al. in 1994 in the New England Journal of Medicine. It was a randomized, double-blind, placebo-controlled trial involving 155 ALS patients. Riluzole (100 mg/day) showed a statistically significant improvement in tracheostomy-free survival at 12 months compared to placebo, with a more pronounced benefit in patients with bulbar-onset disease.
**Accuracy:** 0.50
**Failure mode:** confident fabrication

The model partially addressed 2 of 4 key facts about clinical trials. It matched "new england journal of medicine" but missed "bensimon 1994". It also produced 9 potentially fabricated entities. This represents the model's strongest performance in clinical trials.

#### TRIAL-010: What is the role of neurofilament light chain as a biomarker...

**Question:** What is the role of neurofilament light chain as a biomarker in ALS trials?

>  NfL is used as a biomarker in ALS clinical trials because it is a sensitive and specific marker of axonal degeneration, which is a hallmark of ALS. It is measured in cerebrospinal fluid (CSF) and plasma, and its levels are correlated with clinical progression and disease severity. This makes it a valuable tool for monitoring disease progression and evaluating treatment efficacy in clinical trials. Additionally, NfL levels can be used as a biomarker for early diagnosis of ALS, as it is elevat...

**Expected answer:** Neurofilament light chain (NfL) is a structural protein released into cerebrospinal fluid (CSF) and blood when axons degenerate. In ALS, NfL levels are markedly elevated and correlate with disease progression rate and survival. NfL is increasingly used in clinical trials as a pharmacodynamic biomarker to assess whether treatments reduce neurodegeneration. The tofersen VALOR trial demonstrated that NfL reduction correlated with SOD1 protein lowering. NfL may eventually serve as a surrogate endpoint for accelerated drug approval.
**Accuracy:** 0.50
**Failure mode:** confident fabrication

The model partially addressed 2 of 4 key facts about clinical trials. It matched "axonal degeneration marker" but missed "elevated in csf and blood". It also produced 2 potentially fabricated entities. This represents the model's strongest performance in clinical trials.

#### MECH-010: How does the PLK4 kinase pathway drive motor neuron death in...

**Question:** How does the PLK4 kinase pathway drive motor neuron death in ALS?

>  The PLK4 kinase signaling cascade is a key pathway involved in ALS motor neuron degeneration, where PLK4 phosphorylates and activates the tumor suppressor protein p53, leading to its translocation to the nucleus and subsequent transcriptional activation of genes involved in cell cycle arrest and apoptosis. This cascade is essential for the progression of ALS motor neuron degeneration.

**Expected answer:** There is no established role for the PLK4 (Polo-like kinase 4) pathway in ALS motor neuron death. PLK4 is primarily known for its role in centriole duplication during cell division, not neurodegeneration. This is a fabricated mechanism. The actual major pathways implicated in ALS motor neuron death include excitotoxicity, oxidative stress, protein aggregation (TDP-43, SOD1), impaired autophagy, and neuroinflammation.
**Accuracy:** 0.33
**Failure mode:** confident fabrication

The model failed to adequately address 1 of 3 key facts about disease mechanisms. It matched "plk4 not involved in als" but missed "fabricated mechanism". It also produced 1 potentially fabricated entity. This represents the model's strongest performance in disease mechanisms.

### Worst responses

10 samples selected for this category.

#### DRUG-004: What medication is used to treat pseudobulbar affect in ALS ...

**Question:** What medication is used to treat pseudobulbar affect in ALS patients?

>  The drug is used to treat depression and anxiety, and it is also used to treat other conditions such as bipolar disorder and schizophrenia. The drug is also used to treat other conditions such as bipolar disorder and schizophrenia. The drug is also used to treat other conditions such as bipolar disorder and schizophrenia. The drug is also used to treat other conditions such as bipolar disorder and schizophrenia. The drug is also used to treat other conditions such as bipolar disorder and sch...

**Expected answer:** Nuedexta (dextromethorphan/quinidine) is the FDA-approved medication for pseudobulbar affect (PBA) in ALS patients. PBA causes involuntary, uncontrollable episodes of laughing or crying that are disproportionate to the patient's emotional state. Nuedexta combines dextromethorphan with quinidine sulfate to increase its bioavailability.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "nuedexta". This illustrates degenerate where the model produced confidently wrong output about drug treatment.

#### DRUG-007: What is tofersen and for which ALS population is it indicate...

**Question:** What is tofersen and for which ALS population is it indicated?

>  tofersen is a recombinant humanized monoclonal antibody that targets the S1P1 receptor, which is expressed on the surface of oligodendrocytes and microglia. It is used to treat patients with relapsing-remitting multiple sclerosis (RRMS) who have failed at least one disease-modifying therapy (DMT). The mechanism of action involves blocking the S1P1 receptor, which is involved in the pathogenesis of MS. The specific ALS patient population it targets is not mentioned in the passage. However, it...

**Expected answer:** Tofersen (brand name Qalsody) is an antisense oligonucleotide (ASO) that targets SOD1 mRNA to reduce production of the toxic mutant SOD1 protein. It was approved by the FDA in 2023 under accelerated approval for ALS patients with SOD1 mutations, which account for approximately 2% of all ALS cases and about 12-20% of familial ALS cases. It is administered intrathecally.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "antisense oligonucleotide". It also produced 3 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about drug treatment.

#### DRUG-008: What are common side effects of riluzole?

**Question:** What are common side effects of riluzole?

>  The passage does not explicitly mention any adverse effects of riluzole use in ALS patients, but it does state that riluzole is used to treat ALS. Therefore, it is not possible to determine the specific adverse effects associated with riluzole use in ALS patients based solely on the information provided. However, it is worth noting that riluzole is generally well-tolerated, and its use is associated with a low risk of adverse effects. The passage does not provide any information about the sp...

**Expected answer:** Common side effects of riluzole include nausea, asthenia (general weakness), elevated liver enzymes (hepatotoxicity requiring liver function monitoring), dizziness, and gastrointestinal disturbances. Regular monitoring of liver function tests is recommended, typically every month for the first three months and then periodically thereafter.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "nausea". It also produced 2 potentially fabricated entities. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

#### DRUG-009: What is AMX0035 and what was the outcome of the CENTAUR tria...

**Question:** What is AMX0035 and what was the outcome of the CENTAUR trial?

>  AMX0035 is a novel, oral, small molecule inhibitor of the mTORC1 pathway, which was evaluated in the CENTAUR clinical trial for ALS. The results showed that AMX0035 was well tolerated and had a favorable safety profile, with no serious adverse events reported. The drug demonstrated a statistically significant reduction in disease progression compared to placebo, with a median progression-free survival of 12.5 months. The study also found that AMX0035 had a favorable safety profile, with no s...

**Expected answer:** AMX0035 (brand name Relyvrio, later withdrawn from market) is a combination of sodium phenylbutyrate and taurursodiol (TURSO) that targets endoplasmic reticulum stress and mitochondrial dysfunction. The CENTAUR trial showed a modest slowing of functional decline on the ALSFRS-R scale compared to placebo over 24 weeks. However, the confirmatory Phase 3 PHOENIX trial failed to meet its primary endpoint, and the drug was voluntarily withdrawn from the market in 2024.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "sodium phenylbutyrate". It also produced 1 potentially fabricated entity. This illustrates degenerate where the model produced confidently wrong output about drug treatment.

#### DRUG-010: What is masitinib and what is its proposed mechanism in ALS?

**Question:** What is masitinib and what is its proposed mechanism in ALS?

>  Masitinib is a small molecule inhibitor of the tyrosine kinase receptor c-Kit, which is proposed to be a promising therapeutic agent for the treatment of amyotrophic lateral sclerosis (ALS). The passage suggests that masitinib may be effective in treating ALS by targeting the underlying molecular mechanisms of the disease. However, further research is needed to confirm its efficacy and safety in clinical trials. Masitinib is a promising candidate for ALS treatment due to its potential to inh...

**Expected answer:** Masitinib is not a real approved ALS treatment. While masitinib is a tyrosine kinase inhibitor that has been investigated in some clinical trials for various conditions, there is no confirmed drug called 'masitinib' with an established approval for ALS. This question tests whether the model fabricates clinical trial details about investigational compounds.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "tyrosine kinase inhibitor". It also produced 2 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about drug treatment.

#### GENE-002: What is the SOD1 gene and how do its mutations cause ALS?

**Question:** What is the SOD1 gene and how do its mutations cause ALS?

>  SOD1 mutations are associated with a 10-15% risk of developing ALS, highlighting their significant contribution to the disease's etiology. The passage suggests that these mutations may be responsible for the majority of ALS cases, indicating their central role in the pathogenesis of the disease. The passage also mentions that SOD1 mutations are not present in 80% of patients, implying that other factors may contribute to the development of ALS. Overall, SOD1 mutations play a crucial role in ...

**Expected answer:** SOD1 (superoxide dismutase 1) encodes a copper-zinc superoxide dismutase enzyme that converts superoxide radicals to hydrogen peroxide and oxygen. Over 180 different mutations in SOD1 have been identified in ALS patients, accounting for about 12-20% of familial ALS and 1-2% of sporadic ALS. The mutations cause disease primarily through a toxic gain-of-function mechanism involving protein misfolding and aggregation, rather than loss of enzymatic activity.
**Accuracy:** 0.00
**Failure mode:** plausible blending

The model failed to identify any of 0 of 4 key facts about gene mutation. It missed "superoxide dismutase 1". It also produced 1 potentially fabricated entity. This illustrates plausible blending where the model produced confidently wrong output about gene mutation.

#### GENE-004: What is the FUS gene and how do its mutations contribute to ...

**Question:** What is the FUS gene and how do its mutations contribute to ALS?

>  FUS is a protein that plays a role in the transport of RNA from the nucleus to the cytoplasm. FUS mutations lead to ALS by disrupting this process, resulting in the accumulation of toxic RNA aggregates in motor neurons. This disruption impairs the function of motor neurons, leading to progressive muscle weakness and paralysis. The exact mechanism by which FUS mutations cause ALS is not fully understood, but it is believed to involve the accumulation of toxic RNA aggregates in motor neurons. ...

**Expected answer:** FUS (fused in sarcoma) encodes an RNA-binding protein involved in transcription, RNA splicing, and DNA repair. Mutations in FUS account for approximately 4-5% of familial ALS cases and are often associated with younger age of onset. FUS mutations cause cytoplasmic mislocalization of the protein, leading to formation of stress granules and toxic aggregates. Unlike most ALS cases, FUS-ALS typically does not show TDP-43 pathology but instead shows FUS-positive inclusions.
**Accuracy:** 0.00
**Failure mode:** plausible blending

The model failed to identify any of 0 of 4 key facts about gene mutation. It missed "rna-binding protein". It also produced 2 potentially fabricated entities. This illustrates plausible blending where the model produced confidently wrong output about gene mutation.

#### GENE-005: What percentage of ALS cases are familial versus sporadic?

**Question:** What percentage of ALS cases are familial versus sporadic?

>  The proportion of ALS cases that are familial (inherited) is 10%, whereas the proportion of sporadic cases is 90%. This means that only 10% of all ALS cases are inherited, while the remaining 90% are not inherited. The passage does not provide any information about the specific genetic causes of familial ALS, but it is known that familial ALS is caused by mutations in specific genes. The passage also mentions that the majority of ALS cases are sporadic, with no clear indication of a specific...

**Expected answer:** Approximately 5-10% of ALS cases are familial (inherited), while 90-95% are sporadic with no clear family history. However, the boundary between familial and sporadic ALS has become blurred as genetic testing has identified disease-causing mutations in some apparently sporadic cases, particularly C9orf72 repeat expansions.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about gene mutation. It missed "5-10% familial". It also produced 1 potentially fabricated entity. This illustrates degenerate where the model produced confidently wrong output about gene mutation.

#### GENE-007: How does the C9orf72 repeat expansion cause disease?

**Question:** How does the C9orf72 repeat expansion cause disease?

>  The C9orf72 hexanucleotide repeat expansion is associated with a 50% increase in the risk of developing ALS, but the underlying molecular mechanisms are not fully understood. The expansion of the C9orf72 repeat is linked to a 50% increase in the risk of developing ALS, but the underlying molecular mechanisms are not fully understood. The expansion of the C9orf72 repeat is linked to a 50% increase in the risk of developing ALS, but the underlying molecular mechanisms are not fully understood....

**Expected answer:** The C9orf72 GGGGCC repeat expansion causes disease through three proposed mechanisms: (1) loss of function due to reduced C9orf72 protein expression from the expanded allele, affecting autophagy and endosomal trafficking; (2) RNA toxicity from repeat-containing RNA transcripts that form nuclear RNA foci and sequester RNA-binding proteins; and (3) toxic dipeptide repeat (DPR) proteins produced by repeat-associated non-ATG (RAN) translation of the expanded repeat in all reading frames.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about gene mutation. It missed "loss of function". It also produced 1 potentially fabricated entity. This illustrates degenerate where the model produced confidently wrong output about gene mutation.

#### GENE-009: What is the BRCA3 gene's role in ALS pathogenesis?

**Question:** What is the BRCA3 gene's role in ALS pathogenesis?

>  BRCA3 is a tumor suppressor gene that plays a role in regulating cell cycle progression and DNA repair. It contributes to ALS by promoting the accumulation of mutant proteins, which can lead to neuronal dysfunction and death. The exact mechanism by which BRCA3 contributes to ALS is not fully understood, but it is believed to involve its role in regulating protein stability and function.

**Expected answer:** There is no gene called BRCA3 that is associated with ALS. The BRCA gene family (BRCA1, BRCA2) is associated with breast and ovarian cancer susceptibility, not ALS. This is a fabricated association. The major ALS-associated genes include SOD1, C9orf72, TARDBP, FUS, and TBK1.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 3 key facts about gene mutation. It missed "no such gene associated with als". It also produced 1 potentially fabricated entity. This illustrates confident fabrication where the model produced confidently wrong output about gene mutation.

### Edge cases

10 samples selected for this category.

#### GENE-003: What is the TARDBP gene and what protein does it encode?

**Question:** What is the TARDBP gene and what protein does it encode?

>  The TARDBP gene encodes a protein called TAR DNA-binding protein 43 (TDP-43), which is involved in RNA processing and splicing. The protein is essential for the survival of motor neurons and is associated with the pathogenesis of ALS. The exact mechanism of TDP-43's role in ALS is not fully understood, but it is believed to contribute to the progressive degeneration of motor neurons. The protein is also involved in the regulation of gene expression and has been implicated in the pathogenesis...

**Expected answer:** TARDBP encodes the TAR DNA-binding protein 43 (TDP-43), a nuclear RNA-binding protein involved in RNA processing, splicing, and transport. Mutations in TARDBP account for approximately 4-5% of familial ALS cases. TDP-43 pathology (cytoplasmic mislocalization and aggregation) is the hallmark pathological finding in approximately 97% of all ALS cases, regardless of genetic cause.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about gene mutation. It matched "tdp-43" but missed "cytoplasmic mislocalization". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about gene mutation but missed critical details.

#### GENE-008: What protein does the SOD1 gene encode?

**Question:** What protein does the SOD1 gene encode?

>  The protein product of the SOD1 gene is superoxide dismutase 1 (SOD1), which is a key antioxidant enzyme involved in the regulation of cellular redox homeostasis. It plays a crucial role in maintaining the balance between reactive oxygen species (ROS) and antioxidants in cells. The normal function of SOD1 is to protect cells from oxidative stress and damage caused by ROS. However, mutations in the SOD1 gene can lead to familial amyotrophic lateral sclerosis (FALS), a progressive neurodegener...

**Expected answer:** The SOD1 gene encodes superoxide dismutase 1 (also known as Cu/Zn superoxide dismutase), a cytoplasmic enzyme that catalyzes the conversion of superoxide radicals (O2-) into hydrogen peroxide (H2O2) and molecular oxygen. This enzyme is a critical component of the cell's antioxidant defense system.
**Accuracy:** 0.50
**Failure mode:** confident fabrication

The model partially addressed 2 of 4 key facts about gene mutation. It matched "superoxide dismutase 1" but missed "copper-zinc". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about gene mutation but missed critical details.

#### DIAG-003: What are upper motor neuron signs in ALS?

**Question:** What are upper motor neuron signs in ALS?

>  The clinical signs of upper motor neuron involvement in ALS include spasticity, hyperreflexia, and clonus. These signs are typically present in the limbs and trunk, and may be accompanied by weakness and ataxia. The patient may also experience autonomic dysfunction, such as urinary retention or incontinence. Additionally, patients may present with sensory disturbances, such as numbness or tingling, and may have difficulty with speech and swallowing. The clinical presentation of upper motor n...

**Expected answer:** Upper motor neuron (UMN) signs in ALS include hyperreflexia (exaggerated deep tendon reflexes), spasticity (increased muscle tone), pathological reflexes such as the Babinski sign (extensor plantar response), Hoffmann sign, clonus, and pseudobulbar affect (emotional lability). These signs result from degeneration of corticospinal and corticobulbar motor neurons.
**Accuracy:** 0.50
**Failure mode:** accurate

The model partially addressed 2 of 4 key facts about diagnostic criteria. It matched "hyperreflexia" but missed "babinski sign". It also produced 3 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about diagnostic criteria but missed critical details.

#### DIAG-004: What are lower motor neuron signs in ALS?

**Question:** What are lower motor neuron signs in ALS?

>  The clinical signs of lower motor neuron involvement in ALS include flaccid paralysis, muscle weakness, and muscle atrophy. These symptoms are typically present in the proximal muscles of the limbs and trunk, and may be accompanied by sensory loss. Additionally, patients may experience autonomic dysfunction, such as urinary and bowel dysfunction. The clinical presentation of lower motor neuron involvement in ALS can vary depending on the extent and severity of the underlying motor neuron pat...

**Expected answer:** Lower motor neuron (LMN) signs in ALS include muscle weakness, muscle atrophy (wasting), fasciculations (visible muscle twitching), hyporeflexia or areflexia (reduced or absent deep tendon reflexes), and muscle cramps. These signs result from degeneration of anterior horn cells in the spinal cord and motor nuclei in the brainstem.
**Accuracy:** 0.50
**Failure mode:** accurate

The model partially addressed 2 of 4 key facts about diagnostic criteria. It matched "muscle weakness" but missed "fasciculations". It also produced 2 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about diagnostic criteria but missed critical details.

#### TRIAL-004: What was the pivotal clinical trial that led to riluzole's F...

**Question:** What was the pivotal clinical trial that led to riluzole's FDA approval?

>  The clinical trial was a randomized, double-blind, placebo-controlled study that evaluated the efficacy and safety of riluzole in patients with ALS. The study was conducted at 124 sites in 14 countries, and 112 patients were randomized to receive riluzole or placebo. The primary endpoint was the change in the Expanded Disability Status Scale (EDSS) score from baseline to 12 months, and secondary endpoints included changes in other clinical outcomes. The results showed that riluzole significa...

**Expected answer:** The pivotal trial leading to riluzole's FDA approval was published by Bensimon et al. in 1994 in the New England Journal of Medicine. It was a randomized, double-blind, placebo-controlled trial involving 155 ALS patients. Riluzole (100 mg/day) showed a statistically significant improvement in tracheostomy-free survival at 12 months compared to placebo, with a more pronounced benefit in patients with bulbar-onset disease.
**Accuracy:** 0.50
**Failure mode:** confident fabrication

The model partially addressed 2 of 4 key facts about clinical trials. It matched "new england journal of medicine" but missed "bensimon 1994". It also produced 9 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about clinical trials but missed critical details.

#### TRIAL-010: What is the role of neurofilament light chain as a biomarker...

**Question:** What is the role of neurofilament light chain as a biomarker in ALS trials?

>  NfL is used as a biomarker in ALS clinical trials because it is a sensitive and specific marker of axonal degeneration, which is a hallmark of ALS. It is measured in cerebrospinal fluid (CSF) and plasma, and its levels are correlated with clinical progression and disease severity. This makes it a valuable tool for monitoring disease progression and evaluating treatment efficacy in clinical trials. Additionally, NfL levels can be used as a biomarker for early diagnosis of ALS, as it is elevat...

**Expected answer:** Neurofilament light chain (NfL) is a structural protein released into cerebrospinal fluid (CSF) and blood when axons degenerate. In ALS, NfL levels are markedly elevated and correlate with disease progression rate and survival. NfL is increasingly used in clinical trials as a pharmacodynamic biomarker to assess whether treatments reduce neurodegeneration. The tofersen VALOR trial demonstrated that NfL reduction correlated with SOD1 protein lowering. NfL may eventually serve as a surrogate endpoint for accelerated drug approval.
**Accuracy:** 0.50
**Failure mode:** confident fabrication

The model partially addressed 2 of 4 key facts about clinical trials. It matched "axonal degeneration marker" but missed "elevated in csf and blood". It also produced 2 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about clinical trials but missed critical details.

#### MECH-010: How does the PLK4 kinase pathway drive motor neuron death in...

**Question:** How does the PLK4 kinase pathway drive motor neuron death in ALS?

>  The PLK4 kinase signaling cascade is a key pathway involved in ALS motor neuron degeneration, where PLK4 phosphorylates and activates the tumor suppressor protein p53, leading to its translocation to the nucleus and subsequent transcriptional activation of genes involved in cell cycle arrest and apoptosis. This cascade is essential for the progression of ALS motor neuron degeneration.

**Expected answer:** There is no established role for the PLK4 (Polo-like kinase 4) pathway in ALS motor neuron death. PLK4 is primarily known for its role in centriole duplication during cell division, not neurodegeneration. This is a fabricated mechanism. The actual major pathways implicated in ALS motor neuron death include excitotoxicity, oxidative stress, protein aggregation (TDP-43, SOD1), impaired autophagy, and neuroinflammation.
**Accuracy:** 0.33
**Failure mode:** confident fabrication

The model failed to adequately address 1 of 3 key facts about disease mechanisms. It matched "plk4 not involved in als" but missed "fabricated mechanism". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about disease mechanisms but missed critical details.

#### TEMP-001: When was riluzole approved by the FDA for ALS treatment?

**Question:** When was riluzole approved by the FDA for ALS treatment?

>  Riluzole received FDA approval for treating ALS in 1995. This approval was based on its ability to prolong life and reduce the progression of symptoms in patients with ALS. The drug has since been used to treat patients with ALS, although its effectiveness is not guaranteed. The passage does not mention any other year of approval for riluzole. The passage states that riluzole was approved for treating ALS in 1995, but it does not provide information about any other years of approval. Therefo...

**Expected answer:** Riluzole was approved by the FDA in 1995, making it the first drug specifically approved for the treatment of ALS. It had previously been studied in clinical trials by Bensimon et al. (published 1994) and Lacomblez et al. (published 1996).
**Accuracy:** 0.33
**Failure mode:** degenerate

The model failed to adequately address 1 of 3 key facts about temporal accuracy. It matched "1995" but missed "first fda-approved als drug". It also produced 2 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about temporal accuracy but missed critical details.

#### CARE-010: What is neurostimulation therapy for ALS and when is it reco...

**Question:** What is neurostimulation therapy for ALS and when is it recommended?

>  Neurostimulation therapy for ALS is a promising treatment option, but its efficacy and safety have not been fully established. The passage suggests that more research is needed to determine the long-term effects of this therapy. The passage also mentions that the use of neurostimulation therapy for ALS is not yet widely accepted, and more studies are needed to evaluate its effectiveness and safety. Overall, the passage indicates that neurostimulation therapy for ALS is an emerging treatment ...

**Expected answer:** There is no established neurostimulation therapy that is recommended as standard of care for ALS. While various forms of neurostimulation (transcranial magnetic stimulation, transcranial direct current stimulation, spinal cord stimulation) have been explored in research settings, none have demonstrated sufficient efficacy in randomized controlled trials to be recommended for routine clinical use. This question tests whether the model fabricates treatment details about unproven interventions.
**Accuracy:** 0.33
**Failure mode:** confident fabrication

The model failed to adequately address 1 of 3 key facts about patient care. It matched "no established neurostimulation therapy for als" but missed "research only". It also produced 2 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about patient care but missed critical details.

#### DRUG-020: What is cortexilin and how does it protect motor neurons in ...

**Question:** What is cortexilin and how does it protect motor neurons in ALS?

>  Cortexilin's neuroprotective mechanism involves its ability to inhibit the activity of the NLRP3 inflammasome, which is associated with the pathogenesis of ALS. The clinical evidence suggests that cortexilin may have a potential role in the treatment of ALS. However, further research is needed to confirm its efficacy and safety. The passage does not provide specific details about the mechanism or clinical evidence, but it mentions that the mechanism involves inhibiting the NLRP3 inflammasome...

**Expected answer:** Cortexilin is not a real drug. There is no pharmaceutical compound called cortexilin that has been developed or tested for ALS or any other neurological condition. This is a fabricated drug name designed to test whether the model invents clinical evidence for non-existent compounds. The actual FDA-approved ALS drugs are riluzole, edaravone, and tofersen (for SOD1-ALS).
**Accuracy:** 0.33
**Failure mode:** confident fabrication

The model failed to adequately address 1 of 3 key facts about drug treatment. It matched "no clinical evidence exists" but missed "cortexilin is not a real drug". It also produced 2 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

---

## Disclaimer

This report is a research artifact produced by the ALS-LM hallucination evaluation framework. The model evaluated in this report is not a medical tool and should never be used for medical decision-making. The evaluation framework exists to quantify the model's unreliability and characterize its failure modes for research purposes.

*Generated: 2026-03-22 08:35:22 UTC*
