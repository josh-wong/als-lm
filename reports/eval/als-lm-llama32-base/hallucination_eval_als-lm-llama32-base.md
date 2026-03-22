# ALS-LM hallucination evaluation report

This report presents the results of the ALS-LM hallucination evaluation framework, which systematically measures factual accuracy, fabrication tendencies, and failure modes of a domain-specific language model trained on ALS research literature.

## Metadata

- **Report generated:** 2026-03-21 16:50:51 UTC
- **Model:** `als-lm-llama32-base`
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
| Mean accuracy    | 0.1031 |
| Median accuracy  | 0.0000 |
| Binary pass rate | 0.0813 |
| Total questions  |    160 |

### By category

Accuracy broken down by benchmark question category.

| Category                 | Count | Mean   | Median | Pass rate |
| ------------------------ | ----- | ------ | ------ | --------- |
| clinical trials          |    20 | 0.0500 | 0.0000 | 0.0000    |
| diagnostic criteria      |    20 | 0.1625 | 0.0000 | 0.1500    |
| disease mechanisms       |    20 | 0.0375 | 0.0000 | 0.0000    |
| drug treatment           |    20 | 0.1875 | 0.1250 | 0.1500    |
| epidemiology             |    20 | 0.0125 | 0.0000 | 0.0000    |
| gene mutation            |    20 | 0.1500 | 0.0000 | 0.1500    |
| patient care             |    20 | 0.0708 | 0.0000 | 0.0500    |
| temporal accuracy        |    20 | 0.1542 | 0.0000 | 0.1500    |

### By difficulty

Accuracy broken down by question difficulty level.

| Difficulty | Count | Mean   | Median | Pass rate |
| ---------- | ----- | ------ | ------ | --------- |
| easy       |    38 | 0.1601 | 0.0000 | 0.1579    |
| hard       |    53 | 0.0613 | 0.0000 | 0.0189    |
| medium     |    69 | 0.1039 | 0.0000 | 0.0870    |

### Trap question performance

Trap questions contain fabricated entities or misleading premises to test the model's tendency to agree with incorrect information.

| Metric           | Value  |
| ---------------- | ------ |
| Count            |     16 |
| Mean accuracy    | 0.0573 |
| Binary pass rate | 0.0000 |

## Failure taxonomy distribution

Each response is classified into one of five failure modes (plus accurate and degenerate categories) using rule-based logic.

```
  confident_fabrication        ######### (28)
  plausible_blending           ###### (17)
  outdated_information          (1)
  boundary_confusion            (0)
  accurate_but_misleading       (0)
  accurate                      (1)
  degenerate                   ######################################## (113)
```

| Failure mode             | Count | Pct    | High | Medium | Low |
| ------------------------ | ----- | ------ | ---- | ------ | --- |
| confident fabrication    |    28 |  17.5% |   28 |      0 |   0 |
| plausible blending       |    17 |  10.6% |    1 |     16 |   0 |
| outdated information     |     1 |   0.6% |    0 |      1 |   0 |
| boundary confusion       |     0 |   0.0% |    0 |      0 |   0 |
| accurate but misleading  |     0 |   0.0% |    0 |      0 |   0 |
| accurate                 |     1 |   0.6% |    0 |      0 |   0 |
| degenerate               |   113 |  70.6% |    0 |      0 | 113 |

The dominant failure mode is **confident fabrication**, accounting for 28 responses (17.5% of total). 
Overall, 46 of 160 responses (28.7%) exhibit a classified failure mode, while 1 are accurate and 113 are degenerate.

Severity distribution across all responses:

- **High:** 29 (18.1%)
- **Medium:** 17 (10.6%)
- **Low:** 113 (70.6%)
- **None:** 1 (0.6%)

## Fabrication analysis

Entities (drug names, gene names, clinical trial IDs) extracted from model responses are checked against a training-corpus registry. Entities not found in the registry are flagged as potentially fabricated.

### Summary

- **Total entities extracted:** 1807
- **Total flagged:** 1583
- **Flagged rate:** 0.8760

### By entity type

Breakdown of entity extraction and flagging by type.

| Entity type | Extracted | Flagged | Flagged rate |
| ----------- | --------- | ------- | ------------ |
| drugs       |      1612 |    1511 | 0.9373       |
| genes       |       194 |      71 | 0.3660       |
| trials      |         1 |       1 | 1.0000       |

### Top flagged entities

The most frequently flagged entities across all responses.

| Entity           | Type   | Occurrences | Context                          |
| ---------------- | ------ | ----------- | -------------------------------- |
| This             | drug   |          60 | This helps to slow down the prog |
| Amyotrophic      | drug   |          49 | AMX0035 is a drug that is being  |
| Lateral          | drug   |          47 | AMX0035 is a drug that is being  |
| Sclerosis        | drug   |          47 | AMX0035 is a drug that is being  |
| Amyotrophic Late | drug   |          46 | AMX0035 is a drug that is being  |
| However          | drug   |          40 | However, more research is needed |
| These            | drug   |          35 | These drugs are used to slow the |
| Additionally     | drug   |          35 | Additionally, riluzole has been  |
| Overall          | drug   |          27 | Overall, riluzole is a promising |
| Here             | drug   |          23 | Here is a list of FDA-approved d |

## Hedging behavior summary

Hedging language (words and phrases indicating uncertainty such as "may", "possibly", "it is thought that") is detected in model responses to assess the model's confidence calibration.

- **Total hedging instances:** 223
- **Average hedging per response:** 1.39

Most common hedging phrases:

- "may" (72 occurrences)
- "is thought to" (9 occurrences)
- "in some cases" (8 occurrences)
- "potentially" (5 occurrences)
- "likely" (4 occurrences)
- "could" (2 occurrences)
- "is believed to" (2 occurrences)
- "some studies suggest" (1 occurrences)
- "it is possible that" (1 occurrences)

## Qualitative samples

Representative model responses selected by accuracy score. Best responses show the model's strongest performance, worst responses illustrate systematic failures, and edge cases reveal where the model's knowledge fragments at the pass/fail boundary.

### Best responses

10 samples selected for this category.

#### DRUG-001: What drugs are FDA-approved for treating ALS?

**Question:** What drugs are FDA-approved for treating ALS?

>  The FDA has approved several drugs for the treatment of amyotrophic lateral sclerosis (ALS), a progressive neurodegenerative disease that affects the nerve cells in the brain and spinal cord that control voluntary muscle movement. These drugs are used to slow the progression of the disease and improve the quality of life for patients. Here is a list of FDA-approved drugs for ALS treatment, along with the dates they were approved:

1. **Riluzole (Rilutek)**: Approved in 1995
   - Riluzole is ...

**Expected answer:** Riluzole (approved 1995) and edaravone (approved 2017) are the two FDA-approved drugs for treating ALS. Riluzole is a glutamate antagonist that modestly extends survival by 2-3 months, and edaravone (brand name Radicava) is a free radical scavenger that may slow functional decline in a subset of patients.
**Accuracy:** 1.00
**Failure mode:** degenerate

The model correctly identified 4 of 4 key facts about drug treatment. It matched "riluzole". It also produced 6 potentially fabricated entities. This represents the model's strongest performance in drug treatment.

#### DIAG-004: What are lower motor neuron signs in ALS?

**Question:** What are lower motor neuron signs in ALS?

>  Lower motor neuron (LMN) involvement in Amyotrophic Lateral Sclerosis (ALS) is characterized by weakness and atrophy of muscles, particularly those innervated by the spinal cord. Here are the clinical signs of LMN involvement in ALS:

1. Muscle weakness:
- Decreased muscle strength in the limbs and trunk
- Difficulty with fine motor skills, such as writing or buttoning clothes
- Weakness in the hands, arms, and legs
- Difficulty with activities that require coordination, such as walking or c...

**Expected answer:** Lower motor neuron (LMN) signs in ALS include muscle weakness, muscle atrophy (wasting), fasciculations (visible muscle twitching), hyporeflexia or areflexia (reduced or absent deep tendon reflexes), and muscle cramps. These signs result from degeneration of anterior horn cells in the spinal cord and motor nuclei in the brainstem.
**Accuracy:** 1.00
**Failure mode:** degenerate

The model correctly identified 4 of 4 key facts about diagnostic criteria. It matched "muscle weakness". It also produced 21 potentially fabricated entities. This represents the model's strongest performance in diagnostic criteria.

#### GENE-008: What protein does the SOD1 gene encode?

**Question:** What protein does the SOD1 gene encode?

>  The protein product of the SOD1 gene is superoxide dismutase 1 (SOD1). SOD1 is a copper-zinc superoxide dismutase enzyme that plays a crucial role in the body's antioxidant defense system. It is primarily found in the cytoplasm of cells and is involved in the detoxification of superoxide radicals, which are highly reactive oxygen species that can cause cellular damage. SOD1 is particularly important in the central nervous system, where it helps protect neurons from oxidative stress and damag...

**Expected answer:** The SOD1 gene encodes superoxide dismutase 1 (also known as Cu/Zn superoxide dismutase), a cytoplasmic enzyme that catalyzes the conversion of superoxide radicals (O2-) into hydrogen peroxide (H2O2) and molecular oxygen. This enzyme is a critical component of the cell's antioxidant defense system.
**Accuracy:** 0.75
**Failure mode:** confident fabrication

The model partially addressed 3 of 4 key facts about gene mutation. It matched "superoxide dismutase 1" but missed "converts superoxide to hydrogen peroxide". It also produced 1 potentially fabricated entity. This represents the model's strongest performance in gene mutation.

#### TEMP-003: When was ALS first described as a distinct disease?

**Question:** When was ALS first described as a distinct disease?

>  The first clinical description of amyotrophic lateral sclerosis (ALS) was by Jean-Martin Charcot in 1869. Charcot was a French neurologist who was the first to describe the disease as a distinct entity. He described the disease as a progressive disorder of the nervous system that affected both the upper and lower motor neurons. Charcot's description of ALS was based on his observations of a patient who had been suffering from a progressive weakness and atrophy of the limbs, as well as a loss...

**Expected answer:** ALS was first described as a distinct neurological disease by French neurologist Jean-Martin Charcot in 1869. Charcot identified the combination of muscle atrophy (from lower motor neuron loss) with spasticity (from lateral column sclerosis) and correlated clinical findings with pathological examination of the spinal cord, demonstrating sclerosis of the lateral columns. The disease bears the name 'Charcot's disease' in some countries.
**Accuracy:** 0.75
**Failure mode:** degenerate

The model partially addressed 3 of 4 key facts about temporal accuracy. It matched "1869" but missed "lateral column sclerosis". It also produced 5 potentially fabricated entities. This represents the model's strongest performance in temporal accuracy.

#### DRUG-003: What is edaravone and how does it work in treating ALS?

**Question:** What is edaravone and how does it work in treating ALS?

>  Edaravone (Radicava) is a free radical scavenger that is used to treat amyotrophic lateral sclerosis (ALS). It works by reducing the production of harmful free radicals in the brain and spinal cord, which can cause damage to nerve cells and lead to the progression of ALS. Edaravone is taken orally and is absorbed into the bloodstream, where it circulates to the brain and spinal cord. Once in the brain and spinal cord, edaravone binds to free radicals and neutralizes them, preventing them fro...

**Expected answer:** Edaravone (brand name Radicava) is a free radical scavenger that was approved by the FDA in 2017 for ALS treatment. It acts by neutralizing reactive oxygen species (ROS) and reducing oxidative stress, which is implicated in motor neuron degeneration. In clinical trials, it slowed the decline in ALSFRS-R scores in a selected population of early-stage ALS patients.
**Accuracy:** 0.50
**Failure mode:** accurate

The model partially addressed 2 of 4 key facts about drug treatment. It matched "free radical scavenger" but missed "oxidative stress". It also produced 2 potentially fabricated entities. This represents the model's strongest performance in drug treatment.

#### GENE-001: What is the most common genetic cause of familial ALS?

**Question:** What is the most common genetic cause of familial ALS?

>  The most frequently mutated gene in familial ALS cases is the SOD1 gene. Mutations in this gene are responsible for approximately 20% of familial ALS cases. Other genes that have been implicated in familial ALS include C9orf72, TARDBP, and FUS. However, the SOD1 gene remains the most frequently mutated gene in familial ALS cases.

**Expected answer:** The C9orf72 hexanucleotide repeat expansion is the most common genetic cause of familial ALS, accounting for approximately 40% of familial cases and about 5-10% of apparently sporadic cases in European populations. The mutation involves a GGGGCC repeat expansion in the first intron of the C9orf72 gene on chromosome 9p21.
**Accuracy:** 0.50
**Failure mode:** confident fabrication

The model partially addressed 2 of 4 key facts about gene mutation. It matched "c9orf72" but missed "hexanucleotide repeat expansion". It also produced 3 potentially fabricated entities. This represents the model's strongest performance in gene mutation.

#### GENE-003: What is the TARDBP gene and what protein does it encode?

**Question:** What is the TARDBP gene and what protein does it encode?

>  The TARDBP gene encodes a protein called TAR DNA-binding protein 43 (TDP-43). TDP-43 is a protein that is involved in the regulation of gene expression and the processing of RNA. It is also involved in the transport of RNA from the nucleus to the cytoplasm. TDP-43 is a key player in the regulation of RNA processing and is involved in the regulation of gene expression. It is also involved in the transport of RNA from the nucleus to the cytoplasm. TDP-43 is a key player in the regulation of RN...

**Expected answer:** TARDBP encodes the TAR DNA-binding protein 43 (TDP-43), a nuclear RNA-binding protein involved in RNA processing, splicing, and transport. Mutations in TARDBP account for approximately 4-5% of familial ALS cases. TDP-43 pathology (cytoplasmic mislocalization and aggregation) is the hallmark pathological finding in approximately 97% of all ALS cases, regardless of genetic cause.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about gene mutation. It matched "tdp-43" but missed "cytoplasmic mislocalization". It also produced 1 potentially fabricated entity. This represents the model's strongest performance in gene mutation.

#### DIAG-003: What are upper motor neuron signs in ALS?

**Question:** What are upper motor neuron signs in ALS?

>  Upper motor neuron (UMN) involvement in Amyotrophic Lateral Sclerosis (ALS) typically presents with a distinct set of clinical signs and symptoms. Here are the key clinical features:

1. **Spastic Paralysis**: The most prominent feature is spasticity, which can be generalized or focal. Spasticity is characterized by increased muscle tone, stiffness, and resistance to passive movement.

2. **Hyporeflexia**: This is the opposite of spasticity and refers to decreased muscle tone and reduced res...

**Expected answer:** Upper motor neuron (UMN) signs in ALS include hyperreflexia (exaggerated deep tendon reflexes), spasticity (increased muscle tone), pathological reflexes such as the Babinski sign (extensor plantar response), Hoffmann sign, clonus, and pseudobulbar affect (emotional lability). These signs result from degeneration of corticospinal and corticobulbar motor neurons.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about diagnostic criteria. It matched "hyperreflexia" but missed "babinski sign". It also produced 27 potentially fabricated entities. This represents the model's strongest performance in diagnostic criteria.

#### TEMP-008: When did the Ice Bucket Challenge raise awareness and fundin...

**Question:** When did the Ice Bucket Challenge raise awareness and funding for ALS?

>  The Ice Bucket Challenge was a viral social media campaign that began in 2014. It involved participants pouring a bucket of ice water over their heads and challenging their friends and family to do the same. The challenge was designed to raise awareness and funds for the ALS Association, which is dedicated to finding a cure for amyotrophic lateral sclerosis (ALS), also known as Lou Gehrig's disease.

The Ice Bucket Challenge quickly gained popularity and became a global phenomenon, with mill...

**Expected answer:** The ALS Ice Bucket Challenge went viral in the summer of 2014, becoming one of the largest social media fundraising campaigns in history. The challenge raised over $115 million for the ALS Association in the United States alone between July and August 2014, and over $220 million worldwide. The funds contributed to research breakthroughs including the identification of the NEK1 gene as an ALS risk factor.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about temporal accuracy. It matched "over $115 million raised" but missed "summer 2014". It also produced 12 potentially fabricated entities. This represents the model's strongest performance in temporal accuracy.

#### CARE-004: What augmentative and alternative communication devices are ...

**Question:** What augmentative and alternative communication devices are used for ALS patients?

>  ALS patients who lose speech may use a variety of communication aids and assistive technology to communicate. Some common options include:

1. Speech-generating devices (SGDs): These are electronic devices that help people with speech difficulties to communicate. They can be programmed to generate text, speech, or both. Examples include the Dragon NaturallySpeaking, Dragon Medical Pro, and Dragon Medical Plus.

2. Augmentative and alternative communication (AAC) boards: These are visual comm...

**Expected answer:** Augmentative and alternative communication (AAC) devices for ALS patients include low-tech options (letter boards, picture boards) and high-tech options (speech-generating devices, eye-tracking systems, brain-computer interfaces). Eye-gaze tracking technology allows patients with severe motor impairment to select letters or words on a screen. Voice banking (recording one's voice before speech deteriorates) allows personalized synthetic speech. Brain-computer interfaces (BCIs) are in development for patients who lose all voluntary movement.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about patient care. It matched "eye-tracking systems" but missed "voice banking". It also produced 22 potentially fabricated entities. This represents the model's strongest performance in patient care.

### Worst responses

10 samples selected for this category.

#### DRUG-002: What is the mechanism of action of riluzole in ALS?

**Question:** What is the mechanism of action of riluzole in ALS?

>  Riluzole is a drug that is used to treat amyotrophic lateral sclerosis (ALS), also known as Lou Gehrig's disease. It works by inhibiting the activity of the enzyme glutamine synthetase, which is involved in the production of glutamine, a neurotransmitter that is important for the function of motor neurons in the brain and spinal cord. By inhibiting glutamine synthetase, riluzole reduces the amount of glutamine available to the motor neurons, which can help to slow the progression of ALS. Add...

**Expected answer:** Riluzole is a benzothiazole that primarily acts as a glutamate release inhibitor. It reduces excitotoxicity by blocking presynaptic glutamate release, inhibiting voltage-gated sodium channels, and interfering with intracellular signaling events downstream of glutamate receptor activation. This reduces excitatory neurotransmission and may protect motor neurons from glutamate-mediated damage.
**Accuracy:** 0.00
**Failure mode:** plausible blending

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "glutamate release inhibitor". It also produced 5 potentially fabricated entities. This illustrates plausible blending where the model produced confidently wrong output about drug treatment.

#### DRUG-004: What medication is used to treat pseudobulbar affect in ALS ...

**Question:** What medication is used to treat pseudobulbar affect in ALS patients?

>  Answer: Ropinirole.
A. Ropinirole
B. Levodopa
C. Carbidopa
D. Bromocriptine
Answer: A

The main clinical manifestation of Parkinson's disease is ____
A. Resting tremor
B. Muscle stiffness
C. Reduced movement
D. Postural instability
E. All of the above
Answer: E

The main clinical manifestation of Parkinson's disease is ____
A. Resting tremor
B. Muscle stiffness
C. Reduced movement
D. Postural instability
E. All of the above
Answer: E

The main clinical manifestation of Parkinson's disease is...

**Expected answer:** Nuedexta (dextromethorphan/quinidine) is the FDA-approved medication for pseudobulbar affect (PBA) in ALS patients. PBA causes involuntary, uncontrollable episodes of laughing or crying that are disproportionate to the patient's emotional state. Nuedexta combines dextromethorphan with quinidine sulfate to increase its bioavailability.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "nuedexta". It also produced 9 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about drug treatment.

#### DRUG-007: What is tofersen and for which ALS population is it indicate...

**Question:** What is tofersen and for which ALS population is it indicated?

>  Tofersen is a drug that targets the protein TDP-43, which is overexpressed in ALS patients. It works by binding to TDP-43 and preventing its aggregation, which is a hallmark of ALS. Tofersen is primarily used to treat ALS patients with a specific genetic mutation, known as the C9orf72 mutation. This mutation is the most common genetic cause of ALS and is associated with a higher risk of developing other neurodegenerative diseases, such as frontotemporal dementia. Tofersen has shown promising...

**Expected answer:** Tofersen (brand name Qalsody) is an antisense oligonucleotide (ASO) that targets SOD1 mRNA to reduce production of the toxic mutant SOD1 protein. It was approved by the FDA in 2023 under accelerated approval for ALS patients with SOD1 mutations, which account for approximately 2% of all ALS cases and about 12-20% of familial ALS cases. It is administered intrathecally.
**Accuracy:** 0.00
**Failure mode:** confident fabrication

The model failed to identify any of 0 of 4 key facts about drug treatment. It missed "antisense oligonucleotide". It also produced 3 potentially fabricated entities. This illustrates confident fabrication where the model produced confidently wrong output about drug treatment.

#### GENE-007: How does the C9orf72 repeat expansion cause disease?

**Question:** How does the C9orf72 repeat expansion cause disease?

>  The C9orf72 hexanucleotide repeat expansion is a genetic mutation that can cause amyotrophic lateral sclerosis (ALS) and frontotemporal dementia (FTD). The molecular mechanisms by which this mutation leads to ALS are complex and involve several steps:

1. Expansion of the C9orf72 repeat: The C9orf72 gene contains a hexanucleotide repeat expansion, which is a sequence of six nucleotides (AAGCTT) repeated multiple times. In the vast majority of cases, this expansion is 100-120 repeats, but it ...

**Expected answer:** The C9orf72 GGGGCC repeat expansion causes disease through three proposed mechanisms: (1) loss of function due to reduced C9orf72 protein expression from the expanded allele, affecting autophagy and endosomal trafficking; (2) RNA toxicity from repeat-containing RNA transcripts that form nuclear RNA foci and sequester RNA-binding proteins; and (3) toxic dipeptide repeat (DPR) proteins produced by repeat-associated non-ATG (RAN) translation of the expanded repeat in all reading frames.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about gene mutation. It missed "loss of function". It also produced 6 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about gene mutation.

#### GENE-009: What is the BRCA3 gene's role in ALS pathogenesis?

**Question:** What is the BRCA3 gene's role in ALS pathogenesis?

>  BRCA3 is a gene that is involved in the repair of damaged DNA. It is a member of the BRCA gene family, which is known to be involved in the repair of DNA damage. BRCA3 is located on the X chromosome and is involved in the repair of DNA damage in the cell nucleus. It is also involved in the regulation of cell growth and division. BRCA3 is a tumor suppressor gene, which means that it helps to prevent the development of cancer by repairing damaged DNA and preventing the growth of abnormal cells...

**Expected answer:** There is no gene called BRCA3 that is associated with ALS. The BRCA gene family (BRCA1, BRCA2) is associated with breast and ovarian cancer susceptibility, not ALS. This is a fabricated association. The major ALS-associated genes include SOD1, C9orf72, TARDBP, FUS, and TBK1.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 3 key facts about gene mutation. It missed "no such gene associated with als". It also produced 2 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about gene mutation.

#### GENE-010: What is the genetic relationship between ALS and frontotempo...

**Question:** What is the genetic relationship between ALS and frontotemporal dementia?

>  ALS and FTD are both neurodegenerative diseases that affect the brain and spinal cord. While they share some similarities, they also have distinct genetic and clinical features. Here are some key points to consider:

1. Genetic overlap: Both ALS and FTD are associated with mutations in certain genes, particularly those involved in the regulation of protein synthesis and degradation. Some of the most common genetic mutations in ALS include SOD1, TDP-43, and FUS, while those in FTD include MAP...

**Expected answer:** ALS and frontotemporal dementia (FTD) share significant genetic overlap and are increasingly viewed as a disease spectrum. The C9orf72 repeat expansion is the most common shared genetic cause, found in both ALS and FTD families. Other shared genes include TARDBP, FUS, VCP, UBQLN2, and TBK1. Approximately 15% of ALS patients also develop FTD, and up to 50% show some cognitive or behavioral impairment.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about gene mutation. It missed "disease spectrum". It also produced 15 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about gene mutation.

#### DIAG-001: What are the El Escorial criteria for diagnosing ALS?

**Question:** What are the El Escorial criteria for diagnosing ALS?

>  The El Escorial diagnostic criteria for Amyotrophic Lateral Sclerosis (ALS) are a set of standardized criteria used by the World Health Organization (WHO) and the International Committee for the Definition of Rhabdomyolysis (ICDR) to diagnose ALS. These criteria are based on the presence of motor neuron dysfunction and the absence of other diseases that can mimic ALS.

The El Escorial criteria for ALS include the following:

1. Presence of motor neuron dysfunction: This is the most important...

**Expected answer:** The El Escorial criteria, established by the World Federation of Neurology in 1994 (revised 1998), classify ALS diagnosis into levels based on clinical evidence of upper motor neuron (UMN) and lower motor neuron (LMN) signs across four body regions (bulbar, cervical, thoracic, lumbosacral). The levels are: definite (UMN + LMN signs in 3 regions), probable (UMN + LMN in 2 regions with UMN rostral to LMN), probable laboratory-supported (UMN + LMN in 1 region with EMG evidence), and possible (UMN + LMN in 1 region).
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about diagnostic criteria. It missed "upper motor neuron and lower motor neuron signs". It also produced 24 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about diagnostic criteria.

#### DIAG-002: How do the Awaji criteria differ from the El Escorial criter...

**Question:** How do the Awaji criteria differ from the El Escorial criteria?

>  The Awaji-shima criteria and the revised El Escorial criteria are both used to diagnose Amyotrophic Lateral Sclerosis (ALS), but they differ in several key aspects. Here's a comparison of the two criteria:

1. **Definition of ALS:**
   - **Awaji-shima Criteria:** ALS is defined as a motor neuron disease characterized by the progressive degeneration of motor neurons in the brain and spinal cord.
   - **Revised El Escorial Criteria:** ALS is defined as a motor neuron disease characterized by t...

**Expected answer:** The Awaji-shima criteria (2008) modified the El Escorial criteria by treating electrophysiological evidence (EMG findings) as equivalent to clinical examination for detecting lower motor neuron involvement. This means fasciculation potentials on EMG combined with chronic neurogenic changes are accepted as evidence of LMN degeneration, equal to clinical signs. This change increased diagnostic sensitivity, particularly for bulbar-onset ALS, without significantly reducing specificity.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about diagnostic criteria. It missed "emg equivalent to clinical examination". It also produced 21 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about diagnostic criteria.

#### DIAG-005: What EMG findings are characteristic of ALS?

**Question:** What EMG findings are characteristic of ALS?

>  Electromyography (EMG) is a diagnostic tool used to assess the electrical activity of muscles and the nerves that control them. In Amyotrophic Lateral Sclerosis (ALS), EMG findings can vary, but some common patterns include:

1. Decreased amplitude: The electrical activity of the muscles may be reduced, resulting in decreased muscle response to stimulation.

2. Reduced motor unit recruitment: ALS patients often have a reduced ability to recruit motor units, which are groups of muscle fibers ...

**Expected answer:** Characteristic EMG findings in ALS include widespread denervation evidenced by fibrillation potentials and positive sharp waves at rest, fasciculation potentials, and chronic reinnervation changes including large-amplitude, long-duration motor unit action potentials (MUAPs) with polyphasic morphology. These findings should be present in multiple body regions to support the diagnosis. Reduced recruitment of motor units is also typically observed.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about diagnostic criteria. It missed "fibrillation potentials". It also produced 6 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about diagnostic criteria.

#### DIAG-006: What are the Gold Coast criteria for ALS diagnosis?

**Question:** What are the Gold Coast criteria for ALS diagnosis?

>  The Gold Coast criteria are a set of diagnostic criteria for Amyotrophic Lateral Sclerosis (ALS) that were developed by the American Academy of Neurology (AAN) in 2011. These criteria are designed to be more specific and less restrictive than the previous criteria, which were developed by the National Institute for Neurological Disorders and Stroke (NINDS) in 1994. The Gold Coast criteria are based on the presence of motor neuron loss and the presence of motor neuron degeneration, which are ...

**Expected answer:** The Gold Coast criteria (2019) simplified ALS diagnosis by requiring: (1) progressive motor impairment documented by history or clinical examination, (2) presence of upper and/or lower motor neuron dysfunction on examination (with EMG as equivalent for LMN), and (3) exclusion of alternative diagnoses by investigations. Unlike El Escorial, the Gold Coast criteria eliminated the requirement for combined UMN and LMN signs, accepting isolated LMN disease with progressive course. This significantly improved sensitivity for diagnosing patients who present without UMN signs.
**Accuracy:** 0.00
**Failure mode:** degenerate

The model failed to identify any of 0 of 4 key facts about diagnostic criteria. It missed "simplified diagnosis". It also produced 22 potentially fabricated entities. This illustrates degenerate where the model produced confidently wrong output about diagnostic criteria.

### Edge cases

10 samples selected for this category.

#### DRUG-003: What is edaravone and how does it work in treating ALS?

**Question:** What is edaravone and how does it work in treating ALS?

>  Edaravone (Radicava) is a free radical scavenger that is used to treat amyotrophic lateral sclerosis (ALS). It works by reducing the production of harmful free radicals in the brain and spinal cord, which can cause damage to nerve cells and lead to the progression of ALS. Edaravone is taken orally and is absorbed into the bloodstream, where it circulates to the brain and spinal cord. Once in the brain and spinal cord, edaravone binds to free radicals and neutralizes them, preventing them fro...

**Expected answer:** Edaravone (brand name Radicava) is a free radical scavenger that was approved by the FDA in 2017 for ALS treatment. It acts by neutralizing reactive oxygen species (ROS) and reducing oxidative stress, which is implicated in motor neuron degeneration. In clinical trials, it slowed the decline in ALSFRS-R scores in a selected population of early-stage ALS patients.
**Accuracy:** 0.50
**Failure mode:** accurate

The model partially addressed 2 of 4 key facts about drug treatment. It matched "free radical scavenger" but missed "oxidative stress". It also produced 2 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### GENE-001: What is the most common genetic cause of familial ALS?

**Question:** What is the most common genetic cause of familial ALS?

>  The most frequently mutated gene in familial ALS cases is the SOD1 gene. Mutations in this gene are responsible for approximately 20% of familial ALS cases. Other genes that have been implicated in familial ALS include C9orf72, TARDBP, and FUS. However, the SOD1 gene remains the most frequently mutated gene in familial ALS cases.

**Expected answer:** The C9orf72 hexanucleotide repeat expansion is the most common genetic cause of familial ALS, accounting for approximately 40% of familial cases and about 5-10% of apparently sporadic cases in European populations. The mutation involves a GGGGCC repeat expansion in the first intron of the C9orf72 gene on chromosome 9p21.
**Accuracy:** 0.50
**Failure mode:** confident fabrication

The model partially addressed 2 of 4 key facts about gene mutation. It matched "c9orf72" but missed "hexanucleotide repeat expansion". It also produced 3 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about gene mutation but missed critical details.

#### GENE-003: What is the TARDBP gene and what protein does it encode?

**Question:** What is the TARDBP gene and what protein does it encode?

>  The TARDBP gene encodes a protein called TAR DNA-binding protein 43 (TDP-43). TDP-43 is a protein that is involved in the regulation of gene expression and the processing of RNA. It is also involved in the transport of RNA from the nucleus to the cytoplasm. TDP-43 is a key player in the regulation of RNA processing and is involved in the regulation of gene expression. It is also involved in the transport of RNA from the nucleus to the cytoplasm. TDP-43 is a key player in the regulation of RN...

**Expected answer:** TARDBP encodes the TAR DNA-binding protein 43 (TDP-43), a nuclear RNA-binding protein involved in RNA processing, splicing, and transport. Mutations in TARDBP account for approximately 4-5% of familial ALS cases. TDP-43 pathology (cytoplasmic mislocalization and aggregation) is the hallmark pathological finding in approximately 97% of all ALS cases, regardless of genetic cause.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about gene mutation. It matched "tdp-43" but missed "cytoplasmic mislocalization". It also produced 1 potentially fabricated entity. This borderline response shows fragmented knowledge where the model captured some facts about gene mutation but missed critical details.

#### DIAG-003: What are upper motor neuron signs in ALS?

**Question:** What are upper motor neuron signs in ALS?

>  Upper motor neuron (UMN) involvement in Amyotrophic Lateral Sclerosis (ALS) typically presents with a distinct set of clinical signs and symptoms. Here are the key clinical features:

1. **Spastic Paralysis**: The most prominent feature is spasticity, which can be generalized or focal. Spasticity is characterized by increased muscle tone, stiffness, and resistance to passive movement.

2. **Hyporeflexia**: This is the opposite of spasticity and refers to decreased muscle tone and reduced res...

**Expected answer:** Upper motor neuron (UMN) signs in ALS include hyperreflexia (exaggerated deep tendon reflexes), spasticity (increased muscle tone), pathological reflexes such as the Babinski sign (extensor plantar response), Hoffmann sign, clonus, and pseudobulbar affect (emotional lability). These signs result from degeneration of corticospinal and corticobulbar motor neurons.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about diagnostic criteria. It matched "hyperreflexia" but missed "babinski sign". It also produced 27 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about diagnostic criteria but missed critical details.

#### TEMP-008: When did the Ice Bucket Challenge raise awareness and fundin...

**Question:** When did the Ice Bucket Challenge raise awareness and funding for ALS?

>  The Ice Bucket Challenge was a viral social media campaign that began in 2014. It involved participants pouring a bucket of ice water over their heads and challenging their friends and family to do the same. The challenge was designed to raise awareness and funds for the ALS Association, which is dedicated to finding a cure for amyotrophic lateral sclerosis (ALS), also known as Lou Gehrig's disease.

The Ice Bucket Challenge quickly gained popularity and became a global phenomenon, with mill...

**Expected answer:** The ALS Ice Bucket Challenge went viral in the summer of 2014, becoming one of the largest social media fundraising campaigns in history. The challenge raised over $115 million for the ALS Association in the United States alone between July and August 2014, and over $220 million worldwide. The funds contributed to research breakthroughs including the identification of the NEK1 gene as an ALS risk factor.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about temporal accuracy. It matched "over $115 million raised" but missed "summer 2014". It also produced 12 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about temporal accuracy but missed critical details.

#### CARE-004: What augmentative and alternative communication devices are ...

**Question:** What augmentative and alternative communication devices are used for ALS patients?

>  ALS patients who lose speech may use a variety of communication aids and assistive technology to communicate. Some common options include:

1. Speech-generating devices (SGDs): These are electronic devices that help people with speech difficulties to communicate. They can be programmed to generate text, speech, or both. Examples include the Dragon NaturallySpeaking, Dragon Medical Pro, and Dragon Medical Plus.

2. Augmentative and alternative communication (AAC) boards: These are visual comm...

**Expected answer:** Augmentative and alternative communication (AAC) devices for ALS patients include low-tech options (letter boards, picture boards) and high-tech options (speech-generating devices, eye-tracking systems, brain-computer interfaces). Eye-gaze tracking technology allows patients with severe motor impairment to select letters or words on a screen. Voice banking (recording one's voice before speech deteriorates) allows personalized synthetic speech. Brain-computer interfaces (BCIs) are in development for patients who lose all voluntary movement.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about patient care. It matched "eye-tracking systems" but missed "voice banking". It also produced 22 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about patient care but missed critical details.

#### DRUG-018: What medications are commonly used to manage muscle cramps i...

**Question:** What medications are commonly used to manage muscle cramps in ALS?

>  Muscle cramps are a common symptom in ALS patients. The drugs used to treat muscle cramps in ALS patients include:

1. Baclofen: This is a muscle relaxant that can help reduce muscle spasms and cramps.

2. Tizanidine: This is a muscle relaxant that can help reduce muscle spasms and cramps.

3. Gabapentin: This is an anticonvulsant that can help reduce muscle spasms and cramps.

4. Carbamazepine: This is an anticonvulsant that can help reduce muscle spasms and cramps.

5. Phenytoin: This is a...

**Expected answer:** Muscle cramps in ALS are commonly managed with quinine sulfate (where available and with appropriate risk monitoring), mexiletine (a sodium channel blocker that has shown efficacy in a small randomized trial), baclofen, and levetiracetam. Gabapentin and magnesium supplements are also used. Physical stretching exercises complement pharmacological management. Quinine use has been restricted in some countries due to rare but serious adverse effects including thrombocytopenia.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about drug treatment. It matched "baclofen" but missed "mexiletine". It also produced 8 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about drug treatment but missed critical details.

#### DIAG-011: What specific upper motor neuron signs are assessed in ALS d...

**Question:** What specific upper motor neuron signs are assessed in ALS diagnosis?

>  The UMN signs include:
1. Hyperreflexia
2. Clonus
3. Spasticity
4. Muscle tone
5. Fasciculations
6. Tremor
7. Ataxia
8. Dysarthria
9. Dysphagia
10. Autonomic dysfunction
11. Urinary incontinence
12. Gait disturbance
13. Muscle weakness
14. Muscle atrophy
15. Muscle hypertrophy
16. Muscle rigidity
17. Muscle wasting
18. Muscle cramps
19. Muscle spasms
20. Muscle twitching
21. Muscle fasciculations
22. Muscle tremor
23. Muscle fasciculations
24. Muscle atrophy
25. Muscle hypertrophy
26. Muscle...

**Expected answer:** Upper motor neuron signs assessed in ALS diagnosis include hyperreflexia (exaggerated deep tendon reflexes), spasticity (velocity-dependent increase in muscle tone), pathological reflexes including the Babinski sign (extensor plantar response), Hoffmann sign (thumb flexion on flicking middle finger), clonus (rhythmic involuntary muscle contractions), jaw jerk reflex (brisk in bulbar UMN involvement), and pseudobulbar affect (involuntary emotional expression). The combination of UMN and LMN signs in the same body region is the clinical hallmark of ALS.
**Accuracy:** 0.50
**Failure mode:** confident fabrication

The model partially addressed 2 of 4 key facts about diagnostic criteria. It matched "hyperreflexia" but missed "babinski sign". It also produced 12 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about diagnostic criteria but missed critical details.

#### TEMP-016: When were the major ALS drug approvals granted by the FDA in...

**Question:** When were the major ALS drug approvals granted by the FDA in chronological order?

>  Include a brief description of each therapy and its mechanism of action. FDA-approved therapies for Amyotrophic Lateral Sclerosis (ALS) are a relatively new field, with the first approved therapy being riluzole in 1995. Since then, several other therapies have been approved, including edaravone, riluzole, and edaravone. The most recent approved therapy is aducanumab, which is a monoclonal antibody that targets beta-amyloid plaques in the brain.

1. Riluzole (1995): Riluzole is the first appr...

**Expected answer:** The FDA approved ALS-specific therapies in the following chronological order: riluzole was approved in December 1995 as the first-ever ALS treatment. Edaravone (Radicava) was approved in May 2017. AMX0035 (Relyvrio) received approval in September 2022 but was voluntarily withdrawn in April 2024 after the PHOENIX confirmatory trial failed. Tofersen (Qalsody) received accelerated approval in April 2023 for SOD1-ALS specifically. The oral formulation of edaravone (Radicava ORS) was approved in May 2022. Thus, as of 2024, only riluzole, edaravone, and tofersen remain on the market.
**Accuracy:** 0.50
**Failure mode:** degenerate

The model partially addressed 2 of 4 key facts about temporal accuracy. It matched "riluzole 1995" but missed "amx0035 2022 withdrawn 2024". It also produced 8 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about temporal accuracy but missed critical details.

#### TEMP-001: When was riluzole approved by the FDA for ALS treatment?

**Question:** When was riluzole approved by the FDA for ALS treatment?

>  Riluzole, a drug originally developed to treat cancer, was approved by the FDA in 1995 for the treatment of amyotrophic lateral sclerosis (ALS), also known as Lou Gehrig's disease. The drug works by inhibiting the activity of glutamate, a neurotransmitter that can cause nerve cell death in ALS patients. While it does not cure ALS, it can slow the progression of the disease and improve the quality of life for some patients. Riluzole is the only FDA-approved treatment for ALS, and it is often ...

**Expected answer:** Riluzole was approved by the FDA in 1995, making it the first drug specifically approved for the treatment of ALS. It had previously been studied in clinical trials by Bensimon et al. (published 1994) and Lacomblez et al. (published 1996).
**Accuracy:** 0.33
**Failure mode:** confident fabrication

The model failed to adequately address 1 of 3 key facts about temporal accuracy. It matched "1995" but missed "first fda-approved als drug". It also produced 6 potentially fabricated entities. This borderline response shows fragmented knowledge where the model captured some facts about temporal accuracy but missed critical details.

---

## Disclaimer

This report is a research artifact produced by the ALS-LM hallucination evaluation framework. The model evaluated in this report is not a medical tool and should never be used for medical decision-making. The evaluation framework exists to quantify the model's unreliability and characterize its failure modes for research purposes.

*Generated: 2026-03-21 16:50:51 UTC*
