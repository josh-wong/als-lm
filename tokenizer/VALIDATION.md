# Tokenizer validation report

Three-way comparison of the v0.1 ALS-LM prototype tokenizer, the new v0.2 ALS-LM production tokenizer, and the GPT-2 standard tokenizer. This report validates medical term handling, corpus-level efficiency, and encoding performance.

- **Date:** 2026-02-22T14:08:59.491615+00:00
- **Corpus size:** 35.93 MB (5,382,896 words, 37,680,058 characters)
- **Documents:** 19,163
- **Terms evaluated:** 100

## Corpus-level metrics

Key metrics comparing tokenizer efficiency across the full corpus.

| Metric                      | v0.1 ALS-LM | New v0.2 ALS-LM |     GPT-2 |
|-----------------------------|-------------|-----------------|-----------|
| Vocabulary size             |       3,379 |          32,768 |    50,257 |
| Total corpus tokens         |  12,023,535 |       7,185,473 | 8,194,652 |
| Fertility (tokens/word)     |      2.2337 |          1.3349 |    1.5224 |
| Compression (chars/token)   |      3.1339 |          5.2439 |    4.5981 |
| Encoding speed (tokens/sec) |      66,882 |          64,124 | 3,040,217 |
| Docs within 1024 tokens (%) |       93.4% |           96.8% |     96.3% |

## Medical term analysis by category

Each term is encoded by all three tokenizers. Lower subtoken counts indicate better domain vocabulary coverage.

### Abbreviation

Terms in this category: 16

| Term     | v0.1 ALS-LM | New v0.2 ALS-LM | GPT-2 | New tokens           |
|----------|-------------|-----------------|-------|----------------------|
| ALS      |           1 |               1 |     1 | `ALS`                |
| FVC      |           1 |               1 |     2 | `FVC`                |
| FTD      |           1 |               1 |     2 | `FTD`                |
| RNA      |           1 |               1 |     1 | `RNA`                |
| FDA      |           2 |               1 |     2 | `FDA`                |
| CSF      |           1 |               1 |     2 | `CSF`                |
| NfL      |           1 |               1 |     3 | `NfL`                |
| AMPA     |           3 |               1 |     2 | `AMPA`               |
| UMN      |           1 |               1 |     2 | `UMN`                |
| LMN      |           1 |               1 |     2 | `LMN`                |
| PLS      |           1 |               1 |     2 | `PLS`                |
| PMA      |           1 |               1 |     2 | `PMA`                |
| NMDA     |           2 |               1 |     2 | `NMDA`               |
| ASO      |           1 |               1 |     2 | `ASO`                |
| ALSFRS-R |           4 |               3 |     5 | `ALSFRS` + `-` + `R` |
| VALOR    |           2 |               3 |     2 | `V` + `AL` + `OR`    |

### Clinical

Terms in this category: 58

| Term                                | v0.1 ALS-LM | New v0.2 ALS-LM | GPT-2 | New tokens                                           |
|-------------------------------------|-------------|-----------------|-------|------------------------------------------------------|
| criteria                            |           3 |               1 |     2 | `criteria`                                           |
| degeneration                        |           3 |               1 |     3 | `degeneration`                                       |
| mutation                            |           2 |               1 |     2 | `mutation`                                           |
| diagnosis                           |           2 |               1 |     2 | `diagnosis`                                          |
| communication                       |           3 |               1 |     1 | `communication`                                      |
| quality                             |           2 |               1 |     1 | `quality`                                            |
| ventilation                         |           2 |               1 |     2 | `ventilation`                                        |
| aggregation                         |           4 |               1 |     2 | `aggregation`                                        |
| microglia                           |           3 |               1 |     3 | `microglia`                                          |
| interaction                         |           4 |               1 |     2 | `interaction`                                        |
| gastrostomy                         |           3 |               1 |     4 | `gastrostomy`                                        |
| dementia                            |           3 |               1 |     3 | `dementia`                                           |
| sensitivity                         |           3 |               1 |     2 | `sensitivity`                                        |
| spasticity                          |           2 |               1 |     3 | `spasticity`                                         |
| production                          |           2 |               1 |     1 | `production`                                         |
| asthenia                            |           3 |               1 |     3 | `asthenia`                                           |
| via                                 |           2 |               1 |     1 | `via`                                                |
| autophagy                           |           4 |               1 |     3 | `autophagy`                                          |
| concentration                       |           3 |               1 |     3 | `concentration`                                      |
| tolerability                        |           2 |               1 |     3 | `tolerability`                                       |
| utility                             |           2 |               1 |     2 | `utility`                                            |
| dysphagia                           |           1 |               1 |     5 | `dysphagia`                                          |
| sialorrhea                          |           3 |               1 |     4 | `sialorrhea`                                         |
| stability                           |           2 |               1 |     2 | `stability`                                          |
| reduction                           |           3 |               1 |     2 | `reduction`                                          |
| ataxia                              |           3 |               1 |     3 | `ataxia`                                             |
| formation                           |           2 |               1 |     1 | `formation`                                          |
| dysfunction                         |           2 |               2 |     3 | `dys` + `function`                                   |
| denervation                         |           3 |               2 |     2 | `den` + `ervation`                                   |
| sclerosis                           |           1 |               2 |     2 | `s` + `clerosis`                                     |
| excitotoxicity                      |           5 |               2 |     4 | `excit` + `otoxicity`                                |
| bulbar onset                        |           2 |               2 |     3 | `bulbar` + `Ġonset`                                  |
| combination                         |           2 |               2 |     2 | `comb` + `ination`                                   |
| capacity                            |           3 |               2 |     1 | `c` + `apacity`                                      |
| neuroinflammation                   |           3 |               2 |     4 | `neuro` + `inflammation`                             |
| prognosis                           |           3 |               2 |     3 | `pro` + `gnosis`                                     |
| atrophy                             |           3 |               2 |     2 | `at` + `rophy`                                       |
| limb onset                          |           3 |               2 |     3 | `limb` + `Ġonset`                                    |
| proteostasis                        |           3 |               2 |     4 | `prote` + `ostasis`                                  |
| conduction                          |           2 |               2 |     2 | `cond` + `uction`                                    |
| respiratory failure                 |           3 |               2 |     4 | `respiratory` + `Ġfailure`                           |
| examination                         |           4 |               2 |     1 | `ex` + `amination`                                   |
| hypothesis                          |           4 |               2 |     3 | `hyp` + `othesis`                                    |
| tomography                          |           3 |               2 |     2 | `t` + `omography`                                    |
| hyperreflexia                       |           4 |               2 |     4 | `hyper` + `reflexia`                                 |
| clonus                              |           3 |               2 |     3 | `cl` + `onus`                                        |
| instability                         |           3 |               2 |     2 | `in` + `stability`                                   |
| distribution                        |           4 |               2 |     2 | `dis` + `tribution`                                  |
| modification                        |           4 |               2 |     2 | `mod` + `ification`                                  |
| susceptibility                      |           3 |               2 |     4 | `sus` + `ceptibility`                                |
| mislocalization                     |           4 |               2 |     4 | `mis` + `localization`                               |
| El Escorial criteria                |           4 |               3 |     4 | `El` + `ĠEscorial` + `Ġcriteria`                     |
| Awaji criteria                      |           4 |               3 |     3 | `A` + `waji` + `Ġcriteria`                           |
| fasciculation                       |           3 |               3 |     4 | `f` + `ascic` + `ulation`                            |
| diaphragmatic pacing                |           4 |               3 |     5 | `d` + `iaphragmatic` + `Ġpacing`                     |
| non-invasive ventilation            |           5 |               4 |     5 | `non` + `-` + `invasive` + `Ġventilation`            |
| Babinski sign                       |           4 |               4 |     4 | `B` + `ab` + `inski` + `Ġsign`                       |
| percutaneous endoscopic gastrostomy |           4 |               4 |     8 | `per` + `cutaneous` + `Ġendoscopic` + `Ġgastrostomy` |

### Disease

Terms in this category: 5

| Term                          | v0.1 ALS-LM | New v0.2 ALS-LM | GPT-2 | New tokens                                |
|-------------------------------|-------------|-----------------|-------|-------------------------------------------|
| motor neuron disease          |           4 |               3 |     4 | `motor` + `Ġneuron` + `Ġdisease`          |
| amyotrophic lateral sclerosis |           5 |               3 |     5 | `amyotrophic` + `Ġlateral` + `Ġsclerosis` |
| primary lateral sclerosis     |           5 |               3 |     3 | `primary` + `Ġlateral` + `Ġsclerosis`     |
| progressive muscular atrophy  |           5 |               3 |     5 | `progressive` + `Ġmuscular` + `Ġatrophy`  |
| spinocerebellar ataxia        |           5 |               3 |     9 | `sp` + `inocerebellar` + `Ġataxia`        |

### Drug

Terms in this category: 13

| Term                  | v0.1 ALS-LM | New v0.2 ALS-LM | GPT-2 | New tokens                   |
|-----------------------|-------------|-----------------|-------|------------------------------|
| Phase                 |           3 |               1 |     1 | `Phase`                      |
| riluzole              |           2 |               1 |     3 | `riluzole`                   |
| tofersen              |           2 |               1 |     3 | `tofersen`                   |
| edaravone             |           2 |               1 |     3 | `edaravone`                  |
| one                   |           1 |               1 |     1 | `one`                        |
| release               |           2 |               1 |     1 | `release`                    |
| taurursodiol          |           2 |               1 |     5 | `taurursodiol`               |
| baseline              |           3 |               1 |     2 | `baseline`                   |
| decline               |           3 |               2 |     2 | `de` + `cline`               |
| AMX0035               |           4 |               2 |     4 | `AMX` + `0035`               |
| sodium phenylbutyrate |           3 |               2 |     7 | `sodium` + `Ġphenylbutyrate` |
| dismutase             |           4 |               2 |     4 | `d` + `ismutase`             |
| baclofen              |           3 |               2 |     4 | `b` + `aclofen`              |

### Gene

Terms in this category: 8

| Term     | v0.1 ALS-LM | New v0.2 ALS-LM | GPT-2 | New tokens                |
|----------|-------------|-----------------|-------|---------------------------|
| FUS      |           2 |               1 |     2 | `FUS`                     |
| SOD1     |           2 |               2 |     3 | `SOD` + `1`               |
| ATXN2    |           4 |               2 |     4 | `ATXN` + `2`              |
| BIIB078  |           3 |               2 |     4 | `BIIB` + `078`            |
| TDP-43   |           3 |               3 |     4 | `TDP` + `-` + `43`        |
| GluR2    |           4 |               3 |     4 | `Glu` + `R` + `2`         |
| C9orf72  |           4 |               4 |     4 | `C` + `9` + `orf` + `72`  |
| SOD1-ALS |           4 |               4 |     5 | `SOD` + `1` + `-` + `ALS` |

## Summary statistics

Aggregate comparison across all evaluated terms.

| Metric                     |     v0.1 ALS-LM |      New v0.2 ALS-LM |        GPT-2 |
|----------------------------|-----------------|----------------------|--------------|
| Average subtokens per term |            2.81 |                 1.73 |         2.94 |
| Single-token terms         |              14 |                   50 |           12 |

| Comparison                      |   Wins | Losses |   Ties |
|---------------------------------|--------|--------|--------|
| New v0.2 ALS-LM vs. v0.1 ALS-LM |     74 |      2 |     24 |
| New v0.2 ALS-LM vs. GPT-2       |     73 |      3 |     24 |

---
*Report generated by scripts/retrain_tokenizer.py on 2026-02-22T14:08:59.491615+00:00*
