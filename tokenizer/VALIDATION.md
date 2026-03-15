# Tokenizer validation report

Three-way comparison of the v1.0 ALS-LM tokenizer, the v1.2 ALS-LM tokenizer, and the GPT-2 standard tokenizer. This report validates medical term handling, corpus-level efficiency, and encoding performance.

- **Date:** 2026-03-15T08:04:40.976225+00:00
- **Corpus size:** 673.27 MB (101,274,013 words, 705,973,314 characters)
- **Documents:** 35,063
- **Terms evaluated:** 100

## Corpus-level metrics

Key metrics comparing tokenizer efficiency across the full corpus.

| Metric                              |     v1.0 ALS-LM |      v1.2 ALS-LM |          GPT-2 |
|-------------------------------------|-----------------|------------------|----------------|
| Vocabulary size                     |          50,257 |           50,257 |         50,257 |
| Total corpus tokens                 |     141,734,249 |      141,734,249 |    160,804,284 |
| Fertility (tokens/word)             |          1.3995 |           1.3995 |         1.5878 |
| Compression (chars/token)           |          4.9810 |           4.9810 |         4.3903 |
| Encoding speed (tokens/sec)         |         425,681 |          462,044 |      3,475,465 |
| Docs within 1024 tokens (%)         |           54.5% |            54.5% |          54.1% |

## Medical term analysis by category

Each term is encoded by all three tokenizers. Lower subtoken counts indicate better domain vocabulary coverage.

### Abbreviation

Terms in this category: 42

| Term                                | v1.0 ALS-LM |     v1.2 ALS-LM | GPT-2 | v1.2 ALS-LM tokens                                 |
|-------------------------------------|-------------|-----------------|-------|----------------------------------------------------|
| ALS                                 |           1 |               1 |     1 | `ALS`                                              |
| FVC                                 |           1 |               1 |     2 | `FVC`                                              |
| FTD                                 |           1 |               1 |     2 | `FTD`                                              |
| RNA                                 |           1 |               1 |     1 | `RNA`                                              |
| FDA                                 |           1 |               1 |     2 | `FDA`                                              |
| CSF                                 |           1 |               1 |     2 | `CSF`                                              |
| NfL                                 |           1 |               1 |     3 | `NfL`                                              |
| AMPA                                |           1 |               1 |     2 | `AMPA`                                             |
| UMN                                 |           1 |               1 |     2 | `UMN`                                              |
| LMN                                 |           1 |               1 |     2 | `LMN`                                              |
| PLS                                 |           1 |               1 |     2 | `PLS`                                              |
| PMA                                 |           1 |               1 |     2 | `PMA`                                              |
| NMDA                                |           1 |               1 |     2 | `NMDA`                                             |
| ASO                                 |           1 |               1 |     2 | `ASO`                                              |
| PBA                                 |           1 |               1 |     2 | `PBA`                                              |
| GGGGCC                              |           1 |               1 |     2 | `GGGGCC`                                           |
| CAG                                 |           1 |               1 |     2 | `CAG`                                              |
| iPSC                                |           1 |               1 |     3 | `iPSC`                                             |
| EMG                                 |           1 |               1 |     2 | `EMG`                                              |
| MND                                 |           1 |               1 |     2 | `MND`                                              |
| PBP                                 |           1 |               1 |     2 | `PBP`                                              |
| NIV                                 |           1 |               1 |     2 | `NIV`                                              |
| PEG                                 |           1 |               1 |     2 | `PEG`                                              |
| DTI                                 |           1 |               1 |     2 | `DTI`                                              |
| PET                                 |           1 |               1 |     1 | `PET`                                              |
| AAV                                 |           1 |               1 |     2 | `AAV`                                              |
| MSC                                 |           1 |               1 |     2 | `MSC`                                              |
| SBMA                                |           1 |               1 |     2 | `SBMA`                                             |
| HSP                                 |           1 |               1 |     2 | `HSP`                                              |
| sALS                                |           1 |               1 |     2 | `sALS`                                             |
| fALS                                |           1 |               1 |     2 | `fALS`                                             |
| bvFTD                               |           1 |               1 |     4 | `bvFTD`                                            |
| AAC                                 |           1 |               1 |     2 | `AAC`                                              |
| SVC                                 |           1 |               1 |     2 | `SVC`                                              |
| pNfH                                |           1 |               1 |     4 | `pNfH`                                             |
| RNAi                                |           1 |               1 |     2 | `RNAi`                                             |
| CENTAUR                             |           2 |               2 |     3 | `CE` + `NTAUR`                                     |
| BiPAP                               |           2 |               2 |     3 | `BiP` + `AP`                                       |
| SNIP                                |           2 |               2 |     2 | `SN` + `IP`                                        |
| ALSFRS-R                            |           3 |               3 |     5 | `ALSFRS` + `-` + `R`                               |
| VALOR                               |           3 |               3 |     2 | `V` + `AL` + `OR`                                  |
| ALS-FTD                             |           3 |               3 |     4 | `ALS` + `-` + `FTD`                                |

### Clinical

Terms in this category: 58

| Term                                | v1.0 ALS-LM |     v1.2 ALS-LM | GPT-2 | v1.2 ALS-LM tokens                                 |
|-------------------------------------|-------------|-----------------|-------|----------------------------------------------------|
| dysfunction                         |           1 |               1 |     3 | `dysfunction`                                      |
| degeneration                        |           1 |               1 |     3 | `degeneration`                                     |
| mutation                            |           1 |               1 |     2 | `mutation`                                         |
| diagnosis                           |           1 |               1 |     2 | `diagnosis`                                        |
| communication                       |           1 |               1 |     1 | `communication`                                    |
| capacity                            |           1 |               1 |     1 | `capacity`                                         |
| quality                             |           1 |               1 |     1 | `quality`                                          |
| neuroinflammation                   |           1 |               1 |     4 | `neuroinflammation`                                |
| atrophy                             |           1 |               1 |     2 | `atrophy`                                          |
| aggregation                         |           1 |               1 |     2 | `aggregation`                                      |
| microglia                           |           1 |               1 |     3 | `microglia`                                        |
| interaction                         |           1 |               1 |     2 | `interaction`                                      |
| dementia                            |           1 |               1 |     3 | `dementia`                                         |
| sensitivity                         |           1 |               1 |     2 | `sensitivity`                                      |
| production                          |           1 |               1 |     1 | `production`                                       |
| asthenia                            |           1 |               1 |     3 | `asthenia`                                         |
| examination                         |           1 |               1 |     1 | `examination`                                      |
| via                                 |           1 |               1 |     1 | `via`                                              |
| autophagy                           |           1 |               1 |     3 | `autophagy`                                        |
| concentration                       |           1 |               1 |     3 | `concentration`                                    |
| utility                             |           1 |               1 |     2 | `utility`                                          |
| stability                           |           1 |               1 |     2 | `stability`                                        |
| reduction                           |           1 |               1 |     2 | `reduction`                                        |
| distribution                        |           1 |               1 |     2 | `distribution`                                     |
| modification                        |           1 |               1 |     2 | `modification`                                     |
| ataxia                              |           1 |               1 |     3 | `ataxia`                                           |
| formation                           |           1 |               1 |     1 | `formation`                                        |
| criteria                            |           2 |               2 |     2 | `cri` + `teria`                                    |
| denervation                         |           2 |               2 |     2 | `den` + `ervation`                                 |
| sclerosis                           |           2 |               2 |     2 | `s` + `clerosis`                                   |
| excitotoxicity                      |           2 |               2 |     4 | `ex` + `citotoxicity`                              |
| bulbar onset                        |           2 |               2 |     3 | `bulbar` + `Ġonset`                                |
| combination                         |           2 |               2 |     2 | `com` + `bination`                                 |
| ventilation                         |           2 |               2 |     2 | `v` + `entilation`                                 |
| limb onset                          |           2 |               2 |     3 | `limb` + `Ġonset`                                  |
| proteostasis                        |           2 |               2 |     4 | `prote` + `ostasis`                                |
| conduction                          |           2 |               2 |     2 | `con` + `duction`                                  |
| gastrostomy                         |           2 |               2 |     4 | `gastro` + `stomy`                                 |
| respiratory failure                 |           2 |               2 |     4 | `respiratory` + `Ġfailure`                         |
| spasticity                          |           2 |               2 |     3 | `sp` + `asticity`                                  |
| hypothesis                          |           2 |               2 |     3 | `hyp` + `othesis`                                  |
| tolerability                        |           2 |               2 |     3 | `toler` + `ability`                                |
| tomography                          |           2 |               2 |     2 | `tom` + `ography`                                  |
| dysphagia                           |           2 |               2 |     5 | `dys` + `phagia`                                   |
| hyperreflexia                       |           2 |               2 |     4 | `hyper` + `reflexia`                               |
| clonus                              |           2 |               2 |     3 | `cl` + `onus`                                      |
| sialorrhea                          |           2 |               2 |     4 | `sial` + `orrhea`                                  |
| instability                         |           2 |               2 |     2 | `in` + `stability`                                 |
| susceptibility                      |           2 |               2 |     4 | `sus` + `ceptibility`                              |
| mislocalization                     |           2 |               2 |     4 | `mis` + `localization`                             |
| El Escorial criteria                |           3 |               3 |     4 | `El` + `ĠEscorial` + `Ġcriteria`                   |
| prognosis                           |           3 |               3 |     3 | `pro` + `gn` + `osis`                              |
| fasciculation                       |           3 |               3 |     4 | `fas` + `cic` + `ulation`                          |
| diaphragmatic pacing                |           3 |               3 |     5 | `dia` + `phragmatic` + `Ġpacing`                   |
| Awaji criteria                      |           4 |               4 |     3 | `A` + `wa` + `ji` + `Ġcriteria`                    |
| non-invasive ventilation            |           4 |               4 |     5 | `non` + `-` + `invasive` + `Ġventilation`          |
| Babinski sign                       |           4 |               4 |     4 | `B` + `ab` + `inski` + `Ġsign`                     |
| percutaneous endoscopic gastrostomy |           4 |               4 |     8 | `per` + `cutaneous` + `Ġendoscopic` + `Ġgastrostomy` |

## Summary statistics

Aggregate comparison across all evaluated terms.

| Metric                              |     v1.0 ALS-LM |      v1.2 ALS-LM |          GPT-2 |
|-------------------------------------|-----------------|------------------|----------------|
| Average subtokens per term          |            1.52 |             1.52 |           2.58 |
| Single-token terms                  |              63 |               63 |             10 |

| Comparison                          |   Wins | Losses |   Ties |
|-------------------------------------|--------|--------|--------|
| v1.2 ALS-LM vs v1.0 ALS-LM          |      0 |      0 |    100 |
| v1.2 ALS-LM vs GPT-2                |     77 |      2 |     21 |

---
*Report generated by scripts/retrain_tokenizer.py on 2026-03-15T08:04:40.976225+00:00*
