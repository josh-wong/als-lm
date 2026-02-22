# Tokenizer validation report

Three-way comparison of the v0.1 prototype tokenizer, the new production tokenizer, and the GPT-2 standard tokenizer. This report validates medical term handling, corpus-level efficiency, and encoding performance.

- **Date:** 2026-02-22T08:10:29.065964+00:00
- **Corpus size:** 2.3 MB (330,001 words, 2,415,074 characters)
- **Documents:** 3,199
- **Terms evaluated:** 100

## Corpus-level metrics

Key metrics comparing tokenizer efficiency across the full corpus.

| Metric                              |     v0.1 ALS |      New ALS |        GPT-2 |
|-------------------------------------|--------------|--------------|--------------|
| Vocabulary size                     |        3,379 |        3,379 |       50,257 |
| Total corpus tokens                 |      413,198 |      413,198 |      501,398 |
| Fertility (tokens/word)             |       1.2521 |       1.2521 |       1.5194 |
| Compression (chars/token)           |       5.8448 |       5.8448 |       4.8167 |
| Encoding speed (tokens/sec)         |      237,834 |      315,934 |    3,655,548 |
| Docs within 1024 tokens (%)         |       100.0% |       100.0% |       100.0% |

## Medical term analysis by category

Each term is encoded by all three tokenizers. Lower subtoken counts indicate better domain vocabulary coverage.

### Abbreviation

Terms in this category: 42

| Term                                |  v0.1 |   New | GPT-2 | New tokens                                         |
|-------------------------------------|-------|-------|-------|----------------------------------------------------|
| ALS                                 |     1 |     1 |     1 | `ALS`                                              |
| FVC                                 |     1 |     1 |     2 | `FVC`                                              |
| FTD                                 |     1 |     1 |     2 | `FTD`                                              |
| RNA                                 |     1 |     1 |     1 | `RNA`                                              |
| CSF                                 |     1 |     1 |     2 | `CSF`                                              |
| NfL                                 |     1 |     1 |     3 | `NfL`                                              |
| UMN                                 |     1 |     1 |     2 | `UMN`                                              |
| LMN                                 |     1 |     1 |     2 | `LMN`                                              |
| PLS                                 |     1 |     1 |     2 | `PLS`                                              |
| PMA                                 |     1 |     1 |     2 | `PMA`                                              |
| ASO                                 |     1 |     1 |     2 | `ASO`                                              |
| PBA                                 |     1 |     1 |     2 | `PBA`                                              |
| iPSC                                |     1 |     1 |     3 | `iPSC`                                             |
| EMG                                 |     1 |     1 |     2 | `EMG`                                              |
| MND                                 |     1 |     1 |     2 | `MND`                                              |
| PBP                                 |     1 |     1 |     2 | `PBP`                                              |
| NIV                                 |     1 |     1 |     2 | `NIV`                                              |
| BiPAP                               |     1 |     1 |     3 | `BiPAP`                                            |
| PEG                                 |     1 |     1 |     2 | `PEG`                                              |
| DTI                                 |     1 |     1 |     2 | `DTI`                                              |
| PET                                 |     1 |     1 |     1 | `PET`                                              |
| AAV                                 |     1 |     1 |     2 | `AAV`                                              |
| HSP                                 |     1 |     1 |     2 | `HSP`                                              |
| sALS                                |     1 |     1 |     2 | `sALS`                                             |
| fALS                                |     1 |     1 |     2 | `fALS`                                             |
| bvFTD                               |     1 |     1 |     4 | `bvFTD`                                            |
| AAC                                 |     1 |     1 |     2 | `AAC`                                              |
| SNIP                                |     1 |     1 |     2 | `SNIP`                                             |
| SVC                                 |     1 |     1 |     2 | `SVC`                                              |
| pNfH                                |     1 |     1 |     4 | `pNfH`                                             |
| RNAi                                |     1 |     1 |     2 | `RNAi`                                             |
| FDA                                 |     2 |     2 |     2 | `F` + `DA`                                         |
| NMDA                                |     2 |     2 |     2 | `N` + `MDA`                                        |
| VALOR                               |     2 |     2 |     2 | `VAL` + `OR`                                       |
| CAG                                 |     2 |     2 |     2 | `C` + `AG`                                         |
| MSC                                 |     2 |     2 |     2 | `M` + `SC`                                         |
| SBMA                                |     2 |     2 |     2 | `S` + `BMA`                                        |
| AMPA                                |     3 |     3 |     2 | `A` + `M` + `PA`                                   |
| GGGGCC                              |     3 |     3 |     2 | `GG` + `GG` + `CC`                                 |
| ALS-FTD                             |     3 |     3 |     4 | `ALS` + `-` + `FTD`                                |
| CENTAUR                             |     3 |     3 |     3 | `C` + `EN` + `TAUR`                                |
| ALSFRS-R                            |     4 |     4 |     5 | `ALS` + `FRS` + `-` + `R`                          |

### Clinical

Terms in this category: 58

| Term                                |  v0.1 |   New | GPT-2 | New tokens                                         |
|-------------------------------------|-------|-------|-------|----------------------------------------------------|
| sclerosis                           |     1 |     1 |     2 | `sclerosis`                                        |
| dysphagia                           |     1 |     1 |     5 | `dysphagia`                                        |
| dysfunction                         |     2 |     2 |     3 | `dys` + `function`                                 |
| mutation                            |     2 |     2 |     2 | `mut` + `ation`                                    |
| diagnosis                           |     2 |     2 |     2 | `diagn` + `osis`                                   |
| bulbar onset                        |     2 |     2 |     3 | `bulbar` + `Ġonset`                                |
| combination                         |     2 |     2 |     2 | `com` + `bination`                                 |
| quality                             |     2 |     2 |     1 | `qu` + `ality`                                     |
| ventilation                         |     2 |     2 |     2 | `ventil` + `ation`                                 |
| conduction                          |     2 |     2 |     2 | `con` + `duction`                                  |
| spasticity                          |     2 |     2 |     3 | `sp` + `asticity`                                  |
| production                          |     2 |     2 |     1 | `pro` + `duction`                                  |
| via                                 |     2 |     2 |     1 | `v` + `ia`                                         |
| tolerability                        |     2 |     2 |     3 | `t` + `olerability`                                |
| utility                             |     2 |     2 |     2 | `ut` + `ility`                                     |
| stability                           |     2 |     2 |     2 | `st` + `ability`                                   |
| formation                           |     2 |     2 |     1 | `for` + `mation`                                   |
| criteria                            |     3 |     3 |     2 | `c` + `rit` + `eria`                               |
| degeneration                        |     3 |     3 |     3 | `de` + `gener` + `ation`                           |
| denervation                         |     3 |     3 |     2 | `d` + `ener` + `vation`                            |
| communication                       |     3 |     3 |     1 | `com` + `m` + `unication`                          |
| capacity                            |     3 |     3 |     1 | `cap` + `ac` + `ity`                               |
| neuroinflammation                   |     3 |     3 |     4 | `n` + `euro` + `inflammation`                      |
| prognosis                           |     3 |     3 |     3 | `pro` + `gn` + `osis`                              |
| atrophy                             |     3 |     3 |     2 | `at` + `roph` + `y`                                |
| limb onset                          |     3 |     3 |     3 | `l` + `imb` + `Ġonset`                             |
| proteostasis                        |     3 |     3 |     4 | `pro` + `te` + `ostasis`                           |
| microglia                           |     3 |     3 |     3 | `m` + `ic` + `roglia`                              |
| gastrostomy                         |     3 |     3 |     4 | `g` + `astro` + `stomy`                            |
| respiratory failure                 |     3 |     3 |     4 | `re` + `spiratory` + `Ġfailure`                    |
| fasciculation                       |     3 |     3 |     4 | `f` + `ascic` + `ulation`                          |
| dementia                            |     3 |     3 |     3 | `de` + `ment` + `ia`                               |
| sensitivity                         |     3 |     3 |     2 | `s` + `ens` + `itivity`                            |
| asthenia                            |     3 |     3 |     3 | `as` + `th` + `enia`                               |
| concentration                       |     3 |     3 |     3 | `con` + `cent` + `ration`                          |
| tomography                          |     3 |     3 |     2 | `t` + `om` + `ography`                             |
| clonus                              |     3 |     3 |     3 | `cl` + `on` + `us`                                 |
| sialorrhea                          |     3 |     3 |     4 | `s` + `ial` + `orrhea`                             |
| instability                         |     3 |     3 |     2 | `in` + `st` + `ability`                            |
| reduction                           |     3 |     3 |     2 | `red` + `u` + `ction`                              |
| susceptibility                      |     3 |     3 |     4 | `s` + `us` + `ceptibility`                         |
| ataxia                              |     3 |     3 |     3 | `at` + `ax` + `ia`                                 |
| El Escorial criteria                |     4 |     4 |     4 | `E` + `l` + `ĠEscorial` + `Ġcriteria`              |
| aggregation                         |     4 |     4 |     2 | `ag` + `g` + `reg` + `ation`                       |
| interaction                         |     4 |     4 |     2 | `in` + `t` + `era` + `ction`                       |
| Awaji criteria                      |     4 |     4 |     3 | `A` + `waj` + `i` + `Ġcriteria`                    |
| examination                         |     4 |     4 |     1 | `e` + `x` + `am` + `ination`                       |
| autophagy                           |     4 |     4 |     3 | `a` + `ut` + `ophag` + `y`                         |
| hypothesis                          |     4 |     4 |     3 | `h` + `yp` + `othes` + `is`                        |
| hyperreflexia                       |     4 |     4 |     4 | `h` + `yp` + `errefle` + `xia`                     |
| Babinski sign                       |     4 |     4 |     4 | `B` + `abins` + `ki` + `Ġsign`                     |
| diaphragmatic pacing                |     4 |     4 |     5 | `d` + `iaph` + `ragmatic` + `Ġpacing`              |
| percutaneous endoscopic gastrostomy |     4 |     4 |     8 | `per` + `cutaneous` + `Ġendoscopic` + `Ġgastrostomy` |
| distribution                        |     4 |     4 |     2 | `d` + `is` + `trib` + `ution`                      |
| modification                        |     4 |     4 |     2 | `m` + `od` + `ific` + `ation`                      |
| mislocalization                     |     4 |     4 |     4 | `m` + `is` + `localiz` + `ation`                   |
| excitotoxicity                      |     5 |     5 |     4 | `e` + `x` + `c` + `it` + `otoxicity`               |
| non-invasive ventilation            |     5 |     5 |     5 | `n` + `on` + `-` + `invasive` + `Ġventilation`     |

## Summary statistics

Aggregate comparison across all evaluated terms.

| Metric                              |     v0.1 ALS |      New ALS |        GPT-2 |
|-------------------------------------|--------------|--------------|--------------|
| Average subtokens per term          |         2.32 |         2.32 |         2.58 |
| Single-token terms                  |           33 |           33 |           10 |

| Comparison                          |   Wins | Losses |   Ties |
|-------------------------------------|--------|--------|--------|
| New ALS vs v0.1 ALS                 |      0 |      0 |    100 |
| New ALS vs GPT-2                    |     45 |     24 |     31 |

---
*Report generated by scripts/retrain_tokenizer.py on 2026-02-22T08:10:29.065964+00:00*
