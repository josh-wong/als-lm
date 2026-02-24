# Tokenizer validation report

Three-way comparison of the v0.1 ALS-LM prototype tokenizer, the new v0.2 ALS-LM production tokenizer, and the GPT-2 standard tokenizer. This report validates medical term handling, corpus-level efficiency, and encoding performance.

- **Date:** 2026-02-23T14:31:49.359212+00:00
- **Corpus size:** 673.48 MB (101,303,590 words, 706,191,475 characters)
- **Documents:** 35,088
- **Terms evaluated:** 100

## Corpus-level metrics

Key metrics comparing tokenizer efficiency across the full corpus.

| Metric                      | v0.1 ALS-LM | New v0.2 ALS-LM | GPT-2       |
|-----------------------------|-------------|-----------------|-------------|
| Vocabulary size             |      32,768 |          50,257 |      50,257 |
| Total corpus tokens         | 153,485,846 |     142,833,967 | 162,592,689 |
| Fertility (tokens/word)     |      1.5151 |          1.4100 |      1.6050 |
| Compression (chars/token)   |      4.6010 |          4.9441 |      4.3433 |
| Encoding speed (tokens/sec) |     598,759 |         549,938 |   3,503,665 |
| Docs within 1024 tokens (%) |       54.5% |           54.6% |       54.2% |

## Medical term analysis by category

Each term is encoded by all three tokenizers. Lower subtoken counts indicate better domain vocabulary coverage.

### Abbreviation

Terms in this category: 42

| Term     | v0.1 ALS-LM | New v0.2 ALS-LM | GPT-2 | New v0.2 ALS-LM tokens |
|----------|-------------|-----------------|-------|------------------------|
| ALS      |           1 |               1 |     1 | `ALS`                  |
| FVC      |           1 |               1 |     2 | `FVC`                  |
| FTD      |           1 |               1 |     2 | `FTD`                  |
| RNA      |           1 |               1 |     1 | `RNA`                  |
| FDA      |           1 |               1 |     2 | `FDA`                  |
| CSF      |           1 |               1 |     2 | `CSF`                  |
| NfL      |           1 |               1 |     3 | `NfL`                  |
| AMPA     |           1 |               1 |     2 | `AMPA`                 |
| UMN      |           1 |               1 |     2 | `UMN`                  |
| LMN      |           1 |               1 |     2 | `LMN`                  |
| PLS      |           1 |               1 |     2 | `PLS`                  |
| PMA      |           1 |               1 |     2 | `PMA`                  |
| NMDA     |           1 |               1 |     2 | `NMDA`                 |
| ASO      |           1 |               1 |     2 | `ASO`                  |
| PBA      |           1 |               1 |     2 | `PBA`                  |
| GGGGCC   |           1 |               1 |     2 | `GGGGCC`               |
| CAG      |           1 |               1 |     2 | `CAG`                  |
| iPSC     |           1 |               1 |     3 | `iPSC`                 |
| EMG      |           1 |               1 |     2 | `EMG`                  |
| MND      |           1 |               1 |     2 | `MND`                  |
| PBP      |           1 |               1 |     2 | `PBP`                  |
| NIV      |           1 |               1 |     2 | `NIV`                  |
| PEG      |           1 |               1 |     2 | `PEG`                  |
| DTI      |           1 |               1 |     2 | `DTI`                  |
| PET      |           1 |               1 |     1 | `PET`                  |
| AAV      |           1 |               1 |     2 | `AAV`                  |
| MSC      |           1 |               1 |     2 | `MSC`                  |
| SBMA     |           1 |               1 |     2 | `SBMA`                 |
| HSP      |           1 |               1 |     2 | `HSP`                  |
| sALS     |           1 |               1 |     2 | `sALS`                 |
| fALS     |           1 |               1 |     2 | `fALS`                 |
| bvFTD    |           1 |               1 |     4 | `bvFTD`                |
| AAC      |           1 |               1 |     2 | `AAC`                  |
| SVC      |           1 |               1 |     2 | `SVC`                  |
| pNfH     |           1 |               1 |     4 | `pNfH`                 |
| RNAi     |           1 |               1 |     2 | `RNAi`                 |
| CENTAUR  |           2 |               2 |     3 | `CE` + `NTAUR`         |
| BiPAP    |           1 |               2 |     3 | `BiP` + `AP`           |
| SNIP     |           1 |               2 |     2 | `SN` + `IP`            |
| ALSFRS-R |           3 |               3 |     5 | `ALSFRS` + `-` + `R`   |
| VALOR    |           3 |               3 |     2 | `V` + `AL` + `OR`      |
| ALS-FTD  |           3 |               3 |     4 | `ALS` + `-` + `FTD`    |

### Clinical

Terms in this category: 58

| Term                                | v0.1 ALS-LM | New v0.2 ALS-LM | GPT-2 | New v0.2 ALS-LM tokens                               |
|-------------------------------------|-------------|-----------------|-------|------------------------------------------------------|
| dysfunction                         |           2 |               1 |     3 | `dysfunction`                                        |
| degeneration                        |           1 |               1 |     3 | `degeneration`                                       |
| mutation                            |           1 |               1 |     2 | `mutation`                                           |
| diagnosis                           |           1 |               1 |     2 | `diagnosis`                                          |
| communication                       |           1 |               1 |     1 | `communication`                                      |
| capacity                            |           2 |               1 |     1 | `capacity`                                           |
| quality                             |           1 |               1 |     1 | `quality`                                            |
| neuroinflammation                   |           2 |               1 |     4 | `neuroinflammation`                                  |
| atrophy                             |           2 |               1 |     2 | `atrophy`                                            |
| aggregation                         |           1 |               1 |     2 | `aggregation`                                        |
| microglia                           |           1 |               1 |     3 | `microglia`                                          |
| interaction                         |           1 |               1 |     2 | `interaction`                                        |
| dementia                            |           1 |               1 |     3 | `dementia`                                           |
| sensitivity                         |           1 |               1 |     2 | `sensitivity`                                        |
| production                          |           1 |               1 |     1 | `production`                                         |
| asthenia                            |           1 |               1 |     3 | `asthenia`                                           |
| examination                         |           2 |               1 |     1 | `examination`                                        |
| via                                 |           1 |               1 |     1 | `via`                                                |
| autophagy                           |           1 |               1 |     3 | `autophagy`                                          |
| concentration                       |           1 |               1 |     3 | `concentration`                                      |
| utility                             |           1 |               1 |     2 | `utility`                                            |
| stability                           |           1 |               1 |     2 | `stability`                                          |
| reduction                           |           1 |               1 |     2 | `reduction`                                          |
| distribution                        |           2 |               1 |     2 | `distribution`                                       |
| modification                        |           2 |               1 |     2 | `modification`                                       |
| ataxia                              |           1 |               1 |     3 | `ataxia`                                             |
| formation                           |           1 |               1 |     1 | `formation`                                          |
| criteria                            |           1 |               2 |     2 | `cri` + `teria`                                      |
| denervation                         |           2 |               2 |     2 | `den` + `ervation`                                   |
| sclerosis                           |           2 |               2 |     2 | `s` + `clerosis`                                     |
| excitotoxicity                      |           2 |               2 |     4 | `ex` + `citotoxicity`                                |
| bulbar onset                        |           2 |               2 |     3 | `bulbar` + `Ġonset`                                  |
| combination                         |           2 |               2 |     2 | `com` + `bination`                                   |
| ventilation                         |           1 |               2 |     2 | `v` + `entilation`                                   |
| limb onset                          |           2 |               2 |     3 | `limb` + `Ġonset`                                    |
| proteostasis                        |           2 |               2 |     4 | `prote` + `ostasis`                                  |
| conduction                          |           2 |               2 |     2 | `con` + `duction`                                    |
| gastrostomy                         |           1 |               2 |     4 | `gastro` + `stomy`                                   |
| respiratory failure                 |           2 |               2 |     4 | `respiratory` + `Ġfailure`                           |
| spasticity                          |           1 |               2 |     3 | `sp` + `asticity`                                    |
| hypothesis                          |           2 |               2 |     3 | `hyp` + `othesis`                                    |
| tolerability                        |           1 |               2 |     3 | `toler` + `ability`                                  |
| tomography                          |           2 |               2 |     2 | `tom` + `ography`                                    |
| dysphagia                           |           1 |               2 |     5 | `dys` + `phagia`                                     |
| hyperreflexia                       |           2 |               2 |     4 | `hyper` + `reflexia`                                 |
| clonus                              |           2 |               2 |     3 | `cl` + `onus`                                        |
| sialorrhea                          |           1 |               2 |     4 | `sial` + `orrhea`                                    |
| instability                         |           2 |               2 |     2 | `in` + `stability`                                   |
| susceptibility                      |           2 |               2 |     4 | `sus` + `ceptibility`                                |
| mislocalization                     |           2 |               2 |     4 | `mis` + `localization`                               |
| El Escorial criteria                |           3 |               3 |     4 | `El` + `ĠEscorial` + `Ġcriteria`                     |
| prognosis                           |           2 |               3 |     3 | `pro` + `gn` + `osis`                                |
| fasciculation                       |           3 |               3 |     4 | `fas` + `cic` + `ulation`                            |
| diaphragmatic pacing                |           3 |               3 |     5 | `dia` + `phragmatic` + `Ġpacing`                     |
| Awaji criteria                      |           3 |               4 |     3 | `A` + `wa` + `ji` + `Ġcriteria`                      |
| non-invasive ventilation            |           4 |               4 |     5 | `non` + `-` + `invasive` + `Ġventilation`            |
| Babinski sign                       |           4 |               4 |     4 | `B` + `ab` + `inski` + `Ġsign`                       |
| percutaneous endoscopic gastrostomy |           4 |               4 |     8 | `per` + `cutaneous` + `Ġendoscopic` + `Ġgastrostomy` |

## Summary statistics

Aggregate comparison across all evaluated terms.

| Metric                     | v0.1 ALS-LM | New v0.2 ALS-LM | GPT-2 |
|----------------------------|-------------|-----------------|-------|
| Average subtokens per term |        1.48 |            1.52 |  2.58 |
| Single-token terms         |          65 |              63 |    10 |

| Comparison                     | Wins | Losses | Ties |
|--------------------------------|------|--------|------|
| New v0.2 ALS-LM vs. v0.1 ALS-LM |    7 |     11 |   82 |
| New v0.2 ALS-LM vs. GPT-2       |   77 |      2 |   21 |

---
*Report generated by scripts/retrain_tokenizer.py on 2026-02-23T14:31:49.359212+00:00*
