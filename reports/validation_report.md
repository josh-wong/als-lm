# Tokenizer validation report

This report shows fragmentation analysis for 3 tokenizer(s) against 195 medical terms. Terms with 3+ subtokens are flagged.

- **Date:** 2026-02-21 18:15 UTC
- **Tokenizers evaluated:** 3
- **Total terms:** 195
- **Flagging threshold:** 3+ subtokens

## als_tokenizer_16k

Vocabulary size: 3,379. Flagged 115 of 195 terms (59.0%).

### Category breakdown

Per-category flagging rates show which term types are most affected by fragmentation.

| Category             | Total | Flagged | Flagged % |
|----------------------|-------|---------|-----------|
| abbreviation         |    42 |       5 |     11.9% |
| clinical             |    97 |      69 |     71.1% |
| disease              |    10 |       9 |     90.0% |
| drug                 |    31 |      19 |     61.3% |
| gene                 |    15 |      13 |     86.7% |

### Term details

Every term with its subtoken count and breakdown. Flagged terms (3+ subtokens) are marked.

| Term                                | Category        | Subtokens | Breakdown                                          | Flag |
|-------------------------------------|-----------------|-----------|----------------------------------------------------|------|
| loss-of-function                    | clinical        |         7 | `l` + `os` + `s` + `-` + `of` + `-` + `function`   |   !! |
| multifocal motor neuropathy         | disease         |         6 | `m` + `ult` + `ifoc` + `al` + `Ġmotor` + `Ġneuropathy` |   !! |
| excitotoxicity                      | clinical        |         5 | `e` + `x` + `c` + `it` + `otoxicity`               |   !! |
| non-invasive ventilation            | clinical        |         5 | `n` + `on` + `-` + `invasive` + `Ġventilation`     |   !! |
| disruption                          | clinical        |         5 | `d` + `is` + `ru` + `pt` + `ion`                   |   !! |
| neurodegeneration                   | clinical        |         5 | `n` + `euro` + `de` + `gener` + `ation`            |   !! |
| amyotrophic lateral sclerosis       | disease         |         5 | `am` + `y` + `otrophic` + `Ġlateral` + `Ġsclerosis` |   !! |
| primary lateral sclerosis           | disease         |         5 | `p` + `rim` + `ary` + `Ġlateral` + `Ġsclerosis`    |   !! |
| progressive muscular atrophy        | disease         |         5 | `p` + `rog` + `ressive` + `Ġmuscular` + `Ġatrophy` |   !! |
| spinocerebellar ataxia              | disease         |         5 | `sp` + `inoce` + `rebell` + `ar` + `Ġataxia`       |   !! |
| progressive bulbar palsy            | disease         |         5 | `p` + `rog` + `ressive` + `Ġbulbar` + `Ġpalsy`     |   !! |
| CYP1A2                              | gene            |         5 | `C` + `YP` + `1` + `A` + `2`                       |   !! |
| ALSFRS-R                            | abbreviation    |         4 | `ALS` + `FRS` + `-` + `R`                          |   !! |
| El Escorial criteria                | clinical        |         4 | `E` + `l` + `ĠEscorial` + `Ġcriteria`              |   !! |
| aggregation                         | clinical        |         4 | `ag` + `g` + `reg` + `ation`                       |   !! |
| interaction                         | clinical        |         4 | `in` + `t` + `era` + `ction`                       |   !! |
| Awaji criteria                      | clinical        |         4 | `A` + `waj` + `i` + `Ġcriteria`                    |   !! |
| examination                         | clinical        |         4 | `e` + `x` + `am` + `ination`                       |   !! |
| autophagy                           | clinical        |         4 | `a` + `ut` + `ophag` + `y`                         |   !! |
| hypothesis                          | clinical        |         4 | `h` + `yp` + `othes` + `is`                        |   !! |
| hyperreflexia                       | clinical        |         4 | `h` + `yp` + `errefle` + `xia`                     |   !! |
| Babinski sign                       | clinical        |         4 | `B` + `abins` + `ki` + `Ġsign`                     |   !! |
| diaphragmatic pacing                | clinical        |         4 | `d` + `iaph` + `ragmatic` + `Ġpacing`              |   !! |
| percutaneous endoscopic gastrostomy | clinical        |         4 | `per` + `cutaneous` + `Ġendoscopic` + `Ġgastrostomy` |   !! |
| distribution                        | clinical        |         4 | `d` + `is` + `trib` + `ution`                      |   !! |
| modification                        | clinical        |         4 | `m` + `od` + `ific` + `ation`                      |   !! |
| mislocalization                     | clinical        |         4 | `m` + `is` + `localiz` + `ation`                   |   !! |
| overactivation                      | clinical        |         4 | `o` + `ver` + `activ` + `ation`                    |   !! |
| paraplegia                          | clinical        |         4 | `p` + `ar` + `ap` + `legia`                        |   !! |
| hypoventilation                     | clinical        |         4 | `h` + `y` + `poventil` + `ation`                   |   !! |
| Gold Coast criteria                 | clinical        |         4 | `G` + `old` + `ĠCoast` + `Ġcriteria`               |   !! |
| Hepatotoxicity                      | clinical        |         4 | `H` + `e` + `pat` + `otoxicity`                    |   !! |
| personality                         | clinical        |         4 | `p` + `ers` + `on` + `ality`                       |   !! |
| disinhibition                       | clinical        |         4 | `d` + `is` + `inhib` + `ition`                     |   !! |
| spirometry                          | clinical        |         4 | `sp` + `i` + `rom` + `etry`                        |   !! |
| apoptosis                           | clinical        |         4 | `a` + `po` + `pt` + `osis`                         |   !! |
| motor neuron disease                | disease         |         4 | `m` + `otor` + `Ġneuron` + `Ġdisease`              |   !! |
| Kennedy disease                     | disease         |         4 | `K` + `enned` + `y` + `Ġdisease`                   |   !! |
| hereditary spastic paraplegia       | disease         |         4 | `he` + `reditary` + `Ġspastic` + `Ġparaplegia`     |   !! |
| disease                             | drug            |         4 | `d` + `is` + `e` + `ase`                           |   !! |
| AMX0035                             | drug            |         4 | `A` + `M` + `X` + `0035`                           |   !! |
| dismutase                           | drug            |         4 | `d` + `ism` + `ut` + `ase`                         |   !! |
| botulinum toxin                     | drug            |         4 | `b` + `ot` + `ulinum` + `Ġtoxin`                   |   !! |
| polyglutamine                       | drug            |         4 | `po` + `ly` + `glutam` + `ine`                     |   !! |
| dextromethorphan                    | drug            |         4 | `de` + `xt` + `romethor` + `phan`                  |   !! |
| benzothiazole                       | drug            |         4 | `b` + `en` + `zoth` + `iazole`                     |   !! |
| radicava                            | drug            |         4 | `ra` + `d` + `ic` + `ava`                          |   !! |
| C9orf72                             | gene            |         4 | `C` + `9` + `orf` + `72`                           |   !! |
| ATXN2                               | gene            |         4 | `A` + `TX` + `N` + `2`                             |   !! |
| SOD1-ALS                            | gene            |         4 | `SOD` + `1` + `-` + `ALS`                          |   !! |
| GluR2                               | gene            |         4 | `G` + `lu` + `R` + `2`                             |   !! |
| AMPA                                | abbreviation    |         3 | `A` + `M` + `PA`                                   |   !! |
| GGGGCC                              | abbreviation    |         3 | `GG` + `GG` + `CC`                                 |   !! |
| ALS-FTD                             | abbreviation    |         3 | `ALS` + `-` + `FTD`                                |   !! |
| CENTAUR                             | abbreviation    |         3 | `C` + `EN` + `TAUR`                                |   !! |
| criteria                            | clinical        |         3 | `c` + `rit` + `eria`                               |   !! |
| degeneration                        | clinical        |         3 | `de` + `gener` + `ation`                           |   !! |
| denervation                         | clinical        |         3 | `d` + `ener` + `vation`                            |   !! |
| communication                       | clinical        |         3 | `com` + `m` + `unication`                          |   !! |
| capacity                            | clinical        |         3 | `cap` + `ac` + `ity`                               |   !! |
| neuroinflammation                   | clinical        |         3 | `n` + `euro` + `inflammation`                      |   !! |
| prognosis                           | clinical        |         3 | `pro` + `gn` + `osis`                              |   !! |
| atrophy                             | clinical        |         3 | `at` + `roph` + `y`                                |   !! |
| limb onset                          | clinical        |         3 | `l` + `imb` + `Ġonset`                             |   !! |
| proteostasis                        | clinical        |         3 | `pro` + `te` + `ostasis`                           |   !! |
| microglia                           | clinical        |         3 | `m` + `ic` + `roglia`                              |   !! |
| population                          | clinical        |         3 | `p` + `op` + `ulation`                             |   !! |
| gastrostomy                         | clinical        |         3 | `g` + `astro` + `stomy`                            |   !! |
| respiratory failure                 | clinical        |         3 | `re` + `spiratory` + `Ġfailure`                    |   !! |
| fasciculation                       | clinical        |         3 | `f` + `ascic` + `ulation`                          |   !! |
| dementia                            | clinical        |         3 | `de` + `ment` + `ia`                               |   !! |
| sensitivity                         | clinical        |         3 | `s` + `ens` + `itivity`                            |   !! |
| asthenia                            | clinical        |         3 | `as` + `th` + `enia`                               |   !! |
| concentration                       | clinical        |         3 | `con` + `cent` + `ration`                          |   !! |
| tomography                          | clinical        |         3 | `t` + `om` + `ography`                             |   !! |
| clonus                              | clinical        |         3 | `cl` + `on` + `us`                                 |   !! |
| sialorrhea                          | clinical        |         3 | `s` + `ial` + `orrhea`                             |   !! |
| instability                         | clinical        |         3 | `in` + `st` + `ability`                            |   !! |
| reduction                           | clinical        |         3 | `red` + `u` + `ction`                              |   !! |
| susceptibility                      | clinical        |         3 | `s` + `us` + `ceptibility`                         |   !! |
| ataxia                              | clinical        |         3 | `at` + `ax` + `ia`                                 |   !! |
| reinnervation                       | clinical        |         3 | `re` + `inner` + `vation`                          |   !! |
| electromyography                    | clinical        |         3 | `e` + `lect` + `romyography`                       |   !! |
| formulation                         | clinical        |         3 | `f` + `orm` + `ulation`                            |   !! |
| paralysis                           | clinical        |         3 | `p` + `ar` + `alysis`                              |   !! |
| depletion                           | clinical        |         3 | `de` + `ple` + `tion`                              |   !! |
| accumulation                        | clinical        |         3 | `ac` + `cum` + `ulation`                           |   !! |
| pathogenesis                        | clinical        |         3 | `path` + `ogen` + `esis`                           |   !! |
| condition                           | clinical        |         3 | `con` + `dit` + `ion`                              |   !! |
| duration                            | clinical        |         3 | `d` + `ur` + `ation`                               |   !! |
| malnutrition                        | clinical        |         3 | `m` + `alnutrit` + `ion`                           |   !! |
| degradation                         | clinical        |         3 | `de` + `gra` + `dation`                            |   !! |
| motion                              | clinical        |         3 | `m` + `ot` + `ion`                                 |   !! |
| availability                        | clinical        |         3 | `ava` + `il` + `ability`                           |   !! |
| option                              | clinical        |         3 | `o` + `pt` + `ion`                                 |   !! |
| tracheostomy                        | clinical        |         3 | `trac` + `heost` + `omy`                           |   !! |
| Phase                               | drug            |         3 | `P` + `h` + `ase`                                  |   !! |
| decline                             | drug            |         3 | `de` + `cl` + `ine`                                |   !! |
| sodium phenylbutyrate               | drug            |         3 | `s` + `odium` + `Ġphenylbutyrate`                  |   !! |
| baseline                            | drug            |         3 | `b` + `ase` + `line`                               |   !! |
| baclofen                            | drug            |         3 | `b` + `acl` + `ofen`                               |   !! |
| tizanidine                          | drug            |         3 | `t` + `iz` + `anidine`                             |   !! |
| dantrolene                          | drug            |         3 | `d` + `an` + `trolene`                             |   !! |
| increase                            | drug            |         3 | `in` + `cre` + `ase`                               |   !! |
| April                               | drug            |         3 | `A` + `pr` + `il`                                  |   !! |
| nuedexta                            | drug            |         3 | `n` + `u` + `edexta`                               |   !! |
| rilutek                             | drug            |         3 | `r` + `ilu` + `tek`                                |   !! |
| TDP-43                              | gene            |         3 | `TDP` + `-` + `43`                                 |   !! |
| BIIB078                             | gene            |         3 | `B` + `IIB` + `078`                                |   !! |
| TARDBP                              | gene            |         3 | `TA` + `R` + `DBP`                                 |   !! |
| D90A                                | gene            |         3 | `D` + `90` + `A`                                   |   !! |
| A4V                                 | gene            |         3 | `A` + `4` + `V`                                    |   !! |
| TBK1                                | gene            |         3 | `T` + `BK` + `1`                                   |   !! |
| OPTN                                | gene            |         3 | `O` + `PT` + `N`                                   |   !! |
| NEK1                                | gene            |         3 | `N` + `EK` + `1`                                   |   !! |
| FDA                                 | abbreviation    |         2 | `F` + `DA`                                         |      |
| NMDA                                | abbreviation    |         2 | `N` + `MDA`                                        |      |
| VALOR                               | abbreviation    |         2 | `VAL` + `OR`                                       |      |
| CAG                                 | abbreviation    |         2 | `C` + `AG`                                         |      |
| MSC                                 | abbreviation    |         2 | `M` + `SC`                                         |      |
| SBMA                                | abbreviation    |         2 | `S` + `BMA`                                        |      |
| dysfunction                         | clinical        |         2 | `dys` + `function`                                 |      |
| mutation                            | clinical        |         2 | `mut` + `ation`                                    |      |
| diagnosis                           | clinical        |         2 | `diagn` + `osis`                                   |      |
| bulbar onset                        | clinical        |         2 | `bulbar` + `Ġonset`                                |      |
| combination                         | clinical        |         2 | `com` + `bination`                                 |      |
| quality                             | clinical        |         2 | `qu` + `ality`                                     |      |
| ventilation                         | clinical        |         2 | `ventil` + `ation`                                 |      |
| conduction                          | clinical        |         2 | `con` + `duction`                                  |      |
| spasticity                          | clinical        |         2 | `sp` + `asticity`                                  |      |
| production                          | clinical        |         2 | `pro` + `duction`                                  |      |
| via                                 | clinical        |         2 | `v` + `ia`                                         |      |
| tolerability                        | clinical        |         2 | `t` + `olerability`                                |      |
| utility                             | clinical        |         2 | `ut` + `ility`                                     |      |
| stability                           | clinical        |         2 | `st` + `ability`                                   |      |
| formation                           | clinical        |         2 | `for` + `mation`                                   |      |
| inhibition                          | clinical        |         2 | `inhib` + `ition`                                  |      |
| stimulation                         | clinical        |         2 | `st` + `imulation`                                 |      |
| activation                          | clinical        |         2 | `activ` + `ation`                                  |      |
| generation                          | clinical        |         2 | `gener` + `ation`                                  |      |
| analysis                            | clinical        |         2 | `an` + `alysis`                                    |      |
| arthralgia                          | clinical        |         2 | `arthr` + `algia`                                  |      |
| injection                           | clinical        |         2 | `in` + `jection`                                   |      |
| frontotemporal dementia             | disease         |         2 | `frontotemporal` + `Ġdementia`                     |      |
| riluzole                            | drug            |         2 | `r` + `iluzole`                                    |      |
| tofersen                            | drug            |         2 | `t` + `ofersen`                                    |      |
| edaravone                           | drug            |         2 | `ed` + `aravone`                                   |      |
| release                             | drug            |         2 | `re` + `lease`                                     |      |
| taurursodiol                        | drug            |         2 | `t` + `aurursodiol`                                |      |
| tone                                | drug            |         2 | `t` + `one`                                        |      |
| quinidine                           | drug            |         2 | `qu` + `inidine`                                   |      |
| role                                | drug            |         2 | `ro` + `le`                                        |      |
| kinase                              | drug            |         2 | `kin` + `ase`                                      |      |
| bone                                | drug            |         2 | `b` + `one`                                        |      |
| alone                               | drug            |         2 | `al` + `one`                                       |      |
| SOD1                                | gene            |         2 | `SOD` + `1`                                        |      |
| FUS                                 | gene            |         2 | `F` + `US`                                         |      |
| ALS                                 | abbreviation    |         1 | `ALS`                                              |      |
| FVC                                 | abbreviation    |         1 | `FVC`                                              |      |
| FTD                                 | abbreviation    |         1 | `FTD`                                              |      |
| RNA                                 | abbreviation    |         1 | `RNA`                                              |      |
| CSF                                 | abbreviation    |         1 | `CSF`                                              |      |
| NfL                                 | abbreviation    |         1 | `NfL`                                              |      |
| UMN                                 | abbreviation    |         1 | `UMN`                                              |      |
| LMN                                 | abbreviation    |         1 | `LMN`                                              |      |
| PLS                                 | abbreviation    |         1 | `PLS`                                              |      |
| PMA                                 | abbreviation    |         1 | `PMA`                                              |      |
| ASO                                 | abbreviation    |         1 | `ASO`                                              |      |
| PBA                                 | abbreviation    |         1 | `PBA`                                              |      |
| iPSC                                | abbreviation    |         1 | `iPSC`                                             |      |
| EMG                                 | abbreviation    |         1 | `EMG`                                              |      |
| MND                                 | abbreviation    |         1 | `MND`                                              |      |
| PBP                                 | abbreviation    |         1 | `PBP`                                              |      |
| NIV                                 | abbreviation    |         1 | `NIV`                                              |      |
| BiPAP                               | abbreviation    |         1 | `BiPAP`                                            |      |
| PEG                                 | abbreviation    |         1 | `PEG`                                              |      |
| DTI                                 | abbreviation    |         1 | `DTI`                                              |      |
| PET                                 | abbreviation    |         1 | `PET`                                              |      |
| AAV                                 | abbreviation    |         1 | `AAV`                                              |      |
| HSP                                 | abbreviation    |         1 | `HSP`                                              |      |
| sALS                                | abbreviation    |         1 | `sALS`                                             |      |
| fALS                                | abbreviation    |         1 | `fALS`                                             |      |
| bvFTD                               | abbreviation    |         1 | `bvFTD`                                            |      |
| AAC                                 | abbreviation    |         1 | `AAC`                                              |      |
| SNIP                                | abbreviation    |         1 | `SNIP`                                             |      |
| SVC                                 | abbreviation    |         1 | `SVC`                                              |      |
| pNfH                                | abbreviation    |         1 | `pNfH`                                             |      |
| RNAi                                | abbreviation    |         1 | `RNAi`                                             |      |
| function                            | clinical        |         1 | `function`                                         |      |
| sclerosis                           | clinical        |         1 | `sclerosis`                                        |      |
| dysphagia                           | clinical        |         1 | `dysphagia`                                        |      |
| action                              | clinical        |         1 | `action`                                           |      |
| ability                             | clinical        |         1 | `ability`                                          |      |
| dysarthria                          | clinical        |         1 | `dysarthria`                                       |      |
| one                                 | drug            |         1 | `one`                                              |      |

## als_tokenizer_32k

Vocabulary size: 3,379. Flagged 115 of 195 terms (59.0%).

### Category breakdown

Per-category flagging rates show which term types are most affected by fragmentation.

| Category             | Total | Flagged | Flagged % |
|----------------------|-------|---------|-----------|
| abbreviation         |    42 |       5 |     11.9% |
| clinical             |    97 |      69 |     71.1% |
| disease              |    10 |       9 |     90.0% |
| drug                 |    31 |      19 |     61.3% |
| gene                 |    15 |      13 |     86.7% |

### Term details

Every term with its subtoken count and breakdown. Flagged terms (3+ subtokens) are marked.

| Term                                | Category        | Subtokens | Breakdown                                          | Flag |
|-------------------------------------|-----------------|-----------|----------------------------------------------------|------|
| loss-of-function                    | clinical        |         7 | `l` + `os` + `s` + `-` + `of` + `-` + `function`   |   !! |
| multifocal motor neuropathy         | disease         |         6 | `m` + `ult` + `ifoc` + `al` + `Ġmotor` + `Ġneuropathy` |   !! |
| excitotoxicity                      | clinical        |         5 | `e` + `x` + `c` + `it` + `otoxicity`               |   !! |
| non-invasive ventilation            | clinical        |         5 | `n` + `on` + `-` + `invasive` + `Ġventilation`     |   !! |
| disruption                          | clinical        |         5 | `d` + `is` + `ru` + `pt` + `ion`                   |   !! |
| neurodegeneration                   | clinical        |         5 | `n` + `euro` + `de` + `gener` + `ation`            |   !! |
| amyotrophic lateral sclerosis       | disease         |         5 | `am` + `y` + `otrophic` + `Ġlateral` + `Ġsclerosis` |   !! |
| primary lateral sclerosis           | disease         |         5 | `p` + `rim` + `ary` + `Ġlateral` + `Ġsclerosis`    |   !! |
| progressive muscular atrophy        | disease         |         5 | `p` + `rog` + `ressive` + `Ġmuscular` + `Ġatrophy` |   !! |
| spinocerebellar ataxia              | disease         |         5 | `sp` + `inoce` + `rebell` + `ar` + `Ġataxia`       |   !! |
| progressive bulbar palsy            | disease         |         5 | `p` + `rog` + `ressive` + `Ġbulbar` + `Ġpalsy`     |   !! |
| CYP1A2                              | gene            |         5 | `C` + `YP` + `1` + `A` + `2`                       |   !! |
| ALSFRS-R                            | abbreviation    |         4 | `ALS` + `FRS` + `-` + `R`                          |   !! |
| El Escorial criteria                | clinical        |         4 | `E` + `l` + `ĠEscorial` + `Ġcriteria`              |   !! |
| aggregation                         | clinical        |         4 | `ag` + `g` + `reg` + `ation`                       |   !! |
| interaction                         | clinical        |         4 | `in` + `t` + `era` + `ction`                       |   !! |
| Awaji criteria                      | clinical        |         4 | `A` + `waj` + `i` + `Ġcriteria`                    |   !! |
| examination                         | clinical        |         4 | `e` + `x` + `am` + `ination`                       |   !! |
| autophagy                           | clinical        |         4 | `a` + `ut` + `ophag` + `y`                         |   !! |
| hypothesis                          | clinical        |         4 | `h` + `yp` + `othes` + `is`                        |   !! |
| hyperreflexia                       | clinical        |         4 | `h` + `yp` + `errefle` + `xia`                     |   !! |
| Babinski sign                       | clinical        |         4 | `B` + `abins` + `ki` + `Ġsign`                     |   !! |
| diaphragmatic pacing                | clinical        |         4 | `d` + `iaph` + `ragmatic` + `Ġpacing`              |   !! |
| percutaneous endoscopic gastrostomy | clinical        |         4 | `per` + `cutaneous` + `Ġendoscopic` + `Ġgastrostomy` |   !! |
| distribution                        | clinical        |         4 | `d` + `is` + `trib` + `ution`                      |   !! |
| modification                        | clinical        |         4 | `m` + `od` + `ific` + `ation`                      |   !! |
| mislocalization                     | clinical        |         4 | `m` + `is` + `localiz` + `ation`                   |   !! |
| overactivation                      | clinical        |         4 | `o` + `ver` + `activ` + `ation`                    |   !! |
| paraplegia                          | clinical        |         4 | `p` + `ar` + `ap` + `legia`                        |   !! |
| hypoventilation                     | clinical        |         4 | `h` + `y` + `poventil` + `ation`                   |   !! |
| Gold Coast criteria                 | clinical        |         4 | `G` + `old` + `ĠCoast` + `Ġcriteria`               |   !! |
| Hepatotoxicity                      | clinical        |         4 | `H` + `e` + `pat` + `otoxicity`                    |   !! |
| personality                         | clinical        |         4 | `p` + `ers` + `on` + `ality`                       |   !! |
| disinhibition                       | clinical        |         4 | `d` + `is` + `inhib` + `ition`                     |   !! |
| spirometry                          | clinical        |         4 | `sp` + `i` + `rom` + `etry`                        |   !! |
| apoptosis                           | clinical        |         4 | `a` + `po` + `pt` + `osis`                         |   !! |
| motor neuron disease                | disease         |         4 | `m` + `otor` + `Ġneuron` + `Ġdisease`              |   !! |
| Kennedy disease                     | disease         |         4 | `K` + `enned` + `y` + `Ġdisease`                   |   !! |
| hereditary spastic paraplegia       | disease         |         4 | `he` + `reditary` + `Ġspastic` + `Ġparaplegia`     |   !! |
| disease                             | drug            |         4 | `d` + `is` + `e` + `ase`                           |   !! |
| AMX0035                             | drug            |         4 | `A` + `M` + `X` + `0035`                           |   !! |
| dismutase                           | drug            |         4 | `d` + `ism` + `ut` + `ase`                         |   !! |
| botulinum toxin                     | drug            |         4 | `b` + `ot` + `ulinum` + `Ġtoxin`                   |   !! |
| polyglutamine                       | drug            |         4 | `po` + `ly` + `glutam` + `ine`                     |   !! |
| dextromethorphan                    | drug            |         4 | `de` + `xt` + `romethor` + `phan`                  |   !! |
| benzothiazole                       | drug            |         4 | `b` + `en` + `zoth` + `iazole`                     |   !! |
| radicava                            | drug            |         4 | `ra` + `d` + `ic` + `ava`                          |   !! |
| C9orf72                             | gene            |         4 | `C` + `9` + `orf` + `72`                           |   !! |
| ATXN2                               | gene            |         4 | `A` + `TX` + `N` + `2`                             |   !! |
| SOD1-ALS                            | gene            |         4 | `SOD` + `1` + `-` + `ALS`                          |   !! |
| GluR2                               | gene            |         4 | `G` + `lu` + `R` + `2`                             |   !! |
| AMPA                                | abbreviation    |         3 | `A` + `M` + `PA`                                   |   !! |
| GGGGCC                              | abbreviation    |         3 | `GG` + `GG` + `CC`                                 |   !! |
| ALS-FTD                             | abbreviation    |         3 | `ALS` + `-` + `FTD`                                |   !! |
| CENTAUR                             | abbreviation    |         3 | `C` + `EN` + `TAUR`                                |   !! |
| criteria                            | clinical        |         3 | `c` + `rit` + `eria`                               |   !! |
| degeneration                        | clinical        |         3 | `de` + `gener` + `ation`                           |   !! |
| denervation                         | clinical        |         3 | `d` + `ener` + `vation`                            |   !! |
| communication                       | clinical        |         3 | `com` + `m` + `unication`                          |   !! |
| capacity                            | clinical        |         3 | `cap` + `ac` + `ity`                               |   !! |
| neuroinflammation                   | clinical        |         3 | `n` + `euro` + `inflammation`                      |   !! |
| prognosis                           | clinical        |         3 | `pro` + `gn` + `osis`                              |   !! |
| atrophy                             | clinical        |         3 | `at` + `roph` + `y`                                |   !! |
| limb onset                          | clinical        |         3 | `l` + `imb` + `Ġonset`                             |   !! |
| proteostasis                        | clinical        |         3 | `pro` + `te` + `ostasis`                           |   !! |
| microglia                           | clinical        |         3 | `m` + `ic` + `roglia`                              |   !! |
| population                          | clinical        |         3 | `p` + `op` + `ulation`                             |   !! |
| gastrostomy                         | clinical        |         3 | `g` + `astro` + `stomy`                            |   !! |
| respiratory failure                 | clinical        |         3 | `re` + `spiratory` + `Ġfailure`                    |   !! |
| fasciculation                       | clinical        |         3 | `f` + `ascic` + `ulation`                          |   !! |
| dementia                            | clinical        |         3 | `de` + `ment` + `ia`                               |   !! |
| sensitivity                         | clinical        |         3 | `s` + `ens` + `itivity`                            |   !! |
| asthenia                            | clinical        |         3 | `as` + `th` + `enia`                               |   !! |
| concentration                       | clinical        |         3 | `con` + `cent` + `ration`                          |   !! |
| tomography                          | clinical        |         3 | `t` + `om` + `ography`                             |   !! |
| clonus                              | clinical        |         3 | `cl` + `on` + `us`                                 |   !! |
| sialorrhea                          | clinical        |         3 | `s` + `ial` + `orrhea`                             |   !! |
| instability                         | clinical        |         3 | `in` + `st` + `ability`                            |   !! |
| reduction                           | clinical        |         3 | `red` + `u` + `ction`                              |   !! |
| susceptibility                      | clinical        |         3 | `s` + `us` + `ceptibility`                         |   !! |
| ataxia                              | clinical        |         3 | `at` + `ax` + `ia`                                 |   !! |
| reinnervation                       | clinical        |         3 | `re` + `inner` + `vation`                          |   !! |
| electromyography                    | clinical        |         3 | `e` + `lect` + `romyography`                       |   !! |
| formulation                         | clinical        |         3 | `f` + `orm` + `ulation`                            |   !! |
| paralysis                           | clinical        |         3 | `p` + `ar` + `alysis`                              |   !! |
| depletion                           | clinical        |         3 | `de` + `ple` + `tion`                              |   !! |
| accumulation                        | clinical        |         3 | `ac` + `cum` + `ulation`                           |   !! |
| pathogenesis                        | clinical        |         3 | `path` + `ogen` + `esis`                           |   !! |
| condition                           | clinical        |         3 | `con` + `dit` + `ion`                              |   !! |
| duration                            | clinical        |         3 | `d` + `ur` + `ation`                               |   !! |
| malnutrition                        | clinical        |         3 | `m` + `alnutrit` + `ion`                           |   !! |
| degradation                         | clinical        |         3 | `de` + `gra` + `dation`                            |   !! |
| motion                              | clinical        |         3 | `m` + `ot` + `ion`                                 |   !! |
| availability                        | clinical        |         3 | `ava` + `il` + `ability`                           |   !! |
| option                              | clinical        |         3 | `o` + `pt` + `ion`                                 |   !! |
| tracheostomy                        | clinical        |         3 | `trac` + `heost` + `omy`                           |   !! |
| Phase                               | drug            |         3 | `P` + `h` + `ase`                                  |   !! |
| decline                             | drug            |         3 | `de` + `cl` + `ine`                                |   !! |
| sodium phenylbutyrate               | drug            |         3 | `s` + `odium` + `Ġphenylbutyrate`                  |   !! |
| baseline                            | drug            |         3 | `b` + `ase` + `line`                               |   !! |
| baclofen                            | drug            |         3 | `b` + `acl` + `ofen`                               |   !! |
| tizanidine                          | drug            |         3 | `t` + `iz` + `anidine`                             |   !! |
| dantrolene                          | drug            |         3 | `d` + `an` + `trolene`                             |   !! |
| increase                            | drug            |         3 | `in` + `cre` + `ase`                               |   !! |
| April                               | drug            |         3 | `A` + `pr` + `il`                                  |   !! |
| nuedexta                            | drug            |         3 | `n` + `u` + `edexta`                               |   !! |
| rilutek                             | drug            |         3 | `r` + `ilu` + `tek`                                |   !! |
| TDP-43                              | gene            |         3 | `TDP` + `-` + `43`                                 |   !! |
| BIIB078                             | gene            |         3 | `B` + `IIB` + `078`                                |   !! |
| TARDBP                              | gene            |         3 | `TA` + `R` + `DBP`                                 |   !! |
| D90A                                | gene            |         3 | `D` + `90` + `A`                                   |   !! |
| A4V                                 | gene            |         3 | `A` + `4` + `V`                                    |   !! |
| TBK1                                | gene            |         3 | `T` + `BK` + `1`                                   |   !! |
| OPTN                                | gene            |         3 | `O` + `PT` + `N`                                   |   !! |
| NEK1                                | gene            |         3 | `N` + `EK` + `1`                                   |   !! |
| FDA                                 | abbreviation    |         2 | `F` + `DA`                                         |      |
| NMDA                                | abbreviation    |         2 | `N` + `MDA`                                        |      |
| VALOR                               | abbreviation    |         2 | `VAL` + `OR`                                       |      |
| CAG                                 | abbreviation    |         2 | `C` + `AG`                                         |      |
| MSC                                 | abbreviation    |         2 | `M` + `SC`                                         |      |
| SBMA                                | abbreviation    |         2 | `S` + `BMA`                                        |      |
| dysfunction                         | clinical        |         2 | `dys` + `function`                                 |      |
| mutation                            | clinical        |         2 | `mut` + `ation`                                    |      |
| diagnosis                           | clinical        |         2 | `diagn` + `osis`                                   |      |
| bulbar onset                        | clinical        |         2 | `bulbar` + `Ġonset`                                |      |
| combination                         | clinical        |         2 | `com` + `bination`                                 |      |
| quality                             | clinical        |         2 | `qu` + `ality`                                     |      |
| ventilation                         | clinical        |         2 | `ventil` + `ation`                                 |      |
| conduction                          | clinical        |         2 | `con` + `duction`                                  |      |
| spasticity                          | clinical        |         2 | `sp` + `asticity`                                  |      |
| production                          | clinical        |         2 | `pro` + `duction`                                  |      |
| via                                 | clinical        |         2 | `v` + `ia`                                         |      |
| tolerability                        | clinical        |         2 | `t` + `olerability`                                |      |
| utility                             | clinical        |         2 | `ut` + `ility`                                     |      |
| stability                           | clinical        |         2 | `st` + `ability`                                   |      |
| formation                           | clinical        |         2 | `for` + `mation`                                   |      |
| inhibition                          | clinical        |         2 | `inhib` + `ition`                                  |      |
| stimulation                         | clinical        |         2 | `st` + `imulation`                                 |      |
| activation                          | clinical        |         2 | `activ` + `ation`                                  |      |
| generation                          | clinical        |         2 | `gener` + `ation`                                  |      |
| analysis                            | clinical        |         2 | `an` + `alysis`                                    |      |
| arthralgia                          | clinical        |         2 | `arthr` + `algia`                                  |      |
| injection                           | clinical        |         2 | `in` + `jection`                                   |      |
| frontotemporal dementia             | disease         |         2 | `frontotemporal` + `Ġdementia`                     |      |
| riluzole                            | drug            |         2 | `r` + `iluzole`                                    |      |
| tofersen                            | drug            |         2 | `t` + `ofersen`                                    |      |
| edaravone                           | drug            |         2 | `ed` + `aravone`                                   |      |
| release                             | drug            |         2 | `re` + `lease`                                     |      |
| taurursodiol                        | drug            |         2 | `t` + `aurursodiol`                                |      |
| tone                                | drug            |         2 | `t` + `one`                                        |      |
| quinidine                           | drug            |         2 | `qu` + `inidine`                                   |      |
| role                                | drug            |         2 | `ro` + `le`                                        |      |
| kinase                              | drug            |         2 | `kin` + `ase`                                      |      |
| bone                                | drug            |         2 | `b` + `one`                                        |      |
| alone                               | drug            |         2 | `al` + `one`                                       |      |
| SOD1                                | gene            |         2 | `SOD` + `1`                                        |      |
| FUS                                 | gene            |         2 | `F` + `US`                                         |      |
| ALS                                 | abbreviation    |         1 | `ALS`                                              |      |
| FVC                                 | abbreviation    |         1 | `FVC`                                              |      |
| FTD                                 | abbreviation    |         1 | `FTD`                                              |      |
| RNA                                 | abbreviation    |         1 | `RNA`                                              |      |
| CSF                                 | abbreviation    |         1 | `CSF`                                              |      |
| NfL                                 | abbreviation    |         1 | `NfL`                                              |      |
| UMN                                 | abbreviation    |         1 | `UMN`                                              |      |
| LMN                                 | abbreviation    |         1 | `LMN`                                              |      |
| PLS                                 | abbreviation    |         1 | `PLS`                                              |      |
| PMA                                 | abbreviation    |         1 | `PMA`                                              |      |
| ASO                                 | abbreviation    |         1 | `ASO`                                              |      |
| PBA                                 | abbreviation    |         1 | `PBA`                                              |      |
| iPSC                                | abbreviation    |         1 | `iPSC`                                             |      |
| EMG                                 | abbreviation    |         1 | `EMG`                                              |      |
| MND                                 | abbreviation    |         1 | `MND`                                              |      |
| PBP                                 | abbreviation    |         1 | `PBP`                                              |      |
| NIV                                 | abbreviation    |         1 | `NIV`                                              |      |
| BiPAP                               | abbreviation    |         1 | `BiPAP`                                            |      |
| PEG                                 | abbreviation    |         1 | `PEG`                                              |      |
| DTI                                 | abbreviation    |         1 | `DTI`                                              |      |
| PET                                 | abbreviation    |         1 | `PET`                                              |      |
| AAV                                 | abbreviation    |         1 | `AAV`                                              |      |
| HSP                                 | abbreviation    |         1 | `HSP`                                              |      |
| sALS                                | abbreviation    |         1 | `sALS`                                             |      |
| fALS                                | abbreviation    |         1 | `fALS`                                             |      |
| bvFTD                               | abbreviation    |         1 | `bvFTD`                                            |      |
| AAC                                 | abbreviation    |         1 | `AAC`                                              |      |
| SNIP                                | abbreviation    |         1 | `SNIP`                                             |      |
| SVC                                 | abbreviation    |         1 | `SVC`                                              |      |
| pNfH                                | abbreviation    |         1 | `pNfH`                                             |      |
| RNAi                                | abbreviation    |         1 | `RNAi`                                             |      |
| function                            | clinical        |         1 | `function`                                         |      |
| sclerosis                           | clinical        |         1 | `sclerosis`                                        |      |
| dysphagia                           | clinical        |         1 | `dysphagia`                                        |      |
| action                              | clinical        |         1 | `action`                                           |      |
| ability                             | clinical        |         1 | `ability`                                          |      |
| dysarthria                          | clinical        |         1 | `dysarthria`                                       |      |
| one                                 | drug            |         1 | `one`                                              |      |

## als_tokenizer_50k

Vocabulary size: 3,379. Flagged 115 of 195 terms (59.0%).

### Category breakdown

Per-category flagging rates show which term types are most affected by fragmentation.

| Category             | Total | Flagged | Flagged % |
|----------------------|-------|---------|-----------|
| abbreviation         |    42 |       5 |     11.9% |
| clinical             |    97 |      69 |     71.1% |
| disease              |    10 |       9 |     90.0% |
| drug                 |    31 |      19 |     61.3% |
| gene                 |    15 |      13 |     86.7% |

### Term details

Every term with its subtoken count and breakdown. Flagged terms (3+ subtokens) are marked.

| Term                                | Category        | Subtokens | Breakdown                                          | Flag |
|-------------------------------------|-----------------|-----------|----------------------------------------------------|------|
| loss-of-function                    | clinical        |         7 | `l` + `os` + `s` + `-` + `of` + `-` + `function`   |   !! |
| multifocal motor neuropathy         | disease         |         6 | `m` + `ult` + `ifoc` + `al` + `Ġmotor` + `Ġneuropathy` |   !! |
| excitotoxicity                      | clinical        |         5 | `e` + `x` + `c` + `it` + `otoxicity`               |   !! |
| non-invasive ventilation            | clinical        |         5 | `n` + `on` + `-` + `invasive` + `Ġventilation`     |   !! |
| disruption                          | clinical        |         5 | `d` + `is` + `ru` + `pt` + `ion`                   |   !! |
| neurodegeneration                   | clinical        |         5 | `n` + `euro` + `de` + `gener` + `ation`            |   !! |
| amyotrophic lateral sclerosis       | disease         |         5 | `am` + `y` + `otrophic` + `Ġlateral` + `Ġsclerosis` |   !! |
| primary lateral sclerosis           | disease         |         5 | `p` + `rim` + `ary` + `Ġlateral` + `Ġsclerosis`    |   !! |
| progressive muscular atrophy        | disease         |         5 | `p` + `rog` + `ressive` + `Ġmuscular` + `Ġatrophy` |   !! |
| spinocerebellar ataxia              | disease         |         5 | `sp` + `inoce` + `rebell` + `ar` + `Ġataxia`       |   !! |
| progressive bulbar palsy            | disease         |         5 | `p` + `rog` + `ressive` + `Ġbulbar` + `Ġpalsy`     |   !! |
| CYP1A2                              | gene            |         5 | `C` + `YP` + `1` + `A` + `2`                       |   !! |
| ALSFRS-R                            | abbreviation    |         4 | `ALS` + `FRS` + `-` + `R`                          |   !! |
| El Escorial criteria                | clinical        |         4 | `E` + `l` + `ĠEscorial` + `Ġcriteria`              |   !! |
| aggregation                         | clinical        |         4 | `ag` + `g` + `reg` + `ation`                       |   !! |
| interaction                         | clinical        |         4 | `in` + `t` + `era` + `ction`                       |   !! |
| Awaji criteria                      | clinical        |         4 | `A` + `waj` + `i` + `Ġcriteria`                    |   !! |
| examination                         | clinical        |         4 | `e` + `x` + `am` + `ination`                       |   !! |
| autophagy                           | clinical        |         4 | `a` + `ut` + `ophag` + `y`                         |   !! |
| hypothesis                          | clinical        |         4 | `h` + `yp` + `othes` + `is`                        |   !! |
| hyperreflexia                       | clinical        |         4 | `h` + `yp` + `errefle` + `xia`                     |   !! |
| Babinski sign                       | clinical        |         4 | `B` + `abins` + `ki` + `Ġsign`                     |   !! |
| diaphragmatic pacing                | clinical        |         4 | `d` + `iaph` + `ragmatic` + `Ġpacing`              |   !! |
| percutaneous endoscopic gastrostomy | clinical        |         4 | `per` + `cutaneous` + `Ġendoscopic` + `Ġgastrostomy` |   !! |
| distribution                        | clinical        |         4 | `d` + `is` + `trib` + `ution`                      |   !! |
| modification                        | clinical        |         4 | `m` + `od` + `ific` + `ation`                      |   !! |
| mislocalization                     | clinical        |         4 | `m` + `is` + `localiz` + `ation`                   |   !! |
| overactivation                      | clinical        |         4 | `o` + `ver` + `activ` + `ation`                    |   !! |
| paraplegia                          | clinical        |         4 | `p` + `ar` + `ap` + `legia`                        |   !! |
| hypoventilation                     | clinical        |         4 | `h` + `y` + `poventil` + `ation`                   |   !! |
| Gold Coast criteria                 | clinical        |         4 | `G` + `old` + `ĠCoast` + `Ġcriteria`               |   !! |
| Hepatotoxicity                      | clinical        |         4 | `H` + `e` + `pat` + `otoxicity`                    |   !! |
| personality                         | clinical        |         4 | `p` + `ers` + `on` + `ality`                       |   !! |
| disinhibition                       | clinical        |         4 | `d` + `is` + `inhib` + `ition`                     |   !! |
| spirometry                          | clinical        |         4 | `sp` + `i` + `rom` + `etry`                        |   !! |
| apoptosis                           | clinical        |         4 | `a` + `po` + `pt` + `osis`                         |   !! |
| motor neuron disease                | disease         |         4 | `m` + `otor` + `Ġneuron` + `Ġdisease`              |   !! |
| Kennedy disease                     | disease         |         4 | `K` + `enned` + `y` + `Ġdisease`                   |   !! |
| hereditary spastic paraplegia       | disease         |         4 | `he` + `reditary` + `Ġspastic` + `Ġparaplegia`     |   !! |
| disease                             | drug            |         4 | `d` + `is` + `e` + `ase`                           |   !! |
| AMX0035                             | drug            |         4 | `A` + `M` + `X` + `0035`                           |   !! |
| dismutase                           | drug            |         4 | `d` + `ism` + `ut` + `ase`                         |   !! |
| botulinum toxin                     | drug            |         4 | `b` + `ot` + `ulinum` + `Ġtoxin`                   |   !! |
| polyglutamine                       | drug            |         4 | `po` + `ly` + `glutam` + `ine`                     |   !! |
| dextromethorphan                    | drug            |         4 | `de` + `xt` + `romethor` + `phan`                  |   !! |
| benzothiazole                       | drug            |         4 | `b` + `en` + `zoth` + `iazole`                     |   !! |
| radicava                            | drug            |         4 | `ra` + `d` + `ic` + `ava`                          |   !! |
| C9orf72                             | gene            |         4 | `C` + `9` + `orf` + `72`                           |   !! |
| ATXN2                               | gene            |         4 | `A` + `TX` + `N` + `2`                             |   !! |
| SOD1-ALS                            | gene            |         4 | `SOD` + `1` + `-` + `ALS`                          |   !! |
| GluR2                               | gene            |         4 | `G` + `lu` + `R` + `2`                             |   !! |
| AMPA                                | abbreviation    |         3 | `A` + `M` + `PA`                                   |   !! |
| GGGGCC                              | abbreviation    |         3 | `GG` + `GG` + `CC`                                 |   !! |
| ALS-FTD                             | abbreviation    |         3 | `ALS` + `-` + `FTD`                                |   !! |
| CENTAUR                             | abbreviation    |         3 | `C` + `EN` + `TAUR`                                |   !! |
| criteria                            | clinical        |         3 | `c` + `rit` + `eria`                               |   !! |
| degeneration                        | clinical        |         3 | `de` + `gener` + `ation`                           |   !! |
| denervation                         | clinical        |         3 | `d` + `ener` + `vation`                            |   !! |
| communication                       | clinical        |         3 | `com` + `m` + `unication`                          |   !! |
| capacity                            | clinical        |         3 | `cap` + `ac` + `ity`                               |   !! |
| neuroinflammation                   | clinical        |         3 | `n` + `euro` + `inflammation`                      |   !! |
| prognosis                           | clinical        |         3 | `pro` + `gn` + `osis`                              |   !! |
| atrophy                             | clinical        |         3 | `at` + `roph` + `y`                                |   !! |
| limb onset                          | clinical        |         3 | `l` + `imb` + `Ġonset`                             |   !! |
| proteostasis                        | clinical        |         3 | `pro` + `te` + `ostasis`                           |   !! |
| microglia                           | clinical        |         3 | `m` + `ic` + `roglia`                              |   !! |
| population                          | clinical        |         3 | `p` + `op` + `ulation`                             |   !! |
| gastrostomy                         | clinical        |         3 | `g` + `astro` + `stomy`                            |   !! |
| respiratory failure                 | clinical        |         3 | `re` + `spiratory` + `Ġfailure`                    |   !! |
| fasciculation                       | clinical        |         3 | `f` + `ascic` + `ulation`                          |   !! |
| dementia                            | clinical        |         3 | `de` + `ment` + `ia`                               |   !! |
| sensitivity                         | clinical        |         3 | `s` + `ens` + `itivity`                            |   !! |
| asthenia                            | clinical        |         3 | `as` + `th` + `enia`                               |   !! |
| concentration                       | clinical        |         3 | `con` + `cent` + `ration`                          |   !! |
| tomography                          | clinical        |         3 | `t` + `om` + `ography`                             |   !! |
| clonus                              | clinical        |         3 | `cl` + `on` + `us`                                 |   !! |
| sialorrhea                          | clinical        |         3 | `s` + `ial` + `orrhea`                             |   !! |
| instability                         | clinical        |         3 | `in` + `st` + `ability`                            |   !! |
| reduction                           | clinical        |         3 | `red` + `u` + `ction`                              |   !! |
| susceptibility                      | clinical        |         3 | `s` + `us` + `ceptibility`                         |   !! |
| ataxia                              | clinical        |         3 | `at` + `ax` + `ia`                                 |   !! |
| reinnervation                       | clinical        |         3 | `re` + `inner` + `vation`                          |   !! |
| electromyography                    | clinical        |         3 | `e` + `lect` + `romyography`                       |   !! |
| formulation                         | clinical        |         3 | `f` + `orm` + `ulation`                            |   !! |
| paralysis                           | clinical        |         3 | `p` + `ar` + `alysis`                              |   !! |
| depletion                           | clinical        |         3 | `de` + `ple` + `tion`                              |   !! |
| accumulation                        | clinical        |         3 | `ac` + `cum` + `ulation`                           |   !! |
| pathogenesis                        | clinical        |         3 | `path` + `ogen` + `esis`                           |   !! |
| condition                           | clinical        |         3 | `con` + `dit` + `ion`                              |   !! |
| duration                            | clinical        |         3 | `d` + `ur` + `ation`                               |   !! |
| malnutrition                        | clinical        |         3 | `m` + `alnutrit` + `ion`                           |   !! |
| degradation                         | clinical        |         3 | `de` + `gra` + `dation`                            |   !! |
| motion                              | clinical        |         3 | `m` + `ot` + `ion`                                 |   !! |
| availability                        | clinical        |         3 | `ava` + `il` + `ability`                           |   !! |
| option                              | clinical        |         3 | `o` + `pt` + `ion`                                 |   !! |
| tracheostomy                        | clinical        |         3 | `trac` + `heost` + `omy`                           |   !! |
| Phase                               | drug            |         3 | `P` + `h` + `ase`                                  |   !! |
| decline                             | drug            |         3 | `de` + `cl` + `ine`                                |   !! |
| sodium phenylbutyrate               | drug            |         3 | `s` + `odium` + `Ġphenylbutyrate`                  |   !! |
| baseline                            | drug            |         3 | `b` + `ase` + `line`                               |   !! |
| baclofen                            | drug            |         3 | `b` + `acl` + `ofen`                               |   !! |
| tizanidine                          | drug            |         3 | `t` + `iz` + `anidine`                             |   !! |
| dantrolene                          | drug            |         3 | `d` + `an` + `trolene`                             |   !! |
| increase                            | drug            |         3 | `in` + `cre` + `ase`                               |   !! |
| April                               | drug            |         3 | `A` + `pr` + `il`                                  |   !! |
| nuedexta                            | drug            |         3 | `n` + `u` + `edexta`                               |   !! |
| rilutek                             | drug            |         3 | `r` + `ilu` + `tek`                                |   !! |
| TDP-43                              | gene            |         3 | `TDP` + `-` + `43`                                 |   !! |
| BIIB078                             | gene            |         3 | `B` + `IIB` + `078`                                |   !! |
| TARDBP                              | gene            |         3 | `TA` + `R` + `DBP`                                 |   !! |
| D90A                                | gene            |         3 | `D` + `90` + `A`                                   |   !! |
| A4V                                 | gene            |         3 | `A` + `4` + `V`                                    |   !! |
| TBK1                                | gene            |         3 | `T` + `BK` + `1`                                   |   !! |
| OPTN                                | gene            |         3 | `O` + `PT` + `N`                                   |   !! |
| NEK1                                | gene            |         3 | `N` + `EK` + `1`                                   |   !! |
| FDA                                 | abbreviation    |         2 | `F` + `DA`                                         |      |
| NMDA                                | abbreviation    |         2 | `N` + `MDA`                                        |      |
| VALOR                               | abbreviation    |         2 | `VAL` + `OR`                                       |      |
| CAG                                 | abbreviation    |         2 | `C` + `AG`                                         |      |
| MSC                                 | abbreviation    |         2 | `M` + `SC`                                         |      |
| SBMA                                | abbreviation    |         2 | `S` + `BMA`                                        |      |
| dysfunction                         | clinical        |         2 | `dys` + `function`                                 |      |
| mutation                            | clinical        |         2 | `mut` + `ation`                                    |      |
| diagnosis                           | clinical        |         2 | `diagn` + `osis`                                   |      |
| bulbar onset                        | clinical        |         2 | `bulbar` + `Ġonset`                                |      |
| combination                         | clinical        |         2 | `com` + `bination`                                 |      |
| quality                             | clinical        |         2 | `qu` + `ality`                                     |      |
| ventilation                         | clinical        |         2 | `ventil` + `ation`                                 |      |
| conduction                          | clinical        |         2 | `con` + `duction`                                  |      |
| spasticity                          | clinical        |         2 | `sp` + `asticity`                                  |      |
| production                          | clinical        |         2 | `pro` + `duction`                                  |      |
| via                                 | clinical        |         2 | `v` + `ia`                                         |      |
| tolerability                        | clinical        |         2 | `t` + `olerability`                                |      |
| utility                             | clinical        |         2 | `ut` + `ility`                                     |      |
| stability                           | clinical        |         2 | `st` + `ability`                                   |      |
| formation                           | clinical        |         2 | `for` + `mation`                                   |      |
| inhibition                          | clinical        |         2 | `inhib` + `ition`                                  |      |
| stimulation                         | clinical        |         2 | `st` + `imulation`                                 |      |
| activation                          | clinical        |         2 | `activ` + `ation`                                  |      |
| generation                          | clinical        |         2 | `gener` + `ation`                                  |      |
| analysis                            | clinical        |         2 | `an` + `alysis`                                    |      |
| arthralgia                          | clinical        |         2 | `arthr` + `algia`                                  |      |
| injection                           | clinical        |         2 | `in` + `jection`                                   |      |
| frontotemporal dementia             | disease         |         2 | `frontotemporal` + `Ġdementia`                     |      |
| riluzole                            | drug            |         2 | `r` + `iluzole`                                    |      |
| tofersen                            | drug            |         2 | `t` + `ofersen`                                    |      |
| edaravone                           | drug            |         2 | `ed` + `aravone`                                   |      |
| release                             | drug            |         2 | `re` + `lease`                                     |      |
| taurursodiol                        | drug            |         2 | `t` + `aurursodiol`                                |      |
| tone                                | drug            |         2 | `t` + `one`                                        |      |
| quinidine                           | drug            |         2 | `qu` + `inidine`                                   |      |
| role                                | drug            |         2 | `ro` + `le`                                        |      |
| kinase                              | drug            |         2 | `kin` + `ase`                                      |      |
| bone                                | drug            |         2 | `b` + `one`                                        |      |
| alone                               | drug            |         2 | `al` + `one`                                       |      |
| SOD1                                | gene            |         2 | `SOD` + `1`                                        |      |
| FUS                                 | gene            |         2 | `F` + `US`                                         |      |
| ALS                                 | abbreviation    |         1 | `ALS`                                              |      |
| FVC                                 | abbreviation    |         1 | `FVC`                                              |      |
| FTD                                 | abbreviation    |         1 | `FTD`                                              |      |
| RNA                                 | abbreviation    |         1 | `RNA`                                              |      |
| CSF                                 | abbreviation    |         1 | `CSF`                                              |      |
| NfL                                 | abbreviation    |         1 | `NfL`                                              |      |
| UMN                                 | abbreviation    |         1 | `UMN`                                              |      |
| LMN                                 | abbreviation    |         1 | `LMN`                                              |      |
| PLS                                 | abbreviation    |         1 | `PLS`                                              |      |
| PMA                                 | abbreviation    |         1 | `PMA`                                              |      |
| ASO                                 | abbreviation    |         1 | `ASO`                                              |      |
| PBA                                 | abbreviation    |         1 | `PBA`                                              |      |
| iPSC                                | abbreviation    |         1 | `iPSC`                                             |      |
| EMG                                 | abbreviation    |         1 | `EMG`                                              |      |
| MND                                 | abbreviation    |         1 | `MND`                                              |      |
| PBP                                 | abbreviation    |         1 | `PBP`                                              |      |
| NIV                                 | abbreviation    |         1 | `NIV`                                              |      |
| BiPAP                               | abbreviation    |         1 | `BiPAP`                                            |      |
| PEG                                 | abbreviation    |         1 | `PEG`                                              |      |
| DTI                                 | abbreviation    |         1 | `DTI`                                              |      |
| PET                                 | abbreviation    |         1 | `PET`                                              |      |
| AAV                                 | abbreviation    |         1 | `AAV`                                              |      |
| HSP                                 | abbreviation    |         1 | `HSP`                                              |      |
| sALS                                | abbreviation    |         1 | `sALS`                                             |      |
| fALS                                | abbreviation    |         1 | `fALS`                                             |      |
| bvFTD                               | abbreviation    |         1 | `bvFTD`                                            |      |
| AAC                                 | abbreviation    |         1 | `AAC`                                              |      |
| SNIP                                | abbreviation    |         1 | `SNIP`                                             |      |
| SVC                                 | abbreviation    |         1 | `SVC`                                              |      |
| pNfH                                | abbreviation    |         1 | `pNfH`                                             |      |
| RNAi                                | abbreviation    |         1 | `RNAi`                                             |      |
| function                            | clinical        |         1 | `function`                                         |      |
| sclerosis                           | clinical        |         1 | `sclerosis`                                        |      |
| dysphagia                           | clinical        |         1 | `dysphagia`                                        |      |
| action                              | clinical        |         1 | `action`                                           |      |
| ability                             | clinical        |         1 | `ability`                                          |      |
| dysarthria                          | clinical        |         1 | `dysarthria`                                       |      |
| one                                 | drug            |         1 | `one`                                              |      |

## Comparison summary

Side-by-side comparison of flagging rates across all tokenizers.

| Tokenizer                      | Vocab Size | Flagged | Flagged % |
|--------------------------------|------------|---------|-----------|
| als_tokenizer_16k              |      3,379 |     115 |     59.0% |
| als_tokenizer_32k              |      3,379 |     115 |     59.0% |
| als_tokenizer_50k              |      3,379 |     115 |     59.0% |

## Most fragmented terms

Terms sorted by worst fragmentation across all tokenizers, showing which medical vocabulary is hardest for BPE to learn.

| Term                                | Category        | Max Subtokens | Worst Tokenizer           |
|-------------------------------------|-----------------|---------------|---------------------------|
| loss-of-function                    | clinical        |             7 | als_tokenizer_16k         |
| multifocal motor neuropathy         | disease         |             6 | als_tokenizer_16k         |
| excitotoxicity                      | clinical        |             5 | als_tokenizer_16k         |
| non-invasive ventilation            | clinical        |             5 | als_tokenizer_16k         |
| disruption                          | clinical        |             5 | als_tokenizer_16k         |
| neurodegeneration                   | clinical        |             5 | als_tokenizer_16k         |
| amyotrophic lateral sclerosis       | disease         |             5 | als_tokenizer_16k         |
| primary lateral sclerosis           | disease         |             5 | als_tokenizer_16k         |
| progressive muscular atrophy        | disease         |             5 | als_tokenizer_16k         |
| spinocerebellar ataxia              | disease         |             5 | als_tokenizer_16k         |
| progressive bulbar palsy            | disease         |             5 | als_tokenizer_16k         |
| CYP1A2                              | gene            |             5 | als_tokenizer_16k         |
| ALSFRS-R                            | abbreviation    |             4 | als_tokenizer_16k         |
| El Escorial criteria                | clinical        |             4 | als_tokenizer_16k         |
| aggregation                         | clinical        |             4 | als_tokenizer_16k         |
| interaction                         | clinical        |             4 | als_tokenizer_16k         |
| Awaji criteria                      | clinical        |             4 | als_tokenizer_16k         |
| examination                         | clinical        |             4 | als_tokenizer_16k         |
| autophagy                           | clinical        |             4 | als_tokenizer_16k         |
| hypothesis                          | clinical        |             4 | als_tokenizer_16k         |

*Report generated by scripts/validate_tokenizer.py*
