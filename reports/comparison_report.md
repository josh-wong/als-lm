# Tokenizer comparison report

This report compares custom ALS-domain tokenizers against the GPT-2 baseline tokenizer, measuring medical term handling, text compression, and general-English performance.

## Run metadata

- **Date:** 2026-02-21 18:24:24 UTC
- **Baseline:** gpt2 (vocab: 50,257)
- **Medical terms evaluated:** 195
- **Text file:** data/processed/val.txt
- **Text size:** 239,894 characters, 33,059 words

## als_tokenizer_16k (vocab: 3,379)

### Summary statistics

Term-level results against GPT-2 on 195 medical terms.

| Metric                      | Custom            | GPT-2             |
|-----------------------------|-------------------|--------------------|
| Wins (fewer subtokens)      | 68               |                    |
| Losses (more subtokens)     | 60              |                    |
| Ties                        | 67               |                    |
| Medical text fertility      | 1.2485          | 1.5074           |
| Compression ratio (char/tok)| 5.8124          | 4.8139           |
| Total tokens (val text)     | 41,273          | 49,834           |
| General-English fertility   | 2.8113          | 1.0189           |

### Top wins (custom tokenizer is better)

Terms where the custom tokenizer uses fewer subtokens than GPT-2.

| Term                              | Custom | GPT-2 | Improvement |
|-----------------------------------|--------|-------|-------------|
| dysphagia                           |      1 |     5 |          +4 |
| percutaneous endoscopic gastrostomy |      4 |     8 |          +4 |
| spinocerebellar ataxia              |      5 |     9 |          +4 |
| sodium phenylbutyrate               |      3 |     7 |          +4 |
| bvFTD                               |      1 |     4 |          +3 |
| pNfH                                |      1 |     4 |          +3 |
| dysarthria                          |      1 |     4 |          +3 |
| hereditary spastic paraplegia       |      4 |     7 |          +3 |
| frontotemporal dementia             |      2 |     5 |          +3 |
| taurursodiol                        |      2 |     5 |          +3 |

### Top losses (GPT-2 is better)

Terms where GPT-2 uses fewer subtokens than the custom tokenizer. These represent the cost of domain specialization on a smaller vocabulary.

| Term                              | Custom | GPT-2 | Regression |
|-----------------------------------|--------|-------|------------|
| examination                         |      4 |     1 |         -3 |
| disruption                          |      5 |     2 |         -3 |
| communication                       |      3 |     1 |         -2 |
| capacity                            |      3 |     1 |         -2 |
| aggregation                         |      4 |     2 |         -2 |
| interaction                         |      4 |     2 |         -2 |
| population                          |      3 |     1 |         -2 |
| distribution                        |      4 |     2 |         -2 |
| modification                        |      4 |     2 |         -2 |
| overactivation                      |      4 |     2 |         -2 |

### General-English sanity check

Fertility comparison on 53 common English words. A fertility close to 1.0 means most words are single tokens. Higher values indicate more fragmentation.

- **Custom average fertility:** 2.8113
- **GPT-2 average fertility:** 1.0189
- **Warning:** Custom tokenizer fragments common English words 2.8x more than GPT-2

Words with different tokenization.

| Word          | Custom tokens | GPT-2 tokens |
|---------------|---------------|--------------|
| the           |             2 |            1 |
| was           |             2 |            1 |
| have          |             2 |            1 |
| been          |             2 |            1 |
| would         |             3 |            1 |
| could         |             3 |            1 |
| about         |             2 |            1 |
| people        |             3 |            1 |
| because       |             3 |            1 |
| different     |             4 |            1 |
| important     |             3 |            1 |
| between       |             3 |            1 |
| through       |             2 |            1 |
| another       |             2 |            1 |
| something     |             5 |            1 |

## als_tokenizer_32k (vocab: 3,379)

### Summary statistics

Term-level results against GPT-2 on 195 medical terms.

| Metric                      | Custom            | GPT-2             |
|-----------------------------|-------------------|--------------------|
| Wins (fewer subtokens)      | 68               |                    |
| Losses (more subtokens)     | 60              |                    |
| Ties                        | 67               |                    |
| Medical text fertility      | 1.2485          | 1.5074           |
| Compression ratio (char/tok)| 5.8124          | 4.8139           |
| Total tokens (val text)     | 41,273          | 49,834           |
| General-English fertility   | 2.8113          | 1.0189           |

### Top wins (custom tokenizer is better)

Terms where the custom tokenizer uses fewer subtokens than GPT-2.

| Term                              | Custom | GPT-2 | Improvement |
|-----------------------------------|--------|-------|-------------|
| dysphagia                           |      1 |     5 |          +4 |
| percutaneous endoscopic gastrostomy |      4 |     8 |          +4 |
| spinocerebellar ataxia              |      5 |     9 |          +4 |
| sodium phenylbutyrate               |      3 |     7 |          +4 |
| bvFTD                               |      1 |     4 |          +3 |
| pNfH                                |      1 |     4 |          +3 |
| dysarthria                          |      1 |     4 |          +3 |
| hereditary spastic paraplegia       |      4 |     7 |          +3 |
| frontotemporal dementia             |      2 |     5 |          +3 |
| taurursodiol                        |      2 |     5 |          +3 |

### Top losses (GPT-2 is better)

Terms where GPT-2 uses fewer subtokens than the custom tokenizer. These represent the cost of domain specialization on a smaller vocabulary.

| Term                              | Custom | GPT-2 | Regression |
|-----------------------------------|--------|-------|------------|
| examination                         |      4 |     1 |         -3 |
| disruption                          |      5 |     2 |         -3 |
| communication                       |      3 |     1 |         -2 |
| capacity                            |      3 |     1 |         -2 |
| aggregation                         |      4 |     2 |         -2 |
| interaction                         |      4 |     2 |         -2 |
| population                          |      3 |     1 |         -2 |
| distribution                        |      4 |     2 |         -2 |
| modification                        |      4 |     2 |         -2 |
| overactivation                      |      4 |     2 |         -2 |

### General-English sanity check

Fertility comparison on 53 common English words. A fertility close to 1.0 means most words are single tokens. Higher values indicate more fragmentation.

- **Custom average fertility:** 2.8113
- **GPT-2 average fertility:** 1.0189
- **Warning:** Custom tokenizer fragments common English words 2.8x more than GPT-2

Words with different tokenization.

| Word          | Custom tokens | GPT-2 tokens |
|---------------|---------------|--------------|
| the           |             2 |            1 |
| was           |             2 |            1 |
| have          |             2 |            1 |
| been          |             2 |            1 |
| would         |             3 |            1 |
| could         |             3 |            1 |
| about         |             2 |            1 |
| people        |             3 |            1 |
| because       |             3 |            1 |
| different     |             4 |            1 |
| important     |             3 |            1 |
| between       |             3 |            1 |
| through       |             2 |            1 |
| another       |             2 |            1 |
| something     |             5 |            1 |

## als_tokenizer_50k (vocab: 3,379)

### Summary statistics

Term-level results against GPT-2 on 195 medical terms.

| Metric                      | Custom            | GPT-2             |
|-----------------------------|-------------------|--------------------|
| Wins (fewer subtokens)      | 68               |                    |
| Losses (more subtokens)     | 60              |                    |
| Ties                        | 67               |                    |
| Medical text fertility      | 1.2485          | 1.5074           |
| Compression ratio (char/tok)| 5.8124          | 4.8139           |
| Total tokens (val text)     | 41,273          | 49,834           |
| General-English fertility   | 2.8113          | 1.0189           |

### Top wins (custom tokenizer is better)

Terms where the custom tokenizer uses fewer subtokens than GPT-2.

| Term                              | Custom | GPT-2 | Improvement |
|-----------------------------------|--------|-------|-------------|
| dysphagia                           |      1 |     5 |          +4 |
| percutaneous endoscopic gastrostomy |      4 |     8 |          +4 |
| spinocerebellar ataxia              |      5 |     9 |          +4 |
| sodium phenylbutyrate               |      3 |     7 |          +4 |
| bvFTD                               |      1 |     4 |          +3 |
| pNfH                                |      1 |     4 |          +3 |
| dysarthria                          |      1 |     4 |          +3 |
| hereditary spastic paraplegia       |      4 |     7 |          +3 |
| frontotemporal dementia             |      2 |     5 |          +3 |
| taurursodiol                        |      2 |     5 |          +3 |

### Top losses (GPT-2 is better)

Terms where GPT-2 uses fewer subtokens than the custom tokenizer. These represent the cost of domain specialization on a smaller vocabulary.

| Term                              | Custom | GPT-2 | Regression |
|-----------------------------------|--------|-------|------------|
| examination                         |      4 |     1 |         -3 |
| disruption                          |      5 |     2 |         -3 |
| communication                       |      3 |     1 |         -2 |
| capacity                            |      3 |     1 |         -2 |
| aggregation                         |      4 |     2 |         -2 |
| interaction                         |      4 |     2 |         -2 |
| population                          |      3 |     1 |         -2 |
| distribution                        |      4 |     2 |         -2 |
| modification                        |      4 |     2 |         -2 |
| overactivation                      |      4 |     2 |         -2 |

### General-English sanity check

Fertility comparison on 53 common English words. A fertility close to 1.0 means most words are single tokens. Higher values indicate more fragmentation.

- **Custom average fertility:** 2.8113
- **GPT-2 average fertility:** 1.0189
- **Warning:** Custom tokenizer fragments common English words 2.8x more than GPT-2

Words with different tokenization.

| Word          | Custom tokens | GPT-2 tokens |
|---------------|---------------|--------------|
| the           |             2 |            1 |
| was           |             2 |            1 |
| have          |             2 |            1 |
| been          |             2 |            1 |
| would         |             3 |            1 |
| could         |             3 |            1 |
| about         |             2 |            1 |
| people        |             3 |            1 |
| because       |             3 |            1 |
| different     |             4 |            1 |
| important     |             3 |            1 |
| between       |             3 |            1 |
| through       |             2 |            1 |
| another       |             2 |            1 |
| something     |             5 |            1 |

## Cross-tokenizer summary

Comparison of all custom tokenizers against GPT-2 baseline.

| Vocab size | Wins | Losses | Ties | Medical fertility | General fertility | Compression |
|------------|------|--------|------|-------------------|-------------------|-------------|
|      3,379 |   68 |     60 |   67 |            1.2485 |            2.8113 |      5.8124 |
|      3,379 |   68 |     60 |   67 |            1.2485 |            2.8113 |      5.8124 |
|      3,379 |   68 |     60 |   67 |            1.2485 |            2.8113 |      5.8124 |

**GPT-2 baseline:** fertility=1.5074, general fertility=1.0189, compression=4.8139

## Honest assessment

Across all 3 custom tokenizers compared against GPT-2, the custom tokenizers won on 204 term comparisons, lost on 180, and tied on 201. 
On medical text, the custom tokenizers achieve lower fertility (1.2485 vs 1.5074), producing fewer tokens per word despite smaller vocabularies.
 The general-English penalty is significant (2.8113 vs 1.0189), reflecting the trade-off of training on a domain-specific corpus with a smaller vocabulary.

**Note on vocab size convergence:** All custom tokenizers converged to the same vocabulary size (3,379). This occurs when the training corpus is too small to produce enough BPE merges for the larger target sizes. With a larger corpus (50-100 MB of real ALS literature), the 16K, 32K, and 50K targets would produce distinct vocabularies with meaningfully different trade-offs.

---
*Generated: 2026-02-21 18:24:24 UTC*
