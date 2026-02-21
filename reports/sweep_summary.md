# Tokenizer sweep summary

This report summarizes the evaluation of all trained tokenizer candidates and recommends the best vocabulary size for the ALS-LM project based on medical term handling, text compression, and general-English trade-offs.

## Ranking

Candidates ranked by weighted composite score combining flagged medical terms (40%), medical text fertility (30%), compression ratio (20%), and general-English penalty (10%).

| Rank | Tokenizer          | Vocab  | Flagged | Medical fert. | Compression | General fert. | Score  |
|------|--------------------|--------|---------|---------------|-------------|---------------|--------|
|    1 | als_tokenizer_16k  |  3,379 | 115/195 |        1.2485 |      5.8124 |        2.8113 | 1.0000 |
|    2 | als_tokenizer_32k  |  3,379 | 115/195 |        1.2485 |      5.8124 |        2.8113 | 1.0000 |
|    3 | als_tokenizer_50k  |  3,379 | 115/195 |        1.2485 |      5.8124 |        2.8113 | 1.0000 |

## Recommendation

**Selected tokenizer:** als_tokenizer_32k (vocab size: 3,379)

All three tokenizer candidates converged to the same vocabulary size (3,379) because the sample training corpus (~2 MB) lacks sufficient diversity for the BPE algorithm to produce 16K+ merges with `min_frequency=2`. As a result, all candidates have identical performance metrics and the selection is nominal.

The 32K target is selected as the canonical tokenizer because it represents the middle ground in the sweep range. When re-trained on the full ALS corpus (50-100 MB), the three targets will produce distinct vocabularies and the ranking will be meaningful.

## Trade-off analysis

The custom tokenizer makes several trade-offs compared to GPT-2.

**Advantages over GPT-2:**

- Lower medical text fertility (1.2485 vs 1.5074), producing fewer tokens for ALS content
- Higher compression ratio (5.8124 vs 4.8139 chars/token), more efficient encoding
- Won on 68 of 195 medical term comparisons

**Disadvantages vs GPT-2:**

- Significantly higher general-English fertility (2.8113 vs 1.0189), fragmenting common words more
- Lost on 60 medical term comparisons (mostly generic clinical terms that GPT-2 covers well)
- Much smaller vocabulary (3,379 vs 50,257) limits coverage of rare word forms

The general-English penalty is expected for a domain-specific tokenizer trained on a specialized corpus. For the ALS-LM use case (generating domain-specific text), optimizing for medical term handling is the correct priority.

---
*Generated: 2026-02-21 18:27:57 UTC*
