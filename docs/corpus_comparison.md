# Corpus comparison: v1.0.0 vs v1.2.0

Side-by-side comparison of corpus metrics before and after Phase 35 normalization fixes and Phase 36 source expansion.

## Corpus size

High-level corpus metrics comparing the v1.0.0 baseline to the current v1.2.0 pipeline output.

| Metric               | v1.0.0         | v1.2.0         | Change              | Advisory    |
|----------------------|----------------|----------------|---------------------|-------------|
| Total documents      | 35,088         | 35,063         | -25 (-0.1%)         |             |
| Total words          | 101,303,590    | 101,274,013    | -29,577 (-0.0%)     |             |
| Total size           | 678.05 MB      | 676.07 MB      | -2.0 (-0.3%)        | Target met  |
| Estimated tokens     | ~176,547,868   | ~176,493,328   | -54,540 (-0.0%)     |             |
| Training documents   | 31,581         | 31,560         | -21 (-0.1%)         |             |
| Validation documents | 3,507          | 3,503          | -4 (-0.1%)          |             |
| train.txt size       | 609.69 MB      | 608.05 MB      | -1.6 (-0.3%)        |             |
| val.txt size         | 68.35 MB       | 68.01 MB       | -0.3 (-0.5%)        |             |

## Source distribution

Per-category document and word counts, showing changes between versions.

| Source category       | v1.0.0 docs | v1.2.0 docs | Change       | v1.0.0 words  | v1.2.0 words  | Change       |
|-----------------------|-------------|-------------|--------------|---------------|---------------|--------------|
| biomedical_research   |      33,148 |      33,123 | -25 (-0.1%)  |    99,300,141 |    99,273,443 | -26,698 (-0.0%) |
| clinical_trials       |       1,344 |       1,344 | +0 (+0.0%)   |     1,135,581 |     1,134,013 | -1,568 (-0.1%) |
| educational           |         564 |         564 | +0 (+0.0%)   |       794,661 |       793,350 | -1,311 (-0.2%) |
| supplementary_science |          32 |          32 | +0 (+0.0%)   |        73,207 |        73,207 | +0 (+0.0%)   |

## Deduplication rate

Overall deduplication rate calculated from rejection summary (near_duplicate rejections / total raw documents). Per-category loss rates compare raw JSON file counts against final source distribution counts, reflecting combined cleaning, deduplication, and capping losses.

| Scope                 | v1.0.0 rate | v1.2.0 rate | Change       |
|-----------------------|-------------|-------------|--------------|
| Overall               |       1.8%  |       1.8%  | +0.0 pp      |
| biomedical_research   |       1.0%  |       1.1%  | +0.1 pp      |
| clinical_trials       |       0.5%  |       0.5%  | +0.0 pp      |
| educational           |      61.1%  |      61.1%  | +0.0 pp      |
| supplementary_science |       8.6%  |       8.6%  | +0.0 pp      |

## Document length

Average and median word counts per document.

| Metric  | v1.0.0    | v1.2.0    | Change              |
|---------|-----------|-----------|---------------------|
| Average |   2,887   |   2,888   | +1 (+0.0%)          |
| Median  |     361   |     361   | +0 (+0.0%)          |

## File sizes

Output file sizes for the concatenated training and validation files.

| File      | v1.0.0      | v1.2.0      | Change              |
|-----------|-------------|-------------|---------------------|
| train.txt | 609.69 MB   | 608.05 MB   | -1.6 (-0.3%)        |
| val.txt   | 68.35 MB    | 68.01 MB    | -0.3 (-0.5%)        |

## Per-source regression warnings

No per-source regressions detected.

---

*Generated: 2026-03-14 18:01:18 UTC*
*Script: data/processing/compare_corpus.py*
