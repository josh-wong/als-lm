# SFT training summary

**Run:** 1B-sft_20260320_033822
**Date:** 2026-03-19

## Training configuration

Training parameters used for supervised fine-tuning of the 1B base model.

| Parameter            | Value                                      |
|----------------------|--------------------------------------------|
| Base model           | checkpoints/1B_*/best/best.pt                          |
| Training examples    | 20         |
| Validation examples  | 6           |
| Learning rate        | 2e-5                        |
| Warmup               | 1 steps                           |
| Max epochs           | 3                |
| Batch size           | 2                |
| Gradient accumulation| 4 (effective batch 8)                |
| Block size           | 1024                |
| Dropout              | 0.1                   |

## Early stopping

Completed all 2 epochs (no early stopping triggered).

**Best checkpoint:** epoch 1, val_loss = 9.2643

**Validation loss per epoch:**

- Epoch 1: 9.2643 (best)

## Base model comparison

Comparison of validation loss between the 1B base model and the SFT model. These are computed on different data distributions (raw corpus vs. instruction pairs) and are not directly comparable.

| Metric    | 1B base  | SFT      | Note                              |
|-----------|----------|----------|-----------------------------------|
| Val loss  | 5.6424 | 9.2643  | Different data, not comparable     |

## Qualitative verification

Spot-check result: **1/8** coherent responses (verdict: **FAIL**).

## Training performance

Runtime metrics from the SFT training run.

- **Total steps:** 5
- **Epochs completed:** 2
- **Average tokens/sec:** 1,330.9
- **Peak GPU memory:** 8,820 MB
- **Final train loss:** 9.4855
