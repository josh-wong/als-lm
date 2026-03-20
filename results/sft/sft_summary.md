# SFT training summary

**Run:** unknown
**Date:** 2026-03-20

## Training configuration

Training parameters used for supervised fine-tuning of the 1B base model.

| Parameter            | Value                                      |
|----------------------|--------------------------------------------|
| Base model           | checkpoints/1B_*/best/best.pt                          |
| Training examples    | 776         |
| Validation examples  | 194           |
| Learning rate        | 2e-5                        |
| Warmup               | 50 steps                           |
| Max epochs           | 3                |
| Batch size           | 2                |
| Gradient accumulation| 4 (effective batch 8)                |
| Block size           | 1024                |
| Dropout              | 0.1                   |

## Early stopping

Completed all 2 epochs (no early stopping triggered).

**Best checkpoint:** epoch 1, val_loss = 4.8782

**Validation loss per epoch:**

- Epoch 1: 4.8782 (best)

## Base model comparison

Comparison of validation loss between the 1B base model and the SFT model. These are computed on different data distributions (raw corpus vs. instruction pairs) and are not directly comparable.

| Metric    | 1B base  | SFT      | Note                              |
|-----------|----------|----------|-----------------------------------|
| Val loss  | 5.6424 | 4.8782  | Different data, not comparable     |

## Qualitative verification

Spot-check result: **0/8** coherent responses (verdict: **FAIL**).

## Training performance

Runtime metrics from the SFT training run.

- **Total steps:** 290
- **Epochs completed:** 2
- **Average tokens/sec:** 9,038.1
- **Peak GPU memory:** 10,785 MB
- **Final train loss:** 4.7404
