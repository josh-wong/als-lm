# Training Analysis Report

**Log:** sft_log.jsonl
**Generated:** 2026-03-19 18:47 UTC
**Model:** 30L, 20H, 1600D, ctx=1024, dropout=0.1

## Training overview

The training ran for 0 steps across 1 epoch(s), taking approximately N/A of wall-clock time. The final training loss was 9.4855 and the final validation loss was 9.2643.

## Loss curves

![Loss Curves](train_val_loss.png)

Training loss decreased from 9.2578 to 9.2578 over 0 steps, a 0.0% reduction.

## Per-epoch breakdown

Metrics at each validation checkpoint, grouped by epoch.

|  Epoch | Train Loss | Val Loss | Relative Gap | Train PPL | Val PPL | Classification |
|--------|------------|----------|--------------|-----------|---------|----------------|
|      2 |     9.4855 |   9.2643 |      -0.0233 |  13168.03 | 10554.67 | Underfitting   |

## Perplexity analysis

![Perplexity Gap](perplexity_gap.png)

Train perplexity decreased from 13168.03 to 13168.03. Validation perplexity moved from 10554.67 to 10554.67. The gap between train and validation perplexity remained relatively stable throughout training.

## Learning rate schedule

![LR Schedule](lr_schedule.png)

The training used a WarmupCosineLR schedule with 1 warmup steps over 6 total steps. The minimum LR ratio was set to 0.0, meaning the learning rate decayed to 0.0 of its peak value by the end of training.

## Overfitting diagnosis

**Classification: Underfitting**

The model appears to be underfitting. The validation loss (9.2643) is notably below the training loss (9.4855), with a relative gap of -0.0233 (-2.33%). This is unusual and typically indicates the model has not yet converged, or the validation set is easier than the training set.

## Recommendations

- The model has not fully converged. Consider training for additional epochs to allow the loss to stabilize.
- If the learning rate schedule reaches its minimum before convergence, consider increasing the total training steps or raising the minimum LR ratio.
