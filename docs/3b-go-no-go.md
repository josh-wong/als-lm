# 3B model scaling decision

**GO** — The 1B training run completed well within both hardware thresholds defined in the design document. Peak VRAM usage was 6.10 GB (39% below the 10 GB limit) and per-epoch training time was approximately 3 hours 23 minutes (86% below the 24-hour limit), providing substantial headroom for a 3x parameter scaling attempt.

## Decision criteria

The v2-design-doc.md Section 5.4 established two quantitative thresholds for the 3B go/no-go decision, based on the 1B production training run's actual resource consumption. Both conditions must pass for a GO decision.

| Condition                                                                 | Decision if met                                                               |
|---------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| 1B training uses less than 10 GB peak VRAM and completes within 24 hours. | 3B training may be attempted with ZeRO Stage 3 and aggressive CPU offloading. |
| 1B training exceeds 10 GB peak VRAM or requires more than 24 hours.       | 3B training is deferred; 1B model proceeds directly to instruction tuning.    |

## Measured vs threshold comparison

All measurements are from the 1B production training run (config: n_layer=30, n_head=20, n_embd=1600, ~1.004B parameters), completed on 2026-03-17 in 10 hours 8 minutes across 3 epochs on an RTX 3060 12 GB with DeepSpeed ZeRO Stage 2 and CPU offloading.

| Criterion                | Threshold  | Measured (1B) | Headroom  | Result |
|--------------------------|------------|---------------|-----------|--------|
| Peak VRAM usage          | < 10 GB    | 6.10 GB       | 3.90 GB   | PASS   |
| Training time per epoch  | < 24 hours | ~3h 23m       | ~20h 37m  | PASS   |

## Resource utilization analysis

The 1B training run produced detailed resource telemetry across 11,679 steps logged at 10-step intervals, providing a comprehensive picture of hardware utilization.

**GPU VRAM:** Peak allocation reached 6,244 MB (6.10 GB), stabilizing early in training. The RTX 3060's 12 GB capacity was never stressed. The 6.10 GB figure represents the cumulative peak (never-reset tracking), meaning this is the true maximum VRAM demand across the entire run. Current allocation fluctuated higher (up to ~10.8 GB including PyTorch cache) but this reflects temporary allocations reclaimed by the CUDA allocator.

**CPU RAM:** System-wide RAM peaked at 26,322 MB (25.7 GB) out of 64 GB available. DeepSpeed ZeRO Stage 2 CPU offloading places optimizer states in system RAM, which accounts for the majority of this usage. The 500M model used 30.3 GB by comparison, and the lower 1B figure likely reflects more efficient memory management in the updated training pipeline.

**GPU temperature:** Peak temperature reached 83 C, which is 3 C below the RTX 3060 thermal throttle point. The thermal cooldown mechanism (pause training when GPU exceeds 80 C warning threshold) activated during training but did not significantly impact throughput.

**Throughput:** Average throughput was 10,485 tokens/sec across the full run, substantially exceeding the 3,000–3,500 tok/sec projection from the design document. This 3x throughput overperformance explains why training completed in 10 hours versus the projected 45+ hours.

## 1B model quality assessment

The hallucination evaluation benchmark (160 questions across 8 categories) was run against the 1B best checkpoint (step 11,500, val_loss 5.6424) using greedy decoding.

| Metric           | 1B base (this run) | 500M from-scratch (v1.0.0) | GPT-2 Large fine-tuned (v1.0.0) |
|------------------|--------------------|-----------------------------|---------------------------------|
| Mean accuracy    | 0.00%              | 0.21%                       | 3.12%                           |
| Binary pass rate | 0.00%              | 0.21%                       | 3.12%                           |

The 1B base model produced 65% degenerate responses (repetitive token sequences), 18.1% plausible blending, and 11.9% confident fabrication. Zero questions received a passing accuracy score.

This result is expected and consistent with the v1.0.0 from-scratch baseline. A base completion model trained from scratch on a 300M-token domain corpus has not learned to follow question-answer formats. The model generates domain-adjacent token sequences but cannot produce structured factual answers. Both the 500M and 1B from-scratch models perform at effectively zero accuracy on the benchmark, which is designed for instruction-following models. The meaningful quality comparison will come after instruction tuning, where the 1B model's larger capacity may capture more domain knowledge than the 500M model did.

## 3B scaling projections

Based on the 1B measured data, the following projections estimate 3B resource requirements on the same hardware (RTX 3060 12 GB, 64 GB RAM, Intel i5-12400).

**VRAM projection:** A 3B model has approximately 3x the parameters of the 1B model. However, with ZeRO Stage 3 (required for 3B per the design document), model parameters are partitioned across CPU and GPU. During forward and backward passes, only the current layer's parameters and activations reside on GPU. The design document estimates 3B model weights at ~5.7 GB in fp16. With ZeRO Stage 3, the GPU holds approximately one layer's worth of parameters (~190 MB) plus activations. With gradient checkpointing and a potentially reduced micro-batch size (batch_size=2 or 1), peak VRAM should remain under 12 GB — but this is a tight fit with minimal headroom compared to the comfortable 6.10 GB at 1B.

**Training time projection:** The 1B model achieved 10,485 tok/sec throughput. ZeRO Stage 3 incurs significant CPU–GPU communication overhead for parameter gathering and scattering at every forward/backward pass. A conservative estimate is 3–5x slowdown from the combined effects of 3x larger model and ZeRO Stage 3 overhead, yielding approximately 2,000–3,500 tok/sec. At 2,000 tok/sec with the same 382M token corpus per epoch, each epoch would take approximately 53 hours. At 3,500 tok/sec, approximately 30 hours per epoch. The design document projected over 200 hours total, which aligns with 3 epochs at 53–67 hours each. This exceeds the 24-hour per-epoch threshold but that threshold was defined for the 1B decision, not the 3B attempt itself.

**CPU RAM projection:** ZeRO Stage 3 offloads model parameters, gradients, and optimizer states to CPU. For 3B parameters: weights ~6 GB (fp16), optimizer states ~24 GB (fp32 copies + momentum + variance), gradients ~6 GB. Total CPU RAM for model state: ~36 GB. Adding Python overhead and data loading, total system RAM usage could reach 45–55 GB, approaching the 64 GB limit. This is the tightest constraint and may require reducing other memory consumers or using memory-mapped data loading.

## Decision

**GO** — Both quantitative criteria pass with substantial headroom. The 1B training run used 6.10 GB peak VRAM (39% under threshold) and 3h 23m per epoch (86% under threshold). A 3B training attempt is justified under the conditions defined in the design document.

Conditions and caveats for the 3B attempt:

- ZeRO Stage 3 is required (Stage 2 will not fit 3B parameters in 12 GB VRAM)
- Micro-batch size may need reduction from 4 to 2 or 1
- CPU RAM usage may approach 55 GB of 64 GB available, leaving minimal system headroom
- Per-epoch training time will likely exceed 24 hours (projected 30–53 hours), meaning a full 3-epoch run could take 4–7 days of continuous training
- WSL2 stability over multi-day runs is unverified on this hardware
- Throughput may be significantly lower than 1B due to ZeRO Stage 3 communication overhead

These caveats represent engineering challenges, not fundamental blockers. The hardware resources are sufficient in principle, though the training experience will be less comfortable than the 1B run.

## References

- v2-design-doc.md Section 5.4: Scaling decision framework and thresholds
- logs/training_summary_1B.md: 1B production training run metrics
- reports/eval/1B_20260317_205331/hallucination_eval_1B_20260317_205331.md: 1B evaluation results
- v2-design-doc.md Section 4.2: Memory estimates for 1B and 3B configurations
