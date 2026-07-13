# FreeFlow — Checkpoint Progression on LIBERO-Plus (1 NFE)

- **日期**: 2026-06-13
- **方法**: freeflow
- **Checkpoint**: freeflow/checkpoints/finetuned/freeflow/step_{5000..30000}

## 1. 实验目的

Unknown — raw evaluation output. Likely: evaluate FreeFlow finetuning progress across checkpoints (step_5000 to step_30000) to identify the best checkpoint and characterize training dynamics.

## 2. 实验 Setting

- **数据**: libero-plus, preset "quick" (4 suites, 10 tasks each, 5 episodes per task, 200 total per checkpoint)
- **算法改动**: FreeFlow finetuning from pi05_libero base — adds flow-matching time-embedding heads while freezing the VLM backbone
- **超参 / NFE**: NFE=1, seed=7, replan_steps=5, action_horizon=10

## 3. 实验结果

- **好的结果**: Spatial improves from 70% to 92% across training, Object from 76% to 88%. step_30000 achieves the best overall (60.0%)
- **差的结果**: Goal is near-zero across ALL checkpoints (0-8%) — FreeFlow fundamentally fails on goal-conditioned tasks. No clear monotonic improvement trend; step_25000 regresses
- **量化指标**:

| Checkpoint | Spatial | Object | Goal | LIBERO-10 | Overall |
|---|---|---|---|---|---|
| step_5000 | 70% (35/50) | 76% (38/50) | 2% (1/50) | 50% (25/50) | **49.5%** (99/200) |
| step_10000 | 84% (42/50) | 82% (41/50) | 0% (0/50) | 54% (27/50) | **55.0%** (110/200) |
| step_15000 | 78% (39/50) | 92% (46/50) | 8% (4/50) | 60% (30/50) | **59.5%** (119/200) |
| step_20000 | 86% (43/50) | 90% (45/50) | 4% (2/50) | 50% (25/50) | **57.5%** (115/200) |
| step_25000 | 86% (43/50) | 78% (39/50) | 2% (1/50) | 50% (25/50) | **54.0%** (108/200) |
| step_30000 | 92% (46/50) | 88% (44/50) | 4% (2/50) | 56% (28/50) | **60.0%** (120/200) |

- **可视化/观察**: Source combined JSONs at `eval/results/freeflow/libero_plus/20260613_{184816,190952,193100,195231,201401,203500}_combined_1nfe.json`; per-suite detail in `results/libero_plus/`. Goal suite is systematically at or near zero — not a statistical fluke.

## 4. 分析

Not recorded at eval time. Key observations: (1) FreeFlow plateaus at ~60%, far below the SMF baseline of 96% — a 36-point gap. (2) The libero_goal suite is catastrophic (0-8%), suggesting a fundamental limitation: FreeFlow's flow-matching approach may not properly handle goal-image conditioned tasks. (3) Training is noisy — step_25000 regresses to 54% from 59.5% at step_15000, and the best result (60% at step_30000) is only marginally better than step_15000 (59.5%). (4) The slow/noisy progression suggests either undertraining, learning rate issues, or architectural limitations of the flow-matching heads.

## 5. Next Steps

Not recorded at eval time.
