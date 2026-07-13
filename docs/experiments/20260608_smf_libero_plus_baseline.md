# SMF pi0.5 Baseline — LIBERO-Plus Quick Eval

- **日期**: 2026-06-08
- **方法**: smf (original pi0.5)
- **Checkpoint**: checkpoints/smf_base/pi05_libero

## 1. 实验目的

Unknown — raw evaluation output. Likely: validate SMF evaluation pipeline produces correct results on the known-good pi0.5 checkpoint, establishing a baseline for method comparison.

## 2. 实验 Setting

- **数据**: libero-plus, preset "quick" (4 suites, 10 tasks each, 5 episodes per task = 50 episodes per suite, 200 total)
- **算法改动**: None — baseline pi0.5 model, no finetuning or architectural changes
- **超参 / NFE**: NFE=1, seed=7, replan_steps=5, action_horizon=10

## 3. 实验结果

- **好的结果**: Spatial (98%), Object (100%), Goal (96%) all near-ceiling at 1 NFE
- **差的结果**: LIBERO-10 at 90% — the most challenging suite with 10 diverse tasks
- **量化指标**:

| Suite | Success | Rate |
|---|---|---|
| libero_spatial | 49/50 | 98% |
| libero_object | 50/50 | 100% |
| libero_goal | 48/50 | 96% |
| libero_10 | 45/50 | 90% |
| **Overall** | **192/200** | **96%** |

- **可视化/观察**: Source combined JSON at `eval/results/smf/libero_plus/20260608_150944_combined_1nfe.json`; per-suite detail at `results/libero_plus/20260608_{145732,150107,150404,150944}_*.json`

## 4. 分析

Not recorded at eval time. The 96% overall at 1 NFE establishes a strong baseline. The libero_10 suite (90%) is notably harder than the single-task-category suites (96-100%), consistent with its greater task diversity.

## 5. Next Steps

Not recorded at eval time.
