# Original pi0.5 — LIBERO-Plus Verification (1 NFE)

- **日期**: 2026-06-13
- **方法**: original_pi05
- **Checkpoint**: checkpoints/smf_base/pi05_libero (same as SMF baseline)

## 1. 实验目的

Unknown — raw evaluation output. Likely: verify the SMF baseline result by re-running under the original pi0.5 evaluation code path, or confirm reproducibility on a different date.

## 2. 实验 Setting

- **数据**: libero-plus, preset "quick" (4 suites, 10 tasks each, 5 episodes per task, 200 total)
- **算法改动**: None — same pi0.5 baseline checkpoint as the SMF eval, but run under "original" model type code path
- **超参 / NFE**: NFE=1, seed=7, replan_steps=5, action_horizon=10

## 3. 实验结果

- **好的结果**: Spatial (100%), Goal (100%), LIBERO-10 (100%) all perfect; only Object missed 1 episode
- **差的结果**: Object at 98% (1 failure out of 50)
- **量化指标**:

| Suite | Success | Rate |
|---|---|---|
| libero_spatial | 50/50 | 100% |
| libero_object | 49/50 | 98% |
| libero_goal | 50/50 | 100% |
| libero_10 | 50/50 | 100% |
| **Overall** | **199/200** | **99.5%** |

- **可视化/观察**: Source combined JSON at `eval/results/original_pi05/libero_plus/20260613_210358_combined_1nfe.json`; per-suite detail at `results/libero_plus/20260613_{205219,205604,205909,210358}_*.json`

## 4. 分析

Not recorded at eval time. This run achieved 99.5% vs the SMF eval's 96% on the same checkpoint. The 3.5-point gap may be statistical noise (1 fewer failure in SMF run) or could reflect subtle differences in the SMF vs original evaluation code paths. Both results confirm pi0.5 as a strong baseline.

## 5. Next Steps

Not recorded at eval time.
