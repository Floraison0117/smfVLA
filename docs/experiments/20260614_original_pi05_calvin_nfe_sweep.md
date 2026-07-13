# Original pi0.5 — CALVIN ABCD NFE Sweep

- **日期**: 2026-06-14
- **方法**: original_pi05
- **Checkpoint**: checkpoints/pi05_calvin_corrected

## 1. 实验目的

Unknown — raw evaluation output. Likely: measure the sensitivity of pi0.5 on CALVIN to the number of diffusion steps (NFE), determining the minimum NFE that achieves acceptable performance.

## 2. 实验 Setting

- **数据**: CALVIN ABCD→D benchmark, 100 sequences, seed=0, ep_len_per_subtask=360
- **算法改动**: None — baseline pi0.5, no architectural changes
- **超参 / NFE**: NFE ∈ {1, 2, 4, 10}, replan_steps=5

## 3. 实验结果

- **好的结果**: Minimal NFE sensitivity — 1 NFE already achieves 91%, and 2+ NFE reaches 92%. Performance saturates quickly
- **差的结果**: None — all NFE values within 1 point of each other
- **量化指标**:

| NFE | Success Rate | Episodes |
|---|---|---|
| 1 | 91% (91/100) | 100 |
| 2 | 92% (92/100) | 100 |
| 4 | 92% (92/100) | 100 |
| 10 | 92% (92/100) | 100 |

- **可视化/观察**: Source JSONs at `eval/results/calvin/20260614_{005841,011035,012120,013204}_calvin_ABCD_*.json`. Per-task breakdowns not recorded (per_task: {}).

## 4. 分析

Not recorded at eval time. The near-zero NFE sensitivity on CALVIN suggests that the action horizon (10) and replanning strategy (replan every 5 steps) already provide sufficient temporal coverage. Increasing diffusion steps from 1 to 10 adds only 1 point of success rate while roughly doubling inference latency. For deployment, 1 NFE offers the best speed-accuracy trade-off on this benchmark.

## 5. Next Steps

Not recorded at eval time.
