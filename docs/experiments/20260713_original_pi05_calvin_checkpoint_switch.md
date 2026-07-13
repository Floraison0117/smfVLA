# Original pi0.5 — CALVIN ABCD Checkpoint Switch (pi05_calvin_pt)

- **日期**: 2026-07-13
- **方法**: original_pi05
- **Checkpoint**: checkpoints/pi05_calvin_pt (HuggingFace source)

## 1. 实验目的

Switch CALVIN evaluation from the `pi05_calvin_corrected` checkpoint (used in the June 14 NFE sweep) to the standard `pi05_calvin_pt` checkpoint sourced from HuggingFace. Verify that performance is preserved after the switch.

## 2. 实验 Setting

- **数据**: CALVIN ABCD→D benchmark, 100 sequences, seed=0, ep_len_per_subtask=360
- **算法改动**: None — same baseline pi0.5 model. Only the checkpoint source changed: `pi05_calvin_corrected` → `pi05_calvin_pt`.
- **超参 / NFE**: NFE=1, replan_steps=5
- **验证流程**:
  1. Debug mode (calvin_debug, 5 sequences) — verify pipeline runs without error
  2. Quick smoke test (ABCD, 5 sequences) — verify checkpoint produces reasonable results
  3. Full run (ABCD, 100 sequences) — official benchmark result

## 3. 实验结果

- **好的结果**: Full run achieved 90% at 1 NFE, matching the June 14 result (91%) within 1 point. Checkpoint switch is safe.
- **差的结果**: Debug mode returned 0% as expected (pipeline validation only). Quick test at 60% (5 sequences) had high variance due to tiny sample size — not a concern.
- **量化指标**:

| Stage | Dataset | Seqs | SR | SR1 | SR5 | Duration |
|---|---|---|---|---|---|---|
| Debug | calvin_debug | 5 | 0.0% | — | — | 99s |
| Quick | ABCD | 5 | 60.0% | 60% | 60% | 228s |
| Full | ABCD | 100 | **90.0%** | 90% | 44% | 731s |

**Comparison with previous checkpoint (June 14):**

| Checkpoint | NFE | SR |
|---|---|---|
| pi05_calvin_corrected | 1 | 91% |
| **pi05_calvin_pt** | 1 | **90%** |

- **可视化/观察**: Source JSONs at `eval/results/calvin/20260712_235407_calvin_debug_1nfe_0.0pct.json`, `20260712_235215_calvin_ABCD_1nfe_60.0pct.json`, `20260713_000635_calvin_ABCD_1nfe_90.0pct.json`.

## 4. 分析

The 1-point difference (90% vs 91%) between `pi05_calvin_pt` and `pi05_calvin_corrected` is within normal run-to-run variance on a 100-sequence benchmark. The checkpoint switch is confirmed safe — `pi05_calvin_pt` can serve as the standard CALVIN evaluation checkpoint going forward. The chain length degradation (SR1=90% → SR5=44%) is consistent with the June 14 result (SR5=46%), confirming the checkpoint has equivalent long-horizon behavior.

## 5. Next Steps

1. Standardize all future CALVIN evaluations on `checkpoints/pi05_calvin_pt` — no need to maintain `pi05_calvin_corrected`.
2. If CALVIN eval is extended to DMF or other methods, use this result as the pi0.5 baseline.
