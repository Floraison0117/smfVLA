# Original pi0.5 — LIBERO-Plus Perturbation Robustness (10 NFE)

- **日期**: 2026-07-13
- **方法**: original_pi05
- **Checkpoint**: checkpoints/pi05_libero

## 1. 实验目的

Unknown — raw evaluation output. Likely: measure pi0.5 robustness under perturbation sampling (background textures, camera viewpoints, lighting, object layout, robot initial states, sensor noise, language instructions) at 10 NFE.

## 2. 实验 Setting

- **数据**: libero-plus with perturbation sampling, all suites, all tasks, 5 episodes per (task, perturbation) pair. Total: 2928 episodes across 7 perturbation types. Preset "sample" (full perturbation enumeration, not limited by max_tasks)
- **算法改动**: None — baseline pi0.5, no architectural changes
- **超参 / NFE**: NFE=10, seed=7, replan_steps=5, action_horizon=10

## 3. 实验结果

- **好的结果**: Background Textures (96.67%) and Light Conditions (96.88%) are near-ceiling — pi0.5 is robust to visual texture/lighting changes
- **差的结果**: Sensor Noise (62.71%) is the hardest perturbation — nearly 40% failure rate. Camera Viewpoints (73.33%) and Robot Initial States (75.21%) also show significant degradation
- **量化指标**:

**Per-suite:**

| Suite | Success | Rate |
|---|---|---|
| libero_spatial | 635/732 | 86.75% |
| libero_object | 629/732 | 85.93% |
| libero_goal | 567/732 | 77.46% |
| libero_10 | 592/732 | 80.87% |
| **Overall** | **2423/2928** | **82.75%** |

**Per-perturbation (all suites combined):**

| Perturbation | Success | Rate |
|---|---|---|
| Background Textures | 464/480 | 96.67% |
| Light Conditions | 465/480 | 96.88% |
| Language Instructions | 440/480 | 91.67% |
| Objects Layout | 40/48 | 83.33% |
| Robot Initial States | 361/480 | 75.21% |
| Camera Viewpoints | 352/480 | 73.33% |
| Sensor Noise | 301/480 | 62.71% |

- **可视化/观察**: Source combined JSON at `eval/results/original_pi05/libero_plus_verification/20260713_065252_combined_10nfe.json`; per-suite detail at `results/libero_plus/20260712_230814,20260713_{012220,031234,065252}_*.json`

## 4. 分析

Not recorded at eval time. Key observations: (1) Sensor Noise is clearly the most challenging perturbation — noise in proprioceptive/visual inputs disrupts policy actions dramatically. (2) Camera Viewpoints and Robot Initial States cause moderate degradation, suggesting spatial generalization limits. (3) Visual texture/lighting perturbations barely affect performance, consistent with the frozen VLM backbone's strong visual representations. (4) libero_goal at 77.46% is the weakest suite overall — goal-conditioned tasks appear more sensitive to perturbations.

## 5. Next Steps

Not recorded at eval time.
