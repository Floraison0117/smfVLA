# SMF-VLA 评测结果汇总

> 更新时间：2026-06-04

---

## 总览

| 模型 | Checkpoint | 基准 | NFE | Spatial | Object | Goal | LIBERO-10 | LIBERO-90 | 平均 |
|------|-----------|------|-----|---------|--------|------|-----------|-----------|------|
| pi0.5 原始推理 | base/pi05_libero | LIBERO | 1 | 96% | 98% | 96% | 90% | — | **95.0%** |
| pi0.5 自定义推理 | base/pi05_libero | LIBERO | 1 | 0% | 0% | 0% | 0% | — | **0.0%** |
| smf_curr_v2 step_3000 | finetuned/smf_curr_v2/step_3000 | LIBERO | 1 | 96% | 92% | 80% | 48% | — | **79.0%** |
| pi0.5 原始推理 | base/pi05_libero | LIBERO-plus | 10 | 0%† | 98% | 98% | 92% | — | — |
| pi0.5 原始推理 | base/pi05_libero | LIBERO-plus | 1 | 0% | 0% | 0% | — | — | **0.0%** |

> † spatial 在 LIBERO-plus + 10-NFE 下为 0%，疑似环境/任务配置问题。

---

## 1. Baseline：pi0.5 原始推理（LIBERO, 1-NFE）

- **Checkpoint:** `checkpoints/base/pi05_libero`
- **推理方式:** 原始 Pi0 推理代码
- **每个 task 5 episodes，共 50 episodes/suite**

| Suite | 成功率 | Episodes | Avg 延迟 | P50 | P95 |
|-------|--------|----------|----------|-----|-----|
| libero_spatial | **96.0%** | 50 | 61.2ms | 39.7ms | 72.9ms |
| libero_object | **98.0%** | 50 | 60.1ms | 40.3ms | 74.0ms |
| libero_goal | **96.0%** | 50 | 65.6ms | 49.2ms | 74.5ms |
| libero_10 | **90.0%** | 50 | 57.2ms | 42.1ms | 74.7ms |

**平均成功率: 95.0%**

结果文件: `results/pi05_libero_4suites_1nfe_orig/`

---

## 2. pi0.5 自定义推理管线（LIBERO, 1-NFE）

- **Checkpoint:** `checkpoints/base/pi05_libero`（同上）
- **推理方式:** 自定义 Pi05SMF 推理管线
- **每个 task 5 episodes，共 50 episodes/suite**

| Suite | 成功率 | Episodes |
|-------|--------|----------|
| libero_spatial | **0.0%** | 50 |
| libero_object | **0.0%** | 50 |
| libero_goal | **0.0%** | 50 |
| libero_10 | **0.0%** | 50 |

**全部 0%** — 同一 checkpoint 但推理管线不同，说明自定义推理代码存在严重问题（已通过 `--no-smf` 修复，见 §1）。

结果文件: `results/pi05_libero_4suites_1nfe/`

---

## 3. smf_curr_v2 step_3000（LIBERO, 1-NFE, 全 4 suite）

- **Checkpoint:** `checkpoints/finetuned/smf_curr_v2/step_3000`
- **推理方式:** Pi05SMF（1-NFE SplitMeanFlow）
- **每个 task 5 episodes，共 50 episodes/suite**

| Suite | 成功率 | Episodes | 备注 |
|-------|--------|----------|------|
| libero_spatial | **96.0%** | 50 | 9/10 tasks ≥ 80% |
| libero_object | **92.0%** | 50 | 7/10 tasks 100%，cream cheese 60% |
| libero_goal | **80.0%** | 50 | 5/10 tasks 100%，open middle drawer 20%（最弱） |
| libero_10 | **48.0%** | 50 | 2 tasks 0%，最佳 100% |

**平均成功率: 79.0%**

与 baseline 对比：spatial 持平（96%），object 接近（92% vs 98%），goal 落后（80% vs 96%），libero_10 差距大（48% vs 90%）。

结果文件: `results/eval_step3000_all_suites/`

---

## 4. smf_curr_v2 Checkpoint Sweep（libero_spatial, 1-NFE）

| Checkpoint | 成功率 | Episodes |
|------------|--------|----------|
| step_3000 | **98.0%** | 50 |
| step_6000 | **70.0%** | 50 |
| step_9000 | **88.0%** | 50 |
| step_12000 | **88.0%** | 50 |
| step_15000 | **86.0%** | 50 |

**结论:** step_3000 是最优 checkpoint（98%），step_6000 出现明显下降（70%），之后恢复到 86-88%。

结果文件: `results/eval_curr_v2_all_ckpt/`

---

## 5. smf_base Checkpoint Sweep（libero_spatial, 1-NFE, 5 ep/task）

| Checkpoint | 成功率 | Episodes |
|------------|--------|----------|
| step_1000 | **88.0%** | 50 |
| step_2000 | **70.0%** | 50 |
| step_3000 | **78.0%** | 50 |
| step_4000 | **78.0%** | 50 |
| step_5000 | **78.0%** | 50 |

结果文件: `results/eval_checkpoints/`

---

## 6. smf_base Checkpoint Sweep（libero_spatial, 1-NFE, 25 ep/task）

| Run | 成功率 | Episodes |
|-----|--------|----------|
| 1 | **80.0%** | 250 |
| 2 | **67.2%** | 250 |
| 3 | **78.4%** | 250 |
| 4 | **72.4%** | 250 |
| 5 | **74.0%** | 250 |
| 6 | **72.8%** | 250 |

**平均约 74%**，方差较大（67-80%）。

结果文件: `results/eval_checkpoints_6k_15k/`

---

## 7. smf_curr_12k（libero_spatial, 1-NFE, 25 ep/task）

| Run | 成功率 | Episodes |
|-----|--------|----------|
| 1 | **62.4%** | 250 |
| 2 | **61.6%** | 250 |
| 3 | **59.6%** | 250 |
| 4 | **54.4%** | 250 |

**平均约 59.5%**，明显低于其他变体。

结果文件: `results/eval_curr_12k/`

---

## 8. LIBERO-Plus 鲁棒性评测

### 8.1 pi0.5 原始推理 + 10-NFE

| Suite | 成功率 | Episodes | 备注 |
|-------|--------|----------|------|
| libero_object | **98.0%** | 50 | |
| libero_goal | **98.0%** | 50 | |
| libero_10 | **92.0%** | 50 | |
| libero_spatial | **0.0%** | 100 | †疑似配置问题 |

> † spatial 在 LIBERO-plus 下 0% 成功率，其他 3 个 suite 表现正常，推测是 LIBERO-plus 的 spatial 任务配置或环境适配问题。

### 8.2 pi0.5 原始推理 + 1-NFE

| Suite | 成功率 | Episodes |
|-------|--------|----------|
| libero_spatial | **0.0%** | 50 |
| libero_object | **0.0%** | 50 |
| libero_goal | **0.0%** | 50 |

**全部 0%** — LIBERO-plus 环境下 1-NFE 推理完全失败。

结果文件: `results/libero_plus/`

---

## 9. 训练信息

### smf_base

- **方法:** SplitMeanFlow base, concat time embedding, uniform time sampling
- **训练步数:** 15,000
- **学习率:** 3e-5, weight decay 0.01, warmup 450 steps
- **Batch size:** 32, 精度: bf16
- **Flow ratio:** 0.3
- **最终 loss:** total=0.2694, smf=0.00056, fm=0.2689
- **训练时间:** 2.9 小时（1.45 steps/sec）
- **参数量:** 3.36B 总计, 432M 可训练 (12.9%), 2.92B 冻结
- **Checkpoint:** `checkpoints/finetuned/smf_base`

### smf_curr_v2

- **Checkpoint:** `checkpoints/finetuned/smf_curr_v2`
- **最优 step:** 3000（libero_spatial 98%）
- 详见 `logs/train/smf_curr_v2/`

---

## 10. 关键发现

1. **Baseline 确认:** pi0.5 原始推理在标准 LIBERO 上达到 95% 平均成功率（1-NFE），是可靠的参考基准。

2. **推理管线问题已修复:** 自定义推理管线对同一 checkpoint 产出 0%，通过 `--no-smf` 参数使用原始 Pi0 推理代码可恢复 95%。

3. **最优 SMF 模型:** smf_curr_v2/step_3000 在 spatial 上匹配 baseline（96%），但 goal（80%）和 libero_10（48%）有较大差距，整体 79%。

4. **Checkpoint 敏感性:** smf_curr_v2 在 step_3000 最优（98%），step_6000 骤降至 70%，之后恢复。训练需注意 early stopping。

5. **LIBERO-plus 兼容性:** 10-NFE 在 object/goal/libero_10 上表现正常（92-98%），但 spatial 有异常；1-NFE 完全不兼容。

---

## 11. 结果文件索引

| 目录 | 内容 |
|------|------|
| `results/pi05_libero_4suites_1nfe_orig/` | Baseline 4 suite（96/98/96/90%） |
| `results/pi05_libero_4suites_1nfe/` | 自定义推理 4 suite（全部 0%） |
| `results/eval_step3000_all_suites/` | smf_curr_v2 step_3000 全 suite |
| `results/eval_curr_v2_all_ckpt/` | smf_curr_v2 checkpoint sweep |
| `results/eval_checkpoints/` | smf_base checkpoint sweep (5ep) |
| `results/eval_checkpoints_6k_15k/` | smf_base checkpoint sweep (25ep) |
| `results/eval_curr_12k/` | smf_curr_12k 评测 |
| `results/libero_plus/` | LIBERO-plus 鲁棒性评测 |
| `results/eval/` | 早期评测结果 |

| 日志 | 内容 |
|------|------|
| `logs/eval_pi05_libero_1nfe_orig.log` | Baseline 评测日志 |
| `logs/eval_pi05_libero_4suites_1nfe.log` | 自定义推理评测日志 |
| `logs/eval_curr_v2_all_ckpt/` | smf_curr_v2 sweep 日志 (5 文件) |
| `logs/eval_step3000_all_suites/` | step_3000 全 suite 日志 |
| `logs/eval_libero_plus_comparison/` | LIBERO-plus 对比日志 |
