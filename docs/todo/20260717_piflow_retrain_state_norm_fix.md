# Pi-Flow 重新训练 — 修复 state 归一化后重训

> 日期: 2026-07-17
> 状态: 待执行
> 优先级: **最高 (P0)** — 当前 30k checkpoint 是废权重，所有 Pi-Flow 实验结论需推翻

---

## 1. 背景

Pi-Flow 采用 teacher-student 蒸馏：冻结的 pi0.5 teacher 提供 velocity 监督。但
训练时的 `data_loader.py` 产出的是 **raw state**，而 teacher 训练时用的是
**quantile 归一化** 后的 state（`openpi/training/config.py:187` 对 pi0.5 为
`use_quantile_norm=True`）。Teacher 收到分布外输入 → velocity 预测错误 →
student 学到垃圾动作 → 1-NFE eval **成功率 = 0%**。

详见 `docs/experiments/20260717_piflow_libero_plus_30k_normal.md` 和
`docs/training-debug.md §9`。

## 2. 已完成的修复

代码已修复（本次 commit）：

1. `piflow/src/piflow_vla/training/data_loader.py`
   - `create_data_loader` 新增 `norm_stats_path` 参数
   - 加载 state 的 `q01`/`q99`（之前只加载 action 的 mean/std）
   - 装配 batch 时对 state 做 `(state - q01)/(q99 - q01 + 1e-6)*2 - 1`，
     与 `openpi/transforms.py:141-145` 一致
2. `piflow/scripts/run_train.py`
   - 用 `Path(config["checkpoint"]).rglob("norm_stats.json")` 定位 base
     checkpoint 的 norm_stats（8-dim state，正确），传入 data_loader

Eval 侧无需改动（`policy_loader.py:428/433` 已用 `use_quantiles=True`）。

## 3. 待执行任务

### 3.1 重训 LIBERO（必须）

```bash
cd /root/autodl-tmp/piflow && bash scripts/train.sh
# 默认 config: configs/train/piflow_libero_plus.yaml
# 预计 ~11h (30k steps, ~1340ms/step)
```

- **验证启动日志**包含：`Using checkpoint norm_stats for state normalization:
  .../pi05_libero/assets/physical-intelligence/libero/norm_stats.json` 和
  `Loaded norm_stats ... (state q01/q99 dim=8); applying quantile
  normalization to state`。若出现 `state NOT normalized` warning 说明
  norm_stats 未找到，需先排查 checkpoint 路径。
- **对比 step 200 的 loss**：修复前 raw state 时 step 200 loss≈5.85；
  修复后若 teacher velocity 正确，loss 量级应相似或更低（两者都是拟合
  teacher，但 teacher 输出空间不同，绝对值仅供参考——关键看 eval）。

### 3.2 重新评估（必须）

重训完成后跑 quick 模式验证：

```bash
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type piflow --nfe 1 --mode quick
```

- **通过标准**: quick 模式（10 tasks × 5 ep = 50 ep）成功率应显著 > 0%
  （对照 pi05 base = 100%）。若仍 0%，说明根因判断有误，需回 §9 排查其他
  可能因素（如 action 归一化空间）。
- 通过后再跑 normal 模式（4 suites, 5 ep）。

### 3.3 废弃旧 checkpoint

- `checkpoints/piflow_finetuned/step_0030000` 是用 raw state 训练的废权重，
  重训成功后可删除（建议先保留到新 checkpoint 验证通过）。
- 更新 wandb / 实验记录，标注旧 run `piflow_libero_plus_1nfe` 为 INVALID
  （root cause: state normalization missing）。

## 4. 影响范围

### 4.1 受影响的已记录实验

- `docs/experiments/20260717_piflow_libero_plus_30k_normal.md` — 该实验的
  0% 成功率结论本身正确（确实是 0%），但根因分析与修复已记录在此 todo 和
  `training-debug.md §9` 中。实验报告无需删除，但需在报告顶部加标注：
  "根因已定位并修复，详见 training-debug.md §9；checkpoint 需重训"。
- `docs/todo/20260717_piflow_ablation_frozen_action_expert.md` — 该消融实验
  的 **Baseline 引用的就是废 checkpoint**
  （`checkpoints/piflow_finetuned/step_0030000`，见其 §3.1 "✅ 已有
  step_0030000"）。在 state 归一化修复并重训出新的有效 baseline checkpoint
  之前，**该消融实验的 Baseline 数据无效**，不可直接做 B1/C1 对比。消融实验
  应在新 baseline 训练完成后再启动。

### 4.2 不受影响

- DMF / SMF / SnapFlow / FreeFlow：不用 teacher-student 蒸馏，且其各自的
  `data_loader.py` 是独立副本（虽然与 piflow 同源，但本修复只改了 piflow 的
  副本）。但 DMF 可能存在同类 state 归一化 mismatch（见
  `training-debug.md §9` 教训 4），需单独验证。
- CALVIN eval：Pi-Flow 当前没有 CALVIN 训练 config，不涉及。

## 5. 验收清单

- [ ] 重训启动日志出现 `state q01/q99 dim=8` 和 `applying quantile normalization`
- [ ] 30k 训练完成，loss 正常下降
- [ ] LIBERO-Plus quick 模式成功率 > 0%（目标：接近 pi05 base 的 100%）
- [ ] LIBERO-Plus normal 模式跑完，记录到 `docs/experiments/`
- [ ] 旧 checkpoint 标记为 INVALID 或删除
- [ ] 更新 `docs/todo/20260717_piflow_ablation_frozen_action_expert.md` 的
      Baseline 状态（替换为新的有效 checkpoint）
