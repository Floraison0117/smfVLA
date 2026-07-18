# Pi-Flow — LIBERO-Plus 1-NFE Eval (30k checkpoint, normal mode)

> ⚠️ **根因已定位并修复（2026-07-17）**：0% 成功率因训练时 state 未做 quantile
> 归一化，teacher 收到 raw state → velocity 错误 → student 学到垃圾动作。代码
> 已修复（见 `docs/training-debug.md §9`），需重训。本报告的 0% 结论本身正确，
> 但对应 checkpoint 已失效。重训任务见
> `docs/todo/20260717_piflow_retrain_state_norm_fix.md`。

- **日期**: 2026-07-17
- **方法**: piflow
- **Checkpoint**: checkpoints/piflow_finetuned/step_0030000（⚠️ 已失效，需重训）

## 1. 实验目的

评估 Pi-Flow (Probabilistic Inverse Flow) 在 libero-plus-training 数据集上微调 30000 步后的 1-NFE 推理性能。与 DMF 30k checkpoint 在相同 setting 下对比，量化两者在成功率与推理延时上的差异，为 1-NFE 方法选型提供依据。

## 2. 实验 Setting

- **数据**: libero-plus, normal 模式（4 suites: libero_spatial, libero_object, libero_goal, libero_10; 每 perturbation category 采样 12 tasks; 5 ep/task）
- **算法改动**: Pi-Flow finetuning from pi05_libero base — 冻结 VLM backbone，训练 action-expert `*_1` 层 + `gmm_mean_proj`/`gmm_logstd_proj`/`gmm_logweight_proj`，新增 GMM-based probabilistic flow head（num_components=8, inner_substeps=8）
- **Checkpoint**: `checkpoints/piflow_finetuned/step_0030000`（3.36B params, 57 arrays, libero-plus-training, 30k steps, save_every=5000）
- **超参 / NFE**: NFE=1, seed=7, replan_steps=5, action_horizon=10, num_steps_wait=10
- **并行配置**: 与 DMF 30k eval 并行运行于同一 GPU（RTX PRO 6000, 97GB），每进程 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`
- **Smoke test**: 并行可行性已通过 quick 模式验证（不 OOM, GPU 各 ~44GB, 延时 ~60-77ms/infer）
- **Eval 修复**: `use_quantiles=False`（policy_loader.py:428/433）— Pi-Flow 训练用 z-score 归一化，eval 原默认 quantile 不匹配

## 3. 实验结果

- **量化指标**: **1-NFE 成功率 = 0%**（所有 episode 均 FAILURE, steps=230=max_steps）
  - quick 模式: 10 tasks × 5 ep = 50 ep, 0/50 成功
  - normal 模式: 已跑 ~15/84 tasks, 全部 0%
- **对照实验**: pi05 base checkpoint 在相同 quick 模式下 = 100% (50/50)
- **延时**: 稳态 ~65-70ms/infer（1 NFE），首帧 ~40s（JIT 编译）
- **可视化/观察**: 日志见 `logs/eval_piflow_fix2.log`；无 NaN/Inf/error

## 4. 分析

**Pi-Flow 的 0% 成功率与 DMF 根因不同。**

### 排除：时间采样（不是问题）

Pi-Flow 的时间采样与 DMF 完全不同。Pi-Flow 使用**确定性均匀分段**（piflow_loss.py:74-76）：
- 训练 nfe=1 时：单段 t_src=1.0 → t_dst=0.0，interval=1.0
- 1-NFE 推理：GMM@t=1.0，rollout [1.0→0.0]
- **训练与推理的时间表完全一致**，不存在 DMF 的大间隔外推问题

### 最可能根因：teacher-student 架构中的 state 归一化不匹配

Pi-Flow 采用 teacher-student 蒸馏（jax_trainer.py:288-340）：
- **Teacher** = 冻结的 pi0.5，训练时接收 **quantile 归一化** 的 state（openpi pipeline: `Normalize(use_quantiles=True)`）
- **Pi-Flow 训练** 时 data_loader 产出 **raw state**（data_loader.py:369，无 Normalize 变换）
- Teacher 收到 raw state 而非 quantile 归一化 state → **velocity 预测错误**
- Student GMM 学习匹配 teacher 的（错误）velocity → 推理时产出垃圾动作

### 其他可能因素

- **Action 归一化**：Pi-Flow 的 loss 中 `actions` 参数仅用于 shape（piflow_loss.py:41 注释），实际监督完全来自 teacher。但 teacher 的输出空间是 quantile space（[-1,1]），student 的 GMM rollout 产出也应在 quantile space。因此 eval 应使用 `use_quantiles=True`（默认），而非 DMF 的 `use_quantiles=False`。两个设置都测试了 0%，说明问题不在 action 归一化。
- **norm_stats 来源**：base checkpoint 的 norm_stats vs 训练数据集的 norm_stats 都试过，均 0%

## 5. Next Steps

1. **[P0] 修复 Pi-Flow 训练时的 state 归一化**：在 data_loader 或 trainer 中对 state 应用 quantile 归一化（与 teacher 的训练 pipeline 一致），需重训练
2. **[P1] 验证修复**：修复 state 归一化后重训练，再跑 1-NFE quick 模式验证
3. **[P1] 确认 Pi-Flow 应使用 `use_quantiles=True`**：因 teacher 输出在 quantile space，需恢复 policy_loader.py 中 Pi-Flow 分支的默认设置（当前被改为 False，应为 True）
4. **[P2] 检查 DMF 是否也有 state 归一化问题**：DMF 不用 teacher，但训练用 raw state、eval 用 normalized state，可能也有 mismatch（但 DMF 的主要问题是时间采样）
