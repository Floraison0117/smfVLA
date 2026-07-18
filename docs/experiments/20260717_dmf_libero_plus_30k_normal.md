# DMF — LIBERO-Plus 1-NFE Eval (30k checkpoint, normal mode)

- **日期**: 2026-07-17
- **方法**: dmf
- **Checkpoint**: checkpoints/dmf_finetuned/step_0030000

## 1. 实验目的

评估 DMF (Decoupled MeanFlow) 在 libero-plus-training 数据集上微调 30000 步后的 1-NFE 推理性能。与 Pi-Flow 30k checkpoint 在相同 setting 下对比，量化两者在成功率与推理延时上的差异，为 1-NFE 方法选型提供依据。

## 2. 实验 Setting

- **数据**: libero-plus, normal 模式（4 suites: libero_spatial, libero_object, libero_goal, libero_10; 每 perturbation category 采样 12 tasks; 5 ep/task）
- **算法改动**: DMF finetuning from pi05_libero base — 冻结 VLM backbone，训练 action-expert `*_1` 层 + `logvar_proj` + `time_mlp_in/out`，新增 logvar-based diffusion matching head（dmf_depth_ratio=0.67, use_logvar=True）
- **Checkpoint**: `checkpoints/dmf_finetuned/step_0030000`（3.35B params, 53 arrays, libero-plus-training, 30k steps, save_every=5000）
- **训练 loss**: 从 -0.5 (step 4600) 降至 -2.0 (step 30000)；fm loss ~10 (MSE, 偏高)
- **超参 / NFE**: NFE=1, seed=7, replan_steps=5, action_horizon=10, num_steps_wait=10
- **并行配置**: 与 Pi-Flow 30k eval 并行运行于同一 GPU（RTX PRO 6000, 97GB），每进程 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`
- **Smoke test**: 并行可行性已通过 quick 模式验证（不 OOM, GPU 各 ~44GB, 延时 ~60-77ms/infer）
- **Eval 修复**: `use_quantiles=False`（policy_loader.py:504/509）— DMF 训练用 z-score 归一化，eval 原默认 quantile 不匹配

## 3. 实验结果

- **量化指标**: **1-NFE 成功率 = 0%**（所有 episode 均 FAILURE, steps=230=max_steps）
  - quick 模式: 10 tasks × 5 ep = 50 ep, 0/50 成功
  - normal 模式: 已跑 ~15/84 tasks, 全部 0%
- **对照实验**: pi05 base checkpoint 在相同 quick 模式下 = 100% (50/50)
- **延时**: 稳态 ~65-70ms/infer（1 NFE），首帧 ~40s（JIT 编译）
- **动作范围**: 1-NFE 输出 [-3.9, 5.3]（z-score），远超 pi05 的 [-0.35, 0.52]；10-NFE 输出 [-1.1, 0.29]，合理
- **可视化/观察**: 日志见 `logs/eval_dmf_fix2.log`；无 NaN/Inf/error

## 4. 分析

**根因：DMF 1-NFE 推理在时间端点外推失败。**

DMF 训练使用两个分支的时间采样：
- **FM 分支**: t_fm ~ LogitNormal(P_mean=0.0, P_std=1.0) → sigmoid(N(0,1))，绝大多数采样在 [0.2, 0.8]，中心约 0.5
- **MF 分支**: t ~ LogitNormal(0.4, 1.0) ≈ 0.60, r ~ LogitNormal(-1.2, 1.0) ≈ 0.23

1-NFE 推理需要 model(noise, t=1.0, r=0.0) — **两个端点都不在训练分布内**。模型在此处外推，velocity 预测严重偏差，导致 `action = noise - u` 的结果过大。

10-NFE 推理的步骤在 [0.0, 0.9] 之间，大部分在训练分布内，因此输出合理。

**次要问题：归一化方案不匹配。**
- 训练用 z-score: `x_0 = (a - mean) / std`（dmf_loss.py:66）
- eval 默认用 quantile: `(x+1)/2 * (q99-q01) + q01`
- 已修复（`use_quantiles=False`），但因 1-NFE 外推问题更严重，修复后仍 0%

**libero-plus-training norm_stats 可疑**: actions[3] mean=-2.988（不合理，LIBERO delta action 应在 ±0.1），但即使改用 base checkpoint norm_stats 也无法解决 1-NFE 问题。

## 5. Next Steps

1. **[P0] 调整 DMF 时间采样**：扩大 P_std 或使用 uniform 采样覆盖 [0, 1] 端点，使 1-NFE 推理在训练分布内
2. **[P1] 验证 10-NFE eval**：用 10-NFE 跑 normal 模式，确认模型本身训练有效
3. **[P1] 检查 libero-plus-training norm_stats 正确性**：dim 3 mean=-2.988 异常，需从原始数据重新计算
4. **[P2] 同步检查 Pi-Flow**：Pi-Flow 有相同的 0% 问题，需确认是否有相同的时间端点外推问题
5. **[P2] 检查 SMF/SnapFlow/FreeFlow**：这些方法可能也有归一化方案不匹配（SMF 用 identity norm），需验证
6. **[P2] 注意 Pi-Flow 根因不同**：Pi-Flow 的时间采样无问题（确定性分段，训练=推理），其 0% 的根因是 teacher-student 中 state 归一化不匹配，详见 `20260717_piflow_libero_plus_30k_normal.md`
