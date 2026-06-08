# SMF-Curr-V2: Delta 偏移 + 梯度匹配设计方案

**日期**: 2026-06-04
**状态**: 已批准
**问题**: SMF loss 占总损失不到 1%，训练退化为普通 flow matching

---

## 1. 问题分析

smf_base 训练日志显示：

| step | loss_fm | loss_smf | ratio (smf/fm) |
|---|---|---|---|
| 0 | 0.1988 | 0.00105 | 0.53% |
| 3000 | 0.2019 | 0.00040 | 0.20% |
| 7000 | 0.1473 | 0.00025 | 0.17% |

**根因**：
1. delta_mean ≈ 0.18，大部分 SMF 样本的 r 和 t 间距很小
2. 当 delta 很小时，self-consistency 目标退化为 FM 目标，模型不需要学新东西
3. smf_loss_scale=2.0 远远不够（2 × 0.0005 = 0.001，仍比 loss_fm 小两个数量级）

---

## 2. 设计方案：Delta 偏移 + 梯度匹配

### 2.1 Delta 偏移采样

**原策略**: `delta = Uniform(0, t) * delta_upper`，delta_mean ≈ 0.18

**新策略**: `delta = t * (1 - u^(1/k))`，其中 `u ~ Uniform(0, 1)`，`k = 2`

数学性质：
- k=1 时退化为 Uniform(0, t)（当前行为）
- k=2 时期望 ≈ 0.67t（delta_mean 预期从 ~0.18 提升到 ~0.35+）
- delta 始终 ∈ [0, t]，不改变支撑集
- 保留 curriculum 的 delta 上限约束

### 2.2 梯度匹配动态 Scale

**目标**: 让 loss_smf 的梯度量级自动匹配 loss_fm

**实现**:
```
每 log_every 步计算:
  g_fm  = ||∇_θ loss_fm||
  g_smf = ||∇_θ loss_smf||
  target_ratio = g_fm / (g_smf + 1e-8)

EMA 更新:
  smf_scale = 0.99 * smf_scale + 0.01 * target_ratio
  smf_scale = clip(smf_scale, 1.0, 200.0)

loss_total = smf_scale * loss_smf + loss_fm
```

- EMA 系数 α = 0.99（平滑）
- smf_scale 限制在 [1, 200]
- 只在 log_every 步精确计算梯度比，其余步用 EMA 缓存值（额外开销 ~5%）

### 2.3 Flow Ratio 调整

| 参数 | 原值 | 新值 | 原因 |
|---|---|---|---|
| flow_ratio | 0.5 | 0.15 | 让 SMF 样本占主导（85%） |
| delta_floor | 0.001 | 0.05 | 避免 delta 过小导致 SMF 退化为 FM |

---

## 3. 训练配置

```yaml
method: smf_curr_v2
checkpoint: /root/autodl-tmp/smfVLA/checkpoints/base/pi05_libero
action_dim: 32
action_horizon: 10

# SMF 参数
flow_ratio: 0.15
time_conditioning: concat
smf_loss_scale: dynamic  # 梯度匹配，范围 [1, 200]

# Delta 采样
use_curriculum: true
delta_sampling: biased   # delta = t * (1 - u^(1/2))
delta_min: 0.05
delta_final: 1.0
delta_floor: 0.05

# 训练超参
learning_rate: 3.0e-5
weight_decay: 0.01
warmup_ratio: 0.03
gradient_clipping: 1.0
batch_size: 32
training_steps: 15000
precision: bf16

# 保存
checkpoint_dir: /root/autodl-tmp/smfVLA/checkpoints/finetuned/smf_curr_v2
save_every: 3000
log_every: 100
```

---

## 4. 实验计划

1. **Smoke test**: 单 batch 前向 + 反向，确认 loss_smf 量级提升、scale 收敛
2. **训练 15K steps**: 保存 checkpoint at 3K, 6K, 9K, 12K, 15K
3. **评估**: 每个 checkpoint 在 libero_spatial 上 eval NFE=1, 250 episodes
4. **监控指标**:
   - loss_smf vs loss_fm 比值（目标：从 0.3% 提升到 30%+）
   - smf_scale 收敛值
   - delta_mean（目标：从 0.18 提升到 0.35+）
5. **对比基线**: smf_base (80%) 和 smf_curr_12k (62.4%)

**成功标准**: smf_curr_v2 的 1-step score > smf_base 的 80%

---

## 5. 代码修改清单

1. `src/smf_vla/training/smf_loss.py`:
   - `sample_r_t_curriculum()`: 新增 `delta_sampling="biased"` 分支
   - `compute_full_smf_loss()`: 新增梯度匹配 scale 逻辑

2. `src/smf_vla/training/jax_trainer.py`:
   - 传入 `smf_loss_scale="dynamic"` 参数
   - 每 log_every 步计算梯度比并更新 EMA scale

3. `configs/train/smf_curr_v2_libero.yaml`: 新增配置文件

4. `scripts/train_smf.sh`: 新增训练脚本
