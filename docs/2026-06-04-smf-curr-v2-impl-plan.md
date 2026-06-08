# SMF-Curr-V2 实现计划

基于设计方案 `2026-06-04-smf-curr-v2-design.md`。

---

## Step 1: 修改 `sample_r_t_curriculum` — Delta 偏移采样

**文件**: `src/smf_vla/training/smf_loss.py`
**函数**: `sample_r_t_curriculum()`

**改动**:
- 新增参数 `delta_sampling: str = "uniform"`，支持 `"uniform"`（原行为）和 `"biased"`（新策略）
- 当 `delta_sampling == "biased"` 时，delta 采样改为 `delta = delta_upper * (1 - u^(1/k))`，其中 `u ~ Uniform(0, 1)`，`k = 2`
- `delta_upper = min(t, delta_max)` 保持不变（curriculum 约束）
- 返回的 info_dict 新增 `"delta_sampling"` 字段

**代码片段**:
```python
if delta_sampling == "biased":
    u = jax.random.uniform(rng_delta, (batch_size,), minval=0.0, maxval=1.0)
    delta = delta_upper * (1.0 - jnp.power(u, 1.0 / 2.0))  # k=2
else:
    # 原逻辑: Uniform(0, delta_upper)
    delta = jax.random.uniform(rng_delta, (batch_size,), minval=delta_floor, maxval=1.0) * delta_upper
    delta = jnp.maximum(delta, delta_floor)
```

---

## Step 2: 修改 `compute_full_smf_loss` — 支持动态 scale

**文件**: `src/smf_vla/training/smf_loss.py`
**函数**: `compute_full_smf_loss()`

**改动**:
- 新增参数 `delta_sampling: str = "uniform"`
- 将 `delta_sampling` 传递给 `sample_r_t_curriculum`
- 当 `smf_loss_scale == "dynamic"` 时，不在此函数内做 scale（返回原始 loss_smf，由 trainer 处理梯度匹配）
- 当 `smf_loss_scale` 为 float 时，保持原有行为
- metrics 中新增 `"smf_scale_applied"` 字段，记录实际使用的 scale 值

**注意**: 梯度匹配逻辑放在 trainer 中（Step 3），因为需要 `jax.grad` 分别计算两个 loss 的梯度。

---

## Step 3: 修改 `jax_trainer.py` — 梯度匹配逻辑

**文件**: `src/smf_vla/training/jax_trainer.py`

**改动 3a**: `_setup_jit_train_step()` 方法

当 `smf_loss_scale == "dynamic"` 时：
- 编译两个独立的 grad_fn：
  - `grad_fn_fm`: 只计算 loss_fm 的梯度
  - `grad_fn_smf`: 只计算 loss_smf 的梯度
- 或者更高效的方式：在 `loss_fn` 中分别返回 loss_smf 和 loss_fm，用一次 `jax.value_and_grad` 拿到总梯度，然后用 `jax.grad(loss_smf)` 和 `jax.grad(loss_fm)` 分别计算（需要两次反向传播）
- **推荐方案**: 修改 `loss_fn` 返回 `(loss_total, (loss_smf, loss_fm, metrics))`，然后用 `jax.value_and_grad` 对 `loss_total` 求梯度用于参数更新，同时用 `jax.grad(loss_smf)` 和 `jax.grad(loss_fm)` 计算梯度范数比

**改动 3b**: `train()` 方法主循环

- 初始化 `smf_scale_ema = 1.0`
- 每 `log_every` 步：
  - 调用 grad_fn 获取 `g_smf` 和 `g_fm` 的范数
  - 计算 `target_ratio = g_fm / (g_smf + 1e-8)`
  - 更新 `smf_scale_ema = 0.99 * smf_scale_ema + 0.01 * target_ratio`
  - clip 到 [1, 200]
- 将 `smf_scale_ema` 传入 train_step 用于 loss 计算
- metrics 中记录 `smf_scale_ema`

**改动 3c**: config 解析

- `smf_loss_scale` 从 config 读取时支持字符串 `"dynamic"` 或 float
- 新增 `delta_sampling` 从 config 读取，默认 `"uniform"`

---

## Step 4: 新增配置文件

**文件**: `configs/train/smf_curr_v2_libero.yaml`

```yaml
method: smf_curr_v2
description: "Delta biased sampling + gradient matching scale, 15k steps"

checkpoint: /root/autodl-tmp/smfVLA/checkpoints/base/pi05_libero
pi05: true
action_dim: 32
action_horizon: 10

# SMF 参数
flow_ratio: 0.15
time_conditioning: concat
smf_loss_scale: dynamic

# Delta 采样
use_curriculum: true
delta_sampling: biased
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

optimizer: AdamW
ema_decay: null

dataset: libero
dataset_path: /root/autodl-tmp/smfVLA/data/libero

checkpoint_dir: /root/autodl-tmp/smfVLA/checkpoints/finetuned/smf_curr_v2
log_dir: /root/autodl-tmp/smfVLA/logs/train/smf_curr_v2
save_every: 3000
log_every: 100

wandb:
  project: smfvla
  run_name: null

use_anchor: false
use_bpl: false

freeze:
  - "PaliGemma/img/**"
  - "PaliGemma/llm/embedder/**"
  - "PaliGemma/llm/final_norm/scale"
  - "PaliGemma/llm/layers/attn/q_einsum/w"
  - "PaliGemma/llm/layers/attn/kv_einsum/w"
  - "PaliGemma/llm/layers/attn/attn_vec_einsum/w"
  - "PaliGemma/llm/layers/mlp/gating_einsum"
  - "PaliGemma/llm/layers/mlp/linear"
  - "PaliGemma/llm/layers/pre_attention_norm/scale"
  - "PaliGemma/llm/layers/pre_ffw_norm/scale"

trainable:
  - "PaliGemma/llm/layers/attn/q_einsum_1/**"
  - "PaliGemma/llm/layers/attn/kv_einsum_1/**"
  - "PaliGemma/llm/layers/attn/attn_vec_einsum_1/**"
  - "PaliGemma/llm/layers/mlp_1/**"
  - "PaliGemma/llm/layers/pre_attention_norm_1/**"
  - "PaliGemma/llm/layers/pre_ffw_norm_1/**"
  - "PaliGemma/llm/final_norm_1/**"
  - "action_in_proj/**"
  - "action_out_proj/**"
  - "time_mlp_in/**"
  - "time_mlp_out/**"
  - "time_proj/**"
```

---

## Step 5: 新增训练脚本

**文件**: `scripts/train_curr_v2.sh`

基于 `train_curr_12k.sh` 修改，指向新配置文件。

---

## Step 6: Smoke Test

在正式训练前验证：
1. 单 batch 前向：确认 `delta_sampling="biased"` 时 delta_mean ≈ 0.35+
2. 单 step 反向：确认 loss_smf 量级提升（从 ~0.0005 到 ~0.01+）
3. 梯度匹配：确认 smf_scale_ema 收敛到合理范围（10~100）
4. Checkpoint save/load 正常

---

## Step 7: 训练 15K steps

- 配置: `smf_curr_v2_libero.yaml`
- 保存 checkpoint at 3K, 6K, 9K, 12K, 15K
- 监控: loss_smf/fm ratio, smf_scale_ema, delta_mean

---

## Step 8: 评估

- 每个 checkpoint 在 libero_spatial 上 eval NFE=1, 250 episodes
- 对比基线: smf_base (80%), smf_curr_12k (62.4%)
- 成功标准: 1-step score > 80%

---

## 文件修改清单

| 文件 | 操作 | 改动摘要 |
|---|---|---|
| `src/smf_vla/training/smf_loss.py` | 修改 | `sample_r_t_curriculum` 新增 biased delta 采样；`compute_full_smf_loss` 支持 dynamic scale 和 delta_sampling 参数 |
| `src/smf_vla/training/jax_trainer.py` | 修改 | 支持 `smf_loss_scale="dynamic"`，实现梯度匹配 EMA 逻辑，新增 `delta_sampling` config 解析 |
| `configs/train/smf_curr_v2_libero.yaml` | 新增 | V2 配置文件 |
| `scripts/train_curr_v2.sh` | 新增 | V2 训练脚本 |
