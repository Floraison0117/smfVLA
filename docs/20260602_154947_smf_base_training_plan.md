# SMF-Base 训练计划：SplitMeanFlow 在 pi0.5 Action Head 上的 JAX 全量微调

> 日期: 2026-06-02
> 目标: 在 libero 数据集上使用 SplitMeanFlow 训练 pi0.5 的 action head，实现 1-NFE 高质量动作生成

---

## 1. 方法概述

### 1.1 SplitMeanFlow (SMF) 核心思想

标准 flow matching 训练模型预测 **瞬时速度** v(z_t, t) = ε - x。
SplitMeanFlow 训练模型预测 **平均速度** u_θ(z_t, r, t, c)，即从时间 r 到 t 的平均流速。

关键性质：平均速度满足自举（self-consistency）：
```
u(z_t, r, t) = (1-λ) · u(z_s, r, s) + λ · u(z_t, s, t)
其中 s = (1-λ)·t + λ·r
```

推理时，从纯噪声 z_1 一步生成：z_0 = z_1 - u_θ(z_1, 0, 1, c)

### 1.2 SMF-Base 训练设计（来自 plan0515.md）

- **Time Embedding**: concat [E(t), E(r)] + time_proj（投影回原始维度）
- **Time Proj 初始化**: [I, 0]，t 对应 identity，r 对应 0（初始等价于原始 flow matching）
- **Flow Ratio**: p = 0.3（以概率 p 设置 r = t，退化为普通 flow matching）
- **Loss**: loss_total = loss_smf + loss_fm
- **只微调 action head**，冻结 VLM backbone

---

## 2. 模型架构分析

### 2.1 pi0.5 参数树（已验证）

```
PaliGemma/
├── img/                    # SigLIP 视觉编码器 → 冻结
├── llm/
│   ├── embedder/           # Token embedding → 冻结
│   ├── layers/
│   │   ├── attn/
│   │   │   ├── q_einsum        # VLM attention Q → 冻结
│   │   │   ├── q_einsum_1      # Action expert attention Q → 训练
│   │   │   ├── kv_einsum       # VLM attention KV → 冻结
│   │   │   ├── kv_einsum_1     # Action expert attention KV → 训练
│   │   │   ├── attn_vec_einsum   # VLM attention out → 冻结
│   │   │   └── attn_vec_einsum_1 # Action expert attention out → 训练
│   │   ├── mlp/                # VLM MLP (18层, 2048→16384) → 冻结
│   │   ├── mlp_1/              # Action expert MLP (18层, 1024→4096) → 训练
│   │   ├── pre_attention_norm/   # VLM norm → 冻结
│   │   ├── pre_attention_norm_1/ # Action expert norm → 训练
│   │   ├── pre_ffw_norm/         # VLM norm → 冻结
│   │   └── pre_ffw_norm_1/       # Action expert norm → 训练
│   ├── final_norm/         # VLM final norm → 冻结
│   └── final_norm_1/       # Action expert final norm → 训练
action_in_proj/             # Action → token 投影 → 训练
action_out_proj/            # Token → action 投影 → 训练
time_mlp_in/                # Time embedding MLP in → 训练
time_mlp_out/               # Time embedding MLP out → 训练
```

### 2.2 冻结/训练参数总结

| 组件 | 参数路径模式 | 状态 |
|------|-------------|------|
| SigLIP 视觉编码器 | `PaliGemma/img/**` | ❄️ 冻结 |
| Token embedding | `PaliGemma/llm/embedder/**` | ❄️ 冻结 |
| VLM attention/MLP/norm | `PaliGemma/llm/layers/{attn,mlp,pre_*norm}/{q_einsum,kv_einsum,attn_vec_einsum,gating_einsum,linear,scale}` (不含 `_1` 后缀) | ❄️ 冻结 |
| VLM final norm | `PaliGemma/llm/final_norm/**` | ❄️ 冻结 |
| Action expert attention | `PaliGemma/llm/layers/attn/*_1/**` | 🔥 训练 |
| Action expert MLP | `PaliGemma/llm/layers/mlp_1/**` | 🔥 训练 |
| Action expert norm | `PaliGemma/llm/layers/{pre_*norm_1,final_norm_1}/**` | 🔥 训练 |
| Action projections | `action_in_proj/**`, `action_out_proj/**` | 🔥 训练 |
| Time MLP | `time_mlp_in/**`, `time_mlp_out/**` | 🔥 训练 |
| **新增** time_proj | `time_proj/**` | 🔥 训练 |

---

## 3. 代码修改清单

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `src/smf_vla/models/pi05_smf.py` | SMF 修改版 Pi0 模型（JAX/NNX） |
| `src/smf_vla/training/smf_loss.py` | SplitMeanFlow loss 实现 |
| `src/smf_vla/training/jax_trainer.py` | JAX 训练循环 |
| `src/smf_vla/training/data_loader.py` | libero 数据加载（LeRobot 格式） |
| `src/smf_vla/training/freeze_utils.py` | 参数冻结/过滤工具 |
| `scripts/train.sh` | 训练启动脚本 |
| `configs/train/smf_base_libero.yaml` | SMF-Base 训练配置 |

### 3.2 修改文件

| 文件 | 修改内容 |
|------|----------|
| `src/smf_vla/models/pi05_smf.py` | 继承 Pi0，修改 `embed_suffix` 和 `compute_loss` |

---

## 4. 核心算法实现

### 4.1 Time Embedding 修改（SMF-Base）

原始 pi0.5 的 time embedding：
```python
time_emb = posemb_sincos(t, width)  # [B, width]
time_emb = time_mlp_in(time_emb)    # [B, width]
time_emb = swish(time_emb)
time_emb = time_mlp_out(time_emb)   # [B, width]
time_emb = swish(time_emb)           # → adarms_cond
```

SMF-Base 修改：
```python
e_t = posemb_sincos(t, width)
e_t = time_mlp(swish(time_mlp_in(e_t)))  # [B, width]

e_r = posemb_sincos(r, width)
e_r = time_mlp(swish(time_mlp_in(e_r)))  # [B, width]

# Concat + project
time_emb = concat([e_t, e_r], axis=-1)   # [B, 2*width]
time_emb = time_proj(time_emb)            # [B, width]
# time_proj 初始化为 [I, 0]
# → adarms_cond
```

### 4.2 SplitMeanFlow 训练 Step

```python
def compute_smf_loss(model, observation, actions, rng):
    # Step 1: 归一化 action
    x_norm = (actions - action_mean) / action_std

    # Step 2: 采样噪声
    noise = normal(rng, x_norm.shape)

    # Step 3: 采样时间
    t = uniform(0, 1, batch)
    r = uniform(0, t, batch)

    # Step 4: Bernoulli 采样，以概率 p 设置 r = t
    m = bernoulli(p=0.3, batch)
    r = where(m, t, r)  # m=1 时 r=t

    # Step 5: 线性插值
    z_t = (1 - t) * x_norm + t * noise

    # Step 6: Self-consistency 分支 (m=0, r < t)
    lam = uniform(0, 1, batch)
    s = (1 - lam) * t + lam * r

    u_2 = model(z_t, s, t, c)           # 从 s 到 t 的平均速度
    z_s = z_t - (t - s) * stop_grad(u_2)
    u_1 = model(z_s, r, s, c)           # 从 r 到 s 的平均速度

    target = (1 - lam) * stop_grad(u_1) + lam * stop_grad(u_2)
    pred = model(z_t, r, t, c)
    loss_smf = mean(||pred - target||²)

    # Step 7: Flow matching 分支 (m=1, r = t)
    loss_fm = masked_mean(
        ||u_theta(z_t, t, t, c) - (noise - x_norm)||²,
        mask=m
    )

    # Step 8: 总损失
    loss_total = loss_smf + loss_fm
    return loss_total
```

### 4.3 推理（1-NFE）

```python
def sample_1nfe(model, observation, rng):
    noise = normal(rng, (batch, action_horizon, action_dim))
    # z_0 = z_1 - u_θ(z_1, 0, 1, c)
    actions_norm = noise - model(noise, r=0, t=1, c=observation)
    actions = actions_norm * action_std + action_mean
    return actions
```

---

## 5. 训练配置

```yaml
# SMF-Base 训练配置
method: smf_base
checkpoint: /root/autodl-tmp/smfVLA/checkpoints/base/pi05_libero
dataset: libero  # LeRobot 格式

# 模型
action_dim: 7
action_horizon: 10
pi05: true

# 训练超参
learning_rate: 3e-5
weight_decay: 0.01
warmup_ratio: 0.03
gradient_clipping: 1.0
batch_size: 64  # 根据 GPU 显存调整
training_steps: 15000
precision: bf16

# SMF 特有
flow_ratio: 0.3          # p = 0.3，Bernoulli 概率
time_conditioning: concat # [E(t), E(r)] + time_proj
time_sampling: uniform    # t ~ U(0,1), r ~ U(0,t)

# 冻结策略
freeze:
  - "PaliGemma/img/**"
  - "PaliGemma/llm/embedder/**"
  - "PaliGemma/llm/final_norm/**"
  # VLM 的 attn/mlp/norm（不含 _1 后缀）
trainable:
  - "PaliGemma/llm/layers/attn/*_1/**"
  - "PaliGemma/llm/layers/mlp_1/**"
  - "PaliGemma/llm/layers/pre_*norm_1/**"
  - "PaliGemma/llm/final_norm_1/**"
  - "action_in_proj/**"
  - "action_out_proj/**"
  - "time_mlp_in/**"
  - "time_mlp_out/**"
  - "time_proj/**"

# 优化器
optimizer: AdamW
ema_decay: null  # SMF-Base 不使用 EMA

# 保存
save_every: 3000  # 每 3K steps 保存 checkpoint
log_every: 100
```

---

## 6. 实现步骤

### Phase 1: 基础设施（预计 2-3h）

1. **创建 `freeze_utils.py`**
   - 实现 `build_freeze_filter(params, freeze_patterns, trainable_patterns)`
   - 返回 JAX pytree 的 freeze mask
   - 验证冻结参数数量 vs 训练参数数量

2. **创建 `data_loader.py`**
   - 使用 lerobot 的 `LeRobotDataset` 加载 libero 数据
   - 输出格式: `{observation/image, observation/wrist_image, observation/state, actions, prompt}`
   - 支持 action normalization（从 checkpoint 的 norm_stats 加载）

3. **创建 `smf_loss.py`**
   - 实现 `compute_smf_loss()` 函数
   - 实现 `compute_flow_matching_loss()` 函数
   - 实现 `sample_r_t()` 时间采样（含 Bernoulli p=0.5）

### Phase 2: 模型修改（预计 2-3h）

4. **创建 `pi05_smf.py`**
   - 继承 `openpi.models.pi0.Pi0`
   - 新增 `time_proj` 参数（`nnx.Linear(2*width, width)`）
   - 修改 `embed_suffix()`：支持 `(r, t)` 双时间输入
   - 修改 `compute_loss()`：使用 SMF loss
   - 修改 `sample_actions()`：支持 1-NFE 推理
   - `time_proj` 初始化为 `[I, 0]`

5. **创建 `jax_trainer.py`**
   - JAX JIT 编译训练 step
   - 梯度裁剪 = 1.0
   - 学习率 warmup + cosine decay
   - checkpoint 保存/加载（orbax）
   - 训练日志记录（loss, grad_norm, lr）

### Phase 3: 集成与测试（预计 1-2h）

6. **创建 `configs/train/smf_base_libero.yaml``**
7. **创建 `scripts/train.sh`**
8. **Smoke test**
   - 单 batch 前向传播
   - 单 step 反向传播
   - checkpoint save/load
   - 1-NFE 推理 shape check

---

## 7. 关键技术细节

### 7.1 pi0.5 Time Embedding 结构

pi0.5 使用 adaRMSNorm 注入 timestep：
- `time_mlp_in`: Linear(width → width)
- `time_mlp_out`: Linear(width → width)
- 输出作为 `adarms_cond` 传入每个 transformer block
- Block 内的 `RMSNorm` 使用 `adarms_cond` 做 scale/shift

SMF-Base 在 `time_mlp_out` 之后新增 `time_proj`：
```python
# 原始
adarms_cond = time_mlp_out(swish(time_mlp_in(posemb_sincos(t))))

# SMF-Base
e_t = time_mlp_out(swish(time_mlp_in(posemb_sincos(t))))
e_r = time_mlp_out(swish(time_mlp_in(posemb_sincos(r))))
adarms_cond = time_proj(concat([e_t, e_r]))  # time_proj 初始化为 [I, 0]
```

### 7.2 NNX 参数过滤

JAX/NNX 的 freeze 通过 `nnx.GraphDef` + `nnx.State` 实现：
```python
graphdef, state = nnx.split(model)
# state 是一个 PyTree，可以用 jax.tree.map 操作
frozen_mask = jax.tree.map(lambda path, param: is_frozen(path), state)
# 训练时只更新 non-frozen 参数
```

### 7.3 Action Normalization

从 checkpoint 的 `norm_stats.json` 加载：
```python
# actions 的 q01/q99 用于 quantile normalization
action_mean = norm_stats["actions"]["mean"]  # [7]
action_std = norm_stats["actions"]["std"]    # [7]
```

---

## 8. 验证清单

- [ ] JAX GPU 可用：`jax.devices()` 显示 CUDA device
- [ ] Checkpoint 加载成功：参数 shape 匹配
- [ ] Freeze 正确：冻结参数梯度为 0
- [ ] Time proj 初始化：`time_proj` 权重为 `[I, 0]`
- [ ] SMF loss 数值：初始 loss ≈ 原始 flow matching loss
- [ ] 1-NFE 推理：输出 shape 正确，值域合理
- [ ] GPU 显存：不 OOM
- [ ] Checkpoint 保存/加载：orbax 格式正确

---

## 9. 风险与注意事项

1. **Action expert 与 VLM 共享 Gemma Module**：参数通过 `_1` 后缀区分，freeze filter 需要精确匹配
2. **adaRMSNorm 接口**：`adarms_cond` 需要传入每个 Block，修改 `embed_suffix` 时要保持接口兼容
3. **KV Cache 推理**：`sample_actions` 使用 KV cache，SMF 修改不能破坏 cache 逻辑
4. **Batch size**：pi0.5 模型较大（~3B params），需要根据 GPU 显存调整 batch size
5. **数据集**：libero 数据集尚未下载，需要先下载 LeRobot 格式数据
