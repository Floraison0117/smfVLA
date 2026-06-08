# SMF-Full 四技巧训练计划设计

## 目标

从 pi05-libero checkpoint 出发，同时使用 Curriculum、Anchor loss、BPL loss、Dynamic Loss Scaling 四个技巧，分别在 LIBERO 和 LIBERO-Plus 上训练 SMF 模型，提升 1-NFE 推理性能。

## 背景

### 现有代码问题

1. **Anchor 和 BPL 损失是死代码**：`run_train.py` 从未加载 teacher 模型，`teacher_fn=None`，`teacher_model=None`，这两个损失被静默跳过
2. **Dynamic scaling 未启用**：`smf_full_libero.yaml` 没有设置 `smf_loss_scale: dynamic`，默认固定 1.0
3. **libero-plus-training 数据格式不兼容**：v2.1 格式 vs v2.0，列名不同，action 表示不同

### 数据集对比

| 属性 | libero | libero-plus-training |
|------|--------|---------------------|
| LeRobot 版本 | v2.0 | v2.1 |
| Episodes | 1,693 | 14,347 |
| Frames | 273,465 | 2,238,036 |
| FPS | 10 | 20 |
| 图像列名 | `image`, `wrist_image` | `observation.images.front`, `observation.images.wrist` |
| Action 列名 | `actions` | `action` |
| State 列名 | `state` | `observation.state` |

### Action Space 差异

libero-plus-training 的 action 表示与 libero 显著不同：
- z 分量：均值 -0.78 vs -0.09
- rx 分量：均值 -2.99 vs 0.0007

必须使用各自数据集的 `norm_stats.json` 进行归一化。

## 设计方案

### Phase 1：LIBERO 训练（验证四技巧组合）

#### 1.1 代码修改：加载 Teacher 模型

**文件**: `scripts/run_train.py`

修改内容：
- 创建第二个 `Pi05SMF` 实例作为 teacher，加载同一个 `checkpoints/base/pi05_libero`
- 冻结 teacher 所有参数（不训练，不保留梯度）
- 构造 `teacher_fn` 闭包：`teacher_fn(None, obs, z, r, t) -> velocity`，用于 Anchor loss
- 构造 `teacher_model` 对象，实现 `extract_hidden_states(obs, x, layer_indices)` 方法，用于 BPL loss
- 将两者传入 `SMFTrainer` 的 loss 函数

**显存考虑**：
- Teacher 用 bf16 冻结，不保留梯度
- 预计额外显存开销 ~2-3 GB（432M trainable params 的 2x forward）

#### 1.2 代码修改：启用 Dynamic Scaling

**文件**: `configs/train/smf_full_libero.yaml`

添加配置项：
```yaml
smf_loss_scale: dynamic
```

Dynamic scaling 的实现已存在于 `jax_trainer.py`（`_update_smf_scale` 方法）：
- 两次独立 backward pass 分别计算 fm 和 smf 的梯度范数
- 目标比例 = g_fm / (g_smf + 1e-8)
- EMA 平滑（alpha=0.99），clamp 到 [1.0, 200.0]

#### 1.3 训练配置

```yaml
# configs/train/smf_full_libero.yaml（修改版）
model_path: checkpoints/base/pi05_libero
teacher_path: checkpoints/base/pi05_libero  # 新增
dataset_path: data/libero

flow_ratio: 0.5
time_conditioning: decte
encoder_depth: 6
smf_loss_scale: dynamic  # 新增

use_curriculum: true
use_anchor: true
use_bpl: true

# Curriculum 参数（已有）
delta_min: 0.05
delta_final: 1.0
delta_floor: 0.001

# Anchor 参数（已有）
alpha_anchor_max: 0.1
anchor_warmup_steps: 3000
anchor_cooldown_steps: 7500
anchor_delta_max: 0.3
anchor_teacher_nfe: 2

# BPL 参数（已有）
bpl_warmup_start: 4500
bpl_warmup_end: 10500
bpl_alpha_max: 0.05
bpl_layer_indices: [12, 16]
bpl_layer_weights: [0.5, 1.0]

lr: 3e-5
batch_size: 32
total_steps: 15000
save_every: 3000
```

#### 1.4 损失时间线

```
Step    0 ──── 3000 ──── 4500 ──── 7500 ──── 10500 ──── 15000
Anchor  warmup→0.1  peak 0.1  cooldown→0   (disabled)
BPL     0               warmup→0.05           stay 0.05
SMF     ✓ (全程活跃，自洽性损失)
FM      ✓ (全程活跃，瞬时速度损失)
Curriculum  delta: 0.05 →── cosine ─────→ 1.0
DynScale    ✓ (EMA 平衡 fm/smf 梯度，全程活跃)
```

#### 1.5 评估计划

每 3000 步保存 checkpoint，评估：
- LIBERO 标准环境：1-NFE 和 10-NFE，4 个 suite
- 重点对比 SMF-Base 的 1-NFE 结果

---

### Phase 2：LIBERO-Plus 训练

#### 2.1 代码修改：数据加载器支持 v2.1

**文件**: `src/smf_vla/training/data_loader.py`

修改内容：
- 添加列名映射参数或自动检测格式
- v2.0: `df["image"]`, `df["wrist_image"]`, `df["actions"]`, `df["state"]`
- v2.1: `df["observation.images.front"]`, `df["observation.images.wrist"]`, `df["action"]`, `df["observation.state"]`
- 自动检测：检查 DataFrame 列名中是否存在 `observation.images.front`
- norm_stats 从数据集目录的 `norm_stats.json` 加载（已有逻辑）

#### 2.2 新建训练配置

**文件**: `configs/train/smf_full_libero_plus.yaml`（新建）

```yaml
model_path: checkpoints/base/pi05_libero
teacher_path: checkpoints/base/pi05_libero
dataset_path: data/libero-plus-training

# 与 smf_full_libero.yaml 相同的 SMF 参数
flow_ratio: 0.5
time_conditioning: decte
encoder_depth: 6
smf_loss_scale: dynamic
use_curriculum: true
use_anchor: true
use_bpl: true

lr: 3e-5
batch_size: 32
total_steps: 15000
save_every: 3000
```

#### 2.3 评估

- 在 LIBERO-Plus 环境上评估（`eval_libero_plus.py`）
- 重点：1-NFE 是否能从 0% 提升到可用水平
- 10-NFE 作为基线对比

---

## 训练顺序

```
Phase 1: LIBERO
  └─ SMF-Full (pi05_libero teacher) → 评估 1-NFE / 10-NFE

Phase 2: LIBERO-Plus（Phase 1 完成后）
  ├─ 修改数据加载器
  └─ SMF-Full (pi05_libero teacher) → 评估 1-NFE / 10-NFE
```

## 关键风险

1. **Teacher 质量**：pi05_libero 是原始 checkpoint，teacher 的 2-step Euler 积分可能不够准。如果效果不好，可以换成 SMF-Base checkpoint。
2. **显存开销**：加载两个模型实例可能需要额外显存。Teacher 用 bf16 冻结。
3. **Action Space 泛化**：pi05_libero 在 libero action space 上训练，直接在 libero-plus-training 上 fine-tune 可能需要较长适应期。
4. **libero-plus-training 视频缺失**：chunk 011-014 没有视频目录，可能影响数据加载。
