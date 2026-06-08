# Plan: Curriculum SMF 训练 12000 steps + 修复 SMF Loss 过小

## Context

从之前的 eval 结果来看，step_4000~6000 是最优区间（86%→80%），之后 step_8000 出现 dip（67.2%），step_10k~15k 逐步回升但未超过早期。现在要从 pi05-libero base checkpoint 出发，采用 curriculum 调度重新训练 12000 steps，并解决 SMF loss 量级过小的问题。

### SMF Loss 过小的原因分析

在当前实现中 `loss_total = loss_smf + loss_fm`，存在两个问题：

1. **Self-consistency target 使用 stop-gradient**：随着模型变好，`target = (1-λ)*sg(u_1) + λ*sg(u_2)` 趋近于 `pred = model(z_t, r, t)`，loss 自然衰减。但这不等于模型学到了正确的动作 —— 可能收敛到自洽但错误的固定点。

2. **量级失衡**：SMF loss（3次前向，self-consistency MSE）随着训练快速下降到 ~0.001-0.01 量级，而 FM loss（1次前向，直接监督）保持在 ~0.01-0.1 量级。总 loss 中 FM 分支主导梯度，SMF 分支的信号被淹没。

## 方案

### 1. 新建训练配置 `smf_curr_12k_libero.yaml`

基于 `smf_curr_libero.yaml`，修改：

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `training_steps` | 15000 | 12000 | 用户指定 |
| `save_every` | 3000 | 3000 | 保持不变，产出 step_3k/6k/9k/12k |
| `flow_ratio` | 0.5 | 0.5 | 保持，50% 样本用 FM 防止退化 |
| `smf_loss_scale` | — | **2.0** | **新增**，放大 SMF loss 量级 |
| `delta_floor` | 0.001 | **0.01** | **提升最小 delta**，避免 SMF 样本的 delta 过小导致 loss 可忽略 |
| `checkpoint_dir` | smf_curr | **smf_curr_12k** | 新目录 |
| `log_dir` | smf_curr | **smf_curr_12k** | 新目录 |

其他保持不变：`delta_min=0.05, delta_final=1.0, use_curriculum=true, lr=3e-5, batch_size=32`

### 2. 修改 loss 函数支持 `smf_loss_scale`

**文件**: `smfVLA/src/smf_vla/training/smf_loss.py`

在 `compute_full_smf_loss()` 中：
- 新增参数 `smf_loss_scale: float = 1.0`
- 修改总 loss 计算：`loss_total = smf_loss_scale * loss_smf + loss_fm`
- metrics 中新增 `loss_smf_scaled` 记录放大后的值

### 3. 修改 trainer 传递 `smf_loss_scale`

**文件**: `smfVLA/src/smf_vla/training/jax_trainer.py`

- `__init__` 中读取 `smf_loss_scale` 配置（默认 1.0）
- `loss_fn` 中传递给 `compute_full_smf_loss()`

### 4. 新建训练启动脚本

**文件**: `smfVLA/scripts/train_curr_12k.sh`

基于 `train.sh`，指定新配置文件。

## 涉及文件

| 文件 | 操作 |
|------|------|
| `smfVLA/configs/train/smf_curr_12k_libero.yaml` | **新建** |
| `smfVLA/src/smf_vla/training/smf_loss.py` | **修改** — 添加 `smf_loss_scale` 参数 |
| `smfVLA/src/smf_vla/training/jax_trainer.py` | **修改** — 读取并传递 `smf_loss_scale` |
| `smfVLA/scripts/train_curr_12k.sh` | **新建** |

## 验证

1. 启动训练：`bash scripts/train_curr_12k.sh`
2. 观察日志中 `smf` 和 `fm` loss 的量级比例，确认 SMF loss 不再被 FM loss 淹没
3. 训练完成后在 libero_spatial 上 eval 4 个 checkpoint（step_3k/6k/9k/12k），每个 25 episodes
4. 对比之前 smf_base 的 step_3k~12k 结果
