# Eval Debug Guide

> 目标：记录 LIBERO-Plus / CALVIN eval 中已踩过的坑及根因，供后续 eval 和
> 方法开发参考。每条问题含**症状、根因、修复**三要素，并附 `file:line` 引用。
> 本文与 `docs/training-debug.md` 互补：后者关注训练，本文关注 eval。

---

## 1. 并行 Eval 的 GPU 显存管理

### 症状
两个 eval 进程并行启动时，第二个进程 OOM 崩溃（即使 GPU 有 97GB 空闲）。

### 根因
JAX 默认预分配 **75%** GPU 显存。eval 代码（`eval/common/policy_loader.py`、
`eval/libero_plus/main.py`、`eval/scripts/run_libero_parallel.sh`）**从不设置**
`XLA_PYTHON_CLIENT_MEM_FRACTION`，因此每个进程独占 75%，两进程合计 150% > 100%。

`run_libero_parallel.sh` 的 `--num-workers N` 机制只做 **task 级分片**
（`runner.py:237-239`：`task_ids = task_ids[worker_id::num_workers]`），
**不**做 GPU 分区。每个 worker 是独立的 Python 进程，各自加载完整的 3B 参数模型。

### 修复
在启动 eval 前手动 `export`，按并行进程数均分：
```bash
# 2 进程并行（各 ~44GB，总 ~88GB < 97GB）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

# 单进程（可用满）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
```
eval 代码本身不设此变量，必须外部注入。tmux 中运行时在 send-keys 命令里带上 export。

### 验证方法
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# 两进程各 ~44GB → OK
```

---

## 2. Finetuned Checkpoint 缺少 assets/ 目录

### 症状
日志出现：
```
INFO:root:Norm stats not found in /root/autodl-tmp/assets/pi05_libero/..., skipping.
INFO:eval.common.policy_loader:Using base checkpoint assets: .../checkpoints/pi05_libero/assets
INFO:root:Loaded norm stats from .../checkpoints/pi05_libero/assets/...
```
模型加载成功、推理正常，但 0% 成功率（动作完全错误）。

### 根因
DMF/Pi-Flow 的 finetuned checkpoint 目录（`checkpoints/dmf_finetuned/step_0030000/`）
**没有 `assets/` 子目录**，因为训练脚本只保存 `params/` 和 `config.yaml`，不复制
norm_stats。`policy_loader.py` 的所有方法分支都有相同的 fallback 逻辑：

```python
# policy_loader.py:187-190 (DMF), 266-269 (SMF), 338-341 (Pi-Flow), 417-420 (FreeFlow), 493-496 (Pi05)
assets_dir = checkpoint_path / "assets"
if not assets_dir.exists():
    assets_dir = base_ckpt / "assets"          # 回退到 pi05_libero base
    logger.info(f"Using base checkpoint assets: {assets_dir}")
norm_stats = _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)
```

回退到 base checkpoint 的 norm_stats **在大多数情况下是正确的**——因为
`pi05_libero` 和 libero-plus-training 的 norm_stats 数值接近（state 差异 ~5%，
actions 差异更大但仍在同一量级）。

但当训练数据集的 norm_stats 与 base 差异较大时（如 libero-plus-training 的
`actions[3].mean = -2.988` vs base 的 `0.000`），回退会导致动作反归一化错误。

### 修复
将训练数据集的 `norm_stats.json` 复制到 finetuned checkpoint 的 assets 目录：
```bash
for step in step_0010000 step_0020000 step_0030000; do
  dir="checkpoints/dmf_finetuned/$step/assets/physical-intelligence/libero"
  mkdir -p "$dir"
  cp datasets/libero-plus-training/norm_stats.json "$dir/norm_stats.json"
done
```
**注意：** 这只解决了 norm_stats 来源问题。如果训练和 eval 的归一化**方案**
不同（见 §3），仅换文件无效。

### 更好的长期修复
训练脚本（`run_train.py`）应在保存 checkpoint 时自动复制 `assets/` 目录，
与 openpi 官方 pipeline 保持一致。

---

## 3. 归一化方案不匹配：Z-score vs Quantile

### 症状
DMF checkpoint 加载成功、norm_stats 正确、无 NaN/error，但 1-NFE 成功率 = 0%。
所有 episode 都跑到 `max_steps`（230 步）后 FAILURE。对照实验：pi05 base 在
相同 quick 模式下 = 100%。

### 根因
**训练和 eval 使用了不同的动作归一化方案。**

| | 归一化方案 | 代码位置 |
|---|---|---|
| **DMF 训练** | z-score: `x_0 = (a - mean) / std` | `dmf_loss.py:66` |
| **Pi-Flow 训练** | z-score（同上，通过 teacher） | `piflow_loss.py` |
| **Eval (pi05 base)** | quantile: `(x+1)/2*(q99-q01)+q01` | `policy_loader.py:509` → `transforms.Unnormalize` |

eval 的 `use_quantile_norm` 来自 openpi config：
```python
# openpi/training/config.py:187
use_quantile_norm = model_config.model_type != ModelType.PI0
# pi0.5 → ModelType.PI05 ≠ PI0 → True
```
所有方法分支默认用 `data_config.use_quantile_norm`（= True），但 DMF 训练
用 z-score，两者不兼容。

### 修复
在 `policy_loader.py` 中为 DMF 分支强制 `use_quantiles=False`：
```python
# DMF 分支 (policy_loader.py:504, 509)
_transforms.Normalize(norm_stats, use_quantiles=False),
_transforms.Unnormalize(norm_stats, use_quantiles=False),
```
**注意：** Pi-Flow 的 teacher 输出在 quantile 空间，因此 Pi-Flow 应保持
`use_quantiles=True`（默认）。Pi-Flow 的 0% 有不同的根因（见 §5）。

### 诊断方法
用随机输入直接调用 `policy.infer()`，对比输出动作范围：
```python
# DMF 1-NFE (z-score 空间，未修复): [-3.9, 5.3]  ← 过大
# DMF 10-NFE:                         [-1.1, 0.29] ← 合理
# pi05 base (quantile 空间):          [-0.35, 0.52] ← 正常
```
1-NFE 输出远超正常范围 → 归一化不匹配或推理路径问题。

---

## 4. DMF 时间采样不覆盖 1-NFE 推理所需的大间隔

### 症状
§3 的归一化修复后，DMF 1-NFE 仍然 0%。动作范围 [-3.9, 5.3] 远超正常。
但 DMF 10-NFE 正常（[-1.1, 0.29]）。

### 根因
DMF 训练使用 **logit-normal** 时间采样，导致 `(t-r)` 间隔集中在 ~0.35，
1-NFE 推理需要间隔 = 1.0，完全在训练分布外。

**训练时间采样**（`dmf_loss.py:76-80`）：
```python
t_fm = ln_sampler(fm_rng, ..., p_mean=0.0, p_std=1.0)    # sigmoid(N(0,1))
ln_1 = ln_sampler(l1_rng, ..., p_mean_t=0.4, p_std_t=1.0)
ln_2 = ln_sampler(l2_rng, ..., p_mean_r=-1.2, p_std_r=1.0)
t_mf = max(ln_1, ln_2)    # mean ≈ 0.60
r_mf = min(ln_1, ln_2)    # mean ≈ 0.23
# 间隔 (t_mf - r_mf) 的分布：
#   mean=0.352, p95=0.719, >0.9 仅 0.06%
```

**1-NFE 推理**（`pi05_dmf.py:199-213`）：
```python
t_steps = linspace(1.0, 0.0, num_steps + 1)  # [1.0, 0.0]
# 单步: t=1.0, r=0.0, 间隔=1.0  ← 训练从未见过
```

MeanFlow 的训练目标是 `u_tgt = v_t + (r-t) * du/dt`，当 `(r-t)=-1.0` 时
`du/dt` 的误差被放大，而模型在此处外推 → 动作预测严重偏差。

10-NFE 每步间隔仅 0.1，在训练分布内，因此正常。

### 修复
在 `dmf_loss.py` 中添加 `time_sampling="uniform"` 模式，与 SMF 一致：
```python
if time_sampling == "uniform":
    t_fm = jax.random.uniform(fm_rng, (B,), 0.0, 1.0)
    t_mf = jax.random.uniform(t_rng, (B,), 0.0, 1.0)
    r_mf = jax.random.uniform(r_rng, (B,), 0.0, 1.0) * t_mf
```
配置中设置 `time_sampling: "uniform"`。需重新训练。

**对比 SMF**（`smf_loss.py:52-56`）：SMF 一直用 `t~U(0,1), r~U(0,t)`，
因此 SMF 的 1-NFE 推理正常。

---

## 5. Pi-Flow 的 Teacher-Student State 归一化不匹配

### 症状
Pi-Flow 1-NFE 成功率 = 0%，与 DMF 症状相同。但 Pi-Flow 的时间采样
**没有问题**——训练和推理的时间表完全一致。

### 根因
Pi-Flow 采用 teacher-student 蒸馏（`jax_trainer.py:288-340`）：
- **Teacher** = 冻结的 pi0.5，训练时接收 **quantile 归一化** 的 state
- **Pi-Flow 训练**时 data_loader 产出 **raw state**（`data_loader.py:369`，无 Normalize）
- Teacher 收到 raw state → velocity 预测错误 → student 学到错误的 velocity

Pi-Flow 的 loss 中 `actions` 参数**仅用于 shape**（`piflow_loss.py:41` 注释），
实际监督完全来自 teacher。因此 teacher 的输入错误会直接传播到 student。

### 与 DMF 的区别

| | DMF | Pi-Flow |
|---|---|---|
| 时间采样 | logit-normal（问题） | 确定性分段（无问题） |
| 训练=推理? | ❌ | ✅ |
| 根因 | 大间隔外推 | teacher state 未归一化 |
| action 归一化 | z-score（`use_quantiles=False`） | teacher 输出在 quantile 空间（`use_quantiles=True`） |
| 修复方向 | 改时间采样为 uniform | 训练时对 state 做 quantile 归一化 |

### 修复
在 Pi-Flow 的 trainer 中对 state 应用 quantile 归一化（与 teacher 训练
pipeline 一致），需重训练。eval 侧应保持 `use_quantiles=True`（默认）。

---

## 6. Per-suite 结果文件名不含 model_type

### 症状
两个不同方法的 eval 并行运行时，结果文件可能互相覆盖。

### 根因
`runner.py:401-403` 将 per-suite JSON 保存到**硬编码**的 `results/libero_plus/`
目录，文件名格式（`utils.py:103`）：
```
{timestamp}_{suite}_{nfe}nfe_{success_rate}pct[_w{worker_id}_of_{N}].json
```
**不含 model_type**。两个并行 eval 如果在同一秒完成同一 suite 且成功率
相同，文件会冲突。

combined JSON 按 model_type 分目录（`main.py:107`：
`eval/results/{model_type}/libero_plus/`），不冲突。

### 实际风险评估
**冲突概率极低**：文件名含秒级时间戳 + 成功率，两个方法几乎不可能
在同一秒完成同一 suite 且成功率完全相同。但在大规模并行场景下需注意。

### 修复建议
在文件名中加入 model_type：`utils.py:103` 的 filename 模板添加
`{model_type}_` 前缀。或改为按 model_type 分子目录保存 per-suite 文件。

---

## 7. JIT 编译首次延迟被误计为推理延时

### 症状
`all_latencies_ms` 数组的第一个值异常大（~40000ms），严重拉高 avg latency。

### 根因
首次 `policy.infer()` 调用触发 JAX JIT 编译，耗时 ~40s。`runner.py:156-159`
用 `time.monotonic()` 包裹每次 infer 调用，首次编译被计为一次"推理"。

### 修复（分析侧）
延时统计时**排除 `all_latencies_ms[0]`**（JIT 编译 outlier）。代码已正确记录
p50/p95/p99（这些百分位不受单个 outlier 影响），但 `avg_latency_ms` 会受影响。

---

## 8. 诊断流程速查

遇到 eval 0% 成功率时，按以下顺序排查：

```
1. 对照实验：pi05 base 在 quick 模式跑一次
   ├─ pi05 也 0% → eval 环境问题（数据集、libero_plus 包、GPU 驱动）
   └─ pi05 正常  → 模型/归一化问题，继续 ↓

2. 检查日志中 norm_stats 加载路径
   ├─ "Using base checkpoint assets" → 可能回退到错误的 norm_stats（§2）
   └─ "Loaded norm stats from .../finetuned/.../assets" → 正确

3. 检查动作输出范围
   ├─ 用随机输入直接调 policy.infer()，打印 actions.min/max
   ├─ 范围 > [-3, 3] → 归一化不匹配（§3）或推理路径问题（§4/§5）
   └─ 范围正常 → 模型本身未训练好

4. 对比 1-NFE vs 10-NFE
   ├─ 10-NFE 正常，1-NFE 异常 → 时间采样/推理路径问题（§4）
   └─ 两者都异常 → 归一化或模型问题

5. 检查训练日志
   ├─ loss 是否下降
   ├─ 训练是否中途崩溃（WandB runtime 异常短）
   └─ 训练用的时间采样与推理是否匹配
```

---

## 9. 相关文件索引

| 文件 | 作用 |
|------|------|
| `eval/common/policy_loader.py` | 模型加载 + norm_stats + 归一化方案 |
| `eval/libero_plus/runner.py` | episode 执行 + 延时记录 + 结果保存 |
| `eval/libero_plus/main.py` | CLI 入口 + checkpoint 默认路径 |
| `eval/common/utils.py` | 结果 JSON 构建 + 文件名格式 |
| `eval/libero_plus/presets.py` | quick/normal/fullset 配置 |
| `eval/scripts/run_libero_parallel.sh` | 并行 worker 启动脚本 |
| `dmf/src/dmf_vla/training/dmf_loss.py` | DMF loss + 时间采样 |
| `dmf/src/dmf_vla/models/pi05_dmf.py` | DMF 模型 + sample_actions |
| `piflow/src/piflow_vla/training/piflow_loss.py` | Pi-Flow teacher-student loss |
| `piflow/src/piflow_vla/models/pi05_piflow.py` | Pi-Flow 模型 + sample_actions |
| `smfVLA/src/smf_vla/training/smf_loss.py` | SMF loss + 时间采样（参考正确实现） |
