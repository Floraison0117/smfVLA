# Training Debug Workflow

> 目标：在 GPU smoke test 昂贵（3B 模型 JIT 编译数分钟 + 预分配显存）的前提下，
> 用分层验证把错误**尽早暴露**，最大化每一次 GPU run 的信息密度。
> 本文档为 agent 可执行的 checklist。以 DMF 为主例，泛化到所有方法。

## 0. 核心原则

- **能 CPU/静态抓的，绝不留给 GPU。**
- **上 GPU 时用 fake-data**（去掉磁盘 IO 干扰），一次 run 同时验证 JIT/显存/NaN/grad。
- **关键 env 必须在 `import jax` 前设置**——这是最常见的 OOM 根因。
- **freeze 正确性看启动日志的参数计数**，不靠猜。

---

## 1. 分层 Debug Checklist

### Layer 0 — 静态检查（无 GPU，秒级）

- [ ] config 字段完整：`method` / `training_steps` / `batch_size` / `checkpoint` / `dataset_path`
- [ ] base checkpoint 存在：`ls checkpoints/pi05_libero/params`（所有方法共用此 base）
- [ ] `dataset_path` 存在；若不存在确认会走 fake-data fallback（`dmf/scripts/run_train.py:159-166`）
- [ ] `export WANDB_API_KEY=<your-key>` 后 `echo $WANDB_API_KEY` 非空（否则 WandB 静默跳过，trainer 只 warning）
- [ ] 关键 env 已在 `import jax` **之前**设置（见 §4）。⚠️ **snapflow/freeflow 的 `train.sh` 未设任何 JAX env，piflow 只设了 `JAX_PLATFORMS`**——移植或在这些方法上 smoke test 时必须手动补齐。

### Layer 1 — Freeze / Param-merge 静态验证（无 GPU，看启动日志）

trainer 启动会打印 `print_param_summary`（`freeze_utils.py:104`）和 param merge 计数（`run_train.py:82-85`）。

- [ ] **trainable 参数量**应是几百 M（~300-500M），**不是 ~3B**。若 trainable≈3B 说明 freeze patterns 写错，VLM 被误训练 → 训练慢 + 显存爆炸。
- [ ] param merge 日志：`missing=0`，`unused` 数量合理，`skipped_new` = 新增头参数数
    - DMF：`logvar_proj`（`run_train.py:64-66`）
    - Pi-Flow：`gmm_mean_proj` / `gmm_logstd_proj` / `gmm_logweight_proj`
    - SnapFlow：`target_time_mlp`（零初始化，`models/target_time_mlp.py`）
- [ ] 无 "参数未匹配任何模式，默认冻结" warning（`freeze_utils.py:96`）——出现说明 freeze/trainable patterns 不完整
- [ ] freeze/trainable 并集覆盖全部参数（`freeze_utils.py:90-91` 检查冲突）

### Layer 2 — GPU fake-data smoke test（昂贵，信息量最大）

fake data 不碰磁盘 IO，纯测计算图 + 显存 + JIT 编译。

- [ ] 临时 config：`training_steps=5, log_every=1, batch_size=目标值, dataset_path=/nonexistent`（触发 fallback）
- [ ] 记录：JIT 编译耗时、峰值显存、loss finite、grad_norm 合理（量级 1e-1~1e1）
- [ ] 命令见 §5-B
- [ ] OOM → 查 §3-OOM；NaN → 查 §3-NaN

### Layer 3 — GPU real-data smoke test（10-50 step）

- [ ] `dataset_path` 指向真实数据集
- [ ] `steps/s` 达预期；prefetch queue 不空（数据瓶颈会拖慢 GPU）
- [ ] `norm_stats.json` 在 dataset 根目录存在（`data_loader.py:209`，否则静默用 identity norm）
- [ ] `action_std.min() > 0`（除零风险，见 §3-NaN）
- [ ] 真实 loss 与 fake-data loss 量级一致（差太多说明数据 pipeline 有 bug）

### Layer 4 — 正式训练

- [ ] `tmux new -s train 'bash scripts/train.sh'`（不要裸跑 `run_train.py`，`train.sh` 设了 PYTHONPATH + 激活 env）
- [ ] wandb run 已 init（看日志 "wandb" 不只有 warning）
- [ ] 监控 `steps/s` 和 loss 下降趋势

---

## 2. 高效 smoke test 技巧

- **一次 GPU run 内 sweep batch size**：先大后小，OOM 自动降档（见 §5-C）。比每次手动改 config 重跑省一倍编译时间。
- **metrics 保持 JAX array**，仅在 `log_every` 时 `.item()` 同步（`jax_trainer.py:371-373`）。频繁 host sync 会拖慢训练。
- **`nvidia-smi dmon -s u`** 看峰值显存，不要事后 `nvidia-smi`（峰值已释放）。
- **`XLA_FLAGS=--xla_gpu_autotune_level=0`** 关 GEMM autotune，省 80% 峰值显存，代价 +10% 时间（`run_train.py:23` 注释）。
- **frozen 值只提取一次**作为 JIT 显式参数传入（`jax_trainer.py:230-231`），避免 JIT 反复处理 3B frozen pytree。
- **`stop_gradient(prefix_tokens)`** 必须有（`jax_trainer.py:257-259`），否则 grad 流过 VLM。

---

## 3. 常见坑点（按症状索引）

### 症状：OOM（显存不够）

- [ ] `XLA_PYTHON_CLIENT_MEM_FRACTION` 默认 0.75 太小 → 设 0.90（`run_train.py:26`）。⚠️ snapflow/freeflow/piflow 的 `train.sh` **没设**，必须手动补。
- [ ] `XLA_FLAGS=--xla_gpu_autotune_level=0` 避免 autotune 分配尖峰（`run_train.py:23`）。同样 snapflow/freeflow 缺。
- [ ] 以上两个必须在 `import jax` 之前设置，设了不生效通常是顺序问题
- [ ] bs 理论占用：3B×bf16≈6GB params + JVP 翻倍激活 + optimizer state；bs=32 在 97GB 卡上理论可行，OOM 多半是 env 没设对
- [ ] freeze 漏了 → VLM 进 grad，显存翻倍

### 症状：NaN / Inf loss

- [ ] `action_std` 含 0 → `x_0=(actions-mean)/(std+1e-8)` 除零（`dmf_loss.py:66`）。打印 `action_std.min()`。
- [ ] logvar 发散 → `log_lv_loss` 用 `log(m+eps*exp(lv))` 稳定形式（`dmf_loss.py:37`），`eps=1e-3`
- [ ] 时间采样边界：用 logit-normal `sigmoid(N)` 保证 t∈(0,1)（`dmf_loss.py:20-22`），勿用 uniform（会命中 0/1 边界）
- [ ] 看 metrics：`logvar_fm_mean` 是否持续增长；`t_fm_mean` 是否合理
- [ ] `gradient_clipping: 1.0` 已开（`dmf_libero_plus.yaml:34`）

### 症状：训练慢（X s/step，如 DMF 6s/step）

- [ ] **frozen VLM 没真冻结**：`prefix_tokens` 必须 `stop_gradient`（`jax_trainer.py:257-259`）
- [ ] trainable 误含 3B VLM → 回 §1-Layer1 看 trainable 计数
- [ ] data 单线程阻塞：prefetch 线程内 `jnp.asarray` 触发异步 H2D（`jax_trainer.py:76-79`）。piflow 的 prefetch 没做这一步（`piflow/.../jax_trainer.py:60-70` 直接 put numpy batch）。
- [ ] host sync 过频：metrics 不要每步 `.item()`
- [ ] video 解码：`_decode_video_frames` 单 pass 收集（`data_loader.py:137`），不要 per-frame seek

### 症状：VLM 被误训练

- [ ] freeze pattern 的 NNX 路径：`PaliGemma/llm/layers/attn/q_einsum_1/**` ✅（action expert）vs `attn/q_einsum/w` ✅（VLM）。**坑：写成 `attn_1/q_einsum` 是错的**（`freeze_utils.py:35,49`）
- [ ] 各方法新增头必须在 trainable：DMF `logvar_proj`、Pi-Flow `gmm_*_proj`、SnapFlow `target_time_mlp`
- [ ] 用 `print_param_summary` 的 "训练参数(按组件)" 列表逐项核对

### 症状：WandB 静默跳过

- [ ] `WANDB_API_KEY` 未 export → trainer 只 warning 不报错（`jax_trainer.py:331-332`）
- [ ] 训练前确认日志出现 wandb run URL，而非 "WandB init failed"

### 症状：Checkpoint load/save 错乱

- [ ] `params/` = **EMA** 模型（eval 读这个，`jax_trainer.py:459`）
- [ ] `params_training/` = 训练模型（resume 读这个，`jax_trainer.py:502-508`）
- [ ] `opt_state/` = optimizer state
- [ ] resume 时 `ema_values` 必须从 `params/` 恢复（`jax_trainer.py:515-531`），否则 EMA 丢失
- [ ] eval 读到全 0 → EMA 未初始化，查 "Initialized EMA" 日志（`jax_trainer.py:317`）
- [ ] `detect_checkpoint_type()` 靠头名识别：`logvar_proj`→dmf，`gmm_mean_proj`→piflow，else pi05（`eval/common/policy_loader.py:64-66`）。新方法加新头后要更新此函数

### 症状：数据 pipeline 异常

- [ ] `norm_stats.json` 在 dataset 根目录（`data_loader.py:209`）
- [ ] `action_dim_raw=7` pad 到 32，pad 部分 `std=1.0`（`data_loader.py:225-226`）
- [ ] `image_mask right_wrist_0_rgb=False`（单 wrist 相机，`data_loader.py:367`）
- [ ] LeRobot v2.0（parquet 内嵌图）vs v2.1（mp4 video）自动检测（`data_loader.py:101-132`）
- [ ] **KeyError on episode_video_paths**（v2.1 视频格式）→ 视频文件缺失的 episode 未在 `frame_entries` 构建时过滤。必须在构建 `frame_entries` 前先建 `episode_video_paths` 并跳过无视频的 episode。详见 §8（Pi-Flow 案例）。DMF 已修复，piflow 已同步。snapflow/freeflow 用黑图 fallback（不崩溃但静默训练错误数据，同样需要修）。

---

## 4. 关键环境变量（必须在 `import jax` 前设）

```bash
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_MAX_SIZE=134217728
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export WANDB_API_KEY=<your-key>
```

⚠️ DMF 的 `run_train.py:19-26` 已硬编码前四个；**snapflow / freeflow / piflow 没有**，移植或在这些方法上 smoke test 时必须手动 export 或补进 `train.sh`。

---

## 5. Smoke test 命令模板

### A. GPU fake-data smoke test（验证 JIT / 显存 / NaN，无磁盘 IO）

```bash
cd /root/autodl-tmp/dmf
cp configs/train/dmf_libero_plus.yaml /tmp/smoke.yaml
sed -i 's/training_steps: 30000/training_steps: 5/' /tmp/smoke.yaml
sed -i 's/log_every: 100/log_every: 1/' /tmp/smoke.yaml
sed -i 's|dataset_path: .*|dataset_path: /nonexistent|' /tmp/smoke.yaml   # 触发 fake-data fallback
bash scripts/train.sh /tmp/smoke.yaml
```

### B. Batch size sweep（一次 GPU run，先大后小）

```bash
cd /root/autodl-tmp/dmf
for BS in 64 32 16 8; do
  sed "s/batch_size: .*/batch_size: $BS/" configs/train/dmf_libero_plus.yaml > /tmp/smoke_bs.yaml
  sed -i 's/training_steps: 30000/training_steps: 3/' /tmp/smoke_bs.yaml
  sed -i 's|dataset_path: .*|dataset_path: /nonexistent|' /tmp/smoke_bs.yaml
  echo "=== Trying batch_size=$BS ==="
  if bash scripts/train.sh /tmp/smoke_bs.yaml 2>&1 | tee /tmp/smoke_bs${BS}.log; then
    echo "batch_size=$BS OK"; break
  else
    grep -E "RESOURCE_EXHAUSTED|OOM|out of memory" /tmp/smoke_bs${BS}.log && echo "bs=$BS OOM, trying smaller" || break
  fi
done
```

### C. real-data smoke test（验证 data pipeline，50 step）

```bash
cd /root/autodl-tmp/dmf
cp configs/train/dmf_libero_plus.yaml /tmp/smoke_real.yaml
sed -i 's/training_steps: 30000/training_steps: 50/' /tmp/smoke_real.yaml
sed -i 's/log_every: 100/log_every: 10/' /tmp/smoke_real.yaml
bash scripts/train.sh /tmp/smoke_real.yaml
```

### D. 通用训练启动

```bash
cd /root/autodl-tmp/dmf    && bash scripts/train.sh                         # DMF
cd /root/autodl-tmp/piflow && bash scripts/train.sh                         # Pi-Flow
cd /root/autodl-tmp/snapflow && bash scripts/train.sh configs/train/snapflow_libero.yaml
cd /root/autodl-tmp/freeflow && bash scripts/train.sh
# tmux 后台跑：
tmux new -s train "cd /root/autodl-tmp/dmf && bash scripts/train.sh"
```

---

## 6. 移植新方法 checklist

- [ ] freeze/trainable patterns 匹配新模型 NNX 路径（用 `print_param_summary` 验证计数）
- [ ] param merge：`missing=0`，`skipped_new` = 新增头参数数
- [ ] `stop_gradient` 覆盖所有不该回传的路径（prefix tokens 至少）
- [ ] loss 数值稳定（除法加 `eps`、log 加 `eps`）
- [ ] checkpoint 三件套：`params/`(EMA) + `params_training/` + `opt_state/`
- [ ] `detect_checkpoint_type()`（`eval/common/policy_loader.py:32`）能识别新头名
- [ ] 关键 env 在 `import jax` 前设置（snapflow/freeflow/piflow 的 `train.sh` 默认缺）
- [ ] 双模型方法（Pi-Flow teacher+student）：teacher 必须完全 frozen，grad 不流过

---

## 7. KV Split Optimization — JVP 不穿过 VLM Backbone (2026-07-14)

### 背景

DMF training 速度 **6167ms/step**，GPU util 100%。瓶颈不是 data loading，而是 `jax.jvp`
穿透了全部 18 层 3B transformer。

### 为什么 JVP 会穿过 Frozen 的 VLM Backbone

这是最核心的误解。`stop_gradient(prefix_tokens)` 只能阻止**参数梯度**流向 VLM（即
`∂loss/∂(VLM weights)` = 0），但 JVP 计算的是**对输入求导**：

```
jax.jvp(dmf_model_fn, primals=(z_t, t, r), tangents=(v_t, 1, 0))
→ computes ∂u/∂z_t · v_t + ∂u/∂t
```

链式法则必须从输出 `u` 反向追溯到输入 `z_t`。这条路径是：

```
z_t → action_in_proj → suffix_tokens ─┐
                                       ├→ 18 层 self-attn (prefix + suffix 拼接)
prefix_tokens (已 stop_gradient) ──────┘→ action_out_proj → u
```

即使 `prefix_tokens` 自身不产生参数梯度，在每层 self-attention 中 **suffix query 对
prefix key 的注意力计算**仍是可微的——`∂(softmax(Q_s · K_p)) / ∂Q_s` 不为零。
因此 JVP tracer 必须穿过全部的 QKV 投影、attention softmax、FFN 等操作。

**关键数据**：prefix ~256 tokens，suffix ~10 tokens。每层 self-attention 的
matmul 是 `Q [B,266,H] × K^T [B,H,266]`，其中 96% 的 FLOP 花在 prefix token 上。

### 解决方案：双 pass 模式

正确性前提：attention mask (`pi0.py:make_attn_mask`) 保证 **prefix 不 attend suffix**，
因此 prefix hidden state 数学上与 suffix 独立。

1. **Pass 1**：prefix-only forward→ 得到每层正向 KV cache（`[depth, B, T_prefix, K, H]`）
2. **Pass 2**：仅处理 suffix tokens `[None, suffix_tokens]`，注入 prefix KV，
   只对新 suffix K,V 求导

Pi0.5 inference (`pi0.py:sample_actions`) 已经用这个模式——前向一次 prefix，复用
KV cache 做多次 Euler 步。训练只需把此模式用到 `compute_loss` 路径。

### 代码修改（3 处）

| 文件 | 改动 |
|------|------|
| `openpi/src/openpi/models/gemma.py:514-518` | `forward_with_intermediates` 逐层 slice KV：stacked `[depth, B, T, K, H]`→ 每层 `(k[i], v[i])` |
| `dmf/src/dmf_vla/models/pi05_dmf.py:_dmf_model_fn` (L134-156) | 预计算 prefix KV（`[prefix_tokens, None]`），`stop_gradient`，闭包捕获 |
| `dmf/src/dmf_vla/models/pi05_dmf.py:_dmf_forward` (L93-134) | 新增 `prefix_kv_cache` 参数；suffix-only forward `[None, suffix]` + position/mask slice |
| `dmf/src/dmf_vla/models/pi05_dmf.py:sample_actions` (L179-210) | 同上优化，prefix KV 一次预计算，所有 Euler 步复用 |

### 遇到的错误与修复

#### Error 1: Shape mismatch — positions/mask 未截断

**错误信息**：
```
TypeError: mul got incompatible shapes for broadcasting:
(32, 10, 8, 128), (32, 778, 1, 128).
```
发生在 `gemma.py:_apply_rope(q, positions)`。

**原因**：suffix-only forward `[None, suffix_tokens]` 时，Q/K/V 只有 suffix 长度
（~10 tokens），但 `positions` 和 `mask` 还是全量（prefix + suffix = ~266 tokens）。
RoPE 对 Q 施加 position embedding 时，Q `[B, 10, H, D]` 与 positions `[B, 266]` 形状不匹配。

**修复** (`pi05_dmf.py:119-124`)：
```python
p_len = prefix_tokens.shape[1]
(_, suffix_out), _ = self.PaliGemma.llm(
    [None, suffix_tokens],
    positions=positions[:, p_len:],        # 只取 suffix 位置
    mask=attn_mask[:, p_len:, :],          # Q 维度截断，KV 维度保持全量
    adarms_cond=[None, cond_stack], kv_cache=prefix_kv_cache,
)
```

注意 mask shape `[B, Q_len, KV_len]`，**Q 维度截断但 KV 维度保持全量**（suffix query
仍需看到 prefix keys）。

#### Error 2 (潜在): KV cache per-layer 未逐层 slice

**如果**直接把 `forward_with_intermediates` 的 stacked KV `[depth, B, T, K, H]`
传给每层的 `block.apply`，`Attention.__call__` 会尝试 `concat([stacked_K, new_K])`，
导致维度爆炸（第一维是 depth 而非序列长度）。

**修复** (`gemma.py:514-518`)：
```python
if kv_cache is not None:
    layer_kv_cache = (kv_cache[0][i], kv_cache[1][i])  # 逐层 slice
else:
    layer_kv_cache = None
```

### 其他潜在踩坑点

- **adarms_cond 一致性**：prefix-only pass 产生 KV cache 时 adarms_cond 必须为
  `[None, None]`（VLM 不用 adaptive norm）；suffix-only pass 中 action expert 的
  `cond_stack` 不变（`[None, cond_stack]`）
- **position continuity**：suffix positions 从 `cumsum(input_mask) - 1` 保证
  与 prefix 连接后的位置连续（不是从头编号）
- **JIT compile 时间增加**：新 graph 结构需要额外 JIT trace（约 10 min）。
  这是每次代码改动后首次运行的固定开销，正式训练 amortize 掉
- **FM branch 不需要改**：`compute_dmf_loss` 中 FM 分支 (`t_fm, t_fm`) 不涉及 JVP，
  可继续走统一前向（无 KV cache），或同样走 suffix-only 均可
- **loss 一致性验证**：正确实现时 loss 应 bitwise 一致（数学等价变换）。
  实测 step 100: loss=0.4041，与 baseline 一致

### 结果

| | Baseline | KV Split | 提升 |
|---|---|---|---|
| Step time | 6167ms | **~1350ms** | **4.5×** |
| Loss @ step 100 | 0.4041 | 0.4041 | 一致 |
| Loss @ step 10000+ | — | -0.8 ~ -0.9 | 正常下降 |
| GPU util | 100% | 100% | compute-bound 未变 |
| 预估 30k steps | 51.7h | **~11h** | **4.7×** |

**核心教训**：冻结参数 ≠ 冻结计算图。VLA 架构中 action token 与 prefix token 共享
同一个 transformer 做 full self-attention，必须通过 KV-cache split 把 prefix 完全
移出可微计算图，才能避免 JVP 冗余 FLOP。

---

## 8. Pi-Flow DataLoader KeyError — 视频缺失 episode 未过滤 (2026-07-16)

### 症状

Pi-Flow 训练在 **step 50** 首次 log 后立即崩溃：

```
INFO:jax_trainer:Step     50/30000 | loss=12.1139 | ...
Traceback (most recent call last):
  File ".../run_train.py", line 208, in main
    trainer.train(data_loader, resume_from=args.resume)
  File ".../jax_trainer.py", line 451, in train
    batch = next(prefetch)
  File ".../jax_trainer.py", line 77, in __next__
    raise item
  File ".../jax_trainer.py", line 61, in _worker
    batch = next(self._data_iter)
  File ".../data_loader.py", line 301, in create_data_loader
    front_path, wrist_path = episode_video_paths[ep_idx]
                             ~~~~~~~~~~~~~~~~~~~^^^^^^^^
KeyError: 14086
```

- `piflow_finetuned/` 目录下只有 `config.yaml`，无任何 checkpoint
- 之前的 wandb run (`run-20260715_171953`) 完整记录了崩溃

### 根因

`piflow/src/piflow_vla/training/data_loader.py` 中**三个代码块顺序错误**：

1. `frame_entries`（帧索引）构建时**只检查 parquet 文件存在性**，不检查视频文件
2. `episode_video_paths`（视频路径字典）在 `frame_entries` **之后**构建
3. `parquet_to_episode` 映射也从 `frame_entries` 构建，但时序上已无法过滤

数据集 `libero-plus-training` 的实际情况：

| 资源 | 数量 |
|------|------|
| 元数据 episodes | 14347 |
| parquet 文件 | 14345 |
| 完整视频 (front + wrist) | 14337 |
| **缺失视频的 episode** | **~10** |

Episode 14086 有 parquet 文件和 wrist 视频，但**缺少 front 视频** → 被纳入
`frame_entries`，但不在 `episode_video_paths` 中 → prefetch 线程命中该 episode
时触发 `KeyError`，异常通过 `_PrefetchIterator.__next__` 传播到主线程。

**为什么 step 50 才崩**：`_PrefetchIterator` (maxsize=2) 在后台预取数据。
前 50 步的 batch 恰好没有命中缺失视频的 episode。step 50 后 prefetch queue
耗尽，下一批命中 episode 14086 → 崩溃。崩溃时机的随机性取决于 shuffle seed。

### 修复

DMF 的 `data_loader.py` **已经修复了同样的 bug**（DMF 成功训练到 step 30000）。
修复方式是调整三个代码块的顺序：

```python
# 修复前（piflow 原始代码）:
frame_entries = [...]           # ← 只检查 parquet，不过滤视频
# ... 之后才构建 episode_video_paths
episode_video_paths = {...}     # ← 太晚了，frame_entries 已含缺失视频的 episode

# 修复后（对齐 DMF）:
episode_video_paths = {...}     # ← 先构建视频路径索引
frame_entries = [...]           # ← 构建时过滤：if ep_idx not in episode_video_paths: continue
parquet_to_episode = {...}      # ← 从已过滤的 frame_entries 构建
```

**改动细节**（`piflow/src/piflow_vla/training/data_loader.py`）：

1. 将 `episode_video_paths` 构建移到 `frame_entries` 循环之前
2. 在 `frame_entries` 循环中添加过滤：
   ```python
   if is_video_format and ep_idx not in episode_video_paths:
       continue  # episode has no video files, skip
   ```
3. 将 `parquet_to_episode` 构建移到 `frame_entries` 之后（从已过滤的条目构建）

修复后 `diff` 确认 piflow 与 DMF 的 `data_loader.py` **完全一致**。

### 验证

修复后重新启动训练（`setsid bash scripts/train.sh`）：

| Step | Loss | logstd | grad_norm | 状态 |
|------|------|--------|-----------|------|
| 50 | 12.1145 | -0.003 | 2.59 | ✅ 旧崩溃点已通过 |
| 100 | 12.4034 | -0.030 | 8.74 | ✅ |
| 150 | 11.0196 | -0.211 | 20.55 | ✅ loss 开始下降 |
| 200 | 5.8550 | -2.540 | 26.03 | ✅ loss 大幅下降 |
| 250 | 5.5176 | -4.663 | 9.09 | ✅ |
| 350 | 5.4720 | -4.792 | 6.45 | ✅ 趋于稳定 |
| 400 | 5.5166 | -5.088 | 7.08 | ✅ 稳定（warmup 未结束） |

- 数据加载器正确过滤：`Video episodes available: 14337`，`Total frames: 2236070`（修复前 2237755）
- Loss 从 12.1 降至 5.5，训练稳定无崩溃
- `logstd` 快速下降至 -5.0（GMM 组件方差收缩，模型变确定）
- throughput: 0.5 steps/s (~1340ms/step)，与 DMF 量级一致
- warmup_steps=1000，step 400 时 LR 仍在上坡，预期 loss 会进一步下降

### 教训

1. **数据加载器的过滤必须在索引构建时完成**，不能事后查找。如果某个 episode
   缺少必要文件（视频、parquet），应在 `frame_entries` 构建时就 skip，而不是
   在 yield batch 时才发现 KeyError。
2. **预取线程的异常传播是延迟的**。`_PrefetchIterator` 在后台线程中捕获异常并
   放入 queue，主线程在 `next(prefetch)` 时才 re-raise。崩溃发生在 step 50 而非
   step 1 是因为 shuffle 随机性——前 50 批恰好没命中缺失的 episode。
3. **方法间代码复用时要同步 bugfix**。Pi-Flow 的 `data_loader.py` 是从 DMF 复制
   的早期版本，DMF 后来修复了视频过滤 bug 但没有同步到 Pi-Flow。`diff` 两个方法
   的 `data_loader.py` 是快速发现此类遗漏的方法。
