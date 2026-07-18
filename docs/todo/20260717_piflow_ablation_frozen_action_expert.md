# Pi-Flow Ablation: Frozen Action-Expert — 是否需要微调 backbone？

> 日期: 2026-07-17
> 状态: 计划中（未开始）
> 优先级: 高 — 直接回答 "Pi-Flow 蒸馏是否必须微调 action-expert" 的核心问题

> ⚠️ **注意（2026-07-17 更新）**：本消融实验的 **Baseline 引用的 checkpoint
> 已失效**。`checkpoints/piflow_finetuned/step_0030000`（见 §3.1）是用 raw
> state 训练的废权重（teacher 收到未归一化 state → velocity 错误 → student
> 学到垃圾动作 → 1-NFE eval 成功率 0%）。state 归一化 bug 已修复，需先重训
> 出有效的 baseline checkpoint，本消融实验才能启动。详见
> `docs/todo/20260717_piflow_retrain_state_norm_fix.md` 和
> `docs/training-debug.md §9`。

---

## 1. 背景与动机

### 1.1 问题

Pi-Flow 的设计直觉是：在现有 π₀.₅ action-expert 上**新挂三个 GMM 线性头**
（`gmm_mean_proj` / `gmm_logstd_proj` / `gmm_logweight_proj`，共 2.64M 参数），
让解析 GMFlow rollout 在推理时零网络调用。理论上，这个"新东西"很小。

但当前实现（`piflow_libero_plus.yaml`）不仅训练 GMM 头，还**微调整个 action-expert
transformer**（`*_1` 层，424.78M 参数），加上 `time_mlp` 和 `action_in_proj`，
总可训练参数达 432.7M，和 DMF（430.1M）几乎一样。checkpoint 也是 11G。

问题：**能否冻结 action-expert，只训练 GMM 头（2.64M）？** 如果可以，训练时间和
checkpoint 都会大幅缩小，Pi-Flow 就真正成为一个"挂在小 head 上的轻量方法"。

### 1.2 为什么直觉上应该可以

- π₀.₅ 的 flow matching 训练本身就建模了多峰分布：不同 noise → 不同轨迹 → 不同 mode。
- action-expert 已经学会了从 (image, language, state) 中提取与动作相关的特征。
- GMM 头只需要从这些已有特征中"读出"多峰信息，不需要重新学习感知。

### 1.3 为什么实际上可能不行

核心矛盾在于 **teacher 的 action-expert 在 t=1 处主动塌缩了多峰信息**：

1. **t=1 处的表示塌缩**：π₀.₅ 使用 flow matching，x₁ = 纯噪声。在 t=1 处，
   teacher 的训练目标 = E[x₀ | x₁] ≈ **数据集均值**（无条件期望，因为 x₁ 不携带
   样本信息）。teacher 的 MSE loss 不奖励在 t=1 处保留多峰信息——梯度下降找到了一个
   **主动丢弃**多峰信息的解。所以 t=1 处的 action-expert hidden state 编码的是
   "数据集均值"，而非 K 个 mode。

2. **Pi-Flow 1-NFE 恰好在 t=1 预测 GMM**：`sample_actions` 在 `t_src=1.0` 处做
   唯一一次 GMM 预测（`pi05_piflow.py:254`），然后解析 rollout 8 步到 t=0。
   如果 t=1 处的 hidden state 没有 multi-modal 信息，线性 GMM 头无法凭空榨出
   K 个 mode。

3. **Pooling 是新操作**：原 π₀.₅ 用 `action_out_proj` 对 [B,H,width]→[B,H,D]
   **每步独立**预测；Pi-Flow 先 `mean(action_hidden, axis=1)` pool 成 [B,width]
   再喂 GMM 头（`pi05_piflow.py:175-176`）。原 action-expert 的特征没有为
   pooled 表示优化过。

4. **输出维度膨胀 8 倍**：teacher 输出 [H,D]=320 个数；Pi-Flow 输出
   K·H·D + K + K = 2576 个数，从同一个 2048 维 pooled 向量线性投影。信息密度
   差 8 倍，需要更丰富的特征表示。

### 1.4 多-NFE 可能缓解

对于 nfe=4 训练，segments 为 [1.0→0.75], [0.75→0.5], [0.5→0.25], [0.25→0.0]。
后三段在 t_src < 1.0 处预测 GMM，此时 x_t 携带样本信息，teacher 的 hidden state
可能保留了多峰结构。但第一段仍在 t=1.0，且 errors 会沿 segment 链传播。

---

## 2. 假设

| 编号 | 假设 | 验证方式 |
|------|------|----------|
| H1 | 冻结 action-expert 后，1-NFE 性能显著下降 | Variant B1 vs Baseline |
| H2 | 性能下降的主因是 t=1 处的表示塌缩 | 诊断指标：t=1 vs t=0.5 的 teacher-student vel 差 |
| H3 | 多-NFE 训练能部分缓解（后几段特征更丰富） | Variant B4 vs B1 |
| H4 | 即使冻结 backbone，GMM 头也能学会"有用但不足"的多峰结构 | GMM 参数统计（权重分布、means 展开度） |

---

## 3. 实验设计

### 3.1 实验矩阵

| Variant | 描述 | 可训练参数 | 训练 NFE | 训练时间(估) | 状态 |
|---------|------|-----------|---------|-------------|------|
| **Baseline** | 当前 config：action-expert `*_1` + proj + GMM | 432.7M (100%) | 1 | ~11h | ✅ 已有 step_0030000 |
| **B1** | GMM-only，1-NFE | 2.64M (0.61%) | 1 | ~7-8h* | 待训练 |
| **B4** | GMM-only，4-NFE | 2.64M (0.61%) | 4 | ~7-8h* | 待训练 |
| **C1** | GMM + time_mlp + action_in_proj，1-NFE | 4.77M (1.10%) | 1 | ~7-8h* | 待训练 |
| **C4** | GMM + time_mlp + action_in_proj，4-NFE | 4.77M (1.10%) | 4 | ~7-8h* | 待训练 |

> *训练时间估计：forward pass（student + teacher）不变，backward 只通过 2.6M 而非
> 425M 参数，预计节省 20-30%。JIT 编译时间不变。如果 OOM 风险降低可以增大 batch_size
> 则可能更快。

### 3.2 Variant 定义

所有 variant 共享以下不变量：
- Base checkpoint: `checkpoints/pi05_libero`
- Dataset: `datasets/libero-plus-training`
- training_steps: 30000
- batch_size: 16
- learning_rate: 1e-4
- ema_decay: 0.9999
- save_every: 5000
- num_components: 8, inner_substeps: 8, teacher_query_points: 4

**Variant B（GMM-only）** — 只训练新增的 GMM 线性头：

```yaml
# configs/train/piflow_ablation_B1_gmmonly_1nfe.yaml
trainable:
  - "gmm_mean_proj/**"
  - "gmm_logstd_proj/**"
  - "gmm_logweight_proj/**"
```

**Variant C（GMM + projections）** — 允许学生重新学习时间/动作投影，但不触碰
transformer：

```yaml
# configs/train/piflow_ablation_C1_gmmproj_1nfe.yaml
trainable:
  - "action_in_proj/**"
  - "time_mlp_in/**"
  - "time_mlp_out/**"
  - "gmm_mean_proj/**"
  - "gmm_logstd_proj/**"
  - "gmm_logweight_proj/**"
```

> **注意**：`freeze_utils.py:77-79` 的默认规则是 "not trainable and not frozen →
> default frozen"，所以从 `trainable` 列表中移除 `*_1` 模式后，action-expert 会
> 自动被冻结，无需额外修改 freeze 列表。无需改代码，只改 config。

### 3.3 对照实验

- **Baseline 1-NFE**: `checkpoints/piflow_finetuned/step_0030000`（已有）
- **Baseline 4-NFE**: 需要训练（改 config 的 `nfe: 4`，其余不变）。如果 compute
  紧张，可跳过——B4 vs B1 的对比已经能回答 H3。

---

## 4. 评估方案

### 4.1 主指标：LIBERO-Plus 成功率

所有 variant 训练完成后，统一评估：

```bash
# Quick 模式（快速反馈，~1h/variant）
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type piflow --nfe 1 --mode quick

# Normal 模式（正式结果，~5-10h/variant）
cd /root/autodl-tmp && python -m eval.libero_plus.main --model-type piflow --nfe 1 --mode normal
```

4-NFE 训练的 variant 同时评估 nfe=1 和 nfe=4：

```bash
python -m eval.libero_plus.main --model-type piflow --nfe 4 --mode quick
```

> **Eval 加载说明**：`detect_checkpoint_type()` 通过 `gmm_mean_proj` 键自动识别为
> piflow，无需额外配置。Variant B/C 的 checkpoint 结构与 Baseline 相同（都包含
> 全部参数，只是训练时冻结了部分），eval 代码无需修改。

### 4.2 诊断指标（训练过程）

从 WandB / training logs 收集：

| 指标 | WandB key | 含义 |
|------|-----------|------|
| velocity imitation loss | `loss_total` | 学生 rollout velocity 与 teacher velocity 的 MSE |
| student velocity norm | `student_vel_norm` | 学生预测速度的 L2 范数（接近 0 = 退化） |
| teacher velocity norm | `teacher_vel_norm` | teacher 速度的 L2 范数（参考基准） |
| vel diff norm | `vel_diff_norm` | 师生速度差（= loss_total × batch） |
| GMM means norm | `means_norm` | GMM 均值的 L2 范数（接近 0 = 退化成 no-op） |
| log_stds mean | `log_stds_mean` | 平均 log-std（很负 = 过度自信，很大 = 退化） |

**关键对比**：
- Baseline 的 `loss_total` 收敛曲线 vs Variant B/C 的收敛曲线
- 如果 B/C 的 loss plateau 在比 Baseline 高的水平 → frozen 特征不足以拟合 teacher
- 如果 B/C 的 loss 和 Baseline 一样收敛但 eval 更差 → loss 和下游性能不单调相关

### 4.3 GMM 参数分析（checkpoint 后）

对每个 checkpoint 提取 GMM 参数统计：

```python
# 在一批样本上 forward GMM，收集输出
# 关注：
# 1. log_weights 分布：是否所有 K=8 个分量都有非零权重？
#    还是退化为单峰（一个分量占 99%）？
# 2. means 展开度：K 个分量均值之间的 L2 距离
#    （大 = multi-modal，小 = 塌缩为单峰）
# 3. log_stds 分布：是否随 t_src 变化（官方参数化应该如此）
```

如果 Variant B 的 GMM 退化为单峰（一个 weight=1，其余≈0），说明 frozen 特征
确实无法支持 multi-modal 预测，GMM 头只能学一个 mode。

### 4.4 Teacher-Student 速度差 vs t（可选深度分析）

如果需要直接验证 H2（t=1 塌缩假设），可以在不同 t 值计算 teacher-student 速度差：

```python
# 对 t ∈ {1.0, 0.75, 0.5, 0.25}：
#   1. 采样 x_t（从真实数据前向插值）
#   2. teacher velocity: v_teacher(x_t, t)
#   3. student velocity: GMFlow rollout 的瞬时速度 at (x_t, t)
#   4. 记录 MSE(v_student, v_teacher) as function of t
```

预测：Variant B 的速度差在 t=1 处最大，随 t 减小而减小。
Baseline 的速度差应在所有 t 处均匀较低。

---

## 5. 预期结果与解读

### 5.1 预测

| Variant | 预期 success rate | 预期 loss | 预期 GMM 行为 |
|---------|------------------|-----------|--------------|
| Baseline 1-NFE | 参考（当前 baseline） | 正常收敛 | 多峰，weights 分散 |
| B1 (GMM-only 1NFE) | **显著下降**（H1） | plateau 较高 | 退化为单峰或 2 峰 |
| B4 (GMM-only 4NFE) | 比 B1 好，比 Baseline 差（H3） | 第一段高，后几段低 | 后几段更多峰 |
| C1 (GMM+proj 1NFE) | 与 B1 相近 | 与 B1 相近 | 与 B1 相近 |
| C4 (GMM+proj 4NFE) | 与 B4 相近 | 与 B4 相近 | 与 B4 相近 |

### 5.2 解读规则

- **如果 B1 ≈ Baseline**：frozen action-expert 的特征足够，当前微调 backbone 是
  浪费。可以改为只训 GMM 头，大幅降低训练成本。推翻 H1。
  
- **如果 B1 << Baseline 且 B4 > B1**：确认了 t=1 表示塌缩是主因（H1+H2+H3）。
  action-expert 微调是必要的，且必要的原因是 t=1 处的特征。
  
- **如果 B1 << Baseline 且 B4 ≈ B1**：表示塌缩不是 t-specific 的，而是全局的
  表示不匹配（pooling、输出维度膨胀等）。action-expert 微调是必要的，但原因
  不是 t=1 塌缩，而是整体特征不适合 GMM 预测。推翻 H2/H3。
  
- **如果 C1 > B1（C 的 projections 有帮助）**：time/action 投影需要重学，
  说明 teacher 的投影是为单速度预测优化的。但 transformer 本身仍需微调。

---

## 6. 实施步骤

### Phase 1: Config 准备（~30min）

1. 创建 4 个 ablation config：
   - `piflow/configs/train/piflow_ablation_B1_gmmonly_1nfe.yaml`
   - `piflow/configs/train/piflow_ablation_B4_gmmonly_4nfe.yaml`
   - `piflow/configs/train/piflow_ablation_C1_gmmproj_1nfe.yaml`
   - `piflow/configs/train/piflow_ablation_C4_gmmproj_4nfe.yaml`

2. 每个 config 基于 `piflow_libero_plus.yaml` 修改：
   - `trainable:` 列表按 §3.2 定义
   - `nfe:` 按矩阵设置
   - `checkpoint_dir:` 指向独立目录（避免覆盖 baseline）
   - `wandb.run_name:` 加 ablation 标识
   - 其余超参不变

3. Config diff 示例（B1）：
   ```diff
   - checkpoint_dir: /root/autodl-tmp/checkpoints/piflow_finetuned
   + checkpoint_dir: /root/autodl-tmp/checkpoints/piflow_ablation_B1_gmmonly_1nfe
   - wandb.run_name: piflow_libero_plus_1nfe
   + wandb.run_name: piflow_ablation_B1_gmmonly_1nfe
   - wandb.tags: [...]
   + wandb.tags: ["libero-plus", "1nfe", "piflow", "gmflow", "jax", "ablation", "frozen-backbone"]
     trainable:
   -   - "PaliGemma/llm/layers/attn/q_einsum_1/**"
   -   - "PaliGemma/llm/layers/attn/kv_einsum_1/**"
   -   - "PaliGemma/llm/layers/attn/attn_vec_einsum_1/**"
   -   - "PaliGemma/llm/layers/mlp_1/**"
   -   - "PaliGemma/llm/layers/pre_attention_norm_1/**"
   -   - "PaliGemma/llm/layers/pre_ffw_norm_1/**"
   -   - "PaliGemma/llm/final_norm_1/**"
   -   - "action_in_proj/**"
   -   - "time_mlp_in/**"
   -   - "time_mlp_out/**"
       - "gmm_mean_proj/**"
       - "gmm_logstd_proj/**"
       - "gmm_logweight_proj/**"
   ```

### Phase 2: Smoke test（~30min）

4. 对 B1 config 跑 smoke test（1 step）：
   ```bash
   cd /root/autodl-tmp/piflow && bash scripts/train.sh configs/train/piflow_ablation_B1_gmmonly_1nfe.yaml
   ```
   确认：
   - 参数 summary 显示 trainable=2.64M, frozen=3.36B-2.64M
   - forward/backward 不报错
   - loss 有合理数值（初始应为 ≈ teacher_vel_norm，因为 GMM 头 zero-init → no-op）
   - 不 OOM

### Phase 3: 训练（~7-8h × 4 = 28-32h）

5. 按优先级训练（可串行或并行，取决于 GPU 空闲情况）：

   | 优先级 | Variant | 理由 |
   |--------|---------|------|
   | P0 | B1 | 最关键的对比（H1） |
   | P1 | B4 | 验证多-NFE 缓解（H3） |
   | P2 | C1 | 验证 projections 是否有帮助 |
   | P3 | C4 | 最后补全矩阵 |

   ```bash
   # 每个 variant
   cd /root/autodl-tmp/piflow && bash scripts/train.sh configs/train/piflow_ablation_<variant>.yaml
   ```

6. 训练中监控 WandB：
   - `loss_total` 收敛趋势
   - `student_vel_norm` 不为 0（否则学生退化）
   - `means_norm` 在增长（GMM 在学东西）

### Phase 4: 评估（~2h × 4 = 8h quick, ~20h normal）

7. Quick eval（先跑，快速反馈）：
   ```bash
   for variant in B1 B4 C1 C4; do
     ckpt="checkpoints/piflow_ablation_${variant}/step_0030000"
     python -m eval.libero_plus.main --model-type piflow --nfe 1 --mode quick \
       --checkpoint "$ckpt"
   done
   ```

8. Normal eval（根据 quick 结果决定是否值得跑）：
   ```bash
   # 只对有希望的 variant 跑 normal
   python -m eval.libero_plus.main --model-type piflow --nfe 1 --mode normal
   ```

### Phase 5: 分析与报告（~2h）

9. 汇总结果到 `docs/experiments/` 下（按实验模板）：
   - `202607xx_piflow_ablation_frozen_backbone.md`
   - 包含：成功率对比表、loss 曲线截图、GMM 参数统计、结论

---

## 7. 风险与注意事项

1. **冻结 action-expert 后，`params_training/` 和 `params/` 仍然保存全部 3.36B
   参数**（因为 `jax_trainer.py:384` 重建 full state）。checkpoint 大小不会显著
   缩小（~11G → ~10G，只有 `opt_state/` 从 1.2G 降到 ~10MB）。如需真正缩小
   checkpoint，需要修改 trainer 只保存 trainable params——但这会影响 eval 加载，
   不建议在 ablation 中做。

2. **学习率可能需要调整**：2.64M 参数用 1e-4 的 lr（为 432M 设计）可能偏小。
   第一轮用相同 lr 以保证公平对比；如果 B1 完全不收敛，可尝试 lr=3e-4 或 1e-3
   作为补充实验。但注意这会破坏与 Baseline 的公平性。

3. **GMM 头 zero-init**：三个 GMM 头都用 zero-init（`pi05_piflow.py:76-93`），
   初始输出为 no-op（返回输入 noise）。冻结 backbone 后，GMM 头的梯度只来自
   `pooled` 特征的线性投影。如果 `pooled` 的 Jacobian 对 GMM 参数方向的贡献
   很小，可能出现梯度消失。需要在 smoke test 中检查初始梯度范数。

4. **EMA 对 frozen params 无影响**：EMA 只追踪 trainable params
   （`jax_trainer.py:413-414`），frozen params 直接从 `_student_frozen_dict` 复制。
   所以 EMA 模型 = frozen backbone + EMA(trained heads)，行为正确。

5. **Teacher 仍是同一个 π₀.₅**：所有 variant 的 teacher 都是 `pi05_libero`，
   完全 frozen。teacher 的速度场在所有 variant 中完全相同，确保对比公平。

---

## 8. 成功标准

- **核心结论**（无论哪个方向）：明确回答 "Pi-Flow 是否必须微调 action-expert"
- **论文价值**：这是一个有价值的 ablation，可直接写入论文的 ablation study 部分
- **如 H1 成立**：证明了 distillation 中 backbone fine-tuning 的必要性，
  且根因是 t=1 处的表示塌缩——这是一个 non-trivial 的发现
- **如 H1 不成立**：发现了 Pi-Flow 可以大幅轻量化（432M → 2.6M trainable，
  训练时间降低 ~30%），对实际部署有直接价值

---

## 9. 后续（如果结果有趣）

- 如果 B4 > B1（多-NFE 缓解）：探索 "只微调前几层 action-expert" 或
  "只在 t=1 附近微调" 的混合策略
- 如果 C1 > B1（projections 有帮助）：探索更深/更宽的 GMM 头（如 2 层 MLP
  而非单层 Linear），看是否能弥补 frozen 特征的不足
- 如果 B1 ≈ Baseline：推广到 CALVIN 域验证，并考虑修改 trainer 只保存
  trainable params 以真正缩小 checkpoint
