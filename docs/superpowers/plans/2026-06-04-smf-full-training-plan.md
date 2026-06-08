# SMF-Full 四技巧训练实现计划

## 概述

从 pi05-libero checkpoint 出发，使用 Curriculum + Anchor + BPL + Dynamic Scaling 四技巧，在 LIBERO 和 LIBERO-Plus 上训练 SMF 模型。

---

## Step 1：修改 run_train.py 加载 Teacher 模型

**文件**: `scripts/run_train.py`

**任务**:
1. 在模型加载阶段，创建第二个 `Pi05SMF` 实例作为 teacher
2. 加载同一个 `checkpoints/base/pi05_libero` checkpoint
3. 冻结 teacher 所有参数（不训练，不保留梯度）
4. 构造 `teacher_fn` 闭包传入 loss 函数
5. 构造 `teacher_model` 对象实现 `extract_hidden_states` 方法

**关键代码位置**:
- 模型加载: `run_train.py` 约 line 106-122
- Loss 函数调用: `jax_trainer.py` 约 line 269-292（loss_fn 闭包）
- Anchor loss: `smf_loss.py` line 270-330（`compute_anchor_loss`）
- BPL loss: `smf_loss.py` line 333-372（`compute_bpl_loss`）

**验证**: 运行训练时检查日志中 `loss_anchor` 和 `loss_bpl` 是否为非零值

---

## Step 2：修改 smf_full_libero.yaml 启用 Dynamic Scaling

**文件**: `configs/train/smf_full_libero.yaml`

**任务**:
1. 添加 `smf_loss_scale: dynamic` 配置项
2. 确认 `_update_smf_scale` 方法（jax_trainer.py line 334-383）会被正确调用

**验证**: 训练日志中 `smf_scale` 值应在 [1.0, 200.0] 范围内变化

---

## Step 3：运行 LIBERO 训练（Phase 1）

**命令**:
```bash
cd /root/autodl-tmp/smfVLA
python scripts/run_train.py configs/train/smf_full_libero.yaml
```

**预期时间**: ~3 小时（基于 SMF-Base 的 2.9 小时）

**监控指标**:
- `loss_total`, `loss_fm`, `loss_smf`, `loss_anchor`, `loss_bpl`
- `smf_scale`（dynamic scaling 的值）
- `delta_mean`（curriculum 的平均时间间隔）
- `grad_norm`

**检查点**: 每 3000 步保存，共 5 个中间 checkpoint

---

## Step 4：评估 LIBERO 训练结果

**命令**:
```bash
# 1-NFE 评估
python scripts/eval_direct.py --nfe 1 --use-smf --suite quick

# 10-NFE 评估
python scripts/eval_direct.py --nfe 10 --use-smf --suite quick
```

**对比基线**:
- SMF-Base 1-NFE: ~72-80%（libero_spatial）
- pi05-libero 10-NFE: 96% / 98% / 96% / 90%

---

## Step 5：修改数据加载器支持 v2.1 格式

**文件**: `src/smf_vla/training/data_loader.py`

**任务**:
1. 在 `create_data_loader()` 中添加格式自动检测逻辑
2. 检查 DataFrame 列名是否存在 `observation.images.front`（v2.1 标志）
3. 如果是 v2.1，映射列名：
   - `observation.images.front` → `image`
   - `observation.images.wrist` → `wrist_image`
   - `action` → `actions`
   - `observation.state` → `state`
4. norm_stats 从数据集目录的 `norm_stats.json` 加载（已有逻辑，无需修改）

**关键代码位置**:
- 数据加载: `data_loader.py` line 273-274（列名硬编码处）
- norm_stats 加载: `data_loader.py` line 191

**验证**: 用 libero-plus-training 路径创建 data_loader，检查 batch shape 和数据类型正确

---

## Step 6：新建 LIBERO-Plus 训练配置

**文件**: `configs/train/smf_full_libero_plus.yaml`（新建）

**内容**: 复制 `smf_full_libero.yaml`，修改：
```yaml
dataset_path: data/libero-plus-training
```

其余参数保持不变。

---

## Step 7：运行 LIBERO-Plus 训练（Phase 2）

**命令**:
```bash
cd /root/autodl-tmp/smfVLA
python scripts/run_train.py configs/train/smf_full_libero_plus.yaml
```

**预期时间**: 可能更长（数据量是 libero 的 8.5x，但 batch 大小相同，steps 相同）

---

## Step 8：评估 LIBERO-Plus 训练结果

**命令**:
```bash
# 1-NFE 评估
python scripts/eval_libero_plus.py --nfe 1 --preset quick

# 10-NFE 评估
python scripts/eval_libero_plus.py --nfe 10 --preset quick
```

**对比基线**:
- pi05-libero 1-NFE on LIBERO-Plus: 0%（全面失败）
- pi05-libero 10-NFE on LIBERO-Plus: 92-98%（除 spatial 外）

---

## 执行顺序

```
Step 1 (teacher 加载) → Step 2 (dynamic scaling 配置) → Step 3 (LIBERO 训练)
                                                          ↓
                                                     Step 4 (评估)
                                                          ↓
                                              Step 5 (数据加载器) → Step 6 (新配置)
                                                                      ↓
                                                              Step 7 (LIBERO-Plus 训练)
                                                                      ↓
                                                              Step 8 (评估)
```

---

## 依赖关系

- Step 1-2: 代码修改，无依赖
- Step 3: 依赖 Step 1-2
- Step 4: 依赖 Step 3
- Step 5: 独立于 Step 1-4，可并行
- Step 6: 依赖 Step 5
- Step 7: 依赖 Step 6
- Step 8: 依赖 Step 7

---

## 风险缓解

| 风险 | 缓解措施 |
|------|---------|
| Teacher 质量不够 | 换用 SMF-Base checkpoint 作为 teacher |
| 显存不足 | 减小 batch_size 或用 gradient accumulation |
| libero-plus-training 视频缺失 | 检查 chunk 011-014 是否必须，必要时跳过 |
| Action space 泛化差 | 增加 total_steps 或降低 lr |
