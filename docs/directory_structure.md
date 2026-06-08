# 目录结构说明 (Directory Structure)

本文档描述 `/root/autodl-tmp` 目录重组后的结构。

## 重组日期
2026-06-08

## 重组目标
1. **数据集统一**：所有LIBERO数据集集中存放在 `datasets/`
2. **Checkpoint共享**：所有模型checkpoint集中存放在 `checkpoints/`
3. **评估代码整合**：统一评估框架放在 `eval/`，支持SMF和SnapFlow
4. **训练代码分离**：`smfVLA/` 和 `snapflow/` 仅保留训练相关代码
5. **日志统一管理**：训练和评估日志统一存放在 `logs/`

## 目录结构

```
autodl-tmp/
├── .autodl/                    # ✅ 保持不变
├── .claude/                    # ✅ 保持不变
├── conda_envs/                 # ✅ 保持不变
│   ├── libero_eval/
│   └── libero_client/
│
├── datasets/                   # 🆕 共享数据集
│   ├── libero/                 # LIBERO原始数据集
│   ├── libero-plus/            # LIBERO-Plus鲁棒性基准
│   └── libero-plus-training/  # LIBERO-Plus训练数据
│
├── checkpoints/                # 🆕 共享checkpoint存放
│   ├── smf_base/               # SMF base checkpoints (pi05_libero)
│   ├── smf_finetuned/          # SMF微调checkpoints (smf_base, smf_curr_v2)
│   ├── snapflow_base/          # SnapFlow base (软链接到smf_base)
│   └── snapflow_finetuned/     # SnapFlow微调checkpoints
│
├── eval/                       # 🆕 统一评估框架
│   ├── configs/                # 评估配置
│   │   ├── libero_spatial.yaml
│   │   ├── libero_object.yaml
│   │   ├── libero_goal.yaml
│   │   ├── libero_10.yaml
│   │   └── libero_90.yaml
│   ├── scripts/                # 评估脚本
│   │   ├── eval_utils.py        # 共享工具函数
│   │   ├── eval_direct.py       # 通用LIBERO评估
│   │   ├── eval_libero_plus.py  # LIBERO-Plus评估
│   │   └── eval_libero_plus_real.py
│   ├── models/                 # 模型加载适配器
│   │   ├── __init__.py
│   │   ├── smf_adapter.py      # SMF模型加载
│   │   └── snapflow_adapter.py # SnapFlow模型加载
│   └── results/                # 评估结果
│       ├── smf/
│       └── snapflow/
│
├── logs/                       # 🆕 统一日志存放
│   ├── train/
│   │   ├── smf/
│   │   └── snapflow/
│   └── eval/
│       ├── smf/
│       └── snapflow/
│
├── docs/                       # 🆕 统一文档
│   ├── training.md
│   ├── evaluation.md
│   ├── directory_structure.md
│   └── *.md (其他文档)
│
├── openpi/                     # 第三方库（保持不变）
│
├── smfVLA/                     # 🔧 精简为仅训练
│   ├── configs/
│   │   └── train/              # 仅训练配置
│   ├── scripts/
│   │   ├── run_train.py
│   │   ├── serve_policy.sh
│   │   └── train*.sh
│   ├── src/
│   │   └── smf_vla/
│   ├── third_party/
│   │   └── openpi/ -> ../openpi
│   ├── data -> ../datasets     # 软链接
│   └── checkpoints -> ../checkpoints/smf_*  # 软链接
│
└── snapflow/                   # 🔧 精简为仅训练
    ├── configs/
    │   └── train/              # 仅训练配置
    ├── scripts/
    │   ├── run_train.py
    │   └── train.sh
    ├── src/
    │   └── snapflow/
    ├── third_party/
    │   └── openpi/ -> ../openpi
    ├── data -> ../datasets     # 软链接
    └── checkpoints -> ../checkpoints  # 软链接
```

## 软链接说明

| 项目 | 原路径 | 新路径 |
|------|--------|--------|
| smfVLA/data | smfVLA/data/libero | -> ../datasets |
| smfVLA/checkpoints/base | smfVLA/checkpoints/base | -> ../checkpoints/smf_base |
| smfVLA/checkpoints/finetuned | smfVLA/checkpoints/finetuned | -> ../checkpoints/smf_finetuned |
| snapflow/data | snapflow/data/libero | -> ../datasets |
| snapflow/checkpoints/base | snapflow/checkpoints/base | -> ../checkpoints/smf_base |
| snapflow/checkpoints/finetuned | snapflow/checkpoints/finetuned | -> ../checkpoints/snapflow_finetuned |

## 评估使用方式

### SMF模型评估
```bash
# Quick test
cd /root/autodl-tmp/eval/scripts
python eval_direct.py --preset quick --nfe 1 --model-type smf \
    --checkpoint ../../checkpoints/smf_finetuned/smf_base/step_5000

# Full eval
python eval_direct.py --preset full --nfe 1 --model-type smf \
    --checkpoint ../../checkpoints/smf_finetuned/smf_curr_v2/step_12000
```

### SnapFlow模型评估
```bash
# Quick test
cd /root/autodl-tmp/eval/scripts
python eval_direct.py --preset quick --nfe 1 --model-type snapflow \
    --checkpoint ../../checkpoints/snapflow_finetuned/step_30000

# LIBERO-Plus评估
python eval_libero_plus.py --preset quick --nfe 1 \
    --checkpoint ../../checkpoints/snapflow_finetuned/step_30000
```

## 训练使用方式

### SMF训练
```bash
cd /root/autodl-tmp/smfVLA
bash scripts/train.sh configs/train/smf_base_libero.yaml
```

### SnapFlow训练
```bash
cd /root/autodl-tmp/snapflow
bash scripts/train.sh configs/train/snapflow_libero.yaml
```

## 迁移说明

如果需要在其他环境复现此结构：

1. 创建共享目录
2. 移动数据集到 `datasets/`
3. 移动checkpoints到 `checkpoints/`
4. 移动评估脚本到 `eval/`
5. 创建软链接
6. 更新脚本中的路径引用

## 注意事项

1. **eval/ 目录是独立的**：eval脚本现在位于 `/root/autodl-tmp/eval/`，不再属于smfVLA或snapflow
2. **PROJECT_ROOT已更新**：eval脚本中的PROJECT_ROOT现在指向`/root/autodl-tmp`
3. **checkpoint路径变化**：训练脚本中的checkpoint路径需要使用相对路径或绝对路径
4. **软链接依赖**：smfVLA和snapflow依赖于软链接访问datasets和checkpoints
