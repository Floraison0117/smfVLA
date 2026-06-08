# 评估脚本使用指南

本文档描述 `/root/autodl-tmp/eval/` 目录下的评估脚本使用方法。

## 目录结构

```
eval/
├── configs/              # 评估配置文件
│   ├── libero_spatial.yaml
│   ├── libero_object.yaml
│   ├── libero_goal.yaml
│   ├── libero_10.yaml
│   └── libero_90.yaml
├── scripts/              # 评估脚本
│   ├── eval_utils.py         # 共享工具函数
│   ├── eval_direct.py        # LIBERO 标准评估
│   ├── eval_libero_plus.py  # LIBERO-Plus 鲁棒性评估
│   ├── eval_libero_plus_real.py  # LIBERO-Plus 真实扰动评估
│   └── run_eval.py           # 统一评估入口 ⭐
├── models/               # 模型适配器
│   ├── __init__.py
│   ├── smf_adapter.py
│   └── snapflow_adapter.py
└── results/              # 评估结果
    ├── smf/
    └── snapflow/
```

## 快速开始

### 方法 1: 使用统一评估入口 (推荐)

```bash
cd /root/autodl-tmp/eval/scripts

# LIBERO 标准评估，preset 模式，1-NFE
python run_eval.py --dataset libero --mode preset --nfe 1 --model-type smf

# LIBERO 完整评估，fullset 模式，1-NFE
python run_eval.py --dataset libero --mode fullset --nfe 1 --model-type snapflow

# LIBERO-Plus 鲁棒性评估，quick 模式
python run_eval.py --dataset libero-plus --mode quick --nfe 1 --model-type smf
```

### 方法 2: 直接使用评估脚本

```bash
# LIBERO 评估
python eval_direct.py --preset preset --nfe 1 --model-type smf

# LIBERO-Plus 评估
python eval_libero_plus.py --preset quick --nfe 1
```

## 数据集说明

### LIBERO (标准基准)

标准 LIBERO 评估，每个任务运行多个 episodes。

- **路径**: `datasets/libero/`
- **特点**: 标准评估，多 episode per task
- **任务集**:
  - `libero_spatial` - 空间推理任务 (100 tasks)
  - `libero_object` - 物体操作任务 (100 tasks)
  - `libero_goal` - 目标导向任务 (100 tasks)
  - `libero_10` - 10 任务组合 (100 tasks)
  - `libero_90` - 90 任务组合 (100 tasks)

### LIBERO-Plus (鲁棒性基准)

LIBERO-Plus 鲁棒性评估，包含 7 种扰动维度。

- **路径**: `datasets/libero-plus/LIBERO-plus/`
- **特点**: 鲁棒性评估，1 episode per task（每个扰动是独立任务）
- **扰动维度**:
  - 背景纹理 (background_textures)
  - 机器人初始状态 (robot_init_states)
  - 物体布局 (objects_layout)
  - 相机视角 (camera_viewpoints)
  - 语言指令 (language_instructions)
  - 光照条件 (light_conditions)
  - 物体复制 (background_textures_copy)

## 评估模式

### LIBERO 模式

| 模式 | 任务集 | Episodes/Task | 说明 |
|------|--------|----------------|------|
| `quick` | libero_spatial | 5 | 快速测试 |
| `preset` | 4 suites (spatial, object, goal, 10) | 50 | 标准评估 |
| `fullset` | 5 suites (all) | 50 | 完整评估 |

### LIBERO-Plus 模式

| 模式 | 任务集 | 最大任务数 | 说明 |
|------|--------|-----------|------|
| `quick` | libero_spatial | 50 | 快速测试 |
| `medium` | libero_spatial | 100 | 中等评估 |
| `full` | 4 suites | 全部 | 完整评估 |
| `full90` | libero_90 | 全部 | libero_90 评估 |

## NFE (Number of Function Evaluations) 说明

| NFE | 说明 | 适用模型 |
|-----|------|----------|
| 1 | 单步推理，最快 | SMF, SnapFlow |
| 2 | 两步推理 | 所有模型 |
| 4 | 四步推理 | 所有模型 |
| 10 | 十步推理 (原始 Pi0) | 所有模型 |

**性能预期**:
- 1-NFE: 推理最快，性能略低
- 10-NFE: 推理较慢，性能最高
- 2/4-NFE: 速度与性能的平衡

## 模型类型

| 模型 | checkpoint 路径 | 说明 |
|------|-----------------|------|
| SMF | `checkpoints/smf_base/` | SplitMeanFlow 模型 |
| SnapFlow | `checkpoints/snapflow_finetuned/` | SnapFlow 模型 |

## 详细使用示例

### LIBERO 评估

```bash
# 快速测试 (5 episodes, libero_spatial)
python eval_direct.py --preset quick --nfe 1 --model-type smf

# 标准评估 (50 episodes, 4 suites)
python eval_direct.py --preset preset --nfe 1 --model-type smf \
    --checkpoint ../../checkpoints/smf_finetuned/smf_base/step_5000

# 完整评估 (50 episodes, 5 suites)
python eval_direct.py --preset fullset --nfe 1 --model-type snapflow \
    --checkpoint ../../checkpoints/snapflow_finetuned/step_30000

# 自定义评估
python eval_direct.py \
    --task-suite libero_spatial \
    --num-episodes 10 \
    --nfe 2 \
    --model-type smf

# 测试不同 NFE 值
for nfe in 1 2 4 10; do
    python eval_direct.py --preset quick --nfe $nfe --model-type smf
done
```

### LIBERO-Plus 评估

```bash
# 快速测试 (50 tasks)
python eval_libero_plus.py --preset quick --nfe 1

# 标准评估 (100 tasks)
python eval_libero_plus.py --preset medium --nfe 1

# 完整评估 (4 suites)
python eval_libero_plus.py --preset full --nfe 1

# 指定 suite 和任务数
python eval_libero_plus.py \
    --suite libero_spatial \
    --max-tasks 50 \
    --nfe 1

# 测试不同 NFE 值
for nfe in 1 2 4 10; do
    python eval_libero_plus.py --preset quick --nfe $nfe
done
```

### 使用统一入口

```bash
# LIBERO 评估
python run_eval.py --dataset libero --mode preset --nfe 1 --model-type smf
python run_eval.py --dataset libero --mode fullset --nfe 2 --model-type snapflow

# LIBERO-Plus 评估
python run_eval.py --dataset libero-plus --mode quick --nfe 4 --model-type smf
python run_eval.py --dataset libero-plus --mode full --nfe 1 --model-type snapflow

# 自定义参数
python run_eval.py \
    --dataset libero \
    --task-suite libero_spatial \
    --num-episodes 20 \
    --nfe 2 \
    --model-type smf \
    --checkpoint ../../checkpoints/smf_finetuned/smf_curr_v2/step_12000
```

## 结果文件

评估结果保存在 `eval/results/` 目录下：

```
eval/results/
├── smf/
│   ├── 20240608_120000_libero_spatial_1nfe_85.5pct.json
│   └── 20240608_130000_preset_all_suites_1nfe.json
└── snapflow/
    ├── 20240608_140000_libero_spatial_1nfe_88.2pct.json
    └── ...
```

结果文件包含：
- 总体成功率
- 每个任务的成功率
- 推理延迟统计
- Episode 详细信息

## 常见问题

### Q: 如何选择 NFE 值？
A:
- **1-NFE**: 最快推理，适用于实时应用，使用 SMF/SnapFlow 模型
- **2/4-NFE**: 速度与性能的平衡
- **10-NFE**: 最高性能，使用原始 Pi0 模型

### Q: LIBERO 和 LIBERO-Plus 有什么区别？
A:
- **LIBERO**: 标准基准，每个任务运行多个 episodes，评估标准性能
- **LIBERO-Plus**: 鲁棒性基准，包含扰动任务，评估模型对扰动的鲁棒性

### Q: 如何检查评估结果？
A: 查看生成的 JSON 文件，或查看脚本输出的汇总信息。

### Q: 评估需要多长时间？
A: 取决于模式、NFE 和硬件：
- quick 模式: ~5-10 分钟
- preset/fullset 模式: ~1-3 小时
- NFE 越高，每个 episode 越慢

## 环境要求

- Python 3.8+
- JAX (GPU 版本)
- LIBERO 环境 (`libero_eval` conda 环境)
- 已安装的模型 (smfVLA 或 snapflow)

## 相关文档

- [目录结构说明](directory_structure.md)
- [SMF 训练方法](../smfVLA/docs/20260602_154947_smf_base_training_plan.md)
- [LIBERO 数据集](../memory/libero-datasets.md)
