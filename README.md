# smfVLA: Few-NFE Denoising Vision-Language-Action Model

基于 [openpi](https://github.com/Physical-Intelligence/openpi) 的少 NFE 去噪 VLA 模型。通过重新训练 action-head，实现在更少的去噪步数（NFE）下生成高质量动作。

## 项目结构

```
smfVLA/
├── configs/          # 训练与评估配置
├── scripts/          # 核心脚本
├── src/smf_vla/      # 自定义源码
├── data/             # 数据集（本地，不入 git）
├── checkpoints/      # 模型权重（本地，不入 git）
├── assets/           # 静态资源（norm_stats 等）
├── logs/             # 训练与评估日志（本地，不入 git）
├── results/          # 评估结果汇总
├── notebooks/        # 实验分析 notebook
└── third_party/      # 第三方依赖
    └── openpi -> openpi 仓库
```

## 环境配置

### 1. 安装 openpi 依赖

```bash
cd third_party/openpi
uv sync
```

### 2. 安装 smfVLA 依赖

```bash
cd /root/autodl-tmp/smfVLA
pip install -e .
```

## 使用方法

### 训练

```bash
# 1-NFE 微调
bash scripts/train.sh configs/train/pi05_libero_1nfe.yaml

# 5-NFE 微调
bash scripts/train.sh configs/train/pi05_libero_5nfe.yaml
```

### 评估

评估由统一的入口 `eval/scripts/run_eval.py` 处理，见根目录 `AGENTS.md`。

### 查看结果

```bash
# 评估结果
ls eval/results/smf/
```

## 实验记录

| 模型 | NFE | LIBERO Spatial | 推理延迟 |
|------|-----|----------------|----------|
| pi0.5 base | 1 | 100.00% | 57ms |
| pi0.5 base | 10 | 96.00% | 66ms |
| smfVLA | 1 | TBD | TBD |

## 核心改动

1. **action-head 重新训练**：保持 VLM backbone 不变，仅微调 action-head
2. **少 NFE 采样器**：实现 1-NFE 采样（consistency distillation / progressive distillation）
3. **Flow matching 优化**：针对少 NFE 场景优化 ODE 求解器
