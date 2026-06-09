# FreeFlow for LIBERO-Plus

Data-free distillation of π₀.₅ VLA model for 1-NFE robotics training on LIBERO-Plus robustness benchmark.

## Overview

FreeFlow ([arXiv:2511.19428](https://arxiv.org/abs/2511.19428)) is adapted from image generation to Vision-Language-Action models. The key innovation is **data-free distillation** - the student learns from the teacher's multi-step integration path without requiring training data.

### Key Features

- **Data-free**: Sample from prior distribution, no dataset required
- **Teacher-student**: Frozen π₀.₅ teacher (NFE=10) → 1-NFE student
- **Error correction**: Actively corrects compounding errors
- **LIBERO-Plus**: Robustness evaluation with 7 perturbation dimensions

## Installation

```bash
conda activate openpi_server
cd /root/autodl-tmp/freeflow
pip install -e .
```

## Training

```bash
# Default config
bash scripts/train.sh configs/train/freeflow_base_libero.yaml

# Resume from checkpoint
bash scripts/train.sh configs/train/freeflow_base_libero.yaml \
    --resume checkpoints/finetuned/freeflow/step_10000
```

## Evaluation

```bash
# LIBERO-Plus robustness evaluation
python scripts/eval_libero_plus.py --preset quick --nfe 1

# Via unified eval framework
cd ../eval/scripts
python run_eval.py --dataset libero-plus --mode quick --nfe 1 --model-type freeflow
```

## Architecture

```
freeflow/
├── src/freeflow/
│   ├── models/           # Pi05FreeFlow, TeacherWrapper
│   └── training/         # Loss, trainer, data loader
├── configs/train/        # Training configs
├── scripts/              # Training/evaluation scripts
└── checkpoints/          # Model checkpoints
```

## Comparison

| Method | Data | Teacher | NFE | Innovation |
|--------|------|---------|-----|------------|
| SMF | Required | None | 1 | Self-consistency |
| SnapFlow | Required | None | 1 | Self-distillation |
| **FreeFlow** | **None** | **π₀.₅** | **1** | **Data-free** |

## References

- [FreeFlow Paper](https://arxiv.org/abs/2511.19428)
- [FreeFlow GitHub](https://github.com/ShangyuanTong/FreeFlow)
