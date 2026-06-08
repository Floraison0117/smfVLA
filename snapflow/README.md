# SnapFlow: 1-NFE Action Generation for Flow-Matching VLAs

Implementation of SnapFlow ([arXiv:2604.05656](https://arxiv.org/abs/2604.05656)) on LIBERO benchmark.

## Overview

SnapFlow is a self-distillation method that compresses multi-step denoising (typically 10 ODE steps) into a single forward pass (1-NFE) for flow-matching Vision-Language-Action (VLA) models.

### Key Features

- **Two-Step Euler Shortcut**: Targets are 2-step Euler shortcut velocities
- **Target-Time Embedding**: Zero-initialized MLP for distinguishing FM vs consistency samples
- **Progressive FM/Consistency Mixing**: α·L_FM + (1-α)·λ·L_shortcut

### Target Results (π0.5 on LIBERO)

- **Success Rate**: 98.75% (vs 97.75% baseline 10-step)
- **Speedup**: 9.6× denoising, 3.3× end-to-end
- **Latency**: 274ms → 83ms

## Project Structure

```
snapflow/
├── configs/train/          # Training configurations
├── src/snapflow/
│   ├── models/            # Model implementations
│   ├── training/          # Training infrastructure
│   └── eval/              # Evaluation utilities
├── scripts/               # Training/eval scripts
├── data/                  # LIBERO dataset (symlink)
├── checkpoints/           # Model checkpoints
└── logs/                  # Training logs
```

## Quick Start

### Training

```bash
# Activate conda environment
conda activate openpi_server

# Run training
bash scripts/train.sh configs/train/snapflow_libero.yaml
```

### Evaluation

```bash
# Quick eval (libero_spatial, 5 episodes/task)
python scripts/eval_direct.py --preset quick --nfe 1

# Full eval (all suites, 50 episodes/task)
python scripts/eval_direct.py --preset full --nfe 1
```

## Implementation Status

- [ ] Phase 1: Project Setup
- [ ] Phase 2: Core Components
- [ ] Phase 3: Training Infrastructure
- [ ] Phase 4: Evaluation Infrastructure
- [ ] Phase 5: Training & Experiments
- [ ] Phase 6: Analysis & Documentation

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for details.

## Dependencies

- JAX + Flax (for JAX-based training)
- optax (for optimization)
- WandB (for logging)
- LeRobot (for data loading)

Uses conda environment `openpi_server` from smfVLA project.

## References

1. [SnapFlow Paper](https://arxiv.org/abs/2604.05656)
2. [π0.5 Paper](https://arxiv.org/abs/2504.16054)
3. [smfVLA Project](../smfVLA/) - Reference implementation
4. [openpi Project](../openpi/) - Base VLA framework

## License

This implementation follows the licenses of the underlying projects (openpi, smfVLA).
