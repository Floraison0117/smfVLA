# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SnapFlow implements the paper https://arxiv.org/abs/2604.05656 - a self-distillation method for compressing multi-step denoising of flow-matching VLAs into a single forward pass (1-NFE).

Built on top of [smfVLA](../smfVLA/) and [openpi](../openpi/) from Physical Intelligence.

## Architecture

### Core Algorithm

SnapFlow loss: `L = α·L_FM + (1-α)·λ·L_shortcut`

Where:
- **L_FM**: Standard flow matching at random time t
- **L_shortcut**: Consistency loss with 2-step Euler shortcut target
  - v_1 = F_θ(x_1, 1, 1) — velocity at t=1
  - x_0.5 = x_1 - 0.5 · v_1 — midpoint via Euler
  - v_0.5 = F_θ(x_0.5, 0.5, 0.5) — velocity at t=0.5
  - v_target = 0.5 · (v_1 + v_0.5) — 2-step average velocity
  - L_shortcut = ||F_θ(x_1, 0, 1) - v_target||²

### Key Components

**`Pi05SnapFlow`** (`models/pi05_snapflow.py`):
- Extends `Pi05SMF` from smfVLA
- Adds `target_time_mlp`: Zero-initialized 2-layer MLP
- Modified `embed_suffix()` to inject target-time embedding
- `compute_snapflow_loss()`: SnapFlow-specific loss

**`TargetTimeMLP`** (`models/target_time_mlp.py`):
- Zero-initialized: weights and biases all zeros
- Ensures network starts at teacher behavior (step 0)
- Encodes target time s (s=t for FM, s=0 for consistency)

**`compute_snapflow_loss`** (`training/snapflow_loss.py`):
- Three forward passes per training step:
  1. FM: F_θ(x_t, t, t) at random t
  2. Shortcut v1: F_θ(x_1, 1, 1)
  3. Shortcut v0.5: F_θ(x_0.5, 0.5, 0.5)
- Only F_θ(x_1, 0, 1) receives gradients from consistency loss

## Commands

### Training

```bash
# Default config
bash scripts/train.sh configs/train/snapflow_libero.yaml

# Resume from checkpoint
bash scripts/train.sh configs/train/snapflow_libero.yaml --resume checkpoints/finetuned/snapflow/step_10000
```

### Evaluation

```bash
# Quick test (libero_spatial, 5 episodes/task)
python scripts/eval_direct.py --preset quick --nfe 1

# Full eval (all suites, 50 episodes/task)
python scripts/eval_direct.py --preset full --nfe 1

# Custom
python scripts/eval_direct.py --nfe 1 --task-suite libero_spatial --num-episodes 10
```

### Environment

All scripts use conda env `openpi_server` (`/root/miniconda3/envs/openpi_server`).

PYTHONPATH is set by scripts to:
```
$PROJECT_ROOT/src:$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src
```

## Key Paths

| Item | Path |
|------|------|
| Base checkpoint | `checkpoints/base/pi05_libero/` (symlink to smfVLA) |
| Dataset | `data/libero/` (symlink to smfVLA) |
| Norm stats | `data/libero/norm_stats.json` |
| Training config | `configs/train/snapflow_libero.yaml` |

## Hyperparameters

From paper (Table 10):

| Parameter | Value | Notes |
|-----------|-------|-------|
| α (FM/Consistency ratio) | 0.5 | Balanced mix |
| λ (Consistency weight) | 0.1 | Balances gradient magnitudes |
| Learning rate | 2.5×10⁻⁵ | 1/10 of π0.5 training |
| Warmup steps | 500 | Linear warmup |
| Total steps | 30,000 | ~12h on single A800 |
| Batch size | 4 | Matches paper |
| Precision | bfloat16 | Memory efficiency |

## Code Style

```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```

## Comparison: SMF vs SnapFlow

| Aspect | SMF | SnapFlow |
|--------|-----|----------|
| Target | Average velocity u(z_t, r, t) | 1-step consistency via 2-step shortcut |
| Loss | Self-consistency + FM | α·FM + (1-α)·λ·shortcut |
| Time embedding | Concat E(t), E(r) + time_proj | Time + target-time MLP φ_s |
| Forward passes | 3 (u_2, u_1, pred_sc) | 3 (FM, v1, v0.5) |
| Sampling | (r, t) pairs | Random t + fixed {1, 0.5} |

## Related Memory

- [[smf-training-method]] - SMF training method and performance optimizations
- [[eval-usage]] - Evaluation infrastructure from smfVLA
- [[libero-datasets]] - LIBERO dataset details
