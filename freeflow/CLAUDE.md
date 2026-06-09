# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FreeFlow implements data-free distillation of π₀.₅ VLA model for 1-NFE training on LIBERO-Plus robustness benchmark. Adapted from [FreeFlow](https://arxiv.org/abs/2511.19428) (originally for image generation) to Vision-Language-Action models.

## Commands

### Training

```bash
# Default config (freeflow_base)
bash scripts/train.sh configs/train/freeflow_base_libero.yaml

# Custom config
bash scripts/train.sh configs/train/freeflow_full_libero.yaml

# Resume from checkpoint
bash scripts/train.sh configs/train/freeflow_base_libero.yaml \
    --resume checkpoints/finetuned/freeflow/step_10000
```

### Evaluation

```bash
# LIBERO-Plus robustness evaluation
python scripts/eval_libero_plus.py --preset quick --nfe 1

# Full LIBERO-Plus evaluation
python scripts/eval_libero_plus.py --preset full --nfe 1

# Via unified eval framework
cd ../eval/scripts
python run_eval.py --dataset libero-plus --mode quick --nfe 1 --model-type freeflow
```

### Environment

All scripts use conda env `openpi_server` (`/root/miniconda3/envs/openpi_server`).

PYTHONPATH is set by scripts to:
```
$PROJECT_ROOT/src:$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src:$OPENPI_DIR/third_party/libero
```

### Code style

```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```

## Architecture

### Core Algorithm: FreeFlow for VLA

**Key idea**: Data-free distillation - student learns teacher's multi-step integration path by sampling from prior distribution only.

**Teacher**: π₀.₅ base model (frozen, NFE=10)
- Provides oracle action trajectories
- Multi-step Euler integration

**Student**: FreeFlow model (trainable, NFE=1)
- Learns to predict 1-step actions
- Parameters initialized from teacher

**Loss Function**:
```
L = L_path + λ·L_correction

L_path = ||z_0^S - z_0^T||²
  where z_0^S = z_1 - S_φ(z_1, 0→1)
        z_0^T = Euler(T_θ, z_1, steps=10)

L_correction = ||z_0^S(t) - z_0^T||²
  at intermediate time t for error correction
```

### Key Components

**`Pi05FreeFlow`** (`models/pi05_freeflow.py`):
- Extends `openpi.models.pi0.Pi0`
- Same architecture as teacher initially
- Trained with FreeFlow distillation loss
- Supports 1-NFE inference

**`TeacherWrapper`** (`models/teacher_wrapper.py`):
- Wraps frozen π₀.₅ teacher
- Provides multi-step Euler integration
- No gradient computation
- Cached intermediate states

**`compute_freeflow_loss`** (`training/freeflow_loss.py`):
- Samples z_1 from prior (data-free!)
- Gets teacher path via Euler integration
- Computes path loss + correction loss
- Only student receives gradients

### Training Variants

| Config | Description |
|--------|-------------|
| `freeflow_base_libero.yaml` | Basic FreeFlow |
| `freeflow_no_correction.yaml` | Without error correction |
| `freeflow_full.yaml` | All variants combined |

### Hyperparameters

From FreeFlow paper + adaptation:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Teacher NFE | 10 | π₀.₅ default |
| λ (correction) | 0.1 | Error correction weight |
| Learning rate | 2.5×10⁻⁵ | 1/10 of π₀.₅ |
| Warmup steps | 500 | Linear warmup |
| Total steps | 30,000 | ~12h on A800 |
| Batch size | 4 | Matches paper |

### Data Pipeline

Uses LeRobot v2.0 format (same as smfVLA/snapflow):
- Location: `data/libero/` (symlink to `../datasets/libero`)
- Format: Apache Parquet files
- Normalization: `norm_stats.json`

**Note**: FreeFlow is data-free for distillation, but still needs observation conditioning. The dataset is used only for observation tokens, not for action labels.

### Parameter Freezing

Same patterns as smfVLA/snapflow:
- **Frozen**: VLM backbone (SigLIP, token embeddings, VLM attention/MLP without `_1` suffix)
- **Trainable**: Action expert layers (with `_1` suffix), action_in_proj, action_out_proj, time_mlp

### Comparison: SMF vs SnapFlow vs FreeFlow

| Aspect | SMF | SnapFlow | FreeFlow |
|--------|-----|----------|----------|
| Data required | Yes | Yes | **No** |
| Teacher | None | None | π₀.₅ (frozen) |
| Loss target | Self-consistency | 2-step shortcut | Teacher's 10-step path |
| Forward passes | 2-3 | 3 | 2-4 (student + teacher) |
| Key innovation | Dual time (r,t) | Target-time MLP | **Data-free distillation** |

### Key Paths

| Item | Path |
|------|------|
| Base checkpoint | `checkpoints/base/pi05_libero/` (symlink to smfVLA) |
| Dataset | `data/libero/` (symlink to `../datasets/libero`) |
| Norm stats | `data/libero/norm_stats.json` |
| Training config | `configs/train/freeflow_base_libero.yaml` |

### LIBERO-Plus Evaluation

**Robustness dimensions** (7 perturbations):
1. Visual noise (Gaussian, salt-pepper)
2. Observation missing (wrist camera dropout)
3. Action delay (temporal lag)
4. Action noise (Gaussian perturbation)
5. Goal perturbation (object position shift)
6. Scene distractor (extra objects)
7. Lighting change (brightness/contrast)

**Evaluation modes**:
- `quick`: Fast test (libero_spatial, 5 ep)
- `preset`: Standard LIBERO (4 suites, 50 ep)
- `fullset`: Complete LIBERO-Plus (5 suites, 50 ep)

## Related Documentation

- `FREEFLOW_IMPLEMENTATION_PLAN.md` - Complete implementation plan
- `../smfVLA/CLAUDE.md` - SMF implementation details
- `../snapflow/CLAUDE.md` - SnapFlow implementation details
- `../CLAUDE.md` - Project-level guidance

## References

- [FreeFlow Paper](https://arxiv.org/abs/2511.19428) - Flow Map Distillation Without Data
- [FreeFlow GitHub](https://github.com/ShangyuanTong/FreeFlow) - Original implementation
- [π₀.₅](https://www.physicalintelligence.company/) - Base VLA model
