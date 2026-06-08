# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements two 1-NFE (One Function Evaluation) action generation methods for Vision-Language-Action (VLA) models:

1. **SMF (SplitMeanFlow)** - Trains the model to predict average velocity instead of instantaneous velocity
2. **SnapFlow** - Compresses multi-step denoising into single forward pass via self-distillation

Both methods are built on top of [openpi](https://github.com/Physical-Intelligence/openpi) from Physical Intelligence, fine-tuning the π₀.₅ model's action head for efficient 1-NFE inference on LIBERO benchmark.

**Project structure (reorganized 2026-06-08):**
- `smfVLA/` - SMF training code only
- `snapflow/` - SnapFlow training code only  
- `eval/` - Unified evaluation framework for both models
- `datasets/` - Shared LIBERO datasets
- `checkpoints/` - Shared model checkpoints
- `logs/` - Unified training/evaluation logs
- `docs/` - Project documentation

## Conda Environments

| Environment | Purpose |
|-------------|---------|
| `openpi_server` | Training (JAX/CUDA), serving policies |
| `libero_client` | Evaluation client |
| `libero_eval` | LIBERO simulation environment |

All scripts default to `openpi_server` for training.

## Training Commands

### SMF Training

```bash
cd /root/autodl-tmp/smfVLA

# Default config (smf_base)
bash scripts/train.sh

# Custom config
bash scripts/train.sh configs/train/smf_decte_curr_libero.yaml

# Resume from checkpoint
bash scripts/train.sh configs/train/smf_base_libero.yaml
# Then add: --resume ../checkpoints/smf_finetuned/smf_base/step_5000
```

**Training variants (7 configs):**
- `smf_base` - Basic SMF with uniform time sampling
- `smf_curr` - Curriculum Time Sampling
- `smf_decte` - Decoupled Time Embedding
- `smf_decte_curr` - DecTE + Curriculum
- `smf_decte_curr_anchor` - + Anchor Loss
- `smf_decte_curr_bpl` - + Behavioral Perceptual Loss
- `smf_full` - All variants combined

### SnapFlow Training

```bash
cd /root/autodl-tmp/snapflow

# Default config
bash scripts/train.sh configs/train/snapflow_libero.yaml

# Resume from checkpoint
bash scripts/train.sh configs/train/snapflow_libero.yaml --resume ../checkpoints/snapflow_finetuned/step_10000
```

**Key hyperparameters (from paper):**
- α (FM/Consistency ratio) = 0.5
- λ (Consistency weight) = 0.1
- Learning rate = 2.5×10⁻⁵ (1/10 of π0.5)
- Batch size = 4
- Training steps = 30,000 (~12h on A800)

## Evaluation Commands

### Unified Evaluation Entry (Recommended)

```bash
cd /root/autodl-tmp/eval/scripts

# LIBERO standard evaluation
python run_eval.py --dataset libero --mode preset --nfe 1 --model-type smf

# LIBERO-Plus robustness evaluation
python run_eval.py --dataset libero-plus --mode quick --nfe 1 --model-type snapflow

# Test different NFE values
for nfe in 1 2 4 10; do
    python run_eval.py --dataset libero --mode quick --nfe $nfe --model-type smf
done
```

### Direct Script Usage

```bash
# LIBERO evaluation
python eval_direct.py --preset preset --nfe 1 --model-type smf \
    --checkpoint ../../checkpoints/smf_finetuned/smf_base/step_5000

# LIBERO-Plus evaluation  
python eval_libero_plus.py --preset quick --nfe 1
```

**Evaluation modes:**
- `quick` - Fast test (libero_spatial, 5 ep)
- `preset` - Standard eval (4 suites, 50 ep)
- `fullset` - Complete eval (5 suites, 50 ep)

**NFE options:** 1, 2, 4, 10
- 1-NFE: Fastest inference (SMF/SnapFlow)
- 10-NFE: Original Pi0 performance

## Core Architecture

### SMF (SplitMeanFlow)

**Key idea:** Train model to predict average velocity u(z_t, r, t) from time r to t, not instantaneous velocity.

**Loss branches:**
1. **SMF branch** (r < t): Self-consistency loss - u(z_t, r, t) = (1-λ)·u(z_s, r, s) + λ·u(z_t, s, t)
2. **FM branch** (r = t): Standard flow matching to prevent degeneration (p=0.3)

**1-NFE inference:** z_0 = z_1 - u(z_1, 0, 1, c)

**Key files:**
- `smfVLA/src/smf_vla/models/pi05_smf.py` - Pi05SMF model
- `smfVLA/src/smf_vla/training/smf_loss.py` - SMF loss computation
- `smfVLA/src/smf_vla/training/jax_trainer.py` - JAX training loop

### SnapFlow

**Key idea:** Compress multi-step denoising via 2-step Euler shortcut target.

**Loss:** L = α·L_FM + (1-α)·λ·L_shortcut
- **L_FM**: Standard FM at random time t
- **L_shortcut**: Consistency with 2-step Euler average velocity

**Target-Time MLP:** Zero-initialized 2-layer MLP encoding target time s (s=t for FM, s=0 for consistency)

**Key files:**
- `snapflow/src/snapflow/models/pi05_snapflow.py` - Pi05SnapFlow model
- `snapflow/src/snapflow/training/snapflow_loss.py` - SnapFlow loss
- `snapflow/src/snapflow/models/target_time_mlp.py` - Target-time embedding

### Comparison: SMF vs SnapFlow

| Aspect | SMF | SnapFlow |
|--------|-----|----------|
| Target | Average velocity u(z_t, r, t) | 1-step consistency via 2-step shortcut |
| Time input | Dual time (r, t) | Triple time (r, t, s) with target-time MLP |
| Loss | Self-consistency + FM | α·FM + (1-α)·λ·shortcut |
| Forward passes | 2-3 | 3 (fixed) |
| Sampling | (r, t) pairs | Random t + fixed {1, 0.5} |

## Data Pipeline

**Format:** LeRobot v2.0 (Apache Parquet files)

**Location:** `datasets/libero/` or `datasets/libero-plus/`

**Image processing** (in `smfVLA/src/smf_vla/training/data_loader.py`):
1. Parquet binary → PIL Image → numpy uint8
2. Rotate 180° (`[::-1, ::-1]`)
3. Resize 256×256 → 224×224 (PIL LANCZOS)
4. Pack into LeRobot Observation format

**Output batch:** `{observation: {image, image_mask, state}, actions, action_mean, action_std, prompt}`

**Raw action dim:** 7 (from LIBERO), padded to 32 (model's action_dim)

## Parameter Freezing

Only action expert parameters are trained; VLM backbone is frozen.

**Frozen:** SigLIP, token embeddings, VLM attention/MLP (without `_1` suffix)

**Trainable:** Action expert layers (with `_1` suffix), action_in_proj, action_out_proj, time_mlp_in, time_mlp_out, time_proj, target_time_mlp

Patterns defined in YAML config (`freeze`/`trainable` keys), fallback to defaults in `freeze_utils.py`.

## Key Paths

| Item | Path |
|------|------|
| Datasets | `datasets/libero/`, `datasets/libero-plus/` |
| SMF checkpoints | `checkpoints/smf_base/`, `checkpoints/smf_finetuned/` |
| SnapFlow checkpoints | `checkpoints/snapflow_finetuned/` |
| Evaluation scripts | `eval/scripts/` |
| Training configs (SMF) | `smfVLA/configs/train/` |
| Training configs (SnapFlow) | `snapflow/configs/train/` |

## Evaluation Framework

**Unified entry point:** `eval/scripts/run_eval.py`

**Supports:**
- LIBERO standard benchmark (5 suites)
- LIBERO-Plus robustness benchmark (7 perturbation dimensions)
- Multiple NFE modes (1, 2, 4, 10)
- Both SMF and SnapFlow models

**Results:** Saved as JSON in `eval/results/{smf,snapflow}/`

**LIBERO vs LIBERO-Plus:**
- **LIBERO**: Standard evaluation, multiple episodes per task
- **LIBERO-Plus**: Robustness evaluation with perturbations, 1 episode per task

## Python Path Setup

Training scripts set PYTHONPATH to:
```
$PROJECT_ROOT/src:$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src:$OPENPI_DIR/third_party/libero
```

Where `OPENPI_DIR` is `openpi/` (symlink in smfVLA/snapflow third_party).

## Code Style

```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```

## Related Documentation

- `docs/directory_structure.md` - Detailed directory structure after reorganization
- `docs/evaluation.md` - Complete evaluation guide
- `smfVLA/CLAUDE.md` - SMF-specific implementation details
- `snapflow/CLAUDE.md` - SnapFlow-specific implementation details
