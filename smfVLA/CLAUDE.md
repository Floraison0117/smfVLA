# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

smfVLA fine-tunes the pi0.5 VLA model's action head using **SplitMeanFlow (SMF)** to achieve high-quality action generation in 1 function evaluation (NFE) instead of the usual 10. Built on top of [openpi](https://github.com/Physical-Intelligence/openpi) from Physical Intelligence.

## Commands

### Training
```bash
# Default config (smf_base)
bash scripts/train.sh

# Custom config
bash scripts/train.sh configs/train/smf_decte_curr_libero.yaml

# Resume from checkpoint
bash scripts/train.sh configs/train/smf_base_libero.yaml
# Then in run_train.py: --resume checkpoints/finetuned/smf_base/step_5000
```

The `--resume` flag restores model params, optimizer state, and training step from a saved checkpoint.

### Evaluation

Evaluation is handled by the unified entry point in `eval/scripts/run_eval.py`
(not in this directory). See the root `AGENTS.md` Evaluation section for usage.

### Smoke test (no GPU / no real data)
```python
from smf_vla.training.data_loader import create_fake_data_loader
loader = create_fake_data_loader(batch_size=4, num_batches=2)
batch = next(loader)
```

### Environment
All scripts use conda env `openpi_server` (`/root/miniconda3/envs/openpi_server`).

PYTHONPATH is set by scripts to:
```
$PROJECT_ROOT/src:$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src
```

### Code style (from pyproject.toml)
```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```

## Architecture

### Core algorithm
SplitMeanFlow trains the model to predict **average velocity** u(z_t, r, t) from time r to t, instead of instantaneous velocity. The loss has two branches:
- **SMF branch** (r < t): self-consistency loss — `u(z_t, r, t) = (1-λ)·u(z_s, r, s) + λ·u(z_t, s, t)`
- **FM branch** (r = t): standard flow matching loss to prevent degeneration (Bernoulli p=0.3)

1-NFE inference: `z_0 = z_1 - u(z_1, 0, 1, c)`

### Two code paths
1. **JAX/NNX (production)**: `models/pi05_smf.py`, `training/jax_trainer.py`, `training/smf_loss.py`, `training/freeze_utils.py`, `training/data_loader.py`
2. **PyTorch (legacy/prototyping)**: `action_head.py`, `flow_matching.py`, `training/trainer.py`

Only the JAX path is used for actual training and evaluation.

### Key classes

**`Pi05SMF`** (`models/pi05_smf.py`) — extends `openpi.models.pi0.Pi0`:
- Adds `time_proj`: `Linear(2*width, width)` initialized to `[I, 0]` so it starts equivalent to original pi0.5
- `embed_suffix_smf(obs, noisy_actions, t, r)`: dual time inputs → `time_proj(concat([E(t), E(r)]))`
- `embed_suffix_decte(obs, noisy_actions, t, r, encoder_depth)`: Decoupled Time Embedding — encoder layers use E(t), decoder layers use E(r)
- `compute_loss()`: full SMF loss supporting all variants (base, curr, decte, anchor, bpl, full)
- `sample_actions()`: 1-NFE (direct) or multi-step (Euler)

**`SMFTrainer`** (`training/jax_trainer.py`): JAX JIT-compiled training loop with:
- Selective parameter updates (only trainable params get gradients applied)
- Optax AdamW + linear warmup + cosine decay
- Checkpoint save/load with optimizer state (supports `--resume`)
- Freeze/trainable patterns read from YAML config (falls back to hardcoded defaults)
- WandB logging

**`compute_full_smf_loss`** (`training/smf_loss.py`): Unified loss supporting all variants:
- `sample_r_t()` / `sample_r_t_curriculum()`: time pair sampling (uniform or curriculum)
- `compute_anchor_loss()`: teacher Euler integration supervision
- `compute_bpl_loss()`: behavioral perceptual loss using teacher hidden states

### Training variants (7 configs)

| Config | Method | Key Features |
|--------|--------|-------------|
| `smf_base_libero.yaml` | smf_base | Concat time embedding, uniform sampling |
| `smf_curr_libero.yaml` | smf_curr | Curriculum Time Sampling (cosine schedule) |
| `smf_decte_libero.yaml` | smf_decte | Decoupled Time Embedding (encoder_depth=6) |
| `smf_decte_curr_libero.yaml` | smf_decte_curr | DecTE + Curriculum |
| `smf_decte_curr_anchor_libero.yaml` | smf_decte_curr_anchor | + Anchor Loss (teacher NFE=2) |
| `smf_decte_curr_bpl_libero.yaml` | smf_decte_curr_bpl | + Behavioral Perceptual Loss (layers 12,16) |
| `smf_full_libero.yaml` | smf_full | DecTE + Curriculum + Anchor + BPL |

### Data pipeline
Data is in **LeRobot v2.0 format** (Apache Parquet files at `data/libero/`).

Image processing in `training/data_loader.py`:
1. Parquet binary → PIL Image → numpy uint8 (H, W, C)
2. Rotate 180° (`[::-1, ::-1]`, matching eval preprocessing in `openpi/examples/libero/main.py`)
3. Resize 256×256 → 224×224 (PIL LANCZOS)
4. Pack into `Observation.from_dict` compatible format: `image` dict with keys `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` + `image_mask` dict

Output batch: `{observation: {image, image_mask, state}, actions, action_mean, action_std, prompt}`

Raw action dim is 7 (from LIBERO); padded to 32 (model's `action_dim`) with zeros. Norm stats (`action_mean`, `action_std`) are also padded accordingly.

### Parameter freezing
`freeze_utils.py` uses glob patterns on NNX path tuples. Patterns are read from the YAML config (`freeze`/`trainable` keys), falling back to hardcoded defaults:
- **Frozen**: VLM backbone (SigLIP, token embeddings, VLM attention/MLP/norm without `_1` suffix)
- **Trainable**: Action expert layers with `_1` suffix, action_in_proj, action_out_proj, time_mlp_in, time_mlp_out, time_proj

### Anchor loss detail
The anchor loss teacher uses the correct `r` parameter (not `r=0`) for Euler integration. The alpha schedule is: linear warmup (0→alpha_max over `anchor_warmup_steps`), then linear cooldown (alpha_max→0 over `anchor_cooldown_steps`).

### openpi dependency
`third_party/openpi` is a symlink to `/root/autodl-tmp/openpi`. Key openpi components used:
- `openpi.models.pi0.Pi0` — base model class
- `openpi.models.model.restore_params` — checkpoint loading
- `openpi.models.model.Observation` — observation struct with `from_dict()`
- `openpi.shared.image_tools.resize_with_pad` — JAX-based image resize (used at eval time)

### Conda environment
| Env | Purpose |
|-----|---------|
| `openpi_server` | Training + eval (JAX/CUDA) |

## Key paths

| Item | Path |
|------|------|
| Base checkpoint | `checkpoints/pi05_libero/` (params/ + assets/) |
| Dataset | `datasets/libero/` (data/, meta/, norm_stats.json) — symlinked as `data` |
| Architecture doc | `docs/20260602_154947_smf_base_training_plan.md` |
