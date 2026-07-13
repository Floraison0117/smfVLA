# DMF (Decoupled MeanFlow) for π₀.₅ VLA

JAX implementation of DMF for 1-NFE training on LIBERO-Plus.

## Overview

DMF (Decoupled MeanFlow) converts pretrained flow models into flow maps by decoupling the transformer into encoder and decoder:
- **Encoder** (first 2/3 of layers): conditioned on current time `t`
- **Decoder** (remaining 1/3 of layers): conditioned on target time `r`

**Paper**: https://arxiv.org/abs/2510.24474

## Usage

### Training

```bash
cd /root/autodl-tmp/dmf
conda activate openpi_server
bash scripts/train.sh configs/train/dmf_libero_plus.yaml
```

### Resume from checkpoint

```bash
bash scripts/train.sh configs/train/dmf_libero_plus.yaml --resume checkpoints/dmf_finetuned/step_10000
```

## Architecture

- **Base model**: π₀.₅ VLA from Physical Intelligence
- **Modification**: Per-layer 3D adarms_cond `[depth, B, width]` — first `dmf_depth` layers use E(t), remaining use E(r)
- **E(t)/E(r)**: Reuse base pi0.5's `time_mlp_in`/`time_mlp_out` (perfect warm start, no new time-embedding modules)
- **logvar**: Predicted from hidden state + E(t) + E(r) (learned variance weighting)
- **Loss**: L = 0.5 * (L_FM + L_MF)
  - L_FM: Standard flow matching (model called with t=r)
  - L_MF: MeanFlow loss with JVP to compute du/dt
- **EMA**: decay=0.9999; checkpoints save EMA model to `params/` (eval loads this) and training model to `params_training/` (for resume)
- **No model guidance** (g_type="default")

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| dmf_depth_ratio | 0.67 | Encoder depth as ratio of total layers |
| P_mean | 0.0 | Mean for FM time distribution |
| P_mean_t | 0.4 | Mean for current time (encoder) distribution |
| P_mean_r | -1.2 | Mean for target time (decoder) distribution |
| P_std / P_std_t / P_std_r | 1.0 | Std for time distributions |
| ema_decay | 0.9999 | EMA decay for eval model |
| learning_rate | 1e-4 | AdamW learning rate |
| batch_size | 32 | Training batch size |
| training_steps | 30000 | Total training steps |

## File Structure

```
dmf/
├── configs/train/
│   ├── dmf_libero_plus.yaml     # LIBERO-Plus training config
│   ├── dmf_libero.yaml          # LIBERO training config
│   └── dmf_calvin.yaml          # CALVIN training config
├── scripts/
│   ├── train.sh                  # Training script (activates env, sets PYTHONPATH)
│   └── run_train.py              # Python entry point
├── src/dmf_vla/
│   ├── models/pi05_dmf.py        # DMF model (per-layer encoder/decoder split)
│   └── training/
│       ├── dmf_loss.py           # DMF loss with JVP
│       ├── jax_trainer.py        # JAX training loop (EMA + AdamW + cosine)
│       ├── freeze_utils.py       # Parameter freezing
│       └── data_loader.py        # Data loading
│   └── inference/
│       └── dmf_sampler.py        # Standalone samplers (Euler + 1-NFE)
└── README.md
```

## Key Implementation Details

### 1. Encoder-Decoder Split

The action expert layers (those with `_1` suffix) are split:
- **Encoder**: First `dmf_depth = floor(0.67 * total_layers)` layers use E(t)
- **Decoder**: Remaining layers use E(r)

Implementation: `_cond_stack` creates a 3D `[depth, B, width]` array where each
layer gets its own time embedding. `gemma.__call__` detects the 3D shape and
routes to `forward_with_intermediates` which manually iterates layers (instead of
nn.scan), applying per-layer adaRMS conditioning.

### 2. JVP for MeanFlow Loss

```python
# JVP computation for du/dt
primals = (z_t_mf, t_mf, r_mf)
tangents = (v_t_mf, 1, 0)  # Differentiate w.r.t z_t and t, not r
(u, lv_mf), (du_dt, _) = jax.jvp(dmf_model_fn, primals, tangents)

# Target flow map velocity
u_tgt = v_t_mf + (r_mf - t_mf) * du_dt
```

### 3. EMA

Trainable params have an EMA copy (decay=0.9999). At checkpoint time:
- `params/` ← EMA model (what eval loads)
- `params_training/` ← training model (for resume)
- `opt_state/` ← optimizer state (for resume)

### 4. logvar (Learned Variance)

`logvar_proj` predicts log-variance from the last action token's hidden state + E(t) + E(r):
```python
logvar = logvar_proj(concat([hidden[-1], E(t), E(r)]))  # 3*width → 1
```
Used in `log_lv_loss`: `log(exp(-lv) * mse + eps) + lv`

### 5. Parameter Freezing

Only DMF-specific parameters are trained:
- `time_mlp_in/**`, `time_mlp_out/**`: Base time MLP (reused for E(t)/E(r))
- `logvar_proj/**`: Log-variance prediction head
- Plus standard action expert parameters (attn_1, mlp_1, etc.)

## References

- Paper: https://arxiv.org/abs/2510.24474
- Official repo: https://github.com/kyungmnlee/dmf
- JAX training patterns: `/root/autodl-tmp/smfVLA/`, `/root/autodl-tmp/snapflow/`
