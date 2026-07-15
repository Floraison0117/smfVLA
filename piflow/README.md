# Pi-Flow for π₀.₅ VLA

JAX port of **π-Flow** for 1/2/4-NFE action generation on LIBERO-Plus. Distills the
frozen π₀.₅ teacher into a student that predicts a Gaussian-Mixture (GMM) policy
whose analytic GMFlow rollout needs **zero** teacher calls at inference time.

**Paper**: [π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](https://arxiv.org/abs/2510.14974) (ICLR 2026)
**Official repo (PyTorch)**: [Lakonik/LakonLab](https://github.com/Lakonik/LakonLab)
**GMFlow paper**: [arXiv 2504.05304](https://arxiv.org/abs/2504.05304) (ICML 2025)

## Overview

π-Flow is a *policy-based* distillation method. Instead of regressing a single
denoised state, the student network outputs the parameters of a **GMFlow policy**
(component means, log-stds, log-weights). That policy is then *rolled out
analytically* (no extra network calls) over `inner_substeps` Euler substeps to
reach the denoised action chunk.

Training uses **policy-based imitation distillation (pi-ID)**: a single L2 loss
between the student policy's rollout velocities and the frozen teacher's
instantaneous velocities at a few query states. No JVPs, no auxiliary networks,
no GANs.

## Usage

### Training

```bash
cd /root/autodl-tmp/piflow
bash scripts/train.sh                                    # default config (LIBERO-Plus)
bash scripts/train.sh configs/train/piflow_libero_plus.yaml --resume checkpoints/piflow_finetuned/step_10000
```

`train.sh` activates `openpi_server`, sets `PYTHONPATH`
(`piflow/src` + `openpi/src` + client), then runs `run_train.py`. If the dataset
path is missing it falls back to a fake data loader for smoke testing.

### Evaluation

Pi-Flow is wired into the unified eval entry point (auto-detected by
`detect_checkpoint_type()` sniffing for `gmm_mean_proj`):

```bash
cd /root/autodl-tmp
python -m eval.libero_plus.main --model-type piflow --nfe 1 --mode quick
python -m eval.libero_plus.main --model-type piflow --nfe 1 --mode fullset
```

Eval uses `sample_kwargs={"num_steps": nfe, "method": "gmflow"}` — only the
student transformer runs; the teacher is never called at inference.

### Environment & code style

Same as the other methods: conda env `openpi_server`, `WANDB_API_KEY` required
for logging, and

```bash
black --line-length 100 .
isort --profile black --line-length 100 .
ruff check --line-length 100 .
```

## Architecture

### Core algorithm

**Time convention** (same as π₀.₅): `t=1` is noise, `t=0` is data.
Interpolation: `x_t = (1-t)·x_0 + t·ε`, velocity `u = (x_t - x_0) / t`.

**Student forward** (`Pi05PiFlow.forward_gmm`): one π₀.₅ transformer pass at the
outer time `t_src` → predicts velocity-space GMM parameters. These are converted
to x₀-space at prediction time:
```
vel_means = gmm_mean_proj(pooled)            # velocity means (network output)
means_x0  = x_t_src - t_src * vel_means     # converted to x_0 means
log_stds  = gmm_logstd_proj(pooled)         # velocity-space log std (t-invariant)
log_weights = gmm_logweight_proj(pooled)    # mixture weights
```
The action-token hidden states are mean-pooled over the horizon, then projected
by three heads (all zero-init for a clean no-op start: at init,
`vel_means=0` → `means_x0 = x_t_src`).

**Variance scaling** (official π-Flow parameterization): the component variance
scales with the prediction time:
```
var_k = exp(2 * log_stds_k) * t_src^2
```
At `t_src=1` (1-NFE): `var = exp(2*logstd)` (unscaled). At `t_src=0.5` (2-NFE
second step): `var = 0.25 * exp(2*logstd)` (matches posterior contraction).
This makes the network predict t-invariant quantities; the policy applies the
t_src scaling.

**GMFlow analytic policy** (`gmflow.gmflow_velocity`): for each component `k`
with mean `μ_k` and variance `σ_k² = exp(2·logstd_k)·t_src²` (isotropic,
per-component, scales with prediction time `t_src`), the posterior and velocity
are closed-form:

```
α_{t,k} = (1-t)²·σ_k² + t²
E[x_0 | x_t, k] = (t²·μ_k + (1-t)·σ_k²·x_t) / α_{t,k}
u_k(x_t, t) = (x_t - E[x_0 | x_t, k]) / t          (small-t: u_k ≈ μ_k - x_t)
γ_k ∝ w_k · N(x_t; (1-t)·μ_k, α_{t,k}·I)           (log-sum-exp)
u(x_t, t) = Σ_k γ_k · u_k(x_t, t)
```

**Rollout** (`gmflow.gmflow_rollout`): Euler integration `t: 1→0` over
`inner_substeps` (default 8) using the analytic velocity — no network calls.

**Training loss** (`compute_piflow_loss`, multi-NFE):
1. Sample `x_1 ~ N(0, I)`.
2. For each of `nfe` segments `[t_src_k, t_dst_k]`:
   a. Student GMM forward at `t_src_k`.
   b. GMFlow rollout within segment, recording states + velocities at
      `teacher_query_points` evenly-spaced substeps (clamped to substeps/seg).
   c. Frozen teacher velocity at those query states.
   d. Loss += MSE between student rollout velocities and teacher velocities.
   e. Advance `x` via `stop_gradient(rollout_final)` → next segment.
3. Loss = average over segments.

`inner_substeps` (default 8) is divided equally among segments:
`substeps_per_seg = max(2, inner_substeps // nfe)`. For 1-NFE: 8 substeps, 4
queries. For 4-NFE: 2 substeps, 2 queries per segment.

### Key components

| File | Role |
|------|------|
| `models/pi05_piflow.py` | `Pi05PiFlow` — extends `openpi.models.pi0.Pi0`; adds GMM heads, `forward_gmm()`, overrides `sample_actions()` for GMFlow sampling; `compute_loss()` raises (use `compute_piflow_loss`). |
| `models/gmflow.py` | Analytic GMFlow velocity + Euler rollout (`gmflow_velocity`, `gmflow_rollout`, `gmflow_rollout_with_states`). |
| `training/piflow_loss.py` | `compute_piflow_loss` — velocity imitation distillation. |
| `training/jax_trainer.py` | `PiFlowTrainer` — JIT train step, student/teacher split, EMA (0.9999), AdamW + cosine, checkpoint save/load (`params/` = EMA, `params_training/` = training, `opt_state/`). |
| `training/freeze_utils.py` | Glob-pattern freeze/train mask. Train = action-expert `*_1` + projections + GMM heads; `action_out_proj` is NOT trained (kept from teacher, unused by student). |
| `training/data_loader.py` | LeRobot v2.0/v2.1 loader (shared with DMF; observations only — action labels unused by the loss). |
| `inference/piflow_sampler.py` | Standalone 1-NFE / 2-NFE GMFlow samplers. |
| `configs/train/piflow_libero_plus.yaml` | Training config. |
| `scripts/train.sh`, `scripts/run_train.py` | Entry points (activate env, set PYTHONPATH, build teacher+student, run trainer). |

### File structure

```
piflow/
├── configs/train/piflow_libero_plus.yaml
├── scripts/{train.sh, run_train.py}
└── src/piflow_vla/
    ├── models/{pi05_piflow.py, gmflow.py}
    ├── training/{piflow_loss.py, jax_trainer.py, freeze_utils.py, data_loader.py}
    └── inference/piflow_sampler.py
```

No `third_party/openpi` symlink (like DMF, relies on PYTHONPATH → `openpi/src`).

## Key implementation details

### 1. Teacher / student setup
- **Teacher** = `Pi0(pi05=True)` loaded from `checkpoints/pi05_libero`, fully frozen.
- **Student** = `Pi05PiFlow` initialised from the same base checkpoint; GMM heads
  are zero/xavier-init (skipped during base-ckpt merge via `skip_patterns`).

### 2. Two-model JIT step
`jax_trainer._setup_jit_train_step` reconstructs both models inside one
`jax.value_and_grad` (gradients only on student-trainable params). Teacher is
`stop_gradient`'d. Teacher velocity is computed via `jax.lax.scan` over the
query points; the prefix is embedded once outside the scan.

### 3. EMA + checkpoint layout
- Trainable params carry an EMA copy (decay 0.9999).
- `params/` ← EMA model (what eval loads).
- `params_training/` ← training model (for `--resume`).
- `opt_state/` ← optimizer state; `train_state.json` ← step + metadata.

### 4. Parameter freezing
Same backbone-freeze convention as the other methods (VLM SigLIP + PaliGemma
LLM without `_1` frozen; action-expert `*_1`, `action_in_proj`, `time_mlp_*`,
and the three `gmm_*_proj` heads trainable). `action_out_proj` is frozen — the
student never reads it.

### 5. NFE handling
- **Train**: configurable via `nfe` in YAML (1, 2, or 4). Segments the trajectory
  into `nfe` equal parts; student predicts a fresh GMM at each boundary.
- **Inference `sample_actions(num_steps=N)`**: `N=1` → single GMM + 8-substep
  rollout; `N>1` → `N` outer steps, each GMM + `max(1, 8//N)` inner substeps.
  Each step passes `t_src` to gmflow for correct variance scaling.
- **t_src-dependent variance**: at init, all NFE modes produce a no-op (return
  input noise). The variance `exp(2·logstd)·t_src²` shrinks at lower t_src,
  matching the true Bayesian posterior contraction.

## Hyperparameters

From `configs/train/piflow_libero_plus.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_components (K) | 8 | GMM mixture components |
| inner_substeps | 8 | Total GMFlow Euler substeps (divided by NFE) |
| teacher_query_points | 4 | Teacher query points per segment |
| nfe | 1 | Training NFE (1, 2, or 4) |
| learning_rate | 1e-4 | AdamW |
| weight_decay | 0.01 | |
| warmup_steps | 1000 | Linear warmup |
| training_steps | 30000 | Total |
| batch_size | 16 | |
| ema_decay | 0.9999 | EMA for eval model |
| gradient_clip_norm | 1.0 | |
| precision | bfloat16 | |

## Fidelity to the official π-Flow

This is a JAX port adapted to the π₀.₅ VLA. Key differences from the official
PyTorch LakonLab implementation (all deliberate adaptations for VLA actions):

| Aspect | Official (LakonLab) | This port |
|--------|---------------------|-----------|
| Policy param | Network predicts **velocity**; converts to x₀-means; variance scales with `t_src²` | **Same** (adopted official parameterization) |
| Time schedule | Shifted timestep sampler (shift=3.2 for images) | Plain `linspace(1,0)` (actions, no shift) |
| Training segment | `piid_segment`: per-NFE segment with `num_intermediate_states=2`, teacher-ratio trajectory mixing, window substeps, scheduled decay | Per-NFE segment: single rollout per segment, teacher queried at evenly-spaced substeps; no trajectory mixing / teacher-ratio schedule |
| Student prediction | Average velocity over a window: `(x_start−x_end)/(σ_start−σ_end)` | Instantaneous GMFlow velocity at each substep |
| Data variant | `PiFlowImitation` (data) + `PiFlowImitationDataFree` | Effectively data-free for the loss (observations condition the student; action labels unused beyond shape) |
| Temperature | `temperature_()` between NFE steps for stochasticity | Not implemented |
| Multi-NFE train | Segment-per-NFE | **Same** (segment-per-NFE, `nfe` config) |
| Teacher query cost | KV-cache reused | **KV-cache reused** (prefix computed once per observation, shared across all query steps and NFE segments) |

The core π-Flow idea — student predicts a GMM, analytic GMFlow policy rolls out
substeps, single L2 velocity-imitation loss against a frozen teacher — is
preserved, now including the official velocity parameterization with t_src²
variance scaling and multi-NFE segment training. The remaining simplifications
(no trajectory mixing, no temperature, no KV cache) trade fidelity for
simplicity and are reasonable adaptations to VLA action chunks.

### Code-review notes
- `data_loader.py` header docstring says *"for DMF training"* — copy-paste from DMF; harmless but misleading.
- `compute_piflow_loss` takes an `actions` arg used only for shape `(B,H,D)`; the loss is data-free. Intentional, but the name suggests labels are used.
- No `third_party/openpi` symlink; depends on `train.sh`'s PYTHONPATH (like DMF).
- Eval integration already present: `eval/common/policy_loader.py` (auto-detect via `gmm_mean_proj`) and `eval/libero_plus/main.py` (`--model-type piflow`).
- `pi05_piflow.py` uses `from typing_extensions import override` (not `typing.override`) for Python 3.11 compat.
- Teacher KV cache: the teacher's prefix (images + language tokens) is computed once per observation and the KV cache is reused across all M teacher query steps and all NFE segments. Verified numerically equivalent to the full-forward approach (max diff = 0.0).
- Unit tests: `piflow/tests/test_gmflow.py` covers posterior formula, velocity, small-t approximation, t_src variance scaling, responsibilities, rollout, gradients (18 tests). Run: `cd /root/autodl-tmp && python -m pytest piflow/tests/test_gmflow.py -v`.

## Key paths

| Item | Path |
|------|------|
| Base checkpoint (teacher + student init) | `checkpoints/pi05_libero/` |
| Dataset | `datasets/libero-plus-training/` (like DMF) |
| Finetuned checkpoints | `checkpoints/piflow_finetuned/step_N/` |
| Training config | `piflow/configs/train/piflow_libero_plus.yaml` |

## References

- π-Flow paper: https://arxiv.org/abs/2510.14974
- Official repo: https://github.com/Lakonik/LakonLab (`lakonlab/models/diffusions/piflow.py`, `piflow_policies/gmflow.py`)
- GMFlow paper: https://arxiv.org/abs/2504.05304
- JAX training patterns: `smfVLA/`, `snapflow/`, `freeflow/`, `dmf/`
