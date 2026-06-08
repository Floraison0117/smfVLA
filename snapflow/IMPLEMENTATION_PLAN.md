# SnapFlow Implementation Plan

**Project**: Reproduce SnapFlow (arXiv:2604.05656) on LIBERO
**Date**: 2026-06-06
**Status**: Planning Phase

## Executive Summary

SnapFlow is a self-distillation method that compresses multi-step denoising (10 ODE steps) into a single forward pass (1-NFE) for flow-matching VLAs. Key innovations:

1. **Two-Step Euler Shortcut**: Targets are 2-step Euler shortcut velocities computed from the model's own marginal velocity predictions
2. **Target-Time Embedding**: Zero-initialized MLP encoding target time `s`
3. **Progressive FM/Consistency Mixing**: Loss = α·L_FM + (1-α)·λ·L_shortcut

**Target Results** (from paper on π0.5 3B):
- 98.75% average success on LIBERO (vs 97.75% baseline 10-step)
- 9.6× denoising speedup, E2E latency: 274ms → 83ms
- Training time: ~12h on single GPU, 30k steps

---

## Paper Analysis Summary

### Algorithm Core (Algorithm 1 from paper)

**Training Loop** (each step uses 3 forward passes):
1. **FM component** (probability α=0.5): Standard flow matching at random time t
2. **Consistency component** (probability 1-α=0.5):
   - v_1 = stop_gradient(F_θ(x_1, 1, 1|c)) — velocity at t=1
   - x_0.5 = x_1 - 0.5 · v_1 — midpoint via Euler
   - v_0.5 = stop_gradient(F_θ(x_0.5, 0.5, 0.5|c)) — velocity at t=0.5
   - v_target = 0.5 · (v_1 + v_0.5) — 2-step average velocity
   - L_shortcut = ||F_θ(x_1, 0, 1|c) - v_target||²

**Total Loss**: L = α·L_FM + (1-α)·λ·L_shortcut

### Key Design Choices (from ablation study)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| α (FM/Consistency ratio) | 0.5 | Balanced mix maintains u_θ while providing 1-step supervision |
| λ (Consistency weight) | 0.1 | Balances gradient magnitudes between FM and consistency |
| Learning rate | 2.5×10⁻⁵ | 1/10 of π0.5 training rate (fine-tuning) |
| Training steps | 30,000 | Plateaus by ~3.5k steps, 30k for safety margin |
| Batch size | 4 | Matches paper configuration |
| Prediction clamp | [-20, 20] | Prevents numerical instabilities |

### Theoretical Foundations

**Theorem 1**: Conditional velocity v_t ≠ marginal velocity u_t almost surely
**Theorem 2**: Using conditional velocity in consistency loss introduces trajectory drift
**Theorem 3**: Multi-step integration accumulates local residuals

SnapFlow avoids these issues by:
- Using the model's own marginal velocity predictions (v_1, v_0.5)
- Computing shortcut target via Euler integration
- Progressive mixing to stabilize training

---

## Existing Infrastructure Analysis

### smfVLA Project (Reference Implementation)

Located at `/root/autodl-tmp/smfVLA/`

**Relevant Components**:

1. **Model** (`src/smf_vla/models/pi05_smf.py`):
   - `Pi05SMF` extends `Pi0` with dual time inputs (r, t)
   - `time_proj`: Linear(2*width, width) initialized to [I, 0]
   - Supports `embed_suffix_smf(obs, noisy_actions, t, r)`
   - `compute_loss()` with SMF loss variants
   - `sample_actions()` for 1-NFE and multi-step inference

2. **Loss Functions** (`src/smf_vla/training/smf_loss.py`):
   - `compute_smf_loss()` - Basic SMF loss
   - `compute_full_smf_loss()` - Unified loss supporting all variants
   - Time sampling: `sample_r_t()`, `sample_r_t_curriculum()`

3. **Training** (`src/smf_vla/training/jax_trainer.py`):
   - JAX JIT-compiled training loop
   - Selective parameter updates (trainable params only)
   - Optax AdamW + linear warmup + cosine decay
   - Checkpoint save/load with optimizer state
   - WandB logging

4. **Data Loader** (`src/smf_vla/training/data_loader.py`):
   - LeRobot v2.0 format (Parquet files)
   - Dataset: `data/libero/` - 40 tasks, 1693 episodes
   - Image preprocessing: rotate 180°, resize 256×256 → 224×224

5. **Evaluation** (`scripts/eval_direct.py`, `scripts/eval_utils.py`):
   - Direct evaluation on LIBERO suites
   - Presets: quick (spatial, 5 ep/task), full (all suites, 50 ep/task)
   - NFE selection: `--nfe 1` or `--nfe 10`

### Key Differences: SMF vs SnapFlow

| Aspect | SMF (SplitMeanFlow) | SnapFlow |
|--------|---------------------|----------|
| Target | Average velocity u(z_t, r, t) | 1-step consistency via 2-step shortcut |
| Loss | Self-consistency + FM | α·FM + (1-α)·λ·shortcut |
| Time embedding | Concat E(t), E(r) + time_proj | Time + target-time MLP φ_s |
| Sampling | (r, t) pairs with Bernoulli | Random t + fixed {1, 0.5} for shortcut |
| Training | Single forward pass per sample | 3 forward passes (FM + 2 for shortcut) |

---

## Implementation Architecture

### Directory Structure

```
snapflow/
├── configs/
│   └── train/
│       └── snapflow_libero.yaml          # Training configuration
├── src/
│   └── snapflow/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── pi05_snapflow.py          # Pi0.5 + SnapFlow modifications
│       │   └── target_time_mlp.py        # Target-time embedding φ_s
│       ├── training/
│       │   ├── __init__.py
│       │   ├── snapflow_loss.py          # SnapFlow loss (shortcut + FM)
│       │   ├── jax_trainer.py            # Training loop (adapted from smfVLA)
│       │   └── data_loader.py            # Reuse from smfVLA or symlink
│       └── eval/
│           ├── __init__.py
│           └── eval_utils.py             # Reuse from smfVLA or symlink
├── scripts/
│   ├── train.sh                          # Training entry point
│   ├── run_train.py                      # Python training script
│   └── eval_direct.py                    # Evaluation script
├── data/                                  # Symlink to smfVLA/data/libero
├── checkpoints/
│   └── base/
│       └── pi05_libero/                  # Symlink to smfVLA checkpoint
├── logs/
│   ├── train/
│   └── eval/
├── third_party/
│   └── openpi -> /root/autodl-tmp/openpi  # Symlink to openpi
├── CLAUDE.md                             # Project documentation
├── pyproject.toml                        # Python dependencies
└── README.md                             # Project description
```

### Component Design

#### 1. Target-Time MLP (`target_time_mlp.py`)

```python
class TargetTimeMLP(nnx.Module):
    """Zero-initialized 2-layer MLP encoding target time s.

    Initialized to zero so that at step 0, the network behaves identically
    to the pretrained teacher (no target-time conditioning).
    """
    def __init__(self, width: int, rngs: nnx.Rngs):
        # Layer 1: width -> width
        # Layer 2: width -> width
        # Zero initialization: all weights and biases = 0

    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        # s: [B,] target times
        # Returns: [B, width] embedding
```

#### 2. Pi05SnapFlow Model (`pi05_snapflow.py`)

Extend `Pi05SMF` with:
- `target_time_mlp: TargetTimeMLP` - NEW component
- Modified `embed_suffix()` to inject target-time embedding
- `compute_snapflow_loss()` - SnapFlow-specific loss

```python
class Pi05SnapFlow(Pi05SMF):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.target_time_mlp = TargetTimeMLP(
            width=action_expert_config.width,
            rngs=rngs
        )

    def embed_suffix_with_target_time(
        self, obs, noisy_actions, t, s
    ):
        """Embed suffix with target-time conditioning.

        Args:
            t: current time (for velocity estimation)
            s: target time (s=t for FM, s=0 for consistency)
        """
        # Base time embedding from parent
        time_emb_t = self._compute_time_emb(t)  # [B, width]

        # Target-time embedding
        target_emb = self.target_time_mlp(s)    # [B, width]

        # Combine: base + target (additive, not concat)
        adarms_cond = time_emb_t + target_emb

        # ... rest same as embed_suffix_smf

    def compute_snapflow_loss(self, rng, observation, actions, ...):
        """SnapFlow loss: α·L_FM + (1-α)·λ·L_shortcut"""
        # Implementation in snapflow_loss.py
        pass
```

#### 3. SnapFlow Loss (`snapflow_loss.py`)

```python
def compute_snapflow_loss(
    model_fn, params, observation, actions,
    action_mean, action_std, rng,
    alpha=0.5, lambda_consistency=0.1,
):
    """
    SnapFlow loss with progressive FM/consistency mixing.

    Three forward passes:
    1. FM: F_θ(x_t, t, t) at random t
    2. Shortcut v1: F_θ(x_1, 1, 1)
    3. Shortcut v0.5: F_θ(x_0.5, 0.5, 0.5)
    """
    batch_size = actions.shape[0]
    rng_fm, rng_consist = jax.random.split(rng)

    # Normalize actions
    x_norm = (actions - action_mean) / (action_std + 1e-8)

    # Sample noise
    noise = jax.random.normal(rng_consist, x_norm.shape)

    # --- FM Component (α fraction of batch) ---
    t_fm = jax.random.uniform(rng_fm, (batch_size,))
    z_t = interpolate_z(x_norm, noise, t_fm)
    v_fm = model_fn(params, observation, z_t, t_fm, t_fm)
    target_fm = noise - x_norm
    loss_fm = jnp.mean(jnp.square(v_fm - target_fm))

    # --- Consistency Component (1-α fraction) ---
    # v_1 at t=1
    v_1 = model_fn(params, observation, noise, jnp.ones(batch_size), jnp.ones(batch_size))
    v_1_sg = jax.lax.stop_gradient(v_1)

    # x_0.5 via Euler
    x_0_5 = noise - 0.5 * v_1_sg

    # v_0.5 at t=0.5
    t_half = jnp.full((batch_size,), 0.5)
    v_0_5 = model_fn(params, observation, x_0_5, t_half, t_half)
    v_0_5_sg = jax.lax.stop_gradient(v_0_5)

    # Target: 2-step average velocity
    v_target = 0.5 * (v_1_sg + v_0_5_sg)

    # Prediction: F_θ(x_1, 0, 1)
    r_zero = jnp.zeros(batch_size)
    t_one = jnp.ones(batch_size)
    v_pred = model_fn(params, observation, noise, r_zero, t_one)

    loss_shortcut = jnp.mean(jnp.square(v_pred - v_target))

    # Total loss
    loss_total = alpha * loss_fm + (1 - alpha) * lambda_consistency * loss_shortcut

    metrics = {
        "loss_total": loss_total,
        "loss_fm": loss_fm,
        "loss_shortcut": loss_shortcut,
        "alpha": alpha,
        "lambda": lambda_consistency,
    }

    return loss_total, metrics
```

---

## Implementation Phases

### Phase 1: Project Setup (Day 1)

**Tasks**:
1. Create directory structure
2. Setup symlinks to shared resources:
   - `data/` → smfVLA/data/libero
   - `checkpoints/` → smfVLA/checkpoints
   - `third_party/openpi` → openpi
3. Create `pyproject.toml` with dependencies
4. Create `CLAUDE.md` with project documentation
5. Verify conda environment setup

**Verification**:
- [ ] Directory structure created
- [ ] Symlinks work
- [ ] Can import from openpi and smfVLA

### Phase 2: Core Components (Days 2-3)

**Tasks**:
1. Implement `TargetTimeMLP` with zero initialization
2. Implement `Pi05SnapFlow` extending `Pi05SMF`
3. Implement `compute_snapflow_loss()` function
4. Create training config `snapflow_libero.yaml`
5. Adapt `jax_trainer.py` from smfVLA

**Verification**:
- [ ] TargetTimeMLP outputs zeros at init
- [ ] Model forward pass works with (r, t, s) inputs
- [ ] Loss computation produces correct metrics
- [ ] JIT compilation succeeds

### Phase 3: Training Infrastructure (Days 4-5)

**Tasks**:
1. Implement `jax_trainer.py` with SnapFlow-specific logic
2. Create `run_train.py` entry point
3. Create `train.sh` wrapper script
4. Setup WandB logging for SnapFlow metrics
5. Implement gradient checkpointing for memory efficiency

**Verification**:
- [ ] Can run 1 training step
- [ ] Checkpoint save/load works
- [ ] Metrics logged to WandB
- [ ] Memory usage < 80GB (A800)

### Phase 4: Evaluation Infrastructure (Day 6)

**Tasks**:
1. Adapt `eval_direct.py` from smfVLA for SnapFlow
2. Create `eval_utils.py` with shared evaluation logic
3. Implement 1-NFE inference path
4. Create evaluation presets (quick, medium, full)

**Verification**:
- [ ] Can load trained checkpoint
- [ ] 1-NFE inference produces valid actions
- [ ] Evaluation runs on libero_spatial (quick preset)

### Phase 5: Training & Experiments (Days 7-10)

**Tasks**:
1. Run full 30k-step training
2. Monitor convergence via WandB
3. Evaluate checkpoints at 5k, 10k, 20k, 30k steps
4. Compare baseline vs SnapFlow on all LIBERO suites
5. Run ablation studies (α, λ values)

**Verification**:
- [ ] Training completes 30k steps
- [ ] Loss converges (target: ~0.017)
- [ ] Evaluation metrics match paper expectations
- [ ] 1-NFE success rate ≥ 98% on LIBERO

### Phase 6: Analysis & Documentation (Days 11-12)

**Tasks**:
1. Generate evaluation results report
2. Create comparison plots (MSE, CosSim, success rate)
3. Document hyperparameter ablations
4. Write README with usage instructions
5. Create training/eval scripts for reproducibility

**Verification**:
- [ ] Results documented in report
- [ ] Code is reproducible
- [ ] README is comprehensive

---

## Configuration

### Training Config (`snapflow_libero.yaml`)

```yaml
method: snapflow
description: "SnapFlow: 1-NFE via progressive self-distillation"

# Model
checkpoint: /root/autodl-tmp/snapflow/checkpoints/base/pi05_libero
pi05: true
action_dim: 32
action_horizon: 10

# SnapFlow parameters
alpha: 0.5              # FM/Consistency mixing ratio
lambda_consistency: 0.1 # Consistency loss weight
prediction_clamp: [-20, 20]

# Training hyperparameters (from paper)
learning_rate: 2.5e-5
weight_decay: 0.01
warmup_ratio: 0.017     # 500 steps / 30000 total
gradient_clipping: 1.0
batch_size: 4
training_steps: 30000
precision: bf16

# Optimizer
optimizer: AdamW

# Data
dataset: libero
dataset_path: /root/autodl-tmp/snapflow/data/libero

# Checkpointing
checkpoint_dir: /root/autodl-tmp/snapflow/checkpoints/finetuned/snapflow
log_dir: /root/autodl-tmp/snapflow/logs/train/snapflow
save_every: 5000
log_every: 100
resume: null

# WandB
wandb:
  project: snapflow
  run_name: null
  entity: null

# Freeze strategy (VLM backbone frozen)
freeze:
  - "PaliGemma/img/**"
  - "PaliGemma/llm/embedder/**"
  - "PaliGemma/llm/final_norm/scale"
  # ... (full list from smfVLA)

trainable:
  - "PaliGemma/llm/layers/*/1/**"  # Action expert layers
  - "action_in_proj/**"
  - "action_out_proj/**"
  - "time_mlp_in/**"
  - "time_mlp_out/**"
  - "time_proj/**"
  - "target_time_mlp/**"  # NEW: SnapFlow target-time MLP
```

---

## Data & Dependencies

### Dataset

Reuse existing LIBERO dataset from smfVLA:
- Path: `/root/autodl-tmp/smfVLA/data/libero/`
- Format: LeRobot v2.0 (Parquet + episodes.jsonl)
- Content: 40 tasks, 1693 episodes, 258K frames
- Suites: libero_spatial, libero_object, libero_goal, libero_10

### Conda Environment

Reuse existing `openpi_server` environment:
- Path: `/root/miniconda3/envs/openpi_server`
- Key packages: JAX, Flax, optax, WandB, LeRobot

### Base Checkpoint

Reuse π0.5 LIBERO checkpoint from smfVLA:
- Path: `/root/autodl-tmp/smfVLA/checkpoints/base/pi05_libero/`
- Contents: params/ + assets/

---

## Success Criteria

### Quantitative Metrics

| Metric | Target | Source |
|--------|--------|--------|
| LIBERO Success (1-NFE) | ≥ 98% | Paper: 98.75% |
| MSE Reduction | ≥ 30% | Paper: 33.9% |
| CosSim Improvement | ≥ 0.3% | Paper: 0.31% |
| Denoising Speedup | 9× | Paper: 9.6× |
| E2E Latency | < 90ms | Paper: 83ms |

### Qualitative Checks

- [ ] Training converges without NaN/instability
- [ ] Loss decreases from ~0.021 to ~0.017
- [ ] Gradient norm decreases from ~0.6 to ~0.4
- [ ] Checkpoints can be loaded for inference
- [ ] Evaluation runs without errors

---

## Risks & Mitigations

### Risk 1: No Official Code

**Issue**: SnapFlow has no public GitHub implementation
**Mitigation**: Paper provides detailed algorithm pseudocode and hyperparameters

### Risk 2: Three Forward Passes Memory

**Issue**: SnapFlow requires 3 forward passes per step
**Mitigation**:
- Use gradient checkpointing
- VLM backbone is frozen (no gradients)
- Paper reports ~40GB VRAM on A800

### Risk 3: Target-Time Integration

**Issue**: Adding target-time embedding may break pretrained weights
**Mitigation**: Zero initialization ensures network starts at teacher behavior

### Risk 4: Convergence Speed

**Issue**: Training may require more than 30k steps
**Mitigation**: Monitor loss curves; extend if needed

---

## References

1. **SnapFlow Paper**: https://arxiv.org/abs/2604.05656
2. **π0.5 Paper**: https://arxiv.org/abs/2504.16054
3. **Flow Matching**: https://arxiv.org/abs/2302.05442
4. **Consistency Models**: https://arxiv.org/abs/2309.05155
5. **smfVLA Project**: `/root/autodl-tmp/smfVLA/`
6. **openpi Project**: `/root/autodl-tmp/openpi/`

---

## Appendix: Ablation Study Plan

Based on Table 5 from paper, test these configurations:

| Variant | α | λ | Target-Time Embed? |
|---------|---|---|-------------------|
| Pure consistency | 0.0 | 0.1 | ✓ |
| Consistency-heavy | 0.3 | 0.1 | ✓ |
| **Balanced (default)** | **0.5** | **0.1** | **✓** |
| FM-heavy | 0.7 | 0.1 | ✓ |
| Pure FM | 1.0 | 0.1 | ✓ |
| No embedding | 0.5 | 0.1 | ✗ |

Run 5k steps each, compare MSE and CosSim on held-out set.
