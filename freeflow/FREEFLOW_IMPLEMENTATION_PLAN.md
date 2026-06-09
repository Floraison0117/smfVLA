# FreeFlow for LIBERO-Plus: Implementation Plan

## Project Overview

Adapting **FreeFlow** (Flow Map Distillation Without Data, [arXiv:2511.19428](https://arxiv.org/abs/2511.19428)) from image generation to Vision-Language-Action (VLA) models for 1-NFE robotics training on LIBERO-Plus benchmark.

**Goal:** Train a student VLA that achieves 1-NFE performance comparable to 10-NFE teacher via data-free distillation.

**Base Checkpoint:** `pi05-libero` (same as SMF/SnapFlow)

**Target:** LIBERO-Plus robustness evaluation benchmark

---

## Background: FreeFlow Core Algorithm

### Original FreeFlow (Image Generation)

FreeFlow distills a pre-trained teacher flow model into a 1-step student without requiring training data:

1. **Teacher Model**: T_θ(z_t, t) — multi-step flow model (NFE=10 for π₀.₅)
2. **Student Model**: S_φ(z_t, t) — 1-step model to learn
3. **Key Innovation**: Sample from prior distribution p(z_1) instead of dataset

**Loss Function**:
```
L = L_path + L_correction

L_path = ||S_φ(z_1, 0→1) - T_θ(z_1, 0→1)||²
L_correction = ||S_φ(z_t, t→0) - T_θ(z_0, t→0)||²
```

Where:
- `S_φ(z_1, 0→1)` = Student's predicted velocity from t=1 to t=0
- `T_θ(z_1, 0→1)` = Teacher's multi-step Euler integration path
- `L_correction` actively corrects compounding errors

### Adaptation Challenges for VLA

| Challenge | Image Generation | VLA/Robotics | Solution |
|-----------|------------------|--------------|----------|
| Input space | Pure noise z_t | Observation + z_t | Condition on observation |
| Output space | Image pixels | Action sequences | Same flow matching framework |
| Teacher sampling | From p(z_1) | From data + observation | Use offline data sampling |
| Evaluation metric | FID | LIBERO success rate | Task-specific metrics |

---

## Architecture Design

### Model Components

```
freeflow/
├── src/
│   └── freeflow/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── pi05_freeflow.py          # Main model extending Pi0
│       │   ├── teacher_wrapper.py        # Wrapper for π₀.₅ teacher
│       │   └── student_head.py           # Lightweight student head
│       ├── training/
│       │   ├── __init__.py
│       │   ├── freeflow_loss.py          # Data-free distillation loss
│       │   ├── jax_trainer.py            # Training loop
│       │   ├── freeze_utils.py           # Parameter freezing
│       │   └── data_loader.py            # Reuse from smfVLA
│       └── config/
│           └── default_config.py
├── configs/
│   └── train/
│       ├── freeflow_base_libero.yaml
│       ├── freeflow_curr_libero.yaml
│       └── freeflow_full_libero.yaml
├── scripts/
│   ├── train.sh
│   └── eval_freeflow.sh
├── checkpoints/
│   ├── base/
│   │   └── pi05_libero/                 # Symlink to smfVLA
│   └── finetuned/
│       └── freeflow/
├── data -> ../datasets/libero/          # Symlink to smfVLA
├── third_party/
│   └── openpi -> ../../openpi/          # Symlink
├── CLAUDE.md
├── README.md
└── pyproject.toml
```

### Key Design Decisions

#### 1. Teacher-Student Architecture

**Teacher**: π₀.₅ base model (frozen, NFE=10)
- Pre-trained on LIBERO
- Uses standard 10-step Euler integration
- Provides "oracle" action trajectories

**Student**: Lightweight modification of π₀.₅
- Same backbone as teacher (initially)
- Trained to predict 1-step actions
- Parameters initialized from teacher

**Distillation Strategy**:
- Student learns to mimic teacher's 10-step output in 1 step
- Data-free: sample noise from prior, not from dataset
- Loss on velocity field, not final actions

#### 2. FreeFlow Loss for VLA

```python
def compute_freeflow_loss(
    model_fn,       # Student model S_φ
    teacher_fn,      # Teacher model T_θ (frozen)
    params,          # Student parameters
    observation,     # VLA observation
    actions,         # Ground truth (for normalization only)
    action_mean,     # Normalization stats
    action_std,
    rng,
    num_teacher_steps=10,  # Teacher NFE
    lambda_correction=0.1,  # Error correction weight
):
    """
    Data-free distillation loss adapted from FreeFlow paper.

    Algorithm:
    1. Sample z_1 ~ N(0, I) from prior
    2. Get teacher path: z_0^T = Euler(T_θ, z_1, num_steps=10)
    3. Get student prediction: z_0^S = z_1 - S_φ(z_1, 0, 1)
    4. Path loss: ||z_0^S - z_0^T||²

    5. For error correction at intermediate t:
       - Sample z_t from teacher path
       - Get correction target from teacher
       - Correction loss at intermediate points
    """
    # Step 1: Sample from prior (data-free!)
    noise = jax.random.normal(rng, actions.shape)
    z_1 = noise  # Start from pure noise

    # Step 2: Teacher's multi-step integration (frozen, no grad)
    z_0_teacher = euler_integration(
        teacher_fn, observation, z_1,
        num_steps=num_teacher_steps,
        params=teacher_params  # Frozen
    )

    # Step 3: Student's 1-step prediction
    v_student = model_fn(params, observation, z_1, 0.0, 1.0)  # t=0→1
    z_0_student = z_1 - v_student

    # Step 4: Path loss (main distillation)
    loss_path = jnp.mean(jnp.square(z_0_student - z_0_teacher))

    # Step 5: Error correction (optional, from FreeFlow paper)
    # Sample intermediate time points
    t_correction = jax.random.uniform(rng, (), minval=0.2, maxval=0.8)

    # Get state at t_correction from teacher
    z_t_teacher = intermediate_euler(teacher_fn, observation, z_1, t_correction)

    # Student's correction from z_t
    v_correction = model_fn(params, observation, z_t_teacher, 0.0, t_correction)
    z_0_from_t = z_t_teacher - v_correction

    # Correction should match teacher's path
    loss_correction = jnp.mean(jnp.square(z_0_from_t - z_0_teacher))

    # Total loss
    loss_total = loss_path + lambda_correction * loss_correction

    return loss_total, {
        'loss_path': loss_path,
        'loss_correction': loss_correction,
    }
```

#### 3. Comparison with SMF and SnapFlow

| Aspect | SMF | SnapFlow | FreeFlow |
|--------|-----|----------|----------|
| Teacher | None (self-training) | None (self-training) | π₀.₅ (frozen) |
| Data | Required (offline dataset) | Required (offline dataset) | **Data-free** (prior only) |
| Loss target | Self-consistency | 2-step shortcut | Teacher's 10-step path |
| Forward passes | 2-3 | 3 | 2-4 (student + teacher calls) |
| Key innovation | Dual time (r,t) | Target-time MLP | **Prior sampling + error correction** |

---

## Implementation Phases

### Phase 1: Setup and Infrastructure (Days 1-2)

**Tasks:**
1. Create `freeflow/` directory structure
2. Set up symlinks (data, checkpoints, openpi)
3. Create `pyproject.toml` with dependencies
4. Create `CLAUDE.md` with project documentation
5. Set up conda environment (reuse `openpi_server`)

**Files to create:**
```
freeflow/
├── pyproject.toml
├── CLAUDE.md
├── README.md
├── src/
│   └── freeflow/
│       └── __init__.py
├── configs/
│   └── train/
│       └── freeflow_base_libero.yaml
├── scripts/
│   └── train.sh
└── third_party/
    └── openpi -> ../../openpi/
```

**Dependencies** (same as smfVLA/snapflow):
- JAX + JAXlib (CUDA)
- Optax (optimizer)
- Flax (NNX)
- OpenPI (from third_party)
- LeRobot (data loading)
- WandB (logging)

### Phase 2: Core Model Implementation (Days 3-5)

**Tasks:**
1. Implement `Pi05FreeFlow` model class
2. Implement `TeacherWrapper` for frozen teacher
3. Implement `compute_freeflow_loss()`
4. Implement data loader (reuse from smfVLA)
5. Implement parameter freezing utils

**Key files:**
```
src/freeflow/models/
├── pi05_freeflow.py          # Main model
├── teacher_wrapper.py        # Teacher wrapper
└── student_head.py           # Student head

src/freeflow/training/
├── freeflow_loss.py          # Loss function
├── freeze_utils.py           # Freezing patterns
└── data_loader.py            # Data loading (or symlink)
```

**Implementation details:**

#### `Pi05FreeFlow` Model
```python
class Pi05FreeFlow(Pi0):
    """
    FreeFlow student model extending π₀.₅.

    Key differences from base Pi0:
    - Same architecture as teacher initially
    - Trained with FreeFlow distillation loss
    - Supports 1-NFE inference
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Optional: Lightweight student head
        # Can be added for efficiency
        self.student_head = nn.Dense(
            features=self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros
        )

    def __call__(self, observation, noisy_actions, t, r=0.0):
        """
        Forward pass for FreeFlow training/inference.

        Args:
            observation: VLA observation
            noisy_actions: z_t (interpolated state)
            t: Current time
            r: Reference time (default 0 for 1-NFE)

        Returns:
            velocity: Predicted velocity field
        """
        # Use same architecture as base Pi0
        # The key is in the loss function, not model changes
        return super().__call__(observation, noisy_actions, t, r)
```

#### `TeacherWrapper`
```python
class TeacherWrapper:
    """
    Wrapper for frozen teacher model (π₀.₅ base).

    Provides:
    - Multi-step Euler integration
    - Intermediate path queries
    - No gradient computation
    """

    def __init__(self, teacher_params, teacher_fn):
        self.params = teacher_params
        self.fn = teacher_fn

    @jax.jit
    def integrate(self, observation, z_1, num_steps=10):
        """
        Euler integration from z_1 to z_0.

        Args:
            observation: VLA observation
            z_1: Starting noise (t=1)
            num_steps: Number of integration steps

        Returns:
            z_0: Clean action (t=0)
        """
        dt = 1.0 / num_steps
        z_t = z_1

        for step in range(num_steps):
            t = (step + 1) * dt  # t goes from 0 to 1
            v_t = self.fn(self.params, observation, z_t, t, t)
            z_t = z_t - dt * v_t

        return z_t

    @jax.jit
    def get_intermediate(self, observation, z_1, t_target):
        """
        Get state at intermediate time t_target.

        For error correction loss.
        """
        # Implementation details...
        pass
```

### Phase 3: Training Infrastructure (Days 6-8)

**Tasks:**
1. Implement `FreeFlowTrainer` class
2. Set up optimizer and learning rate schedule
3. Implement checkpoint save/load
4. Set up WandB logging
5. Create training script `run_train.py`

**Key files:**
```
src/freeflow/training/
├── jax_trainer.py            # Main training loop
├── freeflow_loss.py          # Loss (from Phase 2)
└── freeze_utils.py           # Freezing (from Phase 2)

scripts/
├── train.sh                  # Training shell script
└── run_train.py              # Training entry point
```

**Training hyperparameters:**
```yaml
# configs/train/freeflow_base_libero.yaml
model:
  name: freeflow_base
  teacher_nfe: 10
  lambda_correction: 0.1

training:
  batch_size: 4
  learning_rate: 2.5e-5  # Same as SnapFlow
  warmup_steps: 500
  total_steps: 30000
  eval_every: 1000
  save_every: 5000

data:
  dataset_path: ../data/libero
  norm_stats: ../data/libero/norm_stats.json

freeze:
  - "*vit*"  # Freeze VLM backbone
trainable:
  - "*_1"    # Train action expert layers
  - "action_*"
  - "time_*"
```

### Phase 4: Evaluation Integration (Days 9-10)

**Tasks:**
1. Create evaluation script for FreeFlow
2. Integrate with unified `eval/` framework
3. Support 1-NFE inference mode
4. Add LIBERO-Plus evaluation support

**Key files:**
```
scripts/
├── eval_direct.py            # Direct evaluation
└── eval_freeflow.sh

eval/scripts/ (modifications)
├── run_eval.py               # Add freeflow model-type
└── eval_utils.py             # Add freeflow inference
```

**Evaluation modes:**
- `quick`: Fast test (libero_spatial, 5 ep)
- `preset`: Standard LIBERO (4 suites, 50 ep)
- `libero-plus`: Robustness eval (7 perturbations)

### Phase 5: Experimentation and Tuning (Days 11-15)

**Tasks:**
1. Train base FreeFlow model
2. Evaluate on LIBERO-Plus
3. Tune hyperparameters:
   - `lambda_correction` (error correction weight)
   - Learning rate
   - Training steps
4. Ablation studies:
   - With/without error correction
   - Different teacher NFE values
   - Different intermediate sampling strategies

**Ablation configs:**
```
configs/train/
├── freeflow_base.yaml         # Basic FreeFlow
├── freeflow_no_correction.yaml  # Without error correction
├── freeflow_nfe5_teacher.yaml   # Teacher with NFE=5
└── freeflow_full.yaml           # All variants
```

---

## Training Commands

### Initial Training
```bash
cd /root/autodl-tmp/freeflow

# Base config
bash scripts/train.sh configs/train/freeflow_base_libero.yaml

# Resume from checkpoint
bash scripts/train.sh configs/train/freeflow_base_libero.yaml \
    --resume checkpoints/finetuned/freeflow/step_10000
```

### Evaluation
```bash
# LIBERO standard
python scripts/eval_direct.py --preset preset --nfe 1

# LIBERO-Plus robustness
python scripts/eval_libero_plus.py --preset quick --nfe 1

# Via unified eval framework
cd ../eval/scripts
python run_eval.py --dataset libero-plus --mode quick --nfe 1 --model-type freeflow
```

---

## Expected Results

### Target Performance

| Model | NFE | LIBERO Spatial | LIBERO Object | LIBERO Goal | LIBERO-Plus |
|-------|-----|----------------|---------------|-------------|-------------|
| π₀.₅ (teacher) | 10 | ~85% | ~80% | ~75% | ~60% |
| FreeFlow (target) | 1 | ~80% | ~75% | ~70% | ~55% |
| Baseline | 1 | ~50% | ~45% | ~40% | ~30% |

### Key Metrics

1. **Success Rate**: Primary metric for LIBERO tasks
2. **Action MSE**: Measure of action quality
3. **Training Time**: Expected ~12h on single A800
4. **Inference Time**: 1-NFE = ~10x faster than 10-NFE

---

## Risks and Mitigations

### Risk 1: Data-Free Adaptation May Not Work for VLA

**Issue**: FreeFlow was designed for unconditional image generation. VLA models are conditioned on observations and tasks.

**Mitigation**:
- Start with conditional prior: p(z_1 | observation)
- Use observation as conditioning throughout
- Fall back to hybrid: data-free + small dataset subset

### Risk 2: Teacher-Student Mismatch

**Issue**: π₀.₅ was not specifically trained for flow matching (uses different paradigm).

**Mitigation**:
- Fine-tune teacher on LIBERO first (if needed)
- Use SnapFlow/SMF as teacher instead of base π₀.₅
- Ensemble multiple teachers

### Risk 3: Computational Cost

**Issue**: Teacher forward passes in every training step.

**Mitigation**:
- Cache teacher outputs
- Use mixed precision (bfloat16)
- Reduce batch size if needed

### Risk 4: LIBERO-Plus Robustness

**Issue**: Perturbations may expose weaknesses in 1-NFE models.

**Mitigation**:
- Add robustness augmentation during training
- Test on progressively harder perturbations
- Compare against SMF/SnapFlow baselines

---

## Comparison with Existing Methods

| Method | Data Required | Teacher | NFE | Key Innovation |
|--------|---------------|---------|-----|----------------|
| π₀.₅ baseline | Yes | None | 10 | Original multi-step |
| SMF | Yes | None | 1 | Self-consistency loss |
| SnapFlow | Yes | None | 1 | Self-distillation |
| **FreeFlow** | **No** | **π₀.₅** | **1** | **Data-free distillation** |

**FreeFlow advantages:**
1. No training data required for distillation
2. Can leverage better teachers as they become available
3. Theoretically more principled (avoids teacher-data mismatch)

**FreeFlow challenges:**
1. Requires good teacher model
2. May need task-specific adaptation
3. Higher computational cost (teacher forward passes)

---

## Timeline Summary

| Phase | Days | Deliverables |
|-------|------|--------------|
| 1. Setup | 1-2 | Directory structure, configs |
| 2. Model | 3-5 | Pi05FreeFlow, TeacherWrapper, loss |
| 3. Training | 6-8 | Trainer, scripts, infrastructure |
| 4. Evaluation | 9-10 | Eval scripts, integration |
| 5. Experiments | 11-15 | Trained models, ablation studies |

**Total**: ~15 days for full implementation and initial results

---

## References

1. **FreeFlow Paper**: [arXiv:2511.19428](https://arxiv.org/abs/2511.19428) - Flow Map Distillation Without Data
2. **FreeFlow GitHub**: [ShangyuanTong/FreeFlow](https://github.com/ShangyuanTong/FreeFlow) - Original implementation (PyTorch)
3. **π₀.₅ Paper**: [Physical Intelligence](https://www.physicalintelligence.company/) - Base VLA model
4. **SMF Implementation**: `/root/autodl-tmp/smfVLA/` - SplitMeanFlow baseline
5. **SnapFlow Implementation**: `/root/autodl-tmp/snapflow/` - SnapFlow baseline

---

## Next Steps

1. **Review and approve plan** with user
2. **Clone FreeFlow repo** for reference implementation
3. **Create directory structure** and begin Phase 1
4. **Implement core components** iteratively with testing
5. **Train initial model** and evaluate on LIBERO-Plus

**Questions for user:**
1. Should we use base π₀.₅ or SnapFlow/SMF as teacher?
2. Is pure data-free approach acceptable, or should we have hybrid fallback?
3. Priority on LIBERO standard vs LIBERO-Plus robustness?
4. Any specific hyperparameters to prioritize (batch size, training time)?

---

*Plan created: 2025-06-08*
*Based on FreeFlow paper and existing SMF/SnapFlow implementations*
