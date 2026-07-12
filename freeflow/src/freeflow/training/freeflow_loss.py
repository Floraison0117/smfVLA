"""
FreeFlow loss implementation.

Data-free distillation loss adapted from FreeFlow (arXiv:2511.19428)
for Vision-Language-Action models.

Algorithm:
1. Sample z_1 ~ N(0, I) from prior (data-free!)
2. Get teacher path: z_0^T = Euler(T_θ, z_1, num_steps=10)
3. Get student prediction: z_0^S = z_1 - S_φ(z_1, 0→1)
4. Path loss: ||z_0^S - z_0^T||²

5. Error correction at intermediate t:
   - Sample z_t from teacher path
   - Student predicts from z_t
   - Correction loss
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class FreeFlowSample(NamedTuple):
    """FreeFlow training sample."""
    z_1: at.Float[at.Array, "b ah ad"]       # Starting noise (t=1)
    observation: Any                           # VLA observation
    action_mean: at.Float[at.Array, " ad"]    # For normalization
    action_std: at.Float[at.Array, " ad"]     # For normalization
    t_correction: at.Float[at.Array, " b"]     # Intermediate time for correction
    use_correction: at.Bool[at.Array, " b"]   # Whether to apply correction


def sample_from_prior(
    rng: at.KeyArrayLike,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
) -> at.Float[at.Array, "b ah ad"]:
    """
    Sample pure noise from prior distribution (data-free!).

    This is the key innovation: we don't need data samples,
    just sample from N(0, I).

    Args:
        rng: JAX random key
        batch_size: Batch size
        action_horizon: Action horizon (default 1)
        action_dim: Action dimension

    Returns:
        z_1: Pure noise [B, action_horizon, action_dim]
    """
    return jax.random.normal(
        rng,
        (batch_size, action_horizon, action_dim)
    )


def sample_correction_time(
    rng: at.KeyArrayLike,
    batch_size: int,
    correction_prob: float = 0.5,
) -> tuple[
    at.Float[at.Array, " b"],
    at.Bool[at.Array, " b"],
]:
    """
    Sample intermediate time for error correction.

    With probability correction_prob, sample t ~ Uniform(0.1, 0.9).
    Otherwise, set use_correction=False.

    Args:
        rng: JAX random key
        batch_size: Batch size
        correction_prob: Probability of applying correction

    Returns:
        t_correction: Intermediate time
        use_correction: Whether to apply correction for each sample
    """
    rng_t, rng_m = jax.random.split(rng)

    # Sample intermediate time
    t_correction = jax.random.uniform(
        rng_t,
        (batch_size,),
        minval=0.1,
        maxval=0.9
    )

    # Bernoulli mask for correction
    use_correction = jax.random.bernoulli(
        rng_m,
        p=correction_prob,
        shape=(batch_size,)
    )

    return t_correction, use_correction


def interpolate_z(
    x_norm: at.Float[at.Array, "b ah ad"],
    noise: at.Float[at.Array, "b ah ad"],
    t: at.Float[at.Array, " b"],
) -> at.Float[at.Array, "b ah ad"]:
    """
    Linear interpolation: z_t = (1-t)·x_norm + t·noise.

    Uses same convention as π₀.₅:
    - t=0 is clean (x_0)
    - t=1 is pure noise (ε)

    Args:
        x_norm: Normalized clean action
        noise: Pure noise
        t: Time parameter

    Returns:
        z_t: Interpolated state
    """
    t_expanded = t[:, None, None]  # [B, 1, 1]
    z_t = (1 - t_expanded) * x_norm + t_expanded * noise
    return z_t


from freeflow.training.teacher_integration import teacher_euler_integration


def compute_freeflow_loss(
    student_fn: Any,
    student_params: Any,
    observation: Any,
    action_mean: at.Float[at.Array, " ad"],
    action_std: at.Float[at.Array, " ad"],
    rng: at.KeyArrayLike,
    batch_size: int,
    action_horizon: int = 1,
    action_dim: int = 32,
    teacher_nfe: int = 10,
    lambda_correction: float = 0.1,
    correction_prob: float = 0.5,
) -> tuple[
    at.Float[at.Array, ""],
    dict[str, at.Float[at.Array, ""]],
]:
    """
    Compute FreeFlow data-free distillation loss.

    Algorithm:
    1. Sample z_1 ~ N(0, I) from prior (data-free!)
    2. Get teacher's multi-step integration path: z_0^T = Teacher.sample_actions(z_1, num_steps=teacher_nfe)
    3. Get student prediction: z_0^S = z_1 - Student(z_1, t=1, r=0)
    4. Path loss: ||z_0^S - z_0^T||²

    5. Error correction:
       - Sample intermediate time t
       - Get z_t from teacher's path
       - Student predicts from z_t: z_0^S(t) = z_t - Student(z_t, t, 0)
       - Correction loss: ||z_0^S(t) - z_0^T||²

    Args:
        student_fn: Student model function
        student_params: Student parameters (receives gradients)
        observation: VLA observation (images, state, prompt)
        action_mean: Action mean for normalization
        action_std: Action std for normalization
        rng: JAX random key
        batch_size: Batch size
        action_horizon: Action horizon (default 1)
        action_dim: Action dimension (default 32)
        teacher_nfe: Teacher's NFE (default 10)
        lambda_correction: Error correction weight
        correction_prob: Probability of applying correction

    Returns:
        loss_total: Total loss
        metrics: Dictionary of loss components

    Note:
        Teacher model must be set via
        freeflow.training.teacher_integration.set_teacher() before calling.
    """
    rng_noise, rng_teacher, rng_correction = jax.random.split(rng, 3)

    # Step 1: Sample from prior (data-free!)
    z_1 = sample_from_prior(
        rng_noise,
        batch_size,
        action_horizon,
        action_dim
    )

    # Step 2: Get teacher's multi-step integration path
    # Teacher is frozen, so no gradients flow through it
    # Teacher π₀.₅ operates in normalized space (trained on normalized actions),
    # so its output is already normalized
    z_0_teacher_norm = teacher_euler_integration(
        observation=observation,
        z_1=z_1,
        num_steps=teacher_nfe,
        rng=rng_teacher,
    )

    # Step 3: Student's 1-step prediction (from z_1 to z_0)
    # For 1-NFE: r=0, t=1 means "velocity from t=1 to t=0"
    t_one = jnp.ones((batch_size,))
    r_zero = jnp.zeros((batch_size,))

    v_student = student_fn(
        student_params,
        observation,
        z_1,  # Student sees the noise
        t_one,
        r_zero,  # r parameter (reference time, 0 for 1-NFE)
    )

    # Student's predicted clean action: z_0^S = z_1 - v_student
    z_0_student = z_1 - v_student

    # Step 4: Path loss (main distillation objective)
    loss_path = jnp.mean(jnp.square(z_0_student - z_0_teacher_norm))

    # Step 5: Error correction (optional but recommended)
    # Sample intermediate time and apply correction loss
    t_correction, use_correction = sample_correction_time(
        rng_correction,
        batch_size,
        correction_prob
    )

    # Get intermediate state from teacher
    # We need to sample z_t at t_correction along teacher's path
    # For simplicity, use linear interpolation from teacher's path
    # In full version, would query teacher.get_intermediate_states()
    z_t_teacher_norm = (1 - t_correction[:, None, None]) * z_0_teacher_norm + t_correction[:, None, None] * z_1

    # Student's correction from intermediate state
    v_correction = student_fn(
        student_params,
        observation,
        z_t_teacher_norm,
        t_correction,
        r_zero,  # r parameter (reference time, 0 for 1-NFE)
    )

    z_0_from_t = z_t_teacher_norm - v_correction

    # Correction loss
    loss_correction_raw = jnp.mean(jnp.square(z_0_from_t - z_0_teacher_norm))

    # Apply correction mask
    loss_correction = jnp.where(
        use_correction.any(),
        loss_correction_raw,
        jnp.array(0.0)
    )

    # Total loss
    loss_total = loss_path + lambda_correction * loss_correction

    # Metrics
    metrics = {
        "loss_total": loss_total,
        "loss_path": loss_path,
        "loss_correction": loss_correction,
        "z_0_norm": jnp.linalg.norm(z_0_teacher_norm),
        "v_student_norm": jnp.linalg.norm(v_student),
    }

    return loss_total, metrics


def compute_freeflow_loss_with_data(
    student_fn: Any,
    student_params: Any,
    observation: Any,
    actions: at.Float[at.Array, "b ah ad"],
    action_mean: at.Float[at.Array, " ad"],
    action_std: at.Float[at.Array, " ad"],
    rng: at.KeyArrayLike,
    teacher_nfe: int = 10,
    lambda_correction: float = 0.1,
    correction_prob: float = 0.5,
) -> tuple[
    at.Float[at.Array, ""],
    dict[str, at.Float[at.Array, ""]],
]:
    """
    Alternative: Compute FreeFlow loss with ground-truth actions for observation.

    Still data-free in the sense that we don't use action labels for the loss,
    but we use the observation tokens from the dataset.

    Args:
        student_fn: Student model function
        student_params: Student parameters
        observation: VLA observation (from dataset)
        actions: Ground-truth actions (only used for shape/normalization)
        action_mean: Action mean
        action_std: Action std
        rng: JAX random key
        teacher_nfe: Teacher NFE
        lambda_correction: Correction weight
        correction_prob: Correction probability

    Returns:
        loss_total, metrics
    """
    batch_size = actions.shape[0]
    action_horizon = actions.shape[1]
    # Use the actual dimension of the actions array (already padded to target_dim)
    action_dim = actions.shape[2]

    # Normalize actions for reference
    x_norm = (actions - action_mean[None, None, :]) / (action_std[None, None, :] + 1e-8)

    return compute_freeflow_loss(
        student_fn=student_fn,
        student_params=student_params,
        observation=observation,
        action_mean=action_mean,
        action_std=action_std,
        rng=rng,
        batch_size=batch_size,
        action_horizon=action_horizon,
        action_dim=action_dim,
        teacher_nfe=teacher_nfe,
        lambda_correction=lambda_correction,
        correction_prob=correction_prob,
    )
