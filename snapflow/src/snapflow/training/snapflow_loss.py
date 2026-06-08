"""
SnapFlow loss implementation.

Implements the progressive self-distillation loss from:
https://arxiv.org/abs/2604.05656

Loss = α·L_FM + (1-α)·λ·L_shortcut

Where:
- L_FM: Standard flow matching at random time t
- L_shortcut: Consistency loss with 2-step Euler shortcut target

Algorithm 1 from paper:
1. FM component (probability α): L_FM at random t
2. Consistency component (probability 1-α):
   - v_1 = sg(F_θ(x_1, 1, 1))
   - x_0.5 = x_1 - 0.5 · v_1
   - v_0.5 = sg(F_θ(x_0.5, 0.5, 0.5))
   - v_target = 0.5 · (v_1 + v_0.5)
   - L_shortcut = ||F_θ(x_1, 0, 1) - v_target||²
"""

from typing import Any

import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


def interpolate_z(
    x_norm: at.Float[at.Array, "b ah ad"],
    noise: at.Float[at.Array, "b ah ad"],
    t: at.Float[at.Array, " b"],
) -> at.Float[at.Array, "b ah ad"]:
    """
    Linear interpolation: z_t = (1-t)·x_norm + t·noise.

    Uses the same convention as π0.5:
    - t=0 is clean (x_0)
    - t=1 is pure noise (ε)
    """
    t_expanded = t[:, None, None]  # [B, 1, 1]
    z_t = (1 - t_expanded) * x_norm + t_expanded * noise
    return z_t


def compute_snapflow_loss(
    model_fn: Any,
    params: Any,
    observation: Any,
    actions: at.Float[at.Array, "b ah ad"],
    action_mean: at.Float[at.Array, " ad"],
    action_std: at.Float[at.Array, " ad"],
    rng: at.KeyArrayLike,
    alpha: float = 0.5,
    lambda_consistency: float = 0.1,
) -> tuple[at.Float[at.Array, ""], dict[str, at.Float[at.Array, ""]]]:
    """
    Compute SnapFlow loss with progressive FM/consistency mixing.

    Each training step uses THREE forward passes:
    1. FM: F_θ(x_t, t, t) at random t (for L_FM)
    2. Shortcut v1: F_θ(x_1, 1, 1) (for v_1 in shortcut target)
    3. Shortcut v0.5: F_θ(x_0.5, 0.5, 0.5) (for v_0.5 in shortcut target)
    4. Prediction: F_θ(x_1, 0, 1) (receives gradients)

    Args:
        model_fn: Model function f(params, obs, noisy_actions, r, t, s) → velocity
        params: Model parameters (None for NNX modules)
        observation: Observation data
        actions: Ground-truth action chunk [B, action_horizon, action_dim]
        action_mean: Action mean (for normalization)
        action_std: Action std (for normalization)
        rng: JAX random key
        alpha: FM/Consistency mixing ratio (default 0.5)
        lambda_consistency: Consistency loss weight (default 0.1)

    Returns:
        loss_total, metrics_dict
    """
    rng_fm, rng_noise = jax.random.split(rng)
    batch_size = actions.shape[0]

    # Step 1: Normalize actions
    x_norm = (actions - action_mean) / (action_std + 1e-8)

    # Step 2: Sample noise ε ~ N(0, I)
    noise = jax.random.normal(rng_noise, x_norm.shape)

    # Step 3: Sample random time t for FM component
    t_fm = jax.random.uniform(rng_fm, (batch_size,), minval=0.0, maxval=1.0)
    z_t_fm = interpolate_z(x_norm, noise, t_fm)

    # ── FM Component (L_FM) ─────────────────────────────────────
    # v_fm = F_θ(x_t, t, t) should equal ε - x_norm
    # Note: s=t for FM (local velocity estimation)
    v_fm = model_fn(params, observation, z_t_fm, t_fm, t_fm, t_fm)
    target_fm = noise - x_norm
    loss_fm = jnp.mean(jnp.square(v_fm - target_fm))

    # ── Consistency Component (L_shortcut) ───────────────────────
    # From paper Algorithm 1, lines 9-13

    # 1. v_1 = sg(F_θ(x_1, 1, 1)) - velocity at t=1
    #    Note: s=t=1 (both equal)
    t_ones = jnp.ones(batch_size)
    v_1 = model_fn(params, observation, noise, t_ones, t_ones, t_ones)
    v_1_sg = jax.lax.stop_gradient(v_1)

    # 2. x_0.5 = x_1 - 0.5 · v_1 - midpoint via Euler
    x_0_5 = noise - 0.5 * v_1_sg

    # 3. v_0.5 = sg(F_θ(x_0.5, 0.5, 0.5)) - velocity at t=0.5
    t_half = jnp.full((batch_size,), 0.5)
    v_0_5 = model_fn(params, observation, x_0_5, t_half, t_half, t_half)
    v_0_5_sg = jax.lax.stop_gradient(v_0_5)

    # 4. v_target = 0.5 · (v_1 + v_0.5) - 2-step average velocity
    v_target = 0.5 * (v_1_sg + v_0_5_sg)

    # 5. L_shortcut = ||F_θ(x_1, 0, 1) - v_target||²
    #    Note: s=0 (target time), t=1 (current time)
    r_zeros = jnp.zeros(batch_size)
    t_ones_for_pred = jnp.ones(batch_size)
    v_pred = model_fn(params, observation, noise, r_zeros, t_ones_for_pred, r_zeros)
    loss_shortcut = jnp.mean(jnp.square(v_pred - v_target))

    # ── Total Loss ─────────────────────────────────────────────
    # L = α·L_FM + (1-α)·λ·L_shortcut
    loss_total = alpha * loss_fm + (1 - alpha) * lambda_consistency * loss_shortcut

    metrics = {
        "loss_total": loss_total,
        "loss_fm": loss_fm,
        "loss_shortcut": loss_shortcut,
        "alpha": alpha,
        "lambda": lambda_consistency,
    }

    return loss_total, metrics


def compute_1nfe_actions(
    model_fn: Any,
    params: Any,
    observation: Any,
    noise: at.Float[at.Array, "b ah ad"],
    action_mean: at.Float[at.Array, " ad"],
    action_std: at.Float[at.Array, " ad"],
) -> at.Float[at.Array, "b ah ad"]:
    """
    1-NFE inference: x_0 = x_1 - F_θ(x_1, 0, 1).

    Key difference from baseline: s=0 (not s=t) for one-step generation.

    Args:
        model_fn: Model function
        params: Model parameters
        observation: Observation data
        noise: Initial noise z_1 ~ N(0, I)
        action_mean: Action mean (for denormalization)
        action_std: Action std (for denormalization)

    Returns:
        Predicted action chunk [B, action_horizon, action_dim]
    """
    batch_size = noise.shape[0]

    # F_θ(x_1, 0, 1) - from t=1 to s=0
    r = jnp.zeros(batch_size)  # s=0
    t = jnp.ones(batch_size)   # t=1
    v = model_fn(params, observation, noise, r, t, r)

    # x_0 = x_1 - v
    actions_norm = noise - v

    # Denormalize
    actions = actions_norm * (action_std + 1e-8) + action_mean

    return actions
