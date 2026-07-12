"""
Teacher integration utilities for FreeFlow.

Provides JIT-compiled Euler integration that correctly implements
10-NFE sampling from t=1 to t=0.
"""

import jax
import jax.numpy as jnp


# Global reference to teacher model
_TEACHER_MODEL = None


def set_teacher(teacher_model):
    """Set global teacher model for JIT-compiled integration.

    Args:
        teacher_model: A true π₀.₅ model (Pi0, not FreeFlow) with sample_actions method.
    """
    global _TEACHER_MODEL
    _TEACHER_MODEL = teacher_model


def _preprocess_observation(obs):
    """
    Preprocess observation to have consistent dtypes for scan loop.

    Converts uint8 images to float32 to match the Observation.from_dict conversion.
    This ensures the scan loop has consistent input/output dtypes.
    """
    if isinstance(obs, dict):
        # Convert uint8 images to float32 (same as Observation.from_dict does)
        obs = obs.copy()  # Avoid modifying the original
        for key in obs.get("image", {}):
            if obs["image"][key].dtype == jnp.uint8:
                obs["image"][key] = obs["image"][key].astype(jnp.float32) / 255.0 * 2.0 - 1.0
    return obs


def teacher_euler_integration(
    observation: any,
    z_1: jax.Array,
    num_steps: int,
    rng: jax.Array,
) -> jax.Array:
    """
    Teacher Euler integration from z_1 (t=1) to z_0 (t=0).

    This implements the standard 10-NFE Euler integration used by π₀.₅:
    - Start at t=1 (pure noise)
    - Each step: z_{t+dt} = z_t + dt * v_t, where dt = -1.0 / num_steps
    - After num_steps steps, reach t=0 (clean action)

    Args:
        observation: VLA observation
        z_1: Starting noise at t=1
        num_steps: Number of integration steps (static, not traced)
        rng: Random key for noise sampling

    Returns:
        z_0: Integrated action at t=0
    """
    global _TEACHER_MODEL

    if _TEACHER_MODEL is None:
        raise ValueError("Teacher not set. Call set_teacher() first.")

    # Preprocess observation to have consistent dtypes
    obs = _preprocess_observation(observation)

    # Convert observation to proper format if needed
    from openpi.models.model import Observation
    if not isinstance(obs, Observation):
        obs = Observation.from_dict(obs)

    # Use teacher's sample_actions for proper NFE integration
    # This handles the full prefix+suffix computation with KV cache
    z_0 = _TEACHER_MODEL.sample_actions(rng, obs, num_steps=num_steps, noise=z_1)

    return z_0


def teacher_euler_integration_slow(
    observation: any,
    z_1: jax.Array,
    num_steps: int,
    rng: jax.Array,
) -> jax.Array:
    """
    Slow version of teacher Euler integration for debugging.

    This explicitly implements the Euler loop without using sample_actions.
    Use this to verify correctness.

    Args:
        observation: VLA observation
        z_1: Starting noise at t=1
        num_steps: Number of integration steps
        rng: Random key

    Returns:
        z_0: Integrated action at t=0
    """
    global _TEACHER_MODEL

    if _TEACHER_MODEL is None:
        raise ValueError("Teacher not set. Call set_teacher() first.")

    obs = _preprocess_observation(observation)

    from openpi.models.model import Observation
    if not isinstance(obs, Observation):
        obs = Observation.from_dict(obs)

    # Euler integration loop
    z_t = z_1
    t = 1.0
    dt = -1.0 / num_steps  # For num_steps=10, dt = -0.1

    # Pre-compute prefix tokens and KV cache (optimization)
    prefix_tokens, prefix_mask, prefix_ar_mask = _TEACHER_MODEL.embed_prefix(obs)
    from openpi.models.pi0 import make_attn_mask
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = _TEACHER_MODEL.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )

    batch_size = z_1.shape[0]
    einops = __import__("einops")

    for step in range(num_steps):
        # Get suffix tokens at current time t
        t_batch = jnp.full((batch_size,), t)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = _TEACHER_MODEL.embed_suffix(
            obs, z_t, t_batch
        )

        # Build attention masks
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_step = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
        positions_step = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        # Get velocity from teacher
        (prefix_out, suffix_out), _ = _TEACHER_MODEL.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions_step,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = _TEACHER_MODEL.action_out_proj(suffix_out[:, -_TEACHER_MODEL.action_horizon :])

        # Euler step: z_{t+dt} = z_t + dt * v_t
        z_t = z_t + dt * v_t
        t = t + dt  # t goes from 1.0 to 0.0

    return z_t  # z_0
