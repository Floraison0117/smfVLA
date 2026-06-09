"""
Teacher integration utilities for FreeFlow.

Provides JIT-compiled Euler integration that handles the teacher function
properly by closing over it at module level.
"""

import jax
import jax.numpy as jnp


# Global reference to teacher function (set during training setup)
_TEACHER_FN = None
_TEACHER_PARAMS = None


def set_teacher(teacher_fn, teacher_params):
    """Set global teacher reference for JIT-compiled integration."""
    global _TEACHER_FN, _TEACHER_PARAMS
    _TEACHER_FN = teacher_fn
    _TEACHER_PARAMS = teacher_params


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
) -> jax.Array:
    """
    Teacher Euler integration from z_1 (t=1) to z_0 (t=0).

    Uses the global teacher function and parameters set via set_teacher().

    Args:
        observation: VLA observation
        z_1: Starting noise at t=1
        num_steps: Number of integration steps (static, not traced)

    Returns:
        z_0: Integrated action at t=0
    """
    global _TEACHER_FN, _TEACHER_PARAMS

    if _TEACHER_FN is None or _TEACHER_PARAMS is None:
        raise ValueError("Teacher not set. Call set_teacher() first.")

    # Preprocess observation to have consistent dtypes
    obs = _preprocess_observation(observation)

    # Define the integration step function
    def integrate_step(z_t, _):
        """Single Euler integration step.

        Args:
            z_t: Current noisy action state
            _: Dummy carry (unused)

        Returns:
            (z_new, dummy_carry): Next state and dummy carry
        """
        t = jnp.ones((z_t.shape[0],)) * 1.0  # Teacher always uses r=t=1

        # Get velocity from teacher (stop gradient!)
        v_t = _TEACHER_FN(_TEACHER_PARAMS, obs, z_t, t, t)
        v_t = jax.lax.stop_gradient(v_t)

        # Euler step: z_{t-dt} = z_t - dt * v_t
        dt = 1.0 / num_steps
        z_new = z_t - dt * v_t

        return z_new, None  # Return dummy carry to avoid carrying obs through scan

    # Run integration loop
    # Use scan with dummy carry to avoid complex observation structure tracing
    z_0, _ = jax.lax.scan(
        integrate_step,
        z_1,  # Initial state
        None,  # Dummy input
        length=num_steps,
    )

    return z_0
