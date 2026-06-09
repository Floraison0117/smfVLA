"""
Teacher wrapper for frozen π₀.₅ model.

Provides multi-step Euler integration and intermediate state queries
without gradient computation.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class TeacherWrapper:
    """
    Wrapper for frozen teacher model (π₀.₅ base).

    Provides:
    - Multi-step Euler integration from z_1 to z_0
    - Intermediate state queries
    - No gradient computation (stop_gradient)

    Args:
        teacher_fn: Teacher model function (params, obs, z, r, t, s) -> velocity
        teacher_params: Frozen teacher parameters
        num_steps: Number of integration steps (default 10 for π₀.₅)
    """

    def __init__(
        self,
        teacher_fn: Any,
        teacher_params: Any,
        num_steps: int = 10,
    ):
        self.teacher_fn = teacher_fn
        self.teacher_params = teacher_params
        self.num_steps = num_steps

    @jax.jit
    def integrate(
        self,
        observation: Any,
        z_1: at.Float[at.Array, "b ah ad"],
        action_mean: at.Float[at.Array, " ad"],
        action_std: at.Float[at.Array, " ad"],
    ) -> at.Float[at.Array, "b ah ad"]:
        """
        Euler integration from z_1 (t=1) to z_0 (t=0).

        This is the teacher's multi-step sampling path that the student
        will learn to mimic in a single step.

        Args:
            observation: VLA observation (images + state + prompt)
            z_1: Starting noise at t=1
            action_mean: Action mean for denormalization
            action_std: Action std for denormalization

        Returns:
            z_0: Clean action at t=0 (teacher's prediction)
        """
        dt = 1.0 / self.num_steps
        z_t = z_1

        def integrate_step(carry, _):
            """Single Euler integration step."""
            z_t, obs, mean, std = carry
            t = jnp.ones((z_t.shape[0],)) * 1.0  # Teacher always uses r=t=1

            # Get velocity from teacher (stop gradient!)
            v_t = self.teacher_fn(self.teacher_params, obs, z_t, t, t, t)
            v_t = jax.lax.stop_gradient(v_t)

            # Euler step: z_{t-dt} = z_t - dt * v_t
            z_new = z_t - dt * v_t

            return (z_new, obs, mean, std), z_new

        # Run integration loop
        _, z_0 = jax.lax.scan(
            integrate_step,
            (z_t, observation, action_mean, action_std),
            None,
            length=self.num_steps,
        )

        # Denormalize to action space
        actions = z_0 * (action_std[None, None, :] + 1e-8) + action_mean[None, None, :]

        return actions

    @jax.jit
    def get_intermediate_states(
        self,
        observation: Any,
        z_1: at.Float[at.Array, "b ah ad"],
        t_targets: at.Float[at.Array, "n"],
        action_mean: at.Float[at.Array, " ad"],
        action_std: at.Float[at.Array, " ad"],
    ) -> at.Float[at.Array, "n b ah ad"]:
        """
        Get intermediate states along teacher's integration path.

        For error correction loss at specific time points.

        Args:
            observation: VLA observation
            z_1: Starting noise at t=1
            t_targets: Target time points (0 = start, 1 = end)
            action_mean: Action mean
            action_std: Action std

        Returns:
            z_t: States at target time points [n, b, ah, ad]
        """
        dt = 1.0 / self.num_steps

        # Sort targets (descending from t=1 to t=0)
        sorted_indices = jnp.argsort(t_targets)[::-1]
        sorted_t_targets = t_targets[sorted_indices]

        def scan_fn(carry, t_target):
            """Scan through integration path, collecting states."""
            z_t, obs, mean, std, current_t = carry

            # Integrate until we reach t_target
            steps_needed = int(jnp.round((current_t - t_target) / dt))

            def integrate_step(z_inner, _):
                t = jnp.ones((z_inner.shape[0],)) * 1.0
                v = self.teacher_fn(self.teacher_params, obs, z_inner, t, t, t)
                v = jax.lax.stop_gradient(v)
                z_new = z_inner - dt * v
                return z_new, None

            z_at_target, _ = jax.lax.scan(
                integrate_step,
                z_t,
                None,
                length=steps_needed,
            )

            new_carry = (z_at_target, obs, mean, std, t_target)
            return new_carry, z_at_target

        # Scan through targets
        _, states = jax.lax.scan(
            scan_fn,
            (z_1, observation, action_mean, action_std, 1.0),
            sorted_t_targets,
        )

        # Reorder to match original t_targets order
        states = states[sorted_indices.argsort()]

        return states

    @jax.jit
    def get_velocity_path(
        self,
        observation: Any,
        z_1: at.Float[at.Array, "b ah ad"],
        action_mean: at.Float[at.Array, " ad"],
        action_std: at.Float[at.Array, " ad"],
    ) -> tuple[
        at.Float[at.Array, "steps b ah ad"],  # states
        at.Float[at.Array, "steps b ah ad"],  # velocities
    ]:
        """
        Get full state and velocity path from teacher.

        Returns all intermediate states and velocities along the
        integration path for visualization and analysis.

        Args:
            observation: VLA observation
            z_1: Starting noise
            action_mean: Action mean
            action_std: Action std

        Returns:
            states: All states along path [steps, b, ah, ad]
            velocities: All velocities along path [steps, b, ah, ad]
        """
        dt = 1.0 / self.num_steps

        def scan_fn(carry, _):
            """Single integration step, returning state and velocity."""
            z_t, obs, mean, std = carry
            t = jnp.ones((z_t.shape[0],)) * 1.0

            v_t = self.teacher_fn(self.teacher_params, obs, z_t, t, t, t)
            v_t = jax.lax.stop_gradient(v_t)

            z_new = z_t - dt * v_t

            return (z_new, obs, mean, std), (z_t, v_t)

        _, states_and_velocities = jax.lax.scan(
            scan_fn,
            (z_1, observation, action_mean, action_std),
            None,
            length=self.num_steps,
        )

        states = states_and_velocities[0]
        velocities = states_and_velocities[1]

        return states, velocities
