"""
DMF (Decoupled MeanFlow) sampler for inference.

Provides Euler sampler for multi-step generation and 1-NFE direct inference.

Based on the paper "Decoupled MeanFlow: Turning Flow Models into Flow Maps
for Accelerated Sampling" (ICLR 2026).

References:
- Paper: https://arxiv.org/abs/2510.24474
- Official repo: https://github.com/kyungmnlee/dmf (samplers.py)

Note: The primary sampling path used by eval is Pi05DMF.sample_actions() in
pi05_dmf.py. This module provides standalone samplers for testing/debugging.

Time convention: t=1 is noise, t=0 is data (matches pi0.5).
- Encoder layers conditioned on E(t) (current time)
- Decoder layers conditioned on E(r) (target time)
- Euler: x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_{i}) * u(x_{t_i}, t_i, t_{i+1})
- 1-NFE: x_0 = x_1 - u(x_1, t=1, r=0)
"""

import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at
from typing import Callable, Any


def get_timesteps(
    num_steps: int,
    t_start: float = 1.0,
    t_end: float = 0.0,
    shift: float = 1.0,
) -> at.Float[at.Array, " n+1"]:
    """
    Get timestep schedule for Euler integration.

    Args:
        num_steps: Number of steps
        t_start: Start time (usually 1.0 = noise)
        t_end: End time (usually 0.0 = data)
        shift: Time shift parameter (shift=1.5 for ImageNet 512)

    Returns:
        timesteps: Array of length num_steps+1
    """
    import numpy as np

    t_steps = np.linspace(t_start, t_end, num_steps + 1, dtype=np.float64)

    if shift == 1.0:
        return jnp.array(t_steps)

    # Apply shift: t' = shift * t / (1 + (shift - 1) * t)
    t_shifted = shift * t_steps / (1 + (shift - 1.0) * t_steps)
    return jnp.array(t_shifted)


@at.jit
def euler_sampler(
    forward_fn: Callable,
    observation: Any,
    noise: at.Float[at.Array, "b ah ad"],
    num_steps: int = 1,
    shift: float = 1.0,
) -> at.Float[at.Array, "b ah ad"]:
    """
    Euler sampler for DMF inference.

    x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_{i}) * u(x_{t_i}, t_i, t_{i+1})

    Args:
        forward_fn: Model forward function f(obs, x, t, r) -> flow map velocity
            where t = current time (encoder cond), r = target time (decoder cond)
        observation: Observation data
        noise: Initial noise x_1 ~ N(0, I)
        num_steps: Number of integration steps
        shift: Time shift parameter

    Returns:
        actions: Predicted actions at t=0
    """
    batch_size = noise.shape[0]
    t_steps = get_timesteps(num_steps, shift=shift)

    x = noise.astype(jnp.float64)

    def step(carry, i):
        """Single Euler step."""
        (x_cur,) = carry
        t_cur = t_steps[i]
        t_nxt = t_steps[i + 1]

        # Predict flow map: u(x_cur, t_cur, t_nxt)
        # t_cur = current time (encoder), t_nxt = target time (decoder)
        u = forward_fn(observation, x_cur.astype(jnp.float32),
                       jnp.full((batch_size,), t_cur),
                       jnp.full((batch_size,), t_nxt))

        # Euler step: x_{n+1} = x_n + (t_{n+1} - t_n) * u
        x_nxt = x_cur + (t_nxt - t_cur) * u.astype(jnp.float64)

        return (x_nxt,), None

    # Scan over steps
    (x_final,), _ = jax.lax.scan(step, (x,), jnp.arange(num_steps))

    return x_final[0].astype(jnp.float32)


@at.jit
def euler_sampler_multi_step(
    forward_fn: Callable,
    observation: Any,
    noise: at.Float[at.Array, "b ah ad"],
    num_steps: int = 4,
    shift: float = 1.0,
) -> at.Float[at.Array, "b ah ad"]:
    """
    Multi-step Euler sampler (alternative implementation with explicit loop).

    x_{t_{i+1}} = x_{t_i} + (t_{i+1} - t_{i}) * u(x_{t_i}, t_i, t_{i+1})

    Args:
        forward_fn: Model forward function f(obs, x, t, r) -> flow map velocity
        observation: Observation data
        noise: Initial noise x_1 ~ N(0, I)
        num_steps: Number of integration steps
        shift: Time shift parameter

    Returns:
        actions: Predicted actions at t=0
    """
    batch_size = noise.shape[0]
    t_steps = get_timesteps(num_steps, shift=shift)

    x = noise.astype(jnp.float64)

    for i in range(num_steps):
        t_cur = t_steps[i]
        t_nxt = t_steps[i + 1]

        # Predict flow map: u(x, t_cur, t_nxt)
        u = forward_fn(observation, x.astype(jnp.float32),
                       jnp.full((batch_size,), t_cur),
                       jnp.full((batch_size,), t_nxt))

        # Euler step: x_{n+1} = x_n + (t_{n+1} - t_n) * u
        x = x + (t_nxt - t_cur) * u.astype(jnp.float64)

    return x.astype(jnp.float32)


@at.jit
def compute_1nfe(
    forward_fn: Callable,
    observation: Any,
    noise: at.Float[at.Array, "b ah ad"],
) -> at.Float[at.Array, "b ah ad"]:
    """
    1-NFE direct inference: x_0 = x_1 - u(x_1, t=1, r=0).

    t=1 (noise time, encoder cond), r=0 (data time, decoder cond).
    This is equivalent to euler_sampler with num_steps=1, t_start=1.0, t_end=0.0.

    Args:
        forward_fn: Model forward function f(obs, x, t, r) -> flow map velocity
        observation: Observation data
        noise: Initial noise x_1 ~ N(0, I)

    Returns:
        actions: Predicted actions at t=0
    """
    batch_size = noise.shape[0]

    # u = u(x_1, t=1, r=0) — average velocity from t=1 to r=0
    u = forward_fn(observation, noise,
                   jnp.ones((batch_size,)),   # t = 1 (current/noise time, encoder cond)
                   jnp.zeros((batch_size,)))  # r = 0 (target/data time, decoder cond)

    # x_0 = x_1 - u (since u goes from t=1 to r=0, x_0 = x_1 + (0-1)*u = x_1 - u)
    actions = noise - u

    return actions
