"""
Flow Matching Samplers for smfVLA.

This module implements various flow matching samplers optimized for
few-NFE (Number of Function Evaluations) inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from abc import ABC, abstractmethod


class FlowSampler(ABC):
    """Base class for flow matching samplers."""

    @abstractmethod
    def sample(
        self,
        velocity_fn: Callable,
        x_init: torch.Tensor,
        num_steps: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample from the flow matching model.

        Args:
            velocity_fn: Function that predicts velocity field
            x_init: Initial noise [B, T, D]
            num_steps: Number of integration steps

        Returns:
            Sampled trajectory [B, T, D]
        """
        pass


class EulerSampler(FlowSampler):
    """Standard Euler ODE solver for flow matching."""

    def sample(
        self,
        velocity_fn: Callable,
        x_init: torch.Tensor,
        num_steps: int,
        **kwargs,
    ) -> torch.Tensor:
        """Sample using Euler integration."""
        B = x_init.shape[0]
        device = x_init.device
        dt = 1.0 / num_steps
        x = x_init.clone()

        for i in range(num_steps):
            t = torch.full((B, 1), i * dt, device=device)
            velocity = velocity_fn(x, t)
            x = x + velocity * dt

        return x


class MidpointSampler(FlowSampler):
    """Midpoint method (2nd order Runge-Kutta) for better accuracy."""

    def sample(
        self,
        velocity_fn: Callable,
        x_init: torch.Tensor,
        num_steps: int,
        **kwargs,
    ) -> torch.Tensor:
        """Sample using midpoint method."""
        B = x_init.shape[0]
        device = x_init.device
        dt = 1.0 / num_steps
        x = x_init.clone()

        for i in range(num_steps):
            t = torch.full((B, 1), i * dt, device=device)

            # Half step
            v_half = velocity_fn(x, t)
            x_mid = x + v_half * (dt / 2)

            # Full step using midpoint velocity
            t_mid = torch.full((B, 1), (i + 0.5) * dt, device=device)
            v_mid = velocity_fn(x_mid, t_mid)
            x = x + v_mid * dt

        return x


class ConsistencySampler(FlowSampler):
    """
    Consistency sampler for 1-NFE inference.

    Uses a distilled consistency model to directly map noise to samples
    without iterative denoising.
    """

    def __init__(self, consistency_model: nn.Module):
        self.model = consistency_model

    def sample(
        self,
        velocity_fn: Callable,
        x_init: torch.Tensor,
        num_steps: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Sample using consistency model (1-NFE)."""
        B = x_init.shape[0]
        device = x_init.device

        if num_steps == 1:
            # Direct 1-NFE inference
            t = torch.ones(B, 1, device=device)
            velocity = velocity_fn(x_init, t)
            # Use consistency model for direct mapping
            actions = self.model(velocity)
            return actions
        else:
            # Fallback to Euler for multi-step
            sampler = EulerSampler()
            return sampler.sample(velocity_fn, x_init, num_steps)


class AdaptiveSampler(FlowSampler):
    """
    Adaptive sampler that adjusts step size based on velocity magnitude.

    Useful for handling varying difficulty across the action space.
    """

    def __init__(self, min_steps: int = 1, max_steps: int = 20, tolerance: float = 1e-3):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.tolerance = tolerance

    def sample(
        self,
        velocity_fn: Callable,
        x_init: torch.Tensor,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample with adaptive step sizing."""
        B = x_init.shape[0]
        device = x_init.device
        x = x_init.clone()
        t_current = 0.0

        steps_used = 0
        while t_current < 1.0:
            t = torch.full((B, 1), t_current, device=device)
            velocity = velocity_fn(x, t)

            # Estimate step size based on velocity magnitude
            velocity_norm = torch.norm(velocity, dim=-1, keepdim=True).mean()
            dt = min(self.tolerance / (velocity_norm + 1e-8), 1.0 - t_current)
            dt = max(dt, 1e-4)  # Minimum step size

            x = x + velocity * dt
            t_current += dt
            steps_used += 1

            if steps_used >= self.max_steps:
                break

        return x


def get_sampler(sampler_type: str, **kwargs) -> FlowSampler:
    """Factory function to get sampler by type."""
    samplers = {
        "euler": EulerSampler,
        "midpoint": MidpointSampler,
        "adaptive": AdaptiveSampler,
    }

    if sampler_type not in samplers:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Choose from {list(samplers.keys())}")

    return samplers[sampler_type](**kwargs)
