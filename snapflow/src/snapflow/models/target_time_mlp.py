"""
Target-Time Embedding for SnapFlow.

Zero-initialized 2-layer MLP that encodes the target time s.
This allows the network to distinguish between:
- FM samples (s=t, local velocity estimation)
- Consistency samples (s=0, global one-step generation)

Zero initialization ensures the network starts at teacher behavior (step 0).
"""

import flax.nnx as nnx
import jax.numpy as jnp
from typing_extensions import override

from openpi.shared import array_typing as at


class TargetTimeMLP(nnx.Module):
    """
    Zero-initialized 2-layer MLP encoding target time s.

    From paper Section 3.5:
    "A zero-initialized two-layer MLP that encodes s and adds to the
    existing time embedding before each transformer block.
    Zero initialization preserves the teacher at step 0."

    Architecture (from paper):
    - Input: s (scalar time per sample)
    - Layer 1: Linear(1, width) + Swish (scalar to hidden dimension)
    - Layer 2: Linear(width, width)
    - Initialization: All weights and biases = 0

    Args:
        width: Hidden dimension (matches action expert width)
        rngs: Random number generators
    """

    def __init__(self, width: int, rngs: nnx.Rngs):
        # Layer 1: 1 -> width (scalar s to hidden dimension)
        self.layer1 = nnx.Linear(
            in_features=1,
            out_features=width,
            rngs=rngs,
            kernel_init=_zero_init,
            bias_init=_zero_init,
        )
        # Layer 2: width -> width
        self.layer2 = nnx.Linear(
            in_features=width,
            out_features=width,
            rngs=rngs,
            kernel_init=_zero_init,
            bias_init=_zero_init,
        )

    @at.typecheck
    def __call__(
        self,
        s: at.Float[at.Array, " b"],
    ) -> at.Float[at.Array, "b width"]:
        """
        Forward pass: encode target time s.

        Args:
            s: Target time [B,], where s ∈ [0, 1]
              - s=t for FM samples (local velocity)
              - s=0 for consistency samples (global one-step)

        Returns:
            Zero-initialized embedding [B, width]
            (Initially all zeros, learns during training)
        """
        # Expand s to match width (broadcast in first layer)
        # s: [B,] -> [B, width]
        x = s[:, None]  # No actual encoding yet, just dimension expansion

        # Layer 1: width -> width (zero-initialized, output is zeros)
        x = self.layer1(x)
        x = nnx.swish(x)  # Swish(0) = 0

        # Layer 2: width -> width (zero-initialized, output is zeros)
        x = self.layer2(x)

        return x


def _zero_init(key, shape, dtype=jnp.float32):
    """Zero initialization for all parameters."""
    return jnp.zeros(shape, dtype=dtype)
