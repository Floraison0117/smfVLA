"""
Few-NFE Action Head for smfVLA.

This module implements a re-trained action head that can generate high-quality
actions with fewer denoising steps (NFE) compared to the original flow matching
approach in openpi/pi0.5.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ActionHead(nn.Module):
    """
    Action head for few-NFE action generation.

    This replaces the original action expert in pi0.5 with a distilled version
    that can produce good actions in 1 or few NFE steps.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim

        # Action embedding
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer layers for denoising
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the velocity field for flow matching.

        Args:
            noisy_actions: Noisy action trajectory [B, T, action_dim]
            timesteps: Diffusion timesteps [B, 1]
            context: VLM features [B, seq_len, hidden_dim]

        Returns:
            Predicted velocity field [B, T, action_dim]
        """
        B, T, _ = noisy_actions.shape

        # Embed actions and timesteps
        action_emb = self.action_embed(noisy_actions)  # [B, T, hidden_dim]
        time_emb = self.time_embed(timesteps).unsqueeze(1)  # [B, 1, hidden_dim]

        # Combine with timestep
        x = action_emb + time_emb

        # Concatenate with context for cross-attention
        # Here we use self-attention; cross-attention with context can be added
        x = self.transformer(x)

        # Project to action space
        velocity = self.output_proj(x)  # [B, T, action_dim]

        return velocity


class DistilledActionHead(ActionHead):
    """
    Distilled action head for 1-NFE inference.

    Uses consistency distillation to learn a direct mapping from noise to actions,
    bypassing the iterative ODE solving process.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        distillation_steps: int = 1,
    ):
        super().__init__(
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.distillation_steps = distillation_steps

        # Additional consistency head for direct noise-to-action mapping
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, action_dim),
        )

    def sample_actions(
        self,
        context: torch.Tensor,
        num_steps: int = 1,
        action_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample actions using few-NFE inference.

        Args:
            context: VLM features [B, seq_len, hidden_dim]
            num_steps: Number of denoising steps (1 for distilled model)
            action_dim: Action dimension (defaults to self.action_dim)

        Returns:
            Sampled actions [B, T, action_dim]
        """
        if action_dim is None:
            action_dim = self.action_dim

        B = context.shape[0]
        device = context.device

        # Start from pure noise
        x = torch.randn(B, self.action_horizon, action_dim, device=device)

        if num_steps == 1 and hasattr(self, 'consistency_head'):
            # Direct 1-NFE inference using consistency head
            t = torch.ones(B, 1, device=device)
            velocity = self.forward(x, t, context)
            # Use consistency head for direct mapping
            actions = self.consistency_head(velocity)
        else:
            # Multi-step Euler integration (fallback)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.full((B, 1), i * dt, device=device)
                velocity = self.forward(x, t, context)
                x = x + velocity * dt
            actions = x

        return actions
