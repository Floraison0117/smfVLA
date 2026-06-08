"""
SnapFlow-modified Pi0.5 model.

Extends Pi05SMF from smfVLA with:
- target_time_mlp: Zero-initialized 2-layer MLP for target-time conditioning
- Modified embed_suffix to inject target-time embedding
- SnapFlow-specific loss computation

Paper: https://arxiv.org/abs/2604.05656
Reference: smfVLA/src/smf_vla/models/pi05_smf.py
"""

import dataclasses
from typing import TYPE_CHECKING, Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at

from snapflow.models.target_time_mlp import TargetTimeMLP

if TYPE_CHECKING:
    pass


@dataclasses.dataclass(frozen=True)
class Pi05SnapFlowConfig(pi0_config.Pi0Config):
    """SnapFlow-modified Pi0.5 configuration."""

    # SnapFlow-specific parameters
    alpha: float = 0.5  # FM/Consistency mixing ratio
    lambda_consistency: float = 0.1  # Consistency loss weight
    prediction_clamp_min: float = -20
    prediction_clamp_max: float = 20

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi05SnapFlow":
        return Pi05SnapFlow(self, rngs=nnx.Rngs(rng))


class Pi05SnapFlow(pi0.Pi0):
    """
    SnapFlow-modified Pi0.5 model.

    Extends Pi05SMF with target-time embedding for SnapFlow training.

    Key additions:
    - target_time_mlp: Zero-initialized 2-layer MLP
    - embed_suffix_with_target_time: Injects target-time embedding
    - compute_snapflow_loss: SnapFlow-specific loss

    Paper Section 3.5:
    "A zero-initialized two-layer MLP that encodes s and adds to the
    existing time embedding before each transformer block."
    """

    def __init__(self, config: Pi05SnapFlowConfig, rngs: nnx.Rngs):
        # Initialize parent Pi0 class (includes time_proj from Pi05SMF pattern)
        super().__init__(config, rngs=rngs)

        # Save config
        self._snapflow_config = config

        # Get action expert width (same as Pi05SMF)
        import openpi.models.gemma as _gemma
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        width = action_expert_config.width

        # NEW: Target-time MLP (zero-initialized)
        self.target_time_mlp = TargetTimeMLP(width=width, rngs=rngs)

    @at.typecheck
    def embed_suffix_with_target_time(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        t: at.Float[at.Array, " b"],
        s: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """
        Embed suffix with target-time conditioning.

        Extends Pi05SMF's embed_suffix_smf by adding target-time embedding.

        Args:
            obs: Observation data
            noisy_actions: Noisy action chunk
            t: Current time (for velocity estimation)
            s: Target time (s=t for FM, s=0 for consistency)

        Returns:
            tokens: Concatenated prefix + suffix tokens
            input_mask: Valid token mask
            ar_mask: Autoregressive mask
            adarms_cond: Time + target-time embedding
        """
        import jax
        import jax.numpy as jnp
        from openpi.models.pi0 import posemb_sincos

        input_mask = []
        ar_mask = []
        tokens = []

        # Action tokens
        action_tokens = self.action_in_proj(noisy_actions)

        # Base time embedding E(t) (same as Pi05SMF)
        time_emb_t = posemb_sincos(t, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb_t = self.time_mlp_in(time_emb_t)
        time_emb_t = nnx.swish(time_emb_t)
        time_emb_t = self.time_mlp_out(time_emb_t)
        time_emb_t = nnx.swish(time_emb_t)  # [B, width]

        # Target-time embedding E(s) (NEW for SnapFlow)
        target_emb = self.target_time_mlp(s)  # [B, width]

        # Combine: base + target (additive)
        # At step 0, target_emb = 0, so this matches Pi05SMF
        adarms_cond = time_emb_t + target_emb  # [B, width]

        # Action expert tokens (no time mixing here)
        action_expert_tokens = action_tokens

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """
        Compute SnapFlow loss.

        Delegates to snapflow_loss.compute_snapflow_loss.

        Args:
            rng: Random key
            observation: Observation data
            actions: Ground-truth actions
            train: Training mode flag

        Returns:
            Total loss (scalar)
        """
        from snapflow.training.snapflow_loss import compute_snapflow_loss

        # Define model forward function with target-time conditioning
        def model_fn(params, obs, noisy_actions, r, t, s):
            """Forward: F_θ(noisy_actions, r, t, s) → velocity."""
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_target_time(
                obs, noisy_actions, t=t, s=s
            )
            adarms_cond_list = [None, adarms_cond]

            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=adarms_cond_list,
            )
            v = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            # Clamp prediction to prevent numerical instabilities
            v = jnp.clip(v, self.config.prediction_clamp_min, self.config.prediction_clamp_max)
            return v

        # Dummy normalization (actions already normalized in data loader)
        action_mean = jnp.zeros(actions.shape[-1])
        action_std = jnp.ones(actions.shape[-1])

        loss, metrics = compute_snapflow_loss(
            model_fn=model_fn,
            params=None,
            observation=observation,
            actions=actions,
            action_mean=action_mean,
            action_std=action_std,
            rng=rng,
            alpha=self.config.alpha,
            lambda_consistency=self.config.lambda_consistency,
        )

        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 1,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """
        SnapFlow 1-NFE inference.

        For SnapFlow, we always use 1-NFE (num_steps=1).
        The key difference from baseline is using s=0 instead of s=t.

        Args:
            rng: Random key
            observation: Observation data
            num_steps: Number of denoising steps (SnapFlow uses 1)
            noise: Optional noise for reproducibility

        Returns:
            Predicted actions
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # SnapFlow 1-NFE: x_0 = x_1 - F_θ(x_1, s=0, t=1)
        # Key: s=0 (not s=t) for one-step generation
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        # Single forward pass with s=0, t=1
        batch_size = noise.shape[0]
        s = jnp.zeros(batch_size)  # Target time = 0 for 1-NFE
        t = jnp.ones(batch_size)   # Current time = 1

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_with_target_time(
            observation, noise, t=t, s=s
        )
        adarms_cond_list = [None, adarms_cond]

        suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_step = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_mask.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
        positions_step = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions_step,
            kv_cache=kv_cache,
            adarms_cond=adarms_cond_list,
        )
        v = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        # 1-NFE: x_0 = x_1 - v
        actions = noise - v
        return actions
