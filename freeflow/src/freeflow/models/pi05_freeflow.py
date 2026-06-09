"""
FreeFlow student model extending Pi0.5.

Implements 1-NFE student model trained via data-free distillation
from frozen π₀.₅ teacher.
"""

from typing import Any, Optional

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at
from openpi.models.pi0 import posemb_sincos


def _identity_zero_init():
    """Initialize linear layer as [I, 0] for stability (time projection)."""
    def init(key, shape, dtype=jnp.float32):
        # shape = (out_features, in_features)
        out_features, in_features = shape

        # Create identity block for first half, zeros for second half
        kernel = jnp.zeros(shape, dtype=dtype)
        half = in_features // 2

        # For identity matrix, we need out_features >= half
        # Set diagonal of first half columns to 1
        for i in range(min(out_features, half)):
            kernel = kernel.at[i, i].set(1.0)

        bias = jnp.zeros(out_features, dtype=dtype)
        return (kernel, bias)
    return init


class Pi05FreeFlow(pi0.Pi0):
    """
    FreeFlow student model extending π₀.₅.

    Key features:
    - Same architecture as teacher (π₀.₅) initially
    - Trained via data-free distillation loss
    - Supports 1-NFE inference
    - Optional lightweight student head for efficiency

    The model architecture is identical to Pi0, but the training
    procedure differs: we use teacher's multi-step path as target
    instead of ground-truth actions.
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        # Store config for variants
        self._freeflow_config = config

    def embed_suffix_freeflow(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        t: at.Float[at.Array, " b"],
        r: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """
        FreeFlow embed suffix with dual-time input (r, t).

        Args:
            obs: VLA observation
            noisy_actions: Noisy action chunk
            t: End time (1 for sampling, 0→1 for distillation)
            r: Reference time (0 for 1-NFE, varies for training)

        Returns:
            tokens: Action tokens with time conditioning
            input_mask: Valid token mask
            ar_mask: Autoregressive mask
            adarms_cond: Action embedding conditioning
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Project actions to tokens
        action_tokens = self.action_in_proj(noisy_actions)

        # Compute E(t) - time embedding for t
        # The checkpoint uses 1024-dim time embeddings (time_mlp_in kernel is 1024x1024)
        # Hardcode 1024 to match the checkpoint regardless of model config
        time_mlp_dim = 1024
        time_emb_t = posemb_sincos(
            t, time_mlp_dim,
            min_period=4e-3, max_period=4.0
        )
        time_emb_t = self.time_mlp_in(time_emb_t)
        time_emb_t = nnx.swish(time_emb_t)
        time_emb_t = self.time_mlp_out(time_emb_t)
        time_emb_t = nnx.swish(time_emb_t)  # [B, width]

        # Compute E(r) - time embedding for r
        time_emb_r = posemb_sincos(
            r, time_mlp_dim,
            min_period=4e-3, max_period=4.0
        )
        time_emb_r = self.time_mlp_in(time_emb_r)
        time_emb_r = nnx.swish(time_emb_r)
        time_emb_r = self.time_mlp_out(time_emb_r)
        time_emb_r = nnx.swish(time_emb_r)  # [B, width]

        # Average the time embeddings instead of concatenating
        # This avoids the dimension mismatch with time_proj
        adarms_cond = (time_emb_t + time_emb_r) / 2.0  # [B, width]

        # Note: student_head removed due to dimension mismatch
        # Can be added back once we resolve the checkpoint loading

        # Action expert tokens
        action_expert_tokens = action_tokens

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask, adarms_cond

    def embed_prefix(self, obs):
        """
        Override embed_prefix without type check for JIT compatibility.

        Args:
            obs: VLA observation (Observation object or dict)

        Returns:
            tokens: Prefix tokens (image + language)
            input_mask: Valid token mask
            ar_mask: Autoregressive mask
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Convert Observation object to dict if needed
        if hasattr(obs, 'images'):
            # Observation object: has .images and .image_masks attributes
            images = obs.images
            image_masks = obs.image_masks
            tokenized_prompt = obs.tokenized_prompt
            tokenized_prompt_mask = obs.tokenized_prompt_mask
        else:
            # Dict format from data_loader
            images = obs.get("images", obs.get("image", {}))
            image_masks = obs.get("image_masks", obs.get("image_mask", {}))
            tokenized_prompt = obs.get("tokenized_prompt")
            tokenized_prompt_mask = obs.get("tokenized_prompt_mask")

        # Embed images
        for name in images:
            image_tokens, _ = self.PaliGemma.img(images[name], train=False)
            tokens.append(image_tokens)

            # Handle mask format
            if name in image_masks:
                mask = image_masks[name]
            else:
                mask = jnp.ones(image_tokens.shape[:2], dtype=jnp.bool_)

            input_mask.append(
                einops.repeat(
                    mask,
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            ar_mask += [False] * image_tokens.shape[1]

        # Add language (tokenized inputs) if present
        if tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(tokenized_prompt_mask if tokenized_prompt_mask is not None else jnp.ones(tokenized_inputs.shape[:2], dtype=jnp.bool_))
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """
        Override embed_suffix to use FreeFlow version.

        For 1-NFE inference: r=0, t=timestep
        """
        r = jnp.zeros_like(timestep)
        return self.embed_suffix_freeflow(obs, noisy_actions, t=timestep, r=r)

    def __call__(
        self,
        obs: _model.Observation | dict,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        r: Optional[at.Float[at.Array, " b"]] = None,
    ) -> _model.Actions:
        """
        Forward pass with optional reference time r.

        Args:
            obs: VLA observation (Observation object or dict)
            noisy_actions: Noisy action chunk (z_t)
            timestep: Current time t
            r: Reference time (optional, defaults to 0 for 1-NFE)

        Returns:
            Predicted velocity (or actions depending on mode)
        """
        # Convert dict to Observation if needed
        if isinstance(obs, dict):
            obs = _model.Observation.from_dict(obs)

        if r is None:
            r = jnp.zeros_like(timestep)

        # Get prefix tokens (images + language) using parent's method
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)

        # Populate KV cache with prefix tokens (first fill KV cache)
        from openpi.models.pi0 import make_attn_mask
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # Get suffix tokens with FreeFlow dual-time conditioning
        tokens, input_mask, ar_mask, adarms_cond = self.embed_suffix_freeflow(
            obs, noisy_actions, t=timestep, r=r
        )

        # Make attention masks for suffix tokens
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how suffix tokens attend to each other
        suffix_attn_mask = make_attn_mask(input_mask, ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how suffix tokens attend to prefix
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=tokens.shape[1])
        # `full_attn_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how suffix tokens attend to full sequence
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

        # Get positions for suffix tokens (continuing from prefix positions)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(input_mask, axis=-1) - 1

        # Run through action expert with KV cache
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )

        # Project to action space
        actions = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return actions


def create_freeflow_model(
    config: Optional[pi0_config.Pi0Config] = None,
    rngs: Optional[nnx.Rngs] = None,
) -> Pi05FreeFlow:
    """
    Create FreeFlow student model.

    Args:
        config: Pi0 config (uses default if None)
        rngs: Random key (creates new if None)

    Returns:
        FreeFlow model instance
    """
    if config is None:
        config = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_2b",
            pi05=True,
            action_dim=32,
            action_horizon=1,
        )

    if rngs is None:
        rngs = nnx.Rngs(0)

    model = Pi05FreeFlow(config, rngs)
    return model
