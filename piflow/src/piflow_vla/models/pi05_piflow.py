"""Pi-Flow model: pi0.5 student with GMM output head for GMFlow distillation.

Pi05PiFlow extends Pi0 by replacing the direct velocity output with a Gaussian
Mixture Model (GMM) head. The model predicts GMM parameters (velocity-space)
at the start of an outer interval, then the analytic GMFlow policy handles all
inner substeps without further neural network calls.

Uses the official pi-Flow parameterization:
  - gmm_mean_proj predicts *velocity* means (not x_0 means directly)
  - means_x0 = x_t_src - t_src * vel_means   (converted at prediction time)
  - var = exp(2 * log_stds) * t_src^2         (variance scales with t_src^2)

This makes the GMM parameters t-invariant in the network output space, so the
same logstd value naturally produces smaller x_0-variance at lower t_src
(matching the true posterior contraction). At t_src=1 (1-NFE), this reduces to
the simple parameterization: var = exp(2*logstd).
"""

import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0, pi0_config
from openpi.shared import array_typing as at

from . import gmflow


def _zero_init(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype=dtype)


@dataclasses.dataclass(frozen=True)
class Pi05PiFlowConfig(pi0_config.Pi0Config):
    """Pi-Flow configuration with GMM head."""

    num_components: int = 8
    inner_substeps: int = 8

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi05PiFlow":
        return Pi05PiFlow(self, rngs=nnx.Rngs(rng))


class Pi05PiFlow(pi0.Pi0):
    """Pi-Flow pi0.5 student: predicts GMM parameters for GMFlow analytic policy.

    The student network predicts velocity-space GMM parameters. At prediction
    time t_src, these are converted to x_0-space means and t_src-scaled variances
    for the analytic GMFlow rollout.
    """

    def __init__(self, config: Pi05PiFlowConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self._pf_config = config

        from openpi.models import gemma as _gemma

        action_expert_config = _gemma.get_config(config.action_expert_variant)
        width = action_expert_config.width
        K = config.num_components
        H = config.action_horizon
        D = config.action_dim

        # GMM output heads (all zero-init for clean no-op start)
        # gmm_mean_proj: predicts velocity means [B, width] -> [B, K*H*D]
        self.gmm_mean_proj = nnx.Linear(
            in_features=width,
            out_features=K * H * D,
            rngs=rngs,
            kernel_init=_zero_init,
            bias_init=_zero_init,
        )
        # gmm_logstd_proj: predicts velocity-space log std [B, width] -> [B, K]
        self.gmm_logstd_proj = nnx.Linear(
            in_features=width,
            out_features=K,
            rngs=rngs,
            kernel_init=_zero_init,
            bias_init=_zero_init,
        )
        # gmm_logweight_proj: predicts log mixture weights [B, width] -> [B, K]
        self.gmm_logweight_proj = nnx.Linear(
            in_features=width,
            out_features=K,
            rngs=rngs,
            kernel_init=_zero_init,
            bias_init=_zero_init,
        )

    def forward_gmm(
        self,
        obs: _model.Observation,
        noisy_actions: jax.Array,
        time: jax.Array,
        *,
        prefix_tokens=None,
        prefix_mask=None,
        prefix_ar_mask=None,
        prefix_kv_cache=None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Full forward pass returning GMM parameters.

        Predicts velocity-space quantities and converts to x_0-space means.
        The log_stds remain in velocity-space; the caller passes t_src to
        gmflow_rollout for variance scaling (var = exp(2*log_stds) * t_src^2).

        Args:
            obs: observation dict (images, state, prompt)
            noisy_actions: [B, H, D] current noisy action chunk at time `time`
            time: [B] current time (used as t_src for velocity->x_0 conversion
                and variance scaling)
            prefix_tokens: precomputed prefix tokens (images + language).
                If None, computed internally via embed_prefix.
            prefix_mask: precomputed prefix input mask.
            prefix_ar_mask: precomputed prefix AR mask.
            prefix_kv_cache: precomputed prefix KV cache (from a prefix-only
                forward pass). When provided along with prefix_tokens, the
                LLM forward is suffix-only — gradients do not penetrate the
                frozen VLM backbone (3B params), only the action expert.

        Returns:
            means: [B, K, H, D] GMM component x_0 means
                (means_x0 = noisy_actions - time * vel_means)
            log_stds: [B, K] velocity-space log standard deviations
                (var = exp(2*log_stds) * t_src^2 in gmflow_velocity)
            log_weights: [B, K] log mixture weights
        """
        B = noisy_actions.shape[0]
        K = self._pf_config.num_components
        H = self.action_horizon
        D = self.action_dim

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            obs, noisy_actions, jnp.broadcast_to(time, (B,))
        )

        if prefix_kv_cache is not None and prefix_tokens is not None:
            # Suffix-only forward with cached prefix KV — avoids backward
            # through the 3B VLM backbone (frozen). Only the action expert
            # (trainable, ~430M params) participates in the forward + backward.
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            s2p_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([s2p_attn_mask, suffix_attn_mask], axis=-1)
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                adarms_cond=[None, adarms_cond],
                kv_cache=prefix_kv_cache,
            )
        else:
            # Original full forward (backward compat, no KV cache)
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )

        # Extract action token hidden states and mean-pool over horizon
        action_hidden = suffix_out[:, -H:]  # [B, H, width]
        pooled = jnp.mean(action_hidden, axis=1)  # [B, width]

        # Predict velocity-space GMM parameters
        vel_means_flat = self.gmm_mean_proj(pooled)  # [B, K*H*D]
        vel_means = vel_means_flat.reshape(B, K, H, D)  # [B, K, H, D]

        log_stds = self.gmm_logstd_proj(pooled)  # [B, K] velocity-space
        log_weights = self.gmm_logweight_proj(pooled)  # [B, K]

        # Convert velocity means to x_0 means: means_x0 = x_t_src - t_src * vel_means
        # At init (zero vel_means): means_x0 = x_t_src (no-op, returns input)
        t_src = time[:, None, None, None]  # [B, 1, 1, 1]
        means_x0 = noisy_actions[:, None, :, :] - t_src * vel_means  # [B, K, H, D]

        return means_x0, log_stds, log_weights

    @override
    def compute_loss(self, rng, observation, actions, *, train=False, **kwargs):
        raise NotImplementedError(
            "Pi05PiFlow.compute_loss is not used directly. "
            "Use piflow_vla.training.piflow_loss.compute_piflow_loss instead."
        )

    @override
    def sample_actions(
        self,
        rng,
        observation,
        *,
        num_steps=1,
        noise=None,
        num_substeps=None,
        method="gmflow",
        **kwargs,
    ):
        """Sample actions using GMFlow analytic policy.

        Supports 1/2/4-NFE (and arbitrary N). Each outer step predicts a fresh
        GMM at the segment's start time t_src, then rolls out analytically.

        Args:
            rng: random key
            observation: observation dict
            num_steps: number of outer NFE steps (1, 2, 4, ...)
            noise: optional initial noise [B, H, D]
            num_substeps: total GMFlow inner substeps across all segments
                (default: config.inner_substeps=8). Divided equally among segments.
            method: "gmflow" for analytic policy

        Returns:
            denoised action chunk [B, H, D]
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        if num_substeps is None:
            num_substeps = self._pf_config.inner_substeps

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Precompute prefix KV cache once — reused across all NFE segments
        # (observation doesn't change between segments).
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, prefix_kv = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        nfe = num_steps
        if nfe == 1:
            # 1-NFE: single GMM prediction at t=1, then GMFlow rollout to t=0
            means, log_stds, log_weights = self.forward_gmm(
                observation,
                noise,
                jnp.full((batch_size,), 1.0),
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                prefix_ar_mask=prefix_ar_mask,
                prefix_kv_cache=prefix_kv,
            )
            x_0 = gmflow.gmflow_rollout(
                noise,
                means,
                log_stds,
                log_weights,
                num_substeps=num_substeps,
                t_src=1.0,
                stop_gradient=False,
            )
            return x_0
        else:
            # Multi-NFE: GMM prediction at each outer step boundary
            # Segment the trajectory [1.0, 0.0] into nfe equal parts
            outer_ts = jnp.linspace(1.0, 0.0, nfe + 1)
            substeps_per_seg = max(1, num_substeps // nfe)
            x = noise
            for k in range(nfe):
                t_cur = float(outer_ts[k])
                t_nxt = float(outer_ts[k + 1])
                means, log_stds, log_weights = self.forward_gmm(
                    observation,
                    x,
                    jnp.full((batch_size,), t_cur),
                    prefix_tokens=prefix_tokens,
                    prefix_mask=prefix_mask,
                    prefix_ar_mask=prefix_ar_mask,
                    prefix_kv_cache=prefix_kv,
                )
                x = gmflow.gmflow_rollout(
                    x,
                    means,
                    log_stds,
                    log_weights,
                    num_substeps=substeps_per_seg,
                    t_src=t_cur,
                    stop_gradient=False,
                    t_start=t_cur,
                    t_end=t_nxt,
                )
            return x
