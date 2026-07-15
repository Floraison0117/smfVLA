"""Pi-Flow inference samplers: GMFlow-based action sampling for 1/2/4-NFE.

Eval uses zero teacher calls — only student transformer forward + analytic GMFlow.
Each NFE step predicts a fresh GMM at the segment boundary (t_src), then rolls
out analytically with t_src-dependent variance.
"""

import jax
import jax.numpy as jnp

from piflow_vla.models import gmflow


def sample_nfe_gmflow(
    model,
    observation,
    rng: jax.Array,
    nfe: int = 1,
    num_substeps: int = 8,
    noise: jax.Array | None = None,
) -> jax.Array:
    """N-NFE GMFlow sampling: N student forwards, N analytic rollouts.

    Segments [1.0, 0.0] into N equal parts. Each segment predicts a GMM at its
    start time t_src and rolls out to the segment end.

    Args:
        model: Pi05PiFlow model
        observation: preprocessed model observation
        rng: random key
        nfe: number of outer NFE steps (1, 2, 4, ...)
        num_substeps: total substeps across all segments (divided by NFE)
        noise: optional initial noise [B, H, D]

    Returns:
        denoised action chunk [B, H, D]
    """
    B = observation.state.shape[0]
    H = model.action_horizon
    D = model.action_dim

    if noise is None:
        noise = jax.random.normal(rng, (B, H, D))

    if nfe == 1:
        means, log_stds, log_weights = model.forward_gmm(
            observation, noise, jnp.full((B,), 1.0)
        )
        x_0 = gmflow.gmflow_rollout(
            noise, means, log_stds, log_weights,
            num_substeps=num_substeps, t_src=1.0, stop_gradient=False,
        )
        return x_0

    outer_ts = jnp.linspace(1.0, 0.0, nfe + 1)
    substeps_per_seg = max(1, num_substeps // nfe)
    x = noise
    for k in range(nfe):
        t_cur = float(outer_ts[k])
        t_nxt = float(outer_ts[k + 1])
        means, log_stds, log_weights = model.forward_gmm(
            observation, x, jnp.full((B,), t_cur)
        )
        x = gmflow.gmflow_rollout(
            x, means, log_stds, log_weights,
            num_substeps=substeps_per_seg, t_src=t_cur,
            stop_gradient=False, t_start=t_cur, t_end=t_nxt,
        )
    return x


def sample_1nfe_gmflow(
    model,
    observation,
    rng: jax.Array,
    num_substeps: int = 8,
    noise: jax.Array | None = None,
) -> jax.Array:
    """1-NFE GMFlow sampling: one student forward + analytic rollout."""
    return sample_nfe_gmflow(model, observation, rng, nfe=1,
                             num_substeps=num_substeps, noise=noise)


def sample_2nfe_gmflow(
    model,
    observation,
    rng: jax.Array,
    num_substeps: tuple[int, int] = (4, 4),
    noise: jax.Array | None = None,
) -> jax.Array:
    """2-NFE GMFlow sampling: two student forwards, two analytic rollouts.

    Outer intervals: [1.0, 0.5] and [0.5, 0.0].

    Args:
        num_substeps: (substeps in [1.0, 0.5], substeps in [0.5, 0.0])
    """
    B = observation.state.shape[0]
    H = model.action_horizon
    D = model.action_dim

    if noise is None:
        noise = jax.random.normal(rng, (B, H, D))

    # First outer interval: [1.0, 0.5]
    means_1, log_stds_1, log_weights_1 = model.forward_gmm(
        observation, noise, jnp.full((B,), 1.0)
    )
    x_mid = gmflow.gmflow_rollout(
        noise, means_1, log_stds_1, log_weights_1,
        num_substeps=num_substeps[0], t_src=1.0, stop_gradient=False,
        t_start=1.0, t_end=0.5,
    )

    # Second outer interval: [0.5, 0.0]
    means_2, log_stds_2, log_weights_2 = model.forward_gmm(
        observation, x_mid, jnp.full((B,), 0.5)
    )
    x_0 = gmflow.gmflow_rollout(
        x_mid, means_2, log_stds_2, log_weights_2,
        num_substeps=num_substeps[1], t_src=0.5, stop_gradient=False,
        t_start=0.5, t_end=0.0,
    )
    return x_0


def sample_4nfe_gmflow(
    model,
    observation,
    rng: jax.Array,
    num_substeps: int = 8,
    noise: jax.Array | None = None,
) -> jax.Array:
    """4-NFE GMFlow sampling: four student forwards, four analytic rollouts.

    Segments: [1.0,0.75], [0.75,0.5], [0.5,0.25], [0.25,0.0].
    """
    return sample_nfe_gmflow(model, observation, rng, nfe=4,
                             num_substeps=num_substeps, noise=noise)
