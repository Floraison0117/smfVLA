"""GMFlow analytic policy: Gaussian Mixture Flow matching.

Given GMM parameters predicted by the student at outer time t_src, compute the
analytic velocity field at any inner time t without calling a neural network.

Uses the official pi-Flow parameterization: the network predicts *velocity-space*
quantities (t-invariant), and the policy applies t_src scaling:
  means_x0  = x_t_src - t_src * vel_means        (velocity -> x_0 means)
  var_k     = exp(2 * log_stds_vel) * t_src^2     (variance scales with t_src^2)

For a single Gaussian component k with mean mu_k and variance var_k:
  x_0 ~ N(mu_k, var_k I)
  x_t = (1-t)*x_0 + t*eps   where eps ~ N(0, I)

  alpha_{t,k} = (1-t)^2 * var_k + t^2
  E[x_0 | x_t, k] = (t^2*mu_k + (1-t)*var_k*x_t) / alpha_{t,k}
  u_k(x_t, t) = (x_t - E[x_0 | x_t, k]) / t

  gamma_k ~ w_k * N(x_t; (1-t)*mu_k, alpha_{t,k}*I)
  u(x_t, t) = Sum_k gamma_k * u_k(x_t, t)

At t_src=1 (1-NFE): var = exp(2*logstd), identical to the unscaled version.
At t_src<1 (multi-NFE): var shrinks, matching the true posterior contraction.
"""

import jax
import jax.numpy as jnp


def gmflow_velocity(
    x_t: jax.Array,
    t: jax.Array,
    means: jax.Array,
    log_stds: jax.Array,
    log_weights: jax.Array,
    t_src: float = 1.0,
    eps: float = 1e-4,
) -> jax.Array:
    """Compute analytic GMFlow velocity at state x_t and time t.

    Args:
        x_t: current state [B, H, D]
        t: current time, scalar or [B]
        means: GMM component x_0 means [B, K, H, D]
        log_stds: log std per component (velocity-space) [B, K]
        log_weights: log mixture weights [B, K]
        t_src: prediction time at which the GMM was produced (scalar).
            Variance scales as exp(2*log_stds) * t_src^2.
        eps: threshold below which to use small-t approximation

    Returns:
        velocity u(x_t, t) with shape [B, H, D]
    """
    B, K, H, D = means.shape
    var = jnp.exp(2.0 * log_stds) * jnp.square(t_src)  # [B, K] -- t_src scaled

    t_b = jnp.broadcast_to(t, (B,)) if t.ndim == 0 else t
    t_bc = t_b[:, None, None, None]  # [B, 1, 1, 1]
    var_bc = var[:, :, None, None]    # [B, K, 1, 1]

    # alpha_{t,k} = (1-t)^2 * var_k + t^2
    alpha = jnp.square(1.0 - t_bc) * var_bc + jnp.square(t_bc)  # [B, K, 1, 1]

    # E[x_0 | x_t, k] = (t^2*mu_k + (1-t)*var_k*x_t) / alpha
    t_sq = jnp.square(t_bc)
    posterior_mean = (
        t_sq * means + (1.0 - t_bc) * var_bc * x_t[:, None, :, :]
    ) / jnp.maximum(alpha, 1e-10)  # [B, K, H, D]

    # u_k = (x_t - E[x_0|x_t,k]) / t
    # For small t, use approximation: u ~ mu_k - x_t
    use_approx = t_bc < eps
    safe_t = jnp.where(use_approx, 1.0, t_bc)
    u_k = (x_t[:, None, :, :] - posterior_mean) / safe_t  # [B, K, H, D]
    u_k = jnp.where(use_approx, means - x_t[:, None, :, :], u_k)

    # Log probability: log N(x_t; (1-t)*mu_k, alpha_{t,k}*I)
    diff = x_t[:, None, :, :] - (1.0 - t_bc) * means  # [B, K, H, D]
    log_alpha = jnp.log(jnp.maximum(alpha, 1e-20))  # [B, K, 1, 1]

    # Sum log prob over all H*D elements per component
    n_elements = H * D
    log_prob = -0.5 * n_elements * jnp.squeeze(log_alpha, axis=(-1, -2))  # [B, K]
    log_prob = log_prob - 0.5 * jnp.sum(jnp.square(diff), axis=(-1, -2)) / jnp.squeeze(
        jnp.maximum(alpha, 1e-20), axis=(-1, -2)
    )  # [B, K]

    # Responsibilities via log-sum-exp for numerical stability
    log_joint = log_weights + log_prob  # [B, K]
    log_sum = jax.nn.logsumexp(log_joint, axis=-1, keepdims=True)  # [B, 1]
    gamma = jnp.exp(log_joint - log_sum)  # [B, K]

    # Mixture velocity: u = Sum_k gamma_k * u_k
    u = jnp.sum(gamma[:, :, None, None] * u_k, axis=1)  # [B, H, D]

    return u


def gmflow_rollout(
    x_1: jax.Array,
    means: jax.Array,
    log_stds: jax.Array,
    log_weights: jax.Array,
    num_substeps: int = 8,
    t_src: float = 1.0,
    stop_gradient: bool = True,
    t_start: float = 1.0,
    t_end: float = 0.0,
) -> jax.Array:
    """Run GMFlow Euler integration from t_start to t_end.

    Args:
        x_1: initial state [B, H, D] at time t_start
        means: GMM x_0 means [B, K, H, D]
        log_stds: log std per component (velocity-space) [B, K]
        log_weights: log mixture weights [B, K]
        t_src: prediction time at which the GMM was produced (scalar).
            Controls variance scaling: var = exp(2*log_stds) * t_src^2.
        num_substeps: number of Euler substeps
        stop_gradient: if True, stop gradient through rollout states
        t_start: starting time (default 1.0 = noise)
        t_end: ending time (default 0.0 = data)

    Returns:
        final state at time t_end [B, H, D]
    """
    B, H, D = x_1.shape
    ts = jnp.linspace(t_start, t_end, num_substeps + 1)

    def _step(x, i):
        t_cur = ts[i]
        t_nxt = ts[i + 1]
        u = gmflow_velocity(x, t_cur, means, log_stds, log_weights, t_src=t_src)
        x_new = x + (t_nxt - t_cur) * u
        if stop_gradient:
            x_new = jax.lax.stop_gradient(x_new)
        return x_new, None

    x_final, _ = jax.lax.scan(_step, x_1, jnp.arange(num_substeps))
    return x_final


def gmflow_rollout_with_states(
    x_1: jax.Array,
    means: jax.Array,
    log_stds: jax.Array,
    log_weights: jax.Array,
    num_substeps: int = 8,
    t_src: float = 1.0,
    query_indices: jax.Array | None = None,
    stop_gradient: bool = True,
    t_start: float = 1.0,
    t_end: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run GMFlow Euler integration and return selected intermediate states.

    Args:
        x_1: initial state [B, H, D] at time t_start
        means: GMM x_0 means [B, K, H, D]
        log_stds: log std per component (velocity-space) [B, K]
        log_weights: log mixture weights [B, K]
        num_substeps: number of Euler substeps
        t_src: prediction time at which the GMM was produced (scalar).
            Controls variance scaling: var = exp(2*log_stds) * t_src^2.
        query_indices: indices of substeps at which to record velocity and state,
            e.g. jnp.array([0, 4]) to record at the first and fifth substep.
            If None, records all substeps.
        stop_gradient: if True, stop gradient through rollout states
        t_start: starting time (default 1.0 = noise)
        t_end: ending time (default 0.0 = data)

    Returns:
        x_final: final state at time t_end [B, H, D]
        states: recorded intermediate states [B, M, H, D]
        velocities: recorded GMFlow velocities at those states [B, M, H, D]
    """
    B, H, D = x_1.shape
    ts = jnp.linspace(t_start, t_end, num_substeps + 1)
    if query_indices is None:
        query_indices = jnp.arange(num_substeps)

    M = query_indices.shape[0]

    def _step(carry, i):
        x = carry
        t_cur = ts[i]
        t_nxt = ts[i + 1]
        u = gmflow_velocity(x, t_cur, means, log_stds, log_weights, t_src=t_src)
        state_and_vel = (x, u)
        x_new = x + (t_nxt - t_cur) * u
        if stop_gradient:
            x_new = jax.lax.stop_gradient(x_new)
        return x_new, state_and_vel

    x_final, (all_states, all_vels) = jax.lax.scan(_step, x_1, jnp.arange(num_substeps))

    # Select queried timesteps
    selected_states = all_states[query_indices]   # [M, B, H, D]
    selected_vels = all_vels[query_indices]        # [M, B, H, D]
    states = jnp.transpose(selected_states, (1, 0, 2, 3))   # [B, M, H, D]
    velocities = jnp.transpose(selected_vels, (1, 0, 2, 3))  # [B, M, H, D]

    return x_final, states, velocities
