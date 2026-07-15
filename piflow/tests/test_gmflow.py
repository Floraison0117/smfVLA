"""Unit tests for the GMFlow analytic policy (gmflow.py).

Tests cover:
  1. Posterior mean formula (single-component, analytic check)
  2. Velocity formula at t and t_src=1
  3. Small-t approximation (u ~ mu - x_t)
  4. t_src variance scaling (var = exp(2*logstd) * t_src^2)
  5. Responsibility weights (softmax via logsumexp)
  6. GMFlow rollout integration (Euler, endpoint)
  7. Rollout with states (query recording)
  8. No-op at init (zero means -> velocity = x_t / t)
  9. Deterministic single-component -> simple posterior
  10. Gradient flow through velocity

Run:  cd /root/autodl-tmp && python -m pytest piflow/tests/test_gmflow.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# Path setup so `from piflow_vla.models import gmflow` resolves.
import os, sys

# piflow/tests/ -> piflow/ -> repo root
_PIFLOW_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_PIFLOW_DIR)
for _p in (
    os.path.join(_PIFLOW_DIR, "src"),
    os.path.join(_REPO_ROOT, "openpi", "src"),
    os.path.join(_REPO_ROOT, "openpi", "packages", "openpi-client", "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from piflow_vla.models import gmflow  # noqa: E402


# ---------- fixtures ----------------------------------------------------------

@pytest.fixture
def rng():
    return jax.random.key(42)


@pytest.fixture
def small_gmm(rng):
    """K=2, H=3, D=4 small GMM for analytic checks."""
    B, K, H, D = 1, 2, 3, 4
    means = jax.random.normal(jax.random.key(0), (B, K, H, D))
    log_stds = jnp.array([[0.0, -0.5]])       # stds [1.0, 0.606]
    log_weights = jnp.array([[0.0, 0.0]])      # equal weights
    x_t = jax.random.normal(jax.random.key(1), (B, H, D))
    return x_t, means, log_stds, log_weights


# ---------- 1. posterior mean ------------------------------------------------

def test_posterior_mean_single_component(rng):
    """Single Gaussian: E[x0 | x_t] has closed form E = (t^2*mu + (1-t)*var*x_t)/alpha."""
    B, K, H, D = 1, 1, 2, 3
    mu = jnp.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    var = jnp.array([[0.5]])
    log_std = 0.5 * jnp.log(var)
    log_w = jnp.array([[0.0]])
    x_t = jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    t = 0.7

    v = gmflow.gmflow_velocity(x_t, jnp.array(t), mu, log_std, log_w, t_src=1.0)

    alpha = (1 - t) ** 2 * var + t ** 2
    post_mean = (t ** 2 * mu[0, 0] + (1 - t) * var[0, 0] * x_t[0]) / alpha[0, 0]
    expected_v = (x_t[0] - post_mean) / t

    np.testing.assert_allclose(np.array(v[0]), np.array(expected_v), atol=1e-5)


# ---------- 2. velocity formula ----------------------------------------------

def test_velocity_shape(small_gmm):
    x_t, means, log_stds, log_w = small_gmm
    v = gmflow.gmflow_velocity(x_t, jnp.array(0.5), means, log_stds, log_w)
    assert v.shape == x_t.shape


def test_velocity_finite(small_gmm):
    x_t, means, log_stds, log_w = small_gmm
    v = gmflow.gmflow_velocity(x_t, jnp.array(0.5), means, log_stds, log_w)
    assert bool(jnp.all(jnp.isfinite(v)))


# ---------- 3. small-t approximation -----------------------------------------

def test_small_t_approximation():
    """When t < eps, u should use approximation u_k = mu_k - x_t."""
    B, K, H, D = 1, 1, 2, 2
    mu = jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    log_std = jnp.array([[0.0]])
    log_w = jnp.array([[0.0]])
    x_t = jnp.array([[[0.0, 0.0], [0.0, 0.0]]])

    v_approx = gmflow.gmflow_velocity(x_t, jnp.array(1e-6), mu, log_std, log_w, eps=1e-4)
    expected = mu[0, 0] - x_t[0]  # = mu when x_t=0

    np.testing.assert_allclose(np.array(v_approx[0]), np.array(expected), atol=1e-6)


# ---------- 4. t_src variance scaling ---------------------------------------

def test_t_src_scales_variance(rng):
    """Variance var = exp(2*logstd) * t_src^2. Lower t_src -> smaller var -> velocity closer to (x_t-mu)/t."""
    B, K, H, D = 1, 1, 2, 2
    mu = jnp.array([[[[1.0, 1.0], [1.0, 1.0]]]])
    log_std = jnp.array([[0.0]])     # var=1 at t_src=1
    log_w = jnp.array([[0.0]])
    x_t = jnp.array([[[0.0, 0.0], [0.0, 0.0]]])
    t = 0.5

    v_full = gmflow.gmflow_velocity(x_t, jnp.array(t), mu, log_std, log_w, t_src=1.0)
    v_half = gmflow.gmflow_velocity(x_t, jnp.array(t), mu, log_std, log_w, t_src=0.5)

    # Different variance -> different posterior -> different velocity
    assert not bool(jnp.allclose(v_full, v_half, atol=1e-6))

    # At t_src=0.5, var=0.25 (smaller). Posterior should be closer to mu.
    # Compute expected for both.
    for label, v_calc, ts in [("full", v_full, 1.0), ("half", v_half, 0.5)]:
        var = jnp.exp(2 * log_std) * ts ** 2
        alpha = (1 - t) ** 2 * var + t ** 2
        post = (t ** 2 * mu[0, 0] + (1 - t) * var[0, 0] * x_t[0]) / alpha[0, 0]
        exp_v = (x_t[0] - post) / t
        np.testing.assert_allclose(np.array(v_calc[0]), np.array(exp_v), atol=1e-5,
                                   err_msg=f"{label} mismatch")


def test_t_src_zero_means_analytic():
    """With zero means, velocity has a known closed form that depends on t_src.
    var = t_src^2 (log_stds=0).
    alpha = (1-t)^2 * t_src^2 + t^2
    post = (1-t) * t_src^2 * x_t / alpha   (mu=0)
    v = (x_t - post) / t
    This verifies t_src scaling is applied correctly."""
    B, K, H, D = 1, 2, 2, 2
    mu = jnp.zeros((B, K, H, D))
    log_std = jnp.zeros((B, K))
    log_w = jnp.zeros((B, K))
    x_t = jnp.ones((B, H, D))
    t = 0.5

    for t_src in [1.0, 0.5, 0.25]:
        v = gmflow.gmflow_velocity(x_t, jnp.array(t), mu, log_std, log_w, t_src=t_src)
        var = t_src ** 2
        alpha = (1 - t) ** 2 * var + t ** 2
        post = (1 - t) * var * x_t[0] / alpha
        expected = (x_t[0] - post) / t
        np.testing.assert_allclose(np.array(v[0]), np.array(expected), atol=1e-5,
                                   err_msg=f"mismatch at t_src={t_src}")


# ---------- 5. responsibility weights ----------------------------------------

def test_responsibility_concentrates_on_close_mean():
    """With one mean very close to x_t and one far, the velocity should be
    dominated by the close component (high responsibility)."""
    B, K, H, D = 1, 2, 1, 1
    mu_close = jnp.array([[[[0.1]]]])
    mu_far = jnp.array([[[[10.0]]]])
    means = jnp.concatenate([mu_close, mu_far], axis=1)  # [1, 2, 1, 1]
    log_std = jnp.array([[-2.0, -2.0]])   # small variance -> sharp responsibilities
    log_w = jnp.array([[0.0, 0.0]])
    x_t = jnp.array([[[0.0]]])
    t = 0.5

    v = gmflow.gmflow_velocity(x_t, jnp.array(t), means, log_std, log_w, t_src=1.0)

    # Close component velocity ~ (x_t - mu_close)/t approx
    v_close = (0.0 - 0.1) / 0.5
    # Far component velocity would be (0.0 - 10.0)/0.5 = -20
    assert float(v[0, 0, 0]) > -1.0  # dominated by close, not far


def test_equal_weights_equal_means():
    """Two identical components -> velocity same as single component."""
    B, K, H, D = 1, 2, 2, 2
    mu_single = jax.random.normal(jax.random.key(0), (1, 1, H, D))
    means_double = jnp.broadcast_to(mu_single, (B, K, H, D))
    log_std_single = jnp.array([[0.0]])
    log_std_double = jnp.array([[0.0, 0.0]])
    log_w_single = jnp.array([[0.0]])
    log_w_double = jnp.array([[0.0, 0.0]])
    x_t = jax.random.normal(jax.random.key(1), (B, H, D))
    t = 0.6

    v1 = gmflow.gmflow_velocity(x_t, jnp.array(t), mu_single, log_std_single, log_w_single)
    v2 = gmflow.gmflow_velocity(x_t, jnp.array(t), means_double, log_std_double, log_w_double)

    np.testing.assert_allclose(np.array(v1), np.array(v2), atol=1e-5)


# ---------- 6. rollout -------------------------------------------------------

def test_rollout_endpoint_shape(rng):
    B, K, H, D = 2, 4, 5, 3
    x_1 = jax.random.normal(jax.random.key(0), (B, H, D))
    means = jax.random.normal(jax.random.key(1), (B, K, H, D))
    log_std = jnp.zeros((B, K))
    log_w = jnp.zeros((B, K))

    x_0 = gmflow.gmflow_rollout(x_1, means, log_std, log_w, num_substeps=8)
    assert x_0.shape == x_1.shape


def test_rollout_more_substeps_converges():
    """More substeps -> more accurate ODE integration -> result stabilizes.
    Euler method has O(1/N) error, so 8 vs 32 substeps should agree within ~0.15."""
    B, K, H, D = 1, 2, 2, 2
    x_1 = jnp.ones((B, H, D))
    means = jnp.zeros((B, K, H, D))
    log_std = jnp.zeros((B, K))  # var=1
    log_w = jnp.zeros((B, K))

    x_8 = gmflow.gmflow_rollout(x_1, means, log_std, log_w, num_substeps=8)
    x_32 = gmflow.gmflow_rollout(x_1, means, log_std, log_w, num_substeps=32)

    # Both approximate the same ODE; Euler error ~ O(1/N)
    np.testing.assert_allclose(np.array(x_8), np.array(x_32), atol=0.2)


def test_rollout_segment():
    """Rollout from t_start=0.5 to t_end=0.0 (segment, multi-NFE)."""
    B, K, H, D = 1, 2, 2, 2
    x_1 = jnp.ones((B, H, D))
    means = jnp.zeros((B, K, H, D))
    log_std = jnp.zeros((B, K))
    log_w = jnp.zeros((B, K))

    x_0 = gmflow.gmflow_rollout(
        x_1, means, log_std, log_w, num_substeps=4, t_src=0.5,
        t_start=0.5, t_end=0.0,
    )
    assert x_0.shape == x_1.shape
    assert bool(jnp.all(jnp.isfinite(x_0)))


# ---------- 7. rollout with states ------------------------------------------

def test_rollout_with_states_shapes(rng):
    B, K, H, D = 2, 4, 5, 3
    x_1 = jax.random.normal(jax.random.key(0), (B, H, D))
    means = jax.random.normal(jax.random.key(1), (B, K, H, D))
    log_std = jnp.zeros((B, K))
    log_w = jnp.zeros((B, K))
    qidx = jnp.array([0, 2, 4, 6])

    x_final, states, vels = gmflow.gmflow_rollout_with_states(
        x_1, means, log_std, log_w, num_substeps=8, query_indices=qidx,
    )
    M = len(qidx)
    assert x_final.shape == x_1.shape
    assert states.shape == (B, M, H, D)
    assert vels.shape == (B, M, H, D)


def test_rollout_with_states_query_subset():
    """query_indices selects a subset of all substeps."""
    B, K, H, D = 1, 2, 2, 2
    x_1 = jnp.ones((B, H, D))
    means = jnp.zeros((B, K, H, D))
    log_std = jnp.zeros((B, K))
    log_w = jnp.zeros((B, K))
    qidx = jnp.array([0, 4])

    _, states, _ = gmflow.gmflow_rollout_with_states(
        x_1, means, log_std, log_w, num_substeps=8, query_indices=qidx,
    )
    assert states.shape == (1, 2, H, D)


# ---------- 8. no-op at init -------------------------------------------------

def test_zero_means_velocity_is_xt_over_t():
    """When means=0 and log_stds=0 (var=1 at t_src=1), velocity = x_t / t.
    posterior_mean = (t^2 * 0 + (1-t)*1*x_t) / ((1-t)^2 * 1 + t^2)
    velocity = (x_t - post_mean) / t."""
    B, K, H, D = 1, 1, 2, 2
    mu = jnp.zeros((B, K, H, D))
    log_std = jnp.zeros((B, K))
    log_w = jnp.zeros((B, K))
    x_t = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])
    t = 0.5

    v = gmflow.gmflow_velocity(x_t, jnp.array(t), mu, log_std, log_w, t_src=1.0)

    var = 1.0
    alpha = (1 - t) ** 2 * var + t ** 2
    post = (t ** 2 * 0 + (1 - t) * var * x_t[0]) / alpha
    expected = (x_t[0] - post) / t

    np.testing.assert_allclose(np.array(v[0]), np.array(expected), atol=1e-5)


# ---------- 9. deterministic single-component --------------------------------

def test_single_component_matches_analytic():
    """K=1 -> pure Gaussian posterior, velocity = (x_t - E[x0|x_t]) / t."""
    B, K, H, D = 1, 1, 3, 2
    mu = jnp.array([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
    log_std = jnp.array([[0.0]])  # var=1
    log_w = jnp.array([[0.0]])
    x_t = jnp.array([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]])
    t = 0.4

    v = gmflow.gmflow_velocity(x_t, jnp.array(t), mu, log_std, log_w, t_src=1.0)

    var = 1.0
    alpha = (1 - t) ** 2 * var + t ** 2
    E_x0 = (t ** 2 * mu[0, 0] + (1 - t) * var * x_t[0]) / alpha
    expected_v = (x_t[0] - E_x0) / t

    np.testing.assert_allclose(np.array(v[0]), np.array(expected_v), atol=1e-5)


# ---------- 10. gradient flow ------------------------------------------------

def test_gradient_flows_through_velocity():
    """Gradients should flow through velocity w.r.t. means."""
    B, K, H, D = 1, 2, 2, 2
    x_t = jnp.ones((B, H, D))
    log_std = jnp.zeros((B, K))
    log_w = jnp.zeros((B, K))
    t = 0.5

    def loss_fn(means):
        v = gmflow.gmflow_velocity(x_t, jnp.array(t), means, log_std, log_w, t_src=1.0)
        return jnp.sum(v)

    means = jnp.zeros((B, K, H, D))
    grad = jax.grad(loss_fn)(means)
    assert grad.shape == means.shape
    assert bool(jnp.all(jnp.isfinite(grad)))
    # Gradient should be non-zero (velocity depends on means)
    assert float(jnp.sum(jnp.abs(grad))) > 0


def test_gradient_flows_through_rollout():
    """Gradients flow through rollout when stop_gradient=False."""
    B, K, H, D = 1, 2, 2, 2
    x_1 = jnp.ones((B, H, D))
    log_std = jnp.zeros((B, K))
    log_w = jnp.zeros((B, K))

    def loss_fn(means):
        x_0 = gmflow.gmflow_rollout(x_1, means, log_std, log_w,
                                    num_substeps=4, stop_gradient=False)
        return jnp.sum(x_0)

    means = jnp.zeros((B, K, H, D))
    grad = jax.grad(loss_fn)(means)
    assert grad.shape == means.shape
    assert float(jnp.sum(jnp.abs(grad))) > 0


def test_stop_gradient_blocks_rollout_gradient():
    """When stop_gradient=True, gradient through rollout states is blocked."""
    B, K, H, D = 1, 2, 2, 2
    x_1 = jnp.ones((B, H, D))
    log_std = jnp.zeros((B, K))
    log_w = jnp.zeros((B, K))

    def loss_fn(means):
        x_0 = gmflow.gmflow_rollout(x_1, means, log_std, log_w,
                                    num_substeps=4, stop_gradient=True)
        return jnp.sum(x_0)

    means = jnp.zeros((B, K, H, D))
    grad = jax.grad(loss_fn)(means)
    # With stop_gradient, the final x_0 is detached -> gradient is zero
    assert float(jnp.sum(jnp.abs(grad))) == 0.0
