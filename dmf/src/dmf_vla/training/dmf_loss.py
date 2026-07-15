"""
DMF (Decoupled MeanFlow) loss —  L = 0.5 * (L_FM + L_MF).

Time sampling: logit-normal (sigmoid(N)).
- t_fm ~ LN(P_mean, P_std) — standard flow matching time
- ln1 ~ LN(P_mean_t, P_std_t), ln2 ~ LN(P_mean_r, P_std_r)
- t_mf = max(ln1, ln2), r_mf = min(ln1, ln2)  (guarantees t >= r)

Linear interpolation: z_t = (1-t)*x_0 + t*eps, v_t = eps - x_0.
FM: model(z_t, t, t) ≈ v_t  (flow map at r=t reduces to velocity).
MF: JVP computes du/dt, then u_tgt = v_t + (r-t)*du_dt.
Log-variance weighted loss for both branches.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Any, Tuple, Dict


def ln_sampler(rng, shape, mean, std):
    """Logit-normal sample: sigmoid(N(mean, std))."""
    return jax.nn.sigmoid(jax.random.normal(rng, shape) * std + mean)


def log_lv_loss(pred, target, lv, eps=1e-3):
    """Log-variance weighted loss.

    Returns (mse_per_sample[B], log_loss_per_sample[B]).

    Numerically stable via the identity:
      log(exp(-lv) * m + eps) + lv = log(m + eps * exp(lv))
    """
    err = jnp.square(pred - target)
    feat_axes = tuple(range(1, err.ndim))
    mean_loss = jnp.mean(err, axis=feat_axes)
    # Use the algebraically equivalent form that avoids exp(-lv) overflow
    log_loss = jnp.log(mean_loss + eps * jnp.exp(lv))
    mse = jnp.sum(err, axis=feat_axes)
    return mse, log_loss


def compute_dmf_loss(
    model_fn: Callable,
    params: Any,
    observation: Dict,
    actions: jax.Array,
    action_mean: jax.Array,
    action_std: jax.Array,
    rng: jax.Array,
    p_mean: float = 0.0,
    p_std: float = 1.0,
    p_mean_t: float = 0.4,
    p_std_t: float = 1.0,
    p_mean_r: float = -1.2,
    p_std_r: float = 1.0,
    use_logvar: bool = True,
) -> Tuple[jax.Array, Dict]:
    """Compute DMF loss = 0.5 * (L_FM + L_MF).

    model_fn signature: (params, obs, noisy_actions, t[B], r[B], return_logvar=bool)
                        -> u[B,H,D]  or  (u[B,H,D], logvar[B])
    """
    batch_size = actions.shape[0]

    # Normalize actions to zero-mean unit-variance for flow matching
    x_0 = (actions - action_mean) / (action_std + 1e-8)

    rng, noise_rng = jax.random.split(rng)
    eps = jax.random.normal(noise_rng, x_0.shape)

    def _bc(t):
        return t.reshape(batch_size, 1, 1)

    # ── Time sampling ──
    rng, fm_rng, l1_rng, l2_rng = jax.random.split(rng, 4)
    t_fm = ln_sampler(fm_rng, (batch_size,), p_mean, p_std)
    ln_1 = ln_sampler(l1_rng, (batch_size,), p_mean_t, p_std_t)
    ln_2 = ln_sampler(l2_rng, (batch_size,), p_mean_r, p_std_r)
    t_mf = jnp.maximum(ln_1, ln_2)
    r_mf = jnp.minimum(ln_1, ln_2)

    # ── 1. Flow Matching branch ──
    z_t_fm = (1 - _bc(t_fm)) * x_0 + _bc(t_fm) * eps
    v_t_fm = eps - x_0  # target velocity

    if use_logvar:
        v_pred, lv_fm = model_fn(params, observation, z_t_fm, t_fm, t_fm, return_logvar=True)
    else:
        v_pred = model_fn(params, observation, z_t_fm, t_fm, t_fm, return_logvar=False)
        lv_fm = jnp.zeros(batch_size, dtype=v_pred.dtype)

    fm_mse, fm_lp = log_lv_loss(v_pred, v_t_fm, lv_fm)

    # ── 2. MeanFlow branch (JVP) ──
    z_t_mf = (1 - _bc(t_mf)) * x_0 + _bc(t_mf) * eps
    v_t_mf = eps - x_0
    v_t_mf = jax.lax.stop_gradient(v_t_mf)

    def dmf_model_fn(z_t, t, r):
        if use_logvar:
            return model_fn(params, observation, z_t, t, r, return_logvar=True)
        u = model_fn(params, observation, z_t, t, r, return_logvar=False)
        lv = jnp.zeros(batch_size, dtype=u.dtype)
        return u, lv

    primals = (z_t_mf, t_mf, r_mf)
    tangents = (v_t_mf, jnp.ones_like(t_mf), jnp.zeros_like(r_mf))
    (u, lv_mf), (du_dt, _) = jax.jvp(dmf_model_fn, primals, tangents)

    # Target flow map: u_tgt = v_t + (r - t) * du/dt
    u_tgt = v_t_mf + _bc(r_mf - t_mf) * du_dt
    u_tgt = jax.lax.stop_gradient(u_tgt)

    mf_mse, mf_lp = log_lv_loss(u, u_tgt, lv_mf)

    # ── Total loss ──
    loss = 0.5 * (jnp.mean(fm_lp) + jnp.mean(mf_lp))

    metrics = {
        "loss_total": loss,
        "loss_fm": jnp.mean(fm_mse),
        "loss_mf": jnp.mean(mf_mse),
        "loss_fm_logvar": jnp.mean(fm_lp),
        "loss_mf_logvar": jnp.mean(mf_lp),
        "t_fm_mean": jnp.mean(t_fm),
        "t_mf_mean": jnp.mean(t_mf),
        "r_mf_mean": jnp.mean(r_mf),
        "t_mf_r_mf_gap": jnp.mean(t_mf - r_mf),
        "du_dt_norm": jnp.mean(jnp.square(du_dt)),
        "u_norm": jnp.mean(jnp.square(u)),
        "v_t_mf_norm": jnp.mean(jnp.square(v_t_mf)),
    }

    if use_logvar:
        metrics["logvar_fm_mean"] = jnp.mean(lv_fm)
        metrics["logvar_mf_mean"] = jnp.mean(lv_mf)

    return loss, metrics


def compute_fm_loss(
    model_fn: Callable,
    params: Any,
    observation: dict,
    actions: jax.Array,
    action_mean: jax.Array,
    action_std: jax.Array,
    rng: jax.Array,
    p_mean: float = 0.0,
    p_std: float = 1.0,
    use_logvar: bool = True,
) -> tuple[jax.Array, dict]:
    """FM branch only: standard flow matching (no JVP).  Leaner JIT compile."""
    batch_size = actions.shape[0]
    x_0 = (actions - action_mean) / (action_std + 1e-8)

    rng, noise_rng, fm_rng = jax.random.split(rng, 3)
    eps = jax.random.normal(noise_rng, x_0.shape)
    t_fm = ln_sampler(fm_rng, (batch_size,), p_mean, p_std)

    def _bc(t):
        return t.reshape(batch_size, 1, 1)

    z_t_fm = (1 - _bc(t_fm)) * x_0 + _bc(t_fm) * eps
    v_t_fm = eps - x_0

    if use_logvar:
        v_pred, lv_fm = model_fn(params, observation, z_t_fm, t_fm, t_fm, return_logvar=True)
    else:
        v_pred = model_fn(params, observation, z_t_fm, t_fm, t_fm, return_logvar=False)
        lv_fm = jnp.zeros(batch_size, dtype=v_pred.dtype)

    fm_mse, fm_lp = log_lv_loss(v_pred, v_t_fm, lv_fm)
    loss = jnp.mean(fm_lp)

    metrics = {
        "loss_fm": jnp.mean(fm_mse),
        "loss_fm_logvar": jnp.mean(fm_lp),
        "t_fm_mean": jnp.mean(t_fm),
    }
    if use_logvar:
        metrics["logvar_fm_mean"] = jnp.mean(lv_fm)

    return loss, metrics


def compute_mf_loss(
    model_fn: Callable,
    params: Any,
    observation: dict,
    actions: jax.Array,
    action_mean: jax.Array,
    action_std: jax.Array,
    rng: jax.Array,
    p_mean_t: float = 0.4,
    p_std_t: float = 1.0,
    p_mean_r: float = -1.2,
    p_std_r: float = 1.0,
    use_logvar: bool = True,
) -> tuple[jax.Array, dict]:
    """MF branch only: MeanFlow loss with JVP.  Leaner JIT compile."""
    batch_size = actions.shape[0]
    x_0 = (actions - action_mean) / (action_std + 1e-8)

    rng, noise_rng, l1_rng, l2_rng = jax.random.split(rng, 4)
    eps = jax.random.normal(noise_rng, x_0.shape)
    ln_1 = ln_sampler(l1_rng, (batch_size,), p_mean_t, p_std_t)
    ln_2 = ln_sampler(l2_rng, (batch_size,), p_mean_r, p_std_r)
    t_mf = jnp.maximum(ln_1, ln_2)
    r_mf = jnp.minimum(ln_1, ln_2)

    def _bc(t):
        return t.reshape(batch_size, 1, 1)

    z_t_mf = (1 - _bc(t_mf)) * x_0 + _bc(t_mf) * eps
    v_t_mf = eps - x_0
    v_t_mf = jax.lax.stop_gradient(v_t_mf)

    def dmf_model_fn(z_t, t, r):
        if use_logvar:
            return model_fn(params, observation, z_t, t, r, return_logvar=True)
        u = model_fn(params, observation, z_t, t, r, return_logvar=False)
        lv = jnp.zeros(batch_size, dtype=u.dtype)
        return u, lv

    primals = (z_t_mf, t_mf, r_mf)
    tangents = (v_t_mf, jnp.ones_like(t_mf), jnp.zeros_like(r_mf))
    (u, lv_mf), (du_dt, _) = jax.jvp(dmf_model_fn, primals, tangents)

    u_tgt = v_t_mf + _bc(r_mf - t_mf) * du_dt
    u_tgt = jax.lax.stop_gradient(u_tgt)

    mf_mse, mf_lp = log_lv_loss(u, u_tgt, lv_mf)
    loss = jnp.mean(mf_lp)

    metrics = {
        "loss_mf": jnp.mean(mf_mse),
        "loss_mf_logvar": jnp.mean(mf_lp),
        "t_mf_mean": jnp.mean(t_mf),
        "r_mf_mean": jnp.mean(r_mf),
        "t_mf_r_mf_gap": jnp.mean(t_mf - r_mf),
        "du_dt_norm": jnp.mean(jnp.square(du_dt)),
        "u_norm": jnp.mean(jnp.square(u)),
        "v_t_mf_norm": jnp.mean(jnp.square(v_t_mf)),
    }
    if use_logvar:
        metrics["logvar_mf_mean"] = jnp.mean(lv_mf)

    return loss, metrics
