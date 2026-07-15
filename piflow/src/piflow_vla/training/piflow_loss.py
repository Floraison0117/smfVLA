"""Pi-Flow velocity imitation loss: student GMFlow policy vs frozen teacher.

Multi-NFE training: the trajectory [1, 0] is split into `nfe` equal segments.
At each segment boundary, the student predicts a fresh GMM (at t_src), then
rolls out analytically within the segment. The frozen teacher provides velocity
supervision at query states within each segment.

Segment layout (nfe=4 example):
  t: 1.0 --seg0--> 0.75 --seg1--> 0.5 --seg2--> 0.25 --seg3--> 0.0
     GMM@t=1.0    GMM@t=0.75     GMM@t=0.5      GMM@t=0.25

Each segment advances x via stop_gradient(student_rollout_final), so gradients
flow only through the GMM params within each segment (not across segments).
"""

from typing import Callable

import jax
import jax.numpy as jnp
from piflow_vla.models import gmflow


def compute_piflow_loss(
    student_gmm_fn: Callable,
    teacher_vel_fn: Callable,
    observation: dict,
    actions: jax.Array,
    rng: jax.Array,
    nfe: int = 1,
    inner_substeps: int = 8,
    teacher_query_points: int = 4,
    stop_gradient_rollout: bool = True,
) -> tuple[jax.Array, dict]:
    """Compute pi-flow velocity imitation loss for arbitrary NFE.

    Args:
        student_gmm_fn: fn(obs, x_t, t) -> (means[B,K,H,D], log_stds[B,K], log_weights[B,K])
        teacher_vel_fn: fn(obs, states[B,M,H,D], times[B,M]) -> velocity [B,M,H,D]
        observation: model observation dict
        actions: ground truth actions [B,H,D] (used only for shape)
        rng: random key
        nfe: number of outer NFE steps (1, 2, 4). Each segment gets an equal
            share of inner_substeps and teacher_query_points.
        inner_substeps: total GMFlow Euler substeps across all segments.
            Divided equally: substeps_per_seg = max(2, inner_substeps // nfe).
        teacher_query_points: teacher query points per segment.
            Clamped to min(teacher_query_points, substeps_per_seg).
        stop_gradient_rollout: stop gradient through rollout states

    Returns:
        loss: scalar velocity imitation loss (averaged over segments)
        metrics: dict of diagnostic metrics
    """
    B, H, D = actions.shape

    # Outer time schedule: 1.0 -> 0.0 in nfe+1 points
    # Compute as Python floats (not jnp) so they stay concrete under JIT.
    # t_src for segment k = 1.0 - k/nfe, t_dst = 1.0 - (k+1)/nfe

    # Substeps and query points per segment
    substeps_per_seg = max(2, inner_substeps // nfe)
    effective_query_points = min(teacher_query_points, substeps_per_seg)
    query_idx = jnp.linspace(0, substeps_per_seg - 1, effective_query_points, dtype=jnp.int32)

    # Sample initial noise: x_1 ~ N(0, I)
    rng, noise_rng = jax.random.split(rng)
    x = jax.random.normal(noise_rng, (B, H, D))

    total_loss = 0.0
    total_student_vel_norm = 0.0
    total_teacher_vel_norm = 0.0
    total_vel_diff = 0.0

    for k in range(nfe):
        t_src = 1.0 - k / nfe
        t_dst = 1.0 - (k + 1) / nfe

        # Student GMM forward at t_src
        means, log_stds, log_weights = student_gmm_fn(observation, x, jnp.full((B,), t_src))

        # GMFlow rollout within segment [t_src, t_dst], recording query states+vels
        x_final, states, student_vels = gmflow.gmflow_rollout_with_states(
            x,
            means,
            log_stds,
            log_weights,
            num_substeps=substeps_per_seg,
            t_src=t_src,
            query_indices=query_idx,
            stop_gradient=stop_gradient_rollout,
            t_start=t_src,
            t_end=t_dst,
        )
        # states: [B, M, H, D], student_vels: [B, M, H, D]

        # Teacher velocity at query states
        seg_ts = jnp.linspace(t_src, t_dst, substeps_per_seg + 1)
        query_ts = seg_ts[query_idx]  # [M]
        query_times = jnp.repeat(query_ts[None, :], B, axis=0)  # [B, M]
        teacher_vels = teacher_vel_fn(observation, states, query_times)  # [B, M, H, D]
        teacher_vels = jax.lax.stop_gradient(teacher_vels)

        # Velocity imitation loss for this segment
        diff = student_vels - teacher_vels  # [B, M, H, D]
        seg_loss = jnp.mean(jnp.square(diff))

        total_loss = total_loss + seg_loss
        total_student_vel_norm = total_student_vel_norm + jnp.mean(jnp.square(student_vels))
        total_teacher_vel_norm = total_teacher_vel_norm + jnp.mean(jnp.square(teacher_vels))
        total_vel_diff = total_vel_diff + jnp.mean(jnp.square(diff))

        # Advance x to end of segment (detached for next segment)
        x = jax.lax.stop_gradient(x_final)

    # Average over segments
    loss = total_loss / nfe

    metrics = {
        "loss_total": loss,
        "student_vel_norm": total_student_vel_norm / nfe,
        "teacher_vel_norm": total_teacher_vel_norm / nfe,
        "vel_diff_norm": total_vel_diff / nfe,
        "means_norm": jnp.mean(jnp.square(means)),
        "log_stds_mean": jnp.mean(log_stds),
        "nfe": nfe,
        "substeps_per_seg": substeps_per_seg,
        "query_points_per_seg": effective_query_points,
    }

    return loss, metrics
