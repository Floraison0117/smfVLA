"""
SplitMeanFlow 训练损失。

实现 SMF-Base 的损失函数：
- loss_smf: self-consistency loss（r < t 分支）
- loss_fm: flow matching loss（r = t 分支，防止退化）

参考 plan0515.md 中的 SplitMeanFlow 设计。
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class SMFSample(NamedTuple):
    """SMF 训练采样的中间结果。"""
    z_t: at.Float[at.Array, "b ah ad"]       # 插值后的 noisy action
    r: at.Float[at.Array, " b"]              # 起始时间
    t: at.Float[at.Array, " b"]              # 结束时间
    s: at.Float[at.Array, " b"]              # 中间时间 (self-consistency)
    z_s: at.Float[at.Array, "b ah ad"]       # 中间状态
    m: at.Bool[at.Array, " b"]               # Bernoulli mask (True=r=t)
    noise: at.Float[at.Array, "b ah ad"]     # 采样的噪声
    x_norm: at.Float[at.Array, "b ah ad"]    # 归一化的 action


def sample_r_t(
    rng: at.KeyArrayLike,
    batch_size: int,
    flow_ratio: float = 0.3,
) -> tuple[at.Float[at.Array, " b"], at.Float[at.Array, " b"], at.Bool[at.Array, " b"]]:
    """
    采样时间对 (r, t)，满足 0 <= r <= t <= 1。

    以概率 flow_ratio 设置 r = t（退化为普通 flow matching）。
    否则 r ~ Uniform(0, t)。

    Args:
        rng: JAX 随机数 key
        batch_size: batch 大小
        flow_ratio: 设置 r = t 的概率 (p)

    Returns:
        r, t, m (m=True 表示 r=t)
    """
    rng_t, rng_r, rng_m = jax.random.split(rng, 3)

    # t ~ Uniform(0, 1)
    t = jax.random.uniform(rng_t, (batch_size,), minval=0.0, maxval=1.0)

    # r ~ Uniform(0, t)
    r = jax.random.uniform(rng_r, (batch_size,), minval=0.0, maxval=1.0) * t

    # Bernoulli mask: 以概率 flow_ratio 设置 r = t
    m = jax.random.bernoulli(rng_m, p=flow_ratio, shape=(batch_size,))

    # m=True 时 r=t，否则保持 r < t
    r = jnp.where(m, t, r)

    return r, t, m


def sample_lambda_s(
    rng: at.KeyArrayLike,
    r: at.Float[at.Array, " b"],
    t: at.Float[at.Array, " b"],
) -> tuple[at.Float[at.Array, " b"], at.Float[at.Array, " b"]]:
    """
    采样 lambda 和中间时间 s。

    s = (1 - lambda) * t + lambda * r

    Returns:
        lambda, s
    """
    lam = jax.random.uniform(rng, r.shape, minval=0.0, maxval=1.0)
    s = (1 - lam) * t + lam * r
    return lam, s


def interpolate_z(
    x_norm: at.Float[at.Array, "b ah ad"],
    noise: at.Float[at.Array, "b ah ad"],
    t: at.Float[at.Array, " b"],
) -> at.Float[at.Array, "b ah ad"]:
    """
    线性插值: z_t = (1 - t) * x_norm + t * noise。

    注意: 这里使用与 pi0.5 相同的约定，t=0 是 clean，t=1 是 noise。
    """
    t_expanded = t[:, None, None]  # [B, 1, 1]
    z_t = (1 - t_expanded) * x_norm + t_expanded * noise
    return z_t


def compute_smf_loss(
    model_fn: Any,
    params: Any,
    observation: Any,
    actions: at.Float[at.Array, "b ah ad"],
    action_mean: at.Float[at.Array, " ad"],
    action_std: at.Float[at.Array, " ad"],
    rng: at.KeyArrayLike,
    flow_ratio: float = 0.3,
) -> tuple[at.Float[at.Array, ""], dict[str, at.Float[at.Array, ""]]]:
    """
    计算 SplitMeanFlow 总损失。

    loss_total = loss_smf + loss_fm

    其中:
    - loss_smf: self-consistency loss (r < t 分支)
    - loss_fm: flow matching loss (r = t 分支)

    Args:
        model_fn: 模型前向函数 f(params, observation, noisy_actions, r, t) → velocity
        params: 模型参数
        observation: 观测数据
        actions: ground-truth action chunk [B, action_horizon, action_dim]
        action_mean: action 均值 (用于归一化)
        action_std: action 标准差 (用于归一化)
        rng: JAX 随机数 key
        flow_ratio: r=t 的概率 p

    Returns:
        loss_total, metrics_dict
    """
    rng_sample, rng_noise, rng_lambda = jax.random.split(rng, 3)
    batch_size = actions.shape[0]

    # Step 1: 归一化 action
    x_norm = (actions - action_mean) / (action_std + 1e-8)

    # Step 2: 采样噪声 ε ~ N(0, I)
    noise = jax.random.normal(rng_noise, x_norm.shape)

    # Step 3: 采样时间 (r, t) 和 Bernoulli mask
    r, t, m = sample_r_t(rng_sample, batch_size, flow_ratio)

    # Step 4: 线性插值 z_t = (1-t) * x_norm + t * ε
    z_t = interpolate_z(x_norm, noise, t)

    # Step 5: 采样 lambda, 计算 s
    lam, s = sample_lambda_s(rng_lambda, r, t)

    # ── Self-consistency 分支 (r < t, m=False) ──────────────
    # u_2 = u_θ(z_t, s, t, c) — 从 s 到 t 的平均速度
    u_2 = model_fn(params, observation, z_t, s, t)

    # z_s = z_t - (t - s) * stop_gradient(u_2)
    z_s = z_t - (t - s)[:, None, None] * jax.lax.stop_gradient(u_2)

    # u_1 = u_θ(z_s, r, s, c) — 从 r 到 s 的平均速度
    u_1 = model_fn(params, observation, z_s, r, s)

    # target = (1-λ) * stop_gradient(u_1) + λ * stop_gradient(u_2)
    target_sc = (
        (1 - lam)[:, None, None] * jax.lax.stop_gradient(u_1)
        + lam[:, None, None] * jax.lax.stop_gradient(u_2)
    )

    # pred = u_θ(z_t, r, t, c)
    pred_sc = model_fn(params, observation, z_t, r, t)

    # loss_smf = mean(||pred - target||²) — 只在 r < t 的样本上计算
    loss_smf_per_sample = jnp.mean(jnp.square(pred_sc - target_sc), axis=(-2, -1))  # [B]
    # m=False (r < t) 的样本才计算 SMF loss
    loss_smf = jnp.sum(loss_smf_per_sample * (1 - m)) / jnp.maximum(jnp.sum(1 - m), 1)

    # ── Flow matching 分支 (r = t, m=True) ──────────────────
    # 当 r = t 时，u_θ(z_t, t, t, c) 应等于瞬时速度 ε - x_norm
    pred_fm = model_fn(params, observation, z_t, t, t)
    target_fm = noise - x_norm

    loss_fm_per_sample = jnp.mean(jnp.square(pred_fm - target_fm), axis=(-2, -1))  # [B]
    # m=True (r = t) 的样本才计算 FM loss
    loss_fm = jnp.sum(loss_fm_per_sample * m) / jnp.maximum(jnp.sum(m), 1)

    # ── 总损失 ─────────────────────────────────────────────
    loss_total = loss_smf + loss_fm

    metrics = {
        "loss_total": loss_total,
        "loss_smf": loss_smf,
        "loss_fm": loss_fm,
        "flow_ratio_actual": jnp.mean(m.astype(jnp.float32)),
        "t_mean": jnp.mean(t),
        "r_mean": jnp.mean(r),
        "delta_mean": jnp.mean(t - r),
    }

    return loss_total, metrics


def sample_r_t_curriculum(
    rng: at.KeyArrayLike,
    batch_size: int,
    step: int,
    total_steps: int,
    flow_ratio: float = 0.5,
    delta_min: float = 0.05,
    delta_final: float = 1.0,
    delta_floor: float = 1e-3,
    delta_sampling: str = "uniform",
) -> tuple[at.Float[at.Array, " b"], at.Float[at.Array, " b"], at.Bool[at.Array, " b"], dict[str, at.Float[at.Array, ""]]]:
    """
    Curriculum Time Sampling: 训练初期限制时间间隔，逐渐放宽。

    Args:
        rng: JAX 随机数 key
        batch_size: batch 大小
        step: 当前训练步数
        total_steps: 总训练步数
        flow_ratio: r=t 的概率 p
        delta_min: 最小时间间隔
        delta_final: 最终最大时间间隔
        delta_floor: 最小有效间隔
        delta_sampling: 采样策略，"uniform" 或 "biased"
            - "uniform": delta ~ Uniform(0, delta_upper)，原行为
            - "biased": delta = delta_upper * (1 - u^(1/k))，k=2，delta 偏向大值

    Returns:
        r, t, m, info_dict
    """
    rng_t, rng_delta, rng_m = jax.random.split(rng, 3)

    # 训练进度 rho ∈ [0, 1]
    rho = jnp.clip(step / jnp.maximum(total_steps, 1), 0.0, 1.0)

    # cosine curriculum: delta_max 从 delta_min 增长到 delta_final
    delta_max = delta_min + 0.5 * (delta_final - delta_min) * (1 - jnp.cos(jnp.pi * rho))

    # t ~ Uniform(0, 1)
    t = jax.random.uniform(rng_t, (batch_size,), minval=0.0, maxval=1.0)

    # delta_upper = min(t, delta_max)
    delta_upper = jnp.minimum(t, delta_max)

    if delta_sampling == "biased":
        # delta = delta_upper * (1 - u^(1/k))，k=2
        # 期望 ≈ 0.67 * delta_upper，delta 偏向大值
        u = jax.random.uniform(rng_delta, (batch_size,), minval=0.0, maxval=1.0)
        delta = delta_upper * (1.0 - jnp.power(u, 1.0 / 2.0))
        delta = jnp.maximum(delta, delta_floor)  # 确保 >= delta_floor
    else:
        # 原逻辑: delta ~ Uniform(delta_floor, delta_upper)
        delta = jax.random.uniform(rng_delta, (batch_size,), minval=delta_floor, maxval=1.0) * delta_upper
        delta = jnp.maximum(delta, delta_floor)  # 确保 >= delta_floor

    # r = t - delta
    r = t - delta
    r = jnp.maximum(r, 0.0)  # 确保 r >= 0

    # Bernoulli mask: 以概率 flow_ratio 设置 r = t
    m = jax.random.bernoulli(rng_m, p=flow_ratio, shape=(batch_size,))
    r = jnp.where(m, t, r)

    info = {
        "delta_max": delta_max,
        "delta_actual": t - r,
    }

    return r, t, m, info


def compute_anchor_loss(
    model_fn: Any,
    teacher_fn: Any,
    z_t: at.Float[at.Array, "b ah ad"],
    r: at.Float[at.Array, " b"],
    t: at.Float[at.Array, " b"],
    observation: Any,
    delta_max: float = 0.3,
    teacher_nfe: int = 2,
) -> tuple[at.Float[at.Array, ""], dict[str, at.Float[at.Array, ""]]]:
    """
    Anchor Loss: 用 frozen teacher 的 Euler 积分结果监督 student 的平均速度。

    只在 delta <= delta_max 的样本上启用。

    Args:
        model_fn: student 模型 f(params, obs, z, r, t) → velocity
        teacher_fn: frozen teacher 模型（同样接口）
        z_t: noisy action at time t
        r: 起始时间
        t: 结束时间
        observation: 观测数据
        delta_max: 启用 anchor loss 的最大时间间隔
        teacher_nfe: teacher Euler 积分步数

    Returns:
        loss_anchor, metrics_dict
    """
    delta = t - r
    active_mask = delta <= delta_max  # [B]

    # Teacher Euler 积分: 从 z_t 积分到 z_r
    dt_step = (r - t) / teacher_nfe  # 负值，从 t 到 r

    def euler_step(carry):
        z, time = carry
        u = teacher_fn(None, observation, z, r, time)
        return z + dt_step[:, None, None] * u, time + dt_step

    # 执行 teacher_nfe 步 Euler 积分
    z_r_teacher = z_t
    time = t
    for _ in range(teacher_nfe):
        z_r_teacher, time = euler_step((z_r_teacher, time))

    # u_teacher = (z_t - z_r_teacher) / (t - r)
    u_teacher = (z_t - z_r_teacher) / jnp.maximum((t - r)[:, None, None], 1e-8)

    # u_student = u_theta(z_t, r, t, c)
    u_student = model_fn(None, observation, z_t, r, t)

    # loss_anchor = mean(||u_student - stop_gradient(u_teacher)||^2) — 只在 active 样本上
    loss_per_sample = jnp.mean(jnp.square(u_student - jax.lax.stop_gradient(u_teacher)), axis=(-2, -1))
    loss_anchor = jnp.sum(loss_per_sample * active_mask) / jnp.maximum(jnp.sum(active_mask), 1)

    metrics = {
        "loss_anchor": loss_anchor,
        "anchor_active_ratio": jnp.mean(active_mask.astype(jnp.float32)),
    }

    return loss_anchor, metrics


def compute_bpl_loss(
    teacher_model: Any,
    x_pred: at.Float[at.Array, "b ah ad"],
    x_gt: at.Float[at.Array, "b ah ad"],
    observation: Any,
    layer_indices: tuple[int, ...] = (12, 16),
    layer_weights: tuple[float, ...] = (0.5, 1.0),
) -> tuple[at.Float[at.Array, ""], dict[str, at.Float[at.Array, ""]]]:
    """
    Behavioral Perceptual Loss (BPL): 用 frozen teacher 的中间层特征监督 student。

    Args:
        teacher_model: frozen teacher（需支持 extract_hidden_states）
        x_pred: student 的 1-step 预测 (归一化)
        x_gt: ground-truth action (归一化)
        observation: 观测数据
        layer_indices: 提取的中间层索引
        layer_weights: 各层权重

    Returns:
        loss_bpl, metrics_dict
    """
    # 提取 teacher 的 hidden states
    h_pred = teacher_model.extract_hidden_states(observation, x_pred, layer_indices)
    h_gt = teacher_model.extract_hidden_states(observation, x_gt, layer_indices)

    # 计算各层的 normalized MSE
    loss_bpl = jnp.float32(0.0)
    for i, (h_p, h_g, w) in enumerate(zip(h_pred, h_gt, layer_weights)):
        # L2 normalization
        h_p_norm = h_p / (jnp.linalg.norm(h_p, axis=-1, keepdims=True) + 1e-8)
        h_g_norm = h_g / (jnp.linalg.norm(h_g, axis=-1, keepdims=True) + 1e-8)
        layer_loss = jnp.mean(jnp.square(h_p_norm - h_g_norm))
        loss_bpl = loss_bpl + w * layer_loss

    metrics = {
        "loss_bpl": loss_bpl,
    }

    return loss_bpl, metrics


def compute_full_smf_loss(
    model_fn: Any,
    params: Any,
    observation: Any,
    actions: at.Float[at.Array, "b ah ad"],
    action_mean: at.Float[at.Array, " ad"],
    action_std: at.Float[at.Array, " ad"],
    rng: at.KeyArrayLike,
    step: int = 0,
    total_steps: int = 15000,
    # SMF 基础参数
    flow_ratio: float = 0.3,
    smf_loss_scale: float = 1.0,
    smf_scale_value: float = 1.0,
    # Curriculum 参数
    use_curriculum: bool = False,
    delta_min: float = 0.05,
    delta_final: float = 1.0,
    delta_floor: float = 1e-3,
    delta_sampling: str = "uniform",
    # Anchor 参数
    teacher_fn: Any = None,
    use_anchor: bool = False,
    alpha_anchor: float = 0.0,
    anchor_delta_max: float = 0.3,
    # BPL 参数
    teacher_model: Any = None,
    use_bpl: bool = False,
    alpha_bpl: float = 0.0,
) -> tuple[at.Float[at.Array, ""], dict[str, at.Float[at.Array, ""]]]:
    """
    统一的 SMF 损失函数，支持所有训练方法变体。

    loss_total = smf_scale * loss_smf + loss_fm + alpha_anchor * loss_anchor + alpha_bpl * loss_bpl

    当 smf_loss_scale == "dynamic" 时，使用 smf_scale_value（由 trainer 通过梯度匹配 EMA 计算）。
    当 smf_loss_scale 为 float 时，直接使用该值。

    Args:
        model_fn: student 模型 f(params, obs, z, r, t) → velocity
        params: 模型参数
        observation: 观测数据
        actions: ground-truth action chunk
        action_mean/action_std: 归一化参数
        rng: 随机数 key
        step: 当前训练步数
        total_steps: 总训练步数
        flow_ratio: r=t 的概率
        smf_loss_scale: SMF loss 的缩放系数，float 或 "dynamic"
        smf_scale_value: 动态 scale 值（仅当 smf_loss_scale=="dynamic" 时使用）
        use_curriculum: 是否使用 curriculum time sampling
        delta_min/delta_final/delta_floor: curriculum 参数
        delta_sampling: delta 采样策略，"uniform" 或 "biased"
        teacher_fn: frozen teacher 模型（anchor loss 用）
        use_anchor: 是否使用 anchor loss
        alpha_anchor: anchor loss 权重
        anchor_delta_max: anchor loss 的 delta 上限
        teacher_model: frozen teacher model（BPL 用）
        use_bpl: 是否使用 BPL
        alpha_bpl: BPL 权重

    Returns:
        loss_total, metrics_dict
    """
    rng_sample, rng_noise, rng_lambda = jax.random.split(rng, 3)
    batch_size = actions.shape[0]

    # Step 1: 归一化 action
    x_norm = (actions - action_mean) / (action_std + 1e-8)

    # Step 2: 采样噪声
    noise = jax.random.normal(rng_noise, x_norm.shape)

    # Step 3: 采样时间 (r, t)
    if use_curriculum:
        r, t, m, curriculum_info = sample_r_t_curriculum(
            rng_sample, batch_size, step, total_steps, flow_ratio,
            delta_min, delta_final, delta_floor, delta_sampling,
        )
    else:
        r, t, m = sample_r_t(rng_sample, batch_size, flow_ratio)
        curriculum_info = {"delta_max": jnp.float32(1.0), "delta_actual": t - r}

    # Step 4: 插值
    z_t = interpolate_z(x_norm, noise, t)

    # Step 5: 采样 lambda, s
    lam, s = sample_lambda_s(rng_lambda, r, t)

    # ── Self-consistency 分支 (r < t) ──────────────
    u_2 = model_fn(params, observation, z_t, s, t)
    z_s = z_t - (t - s)[:, None, None] * jax.lax.stop_gradient(u_2)
    u_1 = model_fn(params, observation, z_s, r, s)
    target_sc = (
        (1 - lam)[:, None, None] * jax.lax.stop_gradient(u_1)
        + lam[:, None, None] * jax.lax.stop_gradient(u_2)
    )
    pred_sc = model_fn(params, observation, z_t, r, t)
    loss_smf_per_sample = jnp.mean(jnp.square(pred_sc - target_sc), axis=(-2, -1))
    loss_smf = jnp.sum(loss_smf_per_sample * (1 - m)) / jnp.maximum(jnp.sum(1 - m), 1)

    # ── Flow matching 分支 (r = t) ──────────────────
    pred_fm = model_fn(params, observation, z_t, t, t)
    target_fm = noise - x_norm
    loss_fm_per_sample = jnp.mean(jnp.square(pred_fm - target_fm), axis=(-2, -1))
    loss_fm = jnp.sum(loss_fm_per_sample * m) / jnp.maximum(jnp.sum(m), 1)

    # ── 总损失 ─────────────────────────────────────
    # 确定实际使用的 scale
    is_dynamic = isinstance(smf_loss_scale, str) and smf_loss_scale == "dynamic"
    if is_dynamic:
        actual_scale = smf_scale_value
    else:
        actual_scale = float(smf_loss_scale)
    loss_smf_scaled = actual_scale * loss_smf
    loss_total = loss_smf_scaled + loss_fm

    metrics = {
        "loss_total": loss_total,
        "loss_smf": loss_smf,
        "loss_smf_scaled": loss_smf_scaled,
        "loss_fm": loss_fm,
        "smf_scale_applied": actual_scale,
        "flow_ratio_actual": jnp.mean(m.astype(jnp.float32)),
        "t_mean": jnp.mean(t),
        "r_mean": jnp.mean(r),
        "delta_mean": jnp.mean(t - r),
        "delta_max": curriculum_info["delta_max"],
    }

    # ── Anchor Loss ────────────────────────────────
    # 始终计算 anchor loss（避免 JIT 中 Python if 对 traced array 的 bool 转换）
    if use_anchor and teacher_fn is not None:
        loss_anchor, anchor_metrics = compute_anchor_loss(
            model_fn=model_fn,
            teacher_fn=teacher_fn,
            z_t=z_t,
            r=r,
            t=t,
            observation=observation,
            delta_max=anchor_delta_max,
        )
        loss_total = loss_total + alpha_anchor * loss_anchor
        metrics["loss_anchor"] = loss_anchor
        metrics["alpha_anchor"] = jnp.float32(alpha_anchor)
        metrics.update(anchor_metrics)
    else:
        metrics["loss_anchor"] = jnp.float32(0.0)
        metrics["alpha_anchor"] = jnp.float32(0.0)
        metrics["anchor_active_ratio"] = jnp.float32(0.0)

    # ── BPL Loss ───────────────────────────────────
    # 始终计算 BPL loss（避免 JIT 中 Python if 对 traced array 的 bool 转换）
    if use_bpl and teacher_model is not None:
        # 1-step 预测
        r_zero = jnp.zeros(batch_size)
        t_one = jnp.ones(batch_size)
        u_1step = model_fn(params, observation, z_t, r_zero, t_one)
        x_pred = noise - u_1step  # 1-step 预测（归一化空间）

        loss_bpl, bpl_metrics = compute_bpl_loss(
            teacher_model=teacher_model,
            x_pred=x_pred,
            x_gt=x_norm,
            observation=observation,
        )
        loss_total = loss_total + alpha_bpl * loss_bpl
        metrics["loss_bpl"] = loss_bpl
        metrics["alpha_bpl"] = jnp.float32(alpha_bpl)
        metrics.update(bpl_metrics)
    else:
        metrics["loss_bpl"] = jnp.float32(0.0)
        metrics["alpha_bpl"] = jnp.float32(0.0)

    metrics["loss_total"] = loss_total
    return loss_total, metrics


def compute_1nfe_actions(
    model_fn: Any,
    params: Any,
    observation: Any,
    noise: at.Float[at.Array, "b ah ad"],
    action_mean: at.Float[at.Array, " ad"],
    action_std: at.Float[at.Array, " ad"],
) -> at.Float[at.Array, "b ah ad"]:
    """
    1-NFE 推理: z_0 = z_1 - u_θ(z_1, 0, 1, c)

    Args:
        model_fn: 模型前向函数
        params: 模型参数
        observation: 观测数据
        noise: 初始噪声 z_1 ~ N(0, I)
        action_mean: action 均值 (用于反归一化)
        action_std: action 标准差 (用于反归一化)

    Returns:
        预测的 action chunk [B, action_horizon, action_dim]
    """
    # u_θ(z_1, 0, 1, c) — 从 0 到 1 的平均速度
    r = jnp.zeros(noise.shape[0])
    t = jnp.ones(noise.shape[0])
    u = model_fn(params, observation, noise, r, t)

    # z_0 = z_1 - u
    actions_norm = noise - u

    # 反归一化
    actions = actions_norm * (action_std + 1e-8) + action_mean

    return actions
