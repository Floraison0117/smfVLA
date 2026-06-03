"""
SMF 修改版 Pi0.5 模型。

基于 openpi 的 Pi0 模型，修改 time embedding 以支持 SplitMeanFlow 的双时间输入 (r, t)。
使用 concat [E(t), E(r)] + time_proj 作为 adarms_cond。

关键修改：
1. 新增 time_proj 参数: Linear(2*width, width)
2. 修改 embed_suffix: 支持 (r, t) 双时间输入
3. time_proj 初始化为 [I, 0]: 初始等价于原始 flow matching
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
from openpi.models.pi0 import posemb_sincos

if TYPE_CHECKING:
    pass


class Pi05SMF(pi0.Pi0):
    """
    SMF 修改版 Pi0.5。

    在原始 Pi0.5 基础上新增:
    - time_proj: Linear(2*width, width)，初始化为 [I, 0]
    - embed_suffix_smf: 支持 (r, t) 双时间输入
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        # 获取 action expert 的 width
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        width = action_expert_config.width

        # 新增 time_proj: concat([e_t, e_r]) → width
        # 初始化为 [I, 0]，使得初始时只依赖 t，不依赖 r
        self.time_proj = nnx.Linear(
            in_features=2 * width,
            out_features=width,
            rngs=rngs,
            kernel_init=_identity_zero_init(width),
        )

    @at.typecheck
    def embed_suffix_smf(
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
        SMF 版本的 embed_suffix。

        与原始 embed_suffix 的区别：
        - 接受两个时间参数 (r, t) 而非单个 timestep
        - time embedding = time_proj(concat([E(t), E(r)]))
        - time_proj 初始化为 [I, 0]，初始等价于原始模型

        Args:
            obs: 观测数据
            noisy_actions: noisy action chunk
            t: 结束时间 (对应原始 timestep)
            r: 起始时间 (SMF 新增)
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Action tokens
        action_tokens = self.action_in_proj(noisy_actions)

        # 计算 E(t) — 与原始 pi0.5 相同的 time embedding 路径
        time_emb_t = posemb_sincos(t, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb_t = self.time_mlp_in(time_emb_t)
        time_emb_t = nnx.swish(time_emb_t)
        time_emb_t = self.time_mlp_out(time_emb_t)
        time_emb_t = nnx.swish(time_emb_t)  # [B, width]

        # 计算 E(r) — 同样的 time embedding 路径
        time_emb_r = posemb_sincos(r, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb_r = self.time_mlp_in(time_emb_r)
        time_emb_r = nnx.swish(time_emb_r)
        time_emb_r = self.time_mlp_out(time_emb_r)
        time_emb_r = nnx.swish(time_emb_r)  # [B, width]

        # Concat + project: time_proj([E(t), E(r)])
        time_emb_concat = jnp.concatenate([time_emb_t, time_emb_r], axis=-1)  # [B, 2*width]
        adarms_cond = self.time_proj(time_emb_concat)  # [B, width]

        # Action expert tokens（不混入 time，time 通过 adarms_cond 注入）
        action_expert_tokens = action_tokens

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask, adarms_cond

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
        """兼容原始接口的 embed_suffix（推理时 r=0, t=timestep）。"""
        r = jnp.zeros_like(timestep)
        return self.embed_suffix_smf(obs, noisy_actions, t=timestep, r=r)

    @at.typecheck
    def embed_suffix_decte(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        t: at.Float[at.Array, " b"],
        r: at.Float[at.Array, " b"],
        encoder_depth: int = 6,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        list[at.Float[at.Array, "d b emb"] | None],
    ]:
        """
        Decoupled Time Embedding 版本的 embed_suffix。

        与 embed_suffix_smf 的区别：
        - 不使用 time_proj，直接使用 E(t) 和 E(r)
        - 返回 per-layer adarms_cond: layers 0..encoder_depth-1 用 e_t, 其余用 e_r
        - adarms_cond 形状为 [num_experts, depth, B, width]，适配 gemma scan

        Args:
            obs: 观测数据
            noisy_actions: noisy action chunk
            t: 结束时间
            r: 起始时间
            encoder_depth: encoder 层数（前 encoder_depth 层用 e_t, 后面用 e_r）
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Action tokens
        action_tokens = self.action_in_proj(noisy_actions)

        # 计算 E(t)
        time_emb_t = posemb_sincos(t, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb_t = self.time_mlp_in(time_emb_t)
        time_emb_t = nnx.swish(time_emb_t)
        time_emb_t = self.time_mlp_out(time_emb_t)
        time_emb_t = nnx.swish(time_emb_t)  # [B, width]

        # 计算 E(r)
        time_emb_r = posemb_sincos(r, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb_r = self.time_mlp_in(time_emb_r)
        time_emb_r = nnx.swish(time_emb_r)
        time_emb_r = self.time_mlp_out(time_emb_r)
        time_emb_r = nnx.swish(time_emb_r)  # [B, width]

        # 构造 per-layer adarms_cond
        # 获取总层数
        action_expert_config = _gemma.get_config(self.config.action_expert_variant)
        depth = action_expert_config.depth  # 18

        # 前 encoder_depth 层用 e_t，后 depth-encoder_depth 层用 e_r
        # shape: [depth, B, width]
        per_layer_cond = jnp.concatenate([
            jnp.broadcast_to(time_emb_t[None, :, :], (encoder_depth, *time_emb_t.shape)),
            jnp.broadcast_to(time_emb_r[None, :, :], (depth - encoder_depth, *time_emb_r.shape)),
        ], axis=0)

        # adarms_cond: [None(专家0), per_layer_cond(专家1)]
        adarms_cond = [None, per_layer_cond]

        # Action expert tokens
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
        time_conditioning: str = "concat",
        encoder_depth: int = 6,
        # SMF 完整参数（支持所有变体）
        step: int = 0,
        total_steps: int = 15000,
        use_curriculum: bool = False,
        delta_min: float = 0.05,
        delta_final: float = 1.0,
        delta_floor: float = 1e-3,
        teacher_fn: Any = None,
        use_anchor: bool = False,
        alpha_anchor: float = 0.0,
        anchor_delta_max: float = 0.3,
        teacher_model: Any = None,
        use_bpl: bool = False,
        alpha_bpl: float = 0.0,
    ) -> at.Float[at.Array, "*b ah"]:
        """
        使用 SplitMeanFlow loss 替代原始 flow matching loss。

        支持所有 SMF 变体（base, curr, decte, anchor, bpl, full）。

        Args:
            time_conditioning: "concat" (SMF-Base) 或 "decte" (Decoupled Time Embedding)
            encoder_depth: DecTE 模式下 encoder 层数
            step: 当前训练步数（curriculum 用）
            total_steps: 总训练步数
            use_curriculum: 是否使用 Curriculum Time Sampling
            delta_min/delta_final/delta_floor: curriculum 参数
            teacher_fn: frozen teacher（anchor loss 用）
            use_anchor: 是否使用 anchor loss
            alpha_anchor: anchor loss 权重
            anchor_delta_max: anchor loss 的 delta 上限
            teacher_model: frozen teacher model（BPL 用）
            use_bpl: 是否使用 BPL
            alpha_bpl: BPL 权重
        """
        from smf_vla.training.smf_loss import compute_full_smf_loss

        # 定义模型前向函数
        def model_fn(params, obs, noisy_actions, r, t):
            """前向传播：给定 (z_t, r, t, c) 预测平均速度 u_θ。"""
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
            if time_conditioning == "decte":
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_decte(
                    obs, noisy_actions, t=t, r=r, encoder_depth=encoder_depth
                )
            else:
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_smf(
                    obs, noisy_actions, t=t, r=r
                )
                adarms_cond = [None, adarms_cond]

            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=adarms_cond,
            )
            v = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return v

        action_mean = jnp.zeros(actions.shape[-1])
        action_std = jnp.ones(actions.shape[-1])

        loss, metrics = compute_full_smf_loss(
            model_fn=model_fn,
            params=None,
            observation=observation,
            actions=actions,
            action_mean=action_mean,
            action_std=action_std,
            rng=rng,
            step=step,
            total_steps=total_steps,
            flow_ratio=self.config.flow_ratio,
            use_curriculum=use_curriculum,
            delta_min=delta_min,
            delta_final=delta_final,
            delta_floor=delta_floor,
            teacher_fn=teacher_fn,
            use_anchor=use_anchor,
            alpha_anchor=alpha_anchor,
            anchor_delta_max=anchor_delta_max,
            teacher_model=teacher_model,
            use_bpl=use_bpl,
            alpha_bpl=alpha_bpl,
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
        time_conditioning: str = "concat",
        encoder_depth: int = 6,
    ) -> _model.Actions:
        """
        SMF 推理：支持 1-NFE 生成。

        1-NFE: z_0 = z_1 - u_θ(z_1, 0, 1, c)
        Multi-step: Euler 积分

        Args:
            time_conditioning: "concat" 或 "decte"
            encoder_depth: DecTE 模式下 encoder 层数
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 共享的 prefix KV cache 计算
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        dt = -1.0 / num_steps

        def step(carry):
            x_t, time = carry
            r_dummy = jnp.zeros(batch_size)
            if time_conditioning == "decte":
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_decte(
                    observation, x_t, t=jnp.broadcast_to(time, batch_size), r=r_dummy,
                    encoder_depth=encoder_depth,
                )
            else:
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_smf(
                    observation, x_t, t=jnp.broadcast_to(time, batch_size), r=r_dummy
                )
                adarms_cond = [None, adarms_cond]

            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_step = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
            positions_step = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_step,
                kv_cache=kv_cache,
                adarms_cond=adarms_cond,
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2 + 1e-6

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


def _identity_zero_init(width: int):
    """
    初始化 time_proj 为 [I, 0]。

    kernel 形状为 (2*width, width)。
    前半部分 (对应 e_t) 初始化为 identity matrix。
    后半部分 (对应 e_r) 初始化为 zero matrix。
    这样初始时 time_proj([e_t, e_r]) = e_t，等价于原始模型。
    """

    def init(key, shape, dtype=jnp.float32):
        # shape = (2*width, width)
        kernel = jnp.zeros(shape, dtype=dtype)
        # 前半部分 = identity
        kernel = kernel.at[:width, :].set(jnp.eye(width, dtype=dtype))
        return kernel

    return init


# 需要导入 gemma config
import openpi.models.gemma as _gemma  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Pi05SMFConfig(pi0_config.Pi0Config):
    """SMF 修改版 Pi0.5 配置。"""

    # SMF 特有参数
    flow_ratio: float = 0.3

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi05SMF":
        return Pi05SMF(self, rngs=nnx.Rngs(rng))
