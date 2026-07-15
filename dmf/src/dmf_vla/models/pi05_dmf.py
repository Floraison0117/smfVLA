"""
DMF (Decoupled MeanFlow) for π₀.₅ VLA —— 真·逐层 encoder/decoder 解耦实现。

按 DMF 论文 / PyTorch 参考 (dmf/models/dmft.py)：
- 模型输出 flow map u(x, t, r)。
- action-expert 的 18 层**手动展开**（不用 scan）：前 dmf_depth 层（encoder）cond on E(t)，
  其余层（decoder）cond on E(r)——通过 per-layer 3D adarms_cond [depth,B,width] 实现
  （gemma.__call__ 检测 3D cond 自动走 forward_with_intermediates 逐层路径）。
- E(t)/E(r) 复用 base pi0.5 的 time_mlp_in/out（完美 warm start）。
- loss 与 sample 共用 _dmf_forward（消除 train/sample 不一致）。
- 全前向（kv_cache=None，不用增量解码）；sample 每 Euler 步一次全前向（1-NFE=1 次前向，无额外开销）。
- 采样：Euler  x_{k+1} = x_k + (r-t)·u(x_k, t, r)；1-NFE: x_0 = noise - u(noise, 1, 0)。
- logvar 从模型隐藏状态 + 时间嵌入预测（对齐官方 DMFT 的 return_logvar 路径）。
- 无 model guidance（g_type="default"）。

参考: dmf/loss.py, dmf/samplers.py, dmf/models/dmft.py。
"""

import dataclasses
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at
from openpi.models.pi0 import posemb_sincos

import openpi.models.gemma as _gemma  # noqa: E402


def _zero_init(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype=dtype)


@dataclasses.dataclass(frozen=True)
class Pi05DMFConfig(pi0_config.Pi0Config):
    """DMF 配置（逐层 encoder/decoder 解耦）。"""

    dmf_depth_ratio: float = 0.67  # encoder 占 action-expert 总层数的比例
    use_logvar: bool = True

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi05DMF":
        return Pi05DMF(self, rngs=nnx.Rngs(rng))


class Pi05DMF(pi0.Pi0):
    """DMF π₀.₅：逐层 encoder(cond t) / decoder(cond r) 解耦的 flow map。"""

    def __init__(self, config: Pi05DMFConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self._dmf_config = config

        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self._action_depth = action_expert_config.depth
        self.dmf_depth = int(action_expert_config.depth * config.dmf_depth_ratio)

        width = action_expert_config.width
        if config.use_logvar:
            # logvar from hidden state [width] + E(t) [width] + E(r) [width] = 3*width
            self.logvar_proj = nnx.Linear(
                in_features=3 * width, out_features=1, rngs=rngs,
                kernel_init=_zero_init, bias_init=_zero_init,
            )
        else:
            self.logvar_proj = None

    def _time_embed(self, time):
        """time:[B] -> [B,width]，复用 base：swish(time_mlp_out(swish(time_mlp_in(sincos(time)))))。"""
        e = posemb_sincos(time, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        e = nnx.swish(self.time_mlp_in(e))
        e = nnx.swish(self.time_mlp_out(e))
        return e

    def _cond_stack(self, t, r):
        """per-layer cond 栈 [depth,B,width]：前 dmf_depth 层(encoder)=E(t)，其余(decoder)=E(r)。"""
        e_t = self._time_embed(t)  # [B, width]
        e_r = self._time_embed(r)
        idx = jnp.arange(self._action_depth)[:, None, None]  # [depth,1,1]
        return jnp.where(idx < self.dmf_depth, e_t[None, :, :], e_r[None, :, :])  # [depth,B,width]

    def _embed_suffix_tokens(self, noisy_actions):
        action_tokens = self.action_in_proj(noisy_actions)
        input_mask = jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = jnp.array([True] + [False] * (self.action_horizon - 1))
        return action_tokens, input_mask, ar_mask

    def _dmf_forward(self, obs, noisy_actions, t, r, *, prefix_tokens=None, prefix_mask=None,
                     prefix_ar_mask=None, prefix_kv_cache=None, return_logvar=False):
        """共享前向（loss/sample 同源）：flow map u(x,t,r)。

        prefix_kv_cache=None 时走统一前向，非 None 时仅过 action expert（suffix-only +
        注入 freeze 的 prefix KV cache），JVP 不穿透 VLM backbone。
        """
        cond_stack = self._cond_stack(t, r)  # [depth,B,width]
        suffix_tokens, suffix_mask, suffix_ar_mask = self._embed_suffix_tokens(noisy_actions)
        if prefix_tokens is None:
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        if prefix_kv_cache is None:
            # Original unified forward (backward-compat)
            (_, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                positions=positions, mask=attn_mask, adarms_cond=[None, cond_stack],
            )
        else:
            # Suffix-only forward: action expert reads frozen prefix KV cache.
            # Slice positions & mask to suffix-only query length (Q only has suffix tokens).
            p_len = prefix_tokens.shape[1]
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                positions=positions[:, p_len:], mask=attn_mask[:, p_len:, :],
                adarms_cond=[None, cond_stack], kv_cache=prefix_kv_cache,
            )
        u = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        if return_logvar and self.logvar_proj is not None:
            # Predict logvar from last action token's hidden state + time embeddings
            last_hidden = suffix_out[:, -1]  # [B, width]
            logvar = self.logvar_proj(
                jnp.concatenate([last_hidden, self._time_embed(t), self._time_embed(r)], axis=-1)
            ).squeeze(-1)
            return u, logvar
        return u

    def _dmf_model_fn(self, prefix_tokens, prefix_mask, prefix_ar_mask):
        """compute_dmf_loss 用的 model_fn。

        预计算 prefix KV cache（stop_gradient），使得 JVP 仅穿透 action expert，
        不穿过 3B VLM backbone。
        """
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        _, prefix_kv = self.PaliGemma.llm(
            [prefix_tokens, None],
            positions=prefix_positions, mask=prefix_attn_mask,
            adarms_cond=[None, None],
        )
        prefix_kv = jax.lax.stop_gradient(prefix_kv)

        def model_fn(params, obs, noisy_actions, t, r, return_logvar=False):
            return self._dmf_forward(
                obs, noisy_actions, t, r,
                prefix_tokens=prefix_tokens, prefix_mask=prefix_mask,
                prefix_ar_mask=prefix_ar_mask, prefix_kv_cache=prefix_kv,
                return_logvar=return_logvar,
            )
        return model_fn

    @override
    def compute_loss(self, rng, observation, actions, *, train: bool = False, **kwargs):
        from dmf_vla.training.dmf_loss import compute_dmf_loss

        action_mean = kwargs.get("action_mean", jnp.zeros(actions.shape[-1]))
        action_std = kwargs.get("action_std", jnp.ones(actions.shape[-1]))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_tokens = jax.lax.stop_gradient(prefix_tokens)
        prefix_mask = jax.lax.stop_gradient(prefix_mask)
        prefix_ar_mask = jax.lax.stop_gradient(prefix_ar_mask)

        model_fn = self._dmf_model_fn(prefix_tokens, prefix_mask, prefix_ar_mask)
        loss, _metrics = compute_dmf_loss(
            model_fn=model_fn, params=None,
            observation=observation, actions=actions,
            action_mean=action_mean, action_std=action_std,
            rng=rng, use_logvar=self._dmf_config.use_logvar,
        )
        return loss

    @override
    def sample_actions(self, rng, observation, *, num_steps=1, noise=None):
        """DMF Euler 采样：x_{k+1}=x_k+(r-t)·u(x_k,t,r)。prefix KV 预计算一次。"""
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Precompute prefix KV once (reused across all Euler steps)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        _, prefix_kv = self.PaliGemma.llm(
            [prefix_tokens, None],
            positions=prefix_positions, mask=prefix_attn_mask,
            adarms_cond=[None, None],
        )

        t_steps = jnp.linspace(1.0, 0.0, num_steps + 1)

        def step(k, x):
            t_cur, t_nxt = t_steps[k], t_steps[k + 1]
            u = self._dmf_forward(
                observation, x,
                jnp.full((batch_size,), t_cur), jnp.full((batch_size,), t_nxt),
                prefix_tokens=prefix_tokens, prefix_mask=prefix_mask,
                prefix_ar_mask=prefix_ar_mask, prefix_kv_cache=prefix_kv,
            )
            return x + (t_nxt - t_cur) * u

        x = noise
        for k in range(num_steps):
            x = step(k, x)
        return x
