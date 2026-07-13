"""
参数冻结/过滤工具。

用于 pi0.5 模型的参数冻结：冻结 VLM backbone，只训练 action expert。
"""

import fnmatch
import logging
from typing import Any, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def path_to_string(path: tuple[str, ...]) -> str:
    """将 NNX 路径元组转为 '/' 分隔的字符串。"""
    return "/".join(path)


def matches_any_pattern(path_str: str, patterns: Sequence[str]) -> bool:
    """检查路径是否匹配任一 glob 模式。"""
    return any(fnmatch.fnmatch(path_str, p) for p in patterns)


# ── pi0.5 冻结/训练模式 ─────────────────────────────────────
# 冻结: VLM backbone (SigLIP + VLM LLM + shared embedder)
FREEZE_PATTERNS = [
    "PaliGemma/img/**",           # SigLIP 视觉编码器
    "PaliGemma/llm/embedder/**",  # Token embedding
    "PaliGemma/llm/final_norm/scale",  # VLM final norm (无 _1)
    # VLM attention (无 _1 后缀)
    "PaliGemma/llm/layers/attn/q_einsum/w",
    "PaliGemma/llm/layers/attn/kv_einsum/w",
    "PaliGemma/llm/layers/attn/attn_vec_einsum/w",
    # VLM MLP (无 _1 后缀)
    "PaliGemma/llm/layers/mlp/gating_einsum",
    "PaliGemma/llm/layers/mlp/linear",
    # VLM norm (无 _1 后缀)
    "PaliGemma/llm/layers/pre_attention_norm/scale",
    "PaliGemma/llm/layers/pre_ffw_norm/scale",
]

# 训练: Action expert + 投影层 + time MLP
TRAINABLE_PATTERNS = [
    # Action expert attention (_1 后缀)
    "PaliGemma/llm/layers/attn/q_einsum_1/**",
    "PaliGemma/llm/layers/attn/kv_einsum_1/**",
    "PaliGemma/llm/layers/attn/attn_vec_einsum_1/**",
    # Action expert MLP (_1 后缀)
    "PaliGemma/llm/layers/mlp_1/**",
    # Action expert norm (_1 后缀)
    "PaliGemma/llm/layers/pre_attention_norm_1/**",
    "PaliGemma/llm/layers/pre_ffw_norm_1/**",
    # Action expert final norm
    "PaliGemma/llm/final_norm_1/**",
    # Action 投影层
    "action_in_proj/**",
    "action_out_proj/**",
    # Time MLP (base pi0.5，DMF 复用算 E(t)/E(r))
    "time_mlp_in/**",
    "time_mlp_out/**",
    # DMF logvar 头
    "logvar_proj/**",
]


def build_trainable_mask(
    state: nnx.State,
    freeze_patterns: Sequence[str] = FREEZE_PATTERNS,
    trainable_patterns: Sequence[str] = TRAINABLE_PATTERNS,
) -> dict[tuple[str, ...], bool]:
    """
    为每个参数构建 trainable/frozen 标记。

    Returns:
        dict mapping NNX path tuple → True (trainable) / False (frozen)
    """
    flat = state.flat_state()
    mask = {}

    for path, val in sorted(flat.items()):
        path_str = path_to_string(path)

        is_trainable = matches_any_pattern(path_str, trainable_patterns)
        is_frozen = matches_any_pattern(path_str, freeze_patterns)

        if is_trainable and is_frozen:
            logger.warning(f"参数同时匹配 freeze 和 trainable: {path_str}，默认 trainable")
            is_frozen = False

        if not is_trainable and not is_frozen:
            # 未匹配任何模式，默认冻结
            logger.warning(f"参数未匹配任何模式，默认冻结: {path_str}")
            is_frozen = True

        mask[path] = is_trainable and not is_frozen

    return mask


def print_param_summary(
    state: nnx.State,
    mask: dict[tuple[str, ...], bool],
) -> dict[str, Any]:
    """打印参数冻结/训练统计。"""
    flat = state.flat_state()

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    trainable_by_component: dict[str, int] = {}
    frozen_by_component: dict[str, int] = {}

    for path, val in sorted(flat.items()):
        n = val.value.size
        total_params += n
        path_str = path_to_string(path)

        # 提取组件名（前 3 层路径）
        component = "/".join(path[:3]) if len(path) >= 3 else "/".join(path)

        if mask.get(path, False):
            trainable_params += n
            trainable_by_component[component] = trainable_by_component.get(component, 0) + n
        else:
            frozen_params += n
            frozen_by_component[component] = frozen_by_component.get(component, 0) + n

    logger.info("=" * 60)
    logger.info("参数统计")
    logger.info("=" * 60)
    logger.info(f"总参数量:     {total_params:>15,}")
    logger.info(f"训练参数量:   {trainable_params:>15,} ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"冻结参数量:   {frozen_params:>15,} ({frozen_params/total_params*100:.1f}%)")
    logger.info("-" * 60)

    logger.info("训练参数 (按组件):")
    for comp, n in sorted(trainable_by_component.items()):
        logger.info(f"  {comp}: {n:>12,}")

    logger.info("冻结参数 (按组件):")
    for comp, n in sorted(frozen_by_component.items()):
        logger.info(f"  {comp}: {n:>12,}")

    logger.info("=" * 60)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_by_component": trainable_by_component,
        "frozen_by_component": frozen_by_component,
    }


def freeze_model(model: nnx.Module) -> nnx.Module:
    """
    冻结模型中的 VLM backbone 参数。

    将冻结参数的梯度设为 None，使其不参与反向传播。
    返回 trainable_mask 可用于 optax 的参数更新过滤。
    """
    graphdef, state = nnx.split(model)
    mask = build_trainable_mask(state)
    stats = print_param_summary(state, mask)

    # 将 mask 转为 JAX pytree（与 state 结构一致）
    flat = state.flat_state()
    freeze_mask = {}
    for path, val in sorted(flat.items()):
        freeze_mask[path] = jnp.array(mask.get(path, False))

    return model, mask, stats, freeze_mask
