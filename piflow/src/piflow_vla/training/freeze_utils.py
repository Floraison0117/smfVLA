"""Parameter freeze/train utilities for pi-flow.

Freeze: VLM backbone (SigLIP + PaliGemma LLM).
Train: action expert _1 layers + projection layers + GMM heads.
"""

import fnmatch
import logging
from typing import Any, Sequence

import flax.nnx as nnx
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def path_to_string(path: tuple[str, ...]) -> str:
    return "/".join(path)


def matches_any_pattern(path_str: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path_str, p) for p in patterns)


# ── pi0.5 freeze/train patterns ──────────────────────────────────
# Freeze: VLM backbone
FREEZE_PATTERNS = [
    "PaliGemma/img/**",
    "PaliGemma/llm/embedder/**",
    "PaliGemma/llm/final_norm/scale",
    "PaliGemma/llm/layers/attn/q_einsum/w",
    "PaliGemma/llm/layers/attn/kv_einsum/w",
    "PaliGemma/llm/layers/attn/attn_vec_einsum/w",
    "PaliGemma/llm/layers/mlp/gating_einsum",
    "PaliGemma/llm/layers/mlp/linear",
    "PaliGemma/llm/layers/pre_attention_norm/scale",
    "PaliGemma/llm/layers/pre_ffw_norm/scale",
]

# Train: Action expert (_1) + projection layers + time MLP + GMM heads
TRAINABLE_PATTERNS = [
    "PaliGemma/llm/layers/attn/q_einsum_1/**",
    "PaliGemma/llm/layers/attn/kv_einsum_1/**",
    "PaliGemma/llm/layers/attn/attn_vec_einsum_1/**",
    "PaliGemma/llm/layers/mlp_1/**",
    "PaliGemma/llm/layers/pre_attention_norm_1/**",
    "PaliGemma/llm/layers/pre_ffw_norm_1/**",
    "PaliGemma/llm/final_norm_1/**",
    "action_in_proj/**",
    "time_mlp_in/**",
    "time_mlp_out/**",
    # GMM heads (pi-flow specific)
    "gmm_mean_proj/**",
    "gmm_logstd_proj/**",
    "gmm_logweight_proj/**",
]


def build_trainable_mask(
    state: nnx.State,
    freeze_patterns: Sequence[str] = FREEZE_PATTERNS,
    trainable_patterns: Sequence[str] = TRAINABLE_PATTERNS,
) -> dict[tuple[str, ...], bool]:
    """Build per-parameter trainable/frozen mask."""
    flat = state.flat_state()
    mask = {}

    for path, val in sorted(flat.items()):
        path_str = path_to_string(path)
        is_trainable = matches_any_pattern(path_str, trainable_patterns)
        is_frozen = matches_any_pattern(path_str, freeze_patterns)

        if is_trainable and is_frozen:
            logger.warning(f"Param matches both freeze and trainable: {path_str}, default trainable")
            is_frozen = False

        if not is_trainable and not is_frozen:
            logger.warning(f"Param unmatched, default frozen: {path_str}")
            is_frozen = True

        mask[path] = is_trainable and not is_frozen

    return mask


def print_param_summary(
    state: nnx.State,
    mask: dict[tuple[str, ...], bool],
) -> dict[str, Any]:
    """Print parameter freeze/train summary."""
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
        component = "/".join(path[:3]) if len(path) >= 3 else "/".join(path)

        if mask.get(path, False):
            trainable_params += n
            trainable_by_component[component] = trainable_by_component.get(component, 0) + n
        else:
            frozen_params += n
            frozen_by_component[component] = frozen_by_component.get(component, 0) + n

    logger.info("=" * 60)
    logger.info("Parameter Summary")
    logger.info("=" * 60)
    logger.info(f"Total:      {total_params:>15,}")
    logger.info(f"Trainable:  {trainable_params:>15,} ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"Frozen:     {frozen_params:>15,} ({frozen_params/total_params*100:.1f}%)")
    logger.info("-" * 60)

    logger.info("Trainable (by component):")
    for comp, n in sorted(trainable_by_component.items()):
        logger.info(f"  {comp}: {n:>12,}")

    logger.info("Frozen (by component):")
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
