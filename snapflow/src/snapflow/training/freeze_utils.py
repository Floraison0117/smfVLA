"""
Parameter freezing/filtering for SnapFlow.

Adapted from smfVLA for SnapFlow training.
VLM backbone is frozen; only action expert + target_time_mlp are trainable.
"""

import fnmatch
import logging
from typing import Any, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def path_to_string(path: tuple[str, ...]) -> str:
    """Convert NNX path tuple to '/'-separated string."""
    return "/".join(path)


def matches_any_pattern(path_str: str, patterns: Sequence[str]) -> bool:
    """Check if path matches any glob pattern."""
    return any(fnmatch.fnmatch(path_str, p) for p in patterns)


# ── pi0.5 freeze/trainable patterns ─────────────────────────────
# Frozen: VLM backbone (SigLIP + VLM LLM + shared embedder)
FREEZE_PATTERNS = [
    "PaliGemma/img/**",           # SigLIP vision encoder
    "PaliGemma/llm/embedder/**",  # Token embedding
    "PaliGemma/llm/final_norm/scale",  # VLM final norm (no _1 suffix)
    # VLM attention (no _1 suffix)
    "PaliGemma/llm/layers/attn/q_einsum/w",
    "PaliGemma/llm/layers/attn/kv_einsum/w",
    "PaliGemma/llm/layers/attn/attn_vec_einsum/w",
    # VLM MLP (no _1 suffix)
    "PaliGemma/llm/layers/mlp/gating_einsum",
    "PaliGemma/llm/layers/mlp/linear",
    # VLM norm (no _1 suffix)
    "PaliGemma/llm/layers/pre_attention_norm/scale",
    "PaliGemma/llm/layers/pre_ffw_norm/scale",
]

# Trainable: Action expert + projection layers + time MLP + target_time_mlp
TRAINABLE_PATTERNS = [
    # Action expert attention (_1 suffix)
    "PaliGemma/llm/layers/attn/q_einsum_1/**",
    "PaliGemma/llm/layers/attn/kv_einsum_1/**",
    "PaliGemma/llm/layers/attn/attn_vec_einsum_1/**",
    # Action expert MLP (_1 suffix)
    "PaliGemma/llm/layers/mlp_1/**",
    # Action expert norm (_1 suffix)
    "PaliGemma/llm/layers/pre_attention_norm_1/**",
    "PaliGemma/llm/layers/pre_ffw_norm_1/**",
    # Action expert final norm
    "PaliGemma/llm/final_norm_1/**",
    # Action projection layers
    "action_in_proj/**",
    "action_out_proj/**",
    # Time MLP
    "time_mlp_in/**",
    "time_mlp_out/**",
    # Time projection (SMF-style time embedding)
    "time_proj/**",
    # NEW: SnapFlow target-time MLP (must be trainable!)
    "target_time_mlp/**",
]


def build_trainable_mask(
    state: nnx.State,
    freeze_patterns: Sequence[str] = FREEZE_PATTERNS,
    trainable_patterns: Sequence[str] = TRAINABLE_PATTERNS,
) -> dict[tuple[str, ...], bool]:
    """
    Build trainable/frozen mask for each parameter.

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
            logger.warning(f"Parameter matches both freeze and trainable: {path_str}, defaulting to trainable")
            is_frozen = False

        if not is_trainable and not is_frozen:
            # Unmatched parameters default to frozen
            logger.warning(f"Parameter unmatched by any pattern, defaulting to frozen: {path_str}")
            is_frozen = True

        mask[path] = is_trainable and not is_frozen

    return mask


def print_param_summary(
    state: nnx.State,
    mask: dict[tuple[str, ...], bool],
) -> dict[str, Any]:
    """Print parameter freezing/training statistics."""
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

        # Extract component name (first 3 levels of path)
        component = "/".join(path[:3]) if len(path) >= 3 else "/".join(path)

        if mask.get(path, False):
            trainable_params += n
            trainable_by_component[component] = trainable_by_component.get(component, 0) + n
        else:
            frozen_params += n
            frozen_by_component[component] = frozen_by_component.get(component, 0) + n

    logger.info("=" * 60)
    logger.info("Parameter Statistics")
    logger.info("=" * 60)
    logger.info(f"Total parameters:     {total_params:>15,}")
    logger.info(f"Trainable parameters: {trainable_params:>15,} ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"Frozen parameters:     {frozen_params:>15,} ({frozen_params/total_params*100:.1f}%)")
    logger.info("-" * 60)

    logger.info("Trainable parameters (by component):")
    for comp, n in sorted(trainable_by_component.items()):
        logger.info(f"  {comp}: {n:>12,}")

    logger.info("Frozen parameters (by component):")
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
    Freeze VLM backbone parameters in the model.

    Sets frozen parameters' gradients to None so they don't participate in backprop.
    Returns trainable_mask usable with optax for parameter update filtering.
    """
    graphdef, state = nnx.split(model)
    mask = build_trainable_mask(state)
    stats = print_param_summary(state, mask)

    # Convert mask to JAX pytree (structure matches state)
    flat = state.flat_state()
    freeze_mask = {}
    for path, val in sorted(flat.items()):
        freeze_mask[path] = jnp.array(mask.get(path, False))

    return model, mask, stats, freeze_mask
