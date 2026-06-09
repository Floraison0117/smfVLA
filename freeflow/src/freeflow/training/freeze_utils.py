"""
Parameter freezing utilities for FreeFlow.

Reuses patterns from smfVLA/snapflow for consistency.
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


# pi0.5 freeze/trainable patterns (from smfVLA)
FREEZE_PATTERNS = [
    "PaliGemma/img/**",           # SigLIP vision encoder
    "PaliGemma/llm/embedder/**",  # Token embedding
    "PaliGemma/llm/final_norm/scale",  # VLM final norm (no _1)
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
    # Action projection
    "action_in_proj/**",
    "action_out_proj/**",
    # Time MLP (time_proj and student_head not used in current implementation)
    "time_mlp_in/**",
    "time_mlp_out/**",
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
            # No pattern matched, default to frozen
            logger.warning(f"Parameter matched no pattern, defaulting to frozen: {path_str}")
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
        # Handle both Variable objects and raw arrays
        if hasattr(val, 'value'):
            arr = val.value
        else:
            arr = val

        # Get size
        if hasattr(arr, 'size'):
            n = arr.size
        elif isinstance(arr, tuple) and len(arr) == 2:
            # (kernel, bias) tuple
            n = sum(v.size if hasattr(v, 'size') else v.shape[0] if hasattr(v, 'shape') else 0 for v in arr)
        else:
            n = 0

        total_params += n
        path_str = path_to_string(path)

        # Extract component name (first 3 path levels)
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
    logger.info(f"Total params:     {total_params:>15,}")
    logger.info(f"Trainable params: {trainable_params:>15,} ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"Frozen params:    {frozen_params:>15,} ({frozen_params/total_params*100:.1f}%)")
    logger.info("-" * 60)

    logger.info("Trainable params (by component):")
    for comp, n in sorted(trainable_by_component.items()):
        logger.info(f"  {comp}: {n:>12,}")

    logger.info("Frozen params (by component):")
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


def get_trainable_params_from_state(
    state: nnx.State,
    mask: dict[tuple[str, ...], bool],
) -> dict:
    """
    Extract trainable parameters from state using mask.

    Args:
        state: Model state
        mask: Trainable mask from build_trainable_mask

    Returns:
        Dictionary of trainable parameters
    """
    trainable_params = {}
    flat = state.flat_state()

    for path, val in sorted(flat.items()):
        if mask.get(path, False):
            current = trainable_params
            for i, key in enumerate(path[:-1]):
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[path[-1]] = val.value

    return trainable_params


def apply_trainable_params_to_state(
    state: nnx.State,
    trainable_params: dict,
    mask: dict[tuple[str, ...], bool],
) -> nnx.State:
    """
    Apply trainable parameters back to state.

    Args:
        state: Model state
        trainable_params: Trainable parameters
        mask: Trainable mask

    Returns:
        Updated state
    """
    flat = state.flat_state()

    for path, val in sorted(flat.items()):
        if mask.get(path, False):
            # Find corresponding param in trainable_params
            current = trainable_params
            for key in path[:-1]:
                if key in current:
                    current = current[key]
                else:
                    break
            else:
                if path[-1] in current:
                    # Update the value in place
                    flat[path].value = current[path[-1]]

    return state


def freeze_model(model: nnx.Module) -> tuple[nnx.Module, dict, dict, dict]:
    """
    Freeze VLM backbone parameters in model.

    Returns:
        model: Model with frozen parameters
        mask: Trainable mask
        stats: Parameter statistics
        freeze_mask: JAX-compatible freeze mask
    """
    graphdef, state = nnx.split(model)
    mask = build_trainable_mask(state)
    stats = print_param_summary(state, mask)

    # Convert mask to JAX pytree (consistent with state structure)
    flat = state.flat_state()
    freeze_mask = {}
    for path, val in sorted(flat.items()):
        freeze_mask[path] = jnp.array(mask.get(path, False))

    return model, mask, stats, freeze_mask


def get_freeze_patterns(
    freeze_patterns: list[str],
    trainable_patterns: list[str],
) -> tuple[set[str], set[str]]:
    """
    Parse freeze/trainable patterns from config.

    Args:
        freeze_patterns: Glob patterns for frozen parameters
        trainable_patterns: Glob patterns for trainable parameters

    Returns:
        freeze_set: Set of frozen patterns
        trainable_set: Set of trainable patterns
    """
    freeze_set = set(freeze_patterns) if freeze_patterns else set(FREEZE_PATTERNS)
    trainable_set = set(trainable_patterns) if trainable_patterns else set(TRAINABLE_PATTERNS)

    return freeze_set, trainable_set


def get_default_freeze_patterns() -> set[str]:
    """Get default freeze patterns."""
    return set(FREEZE_PATTERNS)


def get_default_trainable_patterns() -> set[str]:
    """Get default trainable patterns."""
    return set(TRAINABLE_PATTERNS)
