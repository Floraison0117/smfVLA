#!/usr/bin/env python3
"""Merge LoRA adapters into base weights, producing a non-LoRA checkpoint.

The InternRobotics pi0.5 LoRA fine-tune (formerly at
``checkpoints/pi05_calvin/29999``) was trained with
``paligemma_variant="gemma_2b_lora"`` + ``action_expert_variant="gemma_300m_lora"``
on InternRobotics/InternData-Calvin_ABC for 30k steps.  During that LoRA
fine-tune the ``.*llm.*`` base weights were frozen; only the low-rank adapters
(``lora_a``/``lora_b``) carry the CALVIN adaptation, together with the
fully-trainable non-LLM weights (SigLIP image encoder, projection layers,
time-MLP).

DMF and Pi-Flow use the *non-LoRA* model variant (``gemma_2b`` +
``gemma_300m``), which has 51 parameter keys and no ``lora_a``/``lora_b``
leaves.  Loading a LoRA checkpoint directly would silently drop all 20 LoRA
adapter keys, discarding the CALVIN domain adaptation in the LLM layers.

This script merges each LoRA adapter back into its base weight::

    w_merged = w_base + matmul(lora_a, lora_b) * scaling_value

Both LoRA configs use ``alpha == rank`` (rank 16 for the 2B VLM, rank 32 for
the 300M action expert), so ``scaling_value = alpha / rank = 1.0`` everywhere.

The output checkpoint has the same 51-key structure as ``pi05_libero`` and is
loadable by ``openpi.models.model.restore_params`` (wrapped under a top-level
``"params"`` key).

The merge has already been performed: the merged non-LoRA checkpoint now lives
at ``checkpoints/pi05_calvin`` and the original LoRA source has been removed.
Usage (only relevant if the LoRA checkpoint is still present)::

    python scripts/merge_lora_checkpoint.py \\
        --src checkpoints/pi05_calvin/29999 \\
        --dst checkpoints/pi05_calvin
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import flax.traverse_util as traverse_util
import numpy as np

# ---------------------------------------------------------------------------
# LoRA merge mapping
# ---------------------------------------------------------------------------
#
# Each entry maps a base-weight tuple-key suffix to the corresponding
# ``lora_a`` / ``lora_b`` tuple-key suffixes.  The suffixes are joined onto the
# shared prefix ``PaliGemma/llm/layers`` (flattened with ``"/"`` separator).
#
# Attention einsums store the base weight under ``<module>/w`` and the adapters
# under ``<module>/lora_a`` / ``<module>/lora_b``.  The FFN modules store the
# base weights directly (``gating_einsum``, ``linear``) and the adapters under
# ``<weight>_lora_a`` / ``<weight>_lora_b``.

_LLM_PREFIX = "PaliGemma/llm/layers"

# (base_suffix, lora_a_suffix, lora_b_suffix)
_LORA_GROUPS: list[tuple[str, str, str]] = [
    # Attention einsums (base weight = <module>/w, adapters = <module>/lora_a|b)
    (
        f"{_LLM_PREFIX}/attn/q_einsum/w",
        f"{_LLM_PREFIX}/attn/q_einsum/lora_a",
        f"{_LLM_PREFIX}/attn/q_einsum/lora_b",
    ),
    (
        f"{_LLM_PREFIX}/attn/kv_einsum/w",
        f"{_LLM_PREFIX}/attn/kv_einsum/lora_a",
        f"{_LLM_PREFIX}/attn/kv_einsum/lora_b",
    ),
    (
        f"{_LLM_PREFIX}/attn/attn_vec_einsum/w",
        f"{_LLM_PREFIX}/attn/attn_vec_einsum/lora_a",
        f"{_LLM_PREFIX}/attn/attn_vec_einsum/lora_b",
    ),
    # Action-expert attention einsums (_1 suffix)
    (
        f"{_LLM_PREFIX}/attn/q_einsum_1/w",
        f"{_LLM_PREFIX}/attn/q_einsum_1/lora_a",
        f"{_LLM_PREFIX}/attn/q_einsum_1/lora_b",
    ),
    (
        f"{_LLM_PREFIX}/attn/kv_einsum_1/w",
        f"{_LLM_PREFIX}/attn/kv_einsum_1/lora_a",
        f"{_LLM_PREFIX}/attn/kv_einsum_1/lora_b",
    ),
    (
        f"{_LLM_PREFIX}/attn/attn_vec_einsum_1/w",
        f"{_LLM_PREFIX}/attn/attn_vec_einsum_1/lora_a",
        f"{_LLM_PREFIX}/attn/attn_vec_einsum_1/lora_b",
    ),
    # FFN (base weight = <weight>, adapters = <weight>_lora_a|b)
    (
        f"{_LLM_PREFIX}/mlp/gating_einsum",
        f"{_LLM_PREFIX}/mlp/gating_einsum_lora_a",
        f"{_LLM_PREFIX}/mlp/gating_einsum_lora_b",
    ),
    (
        f"{_LLM_PREFIX}/mlp/linear",
        f"{_LLM_PREFIX}/mlp/linear_lora_a",
        f"{_LLM_PREFIX}/mlp/linear_lora_b",
    ),
    # Action-expert FFN (_1 suffix)
    (
        f"{_LLM_PREFIX}/mlp_1/gating_einsum",
        f"{_LLM_PREFIX}/mlp_1/gating_einsum_lora_a",
        f"{_LLM_PREFIX}/mlp_1/gating_einsum_lora_b",
    ),
    (
        f"{_LLM_PREFIX}/mlp_1/linear",
        f"{_LLM_PREFIX}/mlp_1/linear_lora_a",
        f"{_LLM_PREFIX}/mlp_1/linear_lora_b",
    ),
]


def _split_key(slash_key: str) -> tuple:
    """Convert a ``"a/b/c"`` string key into the tuple form used by traverse_util."""
    return tuple(slash_key.split("/"))


def merge_lora_params(flat_params: dict[tuple, np.ndarray]) -> dict[tuple, np.ndarray]:
    """Merge LoRA adapters into base weights and remove the adapter keys.

    Args:
        flat_params: flattened params dict (tuple keys -> ndarray), as produced
            by ``traverse_util.flatten_dict``.

    Returns:
        New flattened dict with LoRA adapters merged and removed.
    """
    # Work on a copy so the caller's dict is untouched.
    merged: dict[tuple, np.ndarray] = {k: v for k, v in flat_params.items()}

    for base_suffix, a_suffix, b_suffix in _LORA_GROUPS:
        base_key = _split_key(base_suffix)
        a_key = _split_key(a_suffix)
        b_key = _split_key(b_suffix)

        if base_key not in merged:
            raise KeyError(f"Base weight not found in checkpoint: {base_suffix}")
        if a_key not in merged:
            raise KeyError(f"lora_a not found in checkpoint: {a_suffix}")
        if b_key not in merged:
            raise KeyError(f"lora_b not found in checkpoint: {b_suffix}")

        w_base = np.asarray(merged[base_key])
        w_a = np.asarray(merged[a_key])
        w_b = np.asarray(merged[b_key])

        # scaling_value = alpha / rank = 1.0 for both configs (16/16 and 32/32).
        delta = np.matmul(w_a, w_b)
        if w_base.shape != delta.shape:
            raise ValueError(
                f"Shape mismatch for {base_suffix}: base {w_base.shape} vs delta {delta.shape}"
            )
        merged[base_key] = (w_base + delta).astype(w_base.dtype)
        del merged[a_key]
        del merged[b_key]

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src",
        type=pathlib.Path,
        default=pathlib.Path("/root/autodl-tmp/checkpoints/pi05_calvin/29999"),
        help="Source LoRA checkpoint step directory (containing params/ and assets/).",
    )
    parser.add_argument(
        "--dst",
        type=pathlib.Path,
        default=pathlib.Path("/root/autodl-tmp/checkpoints/pi05_calvin"),
        help="Output directory for the merged non-LoRA checkpoint.",
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(pathlib.Path("/root/autodl-tmp/openpi/src")))

    # Set CPU-only to avoid GPU OOM during checkpoint manipulation.
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import orbax.checkpoint as ocp

    from openpi.models.model import restore_params

    src_params_path = args.src / "params"
    if not src_params_path.exists():
        raise FileNotFoundError(f"Source params dir not found: {src_params_path}")

    print(f"[1/4] Loading LoRA checkpoint: {src_params_path}")
    # Use float32 + numpy for precise addition (bfloat16 addition loses precision).
    params = restore_params(src_params_path, restore_type=np.ndarray, dtype=np.float32)
    flat = traverse_util.flatten_dict(params)
    n_lora_keys = sum(1 for k in flat if "lora" in "/".join(str(p) for p in k).lower())
    n_base_keys = len(flat) - n_lora_keys
    print(f"   Loaded {len(flat)} keys ({n_base_keys} base + {n_lora_keys} LoRA)")

    print(f"[2/4] Merging {len(_LORA_GROUPS)} LoRA adapter groups")
    flat_merged = merge_lora_params(flat)
    remaining_lora = sum(1 for k in flat_merged if "lora" in "/".join(str(p) for p in k).lower())
    if remaining_lora:
        raise RuntimeError(f"Expected 0 LoRA keys after merge, found {remaining_lora}")
    print(f"   After merge: {len(flat_merged)} keys (0 LoRA)")

    # Cast back to bfloat16 for consistency with pi05_libero / eval checkpoints.
    flat_merged = {k: v.astype(np.float32) for k, v in flat_merged.items()}
    merged_tree = traverse_util.unflatten_dict(flat_merged)

    print(f"[3/4] Saving merged checkpoint to {args.dst / 'params'}")
    args.dst.mkdir(parents=True, exist_ok=True)
    params_out = args.dst / "params"
    if params_out.exists():
        shutil.rmtree(params_out)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(str(params_out), {"params": merged_tree})
    print(f"   Saved {len(flat_merged)} params")

    # Copy assets (norm_stats).
    print("[4/4] Copying assets")
    src_assets = args.src / "assets"
    dst_assets = args.dst / "assets"
    if src_assets.exists():
        if dst_assets.exists():
            shutil.rmtree(dst_assets)
        shutil.copytree(src_assets, dst_assets)
        print(f"   Copied assets from {src_assets} -> {dst_assets}")
    else:
        print(f"   WARNING: no assets dir at {src_assets}")

    print()
    print("Merge complete.")
    print(f"  Source: {args.src}")
    print(f"  Output: {args.dst}")
    print(f"  Params: {len(flat_merged)} keys (non-LoRA, compatible with restore_params)")


if __name__ == "__main__":
    main()
