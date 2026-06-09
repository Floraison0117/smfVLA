#!/usr/bin/env python3
"""
Diagnostic script to verify checkpoint compatibility with FreeFlow.

Checks:
1. Checkpoint structure and parameter shapes
2. Model variant (gemma_300m vs gemma_2b)
3. Vision encoder dimensions
4. Action expert dimensions
5. Time embedding dimensions
6. Parameter counts
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_checkpoint_params(ckpt_path: Path) -> dict:
    """Load checkpoint parameters using openpi's restore_params."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "openpi" / "src"))
    from openpi.models.model import restore_params

    params_path = ckpt_path / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Params directory not found: {params_path}")

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    params = restore_params(params_path, restore_type=jnp.ndarray, dtype=jnp.bfloat16)
    return params


def analyze_param_shapes(params: dict) -> dict:
    """Analyze parameter shapes to infer model variants."""
    analysis = {
        "vision_width": None,
        "llm_width": None,
        "action_expert_width": None,
        "time_mlp_dim": None,
        "num_vision_layers": 0,
        "num_llm_layers": 0,
        "total_params": 0,
    }

    def flatten_dict(d, parent_key="", sep="/"):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_params = flatten_dict(params)

    for key, val in flat_params.items():
        if hasattr(val, "shape"):
            shape = val.shape
            analysis["total_params"] += val.size

            # Detect vision encoder (SigLIP)
            if "PaliGemma/img" in key and "kernel" in key:
                if len(shape) >= 2:
                    analysis["vision_width"] = max(analysis["vision_width"] or 0, shape[0])

            # Detect LLM embedding
            if "llm/embedder" in key and "embedding" in key:
                analysis["llm_width"] = shape[-1]

            # Detect transformer layers
            if "llm/layers/" in key:
                layer_num = key.split("layers/")[1].split("/")[0] if "layers/" in key else "0"
                try:
                    layer_num = int(layer_num)
                    analysis["num_llm_layers"] = max(analysis["num_llm_layers"], layer_num + 1)
                except ValueError:
                    pass

            # Detect time MLP
            if "time_mlp" in key and "kernel" in key:
                if len(shape) >= 2:
                    analysis["time_mlp_dim"] = max(analysis["time_mlp_dim"] or 0, shape[0])

            # Detect action expert
            if "action_out_proj" in key and "kernel" in key:
                analysis["action_expert_width"] = shape[0] if len(shape) > 0 else None

    return analysis


def infer_variants(analysis: dict) -> dict:
    """Infer model variants from dimensions."""
    variants = {
        "vision_variant": "unknown",
        "llm_variant": "unknown",
        "action_expert_variant": "unknown",
    }

    # gemma_2b has width 2048, gemma_300m has width 1024
    if analysis["vision_width"]:
        if analysis["vision_width"] == 2048:
            variants["vision_variant"] = "gemma_2b"
        elif analysis["vision_width"] == 1024:
            variants["vision_variant"] = "gemma_300m"
        else:
            variants["vision_variant"] = f"custom_{analysis['vision_width']}"

    if analysis["llm_width"]:
        if analysis["llm_width"] == 2048:
            variants["llm_variant"] = "gemma_2b"
        elif analysis["llm_width"] == 1024:
            variants["llm_variant"] = "gemma_300m"
        else:
            variants["llm_variant"] = f"custom_{analysis['llm_width']}"

    if analysis["action_expert_width"]:
        if analysis["action_expert_width"] == 2048:
            variants["action_expert_variant"] = "gemma_2b"
        elif analysis["action_expert_width"] == 1024:
            variants["action_expert_variant"] = "gemma_300m"
        else:
            variants["action_expert_variant"] = f"custom_{analysis['action_expert_width']}"

    return variants


def test_config_compatibility(ckpt_analysis: dict, config_path: Path) -> dict:
    """Test if FreeFlow config is compatible with checkpoint."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    compatibility = {
        "compatible": True,
        "issues": [],
        "warnings": [],
    }

    # Check dataset path
    dataset_path = config.get("data", {}).get("dataset_path", "")
    if "libero-plus-training" not in dataset_path and "libero-plus" not in dataset_path:
        if "libero" in dataset_path and "libero-plus" not in dataset_path:
            compatibility["issues"].append(
                f"Dataset path '{dataset_path}' uses standard LIBERO, not LIBERO-Plus. "
                f"Expected libero-plus-training for robustness training."
            )

    # Check norm_stats path
    norm_stats_path = config.get("data", {}).get("norm_stats", "")
    if "libero-plus-training" not in norm_stats_path:
        if "libero" in norm_stats_path and "libero-plus" not in norm_stats_path:
            compatibility["warnings"].append(
                f"Norm stats '{norm_stats_path}' from LIBERO, not LIBERO-Plus"
            )

    # Check checkpoint path
    base_ckpt = config.get("checkpointing", {}).get("base_checkpoint", "")
    if not Path(base_ckpt).exists():
        compatibility["issues"].append(f"Base checkpoint not found: {base_ckpt}")

    # Check action dim
    action_dim = config.get("model", {}).get("action_dim", 32)
    if action_dim != 32:
        compatibility["warnings"].append(f"Action dim {action_dim} != 32 (expected)")

    # Check batch size
    batch_size = config.get("training", {}).get("batch_size", 32)
    if batch_size > 16:
        compatibility["warnings"].append(
            f"Batch size {batch_size} may be too large. Consider 16 for stability."
        )

    compatibility["compatible"] = len(compatibility["issues"]) == 0
    return compatibility


def print_diagnostic_report(ckpt_analysis: dict, variants: dict, compatibility: dict):
    """Print comprehensive diagnostic report."""
    logger.info("=" * 70)
    logger.info("CHECKPOINT DIAGNOSTIC REPORT")
    logger.info("=" * 70)

    logger.info("\n📊 Model Dimensions:")
    logger.info(f"  Vision Encoder Width:     {ckpt_analysis['vision_width'] or 'Unknown'}")
    logger.info(f"  LLM Width:                {ckpt_analysis['llm_width'] or 'Unknown'}")
    logger.info(f"  Action Expert Width:      {ckpt_analysis['action_expert_width'] or 'Unknown'}")
    logger.info(f"  Time MLP Dim:             {ckpt_analysis['time_mlp_dim'] or 'Unknown'}")
    logger.info(f"  Total Parameters:          {ckpt_analysis['total_params']:,}")

    logger.info("\n🔍 Inferred Variants:")
    logger.info(f"  Vision Variant:           {variants['vision_variant']}")
    logger.info(f"  LLM Variant:              {variants['llm_variant']}")
    logger.info(f"  Action Expert Variant:    {variants['action_expert_variant']}")

    logger.info("\n✅ Config Compatibility:")
    if compatibility["compatible"]:
        logger.info("  Status: COMPATIBLE")
    else:
        logger.info("  Status: NOT COMPATIBLE - Issues found")

    if compatibility["issues"]:
        logger.info("\n❌ Issues:")
        for issue in compatibility["issues"]:
            logger.info(f"  - {issue}")

    if compatibility["warnings"]:
        logger.info("\n⚠️  Warnings:")
        for warning in compatibility["warnings"]:
            logger.info(f"  - {warning}")

    # Variant mismatch detection
    if variants["vision_variant"] != variants["llm_variant"]:
        logger.info("\n❌ Variant Mismatch:")
        logger.info(f"  Vision: {variants['vision_variant']}, LLM: {variants['llm_variant']}")
        logger.info("  This will cause initialization errors!")

    logger.info("\n" + "=" * 70)


def recommend_fixes(ckpt_analysis: dict, variants: dict, compatibility: dict):
    """Print recommended fixes based on diagnostic results."""
    logger.info("\n🔧 Recommended Fixes:")

    fixes = []

    # Fix dataset path
    if any("libero-plus-training" not in i for i in compatibility["issues"] if "Dataset" in i):
        fixes.append("1. Update dataset path to use libero-plus-training")

    # Fix norm stats
    if any("norm_stats" in w for w in compatibility["warnings"]):
        fixes.append("2. Update norm_stats path to libero-plus-training/norm_stats.json")

    # Fix vision encoder width
    if variants["vision_variant"] != variants["llm_variant"]:
        fixes.append("3. Fix vision encoder width mismatch (see config fix below)")

    for fix in fixes:
        logger.info(f"  {fix}")

    # Print recommended Pi0Config
    logger.info("\n📝 Recommended Pi0Config:")
    logger.info(f"```python")
    logger.info(f"model_config = Pi0Config(")
    logger.info(f"    paligemma_variant=\"{variants['llm_variant']}\",")
    logger.info(f"    action_expert_variant=\"{variants['action_expert_variant']}\",")
    logger.info(f"    pi05=True,")
    logger.info(f"    action_dim=32,")
    logger.info(f"    action_horizon=1,")
    logger.info(f")")
    logger.info(f"```")

    if variants["vision_variant"] == "gemma_300m":
        logger.info("\n⚠️  Note: Checkpoint uses gemma_300m for vision encoder.")
        logger.info("    Pi0Config doesn't support vision_variant parameter.")
        logger.info("    You may need to create the model differently or patch the config.")


def main():
    parser = argparse.ArgumentParser(description="Diagnose FreeFlow checkpoint compatibility")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--config", type=str, help="Path to FreeFlow config YAML")
    args = parser.parse_args()

    # Default paths
    project_root = Path(__file__).parent.parent
    ckpt_path = Path(args.checkpoint) if args.checkpoint else project_root / "checkpoints" / "base" / "pi05_libero"
    config_path = Path(args.config) if args.config else project_root / "configs" / "train" / "freeflow_base_libero.yaml"

    # Load and analyze checkpoint
    params = load_checkpoint_params(ckpt_path)
    analysis = analyze_param_shapes(params)
    variants = infer_variants(analysis)

    # Test config compatibility
    compatibility = test_config_compatibility(analysis, config_path)

    # Print report
    print_diagnostic_report(analysis, variants, compatibility)
    recommend_fixes(analysis, variants, compatibility)


if __name__ == "__main__":
    main()
