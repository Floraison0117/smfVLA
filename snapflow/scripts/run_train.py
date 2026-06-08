#!/usr/bin/env python3
"""SnapFlow training entry script."""
import sys
import os

# Set up paths
project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
openpi_dir = os.path.join(project_root, "third_party", "openpi")
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(openpi_dir, "src"))

import argparse
import logging

# ── JAX 内存限制配置 ─────────────────────────────────────────────────
# 限制 JAX 编译缓存大小，防止 RAM 占用过高
import os
# 限制编译缓存为 128MB（默认可能是无限）
os.environ['JAX_COMPILATION_CACHE_MAX_SIZE'] = '134217728'  # 128MB in bytes

import jax
# 限制编译缓存大小
jax.config.update('jax_compilation_cache_max_size', 128 * 1024 * 1024)

import flax.nnx as nnx
from pathlib import Path

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="configs/train/snapflow_libero.yaml", nargs="?")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (e.g., checkpoints/finetuned/snapflow/step_10000)")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.info(f"Config: {config['method']}, steps={config['training_steps']}")
    logger.info(f"Alpha: {config.get('alpha', 0.5)}, Lambda: {config.get('lambda_consistency', 0.1)}")

    # Load model
    from snapflow.models.pi05_snapflow import Pi05SnapFlow, Pi05SnapFlowConfig

    snapflow_config = Pi05SnapFlowConfig(
        pi05=True,
        action_horizon=config["action_horizon"],
        action_dim=config["action_dim"],
        discrete_state_input=False,
        alpha=config.get("alpha", 0.5),
        lambda_consistency=config.get("lambda_consistency", 0.1),
        prediction_clamp_min=config.get("prediction_clamp_min", -20),
        prediction_clamp_max=config.get("prediction_clamp_max", 20),
    )

    # Load parameters from base checkpoint
    from openpi.models.model import restore_params
    import jax.numpy as jnp

    ckpt_dir = Path(config["checkpoint"])
    logger.info(f"Loading params from {ckpt_dir / 'params'}...")
    params = restore_params(ckpt_dir / "params", dtype=jnp.bfloat16)
    logger.info(f"Loaded {sum(x.size for x in jax.tree.leaves(params)):,} parameters")

    # Create model
    model = snapflow_config.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    pure_state = state.to_pure_dict()

    import flax.traverse_util as traverse_util

    flat_params = traverse_util.flatten_dict(params)
    flat_state = traverse_util.flatten_dict(pure_state)

    loaded_count = 0
    for key in flat_state:
        str_key = "/".join(key)
        if key in flat_params:
            flat_state[key] = flat_params[key]
            loaded_count += 1
        elif "time_proj" in str_key or "target_time_mlp" in str_key:
            logger.info(f"Skipping new parameter: {str_key}")
        else:
            logger.warning(f"Parameter not found: {str_key}")

    logger.info(f"Loaded {loaded_count} parameters")

    pure_state = traverse_util.unflatten_dict(flat_state)
    state.replace_by_pure_dict(pure_state)
    model = nnx.merge(graphdef, state)
    logger.info("Model loaded successfully")

    # Verify target_time_mlp exists and outputs zeros
    logger.info("Checking target_time_mlp initialization...")
    import jax.numpy as jnp
    test_input = jnp.ones((1,))
    test_output = model.target_time_mlp(test_input)
    logger.info(f"target_time_mlp output shape: {test_output.shape}")
    logger.info(f"target_time_mlp output sum: {test_output.sum():.6f} (should be ~0)")

    # Create trainer
    from snapflow.training.jax_trainer import SnapFlowTrainer

    trainer = SnapFlowTrainer(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=int(config["warmup_ratio"] * config["training_steps"]),
        total_steps=config["training_steps"],
        gradient_clip_norm=config["gradient_clipping"],
        checkpoint_dir=config["checkpoint_dir"],
        log_dir=config["log_dir"],
        save_every=config["save_every"],
        log_every=config["log_every"],
        wandb_project=config.get("wandb", {}).get("project", "snapflow"),
        wandb_run_name=config.get("wandb", {}).get("run_name"),
        wandb_config=config.get("wandb", {}),
        train_config=config,
        alpha=config.get("alpha", 0.5),
        lambda_consistency=config.get("lambda_consistency", 0.1),
        prediction_clamp_min=config.get("prediction_clamp_min", -20),
        prediction_clamp_max=config.get("prediction_clamp_max", 20),
    )

    # Create data loader
    from snapflow.training.data_loader import create_data_loader, create_fake_data_loader

    dataset_path = config.get("dataset_path", "")

    if Path(dataset_path).exists():
        logger.info(f"Using dataset: {dataset_path}")
        logger.info(f"Using single-process data loading (optimized)")
        data_loader = create_data_loader(
            dataset_path=dataset_path,
            batch_size=config["batch_size"],
            action_horizon=config["action_horizon"],
            target_action_dim=config["action_dim"],
        )
    else:
        logger.warning(f"Dataset not found: {dataset_path}, using fake data")
        data_loader = create_fake_data_loader(
            batch_size=config["batch_size"],
            action_horizon=config["action_horizon"],
            action_dim=config["action_dim"],
            num_batches=config["training_steps"],
        )

    # Start training
    logger.info("=" * 60)
    logger.info("Starting SnapFlow training")
    logger.info("=" * 60)
    rng = jax.random.key(42)
    model = trainer.train(data_loader, rng, resume_from=args.resume)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
