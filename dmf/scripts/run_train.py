#!/usr/bin/env python3
"""DMF training entry point — loads base pi0.5 checkpoint, injects DMF head, starts training."""

import argparse
import logging
import os
import sys
from pathlib import Path

# PYTHONPATH is set by train.sh; ensure dmf/src and openpi/src are importable.
_project_root = Path(__file__).resolve().parent.parent
_dmf_src = str(_project_root / "src")
_openpi_src = str(_project_root.parent / "openpi" / "src")
for _p in (_dmf_src, _openpi_src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# JAX must use GPU
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["JAX_COMPILATION_CACHE_MAX_SIZE"] = "134217728"

import jax
jax.config.update("jax_platforms", "cuda")
jax.config.update("jax_compilation_cache_max_size", 128 * 1024 * 1024)

import jax.numpy as jnp
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import yaml

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def _load_and_merge_params(model, params_path: Path):
    """Load base checkpoint params and merge into model, leaving DMF-specific keys at init."""
    from openpi.models.model import restore_params

    logger.info(f"Loading base params from {params_path}...")
    params = restore_params(params_path, dtype=jnp.bfloat16)
    logger.info(f"Loaded {sum(x.size for x in jax.tree.leaves(params)):,} parameters")

    _, state = nnx.split(model)
    pure_state = state.to_pure_dict()

    flat_params = traverse_util.flatten_dict(params)
    flat_state = traverse_util.flatten_dict(pure_state)

    loaded = 0
    skipped_new = 0
    missing = 0

    for key in list(flat_state.keys()):
        str_key = "/".join(str(k) for k in key)
        if key in flat_params:
            flat_state[key] = flat_params.pop(key)
            loaded += 1
        elif "logvar_proj" in str_key:
            # DMF-only parameter — keep random init
            skipped_new += 1
        else:
            # Missing from base checkpoint
            missing += 1
            if missing <= 5:
                logger.warning(f"Parameter not in base checkpoint: {str_key}")

    if missing > 5:
        logger.warning(f"... and {missing - 5} more missing parameters")

    # Report unused checkpoint keys (e.g. from other fine-tuning methods)
    unused = len(flat_params)
    if unused > 0:
        unused_keys = ["/".join(str(k) for k in key) for key in list(flat_params.keys())[:5]]
        logger.info(f"Skipped {unused} unused checkpoint keys (e.g. {unused_keys[0] if unused_keys else 'N/A'})")

    logger.info(
        f"Parameter merge: {loaded} loaded, {skipped_new} DMF-new (kept init), "
        f"{missing} missing, {unused} unused"
    )

    pure_state = traverse_util.unflatten_dict(flat_state)
    state.replace_by_pure_dict(pure_state)
    return nnx.merge(model._graphdef if hasattr(model, '_graphdef') else nnx.split(model)[0], state)


def main():
    parser = argparse.ArgumentParser(description="DMF Training")
    parser.add_argument("config", nargs="?", default="configs/train/dmf_libero_plus.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.info(f"Config: method={config['method']}, steps={config['training_steps']}")

    # ── Create DMF model ──
    from dmf_vla.models.pi05_dmf import Pi05DMF, Pi05DMFConfig

    dmf_config = Pi05DMFConfig(
        pi05=True,
        action_horizon=config["action_horizon"],
        action_dim=config["action_dim"],
        discrete_state_input=config.get("discrete_state_input", False),
        dmf_depth_ratio=config.get("dmf_depth_ratio", 0.67),
        use_logvar=config.get("use_logvar", True),
    )
    model = dmf_config.create(jax.random.key(0))

    # ── Load base checkpoint ──
    ckpt_dir = Path(config["checkpoint"])
    if args.resume:
        # Resume: load from the training checkpoint directly
        logger.info(f"Will resume from {args.resume} (skipping base checkpoint load)")
    else:
        model = _load_and_merge_params(model, ckpt_dir / "params")

    # ── Create trainer ──
    from dmf_vla.training.jax_trainer import DMFTrainer

    wandb_config = config.get("wandb", {})
    trainer = DMFTrainer(
        model=model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=int(config.get("warmup_ratio", 0.016) * config["training_steps"]),
        total_steps=config["training_steps"],
        gradient_clip_norm=config["gradient_clipping"],
        checkpoint_dir=config["checkpoint_dir"],
        log_dir=config.get("log_dir", "logs/dmf"),
        save_every=config["save_every"],
        log_every=config["log_every"],
        wandb_project=wandb_config.get("project", "dmf"),
        wandb_run_name=wandb_config.get("run_name"),
        wandb_config=wandb_config,
        train_config=config,
    )

    # ── Create data loader ──
    from dmf_vla.training.data_loader import create_data_loader, create_fake_data_loader

    dataset_path = config.get("dataset_path", "")
    if Path(dataset_path).exists():
        logger.info(f"Using dataset: {dataset_path}")
        data_loader = create_data_loader(
            dataset_path=dataset_path,
            batch_size=config["batch_size"],
            action_horizon=config["action_horizon"],
            action_dim=config.get("action_dim_raw", config["action_dim"]),
            target_action_dim=config["action_dim"],
        )
    else:
        logger.warning(f"Dataset not found: {dataset_path}, using fake data for smoke test")
        data_loader = create_fake_data_loader(
            batch_size=config["batch_size"],
            action_horizon=config["action_horizon"],
            action_dim=config.get("action_dim", 7),
            num_batches=config["training_steps"],
        )

    # ── Run training ──
    logger.info("=" * 60)
    logger.info("Starting DMF Training")
    logger.info("=" * 60)
    logger.info(f"  DMF depth ratio: {config.get('dmf_depth_ratio', 0.67)}")
    logger.info(f"  Use logvar: {config.get('use_logvar', True)}")
    logger.info(f"  P_mean: {config.get('P_mean', 0.0)}")
    logger.info(f"  P_mean_t: {config.get('P_mean_t', 0.4)}")
    logger.info(f"  P_mean_r: {config.get('P_mean_r', -1.2)}")
    logger.info(f"  P_std: {config.get('P_std', 1.0)}")
    logger.info(f"  EMA decay: {config.get('ema_decay', 0.9999)}")
    logger.info("=" * 60)

    trainer.train(data_loader, resume_from=args.resume)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
