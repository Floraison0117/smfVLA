#!/usr/bin/env python3
"""Pi-Flow training entry point.

Loads pi0.5 teacher (frozen) and student (trainable, with GMM heads).
Velocity imitation distillation: student GMFlow policy learns from teacher.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_piflow_src = str(_project_root / "src")
_openpi_src = str(_project_root.parent / "openpi" / "src")
for _p in (_piflow_src, _openpi_src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["JAX_COMPILATION_CACHE_MAX_SIZE"] = "134217728"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import jax

jax.config.update("jax_platforms", "cuda")
jax.config.update("jax_compilation_cache_max_size", 128 * 1024 * 1024)

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax.numpy as jnp
import yaml

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def _load_checkpoint_params(params_path: Path):
    """Load params from orbax checkpoint, returning a pure dict."""
    from openpi.models.model import restore_params

    logger.info(f"Loading params from {params_path}...")
    params = restore_params(params_path, dtype=jnp.bfloat16)
    n = sum(x.size for x in jax.tree.leaves(params))
    logger.info(f"Loaded {n:,} parameters")
    return params


def _merge_params_into_model(model, params, skip_patterns=()):
    """Merge checkpoint params into model state, skipping specified patterns."""
    _, state = nnx.split(model)
    pure_state = state.to_pure_dict()

    flat_params = traverse_util.flatten_dict(params)
    flat_state = traverse_util.flatten_dict(pure_state)

    loaded = 0
    skipped = 0
    missing = 0

    for key in list(flat_state.keys()):
        str_key = "/".join(str(k) for k in key)
        if key in flat_params:
            flat_state[key] = flat_params.pop(key)
            loaded += 1
        elif any(p in str_key for p in skip_patterns):
            skipped += 1
        else:
            missing += 1
            if missing <= 5:
                logger.warning(f"Missing from checkpoint: {str_key}")

    if missing > 5:
        logger.warning(f"... and {missing - 5} more missing params")

    unused = len(flat_params)
    if unused > 0:
        unused_keys = ["/".join(str(k) for k in key) for key in list(flat_params.keys())[:5]]
        logger.info(
            f"Skipped {unused} unused checkpoint keys (e.g. {unused_keys[0] if unused_keys else 'N/A'})"
        )

    logger.info(
        f"Merge: {loaded} loaded, {skipped} new (kept init), {missing} missing, {unused} unused"
    )

    pure_state = traverse_util.unflatten_dict(flat_state)
    state.replace_by_pure_dict(pure_state)
    graphdef, _ = nnx.split(model)
    return nnx.merge(graphdef, state)


def main():
    parser = argparse.ArgumentParser(description="Pi-Flow Training")
    parser.add_argument("config", nargs="?", default="configs/train/piflow_libero_plus.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.info(f"Config: method={config['method']}, steps={config['training_steps']}")

    ckpt_dir = Path(config["checkpoint"])

    # ── Create Teacher (frozen Pi0) ──
    from openpi.models import pi0_config

    logger.info("Creating teacher (Pi0 from pi05_libero)...")
    teacher_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=config["action_horizon"],
        action_dim=config["action_dim"],
        discrete_state_input=config.get("discrete_state_input", False),
    )
    teacher_model = teacher_config.create(jax.random.key(1))
    teacher_params = _load_checkpoint_params(ckpt_dir / "params")
    teacher_model = _merge_params_into_model(teacher_model, teacher_params)
    logger.info("Teacher ready (frozen)")

    # ── Create Student (Pi05PiFlow with GMM heads) ──
    from piflow_vla.models.pi05_piflow import Pi05PiFlow, Pi05PiFlowConfig

    logger.info("Creating student (Pi05PiFlow)...")
    student_config = Pi05PiFlowConfig(
        pi05=True,
        action_horizon=config["action_horizon"],
        action_dim=config["action_dim"],
        discrete_state_input=config.get("discrete_state_input", False),
        num_components=config.get("num_components", 8),
        inner_substeps=config.get("inner_substeps", 8),
    )
    student_model = student_config.create(jax.random.key(0))

    if args.resume:
        logger.info(f"Resuming from {args.resume} (skipping base checkpoint load)")
    else:
        # Load base pi05_libero params, skip GMM head keys
        base_params = _load_checkpoint_params(ckpt_dir / "params")
        student_model = _merge_params_into_model(
            student_model,
            base_params,
            skip_patterns=("gmm_mean_proj", "gmm_logstd_proj", "gmm_logweight_proj"),
        )

    # ── Create Trainer ──
    from piflow_vla.training.jax_trainer import PiFlowTrainer

    wandb_config = config.get("wandb", {})
    trainer = PiFlowTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_steps=config.get(
            "warmup_steps", config.get("warmup_ratio", 0.03) * config["training_steps"]
        ),
        total_steps=config["training_steps"],
        gradient_clip_norm=config["gradient_clip_norm"],
        checkpoint_dir=config["checkpoint_dir"],
        log_dir=config.get("log_dir", "logs/piflow"),
        save_every=config["save_every"],
        log_every=config["log_every"],
        inner_substeps=config.get("inner_substeps", 8),
        teacher_query_points=config.get("teacher_query_points", 4),
        nfe=config.get("nfe", 1),
        ema_decay=config.get("ema_decay", 0.9999),
        wandb_project=wandb_config.get("project", "piflow"),
        wandb_run_name=wandb_config.get("run_name"),
        wandb_config=wandb_config,
        train_config=config,
    )

    # ── Create Data Loader ──
    from piflow_vla.training.data_loader import create_data_loader, create_fake_data_loader

    dataset_path = config.get("dataset_path", "")
    if Path(dataset_path).exists():
        logger.info(f"Using dataset: {dataset_path}")
        # Resolve the base checkpoint's norm_stats.json so the frozen pi0.5
        # teacher receives state in the quantile-normalized [-1, 1] space it
        # was trained on (the dataset's own norm_stats may be partial). See
        # docs/training-debug.md §9.
        norm_stats_path = None
        ckpt_assets = list(Path(config["checkpoint"]).rglob("norm_stats.json"))
        if ckpt_assets:
            norm_stats_path = str(ckpt_assets[0])
            logger.info(f"Using checkpoint norm_stats for state normalization: {norm_stats_path}")
        else:
            logger.warning(
                "No norm_stats.json found under checkpoint; "
                "state will NOT be quantile-normalized"
            )
        data_loader = create_data_loader(
            dataset_path=dataset_path,
            batch_size=config["batch_size"],
            action_horizon=config["action_horizon"],
            action_dim=config.get("action_dim_raw", config["action_dim"]),
            target_action_dim=config["action_dim"],
            norm_stats_path=norm_stats_path,
        )
    else:
        logger.warning(f"Dataset not found: {dataset_path}, using fake data for smoke test")
        data_loader = create_fake_data_loader(
            batch_size=config["batch_size"],
            action_horizon=config["action_horizon"],
            action_dim=config["action_dim"],
            num_batches=config["training_steps"],
        )

    # ── Run Training ──
    logger.info("=" * 60)
    logger.info("Starting Pi-Flow Training")
    logger.info("=" * 60)
    logger.info(f"  GMM components: {config.get('num_components', 8)}")
    logger.info(f"  NFE: {config.get('nfe', 1)}")
    logger.info(f"  Inner substeps (total): {config.get('inner_substeps', 8)}")
    logger.info(f"  Teacher query points/seg: {config.get('teacher_query_points', 4)}")
    logger.info(f"  EMA decay: {config.get('ema_decay', 0.9999)}")
    logger.info("=" * 60)

    trainer.train(data_loader, resume_from=args.resume)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
