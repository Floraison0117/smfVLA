"""
FreeFlow training entry point.

Called by scripts/train.sh: python -m freeflow.training.run_train
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import wandb
import yaml

from freeflow.config.default_config import FreeFlowConfig
from freeflow.models.pi05_freeflow import Pi05FreeFlow, create_freeflow_model
from freeflow.training.jax_trainer import FreeFlowTrainer
from freeflow.training.data_loader import create_data_loader
from freeflow.training.freeze_utils import get_freeze_patterns, FREEZE_PATTERNS, TRAINABLE_PATTERNS
from freeflow.training.teacher_integration import set_teacher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_base_model(config: FreeFlowConfig):
    """
    Load base π₀.₅ model from checkpoint.

    This will serve as both teacher (frozen) and student initialization.
    """
    from openpi.models.model import restore_params
    from openpi.models import pi0_config

    # Path to base checkpoint
    base_ckpt = Path(config.checkpointing.base_checkpoint)

    logger.info(f"Loading base checkpoint from: {base_ckpt}")

    # Load model parameters using openpi's restore_params
    # This handles multi-device → single-device conversion
    params = restore_params(
        base_ckpt / "params",
        restore_type=jnp.ndarray,
        dtype=jnp.bfloat16,
    )

    # Create FreeFlow model
    # NOTE: Checkpoint analysis shows:
    # - Vision encoder: SigLIP with 1152 hidden dim (standard)
    # - LLM width: 2048 (gemma_2b)
    # - Action expert width: 1024 (gemma_300m)
    # So we use paligemma_variant="gemma_2b" for vision→LLM projection
    # and action_expert_variant="gemma_300m" for action expert
    model_config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        action_dim=config.model.action_dim,
        action_horizon=config.model.action_horizon,
    )
    model = create_freeflow_model(model_config)

    # Load parameters
    graphdef, state = nnx.split(model)
    pure_state = state.to_pure_dict()

    import flax.traverse_util as traverse_util
    flat_params = traverse_util.flatten_dict(params)
    flat_state = traverse_util.flatten_dict(pure_state)

    loaded_count = 0
    for key in flat_state:
        if key in flat_params:
            flat_state[key] = flat_params[key]
            loaded_count += 1
        elif "time_proj" in "/".join(key) or "student_head" in "/".join(key):
            logger.info(f"Keeping initialized: {'/'.join(key)}")
        else:
            logger.warning(f"Not in checkpoint: {'/'.join(key)}")

    logger.info(f"Loaded {loaded_count} parameters from base checkpoint")

    pure_state = traverse_util.unflatten_dict(flat_state)
    state.replace_by_pure_dict(pure_state)
    model = nnx.merge(graphdef, state)

    return model, params


def main():
    parser = argparse.ArgumentParser(description="FreeFlow training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Load config
    config = FreeFlowConfig.from_yaml(args.config)

    logger.info("=" * 60)
    logger.info("FreeFlow Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info(f"Total steps: {config.training.total_steps}")
    logger.info(f"Teacher NFE: {config.model.teacher_nfe}")
    logger.info(f"Lambda correction: {config.model.lambda_correction}")
    logger.info("=" * 60)

    # Initialize WandB
    logger.info("Initializing WandB...")
    wandb.init(
        project=config.logging.project,
        name=f"freeflow-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config.__dict__,
        mode=config.logging.mode,
    )
    logger.info("WandB initialized successfully")

    # Load base model (serves as teacher + student initialization)
    logger.info("Loading base π₀.₅ model...")
    model, base_params = load_base_model(config)
    logger.info("Base model loaded successfully")

    # Create teacher forward function (frozen π₀.₅)
    def teacher_fn(params, obs, z, t, r):
        """Teacher forward function (frozen π₀.₅).

        Args:
            params: Model parameters (unused, passed for JIT compatibility)
            obs: Observation (images, state, prompt)
            z: Noisy action (z_t)
            t: Current timestep
            r: Reference timestep (for flow matching)

        Returns:
            Predicted velocity
        """
        # Use the merged model for teacher inference
        # The model is already loaded with parameters
        return model(obs, z, t, r)

    # Set teacher globally for JIT-compiled integration
    logger.info("Setting teacher function...")
    set_teacher(teacher_fn, base_params)
    logger.info("Teacher function set successfully")

    # Create data loader
    logger.info(f"Loading data from: {config.data.dataset_path}")
    data_loader = create_data_loader(
        dataset_path=config.data.dataset_path,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        action_horizon=config.model.action_horizon,
        action_dim=7,  # LIBERO original dim
        target_action_dim=config.model.action_dim,  # Model dim (32)
    )
    logger.info("Data loader created successfully")

    # Create trainer
    logger.info("Creating trainer...")
    log_dir = Path(config.checkpointing.save_dir).parent / "logs" / "train" / "freeflow"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Log directory: {log_dir}")

    # Create trainer
    # Note: We pass teacher_fn and base_params (frozen π₀.₅)
    # The model itself will be trained as student
    logger.info("Initializing FreeFlowTrainer...")
    trainer = FreeFlowTrainer(
        model=model,
        teacher_fn=teacher_fn,
        teacher_params=base_params,
        learning_rate=config.training.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.total_steps,
        gradient_clip_norm=config.training.gradient_clip,
        checkpoint_dir=config.checkpointing.save_dir,
        log_dir=str(log_dir),
        save_every=config.training.save_every,
        log_every=config.training.log_every,
        eval_every=config.training.eval_every,
        teacher_nfe=config.model.teacher_nfe,
        lambda_correction=config.model.lambda_correction,
        correction_prob=0.5,
        wandb_project=config.logging.project,
        freeze_patterns=config.freeze,
        trainable_patterns=config.trainable,
    )
    logger.info("FreeFlowTrainer initialized successfully")

    # Resume from checkpoint if specified
    start_step = 0
    resume_state = None
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        resume_state = trainer.load_checkpoint(args.resume)
        start_step = resume_state.get("step", 0)

    # Start training
    logger.info("Starting training loop...")
    logger.info(f"Training config: batch_size={config.training.batch_size}, total_steps={config.training.total_steps}")
    trainer.train(data_loader, start_step=start_step, resume_state=resume_state)

    logger.info("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
