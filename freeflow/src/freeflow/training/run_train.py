"""
FreeFlow training entry point.

Training setup:
- Teacher: Frozen π₀.₅ model (NFE=10)
- Student: FreeFlow model (NFE=1) initialized from π₀.₅
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import wandb

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


def load_pi05_teacher(config: FreeFlowConfig):
    """
    Load true π₀.₅ teacher model from checkpoint.

    Returns a Pi0 model (not FreeFlow) for teacher inference.
    """
    from openpi.models.model import restore_params
    from openpi.models import pi0_config, pi0

    base_ckpt = Path(config.checkpointing.base_checkpoint)
    logger.info(f"Loading π₀.₅ teacher from: {base_ckpt}")

    # Load parameters
    params = restore_params(
        base_ckpt / "params",
        restore_type=jnp.ndarray,
        dtype=jnp.bfloat16,
    )

    # Create π₀.₅ model (Pi0, not FreeFlow)
    model_config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        action_dim=config.model.action_dim,
        action_horizon=config.model.action_horizon,
    )

    # Create Pi0 model
    teacher_model = pi0.Pi0(model_config, nnx.Rngs(0))

    # Load parameters
    graphdef, state = nnx.split(teacher_model)
    pure_state = state.to_pure_dict()

    import flax.traverse_util as traverse_util
    flat_params = traverse_util.flatten_dict(params)
    flat_state = traverse_util.flatten_dict(pure_state)

    loaded_count = 0
    for key in flat_state:
        if key in flat_params:
            flat_state[key] = flat_params[key]
            loaded_count += 1
        else:
            logger.warning(f"Teacher param not in checkpoint: {'/'.join(str(k) for k in key)}")

    logger.info(f"Teacher loaded: {loaded_count} parameters")

    pure_state = traverse_util.unflatten_dict(flat_state)
    state.replace_by_pure_dict(pure_state)
    teacher_model = nnx.merge(graphdef, state)

    return teacher_model, params


def load_freeflow_student(config: FreeFlowConfig, teacher_params):
    """
    Load FreeFlow student model initialized from π₀.₅ checkpoint.

    Args:
        config: FreeFlow config
        teacher_params: π₀.₅ checkpoint parameters (for initialization)

    Returns:
        FreeFlow student model
    """
    from openpi.models import pi0_config

    logger.info("Creating FreeFlow student model...")

    model_config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        action_dim=config.model.action_dim,
        action_horizon=config.model.action_horizon,
    )

    student_model = create_freeflow_model(model_config)

    # Initialize from teacher parameters
    graphdef, state = nnx.split(student_model)
    pure_state = state.to_pure_dict()

    import flax.traverse_util as traverse_util
    flat_teacher_params = traverse_util.flatten_dict(teacher_params)
    flat_state = traverse_util.flatten_dict(pure_state)

    loaded_count = 0
    for key in flat_state:
        if key in flat_teacher_params:
            flat_state[key] = flat_teacher_params[key]
            loaded_count += 1
        # FreeFlow-specific params (time_mlp_in/out are shared with Pi0, so they should load)
        # If there are truly new params, they stay randomly initialized

    logger.info(f"Student initialized: {loaded_count} parameters from teacher")

    pure_state = traverse_util.unflatten_dict(flat_state)
    state.replace_by_pure_dict(pure_state)
    student_model = nnx.merge(graphdef, state)

    return student_model


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

    # Step 1: Load π₀.₅ teacher (frozen, for Euler integration)
    logger.info("=" * 60)
    logger.info("Step 1: Loading π₀.₅ teacher model...")
    teacher_model, teacher_params = load_pi05_teacher(config)
    logger.info("π₀.₅ teacher loaded successfully")

    # Step 2: Create FreeFlow student (trainable)
    logger.info("Step 2: Creating FreeFlow student model...")
    student_model = load_freeflow_student(config, teacher_params)
    logger.info("FreeFlow student created successfully")

    # Step 3: Set teacher for Euler integration
    logger.info("Step 3: Setting teacher for Euler integration...")
    set_teacher(teacher_model)
    logger.info("Teacher set successfully")

    # Step 4: Create data loader
    logger.info("Step 4: Creating data loader...")
    data_loader = create_data_loader(
        dataset_path=config.data.dataset_path,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        action_horizon=config.model.action_horizon,
        action_dim=7,  # LIBERO original dim
        target_action_dim=config.model.action_dim,  # Model dim (32)
    )
    logger.info("Data loader created successfully")

    # Step 5: Create trainer
    logger.info("Step 5: Creating trainer...")
    log_dir = Path(config.checkpointing.save_dir).parent / "logs" / "train" / "freeflow"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Note: trainer doesn't need teacher_fn anymore
    # The teacher is set globally and used by teacher_euler_integration
    trainer = FreeFlowTrainer(
        model=student_model,
        teacher_fn=None,  # Not used, teacher is global
        teacher_params=None,  # Not used
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
    logger.info("Trainer created successfully")

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
