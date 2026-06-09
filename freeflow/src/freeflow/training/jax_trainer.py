"""
JAX training loop for FreeFlow.

Implements data-free distillation training with frozen teacher and trainable student.
"""

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import yaml
import wandb

from freeflow.training import freeze_utils

logger = logging.getLogger(__name__)


class FreeFlowTrainer:
    """
    FreeFlow JAX trainer for 1-NFE distillation.

    Features:
    - Frozen π₀.₅ teacher (NFE=10)
    - Trainable student model (NFE=1)
    - Data-free distillation loss
    - Selective parameter updates
    - Checkpoint save/load
    - WandB logging
    """

    def __init__(
        self,
        model: nnx.Module,
        teacher_fn: Any,
        teacher_params: Any,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 30000,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints/finetuned/freeflow",
        log_dir: str = "logs/train/freeflow",
        save_every: int = 5000,
        log_every: int = 100,
        eval_every: int = 1000,
        teacher_nfe: int = 10,
        lambda_correction: float = 0.1,
        correction_prob: float = 0.5,
        wandb_project: str = "freeflow",
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[dict[str, Any]] = None,
        freeze_patterns: Optional[list[str]] = None,
        trainable_patterns: Optional[list[str]] = None,
    ):
        """
        Initialize FreeFlow trainer.

        Args:
            model: Student model (trainable)
            teacher_fn: Teacher model function (frozen)
            teacher_params: Teacher parameters (frozen)
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            warmup_steps: Linear warmup steps
            total_steps: Total training steps
            gradient_clip_norm: Gradient clipping norm
            checkpoint_dir: Checkpoint save directory
            log_dir: Log directory
            save_every: Save checkpoint every N steps
            log_every: Log metrics every N steps
            eval_every: Run evaluation every N steps
            teacher_nfe: Teacher's NFE (default 10)
            lambda_correction: Error correction weight
            correction_prob: Probability of applying correction
            wandb_project: WandB project name
            wandb_run_name: WandB run name
            wandb_config: WandB config dict
            freeze_patterns: Parameter freeze patterns
            trainable_patterns: Parameter trainable patterns
        """
        self.model = model
        self.teacher_fn = teacher_fn
        self.teacher_params = teacher_params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.save_every = save_every
        self.log_every = log_every
        self.eval_every = eval_every
        self.teacher_nfe = teacher_nfe
        self.lambda_correction = lambda_correction
        self.correction_prob = correction_prob
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}

        # Freeze/trainable patterns
        self.freeze_patterns = freeze_patterns or list(freeze_utils.get_default_freeze_patterns())
        self.trainable_patterns = trainable_patterns or list(freeze_utils.get_default_trainable_patterns())

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build trainable mask
        graphdef, state = nnx.split(model)
        self.graphdef = graphdef  # Store for reuse in train_step
        self.trainable_mask = freeze_utils.build_trainable_mask(
            state,
            freeze_patterns=set(self.freeze_patterns),
            trainable_patterns=set(self.trainable_patterns),
        )
        self.param_stats = freeze_utils.print_param_summary(state, self.trainable_mask)
        logger.info(f"Parameter stats:\n{self.param_stats}")

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Training state
        self.step = 0
        self.train_log = []

        # Setup WandB
        self._setup_wandb()

        # JIT compile training step
        logger.info("Starting JIT compilation of training step...")
        logger.info("This may take several minutes for the first compilation...")
        import time
        jit_start = time.time()
        self._jit_train_step = self._setup_jit_train_step()
        jit_time = time.time() - jit_start
        logger.info(f"JIT compilation completed in {jit_time:.2f} seconds")

        logger.info("FreeFlow trainer initialized")
        logger.info(f"  Teacher NFE: {self.teacher_nfe}")
        logger.info(f"  Lambda correction: {self.lambda_correction}")
        logger.info(f"  Correction prob: {self.correction_prob}")

    def _build_optimizer(self) -> optax.GradientTransformation:
        """Build optax optimizer: AdamW + warmup + cosine decay + gradient clipping."""
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=self.learning_rate,
            transition_steps=self.warmup_steps,
        )
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=self.learning_rate,
            decay_steps=self.total_steps - self.warmup_steps,
            alpha=0.0,
        )
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[self.warmup_steps],
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(self.gradient_clip_norm),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.weight_decay,
            ),
        )

        return optimizer

    def _setup_wandb(self):
        """Initialize WandB logging."""
        if wandb.run is None:
            run_name = self.wandb_run_name or f"freeflow-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=self.wandb_project,
                name=run_name,
                config=self.wandb_config,
                mode="online",
            )
            logger.info(f"WandB initialized: {run_name}")

    def _setup_jit_train_step(self):
        """
        Compile training step with JAX JIT.

        Only trainable parameters receive gradients.
        """
        from freeflow.training.freeflow_loss import compute_freeflow_loss_with_data

        def train_step(
            student_state: nnx.State,
            opt_state: optax.OptState,
            rng: jax.Array,
            batch: dict[str, Any],
        ) -> tuple[nnx.State, optax.OptState, dict[str, Any]]:
            """
            Single training step.

            Args:
                student_state: Student model state
                opt_state: Optimizer state
                rng: Random key
                batch: Training batch (filtered to only JAX arrays)

            Returns:
                student_state: Updated student state
                opt_state: Updated optimizer state
                metrics: Training metrics
            """
            # Split random keys
            rng_loss, rng_new = jax.random.split(rng)

            # Get observation from batch
            observation = {
                "image": batch["observation"]["image"],
                "image_mask": batch["observation"]["image_mask"],
                "state": batch["observation"]["state"],
            }

            actions = batch["actions"]
            action_mean = batch["action_mean"]
            action_std = batch["action_std"]

            # Define loss function
            # Use self.graphdef (created in __init__)
            graphdef = self.graphdef

            def loss_fn(params):
                # Create a temporary state from params
                # params is a flat dict of trainable parameters
                # We need to merge it back into the full state structure
                temp_state = freeze_utils.apply_trainable_params_to_state(
                    student_state, params, self.trainable_mask
                )

                # Create a merged model with the temp state
                temp_model = nnx.merge(graphdef, temp_state)

                loss, metrics = compute_freeflow_loss_with_data(
                    student_fn=lambda p, *args: temp_model(*args),
                    student_params=params,
                    observation=observation,
                    actions=actions,
                    action_mean=action_mean,
                    action_std=action_std,
                    rng=rng_loss,
                    teacher_nfe=self.teacher_nfe,
                    lambda_correction=self.lambda_correction,
                    correction_prob=self.correction_prob,
                )
                return loss, metrics

            # Compute gradients only for trainable parameters
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            # Get trainable parameters from state
            trainable_params = freeze_utils.get_trainable_params_from_state(
                student_state, self.trainable_mask
            )

            (loss, metrics), grads = grad_fn(trainable_params)

            # Apply gradients only to trainable parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, trainable_params)
            trainable_params = optax.apply_updates(trainable_params, updates)

            # Update state with trainable parameters
            student_state = freeze_utils.apply_trainable_params_to_state(
                student_state, trainable_params, self.trainable_mask
            )

            return student_state, opt_state, metrics

        return jax.jit(train_step)

    def train(
        self,
        data_loader,
        start_step: int = 0,
        resume_state: Optional[dict[str, Any]] = None,
    ):
        """
        Main training loop.

        Args:
            data_loader: Training data iterator
            start_step: Starting step (for resume)
            resume_state: Optional resume state (model, optimizer, step)
        """
        # Handle resume
        if resume_state is not None:
            logger.info(f"Resuming from step {start_step}")
            self.step = start_step
            if "opt_state" in resume_state:
                self.opt_state = resume_state["opt_state"]
            else:
                # Initialize optimizer state from current model parameters
                _, state = nnx.split(self.model)
                trainable_params = freeze_utils.get_trainable_params_from_state(
                    state, self.trainable_mask
                )
                self.opt_state = self.optimizer.init(trainable_params)
        else:
            self.step = 0
            graphdef, state = nnx.split(self.model)
            trainable_params = freeze_utils.get_trainable_params_from_state(
                state, self.trainable_mask
            )
            self.opt_state = self.optimizer.init(trainable_params)

        logger.info(f"Starting training from step {self.step} to {self.total_steps}")
        logger.info("=" * 60)

        start_time = time.time()
        last_log_time = start_time

        for epoch in range(100):  # Maximum epochs
            for batch_idx, batch in enumerate(data_loader):
                if self.step >= self.total_steps:
                    logger.info(f"Training complete at step {self.step}")
                    break

                # Filter batch to only JAX-compatible arrays
                # Remove string fields like 'prompt' that can't be passed to JIT
                filtered_batch = {
                    "observation": {
                        "image": batch["observation"]["image"],
                        "image_mask": batch["observation"]["image_mask"],
                        "state": batch["observation"]["state"],
                    },
                    "actions": batch["actions"],
                    "action_mean": batch["action_mean"],
                    "action_std": batch["action_std"],
                }

                # Training step
                rng = jax.random.PRNGKey(self.step)
                graphdef, state = nnx.split(self.model)
                state, self.opt_state, metrics = self._jit_train_step(
                    state, self.opt_state, rng, filtered_batch
                )
                self.model = nnx.merge(graphdef, state)

                # Convert metrics to float
                metrics_float = jax.device_get(metrics)
                for k, v in metrics_float.items():
                    if hasattr(v, "item"):
                        metrics_float[k] = v.item()

                # Update step
                self.step += 1

                # Logging
                current_time = time.time()
                should_log = (self.step % self.log_every == 0)
                should_save = (self.step % self.save_every == 0) or (self.step == self.total_steps)

                if should_log:
                    elapsed = current_time - last_log_time
                    steps_per_sec = self.log_every / elapsed
                    last_log_time = current_time

                    # Log to console
                    logger.info(
                        f"Step {self.step}/{self.total_steps} | "
                        f"loss: {metrics_float['loss_total']:.4f} | "
                        f"path: {metrics_float.get('loss_path', 0):.4f} | "
                        f"corr: {metrics_float.get('loss_correction', 0):.4f} | "
                        f"{steps_per_sec:.2f} steps/s"
                    )

                    # Log to WandB
                    wandb.log({
                        "step": self.step,
                        "train/loss_total": metrics_float["loss_total"],
                        "train/loss_path": metrics_float.get("loss_path", 0),
                        "train/loss_correction": metrics_float.get("loss_correction", 0),
                        "train/steps_per_sec": steps_per_sec,
                        "train/lr": self.learning_rate,  # Use the stored learning rate
                    })

                # Save checkpoint
                if should_save:
                    self.save_checkpoint(self.step)

        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time / 3600:.2f} hours")

    def save_checkpoint(self, step: int):
        """Save training checkpoint."""
        path = self.checkpoint_dir / f"step_{step}"

        # Split model into graphdef and state for saving
        graphdef, state = nnx.split(self.model)

        # Save with Orbax
        checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(state)

        checkpointer.save(
            path,
            {
                "model": state,
                "optimizer": self.opt_state,
                "step": step,
            },
            save_args=save_args,
        )

        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str | Path):
        """Load training checkpoint."""
        checkpointer = ocp.PyTreeCheckpointer()
        checkpoint = checkpointer.load(path)

        # Restore model state
        graphdef, _ = nnx.split(self.model)
        self.model = nnx.merge(graphdef, checkpoint["model"])

        # Restore optimizer state
        self.opt_state = checkpoint["optimizer"]

        # Restore step
        self.step = checkpoint["step"]

        logger.info(f"Checkpoint loaded: {path} (step {self.step})")

        return {
            "model": checkpoint["model"],
            "optimizer": checkpoint["optimizer"],
            "step": checkpoint["step"],
        }


# Orbax utilities
class orbax_utils:
    """Orbax checkpoint utilities."""

    @staticmethod
    def save_args_from_target(target):
        """Create save args from target structure."""
        from orbax.checkpoint import utils as orbax_utils

        def _create_save_args(pytree):
            if isinstance(pytree, dict):
                return {k: _create_save_args(v) for k, v in pytree.items()}
            elif isinstance(pytree, (jnp.ndarray, np.ndarray)):
                return ocp.array.ArrayArgs()
            else:
                return None

        return _create_save_args(target)
