"""
JAX training loop for DMF (Decoupled MeanFlow).

Implements DMF training with:
- Selective parameter updates (trainable/frozen split)
- AdamW optimizer with warmup + cosine decay
- Gradient clipping
- EMA of trainable params — eval loads EMA model
- Orbax checkpoint save/load with resume support
- WandB logging

Based on DMF paper: "Decoupled MeanFlow" (ICLR 2026).
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from dmf_vla.training import freeze_utils

logger = logging.getLogger(__name__)


def _to_array(v):
    """Extract raw JAX array from NNX Variable or pass through."""
    return v.value if hasattr(v, "value") else v


def _prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy arrays in a batch dict to JAX arrays."""
    result = {}
    for key, val in batch.items():
        if isinstance(val, dict):
            result[key] = _prepare_batch(val)
        elif isinstance(val, np.ndarray):
            result[key] = jnp.asarray(val)
        elif isinstance(val, list):
            result[key] = val
        else:
            result[key] = val
    return result


class _PrefetchIterator:
    """Threaded prefetch iterator that prepares the next batch while GPU computes."""

    def __init__(self, data_loader, maxsize: int = 2):
        self._data_iter = iter(data_loader)
        self._queue = queue.Queue(maxsize=maxsize)
        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop:
            try:
                batch = next(self._data_iter)
            except StopIteration:
                continue  # data loader is infinite
            except Exception as e:
                if not self._stop:
                    self._queue.put(e)
                return
            if not self._stop:
                self._queue.put(batch)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self._stop = True
        # Drain queue to unblock worker
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass


class DMFTrainer:
    def __init__(
        self,
        model: nnx.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 500,
        total_steps: int = 40000,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints/dmf_finetuned",
        log_dir: str = "logs/train/dmf",
        save_every: int = 5000,
        log_every: int = 100,
        wandb_project: str = "dmf-vla",
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        train_config: dict[str, Any] | None = None,
    ):
        if len(jax.devices("gpu")) > 0:
            jax.config.update("jax_platforms", "cuda")
            logger.info(f"Using GPU: {jax.devices('gpu')[0]}")
        else:
            logger.warning("No GPU devices found, falling back to CPU")

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.save_every = save_every
        self.log_every = log_every
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self.train_config = train_config or {}

        # DMF-specific hyperparams
        self.dmf_depth_ratio = self.train_config.get("dmf_depth_ratio", 0.67)
        self.use_logvar = self.train_config.get("use_logvar", True)
        self.P_mean = self.train_config.get("P_mean", 0.0)
        self.P_mean_t = self.train_config.get("P_mean_t", 0.4)
        self.P_mean_r = self.train_config.get("P_mean_r", -1.2)
        self.P_std = self.train_config.get("P_std", 1.0)
        self.P_std_t = self.train_config.get("P_std_t", 1.0)
        self.P_std_r = self.train_config.get("P_std_r", 1.0)
        self.ema_decay = self.train_config.get("ema_decay", 0.9999)

        # Freeze/trainable patterns from config
        self.freeze_patterns = self.train_config.get("freeze", freeze_utils.FREEZE_PATTERNS)
        self.trainable_patterns = self.train_config.get("trainable", freeze_utils.TRAINABLE_PATTERNS)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build trainable mask
        graphdef, state = nnx.split(model)
        self._graphdef = graphdef
        self.trainable_mask = freeze_utils.build_trainable_mask(
            state,
            freeze_patterns=self.freeze_patterns,
            trainable_patterns=self.trainable_patterns,
        )
        self.param_stats = freeze_utils.print_param_summary(state, self.trainable_mask)

        # Build optimizer
        self.optimizer = self._build_optimizer()

        # Training state
        self.step = 0
        self.train_log = []
        self.ema_values = None

        # Save config alongside checkpoints
        config_path = self.checkpoint_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.train_config, f)

    def _build_optimizer(self) -> optax.GradientTransformation:
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
        return optax.chain(
            optax.clip_by_global_norm(self.gradient_clip_norm),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.weight_decay,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
            ),
        )

    def _setup_jit_train_step(self):
        """JIT-compile the training step: split frozen/trainable params, compile grad_fn."""
        logger.info("=" * 60)
        logger.info("Compiling JIT training step...")
        logger.info("=" * 60)

        _, state = nnx.split(self.model)
        flat = state.flat_state()
        all_paths = sorted(flat.keys())

        self._trainable_paths = [p for p in all_paths if self.trainable_mask.get(p, False)]
        frozen_paths = [p for p in all_paths if not self.trainable_mask.get(p, False)]

        self._frozen_dict = {p: flat[p] for p in frozen_paths}

        n_trainable = sum(_to_array(flat[p]).size for p in self._trainable_paths)
        n_frozen = sum(_to_array(flat[p]).size for p in frozen_paths)
        logger.info(f"  Trainable: {n_trainable:,} params ({n_trainable / 1e6:.1f}M)")
        logger.info(f"  Frozen:    {n_frozen:,} params ({n_frozen / 1e6:.1f}M)")

        frozen_dict = self._frozen_dict
        trainable_paths = self._trainable_paths
        graphdef = self._graphdef
        trainer = self

        def loss_fn(trainable_values, jax_batch, rng):
            from openpi.models.model import Observation
            from dmf_vla.training.dmf_loss import compute_dmf_loss

            # Reconstruct model
            full = dict(frozen_dict)
            for p, v in zip(trainable_paths, trainable_values):
                full[p] = v
            st = nnx.State.from_flat_path(full)
            model = nnx.merge(graphdef, st)

            # Pre-compute vision encoder outputs with stop_gradient to prevent
            # JVP from tracing through the frozen SigLIP vision encoder.
            observation = Observation.from_dict(jax_batch["observation"])
            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
            prefix_tokens = jax.lax.stop_gradient(prefix_tokens)
            prefix_mask = jax.lax.stop_gradient(prefix_mask)
            prefix_ar_mask = jax.lax.stop_gradient(prefix_ar_mask)

            actions = jax_batch["actions"]
            action_mean = jax_batch["action_mean"]
            action_std = jax_batch["action_std"]

            model_fn = model._dmf_model_fn(prefix_tokens, prefix_mask, prefix_ar_mask)

            loss, metrics = compute_dmf_loss(
                model_fn=model_fn,
                params=None,
                observation=observation,
                actions=actions,
                action_mean=action_mean,
                action_std=action_std,
                rng=rng,
                p_mean=trainer.P_mean,
                p_mean_t=trainer.P_mean_t,
                p_mean_r=trainer.P_mean_r,
                p_std=trainer.P_std,
                p_std_t=trainer.P_std_t,
                p_std_r=trainer.P_std_r,
                use_logvar=trainer.use_logvar,
            )
            return loss, metrics

        self._grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))
        logger.info("JIT compilation complete")

    def train_step(
        self,
        model: nnx.Module,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        jax_batch: dict[str, Any],
        rng: jax.Array,
    ) -> tuple[nnx.Module, optax.OptState, dict[str, float]]:
        """Single training step.  jax_batch must already be JAX arrays (strings stripped)."""

        _, state = nnx.split(model)
        flat = state.flat_state()
        trainable_values = [flat[p] for p in self._trainable_paths]

        # Strip non-array fields that JIT cannot trace
        jit_batch = {
            k: v for k, v in jax_batch.items()
            if k in ("observation", "actions", "action_mean", "action_std")
        }

        (loss, metrics), grads = self._grad_fn(trainable_values, jit_batch, rng)

        updates, new_opt_state = optimizer.update(grads, opt_state, params=trainable_values)
        new_trainable_values = optax.apply_updates(trainable_values, updates)

        # EMA update on trainable params
        if self.ema_values is not None:
            new_arrs = [_to_array(v) for v in new_trainable_values]
            self.ema_values = [
                self.ema_decay * ema + (1.0 - self.ema_decay) * new
                for ema, new in zip(self.ema_values, new_arrs)
            ]

        # Reconstruct model
        full = dict(self._frozen_dict)
        for p, v in zip(self._trainable_paths, new_trainable_values):
            full[p] = v
        new_state = nnx.State.from_flat_path(full)
        model = nnx.merge(self._graphdef, new_state)

        # Gradient norm
        grad_norm = jnp.sqrt(sum(
            jnp.sum(jnp.square(_to_array(g))) for g in grads if g is not None
        ))
        metrics["grad_norm"] = float(grad_norm)

        return model, new_opt_state, {k: float(v) for k, v in metrics.items()}

    def train(
        self,
        data_loader,
        resume_from: str | None = None,
    ):
        if resume_from:
            self._load_checkpoint(resume_from)

        self._setup_jit_train_step()

        # Initialize optimizer state from current trainable params
        _, state = nnx.split(self.model)
        flat = state.flat_state()
        trainable_values = [flat[p] for p in self._trainable_paths]
        opt_state = self.optimizer.init(trainable_values)

        # Initialize EMA from current trainable params
        if self.ema_values is None:
            self.ema_values = [_to_array(v).copy() for v in trainable_values]
            logger.info(f"Initialized EMA ({len(self.ema_values)} arrays, decay={self.ema_decay})")

        # WandB
        use_wandb = False
        if self.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=self.wandb_project,
                    name=self.wandb_run_name,
                    config=self.wandb_config,
                )
                use_wandb = True
            except Exception as e:
                logger.warning(f"WandB init failed: {e}")

        model = self.model
        rng = jax.random.PRNGKey(42)
        start_time = time.time()

        logger.info(f"Starting training: {self.total_steps} steps")
        logger.info(f"  Batch size: {self.train_config.get('batch_size', 'N/A')}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  DMF depth ratio: {self.dmf_depth_ratio}")
        logger.info(f"  EMA decay: {self.ema_decay}")
        logger.info(f"  P_mean: {self.P_mean}, P_mean_t: {self.P_mean_t}, P_mean_r: {self.P_mean_r}")
        logger.info(f"  JAX backend: {jax.default_backend()}, devices: {jax.devices('gpu')}")

        # Use prefetch to overlap data loading with GPU computation
        prefetch = _PrefetchIterator(data_loader, maxsize=2)

        while self.step < self.total_steps:
            batch = next(prefetch)

            jax_batch = _prepare_batch(batch)

            rng, step_rng = jax.random.split(rng)
            step_start = time.time()

            model, opt_state, metrics = self.train_step(
                model, self.optimizer, opt_state, jax_batch, step_rng
            )

            self.step += 1

            if self.step % self.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.step / elapsed
                step_time_ms = (time.time() - step_start) * 1000

                log_msg = (
                    f"Step {self.step:6d}/{self.total_steps} | "
                    f"loss={metrics['loss_total']:.4f} | "
                    f"fm={metrics['loss_fm']:.4f} | "
                    f"mf={metrics['loss_mf']:.4f} | "
                    f"grad={metrics['grad_norm']:.4f} | "
                    f"t_fm={metrics['t_fm_mean']:.3f} | "
                    f"t_mf={metrics['t_mf_mean']:.3f} | "
                    f"r_mf={metrics['r_mf_mean']:.3f} | "
                    f"{steps_per_sec:.1f} steps/s | "
                    f"{step_time_ms:.0f}ms/step"
                )
                logger.info(log_msg)

                if use_wandb:
                    try:
                        import wandb
                        wandb_metrics = {
                            "train/loss_total": metrics["loss_total"],
                            "train/loss_fm": metrics["loss_fm"],
                            "train/loss_mf": metrics["loss_mf"],
                            "train/loss_fm_logvar": metrics.get("loss_fm_logvar", 0),
                            "train/loss_mf_logvar": metrics.get("loss_mf_logvar", 0),
                            "train/grad_norm": metrics["grad_norm"],
                            "train/t_fm_mean": metrics["t_fm_mean"],
                            "train/t_mf_mean": metrics["t_mf_mean"],
                            "train/r_mf_mean": metrics["r_mf_mean"],
                            "train/t_mf_r_mf_gap": metrics.get("t_mf_r_mf_gap", 0),
                            "train/du_dt_norm": metrics.get("du_dt_norm", 0),
                            "train/u_norm": metrics.get("u_norm", 0),
                            "train/logvar_fm_mean": metrics.get("logvar_fm_mean", 0),
                            "train/logvar_mf_mean": metrics.get("logvar_mf_mean", 0),
                            "train/steps_per_sec": steps_per_sec,
                            "train/step_time_ms": step_time_ms,
                            "train/lr": float(self.optimizer[1].learning_rate(self.step)),
                            "step": self.step,
                        }
                        wandb.log(wandb_metrics)
                    except Exception:
                        pass

            if self.step % self.save_every == 0:
                self._save_checkpoint(model, opt_state)
                logger.info(f"Saved checkpoint at step {self.step}")

        # Final save
        self._save_checkpoint(model, opt_state)
        logger.info("Training complete!")

        prefetch.close()

        if use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        self.model = model
        return model

    def _save_checkpoint(self, model, opt_state):
        """Save EMA model to params/ (eval), training model to params_training/ (resume)."""
        import orbax.checkpoint as ocp

        ckpt_dir = self.checkpoint_dir / f"step_{self.step:07d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpointer = ocp.PyTreeCheckpointer()

        # EMA model -> params/ (loaded by eval)
        if self.ema_values is not None:
            full_ema = dict(self._frozen_dict)
            for p, v in zip(self._trainable_paths, self.ema_values):
                full_ema[p] = v
            ema_state = nnx.State.from_flat_path(full_ema)
            _, ema_nnx_state = nnx.split(nnx.merge(self._graphdef, ema_state))
            checkpointer.save(str(ckpt_dir / "params"), ema_nnx_state.to_pure_dict())
            logger.info(f"  Saved EMA model to params/")
        else:
            _, train_nnx_state = nnx.split(model)
            checkpointer.save(str(ckpt_dir / "params"), train_nnx_state.to_pure_dict())

        # Training model -> params_training/ (for resume)
        _, train_nnx_state = nnx.split(model)
        checkpointer.save(str(ckpt_dir / "params_training"), train_nnx_state.to_pure_dict())

        # Optimizer state -> opt_state/
        checkpointer.save(str(ckpt_dir / "opt_state"), opt_state)

        # Metadata
        train_state = {
            "step": self.step,
            "param_stats": self.param_stats,
            "ema_decay": self.ema_decay,
        }
        with open(ckpt_dir / "train_state.json", "w") as f:
            json.dump(train_state, f, indent=2, default=str)

        # Copy config alongside checkpoint for traceability
        config_src = self.checkpoint_dir / "config.yaml"
        if config_src.exists():
            import shutil
            shutil.copy2(config_src, ckpt_dir / "config.yaml")

        logger.info(f"Saved checkpoint to {ckpt_dir}")

    def _load_checkpoint(self, path: str):
        """Load training checkpoint: training model + EMA + optimizer state."""
        import orbax.checkpoint as ocp
        import flax.traverse_util as traverse_util

        logger.info(f"Loading checkpoint from {path}")
        ckpt_path = Path(path)
        checkpointer = ocp.PyTreeCheckpointer()

        # Load training model
        train_params_path = ckpt_path / "params_training"
        if train_params_path.exists():
            params = checkpointer.restore(str(train_params_path))
            logger.info("Loaded training model from params_training/")
        else:
            params = checkpointer.restore(str(ckpt_path / "params"))
            logger.info("Loaded model from params/ (no params_training/)")

        _, state = nnx.split(self.model)
        state.replace_by_pure_dict(params)
        self.model = nnx.merge(self._graphdef, state)

        # Load EMA values
        ema_params_path = ckpt_path / "params"
        if ema_params_path.exists():
            ema_params = checkpointer.restore(str(ema_params_path))
            flat_ema = traverse_util.flatten_dict(ema_params)

            _, state2 = nnx.split(self.model)
            flat_state = traverse_util.flatten_dict(state2.to_pure_dict())

            self.ema_values = []
            for p in self._trainable_paths:
                if p in flat_ema:
                    self.ema_values.append(jnp.array(flat_ema[p]))
                elif p in flat_state:
                    self.ema_values.append(_to_array(flat_state[p]).copy())
                else:
                    self.ema_values.append(jnp.zeros(1))
            logger.info(f"Loaded EMA values ({len(self.ema_values)} arrays)")

        # Load step
        train_state_path = ckpt_path / "train_state.json"
        if train_state_path.exists():
            with open(train_state_path) as f:
                train_state = json.load(f)
            self.step = train_state.get("step", 0)
            self.ema_decay = train_state.get("ema_decay", self.ema_decay)
        else:
            try:
                self.step = int(ckpt_path.name.split("_")[1])
            except (IndexError, ValueError):
                self.step = 0

        logger.info(f"Resumed from step {self.step}")
