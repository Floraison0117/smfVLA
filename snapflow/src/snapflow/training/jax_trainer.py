"""
JAX training loop for SnapFlow.

Implements SnapFlow training with progressive FM/consistency mixing.
Adapted from smfVLA's jax_trainer.py for SnapFlow's loss function.

Paper: https://arxiv.org/abs/2604.05656
"""

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from snapflow.training import freeze_utils

logger = logging.getLogger(__name__)


class SnapFlowTrainer:
    """
    SnapFlow JAX trainer.

    Implements progressive FM/consistency mixing:
    - FM component (probability alpha): Standard flow matching at random t
    - Consistency component (probability 1-alpha): 2-step Euler shortcut target

    Loss = alpha * L_FM + (1-alpha) * lambda * L_shortcut
    """

    def __init__(
        self,
        model: nnx.Module,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 30000,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints/finetuned/snapflow",
        log_dir: str = "logs/train/snapflow",
        save_every: int = 5000,
        log_every: int = 100,
        wandb_project: str = "snapflow",
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        train_config: dict[str, Any] | None = None,
        alpha: float = 0.5,
        lambda_consistency: float = 0.1,
        prediction_clamp_min: float = -20,
        prediction_clamp_max: float = 20,
    ):
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

        # SnapFlow-specific parameters
        self.alpha = alpha
        self.lambda_consistency = lambda_consistency
        self.prediction_clamp_min = prediction_clamp_min
        self.prediction_clamp_max = prediction_clamp_max

        # Freeze strategy (from config)
        self.freeze_patterns = self.train_config.get("freeze", freeze_utils.FREEZE_PATTERNS)
        self.trainable_patterns = self.train_config.get("trainable", freeze_utils.TRAINABLE_PATTERNS)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build trainable mask
        graphdef, state = nnx.split(model)
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

    def _setup_jit_train_step(self):
        """
        Pre-compile training step: split model into trainable/frozen,
        use jax.value_and_grad(fn, argnums=0) to build gradient graph only for trainable params.

        nnx.merge inside JIT → only traced once during compilation,
        subsequent calls are pure JAX array operations.
        """
        logger.info("=" * 60)
        logger.info("JIT compiling training step (first call slow, ~5s/step after)")
        logger.info("=" * 60)

        # Split model
        graphdef, state = nnx.split(self.model)
        flat = state.flat_state()
        all_paths = sorted(flat.keys())

        self._trainable_paths = [p for p in all_paths if self.trainable_mask.get(p, False)]
        frozen_paths = [p for p in all_paths if not self.trainable_mask.get(p, False)]

        # Frozen dict: JIT constants, no gradient graph
        self._frozen_dict = {p: flat[p] for p in frozen_paths}
        self._graphdef = graphdef

        def _size(v):
            arr = v.value if hasattr(v, 'value') else v
            return arr.size

        n_trainable = sum(_size(flat[p]) for p in self._trainable_paths)
        n_frozen = sum(_size(flat[p]) for p in frozen_paths)
        logger.info(f"  Trainable: {n_trainable:,} params ({n_trainable/1e6:.1f}M)")
        logger.info(f"  Frozen:    {n_frozen:,} params ({n_frozen/1e6:.1f}M)")

        # Capture closure variables (JIT constants)
        frozen_dict = self._frozen_dict
        trainable_paths = self._trainable_paths
        trainer = self  # Capture self for config access

        # Loss function: only differentiate w.r.t trainable params
        def loss_fn(trainable_values, batch, rng):
            from openpi.models.model import Observation
            from openpi.models.pi0 import make_attn_mask
            from snapflow.training.snapflow_loss import compute_snapflow_loss

            # Reconstruct full model (traced on first JIT call, compiled after)
            full = dict(frozen_dict)
            for p, v in zip(trainable_paths, trainable_values):
                full[p] = v
            st = nnx.State.from_flat_path(full)
            model = nnx.merge(graphdef, st)

            observation = Observation.from_dict(batch["observation"])
            actions = batch["actions"]
            action_mean = batch["action_mean"]
            action_std = batch["action_std"]

            # ── 预计算 prefix KV cache (优化: 只计算一次，4次前向共享) ──
            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
            # stop_gradient: prefix 来自 frozen VLM，切断 JVP 对 VLM 的回溯（见 docs §2/§7）
            prefix_tokens = jax.lax.stop_gradient(prefix_tokens)
            prefix_mask = jax.lax.stop_gradient(prefix_mask)
            prefix_ar_mask = jax.lax.stop_gradient(prefix_ar_mask)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

            # 计算 prefix 的 KV cache (用于后续的 suffix-only forward)
            (_, _), kv_cache = model.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
            )

            # 获取 batch_size 和序列长度 (用于构建 attention mask)
            batch_size = prefix_mask.shape[0]
            prefix_len = prefix_mask.shape[1]

            # Model forward function with target-time conditioning (suffix-only)
            def model_fn(params, obs, noisy_actions, r, t, s):
                """Forward: F_θ(noisy_actions, r, t, s) → velocity (suffix-only, 使用 kv_cache)."""
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix_with_target_time(
                    obs, noisy_actions, t=t, s=s
                )
                adarms_cond_list = [None, adarms_cond]

                # Suffix-only forward pass (使用 kv_cache 跳过 prefix 重复计算)
                suffix_len = suffix_mask.shape[1]
                suffix_causal_mask = make_attn_mask(suffix_mask, suffix_ar_mask)  # (batch, suffix_len, suffix_len)
                # suffix 的全局位置必须接在 prefix 之后（与 sample_actions 一致），
                # 否则 RoPE 位置与推理不一致 → train/infer mismatch。见 openpi pi0.py:259。
                suffix_positions = (
                    jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=1) - 1
                )

                # 构建完整的 attention mask: suffix tokens 可以 attend 到 prefix + suffix
                # 前缀部分: suffix tokens 可以 attend 到所有 prefix tokens
                prefix_to_suffix_attn = jnp.ones((batch_size, suffix_len, prefix_len), dtype=jnp.bool_)
                # 后缀部分: suffix tokens 只能 causal attend 到其他 suffix tokens
                # 连接成完整 mask: (batch, suffix_len, prefix_len + suffix_len)
                full_attn_mask = jnp.concatenate([prefix_to_suffix_attn, suffix_causal_mask], axis=2)

                (_, suffix_out), _ = model.PaliGemma.llm(
                    [None, suffix_tokens],  # ← prefix=None, 使用 kv_cache
                    mask=full_attn_mask,
                    positions=suffix_positions,
                    kv_cache=kv_cache,  # ← 复用 prefix 的 KV cache
                    adarms_cond=adarms_cond_list,
                )
                v = model.action_out_proj(suffix_out[:, -model.action_horizon:])

                # Clamp prediction to prevent numerical instabilities
                v = jnp.clip(v, trainer.prediction_clamp_min, trainer.prediction_clamp_max)
                return v

            loss, metrics = compute_snapflow_loss(
                model_fn=model_fn,
                params=None,
                observation=observation,
                actions=actions,
                action_mean=action_mean,
                action_std=action_std,
                rng=rng,
                alpha=trainer.alpha,
                lambda_consistency=trainer.lambda_consistency,
            )

            return loss, metrics

        # JIT compile: argnums=0 → only differentiate w.r.t trainable_values
        self._grad_fn = jax.jit(
            jax.value_and_grad(loss_fn, argnums=0, has_aux=True),
        )

        logger.info("JIT compilation complete")
        logger.info("=" * 60)

    def train_step(
        self,
        model: nnx.Module,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: dict[str, Any],
        rng: jax.Array,
    ) -> tuple[nnx.Module, optax.OptState, dict[str, float]]:
        """
        Single training step.

        Uses pre-compiled _grad_fn (jax.value_and_grad, argnums=0),
        only builds gradient graph for trainable params, frozen params are JIT constants.

        Returns:
            Updated model, opt_state, metrics
        """
        # Extract trainable params
        _, state = nnx.split(model)
        flat = state.flat_state()
        trainable_values = [flat[p] for p in self._trainable_paths]

        # Filter batch: only keep JAX-compatible arrays
        jax_batch = {
            "observation": batch["observation"],
            "actions": batch["actions"],
            "action_mean": batch["action_mean"],
            "action_std": batch["action_std"],
        }

        # Forward + gradient only for trainable params
        (loss, metrics), grads = self._grad_fn(trainable_values, jax_batch, rng)

        # Parameter update
        updates, new_opt_state = optimizer.update(grads, opt_state, params=trainable_values)
        new_trainable_values = optax.apply_updates(trainable_values, updates)

        # Reconstruct full model
        full = dict(self._frozen_dict)
        for p, v in zip(self._trainable_paths, new_trainable_values):
            full[p] = v
        new_state = nnx.State.from_flat_path(full)
        model = nnx.merge(self._graphdef, new_state)

        # Gradient/param norms
        def _to_array(x):
            if hasattr(x, 'value'):
                return x.value
            return x

        grad_norms = jnp.array([jnp.linalg.norm(_to_array(g)) for g in grads if g is not None])
        total_grad_norm = jnp.linalg.norm(grad_norms)
        param_norms = jnp.array([jnp.linalg.norm(_to_array(p)) for p in new_trainable_values])
        total_param_norm = jnp.linalg.norm(param_norms)

        metrics["grad_norm"] = total_grad_norm
        metrics["param_norm"] = total_param_norm

        return model, new_opt_state, {k: float(v) for k, v in metrics.items()}

    def _init_wandb(self):
        """Initialize wandb."""
        try:
            import wandb

            wandb.login()
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "method": "snapflow",
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "warmup_steps": self.warmup_steps,
                    "total_steps": self.total_steps,
                    "gradient_clip_norm": self.gradient_clip_norm,
                    "alpha": self.alpha,
                    "lambda_consistency": self.lambda_consistency,
                    "prediction_clamp": (self.prediction_clamp_min, self.prediction_clamp_max),
                    "freeze_patterns": self.freeze_patterns,
                    "trainable_patterns": self.trainable_patterns,
                    "param_stats": self.param_stats,
                    **self.wandb_config,
                },
            )
            logger.info(f"WandB initialized: project={self.wandb_project}, run={wandb.run.name}")
            return True
        except Exception as e:
            logger.warning(f"WandB initialization failed, skipping wandb logging: {e}")
            return False

    def train(self, data_loader, rng: jax.Array, resume_from: str | None = None):
        """
        Main training loop.

        Args:
            data_loader: Data loader (iterator)
            rng: JAX random key
            resume_from: Checkpoint path for resuming training (optional)
        """
        logger.info("=" * 60)
        logger.info(f"Starting SnapFlow training")
        logger.info("=" * 60)
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Alpha (FM ratio): {self.alpha}")
        logger.info(f"Lambda (consistency weight): {self.lambda_consistency}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        logger.info("-" * 60)

        # Initialize wandb
        use_wandb = self._init_wandb()

        # Initialize optimizer state
        graphdef, state = nnx.split(self.model)
        trainable_params = []
        flat = state.flat_state()
        for path in sorted(flat.keys()):
            if self.trainable_mask.get(path, False):
                trainable_params.append(flat[path])

        opt_state = self.optimizer.init(trainable_params)

        model = self.model

        # Resume from checkpoint if specified
        if resume_from is not None:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            model, restored_opt_state, resume_step = self.load_checkpoint(
                model, resume_from, opt_state
            )
            if restored_opt_state is not None:
                opt_state = restored_opt_state
            self.step = resume_step
            logger.info(f"Resumed to step {resume_step}, continuing training...")

        # JIT compile training step (critical optimization)
        self._setup_jit_train_step()

        # Note: opt_state already initialized at line 359, no need to rebuild
        # The trainable_params from first init are identical to _trainable_paths

        start_time = time.time()

        for step in range(self.step, self.total_steps):
            self.step = step

            # Get batch
            try:
                batch = next(data_iter)
            except (StopIteration, NameError):
                data_iter = iter(data_loader)
                batch = next(data_iter)

            # Prepare batch
            jax_batch = _prepare_batch(batch)

            # Training step
            rng, step_rng = jax.random.split(rng)
            model, opt_state, metrics = self.train_step(
                model, self.optimizer, opt_state, jax_batch, step_rng
            )

            # Debug logging (every 100 steps)
            if step % 100 == 0:
                logger.info(f"Step {step}: loss={metrics.get('loss_total', 'N/A'):.4f}")

            # Log metrics
            metrics["step"] = step
            try:
                lr_fn = self.optimizer[1].learning_rate
                metrics["lr"] = float(lr_fn(step))
            except (AttributeError, IndexError, TypeError):
                metrics["lr"] = self.learning_rate
            metrics["wall_time"] = time.time() - start_time
            self.train_log.append(metrics)

            # wandb logging
            if use_wandb and step % self.log_every == 0:
                import wandb

                wandb_metrics = {
                    "train/loss_total": metrics["loss_total"],
                    "train/loss_fm": metrics["loss_fm"],
                    "train/loss_shortcut": metrics["loss_shortcut"],
                    "train/lr": metrics["lr"],
                    "train/wall_time": metrics["wall_time"],
                    "train/grad_norm": metrics.get("grad_norm", 0),
                    "train/param_norm": metrics.get("param_norm", 0),
                }

                try:
                    devices = jax.devices()
                    if hasattr(devices[0], "memory_stats"):
                        mem = devices[0].memory_stats()
                        wandb_metrics["system/gpu_memory_used_gb"] = mem.get("bytes_in_use", 0) / 1e9
                except Exception:
                    pass

                wandb.log(wandb_metrics, step=step)

            if step % self.log_every == 0:
                elapsed = time.time() - start_time
                msg = (
                    f"Step {step}/{self.total_steps} | "
                    f"loss={metrics['loss_total']:.4f} | "
                    f"fm={metrics['loss_fm']:.4f} | "
                    f"shortcut={metrics['loss_shortcut']:.4f} | "
                    f"grad={metrics.get('grad_norm', 0):.4f} | "
                    f"time={elapsed:.1f}s"
                )
                logger.info(msg)

            # Save checkpoint
            if (step + 1) % self.save_every == 0:
                self.save_checkpoint(model, opt_state, step + 1, metrics)

        # Save final checkpoint
        self.save_checkpoint(model, opt_state, self.total_steps, metrics)

        # Save training log
        self._save_train_log()

        total_time = time.time() - start_time

        # Save training summary
        summary_dir = self._save_training_summary(total_time, metrics)

        # Finish wandb
        if use_wandb:
            import wandb
            wandb.finish()

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Training summary: {summary_dir}")
        logger.info("=" * 60)

        self.model = model
        return model

    def save_checkpoint(
        self,
        model: nnx.Module,
        opt_state: optax.OptState,
        step: int,
        metrics: dict[str, float],
    ):
        """Save checkpoint (model params + optimizer state + training metadata)."""
        import orbax.checkpoint as ocp

        ckpt_dir = self.checkpoint_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model params
        _, state = nnx.split(model)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(str(ckpt_dir / "params"), state.to_pure_dict())

        # Save optimizer state
        checkpointer.save(str(ckpt_dir / "opt_state"), opt_state)

        # Save training metadata
        train_state = {
            "step": step,
            "metrics": metrics,
            "param_stats": self.param_stats,
        }
        with open(ckpt_dir / "train_state.json", "w") as f:
            json.dump(train_state, f, indent=2)

        logger.info(f"Saved checkpoint to {ckpt_dir}")

    def load_checkpoint(
        self,
        model: nnx.Module,
        ckpt_path: str,
        opt_state: optax.OptState | None = None,
    ) -> tuple[nnx.Module, optax.OptState | None, int]:
        """Load checkpoint (model params + optimizer state + training step)."""
        import orbax.checkpoint as ocp

        ckpt_path = Path(ckpt_path)
        checkpointer = ocp.PyTreeCheckpointer()

        # Load model params
        params = checkpointer.restore(str(ckpt_path / "params"))
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(params)
        model = nnx.merge(graphdef, state)

        # Load optimizer state if exists
        opt_state_path = ckpt_path / "opt_state"
        if opt_state_path.exists() and opt_state is not None:
            try:
                restored_opt = checkpointer.restore(str(opt_state_path))
                opt_state = jax.tree.map(
                    lambda old, new: new if new is not None else old,
                    opt_state,
                    restored_opt,
                )
                logger.info(f"Loaded optimizer state from {opt_state_path}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Optimizer state mismatch, using fresh state: {e}")

        # Load training step
        step = 0
        train_state_path = ckpt_path / "train_state.json"
        if train_state_path.exists():
            with open(train_state_path) as f:
                train_state = json.load(f)
            step = train_state["step"]

        logger.info(f"Loaded checkpoint from {ckpt_path}, step={step}")
        return model, opt_state, step

    def _save_train_log(self):
        """Save training log as JSONL file."""
        log_path = self.log_dir / "train_log.jsonl"
        with open(log_path, "w") as f:
            for entry in self.train_log:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Saved training log to {log_path}")

    def _save_training_summary(self, total_time: float, final_metrics: dict[str, float]) -> Path:
        """Save training summary to timestamped directory."""
        now = datetime.now()
        desc_short = self.train_config.get("description", "")[:30].replace(" ", "_")
        dir_name = f"{now.strftime('%Y%m%d_%H%M%S')}_snapflow"
        if desc_short:
            dir_name += f"_{desc_short}"

        summary_dir = self.log_dir / dir_name
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "method": "snapflow",
            "description": self.train_config.get("description", ""),
            "training": {
                "total_steps": self.total_steps,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "gradient_clip_norm": self.gradient_clip_norm,
                "batch_size": self.train_config.get("batch_size"),
                "precision": self.train_config.get("precision"),
                "alpha": self.alpha,
                "lambda_consistency": self.lambda_consistency,
                "prediction_clamp": (self.prediction_clamp_min, self.prediction_clamp_max),
                "freeze_patterns": self.freeze_patterns,
                "trainable_patterns": self.trainable_patterns,
            },
            "results": {
                "final_loss_total": final_metrics.get("loss_total"),
                "final_loss_fm": final_metrics.get("loss_fm"),
                "final_loss_shortcut": final_metrics.get("loss_shortcut"),
                "final_grad_norm": final_metrics.get("grad_norm", 0),
                "total_time_seconds": round(total_time, 1),
                "total_time_human": f"{total_time / 3600:.1f}h" if total_time > 3600 else f"{total_time / 60:.1f}min",
                "steps_per_second": round(self.total_steps / total_time, 2) if total_time > 0 else 0,
            },
            "param_stats": self.param_stats,
            "paths": {
                "checkpoint_dir": str(self.checkpoint_dir),
                "log_dir": str(self.log_dir),
            },
            "timestamp": now.isoformat(),
        }
        with open(summary_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        log_src = self.log_dir / "train_log.jsonl"
        if log_src.exists():
            shutil.copy2(log_src, summary_dir / "train_log.jsonl")

        if self.train_config:
            with open(summary_dir / "config.yaml", "w") as f:
                yaml.dump(self.train_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Training summary saved to: {summary_dir}")
        return summary_dir


def _prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy batch to JAX arrays (supports nested dicts)."""
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
