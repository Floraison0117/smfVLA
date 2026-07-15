"""JAX training loop for Pi-Flow: velocity imitation distillation.

Two-model setup:
- Student (Pi05PiFlow): predicts GMM parameters, trained via velocity imitation
- Teacher (Pi0): frozen pi0.5, provides velocity supervision
"""

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from piflow_vla.training import freeze_utils
from piflow_vla.training.piflow_loss import compute_piflow_loss

from openpi.models import model as _model
from openpi.models import pi0 as _pi0

logger = logging.getLogger(__name__)


def _to_array(v):
    return v.value if hasattr(v, "value") else v


def _prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
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
                break
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
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass


class PiFlowTrainer:
    def __init__(
        self,
        student_model: nnx.Module,
        teacher_model: nnx.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        total_steps: int = 30000,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints/piflow_finetuned",
        log_dir: str = "logs/train/piflow",
        save_every: int = 5000,
        log_every: int = 50,
        inner_substeps: int = 8,
        teacher_query_points: int = 4,
        nfe: int = 1,
        ema_decay: float = 0.9999,
        wandb_project: str = "piflow",
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        train_config: dict[str, Any] | None = None,
    ):
        try:
            gpu_devs = jax.devices("cuda")
            if len(gpu_devs) > 0:
                jax.config.update("jax_platforms", "cuda")
                logger.info(f"Using GPU: {gpu_devs[0]}")
            else:
                logger.warning("No GPU devices found, falling back to CPU")
        except RuntimeError:
            logger.warning("CUDA backend not available, using CPU")

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.save_every = save_every
        self.log_every = log_every
        self.inner_substeps = inner_substeps
        self.teacher_query_points = teacher_query_points
        self.nfe = nfe
        self.ema_decay = ema_decay
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self.train_config = train_config or {}

        self.freeze_patterns = self.train_config.get("freeze", freeze_utils.FREEZE_PATTERNS)
        self.trainable_patterns = self.train_config.get(
            "trainable", freeze_utils.TRAINABLE_PATTERNS
        )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Build student trainable mask
        graphdef, state = nnx.split(student_model)
        self._student_graphdef = graphdef
        self.trainable_mask = freeze_utils.build_trainable_mask(
            state,
            freeze_patterns=self.freeze_patterns,
            trainable_patterns=self.trainable_patterns,
        )
        self.param_stats = freeze_utils.print_param_summary(state, self.trainable_mask)

        # Extract teacher state (all frozen)
        self._teacher_graphdef, teacher_state = nnx.split(teacher_model)
        self._teacher_flat = teacher_state.flat_state()
        self._teacher_paths = sorted(self._teacher_flat.keys())
        logger.info(f"Teacher params: {len(self._teacher_paths)} arrays, all frozen")

        self.optimizer = self._build_optimizer()
        self.opt_state = None
        self.step = 0
        self.train_log = []
        self.ema_values = None

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
        logger.info("=" * 60)
        logger.info("Compiling JIT training step (Pi-Flow)...")
        logger.info("=" * 60)

        # Student param split
        _, state = nnx.split(self.student_model)
        flat = state.flat_state()
        all_paths = sorted(flat.keys())

        self._student_trainable_paths = [p for p in all_paths if self.trainable_mask.get(p, False)]
        self._student_frozen_paths = [p for p in all_paths if not self.trainable_mask.get(p, False)]
        self._student_frozen_dict = {p: flat[p] for p in self._student_frozen_paths}

        n_trainable = sum(_to_array(flat[p]).size for p in self._student_trainable_paths)
        n_frozen = sum(_to_array(flat[p]).size for p in self._student_frozen_paths)
        n_teacher = sum(_to_array(self._teacher_flat[p]).size for p in self._teacher_paths)
        logger.info(f"  Student trainable: {n_trainable:,} ({n_trainable/1e6:.1f}M)")
        logger.info(f"  Student frozen:    {n_frozen:,} ({n_frozen/1e6:.1f}M)")
        logger.info(f"  Teacher (frozen):  {n_teacher:,} ({n_teacher/1e6:.1f}M)")

        student_trainable_paths = self._student_trainable_paths
        student_frozen_paths = self._student_frozen_paths
        teacher_paths = self._teacher_paths
        student_graphdef = self._student_graphdef
        teacher_graphdef = self._teacher_graphdef
        inner_substeps = self.inner_substeps
        teacher_query_points = self.teacher_query_points
        nfe = self.nfe
        logger.info(
            f"  NFE: {nfe}, inner_substeps: {inner_substeps}, "
            f"teacher_query_points/seg: {teacher_query_points}"
        )

        def loss_fn(student_trainable_vals, student_frozen_vals, teacher_vals, jax_batch, rng):
            teacher_vals = jax.lax.stop_gradient(teacher_vals)

            # ── Reconstruct student ──
            full = {}
            for p, v in zip(student_frozen_paths, student_frozen_vals):
                full[p] = v
            for p, v in zip(student_trainable_paths, student_trainable_vals):
                full[p] = v
            student_state = nnx.State.from_flat_path(full)
            student = nnx.merge(student_graphdef, student_state)

            # ── Reconstruct teacher ──
            teacher_full = {}
            for p, v in zip(teacher_paths, teacher_vals):
                teacher_full[p] = v
            teacher_state = nnx.State.from_flat_path(teacher_full)
            teacher = nnx.merge(teacher_graphdef, teacher_state)

            observation = _model.Observation.from_dict(jax_batch["observation"])

            # ── Precompute student prefix KV cache (stop_gradient) ──
            # The VLM backbone (3B params) is frozen. By precomputing the prefix
            # KV cache once and stop_gradient'ing it, the backward pass only
            # penetrates the action expert (~430M trainable params), not the
            # VLM. This matches DMF's optimization and avoids storing 3B
            # activations for backward.
            s_prefix_tokens, s_prefix_mask, s_prefix_ar_mask = student.embed_prefix(observation)
            s_prefix_tokens = jax.lax.stop_gradient(s_prefix_tokens)
            s_prefix_mask = jax.lax.stop_gradient(s_prefix_mask)
            s_prefix_ar_mask = jax.lax.stop_gradient(s_prefix_ar_mask)
            s_prefix_attn_mask = _pi0.make_attn_mask(s_prefix_mask, s_prefix_ar_mask)
            s_prefix_positions = jnp.cumsum(s_prefix_mask, axis=1) - 1
            _, s_prefix_kv = student.PaliGemma.llm(
                [s_prefix_tokens, None],
                mask=s_prefix_attn_mask,
                positions=s_prefix_positions,
            )
            s_prefix_kv = jax.lax.stop_gradient(s_prefix_kv)

            # Student GMM forward closure — uses cached prefix KV
            def student_gmm_fn(obs, x_t, t):
                return student.forward_gmm(
                    obs,
                    x_t,
                    t,
                    prefix_tokens=s_prefix_tokens,
                    prefix_mask=s_prefix_mask,
                    prefix_ar_mask=s_prefix_ar_mask,
                    prefix_kv_cache=s_prefix_kv,
                )

            # Teacher velocity with KV cache (prefix computed once per observation).
            # For multi-NFE: teacher_vel_fn is called once per segment, but all
            # segments share the same observation -> same prefix -> same KV cache.
            # We fill the cache on first call and reuse it for all M query steps.
            # The cache is built lazily inside teacher_vel_fn (captured via mutable
            # container) so jit traces it as a fixed-shape computation.
            t_prefix_tokens, t_prefix_mask, t_prefix_ar_mask = teacher.embed_prefix(observation)
            # Build prefix attention mask and positions for cache-fill pass
            prefix_attn_mask = _pi0.make_attn_mask(t_prefix_mask, t_prefix_ar_mask)
            prefix_positions = jnp.cumsum(t_prefix_mask, axis=1) - 1
            # Fill KV cache once (prefix-only forward, no suffix)
            _, kv_cache = teacher.PaliGemma.llm(
                [t_prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
            )

            def teacher_vel_fn(obs, states, times):
                """states: [B, M, H, D], times: [B, M] -> velocities: [B, M, H, D]"""
                B, M, H, D = states.shape

                def query_step(_carry, m):
                    x_t = states[:, m, :, :]
                    t = times[:, m]
                    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = teacher.embed_suffix(
                        obs, x_t, jnp.broadcast_to(t, (B,))
                    )
                    # Build combined attention mask for suffix queries:
                    # [suffix->prefix (broadcast), suffix->suffix (causal)]
                    suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
                    s2p_attn_mask = einops.repeat(
                        t_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
                    )
                    full_attn_mask = jnp.concatenate([s2p_attn_mask, suffix_attn_mask], axis=-1)
                    # Suffix positions continue after prefix
                    suffix_positions = (
                        jnp.sum(t_prefix_mask, axis=-1)[:, None]
                        + jnp.cumsum(suffix_mask, axis=-1)
                        - 1
                    )

                    (_, suffix_out), _ = teacher.PaliGemma.llm(
                        [None, suffix_tokens],
                        mask=full_attn_mask,
                        positions=suffix_positions,
                        kv_cache=kv_cache,
                        adarms_cond=[None, adarms_cond],
                    )
                    v_t = teacher.action_out_proj(suffix_out[:, -teacher.action_horizon :])
                    return None, v_t  # v_t: [B, H, D]

                _, vels = jax.lax.scan(query_step, None, jnp.arange(M))
                return jnp.transpose(vels, (1, 0, 2, 3))  # [B, M, H, D]

            return compute_piflow_loss(
                student_gmm_fn=student_gmm_fn,
                teacher_vel_fn=teacher_vel_fn,
                observation=observation,
                actions=jax_batch["actions"],
                rng=rng,
                nfe=nfe,
                inner_substeps=inner_substeps,
                teacher_query_points=teacher_query_points,
                stop_gradient_rollout=True,
            )

        self._grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))
        logger.info("JIT compilation complete")

    def train_step(self, jax_batch, rng):
        _, state = nnx.split(self.student_model)
        flat = state.flat_state()
        trainable_values = [flat[p] for p in self._student_trainable_paths]
        frozen_values = [flat[p] for p in self._student_frozen_paths]
        teacher_values = [self._teacher_flat[p] for p in self._teacher_paths]

        jit_batch = {k: v for k, v in jax_batch.items() if k in ("observation", "actions")}

        (loss, metrics), grads = self._grad_fn(
            trainable_values, frozen_values, teacher_values, jit_batch, rng
        )

        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, params=trainable_values
        )
        new_trainable_values = optax.apply_updates(trainable_values, updates)

        # EMA update
        if self.ema_values is not None:
            new_arrs = [_to_array(v) for v in new_trainable_values]
            self.ema_values = [
                self.ema_decay * ema + (1.0 - self.ema_decay) * new
                for ema, new in zip(self.ema_values, new_arrs)
            ]

        # Reconstruct student model
        full = dict(self._student_frozen_dict)
        for p, v in zip(self._student_trainable_paths, new_trainable_values):
            full[p] = v
        new_state = nnx.State.from_flat_path(full)
        self.student_model = nnx.merge(self._student_graphdef, new_state)

        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(_to_array(g))) for g in grads if g is not None))
        metrics["grad_norm"] = float(grad_norm)

        float_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, "shape") and v.shape == ():
                float_metrics[k] = float(v)
            elif isinstance(v, (int, float)):
                float_metrics[k] = float(v)
            else:
                float_metrics[k] = v
        return float_metrics

    def train(self, data_loader, resume_from: str | None = None):
        if resume_from:
            self._load_checkpoint(resume_from)

        self._setup_jit_train_step()

        # Initialize EMA and optimizer state
        if self.ema_values is None:
            _, state = nnx.split(self.student_model)
            flat = state.flat_state()
            trainable_values = [flat[p] for p in self._student_trainable_paths]
            self.ema_values = [_to_array(v).copy() for v in trainable_values]
            logger.info(f"Initialized EMA ({len(self.ema_values)} arrays, decay={self.ema_decay})")

        if self.opt_state is None:
            _, state = nnx.split(self.student_model)
            flat = state.flat_state()
            trainable_values = [flat[p] for p in self._student_trainable_paths]
            self.opt_state = self.optimizer.init(trainable_values)
            logger.info("Initialized optimizer state")

        use_wandb = False
        if self.wandb_project:
            try:
                import wandb

                wandb.init(
                    project=self.wandb_project, name=self.wandb_run_name, config=self.wandb_config
                )
                use_wandb = True
            except Exception as e:
                logger.warning(f"WandB init failed: {e}")

        rng = jax.random.PRNGKey(42)
        start_time = time.time()

        logger.info(f"Starting Pi-Flow training: {self.total_steps} steps")
        logger.info(f"  Batch size: {self.train_config.get('batch_size', 'N/A')}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  NFE: {self.nfe}")
        logger.info(f"  Inner substeps (total): {self.inner_substeps}")
        logger.info(f"  Teacher query points/seg: {self.teacher_query_points}")
        logger.info(f"  EMA decay: {self.ema_decay}")

        prefetch = _PrefetchIterator(data_loader, maxsize=2)

        try:
            while self.step < self.total_steps:
                batch = next(prefetch)
                jax_batch = _prepare_batch(batch)
                rng, step_rng = jax.random.split(rng)
                step_start = time.time()

                metrics = self.train_step(jax_batch, step_rng)
                self.step += 1

                if self.step % self.log_every == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.step / max(elapsed, 1e-8)
                    step_time_ms = (time.time() - step_start) * 1000

                    log_msg = (
                        f"Step {self.step:6d}/{self.total_steps} | "
                        f"loss={metrics['loss_total']:.4f} | "
                        f"vel_diff={metrics.get('vel_diff_norm', 0):.4f} | "
                        f"grad={metrics['grad_norm']:.4f} | "
                        f"means_norm={metrics.get('means_norm', 0):.3f} | "
                        f"logstd={metrics.get('log_stds_mean', 0):.3f} | "
                        f"{steps_per_sec:.1f} steps/s | {step_time_ms:.0f}ms/step"
                    )
                    logger.info(log_msg)

                    if use_wandb:
                        try:
                            import wandb

                            wandb_metrics = {"train/" + k: v for k, v in metrics.items()}
                            wandb_metrics["train/steps_per_sec"] = steps_per_sec
                            wandb_metrics["train/step_time_ms"] = step_time_ms
                            wandb_metrics["step"] = self.step
                            wandb.log(wandb_metrics)
                        except Exception:
                            pass

                if self.step % self.save_every == 0:
                    self._save_checkpoint()
                    logger.info(f"Saved checkpoint at step {self.step}")

            self._save_checkpoint()
            logger.info("Training complete!")
        finally:
            prefetch.close()

        if use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

    def _save_checkpoint(self):
        import orbax.checkpoint as ocp

        ckpt_dir = self.checkpoint_dir / f"step_{self.step:07d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpointer = ocp.PyTreeCheckpointer()

        # Save EMA model -> params/ (loaded by eval)
        _, state = nnx.split(self.student_model)
        ema_state_dict = dict(self._student_frozen_dict)
        for p, v in zip(self._student_trainable_paths, self.ema_values):
            ema_state_dict[p] = v
        ema_state = nnx.State.from_flat_path(ema_state_dict)
        _, ema_nnx_state = nnx.split(nnx.merge(self._student_graphdef, ema_state))
        checkpointer.save(str(ckpt_dir / "params"), ema_nnx_state.to_pure_dict())

        # Save training model -> params_training/
        _, train_nnx_state = nnx.split(self.student_model)
        checkpointer.save(str(ckpt_dir / "params_training"), train_nnx_state.to_pure_dict())

        # Save optimizer state
        if self.opt_state is not None:
            checkpointer.save(str(ckpt_dir / "opt_state"), self.opt_state)

        # Metadata
        train_state = {
            "step": self.step,
            "param_stats": self.param_stats,
            "ema_decay": self.ema_decay,
        }
        with open(ckpt_dir / "train_state.json", "w") as f:
            json.dump(train_state, f, indent=2, default=str)

        config_src = self.checkpoint_dir / "config.yaml"
        if config_src.exists():
            import shutil

            shutil.copy2(config_src, ckpt_dir / "config.yaml")

        logger.info(f"Saved checkpoint to {ckpt_dir}")

    def _load_checkpoint(self, path: str):
        import flax.traverse_util as traverse_util
        import orbax.checkpoint as ocp

        logger.info(f"Loading checkpoint from {path}")
        ckpt_path = Path(path)
        checkpointer = ocp.PyTreeCheckpointer()

        train_params_path = ckpt_path / "params_training"
        load_path = (
            str(train_params_path) if train_params_path.exists() else str(ckpt_path / "params")
        )
        params = checkpointer.restore(load_path)
        logger.info(f"Loaded model from {load_path}")

        _, state = nnx.split(self.student_model)
        state.replace_by_pure_dict(params)
        self.student_model = nnx.merge(self._student_graphdef, state)

        ema_params_path = ckpt_path / "params"
        if ema_params_path.exists():
            ema_params = checkpointer.restore(str(ema_params_path))
            flat_ema = traverse_util.flatten_dict(ema_params)
            _, state2 = nnx.split(self.student_model)
            flat_state = traverse_util.flatten_dict(state2.to_pure_dict())
            self.ema_values = []
            for p in self._student_trainable_paths:
                if p in flat_ema:
                    self.ema_values.append(jnp.array(flat_ema[p]))
                elif p in flat_state:
                    self.ema_values.append(_to_array(flat_state[p]).copy())
                else:
                    self.ema_values.append(jnp.zeros(1))
            logger.info(f"Loaded EMA values ({len(self.ema_values)} arrays)")

        train_state_path = ckpt_path / "train_state.json"
        if train_state_path.exists():
            with open(train_state_path) as f:
                ts = json.load(f)
            self.step = ts.get("step", 0)
            self.ema_decay = ts.get("ema_decay", self.ema_decay)
        else:
            try:
                self.step = int(ckpt_path.name.split("_")[1])
            except (IndexError, ValueError):
                self.step = 0

        # Load optimizer state
        opt_state_path = ckpt_path / "opt_state"
        if opt_state_path.exists():
            self.opt_state = checkpointer.restore(str(opt_state_path))
            logger.info("Loaded optimizer state")

        logger.info(f"Resumed from step {self.step}")
