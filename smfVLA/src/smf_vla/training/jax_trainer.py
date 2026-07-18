"""
JAX 训练循环。

实现 SplitMeanFlow 及其变体的 JAX JIT 编译训练循环。
支持：梯度裁剪、学习率 warmup + decay、checkpoint 保存/加载、wandb 记录。
训练方法：SMF-Base, SMF-DecTE, SMF-Curr, SMF-DecTE-Curr, + Anchor, + BPL
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

from smf_vla.training import freeze_utils

logger = logging.getLogger(__name__)


class SMFTrainer:
    """
    SplitMeanFlow JAX 训练器（支持所有变体）。

    支持的训练方法（通过 train_config 配置）：
    - smf_base: concat time embedding, uniform time sampling
    - smf_decte: Decoupled Time Embedding
    - smf_curr: Curriculum Time Sampling
    - smf_decte_curr: DecTE + Curriculum
    - smf_decte_curr_anchor: + Anchor Loss
    - smf_decte_curr_bpl: + BPL Loss
    - smf_full: DecTE + Curriculum + Anchor + BPL
    """

    def __init__(
        self,
        model: nnx.Module,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 15000,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints/finetuned/smf_base",
        log_dir: str = "logs/train/smf_base",
        save_every: int = 3000,
        log_every: int = 100,
        wandb_project: str = "smfvla",
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
        train_config: dict[str, Any] | None = None,
        teacher_model: nnx.Module | None = None,
    ):
        self.model = model
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
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_config = wandb_config or {}
        self.train_config = train_config or {}

        # 训练方法配置
        self.method = self.train_config.get("method", "smf_base")
        self.time_conditioning = self.train_config.get("time_conditioning", "concat")
        self.use_curriculum = self.train_config.get("use_curriculum", False)
        self.use_anchor = self.train_config.get("use_anchor", False)
        self.use_bpl = self.train_config.get("use_bpl", False)
        self.flow_ratio = self.train_config.get("flow_ratio", 0.3)
        self.smf_loss_scale = self.train_config.get("smf_loss_scale", 1.0)

        # Curriculum 参数
        self.delta_min = self.train_config.get("delta_min", 0.05)
        self.delta_final = self.train_config.get("delta_final", 1.0)
        self.delta_floor = self.train_config.get("delta_floor", 1e-3)
        self.delta_sampling = self.train_config.get("delta_sampling", "uniform")

        # 动态 SMF scale（梯度匹配）
        self._is_dynamic_scale = isinstance(self.smf_loss_scale, str) and self.smf_loss_scale == "dynamic"
        self._smf_scale_ema = 1.0
        self._smf_scale_ema_alpha = 0.99
        self._smf_scale_min = 1.0
        self._smf_scale_max = 200.0

        # Anchor 参数
        self.anchor_warmup_steps = self.train_config.get("anchor_warmup_steps", 3000)
        self.anchor_cooldown_steps = self.train_config.get("anchor_cooldown_steps", 7500)
        self.anchor_alpha_max = self.train_config.get("anchor_alpha_max", 0.1)
        self.anchor_delta_max = self.train_config.get("anchor_delta_max", 0.3)

        # BPL 参数
        self.bpl_warmup_start = self.train_config.get("bpl_warmup_start", 4500)
        self.bpl_warmup_end = self.train_config.get("bpl_warmup_end", 10500)
        self.bpl_alpha_max = self.train_config.get("bpl_alpha_max", 0.05)

        # DecTE 参数
        self.encoder_depth = self.train_config.get("encoder_depth", 6)

        # 冻结策略（从 config 读取，fallback 到默认值）
        self.freeze_patterns = self.train_config.get("freeze", freeze_utils.FREEZE_PATTERNS)
        self.trainable_patterns = self.train_config.get("trainable", freeze_utils.TRAINABLE_PATTERNS)

        # 创建目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 构建冻结 mask（使用配置中的 patterns）
        graphdef, state = nnx.split(model)
        self.trainable_mask = freeze_utils.build_trainable_mask(
            state,
            freeze_patterns=self.freeze_patterns,
            trainable_patterns=self.trainable_patterns,
        )
        self.param_stats = freeze_utils.print_param_summary(state, self.trainable_mask)

        # 构建优化器
        self.optimizer = self._build_optimizer()

        # 训练状态
        self.step = 0
        self.train_log = []

    def _build_optimizer(self) -> optax.GradientTransformation:
        """构建 optax 优化器：AdamW + warmup + cosine decay + gradient clipping。"""
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

    def _get_anchor_alpha(self, step: int) -> float:
        """计算 anchor loss 的 alpha（线性 warmup + cooldown）。"""
        if step < self.anchor_warmup_steps:
            # 线性 warmup: 0 → alpha_max
            progress = step / max(self.anchor_warmup_steps, 1)
            return self.anchor_alpha_max * progress
        elif step < self.anchor_cooldown_steps:
            # 线性下降: alpha_max → 0
            progress = (step - self.anchor_warmup_steps) / max(
                self.anchor_cooldown_steps - self.anchor_warmup_steps, 1
            )
            return self.anchor_alpha_max * (1.0 - progress)
        else:
            return 0.0

    def _get_bpl_alpha(self, step: int) -> float:
        """计算 BPL 的 alpha（延迟 warmup）。"""
        if step < self.bpl_warmup_start:
            return 0.0
        elif step < self.bpl_warmup_end:
            progress = (step - self.bpl_warmup_start) / max(
                self.bpl_warmup_end - self.bpl_warmup_start, 1
            )
            return self.bpl_alpha_max * progress
        else:
            return self.bpl_alpha_max

    def _setup_jit_train_step(self):
        """
        预编译训练步骤：把模型拆成 trainable / frozen，
        用 jax.value_and_grad(fn, argnums=0) 只对 trainable 参数建梯度图。

        nnx.merge 放在 JIT 内部 → 只在首次编译时执行一次，
        后续调用是纯 JAX array 操作，无 NNX 对象图遍历。
        """
        logger.info("=" * 60)
        logger.info("JIT 编译训练步骤（首次调用较慢，后续 ~5s/step）")
        logger.info("=" * 60)

        # ── 拆分模型 ────────────────────────────────────────────
        graphdef, state = nnx.split(self.model)
        flat = state.flat_state()
        all_paths = sorted(flat.keys())

        self._trainable_paths = [p for p in all_paths if self.trainable_mask.get(p, False)]
        frozen_paths = [p for p in all_paths if not self.trainable_mask.get(p, False)]

        # frozen dict: closure 常量，不建梯度图
        self._frozen_dict = {p: flat[p] for p in frozen_paths}
        self._graphdef = graphdef

        def _size(v):
            arr = v.value if hasattr(v, 'value') else v
            return arr.size

        n_trainable = sum(_size(flat[p]) for p in self._trainable_paths)
        n_frozen = sum(_size(flat[p]) for p in frozen_paths)
        logger.info(f"  Trainable: {n_trainable:,} params ({n_trainable/1e6:.1f}M)")
        logger.info(f"  Frozen:    {n_frozen:,} params ({n_frozen/1e6:.1f}M)")

        # ── 拆分 Teacher 模型（如果存在）────────────────────────
        teacher_graphdef = None
        teacher_frozen_dict = None
        if self.teacher_model is not None:
            teacher_graphdef, teacher_state = nnx.split(self.teacher_model)
            teacher_flat = teacher_state.flat_state()
            teacher_frozen_dict = {p: teacher_flat[p] for p in teacher_flat}
            n_teacher = sum(_size(teacher_flat[p]) for p in teacher_flat)
            logger.info(f"  Teacher:   {n_teacher:,} params ({n_teacher/1e6:.1f}M) [frozen]")

        # ── 捕获到 closure 的变量（JIT 常量）────────────────────
        frozen_dict = self._frozen_dict
        trainable_paths = self._trainable_paths
        trainer = self  # 捕获 self 以访问配置（都是 Python 标量）

        # ── loss function: 只对 trainable_values 求导 ────────────
        def loss_fn(trainable_values, batch, rng, step, alpha_anchor, alpha_bpl, smf_scale_value):
            from openpi.models.model import Observation
            from openpi.models.pi0 import make_attn_mask
            from smf_vla.training.smf_loss import compute_full_smf_loss

            # 重组完整模型（JIT 首次调用时 trace，之后编译掉）
            full = dict(frozen_dict)
            for p, v in zip(trainable_paths, trainable_values):
                full[p] = v
            st = nnx.State.from_flat_path(full)
            model = nnx.merge(graphdef, st)

            observation = Observation.from_dict(batch["observation"])
            actions = batch["actions"]
            action_mean = batch["action_mean"]
            action_std = batch["action_std"]

            # ── model_fn: KV-cache split（prefix 只算一次，4 次前向共享）────────
            # 之前的 suffix-only ([None, suffix_tokens]) 写法丢弃了 prefix，导致 action
            # token 无法 attend 到 image/language —— 模型无法依赖观测，是严重正确性 bug。
            # 现采用与 model.sample_actions 完全一致的 KV-cache 模式（见 pi05_smf.py）：
            # prefix 前向一次得到 kv_cache（stop_gradient，frozen VLM 不进 grad），
            # model_fn 做 suffix-only 前向复用 kv_cache。数学上与完整前向等价，但 prefix
            # 只算一次（见 docs/training-debug.md §2/§7）。
            prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
            prefix_tokens = jax.lax.stop_gradient(prefix_tokens)
            prefix_mask = jax.lax.stop_gradient(prefix_mask)
            prefix_ar_mask = jax.lax.stop_gradient(prefix_ar_mask)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (_, _), kv_cache = model.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
            )
            batch_size = prefix_mask.shape[0]
            prefix_len = prefix_mask.shape[1]

            def model_fn(params, obs, noisy_actions, r, t):
                if trainer.time_conditioning == "decte":
                    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix_decte(
                        obs, noisy_actions, t=t, r=r, encoder_depth=trainer.encoder_depth
                    )
                else:
                    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix_smf(
                        obs, noisy_actions, t=t, r=r
                    )
                adarms_cond = [None, adarms_cond]

                # suffix-only 前向，复用 prefix 的 kv_cache
                suffix_len = suffix_mask.shape[1]
                suffix_causal_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                # suffix 可 attend 全部 prefix（全 1）+ causal 自身
                prefix_to_suffix_attn = jnp.ones((batch_size, suffix_len, prefix_len), dtype=jnp.bool_)
                full_attn_mask = jnp.concatenate([prefix_to_suffix_attn, suffix_causal_mask], axis=2)
                # suffix 全局位置接在 prefix 之后（与 sample_actions 一致，避免 train/infer 位置 mismatch）
                suffix_positions = (
                    jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=1) - 1
                )

                (_, suffix_out), _ = model.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=suffix_positions,
                    kv_cache=kv_cache,
                    adarms_cond=adarms_cond,
                )
                v = model.action_out_proj(suffix_out[:, -model.action_horizon:])
                return v

            # ── Teacher model（frozen，用于 Anchor / BPL）────────
            teacher_fn = None
            teacher_model_wrapper = None

            if teacher_frozen_dict is not None and teacher_graphdef is not None:
                # 重组 teacher 模型
                t_st = nnx.State.from_flat_path(teacher_frozen_dict)
                teacher = nnx.merge(teacher_graphdef, t_st)

                def teacher_fn(params, obs, noisy_actions, r, t):
                    """Teacher 前向传播（与 student 相同接口）。"""
                    if trainer.time_conditioning == "decte":
                        s_tokens, s_mask, s_ar_mask, adarms = teacher.embed_suffix_decte(
                            obs, noisy_actions, t=t, r=r, encoder_depth=trainer.encoder_depth
                        )
                    else:
                        s_tokens, s_mask, s_ar_mask, adarms = teacher.embed_suffix_smf(
                            obs, noisy_actions, t=t, r=r
                        )
                    adarms = [None, adarms]

                    s_attn_mask = make_attn_mask(s_mask, s_ar_mask)
                    s_positions = jnp.cumsum(s_mask, axis=1) - 1

                    (_, s_out), _ = teacher.PaliGemma.llm(
                        [None, s_tokens],
                        mask=s_attn_mask,
                        positions=s_positions,
                        adarms_cond=adarms,
                    )
                    v = teacher.action_out_proj(s_out[:, -teacher.action_horizon:])
                    return v

                # BPL teacher_model wrapper: 实现 extract_hidden_states 接口
                if trainer.use_bpl:
                    class _TeacherModelWrapper:
                        """Teacher wrapper for BPL loss: implements extract_hidden_states."""
                        def __init__(self, teacher_module):
                            self._teacher = teacher_module

                        def extract_hidden_states(self, obs, x, layer_indices):
                            return self._teacher.extract_hidden_states(obs, x, layer_indices)

                    teacher_model_wrapper = _TeacherModelWrapper(teacher)

            loss, metrics = compute_full_smf_loss(
                model_fn=model_fn,
                params=None,
                observation=observation,
                actions=actions,
                action_mean=action_mean,
                action_std=action_std,
                rng=rng,
                step=step,
                total_steps=trainer.total_steps,
                flow_ratio=trainer.flow_ratio,
                smf_loss_scale=trainer.smf_loss_scale,
                smf_scale_value=smf_scale_value,
                use_curriculum=trainer.use_curriculum,
                delta_min=trainer.delta_min,
                delta_final=trainer.delta_final,
                delta_floor=trainer.delta_floor,
                delta_sampling=trainer.delta_sampling,
                use_anchor=trainer.use_anchor,
                alpha_anchor=alpha_anchor,
                anchor_delta_max=trainer.anchor_delta_max,
                use_bpl=trainer.use_bpl,
                alpha_bpl=alpha_bpl,
                teacher_fn=teacher_fn,
                teacher_model=teacher_model_wrapper,
            )

            return loss, metrics

        # ── JIT 编译: argnums=0 → 只对 trainable_values 求导 ───
        self._grad_fn = jax.jit(
            jax.value_and_grad(loss_fn, argnums=0, has_aux=True),
            static_argnums=(),  # 所有参数都是动态的
        )

        logger.info("JIT 编译完成")
        logger.info("=" * 60)

        # ── 动态 scale: 编译梯度范数计算函数 ───────────────────
        if self._is_dynamic_scale:
            logger.info("编译梯度匹配函数（用于动态 SMF scale）...")

            def _grad_norm_fn(trainable_values, batch, rng, step, alpha_anchor, alpha_bpl):
                """计算 loss_fm 和 loss_smf 各自的梯度范数。"""
                # scale=0 → grad 只含 loss_fm
                _, grads_fm = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
                    trainable_values, batch, rng, step, alpha_anchor, alpha_bpl, 0.0,
                )
                # scale=1 → grad = grad_smf + grad_fm
                _, grads_both = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(
                    trainable_values, batch, rng, step, alpha_anchor, alpha_bpl, 1.0,
                )
                # grad_smf = grad_both - grad_fm
                def _to_val(x):
                    return x.value if hasattr(x, 'value') else x

                def _norm(grads):
                    norms = jnp.array([jnp.linalg.norm(_to_val(g)) for g in grads if g is not None])
                    return jnp.linalg.norm(norms)

                g_fm = _norm(grads_fm)
                g_smf = _norm([_to_val(b) - _to_val(f) if b is not None else None for b, f in zip(grads_both, grads_fm)])
                return g_fm, g_smf

            self._grad_norm_fn = jax.jit(_grad_norm_fn)
            logger.info("梯度匹配函数编译完成")

    def _update_smf_scale(self, model: nnx.Module, jax_batch: dict, rng: jax.Array, step: int, alpha_anchor: float, alpha_bpl: float) -> float:
        """
        计算梯度范数比并更新 smf_scale_ema。

        通过两次反向传播分别计算 loss_fm 和 loss_smf 的梯度范数，
        用 EMA 平滑更新 scale 值。

        Args:
            model: 当前模型
            jax_batch: 已准备好的 JAX batch（包含 observation, actions, action_mean, action_std）
            rng: 随机数 key
            step: 当前步数
            alpha_anchor: anchor loss alpha
            alpha_bpl: BPL loss alpha

        Returns:
            更新后的 smf_scale_ema
        """
        if not self._is_dynamic_scale:
            return self._smf_scale_ema

        _, state = nnx.split(model)
        flat = state.flat_state()
        trainable_values = [flat[p] for p in self._trainable_paths]

        # 只保留 JIT 需要的 array 值（与 train_step 一致）
        filtered_batch = {k: v for k, v in jax_batch.items() if k in ("observation", "actions", "action_mean", "action_std")}

        g_fm, g_smf = self._grad_norm_fn(
            trainable_values, filtered_batch, rng, step, alpha_anchor, alpha_bpl,
        )

        g_fm = float(g_fm)
        g_smf = float(g_smf)

        # 计算目标 scale
        target_ratio = g_fm / (g_smf + 1e-8)

        # EMA 更新
        self._smf_scale_ema = (
            self._smf_scale_ema_alpha * self._smf_scale_ema
            + (1 - self._smf_scale_ema_alpha) * target_ratio
        )

        # 限制范围
        self._smf_scale_ema = max(self._smf_scale_min, min(self._smf_scale_max, self._smf_scale_ema))

        logger.debug(f"梯度匹配: g_fm={g_fm:.4f}, g_smf={g_smf:.6f}, ratio={target_ratio:.2f}, ema={self._smf_scale_ema:.2f}")

        return self._smf_scale_ema

    def train_step(
        self,
        model: nnx.Module,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: dict[str, Any],
        rng: jax.Array,
        step: int,
        alpha_anchor: float,
        alpha_bpl: float,
        smf_scale_value: float = 1.0,
    ) -> tuple[nnx.Module, optax.OptState, dict[str, float]]:
        """
        单步训练。

        使用预编译的 _grad_fn（jax.value_and_grad, argnums=0），
        只对 ~300M trainable 参数建梯度图，frozen 参数是 JIT 常量。

        Returns:
            更新后的 model, opt_state, metrics
        """
        # ── 提取 trainable 参数 ────────────────────────────────
        _, state = nnx.split(model)
        flat = state.flat_state()
        trainable_values = [flat[p] for p in self._trainable_paths]

        # ── 过滤 batch: 只保留 JIT 需要的 array 值 ──────────
        jax_batch = {
            "observation": batch["observation"],
            "actions": batch["actions"],
            "action_mean": batch["action_mean"],
            "action_std": batch["action_std"],
        }

        # ── 前向 + 只对 trainable 求梯度 ─────────────────────
        (loss, metrics), grads = self._grad_fn(
            trainable_values, jax_batch, rng, step, alpha_anchor, alpha_bpl, smf_scale_value,
        )

        # ── 参数更新 ─────────────────────────────────────────
        updates, new_opt_state = optimizer.update(
            grads, opt_state, params=trainable_values
        )
        new_trainable_values = optax.apply_updates(trainable_values, updates)

        # ── 重组完整模型 ─────────────────────────────────────
        full = dict(self._frozen_dict)
        for p, v in zip(self._trainable_paths, new_trainable_values):
            full[p] = v
        new_state = nnx.State.from_flat_path(full)
        model = nnx.merge(self._graphdef, new_state)

        # ── 梯度/参数范数 ────────────────────────────────────
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
        """初始化 wandb。"""
        try:
            import wandb

            wandb.login()
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "method": self.method,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "warmup_steps": self.warmup_steps,
                    "total_steps": self.total_steps,
                    "gradient_clip_norm": self.gradient_clip_norm,
                    "flow_ratio": self.flow_ratio,
                    "smf_loss_scale": self.smf_loss_scale,
                    "time_conditioning": self.time_conditioning,
                    "use_curriculum": self.use_curriculum,
                    "delta_min": self.delta_min,
                    "delta_final": self.delta_final,
                    "delta_floor": self.delta_floor,
                    "delta_sampling": self.delta_sampling,
                    "is_dynamic_scale": self._is_dynamic_scale,
                    "use_anchor": self.use_anchor,
                    "use_bpl": self.use_bpl,
                    "encoder_depth": self.encoder_depth,
                    "freeze_patterns": self.freeze_patterns,
                    "trainable_patterns": self.trainable_patterns,
                    "param_stats": self.param_stats,
                    **self.wandb_config,
                },
            )
            logger.info(f"WandB 初始化完成: project={self.wandb_project}, run={wandb.run.name}")
            return True
        except Exception as e:
            logger.warning(f"WandB 初始化失败，跳过 wandb 记录: {e}")
            return False

    def train(self, data_loader, rng: jax.Array, resume_from: str | None = None):
        """
        主训练循环。

        Args:
            data_loader: 数据加载器（迭代器）
            rng: JAX 随机数 key
            resume_from: 恢复训练的 checkpoint 路径（可选）
        """
        logger.info("=" * 60)
        logger.info(f"开始训练: {self.method}")
        logger.info("=" * 60)
        logger.info(f"总步数: {self.total_steps}")
        logger.info(f"学习率: {self.learning_rate}")
        logger.info(f"梯度裁剪: {self.gradient_clip_norm}")
        logger.info(f"时间条件: {self.time_conditioning}")
        logger.info(f"Curriculum: {self.use_curriculum}")
        logger.info(f"SMF Loss Scale: {self.smf_loss_scale}")
        logger.info(f"Anchor: {self.use_anchor}")
        logger.info(f"BPL: {self.use_bpl}")
        logger.info(f"Checkpoint 目录: {self.checkpoint_dir}")
        logger.info("-" * 60)

        # 初始化 wandb
        use_wandb = self._init_wandb()

        # 初始化优化器状态
        graphdef, state = nnx.split(self.model)
        trainable_params = []
        flat = state.flat_state()
        for path in sorted(flat.keys()):
            if self.trainable_mask.get(path, False):
                trainable_params.append(flat[path])

        opt_state = self.optimizer.init(trainable_params)

        model = self.model

        # 恢复 checkpoint（如果指定）
        if resume_from is not None:
            logger.info(f"从 checkpoint 恢复: {resume_from}")
            model, restored_opt_state, resume_step = self.load_checkpoint(
                model, resume_from, opt_state
            )
            if restored_opt_state is not None:
                opt_state = restored_opt_state
            self.step = resume_step
            logger.info(f"恢复到 step {resume_step}，继续训练...")

        # ── JIT 编译训练步骤（关键优化）──────────────────────
        # 把模型拆成 trainable/frozen，用 jax.value_and_grad(argnums=0)
        # 只对 trainable 参数建梯度图。nnx.merge 在 JIT 内部只执行一次。
        self._setup_jit_train_step()

        # 用 _setup_jit_train_step 提取的 trainable_paths 重建 opt_state
        flat = state.flat_state()
        trainable_params = [flat[p] for p in self._trainable_paths]
        opt_state = self.optimizer.init(trainable_params)

        start_time = time.time()

        for step in range(self.step, self.total_steps):
            self.step = step

            # 获取 batch
            try:
                batch = next(data_iter)
            except (StopIteration, NameError):
                data_iter = iter(data_loader)
                batch = next(data_iter)

            # 准备 batch
            jax_batch = _prepare_batch(batch)

            # 计算当前 step 的 alpha 值
            alpha_anchor = self._get_anchor_alpha(step) if self.use_anchor else 0.0
            alpha_bpl = self._get_bpl_alpha(step) if self.use_bpl else 0.0

            # 动态 SMF scale: 每 log_every 步更新一次
            rng, scale_rng = jax.random.split(rng)
            smf_scale_value = self._update_smf_scale(model, jax_batch, scale_rng, step, alpha_anchor, alpha_bpl)

            # 训练 step
            rng, step_rng = jax.random.split(rng)
            model, opt_state, metrics = self.train_step(
                model, self.optimizer, opt_state, jax_batch, step_rng,
                step, alpha_anchor, alpha_bpl, smf_scale_value,
            )

            # 记录日志
            metrics["step"] = step
            try:
                # optax.chain: [0]=clip, [1]=adamw
                lr_fn = self.optimizer[1].learning_rate
                metrics["lr"] = float(lr_fn(step))
            except (AttributeError, IndexError, TypeError):
                metrics["lr"] = self.learning_rate
            metrics["wall_time"] = time.time() - start_time
            metrics["alpha_anchor"] = alpha_anchor
            metrics["alpha_bpl"] = alpha_bpl
            metrics["smf_scale_ema"] = self._smf_scale_ema
            self.train_log.append(metrics)

            # wandb 记录
            if use_wandb and step % self.log_every == 0:
                import wandb

                wandb_metrics = {
                    "train/loss_total": metrics["loss_total"],
                    "train/loss_smf": metrics["loss_smf"],
                    "train/loss_smf_scaled": metrics.get("loss_smf_scaled", metrics["loss_smf"]),
                    "train/loss_fm": metrics["loss_fm"],
                    "train/flow_ratio": metrics.get("flow_ratio_actual", 0),
                    "train/smf_scale_ema": self._smf_scale_ema,
                    "train/smf_scale_applied": metrics.get("smf_scale_applied", 1.0),
                    "train/lr": metrics["lr"],
                    "train/wall_time": metrics["wall_time"],
                    "train/grad_norm": metrics.get("grad_norm", 0),
                    "train/param_norm": metrics.get("param_norm", 0),
                    "train/delta_mean": metrics.get("delta_mean", 0),
                    "train/delta_max": metrics.get("delta_max", 1),
                    "train/t_mean": metrics.get("t_mean", 0),
                    "train/r_mean": metrics.get("r_mean", 0),
                }
                # Anchor 指标
                if self.use_anchor:
                    wandb_metrics["anchor/loss"] = metrics.get("loss_anchor", 0)
                    wandb_metrics["anchor/alpha"] = alpha_anchor
                    wandb_metrics["anchor/active_ratio"] = metrics.get("anchor_active_ratio", 0)
                # BPL 指标
                if self.use_bpl:
                    wandb_metrics["bpl/loss"] = metrics.get("loss_bpl", 0)
                    wandb_metrics["bpl/alpha"] = alpha_bpl
                # GPU 内存
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
                    f"smf={metrics['loss_smf']:.4f} | "
                    f"fm={metrics['loss_fm']:.4f} | "
                    f"scale={self._smf_scale_ema:.1f} | "
                    f"delta={metrics.get('delta_mean', 0):.3f} | "
                    f"grad={metrics.get('grad_norm', 0):.4f} | "
                    f"time={elapsed:.1f}s"
                )
                if self.use_anchor:
                    msg += f" | anchor={metrics.get('loss_anchor', 0):.4f}(α={alpha_anchor:.3f})"
                if self.use_bpl:
                    msg += f" | bpl={metrics.get('loss_bpl', 0):.4f}(α={alpha_bpl:.4f})"
                logger.info(msg)

            # 保存 checkpoint
            if (step + 1) % self.save_every == 0:
                self.save_checkpoint(model, opt_state, step + 1, metrics)

        # 保存最终 checkpoint
        self.save_checkpoint(model, opt_state, self.total_steps, metrics)

        # 保存训练日志
        self._save_train_log()

        total_time = time.time() - start_time

        # 保存训练总结
        summary_dir = self._save_training_summary(total_time, metrics)

        # 完成 wandb
        if use_wandb:
            import wandb
            wandb.finish()

        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info(f"总耗时: {total_time:.1f}s")
        logger.info(f"训练总结: {summary_dir}")
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
        """保存 checkpoint（模型参数 + 优化器状态 + 训练元数据）。"""
        import orbax.checkpoint as ocp

        ckpt_dir = self.checkpoint_dir / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型参数
        _, state = nnx.split(model)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(str(ckpt_dir / "params"), state.to_pure_dict())

        # 保存优化器状态（用于训练恢复）
        checkpointer.save(str(ckpt_dir / "opt_state"), opt_state)

        # 保存训练元数据
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
        """
        加载 checkpoint（模型参数 + 优化器状态 + 训练步数）。

        Returns:
            (model, opt_state, step) — opt_state 为 None 如果 checkpoint 中没有保存
        """
        import orbax.checkpoint as ocp

        ckpt_path = Path(ckpt_path)
        checkpointer = ocp.PyTreeCheckpointer()

        # 加载模型参数
        params = checkpointer.restore(str(ckpt_path / "params"))
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(params)
        model = nnx.merge(graphdef, state)

        # 加载优化器状态（如果存在）
        opt_state_path = ckpt_path / "opt_state"
        if opt_state_path.exists() and opt_state is not None:
            try:
                restored_opt = checkpointer.restore(str(opt_state_path))
                # 将恢复的 state 填入当前 opt_state 的 pytree 结构
                opt_state = jax.tree.map(
                    lambda old, new: new if new is not None else old,
                    opt_state,
                    restored_opt,
                )
                logger.info(f"Loaded optimizer state from {opt_state_path}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Optimizer state structure mismatch (likely due to changed total_steps), using fresh optimizer state: {e}")
        elif opt_state_path.exists():
            logger.warning(f"Opt state found at {opt_state_path} but no opt_state provided, skipping")
        else:
            logger.warning(f"No opt_state found at {opt_state_path}, using fresh optimizer state")

        # 加载训练步数
        step = 0
        train_state_path = ckpt_path / "train_state.json"
        if train_state_path.exists():
            with open(train_state_path) as f:
                train_state = json.load(f)
            step = train_state["step"]

        logger.info(f"Loaded checkpoint from {ckpt_path}, step={step}")
        return model, opt_state, step

    def _save_train_log(self):
        """保存训练日志为 JSONL 文件。"""
        log_path = self.log_dir / "train_log.jsonl"
        with open(log_path, "w") as f:
            for entry in self.train_log:
                f.write(json.dumps(entry) + "\n")
        logger.info(f"Saved training log to {log_path}")

    def _save_training_summary(self, total_time: float, final_metrics: dict[str, float]) -> Path:
        """保存训练总结到以日期+内容命名的目录。"""
        now = datetime.now()
        method = self.train_config.get("method", "smf")
        desc_short = self.train_config.get("description", "")[:30].replace(" ", "_")
        dir_name = f"{now.strftime('%Y%m%d_%H%M%S')}_{method}"
        if desc_short:
            dir_name += f"_{desc_short}"

        summary_dir = self.log_dir / dir_name
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "method": method,
            "description": self.train_config.get("description", ""),
            "training": {
                "total_steps": self.total_steps,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_steps": self.warmup_steps,
                "gradient_clip_norm": self.gradient_clip_norm,
                "batch_size": self.train_config.get("batch_size"),
                "precision": self.train_config.get("precision"),
                "flow_ratio": self.flow_ratio,
                "smf_loss_scale": self.smf_loss_scale,
                "time_conditioning": self.time_conditioning,
                "use_curriculum": self.use_curriculum,
                "delta_min": self.delta_min,
                "delta_final": self.delta_final,
                "delta_floor": self.delta_floor,
                "use_anchor": self.use_anchor,
                "use_bpl": self.use_bpl,
                "encoder_depth": self.encoder_depth,
                "freeze_patterns": self.freeze_patterns,
                "trainable_patterns": self.trainable_patterns,
            },
            "results": {
                "final_loss_total": final_metrics.get("loss_total"),
                "final_loss_smf": final_metrics.get("loss_smf"),
                "final_loss_smf_scaled": final_metrics.get("loss_smf_scaled", final_metrics.get("loss_smf")),
                "final_loss_fm": final_metrics.get("loss_fm"),
                "final_loss_anchor": final_metrics.get("loss_anchor", 0),
                "final_loss_bpl": final_metrics.get("loss_bpl", 0),
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

        logger.info(f"训练总结已保存到: {summary_dir}")
        return summary_dir


def _prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """将 numpy batch 转为 JAX array（支持嵌套 dict）。"""
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
