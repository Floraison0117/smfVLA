#!/usr/bin/env python3
"""SMF 训练入口脚本。"""

import sys
import os

# 设置路径
project_root = os.environ.get(
    "PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
openpi_dir = os.path.join(project_root, "third_party", "openpi")
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, os.path.join(openpi_dir, "src"))

import yaml
import logging

# ── JAX 环境变量（必须在 import jax 之前设置）─────────────────────
# 强制使用 GPU
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["JAX_COMPILATION_CACHE_MAX_SIZE"] = "134217728"  # 128MB
# 关闭 XLA GEMM autotune，避免显存尖峰（默认 autotune_level=4；level=0 省约 80% 峰值显存，代价 +10% 时间）
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
# 提高显存占用比例（默认 0.75 太小，JVP 穿过 3B 模型在 bs=32 时会 OOM）
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import jax

jax.config.update("jax_platforms", "cuda")
jax.config.update("jax_compilation_cache_max_size", 128 * 1024 * 1024)

import flax.nnx as nnx
from pathlib import Path

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="configs/train/smf_base_libero.yaml", nargs="?")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 恢复训练（路径如 checkpoints/finetuned/smf_base/step_5000）",
    )
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)
    logger.info(f"配置: {config['method']}, steps={config['training_steps']}")

    # 加载模型
    from smf_vla.models.pi05_smf import Pi05SMF, Pi05SMFConfig

    smf_config = Pi05SMFConfig(
        pi05=True,
        action_horizon=config["action_horizon"],
        action_dim=config["action_dim"],
        discrete_state_input=False,
        flow_ratio=config["flow_ratio"],
    )

    # 加载参数
    from openpi.models.model import restore_params
    import jax.numpy as jnp

    ckpt_dir = Path(config["checkpoint"])

    # 创建模型
    model = smf_config.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    pure_state = state.to_pure_dict()

    import flax.traverse_util as traverse_util

    if args.resume:
        # Resume: skip base checkpoint load, trainer will restore from --resume path
        logger.info(f"将从 {args.resume} 恢复训练（跳过 base checkpoint 加载）")
    else:
        logger.info(f"加载参数: {ckpt_dir / 'params'}...")
        params = restore_params(ckpt_dir / "params", dtype=jnp.bfloat16)
        logger.info(f"参数数量: {sum(x.size for x in jax.tree.leaves(params)):,}")

        flat_params = traverse_util.flatten_dict(params)
        flat_state = traverse_util.flatten_dict(pure_state)

        loaded_count = 0
        skipped_new = 0
        missing = 0
        for key in flat_state:
            str_key = "/".join(key)
            if key in flat_params:
                flat_state[key] = flat_params.pop(key)
                loaded_count += 1
            elif "time_proj" in str_key:
                # SMF 新增参数 — 保留随机初始化（time_proj 初始化为 [I, 0]）
                skipped_new += 1
            else:
                missing += 1
                if missing <= 5:
                    logger.warning(f"未在 base checkpoint 中找到: {str_key}")
        if missing > 5:
            logger.warning(f"... 还有 {missing - 5} 个参数未找到")

        unused = len(flat_params)
        if unused > 0:
            logger.info(f"跳过 {unused} 个未使用的 checkpoint 键")

        logger.info(
            f"参数合并: {loaded_count} loaded, {skipped_new} SMF-new (kept init), "
            f"{missing} missing, {unused} unused"
        )

        pure_state = traverse_util.unflatten_dict(flat_state)
        state.replace_by_pure_dict(pure_state)
        model = nnx.merge(graphdef, state)
        logger.info("模型加载完成")

    # ── 加载 Teacher 模型（用于 Anchor / BPL）──────────────────
    teacher_model = None
    use_anchor = config.get("use_anchor", False)
    use_bpl = config.get("use_bpl", False)

    if use_anchor or use_bpl:
        teacher_ckpt = Path(config.get("teacher_path", config["checkpoint"]))
        logger.info(f"加载 Teacher 模型: {teacher_ckpt / 'params'}...")

        teacher_model = smf_config.create(jax.random.key(1))
        teacher_graphdef, teacher_state = nnx.split(teacher_model)
        teacher_pure = teacher_state.to_pure_dict()

        flat_teacher = traverse_util.flatten_dict(teacher_pure)
        flat_ckpt = traverse_util.flatten_dict(params)  # 复用已加载的 checkpoint params

        teacher_loaded = 0
        for key in flat_teacher:
            str_key = "/".join(key)
            if key in flat_ckpt:
                flat_teacher[key] = flat_ckpt[key]
                teacher_loaded += 1
            elif "time_proj" in str_key:
                logger.info(f"Teacher 跳过新增参数: {str_key}")
            else:
                logger.warning(f"Teacher 未找到: {str_key}")

        logger.info(f"Teacher 加载了 {teacher_loaded} 个参数")

        teacher_pure = traverse_util.unflatten_dict(flat_teacher)
        teacher_state.replace_by_pure_dict(teacher_pure)
        teacher_model = nnx.merge(teacher_graphdef, teacher_state)
        logger.info("Teacher 模型加载完成")

    # 创建训练器
    from smf_vla.training.jax_trainer import SMFTrainer

    trainer = SMFTrainer(
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
        wandb_project=config.get("wandb", {}).get("project", "smfvla"),
        wandb_run_name=config.get("wandb", {}).get("run_name"),
        wandb_config=config.get("wandb", {}),
        train_config=config,
        teacher_model=teacher_model,
    )

    # 创建数据加载器
    from smf_vla.training.data_loader import create_data_loader, create_fake_data_loader

    dataset_path = config.get("dataset_path", "")
    if Path(dataset_path).exists():
        logger.info(f"使用数据集: {dataset_path}")
        data_loader = create_data_loader(
            dataset_path=dataset_path,
            batch_size=config["batch_size"],
            action_horizon=config["action_horizon"],
            target_action_dim=config["action_dim"],
        )
    else:
        logger.warning(f"数据集不存在: {dataset_path}，使用假数据")
        data_loader = create_fake_data_loader(
            batch_size=config["batch_size"],
            action_horizon=config["action_horizon"],
            action_dim=config["action_dim"],
            num_batches=config["training_steps"],
        )

    # 开始训练
    rng = jax.random.key(42)
    model = trainer.train(data_loader, rng, resume_from=args.resume)
    logger.info("训练完成!")


if __name__ == "__main__":
    main()
