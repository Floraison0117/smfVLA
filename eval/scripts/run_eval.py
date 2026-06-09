#!/usr/bin/env python3
"""
统一评估入口脚本。

支持 LIBERO 和 LIBERO-Plus 数据集，提供统一的命令行接口。

用法:
    # LIBERO 标准评估，preset 模式，1-NFE，SMF 模型
    python run_eval.py --dataset libero --mode preset --nfe 1 --model-type smf

    # LIBERO 标准评估，fullset 模式，1-NFE，SnapFlow 模型
    python run_eval.py --dataset libero --mode fullset --nfe 1 --model-type snapflow

    # LIBERO 标准评估，quick 模式，1-NFE，FreeFlow 模型
    python run_eval.py --dataset libero --mode quick --nfe 1 --model-type freeflow

    # LIBERO-Plus 鲁棒性评估，quick 模式，1-NFE，FreeFlow 模型
    python run_eval.py --dataset libero-plus --mode quick --nfe 1 --model-type freeflow

    # 测试不同 NFE 值
    python run_eval.py --dataset libero --mode quick --nfe 2 --model-type smf
    python run_eval.py --dataset libero --mode quick --nfe 4 --model-type smf
    python run_eval.py --dataset libero --mode quick --nfe 10 --model-type smf
"""

import argparse
import logging
import os
import subprocess
import sys
import pathlib

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# 获取脚本所在目录
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EVAL_ROOT.parent


def get_default_checkpoint(model_type: str, dataset: str) -> str:
    """
    根据模型类型和数据集返回默认 checkpoint 路径。

    Args:
        model_type: 'smf', 'snapflow', 或 'freeflow'
        dataset: 'libero' 或 'libero-plus'

    Returns:
        checkpoint 目录路径
    """
    if model_type == "smf":
        # SMF 使用 base checkpoint
        return str(PROJECT_ROOT / "checkpoints" / "smf_base" / "pi05_libero")
    elif model_type == "snapflow":
        # SnapFlow 也使用 base checkpoint (软链接到 smf_base)
        return str(PROJECT_ROOT / "checkpoints" / "smf_base" / "pi05_libero")
    elif model_type == "freeflow":
        # FreeFlow 使用 base checkpoint
        return str(PROJECT_ROOT / "checkpoints" / "freeflow" / "pi05_libero")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_libero_eval(args):
    """运行 LIBERO 标准评估。"""
    logger.info("=" * 60)
    logger.info("Running LIBERO Evaluation")
    logger.info("=" * 60)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "eval_direct.py"),
        "--nfe", str(args.nfe),
        "--model-type", args.model_type,
    ]

    # 添加 preset 或自定义参数
    if args.mode:
        cmd.extend(["--preset", args.mode])
    else:
        if args.task_suite:
            cmd.extend(["--task-suite", args.task_suite])
        if args.num_episodes:
            cmd.extend(["--num-episodes", str(args.num_episodes)])

    # 添加 checkpoint
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    else:
        default_ckpt = get_default_checkpoint(args.model_type, "libero")
        cmd.extend(["--checkpoint", default_ckpt])

    # 添加其他参数
    if args.seed:
        cmd.extend(["--seed", str(args.seed)])
    if args.replan_steps:
        cmd.extend(["--replan-steps", str(args.replan_steps)])

    logger.info(f"Command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def run_libero_plus_eval(args):
    """运行 LIBERO-Plus 鲁棒性评估。"""
    logger.info("=" * 60)
    logger.info("Running LIBERO-Plus Evaluation")
    logger.info("=" * 60)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "eval_libero_plus.py"),
        "--nfe", str(args.nfe),
    ]

    # 添加 preset 或自定义参数
    if args.mode:
        cmd.extend(["--preset", args.mode])
    else:
        if args.suite:
            cmd.extend(["--suite", args.suite])
        if args.max_tasks:
            cmd.extend(["--max-tasks", str(args.max_tasks)])

    # 添加 checkpoint
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    else:
        default_ckpt = get_default_checkpoint(args.model_type, "libero-plus")
        cmd.extend(["--checkpoint", default_ckpt])

    # 添加其他参数
    if args.seed:
        cmd.extend(["--seed", str(args.seed)])
    if args.replan_steps:
        cmd.extend(["--replan-steps", str(args.replan_steps)])

    logger.info(f"Command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="统一评估入口脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # LIBERO preset 模式
  python run_eval.py --dataset libero --mode preset --nfe 1 --model-type smf

  # LIBERO fullset 模式
  python run_eval.py --dataset libero --mode fullset --nfe 1 --model-type snapflow

  # LIBERO-Plus quick 模式 (FreeFlow)
  python run_eval.py --dataset libero-plus --mode quick --nfe 1 --model-type freeflow

  # 自定义 LIBERO 评估
  python run_eval.py --dataset libero --task-suite libero_spatial --num-episodes 10 --nfe 2

数据集说明:
  libero    - 标准 LIBERO 基准，支持多 episode 评估
  libero-plus - LIBERO-Plus 鲁棒性基准，包含扰动任务

模式说明 (LIBERO):
  quick     - 快速测试 (libero_spatial, 5 ep)
  preset    - 标准评估 (4 suites, 50 ep)
  fullset   - 完整评估 (5 suites, 50 ep)

模式说明 (LIBERO-Plus):
  quick     - 快速测试 (50 tasks)
  medium    - 中等评估 (100 tasks)
  full      - 完整评估 (4 suites)
  full90    - libero_90 suite

NFE 说明:
  1  - 1 步推理 (SMF/SnapFlow/FreeFlow)
  2  - 2 步推理
  4  - 4 步推理
  10 - 10 步推理 (Pi0 原始)
        """
    )

    # 数据集选择
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["libero", "libero-plus"],
                        help="选择评估数据集")

    # 模式选择
    parser.add_argument("--mode", type=str, default=None,
                        help="评估模式 (见上方说明)")

    # 自定义参数 (当不使用 preset 时)
    parser.add_argument("--task-suite", type=str, default=None,
                        help="LIBERO 任务集 (libero_spatial, libero_object, etc.)")
    parser.add_argument("--suite", type=str, default=None,
                        help="LIBERO-Plus 任务集")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="每个任务的 episode 数")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="LIBERO-Plus 最大任务数")

    # 通用参数
    parser.add_argument("--nfe", type=int, default=1, choices=[1, 2, 4, 10],
                        help="Number of Function Evaluations")
    parser.add_argument("--model-type", type=str, default="smf",
                        choices=["smf", "snapflow", "freeflow"],
                        help="模型类型")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint 目录路径")
    parser.add_argument("--seed", type=int, default=7,
                        help="随机种子")
    parser.add_argument("--replan-steps", type=int, default=5,
                        help="Replan 步数")

    args = parser.parse_args()

    # 根据 dataset 选择评估脚本
    if args.dataset == "libero":
        return run_libero_eval(args)
    elif args.dataset == "libero-plus":
        return run_libero_plus_eval(args)
    else:
        parser.error(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    sys.exit(main())
