#!/usr/bin/env python3
"""
LIBERO-Plus 评测脚本：在 LIBERO-Plus 鲁棒性基准上评测 pi05-libero checkpoints。

支持 NFE = 1, 2, 4, 10，统一使用 Pi05SMF 模型。

用法:
    # 快速冒烟测试（10 个任务，1-NFE）
    python scripts/eval_libero_plus.py --preset quick --nfe 1 --checkpoint checkpoints/finetuned/smf_base/step_5000

    # 单 suite 完整评测
    python scripts/eval_libero_plus.py --suite libero_spatial --nfe 1 --checkpoint checkpoints/finetuned/smf_base/step_5000

    # 多 NFE 评测
    python scripts/eval_libero_plus.py --suite libero_spatial --nfe 1 2 4 10 --max-tasks 100

    # 分批并行（第 0~99 个任务）
    python scripts/eval_libero_plus.py --suite libero_spatial --nfe 1 --task-offset 0 --max-tasks 100
"""

import argparse
import datetime
import json
import logging
import os
import pathlib
import re
import socket
import sys
import time

import numpy as np

# ── 设置路径 ──────────────────────────────────────────────
from eval_utils import (
    LIBERO_ENV_RESOLUTION,
    MAX_STEPS_MAP,
    PROJECT_ROOT,
    build_result_json,
    load_policy,
    run_single_task_episode,
    save_result_json,
    setup_paths,
)

# ── LIBERO-Plus 配置（必须在 import libero 之前完成） ──────
LIBERO_PLUS_DIR = PROJECT_ROOT / "data" / "libero-plus" / "LIBERO-plus"
LIBERO_PLUS_BENCHMARK = LIBERO_PLUS_DIR / "libero" / "libero"
LIBERO_PLUS_CONFIG_DIR = PROJECT_ROOT / "data" / "libero-plus" / ".libero_config"
LIBERO_PLUS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LIBERO_PLUS_CONFIG_FILE = LIBERO_PLUS_CONFIG_DIR / "config.yaml"

import yaml as _yaml
_libero_plus_config = {
    "benchmark_root": str(LIBERO_PLUS_BENCHMARK),
    "bddl_files": str(LIBERO_PLUS_BENCHMARK / "bddl_files"),
    "init_states": str(LIBERO_PLUS_BENCHMARK / "init_files"),
    "datasets": str(LIBERO_PLUS_DIR / "datasets"),
    "assets": str(LIBERO_PLUS_BENCHMARK / "assets"),
}
with open(LIBERO_PLUS_CONFIG_FILE, "w") as f:
    _yaml.dump(_libero_plus_config, f)
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_PLUS_CONFIG_DIR)

# 优先加载 libero-plus（替换原始 libero）
sys.path.insert(0, str(LIBERO_PLUS_DIR))
setup_paths()

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ── Preset 定义 ──────────────────────────────────────────
PRESETS = {
    "quick": {
        "suites": ["libero_spatial"],
        "max_tasks": 100,
        "description": "快速测试：libero_spatial 前 100 个任务",
    },
    "medium": {
        "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        "max_tasks": 100,
        "description": "中等评测：4 个 suite 各 100 个任务",
    },
    "full": {
        "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        "max_tasks": None,
        "description": "完整评测：4 个 suite 全部任务",
    },
    "full90": {
        "suites": ["libero_90"],
        "max_tasks": None,
        "description": "LIBERO-90 全部任务",
    },
}

# ── 扰动类型识别 ──────────────────────────────────────────
PERTURBATION_PATTERNS = {
    "objects_layout_add": r"_add_\d+",
    "camera_viewpoints_view": r"_view_\d+",
    "robot_init_states_table": r"_table_\d+",
    "robot_init_states_tb": r"_tb_\d+",
    "language_instructions_language": r"_language_",
    "light_conditions_light": r"_light_\d+",
    "sensor_noise_level": r"_level\d+",
    "background_textures_copy": r" copy",
}


def detect_perturbation(task_name: str) -> str:
    """从任务名识别扰动类型。"""
    for perturb_type, pattern in PERTURBATION_PATTERNS.items():
        if re.search(pattern, task_name):
            return perturb_type
    return "original"


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def run_eval_suite(nfe, suite_name, policy, seed, max_tasks, task_offset,
                   replan_steps, num_steps_wait, checkpoint, num_episodes=1):
    """评测单个 suite。"""
    np.random.seed(seed)
    start_time = time.time()

    # 加载 task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    num_tasks = task_suite.n_tasks
    logger.info(f"Task suite: {suite_name} ({num_tasks} tasks total)")

    # 应用 task 范围
    task_start = task_offset
    task_end = num_tasks if max_tasks is None else min(task_start + max_tasks, num_tasks)
    actual_tasks = task_end - task_start
    logger.info(f"Evaluating tasks [{task_start}, {task_end}) = {actual_tasks} tasks, {num_episodes} episodes/task")

    max_steps = MAX_STEPS_MAP.get(suite_name, 300)

    total_episodes, total_successes = 0, 0
    all_latencies = []
    task_results = {}
    episode_details = []
    perturbation_counts = {}

    for task_id in range(task_start, task_end):
        task = task_suite.get_task(task_id)
        task_description = task.language
        task_name = task.name
        perturbation = detect_perturbation(task_name)

        initial_states = task_suite.get_task_init_states(task_id)

        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        rel_id = task_id - task_start
        logger.info(f"[{rel_id+1}/{actual_tasks}] {task_description} ({perturbation})")

        task_successes = 0
        for ep in range(num_episodes):
            ep_idx = ep % len(initial_states)

            done, steps, ep_latencies = run_single_task_episode(
                env, initial_states[ep_idx], task_description, policy,
                max_steps, num_steps_wait, replan_steps,
            )

            all_latencies.extend(ep_latencies)
            if done:
                task_successes += 1
                total_successes += 1

            total_episodes += 1
            avg_lat = round(float(np.mean(ep_latencies)), 2) if ep_latencies else 0.0

            episode_details.append({
                "task_id": task_id,
                "task_name": task_name,
                "task_description": task_description,
                "perturbation": perturbation,
                "episode": ep,
                "success": done,
                "steps": steps,
                "avg_latency_ms": avg_lat,
            })

        task_rate = task_successes / num_episodes
        logger.info(f"  {task_successes}/{num_episodes} ({task_rate*100:.0f}%)")

        task_results[task_description] = {
            "task_name": task_name,
            "task_description": task_description,
            "success_rate": round(task_rate, 4),
            "successes": task_successes,
            "episodes": num_episodes,
            "perturbation": perturbation,
        }

        # 统计扰动类型
        if perturbation not in perturbation_counts:
            perturbation_counts[perturbation] = {"total": 0, "success": 0}
        perturbation_counts[perturbation]["total"] += num_episodes
        perturbation_counts[perturbation]["success"] += task_successes

        env.close()

    # 汇总
    end_time = time.time()
    total_rate = total_successes / total_episodes if total_episodes > 0 else 0
    avg_latency = np.mean(all_latencies) if all_latencies else 0
    p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0

    # 构建扰动统计
    perturbation_stats = {}
    for ptype, counts in sorted(perturbation_counts.items()):
        rate = counts["success"] / counts["total"] if counts["total"] > 0 else 0
        perturbation_stats[ptype] = {
            "total": counts["total"],
            "success": counts["success"],
            "rate": round(rate, 4),
        }

    logger.info("=" * 60)
    logger.info(f"RESULTS: {suite_name} | NFE={nfe} | {actual_tasks} tasks")
    logger.info("=" * 60)
    logger.info(f"Total: {total_successes}/{total_episodes} ({total_rate*100:.1f}%)")
    logger.info(f"Avg latency: {avg_latency:.1f}ms | P95: {p95_latency:.1f}ms")
    logger.info("-" * 60)
    logger.info("Per-perturbation breakdown:")
    for ptype, stats in perturbation_stats.items():
        logger.info(f"  {ptype}: {stats['success']}/{stats['total']} ({stats['rate']*100:.1f}%)")
    logger.info("=" * 60)

    # 保存结果
    config_dict = {
        "benchmark": "libero-plus",
        "task_suite": suite_name,
        "nfe": nfe,
        "max_tasks": max_tasks,
        "task_offset": task_offset,
        "replan_steps": replan_steps,
        "num_steps_wait": num_steps_wait,
        "seed": seed,
        "checkpoint_path": str(checkpoint),
        "action_horizon": 10,
    }
    result = build_result_json(
        config_dict, task_results, episode_details, all_latencies,
        total_successes, total_episodes, start_time, end_time,
    )
    result["perturbation_stats"] = perturbation_stats

    results_dir = str(PROJECT_ROOT / "results" / "libero_plus")
    filepath = save_result_json(result, results_dir, suite_name)

    return result, filepath


def main():
    parser = argparse.ArgumentParser(
        description="LIBERO-Plus evaluation for pi05-libero checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test
  python scripts/eval_libero_plus.py --preset quick --nfe 1

  # Full suite with multiple NFE
  python scripts/eval_libero_plus.py --suite libero_spatial --nfe 1 2 4 10

  # Batched evaluation
  python scripts/eval_libero_plus.py --suite libero_spatial --nfe 1 --task-offset 0 --max-tasks 500
        """,
    )
    parser.add_argument("--nfe", type=int, nargs="+", default=[1],
                        choices=[1, 2, 4, 10],
                        help="NFE steps to evaluate (default: 1)")
    parser.add_argument("--suite", type=str, default=None,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="Task suite (ignored if --preset is set)")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["quick", "medium", "full", "full90"],
                        help="Eval preset")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Max tasks per suite (default: all)")
    parser.add_argument("--task-offset", type=int, default=0,
                        help="Start from this task index (for batching)")
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "checkpoints" / "base" / "pi05_libero"),
                        help="Checkpoint directory")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Episodes per task (default: 1, libero-plus standard)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--results-dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "libero_plus"),
                        help="Directory to save results")
    args = parser.parse_args()

    # 确定 suites 和 max_tasks
    if args.preset:
        preset = PRESETS[args.preset]
        suites = preset["suites"]
        if args.max_tasks is None:
            args.max_tasks = preset["max_tasks"]
    elif args.suite:
        suites = [args.suite]
    else:
        parser.error("Must specify --suite or --preset")

    logger.info(f"LIBERO-Plus evaluation")
    logger.info(f"  Suites: {suites}")
    logger.info(f"  NFE: {args.nfe}")
    logger.info(f"  Max tasks: {args.max_tasks or 'all'}")
    logger.info(f"  Task offset: {args.task_offset}")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Replan steps: {args.replan_steps}")

    # 验证 libero-plus 已加载
    import libero.libero as libero_pkg
    libero_path = pathlib.Path(libero_pkg.__file__).parent
    if "libero-plus" not in str(libero_path):
        logger.warning(f"Loaded libero from {libero_path}, expected libero-plus!")
    else:
        logger.info(f"Using libero-plus: {libero_path}")

    # 为每个 NFE 分别加载模型并评测
    all_results = []
    for nfe in args.nfe:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading model for NFE={nfe}...")
        logger.info(f"{'='*60}\n")

        policy = load_policy(nfe, args.checkpoint)

        for suite_name in suites:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running eval: suite={suite_name}, nfe={nfe}")
            logger.info(f"{'='*60}\n")

            result, filepath = run_eval_suite(
                nfe=nfe,
                suite_name=suite_name,
                policy=policy,
                seed=args.seed,
                max_tasks=args.max_tasks,
                task_offset=args.task_offset,
                replan_steps=args.replan_steps,
                num_steps_wait=args.num_steps_wait,
                checkpoint=args.checkpoint,
                num_episodes=args.num_episodes,
            )
            all_results.append(result)

    # 多 suite/NFE 汇总
    if len(all_results) > 1:
        nfe_values = sorted(set(r["config"]["nfe"] for r in all_results))
        combined = {
            "benchmark": "libero-plus",
            "preset": args.preset,
            "nfe_values": nfe_values,
            "max_tasks": args.max_tasks,
            "task_offset": args.task_offset,
            "suites": {},
            "grand_total_episodes": sum(r["overall"]["total_episodes"] for r in all_results),
            "grand_total_successes": sum(r["overall"]["total_successes"] for r in all_results),
            "metadata": {
                "end_time": datetime.datetime.now().isoformat(),
                "hostname": socket.gethostname(),
                "checkpoint": args.checkpoint,
            },
        }
        combined["grand_total_rate"] = round(
            combined["grand_total_successes"] / max(combined["grand_total_episodes"], 1),
            4,
        )
        for r in all_results:
            key = f"{r['config']['task_suite']}_nfe{r['config']['nfe']}"
            combined["suites"][key] = r["overall"]

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nfe_str = "_".join(str(n) for n in nfe_values)
        combined_path = pathlib.Path(args.results_dir) / f"{ts}_combined_{nfe_str}nfe.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        logger.info(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
