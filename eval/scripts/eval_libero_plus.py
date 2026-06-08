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
LIBERO_PLUS_DIR = PROJECT_ROOT / "datasets" / "libero-plus" / "LIBERO-plus"
LIBERO_PLUS_BENCHMARK = LIBERO_PLUS_DIR / "libero" / "libero"
LIBERO_PLUS_CONFIG_DIR = PROJECT_ROOT / "datasets" / "libero-plus" / ".libero_config"
LIBERO_PLUS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LIBERO_PLUS_CONFIG_FILE = LIBERO_PLUS_CONFIG_DIR / "config.yaml"

import yaml as _yaml
_libero_plus_config = {
    "benchmark_root": str(LIBERO_PLUS_BENCHMARK),
    "bddl_files": str(LIBERO_PLUS_BENCHMARK / "bddl_files"),
    "init_states": str(LIBERO_PLUS_BENCHMARK / "init_files"),
    "datasets": str(LIBERO_PLUS_DIR.parent / "datasets"),
    "assets": str(LIBERO_PLUS_BENCHMARK / "assets"),
}
with open(LIBERO_PLUS_CONFIG_FILE, "w") as f:
    _yaml.dump(_libero_plus_config, f)
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_PLUS_CONFIG_DIR)

# 优先加载 libero-plus（替换原始 libero）
sys.path.insert(0, str(LIBERO_PLUS_DIR.parent))
setup_paths()

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ── Preset 定义 ──────────────────────────────────────────
PRESETS = {
    "quick": {
        "suites": ["libero_spatial"],
        "max_tasks": 50,
        "description": "快速测试：libero_spatial 前 50 个任务",
    },
    "medium": {
        "suites": ["libero_spatial"],
        "max_tasks": 100,
        "description": "中等评测：libero_spatial 100 个任务",
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

# ── Task name parsing for LIBERO-Plus ────────────────────
# Based on SnapFlow's correct implementation
def parse_task_name(task_name: str) -> dict:
    """
    Parse LIBERO-Plus task name to extract perturbation info.

    Examples:
        "libero_spatial_add_0_view_0" -> {"type": "add", "add_id": 0, "view_id": 0}
        "libero_object_view_1" -> {"type": "view", "view_id": 1}
    """
    parts = task_name.split("_")
    result = {"base_suite": parts[0]}

    for i, part in enumerate(parts):
        if part == "add" and i + 1 < len(parts):
            result["add_id"] = int(parts[i + 1])
        elif part == "view" and i + 1 < len(parts):
            result["view_id"] = int(parts[i + 1])

    return result


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def run_eval_suite(nfe, suite_name, policy, seed, max_tasks, task_offset,
                   replan_steps, num_steps_wait, checkpoint):
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
    logger.info(f"Evaluating tasks [{task_start}, {task_end}) = {actual_tasks} tasks")

    max_steps = MAX_STEPS_MAP.get(suite_name, 300)

    total_episodes = 0
    total_successes = 0
    all_latencies = []
    task_results = {}
    episode_details = []

    for task_id in range(task_start, task_end):
        task = task_suite.get_task(task_id)
        task_description = task.language
        # ⭐ FIX: Use bddl_file instead of task.name (matches SnapFlow)
        task_name = task.bddl_file.replace(".yaml", "")
        perturb_info = parse_task_name(task_name)

        initial_states = task_suite.get_task_init_states(task_id)
        bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        rel_id = task_id - task_start
        logger.info(f"[{rel_id+1}/{actual_tasks}] {task_description[:80]}...")

        # LIBERO-Plus: 1 episode per task (different perturbations are different tasks)
        episode_idx = 0

        try:
            done, steps, ep_latencies = run_single_task_episode(
                env=env,
                initial_state=initial_states[episode_idx],
                task_description=task_description,
                policy=policy,
                max_steps=max_steps,
                num_steps_wait=num_steps_wait,
                replan_steps=replan_steps,
            )

            all_latencies.extend(ep_latencies)
            if done:
                total_successes += 1

            total_episodes += 1
            status = "✓ SUCCESS" if done else "✗ FAILURE"
            logger.info(f"  {status} (steps={steps})")

            episode_details.append({
                "task_id": task_id,
                "task_name": task_name,
                "task_description": task_description,
                "perturb_info": perturb_info,
                "success": done,
                "steps": steps,
                "avg_latency_ms": round(float(np.mean(ep_latencies)), 2) if ep_latencies else 0.0,
            })

            # ⭐ FIX: Use task_name as key (not task_description)
            task_results[task_name] = {
                "task_id": task_id,
                "task_name": task_name,
                "task_description": task_description,
                "perturb_info": perturb_info,
                "success": int(done),
            }

        except Exception as e:
            logger.warning(f"Task {task_id} failed with error: {e}")
            episode_details.append({
                "task_id": task_id,
                "task_name": task_name,
                "task_description": task_description,
                "perturb_info": perturb_info,
                "success": False,
                "steps": 0,
                "error": str(e),
            })

        env.close()

    # 汇总
    end_time = time.time()
    total_rate = total_successes / total_episodes if total_episodes > 0 else 0
    avg_latency = np.mean(all_latencies) if all_latencies else 0
    p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0

    logger.info("=" * 60)
    logger.info(f"RESULTS: {suite_name} | NFE={nfe} | {actual_tasks} tasks")
    logger.info("=" * 60)
    logger.info(f"Total: {total_successes}/{total_episodes} ({total_rate*100:.1f}%)")
    logger.info(f"Avg latency: {avg_latency:.1f}ms | P95: {p95_latency:.1f}ms")
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
                        default=str(PROJECT_ROOT / "checkpoints" / "smf_base" / "pi05_libero"),
                        help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--results-dir", type=str,
                        default=str(PROJECT_ROOT / "eval" / "results" / "smf" / "libero_plus"),
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
