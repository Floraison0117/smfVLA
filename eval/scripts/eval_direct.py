#!/usr/bin/env python3
"""
直接评测脚本：无需 WebSocket server，直接加载模型运行 LIBERO 评测。

用法:
    # 轻量级评测（libero_spatial, 5 ep/task）
    python scripts/eval_direct.py --preset quick --nfe 1

    # 正式评测（全部 5 suites, 50 ep/task）
    python scripts/eval_direct.py --preset full --nfe 1

    # 自定义评测（向后兼容）
    python scripts/eval_direct.py --nfe 10 --num-episodes 5
"""

import argparse
import datetime
import json
import logging
import pathlib
import socket
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

setup_paths()

import jax
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

PRESETS = {
    "quick": {
        "suites": ["libero_spatial"],
        "num_episodes": 5,  # 快速测试：5 episodes/task
    },
    "preset": {
        "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        "num_episodes": 5,  # 标准评估：5 episodes/task, 4 suites
    },
    "fullset": {
        "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        "num_episodes": 50,  # 完整评估：50 episodes/task, all 5 suites
    },
}

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def run_eval(args):
    np.random.seed(args.seed)
    start_time = time.time()

    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"JAX devices: {jax.devices()}")

    # 加载 policy
    use_smf = (args.model_type == "smf") and not args.no_smf
    use_snapflow = args.model_type == "snapflow"
    use_freeflow = args.model_type == "freeflow"
    policy = load_policy(args.nfe, args.checkpoint, use_smf=use_smf, use_snapflow=use_snapflow, use_freeflow=use_freeflow)

    # 加载 LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks
    logger.info(f"Task suite: {args.task_suite} ({num_tasks} tasks)")

    max_steps = MAX_STEPS_MAP.get(args.task_suite, 300)

    total_episodes, total_successes = 0, 0
    all_latencies = []
    task_results = {}
    episode_details = []

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)

        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(args.seed)

        task_episodes, task_successes = 0, 0

        max_available = len(initial_states)
        actual_episodes = min(args.num_episodes, max_available)
        if actual_episodes < args.num_episodes:
            logger.warning(f"Only {max_available} initial states available, capping from {args.num_episodes}")

        for episode_idx in range(actual_episodes):
            logger.info(f"[Task {task_id}] Ep {episode_idx+1}/{args.num_episodes}: {task_description}")

            done, steps, ep_latencies = run_single_task_episode(
                env, initial_states[episode_idx], task_description, policy,
                max_steps, args.num_steps_wait, args.replan_steps,
            )

            all_latencies.extend(ep_latencies)
            if done:
                task_successes += 1
                total_successes += 1

            task_episodes += 1
            total_episodes += 1
            status = "✓ SUCCESS" if done else "✗ FAILURE"
            logger.info(f"  {status} (steps={steps})")

            episode_details.append({
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "success": done,
                "steps": steps,
                "avg_latency_ms": round(float(np.mean(ep_latencies)), 2) if ep_latencies else 0.0,
            })

        task_rate = task_successes / task_episodes if task_episodes > 0 else 0
        task_results[task_description] = {
            "task_description": task_description,
            "successes": task_successes,
            "episodes": task_episodes,
            "rate": round(task_rate, 4),
        }
        logger.info(f"[Task {task_id}] {task_successes}/{task_episodes} ({task_rate*100:.1f}%)")
        env.close()

    # 汇总
    end_time = time.time()
    total_rate = total_successes / total_episodes if total_episodes > 0 else 0
    avg_latency = np.mean(all_latencies) if all_latencies else 0
    p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0

    logger.info("=" * 60)
    logger.info(f"RESULTS: {args.task_suite} | NFE={args.nfe} | {args.num_episodes} eps/task")
    logger.info("=" * 60)
    for desc, res in task_results.items():
        logger.info(f"  {desc}: {res['successes']}/{res['episodes']} ({res['rate']*100:.1f}%)")
    logger.info("-" * 60)
    logger.info(f"Total: {total_successes}/{total_episodes} ({total_rate*100:.1f}%)")
    logger.info(f"Avg latency: {avg_latency:.1f}ms | P95: {p95_latency:.1f}ms")
    logger.info("=" * 60)

    issues = []
    if total_rate < 0.9:
        issues.append(f"成功率 {total_rate*100:.1f}% < 90%")
    if avg_latency > 500:
        issues.append(f"平均延时 {avg_latency:.1f}ms > 500ms")

    if issues:
        logger.warning("ISSUES:")
        for issue in issues:
            logger.warning(f"  ⚠ {issue}")
    else:
        logger.info("✓ All checks passed!")

    # 保存结构化结果
    config_dict = {
        "task_suite": args.task_suite,
        "nfe": args.nfe,
        "num_episodes": args.num_episodes,
        "replan_steps": args.replan_steps,
        "num_steps_wait": args.num_steps_wait,
        "seed": args.seed,
        "checkpoint_path": str(args.checkpoint),
        "action_horizon": 10,
    }
    result = build_result_json(
        config_dict, task_results, episode_details, all_latencies,
        total_successes, total_episodes, start_time, end_time,
    )
    results_dir = getattr(args, "results_dir", str(PROJECT_ROOT / "results" / "eval"))
    filepath = save_result_json(result, results_dir, args.task_suite)

    return total_rate, avg_latency, result, filepath


def main():
    parser = argparse.ArgumentParser(description="LIBERO direct evaluation")
    parser.add_argument("--nfe", type=int, default=10, choices=[1, 2, 4, 10])
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "checkpoints" / "smf_base" / "pi05_libero"),
                        help="Checkpoint directory to evaluate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--preset", type=str, default=None, choices=["quick", "preset", "fullset"],
                        help="Eval preset: 'quick' (5ep), 'preset' (4 suites, 50ep), 'fullset' (5 suites, 50ep)")
    parser.add_argument("--results-dir", type=str,
                        default=str(PROJECT_ROOT / "eval" / "results"),
                        help="Directory to save structured JSON results")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording (useful for full eval to save disk)")
    parser.add_argument("--no-smf", action="store_true",
                        help="Use original Pi05 model (instead of SMF) for any NFE")
    parser.add_argument("--model-type", type=str, default="smf", choices=["smf", "snapflow", "freeflow", "dmf"],
                        help="Model type: 'smf', 'snapflow', 'freeflow', 'dmf', or original (use --no-smf)")
    args = parser.parse_args()

    # 应用预设覆盖
    if args.preset is not None:
        preset = PRESETS[args.preset]
        args.num_episodes = preset["num_episodes"]
        suites_to_run = preset["suites"]
    else:
        suites_to_run = [args.task_suite]

    all_results = []
    for suite_name in suites_to_run:
        args.task_suite = suite_name
        logger.info(f"\n{'='*60}")
        logger.info(f"Running eval: suite={suite_name}, nfe={args.nfe}, episodes={args.num_episodes}")
        logger.info(f"{'='*60}\n")
        total_rate, avg_latency, result, filepath = run_eval(args)
        all_results.append(result)

    # 多 suite 时保存汇总结果
    if len(suites_to_run) > 1:
        combined = {
            "preset": args.preset,
            "nfe": args.nfe,
            "num_episodes_per_task": args.num_episodes,
            "suites": {r["config"]["task_suite"]: r["overall"] for r in all_results},
            "grand_total_episodes": sum(r["overall"]["total_episodes"] for r in all_results),
            "grand_total_successes": sum(r["overall"]["total_successes"] for r in all_results),
            "grand_total_rate": round(
                sum(r["overall"]["total_successes"] for r in all_results)
                / max(sum(r["overall"]["total_episodes"] for r in all_results), 1),
                4,
            ),
            "metadata": {
                "end_time": datetime.datetime.now().isoformat(),
                "hostname": socket.gethostname(),
            },
        }
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = pathlib.Path(args.results_dir) / f"{ts}_{args.preset}_all_suites_{args.nfe}nfe.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        logger.info(f"Combined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
