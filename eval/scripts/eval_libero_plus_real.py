#!/usr/bin/env python3
"""
SnapFlow LIBERO-Plus evaluation script with true perturbation tasks.

Loads tasks from task_classification.json (2402+ tasks with perturbations).

Usage:
    python scripts/eval_libero_plus_real.py --preset quick --nfe 1 --checkpoint checkpoints/finetuned/snapflow/step_30000
"""
import argparse
import json
import logging
import os
import pathlib
import socket
import sys
import time
import yaml

import numpy as np

# Set up paths
# eval/scripts/ 目录结构：project_root 应该是 autodl-tmp/
project_root = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
openpi_dir = os.path.join(project_root, "third_party", "openpi")
snapflow_dir = os.path.join(project_root, "snapflow")
sys.path.insert(0, os.path.join(snapflow_dir, "src"))
sys.path.insert(0, os.path.join(openpi_dir, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # eval/

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(project_root)
LIBERO_PLUS_DIR = PROJECT_ROOT / "datasets" / "libero-plus" / "LIBERO-plus"
LIBERO_PLUS_BENCHMARK = LIBERO_PLUS_DIR / "libero" / "libero"
LIBERO_PLUS_CONFIG_DIR = PROJECT_ROOT / "datasets" / "libero-plus" / ".libero_config"
LIBERO_PLUS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LIBERO_PLUS_CONFIG_FILE = LIBERO_PLUS_CONFIG_DIR / "config.yaml"

_libero_plus_config = {
    "benchmark_root": str(LIBERO_PLUS_BENCHMARK),
    "bddl_files": str(LIBERO_PLUS_BENCHMARK / "bddl_files"),
    "init_states": str(LIBERO_PLUS_BENCHMARK / "init_files"),
    "datasets": str(LIBERO_PLUS_DIR.parent / "datasets"),
    "assets": str(LIBERO_PLUS_BENCHMARK / "assets"),
}
with open(LIBERO_PLUS_CONFIG_FILE, "w") as f:
    yaml.dump(_libero_plus_config, f)
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_PLUS_CONFIG_DIR)

# Load standard LIBERO for environment creation
sys.path.insert(0, str(LIBERO_PLUS_DIR.parent))

from eval_utils import (
    MAX_STEPS_MAP,
    load_policy,
    run_single_task_episode,
    build_result_json,
    save_result_json,
)

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# Load task classification for true LIBERO-Plus tasks
TASK_CLASSIFICATION_FILE = LIBERO_PLUS_BENCHMARK / "benchmark" / "task_classification.json"
with open(TASK_CLASSIFICATION_FILE, "r") as f:
    TASK_CLASSIFICATION = json.load(f)

PRESETS = {
    "quick": {
        "suites": ["libero_spatial"],
        "max_tasks": 50,
        "description": "Quick: 50 perturbation tasks"
    },
    "medium": {
        "suites": ["libero_spatial"],
        "max_tasks": 100,
        "description": "Medium: 100 perturbation tasks"
    },
    "full": {
        "suites": ["libero_spatial", "libero_object", "libero_goal"],
        "max_tasks": None,
        "description": "Full: all perturbation tasks"
    },
}


def get_base_task_name(task_name: str) -> str:
    """Extract base task name from perturbation task name."""
    # Remove perturbation suffixes like _table_1, _tb_12, etc.
    parts = task_name.split("_")
    base_parts = []
    for i, part in enumerate(parts):
        if part in ["table", "tb"] and i + 1 < len(parts) and parts[i + 1].isdigit():
            break
        base_parts.append(part)
    return "_".join(base_parts)


def detect_perturbation(task_name: str) -> str:
    """Detect perturbation type from task name."""
    if "_table_" in task_name:
        return "background_textures"
    elif "_tb_" in task_name:
        return "robot_init_states"
    elif "_add_" in task_name:
        return "objects_layout_add"
    elif "_view_" in task_name:
        return "camera_viewpoints"
    elif "_language_" in task_name:
        return "language_instructions"
    elif "_light_" in task_name:
        return "light_conditions"
    elif "copy" in task_name.lower():
        return "background_textures_copy"
    else:
        return "original"


def load_libero_plus_tasks(suite_name: str, max_tasks: int = None, task_offset: int = 0):
    """Load LIBERO-Plus tasks from task_classification.json."""
    if suite_name not in TASK_CLASSIFICATION:
        logger.error(f"Suite {suite_name} not found in task_classification.json")
        return []

    all_tasks = TASK_CLASSIFICATION[suite_name]
    logger.info(f"LIBERO-Plus {suite_name} has {len(all_tasks)} tasks in task_classification.json")

    # Apply task range
    tasks = all_tasks[task_offset:task_offset + max_tasks] if max_tasks else all_tasks[task_offset:]
    logger.info(f"Evaluating {len(tasks)} tasks (offset {task_offset})")

    return tasks


def main():
    parser = argparse.ArgumentParser(description="SnapFlow LIBERO-Plus evaluation with perturbation tasks")
    parser.add_argument("--preset", choices=["quick", "medium", "full"], default="quick",
                        help="Evaluation preset")
    parser.add_argument("--suite", type=str, default=None,
                        help="Override task suite")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Maximum number of tasks to evaluate")
    parser.add_argument("--task-offset", type=int, default=0,
                        help="Task offset for parallel eval")
    parser.add_argument("--nfe", type=int, default=1, choices=[1, 2, 4, 10],
                        help="Number of function evaluations")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--results-dir", type=str, default="results/eval_libero_plus_real",
                        help="Results output directory")
    args = parser.parse_args()

    # Load preset
    if args.preset:
        preset = PRESETS[args.preset]
        suites = preset["suites"]
        max_tasks = preset["max_tasks"]
        logger.info(f"Using preset: {args.preset} - {preset['description']}")

    if args.suite:
        suites = [args.suite]
    if args.max_tasks is not None:
        max_tasks = args.max_tasks

    logger.info(f"Evaluation config:")
    logger.info(f"  Suites: {suites}")
    logger.info(f"  Max tasks: {max_tasks or 'All'}")
    logger.info(f"  Task offset: {args.task_offset}")
    logger.info(f"  NFE: {args.nfe}")

    # Default checkpoint
    if args.checkpoint is None:
        args.checkpoint = "checkpoints/finetuned/snapflow/step_30000"
        logger.info(f"Using default checkpoint: {args.checkpoint}")

    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load SnapFlow policy
    logger.info(f"Loading SnapFlow policy from {ckpt_path}...")
    policy = load_policy(
        nfe=args.nfe,
        checkpoint_dir=str(ckpt_path),
        use_smf=False,
        use_snapflow=True,
    )
    logger.info("Policy loaded successfully")

    np.random.seed(args.seed)
    start_time = time.time()

    all_suite_results = []

    for suite_name in suites:
        logger.info("=" * 60)
        logger.info(f"Evaluating on {suite_name} (LIBERO-Plus perturbation tasks)")
        logger.info("=" * 60)

        # Load LIBERO-Plus tasks from task_classification.json
        tasks_data = load_libero_plus_tasks(suite_name, max_tasks, args.task_offset)

        if not tasks_data:
            logger.warning(f"No tasks to evaluate in {suite_name}")
            continue

        max_steps = MAX_STEPS_MAP.get(suite_name, 300)

        total_episodes = 0
        total_successes = 0
        all_latencies = []
        task_results = {}
        episode_details = []
        perturbation_counts = {}

        for task_idx, task_info in enumerate(tasks_data):
            task_id = task_info["id"]
            task_name = task_info["name"]
            task_description = task_name.replace("_", " ")
            perturbation = detect_perturbation(task_name)
            category = task_info.get("category", "Unknown")
            difficulty = task_info.get("difficulty_level", "N/A")

            # Get base task name for BDDL file lookup
            base_task_name = get_base_task_name(task_name)

            # Find BDDL file
            bddl_file = pathlib.Path(get_libero_path("bddl_files")) / suite_name / f"{base_task_name}.bddl"
            if not bddl_file.exists():
                # Try with full task name
                bddl_file = pathlib.Path(get_libero_path("bddl_files")) / suite_name / f"{task_name}.bddl"

            if not bddl_file.exists():
                logger.warning(f"BDDL file not found for {task_name}, skipping...")
                continue

            # Init states file
            init_file = pathlib.Path(get_libero_path("init_states")) / suite_name / f"{task_name}.pruned_init"
            if not init_file.exists():
                init_file = pathlib.Path(get_libero_path("init_states")) / suite_name / f"{base_task_name}.pruned_init"

            logger.info(f"[{task_idx+1}/{len(tasks_data)}] {task_description[:60]}... ({perturbation}, {category})")

            try:
                env_args = {
                    "bddl_file_name": bddl_file,
                    "camera_heights": 256,
                    "camera_widths": 256,
                }
                env = OffScreenRenderEnv(**env_args)
                env.seed(args.seed)

                # Load initial states if available
                initial_state = None
                if init_file.exists():
                    try:
                        import pickle
                        with open(init_file, "rb") as f:
                            initial_states = pickle.load(f)
                            initial_state = initial_states[0] if len(initial_states) > 0 else None
                    except:
                        pass

                # For LIBERO-Plus perturbation tasks, use env.reset() as default initial state
                env.reset()

                done, steps, ep_latencies = run_single_task_episode(
                    env=env,
                    initial_state=None,
                    task_description=task_description,
                    policy=policy,
                    max_steps=max_steps,
                    num_steps_wait=args.num_steps_wait,
                    replan_steps=args.replan_steps,
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
                    "perturbation": perturbation,
                    "category": category,
                    "difficulty": difficulty,
                    "success": done,
                    "steps": steps,
                    "avg_latency_ms": round(float(np.mean(ep_latencies)), 2) if ep_latencies else 0.0,
                })

                # Track perturbation stats
                perturbation_key = f"{perturbation}_{category}" if category != "Unknown" else perturbation
                if perturbation_key not in perturbation_counts:
                    perturbation_counts[perturbation_key] = {"total": 0, "success": 0}
                perturbation_counts[perturbation_key]["total"] += 1
                if done:
                    perturbation_counts[perturbation_key]["success"] += 1

                env.close()

            except Exception as e:
                logger.warning(f"Task {task_name} failed: {e}")
                episode_details.append({
                    "task_id": task_id,
                    "task_name": task_name,
                    "task_description": task_description,
                    "perturbation": perturbation,
                    "category": category,
                    "difficulty": difficulty,
                    "success": False,
                    "steps": 0,
                    "error": str(e),
                })

        # Suite summary
        suite_rate = total_successes / total_episodes if total_episodes > 0 else 0
        avg_latency = np.mean(all_latencies) if all_latencies else 0
        p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0

        logger.info(f"{suite_name} success rate: {suite_rate*100:.1f}%")
        logger.info(f"Avg latency: {avg_latency:.1f}ms | P95: {p95_latency:.1f}ms")

        # Perturbation breakdown
        if perturbation_counts:
            logger.info("Per-perturbation breakdown:")
            for ptype, counts in sorted(perturbation_counts.items()):
                rate = counts["success"] / counts["total"] if counts["total"] > 0 else 0
                logger.info(f"  {ptype}: {counts['success']}/{counts['total']} ({rate*100:.1f}%)")

        # Save results
        end_time = time.time()
        config_dict = {
            "benchmark": "libero-plus-perturbation",
            "task_suite": suite_name,
            "nfe": args.nfe,
            "max_tasks": max_tasks,
            "task_offset": args.task_offset,
            "replan_steps": args.replan_steps,
            "num_steps_wait": args.num_steps_wait,
            "seed": args.seed,
            "checkpoint_path": str(ckpt_path),
            "action_horizon": 10,
        }

        # Build task_results from episode_details
        task_results = {}
        for ep in episode_details:
            task_name = ep["task_name"]
            if task_name not in task_results:
                task_results[task_name] = {
                    "task_name": task_name,
                    "task_description": ep["task_description"],
                    "perturbation": ep["perturbation"],
                    "category": ep["category"],
                    "success": 0,
                    "total": 0,
                }
            task_results[task_name]["total"] += 1
            if ep["success"]:
                task_results[task_name]["success"] += 1

        result = build_result_json(
            config_dict, task_results, episode_details, all_latencies,
            total_successes, total_episodes, start_time, end_time,
        )
        all_suite_results.append(result)

        results_dir = pathlib.Path(args.results_dir)
        filepath = save_result_json(result, results_dir, suite_name)
        logger.info(f"Results saved to {filepath}")

    # Overall summary
    if len(suites) > 1:
        total_episodes_all = sum(r["overall"]["total_episodes"] for r in all_suite_results)
        total_successes_all = sum(r["overall"]["total_successes"] for r in all_suite_results)
        grand_total_rate = total_successes_all / total_episodes_all if total_episodes_all > 0 else 0

        logger.info("=" * 60)
        logger.info("Grand Total Results")
        logger.info("=" * 60)
        logger.info(f"Total episodes: {total_episodes_all}")
        logger.info(f"Total successes: {total_successes_all}")
        logger.info(f"Grand total rate: {grand_total_rate*100:.1f}%")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
