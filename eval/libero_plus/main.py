#!/usr/bin/env python3
"""LIBERO-Plus 评测入口。

用法:
    python -m eval.libero_plus.main --model-type pi05 --nfe 1 --mode quick
    python -m eval.libero_plus.main --model-type dmf --nfe 10 --mode normal
    python -m eval.libero_plus.main --model-type pi05 --nfe 1 --mode fullset
"""

import argparse
import datetime
import json
import logging
import os
import pathlib
import socket
import sys

# headless rendering
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from eval.common import setup_paths
setup_paths()

from eval.common.policy_loader import load_policy
from eval.common.constants import PROJECT_ROOT
from eval.libero_plus.presets import PRESETS, SAFE_SUITES
from eval.libero_plus.runner import (
    run_eval_suite,
    sample_tasks_by_category,
    compute_per_perturbation_summary,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LIBERO-Plus evaluation")
    parser.add_argument("--model-type", type=str, default="pi05", choices=["pi05", "dmf"])
    parser.add_argument("--nfe", type=int, nargs="+", default=[1], choices=[1, 2, 4, 10])
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "normal", "fullset"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    # 默认 checkpoint
    if args.checkpoint is None:
        if args.model_type == "dmf":
            # 查找 dmf_finetuned 下最新 step
            dmf_dir = PROJECT_ROOT / "checkpoints" / "dmf_finetuned"
            steps = sorted(dmf_dir.glob("step_*")) if dmf_dir.exists() else []
            if steps:
                args.checkpoint = str(steps[-1])
            else:
                args.checkpoint = str(PROJECT_ROOT / "checkpoints" / "pi05_libero")
        else:
            args.checkpoint = str(PROJECT_ROOT / "checkpoints" / "pi05_libero")

    # 默认 results dir
    if args.results_dir is None:
        args.results_dir = str(PROJECT_ROOT / "eval" / "results" / args.model_type / "libero_plus")

    preset = PRESETS[args.mode]
    suites = [s for s in preset["suites"] if s in SAFE_SUITES]
    max_tasks = preset["max_tasks"]
    num_episodes = preset["num_episodes"]
    use_sampling = preset.get("use_sampling", False)

    sampled_task_names_map = {}
    if use_sampling:
        tasks_per_category = preset.get("tasks_per_category", 12)
        for s in suites:
            sampled_task_names_map[s] = sample_tasks_by_category(
                s, num_per_category=tasks_per_category, seed=args.seed
            )
        logger.info(f"Sampled tasks for {len(sampled_task_names_map)} suites")
        for s, names in sampled_task_names_map.items():
            logger.info(f"  {s}: {len(names)} tasks")

    logger.info(f"LIBERO-Plus evaluation | model={args.model_type} | mode={args.mode}")
    logger.info(f"  Suites: {suites} | NFE: {args.nfe} | Ep/task: {num_episodes}")
    logger.info(f"  Checkpoint: {args.checkpoint}")

    all_results = []
    for nfe in args.nfe:
        logger.info(f"\n{'='*60}\nLoading model for NFE={nfe}...\n{'='*60}")
        policy = load_policy(nfe, args.checkpoint, args.model_type)

        for suite_name in suites:
            logger.info(f"\n{'='*60}\nRunning: suite={suite_name}, nfe={nfe}\n{'='*60}")
            sampled = sampled_task_names_map.get(suite_name) if use_sampling else None
            result, filepath = run_eval_suite(
                nfe=nfe, suite_name=suite_name, policy=policy,
                seed=args.seed, max_tasks=max_tasks, task_offset=0,
                replan_steps=args.replan_steps, num_steps_wait=args.num_steps_wait,
                checkpoint=args.checkpoint, num_episodes=num_episodes,
                sampled_task_names=sampled,
            )
            if result is not None:
                all_results.append(result)

    # 多 suite/NFE 汇总
    if len(all_results) > 1:
        nfe_values = sorted(set(r["config"]["nfe"] for r in all_results))
        combined = {
            "benchmark": "libero-plus",
            "mode": args.mode,
            "nfe_values": nfe_values,
            "max_tasks": max_tasks,
            "suites": {},
            "grand_total_episodes": sum(r["overall"]["total_episodes"] for r in all_results),
            "grand_total_successes": sum(r["overall"]["total_successes"] for r in all_results),
            "metadata": {
                "end_time": datetime.datetime.now().isoformat(),
                "hostname": socket.gethostname(),
                "checkpoint": args.checkpoint,
                "model_type": args.model_type,
                "replan_steps": args.replan_steps,
            },
        }
        combined["grand_total_rate"] = round(
            combined["grand_total_successes"] / max(combined["grand_total_episodes"], 1), 4,
        )
        for r in all_results:
            key = f"{r['config']['task_suite']}_nfe{r['config']['nfe']}"
            combined["suites"][key] = r["overall"]

        combined["per_perturbation"] = compute_per_perturbation_summary(all_results)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nfe_str = "_".join(str(n) for n in nfe_values)
        combined_path = pathlib.Path(args.results_dir) / f"{ts}_combined_{nfe_str}nfe.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        logger.info(f"\nCombined results saved to: {combined_path}")
        logger.info(f"Grand total: {combined['grand_total_rate']*100:.1f}%")
        for cat, stats in combined.get("per_perturbation", {}).items():
            logger.info(f"  {cat}: {stats['rate']*100:.1f}%")


if __name__ == "__main__":
    main()
