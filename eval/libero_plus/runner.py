"""LIBERO-Plus 评测 runner：环境创建、episode 执行、扰动处理。"""

import collections
import json
import logging
import pathlib
import random
import sys
import time

import numpy as np

from eval.common.constants import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    MAX_STEPS_MAP,
    PROJECT_ROOT,
)
from eval.common.utils import build_result_json, save_result_json, quat2axisangle

logger = logging.getLogger(__name__)

# ── LIBERO-Plus 路径配置（必须在 import libero 之前）──────
LIBERO_PLUS_DIR = PROJECT_ROOT / "datasets" / "libero-plus" / "LIBERO-plus"
LIBERO_PLUS_BENCHMARK = LIBERO_PLUS_DIR / "libero" / "libero"
LIBERO_PLUS_CONFIG_DIR = PROJECT_ROOT / "datasets" / "libero-plus" / ".libero_config"
LIBERO_PLUS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LIBERO_PLUS_CONFIG_FILE = LIBERO_PLUS_CONFIG_DIR / "config.yaml"

import os as _os
import yaml as _yaml

_libero_plus_config = {
    "benchmark_root": str(LIBERO_PLUS_BENCHMARK),
    "bddl_files": str(LIBERO_PLUS_BENCHMARK / "bddl_files"),
    "init_files": str(LIBERO_PLUS_BENCHMARK / "init_files"),
    "init_states": str(LIBERO_PLUS_BENCHMARK / "init_files"),
    "datasets": str(LIBERO_PLUS_DIR.parent / "datasets"),
    "assets": str(LIBERO_PLUS_BENCHMARK / "assets"),
}
with open(LIBERO_PLUS_CONFIG_FILE, "w") as f:
    _yaml.dump(_libero_plus_config, f)
_os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_PLUS_CONFIG_DIR)

# 注入 libero-plus 到 sys.path 头部以覆盖标准 libero
sys.path.insert(0, str(LIBERO_PLUS_DIR))

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ── 扰动类别 ──────────────────────────────────────────────
TASK_CLASSIFICATION_PATH = LIBERO_PLUS_BENCHMARK / "benchmark" / "task_classification.json"


def load_task_classification():
    with open(TASK_CLASSIFICATION_PATH, "r") as f:
        data = json.load(f)
    mapping = {}
    for suite_name, tasks in data.items():
        mapping[suite_name] = {}
        for task in tasks:
            mapping[suite_name][task["name"]] = task["category"]
    return mapping


TASK_CLASSIFICATION = load_task_classification()


def sample_tasks_by_category(suite_name, num_per_category=12, seed=7):
    """按扰动类别均匀采样 task。"""
    if suite_name not in TASK_CLASSIFICATION:
        return None

    cat_to_tasks = collections.defaultdict(list)
    for task_name, category in TASK_CLASSIFICATION[suite_name].items():
        cat_to_tasks[category].append(task_name)

    rng = random.Random(seed)
    sampled = []
    for category, task_names in sorted(cat_to_tasks.items()):
        if len(task_names) <= num_per_category:
            sampled.extend(task_names)
        else:
            stride = len(task_names) / num_per_category
            indices = [int(i * stride) for i in range(num_per_category)]
            sampled.extend(task_names[i] for i in indices)

    return sampled


# ── Episode 执行 ──────────────────────────────────────────


def preprocess_obs(obs, resize_size=224):
    """将 LIBERO 环境观测预处理为模型输入格式。"""
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    return img, wrist_img, state


def run_single_task_episode(env, initial_state, task_description, policy,
                            max_steps, num_steps_wait, replan_steps):
    """运行单个 episode，返回 (success, steps, latencies)。"""
    env.reset()
    action_plan = collections.deque()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs, _, _, _ = env.step([0.0] * 7)

    t = 0
    done = False
    episode_latencies = []

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        img, wrist_img, state = preprocess_obs(obs)

        if not action_plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(task_description),
            }

            infer_start = time.monotonic()
            result = policy.infer(element)
            infer_ms = (time.monotonic() - infer_start) * 1000
            episode_latencies.append(infer_ms)

            action_chunk = result["actions"]
            action_plan.extend(action_chunk[:replan_steps])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action.tolist())
        if done:
            break
        t += 1

    return bool(done), t, episode_latencies


def clear_jax_cache():
    """清理 JAX GPU 显存，防止长评测 OOM。"""
    import gc
    try:
        import jax
        jax.clear_backends()
        gc.collect()
    except Exception:
        pass


# ── Suite 评测 ────────────────────────────────────────────


def run_eval_suite(nfe, suite_name, policy, seed, max_tasks, task_offset,
                   replan_steps, num_steps_wait, checkpoint, num_episodes=1,
                   sampled_task_names=None):
    """评测单个 suite。"""
    np.random.seed(seed)
    start_time = time.time()

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        logger.error(f"Suite '{suite_name}' not in benchmark dict, skipping.")
        return None, None
    task_suite = benchmark_dict[suite_name]()
    num_tasks = task_suite.n_tasks
    logger.info(f"Task suite: {suite_name} ({num_tasks} tasks total)")

    if sampled_task_names is not None:
        name_to_id = {}
        for tid in range(num_tasks):
            task = task_suite.get_task(tid)
            tn = task.bddl_file.replace(".bddl", "").replace(".yaml", "")
            name_to_id[tn] = tid
        task_ids = []
        for tn in sampled_task_names:
            if tn in name_to_id:
                task_ids.append(name_to_id[tn])
            else:
                logger.warning(f"Sampled task '{tn}' not found in suite, skipping.")
        actual_tasks = len(task_ids)
        logger.info(f"Sampled {actual_tasks} tasks across perturbation categories")
    else:
        task_start = task_offset
        task_end = num_tasks if max_tasks is None else min(task_start + max_tasks, num_tasks)
        task_ids = list(range(task_start, task_end))
        actual_tasks = task_end - task_start
        logger.info(f"Evaluating tasks [{task_offset}, {task_offset + actual_tasks}) = {actual_tasks} tasks")

    max_steps = MAX_STEPS_MAP.get(suite_name, 300)

    total_episodes = 0
    total_successes = 0
    all_latencies = []
    task_results = {}
    episode_details = []

    for idx, task_id in enumerate(task_ids):
        task = task_suite.get_task(task_id)
        task_description = task.language
        task_name = task.bddl_file.replace(".bddl", "").replace(".yaml", "")

        perturbation_category = "Unknown"
        if suite_name in TASK_CLASSIFICATION and task_name in TASK_CLASSIFICATION[suite_name]:
            perturbation_category = TASK_CLASSIFICATION[suite_name][task_name]

        initial_states = task_suite.get_task_init_states(task_id)
        bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(seed)

        logger.info(f"[{idx+1}/{actual_tasks}] [{perturbation_category}] {task_description[:70]}...")

        task_episodes = 0
        task_successes = 0

        max_available = len(initial_states)
        actual_episodes = min(num_episodes, max_available)

        for episode_idx in range(actual_episodes):
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
                    task_successes += 1
                    total_successes += 1

                task_episodes += 1
                total_episodes += 1
                status = "SUCCESS" if done else "FAILURE"
                logger.info(f"  Ep {episode_idx+1}/{actual_episodes}: {status} (steps={steps})")

                episode_details.append({
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "task_name": task_name,
                    "task_description": task_description,
                    "perturbation_category": perturbation_category,
                    "success": done,
                    "steps": steps,
                    "avg_latency_ms": round(float(np.mean(ep_latencies)), 2) if ep_latencies else 0.0,
                })

                clear_jax_cache()

            except Exception as e:
                logger.warning(f"Task {task_id} Ep {episode_idx+1} failed with error: {e}")
                episode_details.append({
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "task_name": task_name,
                    "task_description": task_description,
                    "perturbation_category": perturbation_category,
                    "success": False,
                    "steps": 0,
                    "error": str(e),
                })
                task_episodes += 1
                total_episodes += 1

        task_rate = task_successes / task_episodes if task_episodes > 0 else 0
        logger.info(f"  Task {task_name[:50]}: {task_successes}/{task_episodes} ({task_rate*100:.1f}%)")

        task_results[task_name] = {
            "task_id": task_id,
            "task_name": task_name,
            "task_description": task_description,
            "perturbation_category": perturbation_category,
            "successes": task_successes,
            "episodes": task_episodes,
            "rate": task_rate,
        }

        env.close()
        clear_jax_cache()

    end_time = time.time()
    total_rate = total_successes / total_episodes if total_episodes > 0 else 0
    avg_latency = np.mean(all_latencies) if all_latencies else 0
    p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0

    logger.info("=" * 60)
    logger.info(f"RESULTS: {suite_name} | NFE={nfe} | {actual_tasks} tasks | {num_episodes} ep/task")
    logger.info("=" * 60)
    logger.info(f"Total: {total_successes}/{total_episodes} ({total_rate*100:.1f}%)")
    logger.info(f"Avg latency: {avg_latency:.1f}ms | P95: {p95_latency:.1f}ms")
    logger.info("=" * 60)

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


def compute_per_perturbation_summary(all_results):
    """跨 suite 按扰动类别汇总。"""
    cat_stats = collections.defaultdict(lambda: {"successes": 0, "episodes": 0})
    for result in all_results:
        for ep in result.get("episode_details", []):
            cat = ep.get("perturbation_category", "Unknown")
            cat_stats[cat]["successes"] += 1 if ep["success"] else 0
            cat_stats[cat]["episodes"] += 1

    summary = {}
    for cat, stats in sorted(cat_stats.items()):
        rate = stats["successes"] / stats["episodes"] if stats["episodes"] > 0 else 0
        summary[cat] = {
            "successes": stats["successes"],
            "episodes": stats["episodes"],
            "rate": round(rate, 4),
        }
    return summary
