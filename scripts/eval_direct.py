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
import collections
import datetime
import json
import logging
import math
import pathlib
import socket
import sys
import time

import numpy as np

# ── 设置路径 ──────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OPENPI_DIR = PROJECT_ROOT / "third_party" / "openpi"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(OPENPI_DIR / "src"))
sys.path.insert(0, str(OPENPI_DIR / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(OPENPI_DIR / "third_party" / "libero"))

import jax
import jax.numpy as jnp
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

PRESETS = {
    "quick": {
        "suites": ["libero_spatial"],
        "num_episodes": 5,
    },
    "full": {
        "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        "num_episodes": 50,
    },
}

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def quat2axisangle(quat):
    quat = np.array(quat)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def preprocess_obs(obs, resize_size=224):
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


def load_policy(nfe: int, checkpoint_dir: str):
    """
    加载 policy。

    nfe=10: 使用原始 Pi0 模型（默认 num_steps=10）
    nfe=1:  使用 Pi05SMF 模型（num_steps=1）
    """
    checkpoint_path = pathlib.Path(checkpoint_dir).resolve()
    train_config = _config.get_config("pi05_libero")

    if nfe == 1:
        # 使用 Pi05SMF 模型
        logger.info("Loading Pi05SMF model for 1-NFE inference...")
        import flax.nnx as nnx
        import flax.traverse_util as traverse_util
        import orbax.checkpoint as ocp
        from smf_vla.models.pi05_smf import Pi05SMF, Pi05SMFConfig

        smf_config = Pi05SMFConfig(
            pi05=True,
            action_horizon=train_config.model.action_horizon,
            action_dim=train_config.model.action_dim,
            discrete_state_input=False,
        )

        # 直接用 orbax 加载（SMF checkpoint 没有 "params" wrapper）
        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(str(checkpoint_path / "params"))

        model = smf_config.create(jax.random.key(0))
        graphdef, state = nnx.split(model)
        pure_state = state.to_pure_dict()

        flat_params = traverse_util.flatten_dict(params)
        flat_state = traverse_util.flatten_dict(pure_state)

        loaded_count = 0
        for key in flat_state:
            if key in flat_params:
                flat_state[key] = flat_params[key]
                loaded_count += 1
            elif "time_proj" in "/".join(key):
                logger.info(f"Skipping (keeping init): {'/'.join(key)}")
            else:
                logger.warning(f"Not in checkpoint: {'/'.join(key)}")

        logger.info(f"Loaded {loaded_count} parameters")
        pure_state = traverse_util.unflatten_dict(flat_state)
        state.replace_by_pure_dict(pure_state)
        model = nnx.merge(graphdef, state)

        # 创建 policy，手动传入 model
        from openpi.policies import policy as _policy
        from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
        from openpi import transforms as _transforms
        from openpi.training import checkpoints as _checkpoints
        from openpi.shared import nnx_utils

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        # SMF checkpoint 无 assets 目录，使用 base checkpoint 的 norm_stats
        base_ckpt = PROJECT_ROOT / "checkpoints" / "base" / "pi05_libero"
        assets_dir = checkpoint_path / "assets"
        if not assets_dir.exists():
            assets_dir = base_ckpt / "assets"
            logger.info(f"Using base checkpoint assets: {assets_dir}")
        norm_stats = _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)

        policy = _policy.Policy(
            model,
            transforms=[
                _transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ],
            sample_kwargs={"num_steps": nfe},
        )
        return policy
    else:
        # 使用原始 Pi0 模型
        logger.info("Loading Pi0 model for 10-NFE inference...")
        policy = _policy_config.create_trained_policy(
            train_config,
            str(checkpoint_path),
            sample_kwargs={"num_steps": nfe},
        )
        return policy


def build_result_json(args, task_results, episode_details, all_latencies,
                      total_successes, total_episodes, start_time, end_time, suite_name):
    """Build structured result dictionary for JSON output."""
    latencies_arr = np.array(all_latencies) if all_latencies else np.array([0.0])
    duration = end_time - start_time

    return {
        "overall": {
            "total_success_rate": round(total_successes / total_episodes, 4) if total_episodes > 0 else 0.0,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
        },
        "config": {
            "task_suite": suite_name,
            "nfe": args.nfe,
            "num_episodes": args.num_episodes,
            "replan_steps": args.replan_steps,
            "num_steps_wait": args.num_steps_wait,
            "seed": args.seed,
            "checkpoint_path": str(args.checkpoint),
            "action_horizon": 10,
        },
        "per_task": {
            desc: {
                "task_description": desc,
                "successes": res["successes"],
                "episodes": res["episodes"],
                "rate": round(res["rate"], 4),
            }
            for desc, res in task_results.items()
        },
        "timing": {
            "all_latencies_ms": [round(x, 2) for x in all_latencies],
            "avg_latency_ms": round(float(np.mean(latencies_arr)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies_arr, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies_arr, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies_arr, 99)), 2),
        },
        "episode_details": episode_details,
        "metadata": {
            "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
            "end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
            "duration_seconds": round(duration, 1),
            "hostname": socket.gethostname(),
        },
    }


def save_result_json(result_dict, results_dir, suite_name):
    """Save result JSON to results_dir with timestamped filename."""
    results_path = pathlib.Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total_rate = result_dict["overall"]["total_success_rate"]
    nfe = result_dict["config"]["nfe"]
    pct_str = f"{total_rate * 100:.1f}pct"
    filename = f"{ts}_{suite_name}_{nfe}nfe_{pct_str}.json"
    filepath = results_path / filename

    with open(filepath, "w") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {filepath}")
    return filepath


def run_eval(args):
    np.random.seed(args.seed)
    start_time = time.time()

    logger.info(f"JAX backend: {jax.default_backend()}")
    logger.info(f"JAX devices: {jax.devices()}")

    # 加载 policy
    policy = load_policy(args.nfe, args.checkpoint)

    if getattr(args, "no_video", False):
        logger.info("Video recording disabled (--no-video)")

    # 加载 LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks
    logger.info(f"Task suite: {args.task_suite} ({num_tasks} tasks)")

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_map.get(args.task_suite, 300)

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

        # 检查可用初始状态数量
        max_available = len(initial_states)
        actual_episodes = min(args.num_episodes, max_available)
        if actual_episodes < args.num_episodes:
            logger.warning(f"Only {max_available} initial states available, capping from {args.num_episodes}")

        for episode_idx in range(actual_episodes):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            episode_latencies = []
            logger.info(f"[Task {task_id}] Ep {episode_idx+1}/{args.num_episodes}: {task_description}")

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
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
                    all_latencies.append(infer_ms)
                    episode_latencies.append(infer_ms)

                    action_chunk = result["actions"]
                    action_plan.extend(action_chunk[:args.replan_steps])

                    if episode_idx == 0 and task_id == 0:
                        logger.info(f"  First inference latency: {infer_ms:.1f}ms")

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1
            status = "✓ SUCCESS" if done else "✗ FAILURE"
            logger.info(f"  {status} (steps={t})")

            episode_details.append({
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "success": bool(done),
                "steps": t,
                "avg_latency_ms": round(float(np.mean(episode_latencies)), 2) if episode_latencies else 0.0,
            })

        task_rate = task_successes / task_episodes if task_episodes > 0 else 0
        task_results[task_description] = {
            "successes": task_successes,
            "episodes": task_episodes,
            "rate": task_rate,
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
    result = build_result_json(
        args, task_results, episode_details, all_latencies,
        total_successes, total_episodes, start_time, end_time,
        suite_name=args.task_suite,
    )
    results_dir = getattr(args, "results_dir", str(PROJECT_ROOT / "results" / "eval"))
    filepath = save_result_json(result, results_dir, args.task_suite)

    return total_rate, avg_latency, result, filepath


def main():
    parser = argparse.ArgumentParser(description="LIBERO direct evaluation")
    parser.add_argument("--nfe", type=int, default=10, choices=[1, 10])
    parser.add_argument("--task-suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "checkpoints" / "base" / "pi05_libero"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--preset", type=str, default=None, choices=["quick", "full"],
                        help="Eval preset: 'quick' (spatial, 5ep) or 'full' (all suites, 50ep)")
    parser.add_argument("--results-dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "eval"),
                        help="Directory to save structured JSON results")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip video recording (useful for full eval to save disk)")
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
