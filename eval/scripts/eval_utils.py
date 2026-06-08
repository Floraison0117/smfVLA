#!/usr/bin/env python3
"""
共享评测工具模块。

提供 eval_direct.py 和 eval_libero_plus.py 的公共函数和常量。
"""

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

# ── 路径常量 ──────────────────────────────────────────────
# eval/ 目录是独立的，PROJECT_ROOT 指向 autodl-tmp/
EVAL_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOT = EVAL_ROOT.parent
OPENPI_DIR = PROJECT_ROOT / "third_party" / "openpi"

# ── 评测常量 ──────────────────────────────────────────────
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

logger = logging.getLogger(__name__)


def setup_paths():
    """添加评测所需的 sys.path 条目（幂等）。"""
    for p in [
        str(PROJECT_ROOT / "src"),
        str(OPENPI_DIR / "src"),
        str(OPENPI_DIR / "packages" / "openpi-client" / "src"),
        str(OPENPI_DIR / "third_party" / "libero"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def quat2axisangle(quat):
    """四元数 → 轴角表示。"""
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


def load_policy(nfe: int, checkpoint_dir: str, use_smf: bool = True, use_snapflow: bool = False):
    """
    加载 policy。

    nfe=10: 使用原始 Pi0 模型（默认 num_steps=10）
    nfe=1 + use_smf=True:  使用 Pi05SMF 模型（需要 SMF checkpoint）
    nfe=1 + use_smf=False: 使用原始 Pi0 模型（用于 pi05-libero 等非 SMF checkpoint）
    nfe=1 + use_snapflow=True: 使用 Pi05SnapFlow 模型（需要 SnapFlow checkpoint）
    """
    import jax
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    checkpoint_path = pathlib.Path(checkpoint_dir).resolve()
    train_config = _config.get_config("pi05_libero")

    if nfe == 1 and use_snapflow:
        logger.info("Loading Pi05SnapFlow model for 1-NFE inference...")
        import flax.nnx as nnx
        import flax.traverse_util as traverse_util
        import orbax.checkpoint as ocp
        # Note: This requires snapflow to be installed/available
        try:
            from snapflow.models.pi05_snapflow import Pi05SnapFlow, Pi05SnapFlowConfig
        except ImportError:
            logger.error("SnapFlow not available. Install snapflow to use use_snapflow=True")
            raise

        snapflow_config = Pi05SnapFlowConfig(
            pi05=True,
            action_horizon=train_config.model.action_horizon,
            action_dim=train_config.model.action_dim,
            discrete_state_input=False,
            alpha=0.5,
            lambda_consistency=0.1,
        )

        checkpointer = ocp.PyTreeCheckpointer()
        try:
            params = checkpointer.restore(str(checkpoint_path / "params"))
        except ValueError as e:
            if "sharding" in str(e).lower():
                logger.info("Multi-device checkpoint detected, restoring with single-device sharding...")
                from jax.sharding import SingleDeviceSharding
                single_sharding = SingleDeviceSharding(jax.devices()[0])
                restore_args = jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=single_sharding),
                    checkpointer.metadata(str(checkpoint_path / "params"))
                )
                params = checkpointer.restore(str(checkpoint_path / "params"), restore_args=restore_args)
            else:
                raise

        model = snapflow_config.create(jax.random.key(0))
        graphdef, state = nnx.split(model)
        pure_state = state.to_pure_dict()

        flat_params = traverse_util.flatten_dict(params)
        flat_state = traverse_util.flatten_dict(pure_state)

        loaded_count = 0
        for key in flat_state:
            if key in flat_params:
                flat_state[key] = flat_params[key]
                loaded_count += 1
            elif "time_proj" in "/".join(key) or "target_time_mlp" in "/".join(key):
                logger.info(f"Skipping (keeping init): {'/'.join(key)}")
            else:
                logger.warning(f"Not in checkpoint: {'/'.join(key)}")

        logger.info(f"Loaded {loaded_count} parameters")
        pure_state = traverse_util.unflatten_dict(flat_state)
        state.replace_by_pure_dict(pure_state)
        model = nnx.merge(graphdef, state)

        from openpi.policies import policy as _policy
        from openpi import transforms as _transforms
        from openpi.training import checkpoints as _checkpoints

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        base_ckpt = PROJECT_ROOT / "checkpoints" / "snapflow_base" / "pi05_libero"
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
    elif nfe == 1 and use_smf:
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

        checkpointer = ocp.PyTreeCheckpointer()
        try:
            params = checkpointer.restore(str(checkpoint_path / "params"))
        except ValueError as e:
            if "sharding" in str(e).lower():
                logger.info("Multi-device checkpoint detected, restoring with single-device sharding...")
                from jax.sharding import SingleDeviceSharding
                single_sharding = SingleDeviceSharding(jax.devices()[0])
                restore_args = jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=single_sharding),
                    checkpointer.metadata(str(checkpoint_path / "params"))
                )
                params = checkpointer.restore(str(checkpoint_path / "params"), restore_args=restore_args)
            else:
                raise

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

        from openpi.policies import policy as _policy
        from openpi import transforms as _transforms
        from openpi.training import checkpoints as _checkpoints

        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        base_ckpt = PROJECT_ROOT / "checkpoints" / "smf_base" / "pi05_libero"
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
        logger.info("Loading Pi0 model for 10-NFE inference...")
        policy = _policy_config.create_trained_policy(
            train_config,
            str(checkpoint_path),
            sample_kwargs={"num_steps": nfe},
        )
        return policy


def build_result_json(config_dict, task_results, episode_details, all_latencies,
                      total_successes, total_episodes, start_time, end_time):
    """
    构建结构化结果 JSON。

    config_dict: 评测配置字典，直接写入 result["config"]
    """
    latencies_arr = np.array(all_latencies) if all_latencies else np.array([0.0])
    duration = end_time - start_time

    return {
        "overall": {
            "total_success_rate": round(total_successes / total_episodes, 4) if total_episodes > 0 else 0.0,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
        },
        "config": config_dict,
        "per_task": task_results,
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
    """保存结果 JSON 到 results_dir，文件名含时间戳。"""
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


def run_single_task_episode(env, initial_state, task_description, policy,
                            max_steps, num_steps_wait, replan_steps):
    """
    运行单个 episode，返回 (success, steps, latencies)。

    这是评测循环的核心逻辑，被 eval_direct.py 和 eval_libero_plus.py 共用。

    Args:
        env: LIBERO environment
        initial_state: Initial state for the environment (None = use env.reset() default)
        task_description: Task description string
        policy: Policy model
        max_steps: Maximum steps per episode
        num_steps_wait: Number of dummy actions at start
        replan_steps: Number of actions to execute per inference
    """
    env.reset()
    action_plan = collections.deque()

    # Set initial state if provided, otherwise use default from env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        # Get initial observation after reset
        obs, _, _, _ = env.step([0.0] * 7)  # Dummy action to get obs

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
