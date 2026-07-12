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
OPENPI_DIR = PROJECT_ROOT / "openpi"

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
    paths = [
        str(PROJECT_ROOT / "smfVLA" / "src"),
        str(PROJECT_ROOT / "snapflow" / "src"),
        str(PROJECT_ROOT / "freeflow" / "src"),
        str(PROJECT_ROOT / "dmf" / "src"),
        str(OPENPI_DIR / "src"),
        str(OPENPI_DIR / "packages" / "openpi-client" / "src"),
    ]
    # Only add openpi/third_party/libero if it exists and is non-empty
    libero_tp = OPENPI_DIR / "third_party" / "libero"
    if libero_tp.exists() and any(libero_tp.iterdir()):
        paths.append(str(libero_tp))
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_checkpoint_with_single_device_sharding(checkpointer, checkpoint_path):
    """
    Load checkpoint with single-device sharding support.

    Handles checkpoints saved with multiple devices by restoring with
    single-device sharding when only one device is available.

    Args:
        checkpointer: Orbax checkpointer instance
        checkpoint_path: Path to checkpoint directory (string or pathlib.Path)

    Returns:
        Loaded checkpoint parameters

    Raises:
        ValueError: If checkpoint loading fails for non-sharding reasons
    """
    try:
        return checkpointer.restore(str(checkpoint_path))
    except ValueError as e:
        if "sharding" in str(e).lower() or "devices" in str(e).lower():
            logger.info("Multi-device checkpoint detected, restoring with single-device sharding...")
            import jax
            from jax.sharding import SingleDeviceSharding
            import orbax.checkpoint as ocp

            single_sharding = SingleDeviceSharding(jax.devices()[0])
            restore_args = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(sharding=single_sharding),
                checkpointer.metadata(str(checkpoint_path))
            )
            return checkpointer.restore(str(checkpoint_path), restore_args=restore_args)
        else:
            raise


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


def _detect_checkpoint_type(checkpoint_path: pathlib.Path) -> str:
    """
    检测 checkpoint 类型：original, smf, snapflow, freeflow, 或 dmf。

    通过检查是否存在特定参数：
    - original: 没有 time_proj
    - smf: 有 time_proj 但没有 target_time_mlp, 没有 time_mlp_in
    - snapflow: 有 target_time_mlp
    - freeflow: 有 time_mlp_in (FreeFlow dual time embeddings)
    - dmf: 有 logvar_proj (DMF learned-variance head; DMF reuses base time_mlp_in/out
      for both E(t) and E(r), so logvar_proj is the unique DMF identifier)

    支持两种 checkpoint 格式：
    - 标准格式: checkpoint_path/params/ (SMF, SnapFlow, DMF, base)
    - FreeFlow 格式: checkpoint_path/ 直接保存 (无 params/ 子目录)
    """
    import jax
    import flax.traverse_util as traverse_util
    import orbax.checkpoint as ocp

    # Check if this is FreeFlow format (has _METADATA directly at checkpoint_path)
    is_freeflow_format = (checkpoint_path / "_METADATA").exists()

    if is_freeflow_format:
        load_path = str(checkpoint_path)
    else:
        load_path = str(checkpoint_path / "params")

    checkpointer = ocp.PyTreeCheckpointer()

    try:
        params = checkpointer.restore(load_path)
    except FileNotFoundError:
        # Try alternate path
        if is_freeflow_format:
            load_path = str(checkpoint_path / "params")
        else:
            load_path = str(checkpoint_path)
        try:
            params = checkpointer.restore(load_path)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    except ValueError as e:
        if "sharding" in str(e).lower():
            from jax.sharding import SingleDeviceSharding
            single_sharding = SingleDeviceSharding(jax.devices()[0])
            restore_args = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(sharding=single_sharding),
                checkpointer.metadata(load_path)
            )
            params = checkpointer.restore(load_path, restore_args=restore_args)
        else:
            raise

    # Check checkpoint structure BEFORE extracting nested params
    # FreeFlow: {'model': ..., 'optimizer': ..., 'step': ...}
    # Standard: {'PaliGemma': ...} or {'params': {'PaliGemma': ...}}
    is_freeflow_structure = 'model' in params

    # FreeFlow checkpoint has nested structure: {'model': ..., 'optimizer': ..., 'step': ...}
    if is_freeflow_structure:
        params = params['model']

    flat = traverse_util.flatten_dict(params)
    has_time_proj = any('time_proj' in '/'.join(k) for k in flat.keys())
    has_target_time_mlp = any('target_time_mlp' in '/'.join(k) for k in flat.keys())
    has_time_mlp_in = any('time_mlp_in' in '/'.join(k) for k in flat.keys())
    has_logvar_proj = any('logvar_proj' in '/'.join(k) for k in flat.keys())

    # Detection order matters! Check unique identifiers first.
    # DMF: has logvar_proj (unique to DMF; DMF reuses base time_mlp_in/out)
    # SnapFlow: has target_time_mlp (unique)
    # SMF: has time_proj (unique)
    # FreeFlow: has time_mlp_in AND FreeFlow checkpoint structure
    # Original: has time_mlp_in but standard checkpoint structure
    if has_logvar_proj:
        return "dmf"
    elif has_target_time_mlp:
        return "snapflow"
    elif has_time_proj:
        return "smf"
    elif has_time_mlp_in and is_freeflow_structure:
        return "freeflow"
    elif has_time_mlp_in:
        return "original"
    else:
        return "original"


def load_policy(nfe: int, checkpoint_dir: str, use_smf: bool = True, use_snapflow: bool = False, use_freeflow: bool = False, use_dmf: bool = False):
    """
    加载 policy。

    支持多种方式（均支持任意 NFE）：
    1. 原生 Pi05 (use_smf=False, use_snapflow=False, use_dmf=False): 使用原始 Pi05 模型
    2. SMF (use_smf=True, use_snapflow=False, use_dmf=False): 使用 Pi05SMF 模型
    3. SnapFlow (use_snapflow=True): 使用 Pi05SnapFlow 模型
    4. DMF (use_dmf=True, auto-detected from checkpoint): 使用 Pi05DMF 模型
    5. FreeFlow (use_freeflow=True): 使用 Pi05FreeFlow 模型
    """
    import jax
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    checkpoint_path = pathlib.Path(checkpoint_dir).resolve()
    train_config = _config.get_config("pi05_libero")

    # 检测 checkpoint 类型
    ckpt_type = _detect_checkpoint_type(checkpoint_path)
    logger.info(f"Checkpoint type: {ckpt_type}, NFE={nfe}, use_smf={use_smf}, use_snapflow={use_snapflow}")

    if use_snapflow:
        logger.info(f"Loading Pi05SnapFlow model for {nfe}-NFE inference...")
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
        params = load_checkpoint_with_single_device_sharding(checkpointer, checkpoint_path / "params")

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
    elif ckpt_type == "dmf":
        logger.info(f"Loading Pi05DMF model for {nfe}-NFE inference...")
        import flax.nnx as nnx
        import flax.traverse_util as traverse_util
        import orbax.checkpoint as ocp
        # Note: This requires dmf to be installed/available
        try:
            from dmf_vla.models.pi05_dmf import Pi05DMF, Pi05DMFConfig
        except ImportError:
            logger.error("DMF not available. Install dmf to use DMF checkpoints")
            raise

        dmf_config = Pi05DMFConfig(
            pi05=True,
            action_horizon=train_config.model.action_horizon,
            action_dim=train_config.model.action_dim,
            discrete_state_input=False,
            dmf_depth_ratio=0.67,
            use_logvar=True,
        )

        checkpointer = ocp.PyTreeCheckpointer()
        params = load_checkpoint_with_single_device_sharding(checkpointer, checkpoint_path / "params")

        model = dmf_config.create(jax.random.key(0))
        graphdef, state = nnx.split(model)
        pure_state = state.to_pure_dict()

        flat_params = traverse_util.flatten_dict(params)
        flat_state = traverse_util.flatten_dict(pure_state)

        loaded_count = 0
        for key in flat_state:
            if key in flat_params:
                flat_state[key] = flat_params[key]
                loaded_count += 1
            elif "logvar_proj" in "/".join(key):
                logger.info(f"Skipping (keeping init): {'/'.join(key)}")
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
        base_ckpt = PROJECT_ROOT / "checkpoints" / "pi05_libero"
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
    elif use_freeflow:
        logger.info(f"Loading Pi05FreeFlow model for {nfe}-NFE inference...")
        import flax.nnx as nnx
        import flax.traverse_util as traverse_util
        import orbax.checkpoint as ocp
        from freeflow.models.pi05_freeflow import Pi05FreeFlow, create_freeflow_model
        from openpi.models import pi0_config

        checkpointer = ocp.PyTreeCheckpointer()

        # FreeFlow saves checkpoints directly at step_N/ (no params/ subdirectory)
        # Try params/ subdirectory first (for compatibility), then direct path
        params_path = checkpoint_path / "params"
        model_path = checkpoint_path

        # Check if this is a FreeFlow-format checkpoint (has _METADATA directly)
        is_freeflow_format = (checkpoint_path / "_METADATA").exists()

        if is_freeflow_format:
            logger.info(f"Detected FreeFlow checkpoint format (no params/ subdirectory)")
            load_from = str(checkpoint_path)
        else:
            load_from = str(params_path)

        params = load_checkpoint_with_single_device_sharding(checkpointer, load_from)

        # FreeFlow checkpoint has nested structure: {'model': ..., 'optimizer': ..., 'step': ...}
        if 'model' in params:
            logger.info(f"Extracting model params from FreeFlow checkpoint")
            params = params['model']

        # Check if this is a FreeFlow trainable-only checkpoint vs full checkpoint
        # Full VLM checkpoints have layer-stacked format: ~51 keys for entire VLM
        # Trainable-only checkpoints would have only action expert layers (~10-20 keys)
        flat_params = traverse_util.flatten_dict(params)

        # Detect trainable-only by checking for VLM backbone presence
        has_vlm_backbone = any('PaliGemma' in '/'.join(str(k) for k in key) for key in flat_params.keys())
        # Check if it has vision encoder or language model (not just action expert)
        has_img_encoder = any('PaliGemma/img' in '/'.join(str(k) for k in key) for key in flat_params.keys())
        has_llm_layers = any('PaliGemma/llm/layers' in '/'.join(str(k) for k in key) for key in flat_params.keys())

        # Full checkpoint has VLM backbone, trainable-only has only action layers
        is_trainable_only = has_vlm_backbone and not (has_img_encoder and has_llm_layers)

        if is_trainable_only:
            logger.info(f"Detected FreeFlow trainable-only checkpoint ({len(flat_params)} params)")
            logger.info(f"Loading base model from SMF base checkpoint first...")

            # Load base model from SMF base checkpoint
            base_ckpt_path = PROJECT_ROOT / "checkpoints" / "pi05_libero"
            base_checkpointer = ocp.PyTreeCheckpointer()

            base_params = load_checkpoint_with_single_device_sharding(base_checkpointer, base_ckpt_path / "params")

            # Merge: base_params + freeflow_params (freeflow overrides)
            flat_base = traverse_util.flatten_dict(base_params)
            for key, value in flat_params.items():
                flat_base[key] = value
            params = traverse_util.unflatten_dict(flat_base)
            flat_params = flat_base  # Update for subsequent code
            logger.info(f"Merged base model ({len(flat_base)} params) with FreeFlow trainable params")
        else:
            logger.info(f"Detected FreeFlow full checkpoint ({len(flat_params)} params)")

        # Create FreeFlow model
        config = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            pi05=True,
            action_dim=train_config.model.action_dim,
            action_horizon=train_config.model.action_horizon,
        )
        model = create_freeflow_model(config)
        graphdef, state = nnx.split(model)
        pure_state = state.to_pure_dict()

        # Strip 'value' suffix from checkpoint keys (Orbax serialization format)
        flat_params_clean = {}
        for k, v in flat_params.items():
            if k and k[-1] == 'value':
                k = k[:-1]  # Remove 'value' suffix
            flat_params_clean[k] = v
        flat_params = flat_params_clean

        flat_state = traverse_util.flatten_dict(pure_state)

        loaded_count = 0
        for key in flat_state:
            if key in flat_params:
                flat_state[key] = flat_params[key]
                loaded_count += 1
            elif "time_proj" in "/".join(key) or "student_head" in "/".join(key):
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

        # Try checkpoint assets first, then fall back to SMF base
        assets_dir = checkpoint_path / "assets"
        if not assets_dir.exists():
            assets_dir = PROJECT_ROOT / "checkpoints" / "pi05_libero" / "assets"
            logger.info(f"Using SMF base checkpoint assets: {assets_dir}")
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
    elif use_smf and not use_snapflow:
        logger.info(f"Loading Pi05SMF model for {nfe}-NFE inference...")
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
        params = load_checkpoint_with_single_device_sharding(checkpointer, checkpoint_path / "params")

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
        base_ckpt = PROJECT_ROOT / "checkpoints" / "pi05_libero"
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
        logger.info(f"Loading original Pi05 model for {nfe}-NFE inference...")
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
