"""LIBERO-Plus 模型加载：pi0.5（auto-detect JAX/PyTorch）、DMF 和 Pi-Flow（JAX/nnx）。"""

import logging
import pathlib

from .constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


def load_checkpoint_with_single_device_sharding(checkpointer, checkpoint_path):
    """加载 checkpoint，自动处理多设备 -> 单设备 sharding。"""
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
                checkpointer.metadata(str(checkpoint_path)),
            )
            return checkpointer.restore(str(checkpoint_path), restore_args=restore_args)
        else:
            raise


def _detect_checkpoint_type(checkpoint_path: pathlib.Path) -> str:
    """检测 checkpoint 类型：original（pi0.5）、dmf 或 piflow。"""
    import jax
    import flax.traverse_util as traverse_util
    import orbax.checkpoint as ocp

    is_freeflow_format = (checkpoint_path / "_METADATA").exists()
    load_path = str(checkpoint_path) if is_freeflow_format else str(checkpoint_path / "params")

    checkpointer = ocp.PyTreeCheckpointer()

    try:
        params = checkpointer.restore(load_path)
    except FileNotFoundError:
        alt_path = str(checkpoint_path / "params") if is_freeflow_format else str(checkpoint_path)
        params = checkpointer.restore(alt_path)
    except ValueError as e:
        if "sharding" in str(e).lower():
            from jax.sharding import SingleDeviceSharding
            single_sharding = SingleDeviceSharding(jax.devices()[0])
            restore_args = jax.tree.map(
                lambda _: ocp.ArrayRestoreArgs(sharding=single_sharding),
                checkpointer.metadata(load_path),
            )
            params = checkpointer.restore(load_path, restore_args=restore_args)
        else:
            raise

    if "model" in params:
        params = params["model"]

    flat = traverse_util.flatten_dict(params)
    if any("gmm_mean_proj" in "/".join(k) for k in flat.keys()):
        return "piflow"
    if any("logvar_proj" in "/".join(k) for k in flat.keys()):
        return "dmf"
    return "original"


def load_policy(nfe: int, checkpoint_dir: str, model_type: str):
    """加载 LIBERO-Plus 评测 policy。

    Args:
        nfe: 采样步数 (1, 2, 4, 10)
        checkpoint_dir: checkpoint 目录路径
        model_type: "pi05"、"dmf" 或 "piflow"
    """
    import jax
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    checkpoint_path = pathlib.Path(checkpoint_dir).resolve()
    train_config = _config.get_config("pi05_libero")
    ckpt_type = _detect_checkpoint_type(checkpoint_path)
    logger.info(f"Checkpoint type: {ckpt_type}, model_type={model_type}, NFE={nfe}")

    if model_type == "piflow" or ckpt_type == "piflow":
        logger.info(f"Loading Pi05PiFlow model for {nfe}-NFE inference...")
        import flax.nnx as nnx
        import flax.traverse_util as traverse_util
        import orbax.checkpoint as ocp

        try:
            from piflow_vla.models.pi05_piflow import Pi05PiFlow, Pi05PiFlowConfig
        except ImportError:
            logger.error("Pi-Flow not available. Install piflow to use Pi-Flow checkpoints")
            raise

        pf_config = Pi05PiFlowConfig(
            pi05=True,
            action_horizon=train_config.model.action_horizon,
            action_dim=train_config.model.action_dim,
            discrete_state_input=False,
            num_components=8,
            inner_substeps=8,
        )

        checkpointer = ocp.PyTreeCheckpointer()
        params = load_checkpoint_with_single_device_sharding(
            checkpointer, checkpoint_path / "params"
        )

        model = pf_config.create(jax.random.key(0))
        graphdef, state = nnx.split(model)
        pure_state = state.to_pure_dict()

        flat_params = traverse_util.flatten_dict(params)
        flat_state = traverse_util.flatten_dict(pure_state)

        loaded_count = 0
        for key in flat_state:
            if key in flat_params:
                flat_state[key] = flat_params[key]
                loaded_count += 1
            elif any(p in "/".join(key) for p in ("gmm_mean_proj", "gmm_logstd_proj", "gmm_logweight_proj")):
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
            sample_kwargs={"num_steps": nfe, "method": "gmflow"},
        )
        return policy
    elif model_type == "dmf" or ckpt_type == "dmf":
        logger.info(f"Loading Pi05DMF model for {nfe}-NFE inference...")
        import flax.nnx as nnx
        import flax.traverse_util as traverse_util
        import orbax.checkpoint as ocp

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
        params = load_checkpoint_with_single_device_sharding(
            checkpointer, checkpoint_path / "params"
        )

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
        logger.info(f"Loading pi0.5 model for {nfe}-NFE inference...")
        policy = _policy_config.create_trained_policy(
            train_config,
            str(checkpoint_path),
            sample_kwargs={"num_steps": nfe},
        )
        return policy
