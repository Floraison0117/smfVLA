"""
SMF (State-Machine-Free) 模型适配器。

提供加载SMF模型checkpoint的接口，用于LIBERO评估。
"""

import logging
import pathlib

logger = logging.getLogger(__name__)

# ── 全局路径设置 ──────────────────────────────────────────────
EVAL_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = EVAL_ROOT.parent
OPENPI_DIR = PROJECT_ROOT / "third_party" / "openpi"
SMF_DIR = PROJECT_ROOT / "smfVLA"


def setup_smf_paths():
    """设置SMF模型加载所需的Python路径。"""
    import sys
    paths_to_add = [
        str(SMF_DIR / "src"),
        str(OPENPI_DIR / "src"),
        str(OPENPI_DIR / "packages" / "openpi-client" / "src"),
        str(OPENPI_DIR / "third_party" / "libero"),
    ]
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_smf_config(
    checkpoint_dir: str,
    action_horizon: int = 8,
    action_dim: int = 7,
    discrete_state_input: bool = False,
):
    """
    加载SMF模型配置。

    Args:
        checkpoint_dir: Checkpoint目录路径
        action_horizon: Action horizon
        action_dim: Action维度
        discrete_state_input: 是否使用离散状态输入

    Returns:
        Pi05SMFConfig对象
    """
    setup_smf_paths()

    from smf_vla.models.pi05_smf import Pi05SMFConfig

    return Pi05SMFConfig(
        pi05=True,
        action_horizon=action_horizon,
        action_dim=action_dim,
        discrete_state_input=discrete_state_input,
    )


def load_smf_model(
    checkpoint_dir: str,
    nfe: int = 1,
    base_checkpoint_dir: str = None,
    action_horizon: int = 8,
    action_dim: int = 7,
):
    """
    加载SMF模型并构建Policy。

    Args:
        checkpoint_dir: SMF checkpoint目录路径
        nfe: Number of function evaluations (1 for SMF)
        base_checkpoint_dir: Base checkpoint目录（用于加载assets/norm_stats）
        action_horizon: Action horizon
        action_dim: Action维度

    Returns:
        Policy对象，可直接用于推理
    """
    setup_smf_paths()

    import jax
    import flax.nnx as nnx
    import flax.traverse_util as traverse_util
    import orbax.checkpoint as ocp
    from smf_vla.models.pi05_smf import Pi05SMF, Pi05SMFConfig
    from openpi.policies import policy as _policy
    from openpi import transforms as _transforms
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import config as _config

    checkpoint_path = pathlib.Path(checkpoint_dir).resolve()

    # 使用pi05_libero配置作为训练配置
    train_config = _config.get_config("pi05_libero")

    # 创建SMF配置
    smf_config = Pi05SMFConfig(
        pi05=True,
        action_horizon=action_horizon,
        action_dim=action_dim,
        discrete_state_input=False,
    )

    # 加载模型参数
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

    # 创建模型并加载参数
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
            logger.debug(f"Skipping (keeping init): {'/'.join(key)}")
        else:
            logger.warning(f"Not in checkpoint: {'/'.join(key)}")

    logger.info(f"Loaded {loaded_count} parameters for SMF model")
    pure_state = traverse_util.unflatten_dict(flat_state)
    state.replace_by_pure_dict(pure_state)
    model = nnx.merge(graphdef, state)

    # 构建数据配置（用于获取transforms和norm_stats）
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    # 确定base checkpoint路径（用于加载assets）
    if base_checkpoint_dir is None:
        base_ckpt = PROJECT_ROOT / "checkpoints" / "smf_base" / "pi05_libero"
    else:
        base_ckpt = pathlib.Path(base_checkpoint_dir)

    assets_dir = checkpoint_path / "assets"
    if not assets_dir.exists():
        assets_dir = base_ckpt / "assets"
        logger.info(f"Using base checkpoint assets: {assets_dir}")

    norm_stats = _checkpoints.load_norm_stats(assets_dir, data_config.asset_id)

    # 构建Policy
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


def load_smf_config_dict(checkpoint_dir: str) -> dict:
    """
    从SMF checkpoint目录加载配置字典（如果存在）。

    Args:
        checkpoint_dir: Checkpoint目录路径

    Returns:
        配置字典，如果不存在返回空字典
    """
    import json

    config_path = pathlib.Path(checkpoint_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}
