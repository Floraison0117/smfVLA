#!/usr/bin/env python3
"""
CALVIN 数据集工具模块。

提供 CALVIN 数据集加载、预处理和评估相关的工具函数。
"""

import logging
import pathlib
import sys
from typing import Any, Dict

import numpy as np

# 路径设置
CALVIN_DATASET_DIR = pathlib.Path("/root/autodl-tmp/datasets")
CALVIN_DEBUG_PATH = CALVIN_DATASET_DIR / "calvin/calvin/dataset/calvin_debug_dataset"
CALVIN_DD_PATH = CALVIN_DATASET_DIR / "calvin_D-D"

logger = logging.getLogger(__name__)


def load_calvin_obs(obs: Dict[str, Any], resize_size: int = 224) -> tuple:
    """
    将 CALVIN 环境观测预处理为模型输入格式。

    Args:
        obs: CALVIN 环境观测字典，包含 rgb_obs 和 robot_obs
        resize_size: 目标图像大小 (默认 224)

    Returns:
        (img, wrist_img, state) tuple
        - img: 预处理后的静态相机图像 [H, W, 3] uint8
        - wrist_img: 预处理后的抓手相机图像 [H, W, 3] uint8 (如果没有，返回 zeros)
        - state: 机器人状态向量 [8] float32
    """
    # 获取静态相机图像
    rgb_obs = obs.get("rgb_obs", {})
    if "rgb_static" in rgb_obs:
        img = rgb_obs["rgb_static"]  # (200, 200, 3)
    else:
        # 回退到第一个可用的相机
        first_key = next(iter(rgb_obs.keys())) if rgb_obs else None
        img = rgb_obs[first_key] if first_key else np.zeros((200, 200, 3), dtype=np.uint8)

    # 获取抓手相机图像 (如果不存在，创建 zeros)
    if "rgb_gripper" in rgb_obs:
        wrist_img = rgb_obs["rgb_gripper"]  # (84, 84, 3)
    else:
        wrist_img = np.zeros((84, 84, 3), dtype=np.uint8)

    # 获取机器人状态
    robot_obs = obs.get("robot_obs", np.zeros(15, dtype=np.float32))
    if robot_obs.shape == (15,):
        # CALVIN robot_obs[15]: [tcp_pos(3), tcp_orn_euler(3), gripper_width(1), arm_joints(7), gripper_action(1)]
        # RLinf 的 7 维 state = [tcp_pos(3), tcp_orn(3), gripper_ACTION(1)]。
        # 关键: 第 7 维是 gripper ACTION (robot_obs[14], 范围 ±1)，不是 gripper WIDTH (robot_obs[6], 0~0.08)。
        # （norm_stats.state dim6 的 q01/q99 = ±1 证实是动作。）
        tcp_pos = robot_obs[0:3]
        tcp_orn = robot_obs[3:6]          # Euler angles
        gripper_action = robot_obs[14:15]  # 夹爪动作 ±1（不是宽度）
        state = np.concatenate([tcp_pos, tcp_orn, gripper_action])  # (7,)
    else:
        state = np.zeros(7, dtype=np.float32)

    # 调整图像大小到 resize_size (使用简单的 resize，保持与 LIBERO 一致)
    from PIL import Image

    img_pil = Image.fromarray(img).resize((resize_size, resize_size), Image.LANCZOS)
    wrist_img_pil = Image.fromarray(wrist_img).resize((resize_size, resize_size), Image.LANCZOS)

    img = np.array(img_pil, dtype=np.uint8)
    wrist_img = np.array(wrist_img_pil, dtype=np.uint8)

    # CALVIN robot state 为 7 维 (tcp_pos3 + tcp_orn3 + gripper1)，与 CALVIN norm_stats(7维) 对齐；
    # 不做 8 维填充（那是 LIBERO 约定，会与 CALVIN 7维归一化 broadcast 冲突）。
    return img, wrist_img, state.astype(np.float32)


def get_calvin_dataset_path(dataset_type: str = "debug") -> pathlib.Path:
    """
    获取 CALVIN 数据集路径。

    Args:
        dataset_type: "debug", "D", "ABC", or "ABCD"

    Returns:
        数据集路径
    """
    if dataset_type == "debug":
        path = CALVIN_DEBUG_PATH
    elif dataset_type == "D":
        path = CALVIN_DD_PATH / "task_D_D"
    elif dataset_type == "ABC":
        path = CALVIN_DD_PATH / "task_ABC_D"
    elif dataset_type == "ABCD":
        path = CALVIN_DD_PATH / "task_ABCD_D"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if not path.exists():
        logger.warning(f"CALVIN dataset path not found: {path}")

    return path


def get_calvin_validation_path(dataset_type: str = "debug") -> pathlib.Path:
    """
    获取 CALVIN validation 数据集路径。

    Args:
        dataset_type: "debug", "D", "ABC", or "ABCD"

    Returns:
        validation 数据集路径
    """
    if dataset_type == "debug":
        path = CALVIN_DEBUG_PATH / "validation"
    else:
        path = get_calvin_dataset_path(dataset_type) / "validation"

    if not path.exists():
        logger.warning(f"CALVIN validation path not found: {path}")

    return path


def load_lang_annotations(val_path: pathlib.Path) -> Dict[str, Any]:
    """
    加载 CALVIN 语言标注。

    Args:
        val_path: validation 数据集路径

    Returns:
        语言标注字典
    """
    lang_ann_path = val_path / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_ann_path.exists():
        logger.error(f"Language annotations not found: {lang_ann_path}")
        return {}

    lang_ann = np.load(lang_ann_path, allow_pickle=True).item()
    return lang_ann


def get_episode_files(val_path: pathlib.Path) -> list:
    """
    获取 validation 数据集中的所有 episode 文件。

    Args:
        val_path: validation 数据集路径

    Returns:
        episode 文件路径列表
    """
    episode_files = sorted(val_path.glob("episode_*.npz"))
    logger.info(f"Found {len(episode_files)} episode files in {val_path}")
    return episode_files


# CALVIN 评估常量
CALVIN_MAX_STEPS = 360  # 每个 subtask 的最大步数
CALVIN_NUM_SEQUENCES = 1000  # 评估序列数 (full dataset)
CALVIN_DEBUG_NUM_SEQUENCES = 5  # debug 数据集序列数

# CALVIN 相机配置
CALVIN_STATIC_CAMERA_SIZE = (200, 200)
CALVIN_GRIPPER_CAMERA_SIZE = (84, 84)


def calvin_action_to_libero_format(action: np.ndarray) -> np.ndarray:
    """
    将 CALVIN action 格式转换为 LIBERO 格式。

    CALVIN rel_actions: [x, y, z, euler_x, euler_y, euler_z, gripper]
    LIBERO actions: [x, y, z, qx, qy, qz, qw, gripper_open]

    注意: CALVIN 使用相对动作，LIBERO 使用绝对动作 + 四元数。

    Args:
        action: CALVIN action (7,)

    Returns:
        LIBERO 格式 action (8,)
    """
    # 简化版本: 直接复制位置和 gripper，旋转部分用 zeros
    if action.shape[0] == 7:
        pos = action[0:3]
        euler = action[3:6]
        gripper = action[6:7]
        # 将 euler 转换为简单的 quaternion (简化处理)
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # 单位四元数
        return np.concatenate([pos, quat, gripper])
    else:
        return np.zeros(8, dtype=np.float32)


def setup_calvin_paths():
    """设置 CALVIN 相关路径到 sys.path。"""
    calvin_root = CALVIN_DATASET_DIR / "calvin/calvin"
    calvin_env = calvin_root / "calvin_env"

    paths_to_add = [
        str(calvin_root),
        str(calvin_env),
        str(calvin_env / "calvin_env"),
        str(calvin_root / "calvin_models"),
    ]

    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)


def calvin_quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """
    四元数 → 轴角表示 (用于 LIBERO 兼容)。

    Args:
        quat: 四元数 [qx, qy, qz, qw]

    Returns:
        轴角向量 [3,]
    """
    quat = np.array(quat)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den
