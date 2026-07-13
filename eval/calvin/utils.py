"""CALVIN 数据集工具。"""

import logging
import pathlib
import sys
from typing import Any, Dict

import numpy as np

CALVIN_DATASET_DIR = pathlib.Path("/root/autodl-tmp/datasets")
CALVIN_DEBUG_PATH = CALVIN_DATASET_DIR / "calvin" / "calvin" / "dataset" / "calvin_debug_dataset"
CALVIN_DD_PATH = CALVIN_DATASET_DIR / "calvin_D-D"

logger = logging.getLogger(__name__)


def load_calvin_obs(obs: Dict[str, Any], resize_size: int = 224):
    """将 CALVIN 环境观测预处理为模型输入格式。

    Returns: (img, wrist_img, state) — img/wrist_img 224x224 uint8, state (7,) float32
    """
    rgb_obs = obs.get("rgb_obs", {})
    if "rgb_static" in rgb_obs:
        img = rgb_obs["rgb_static"]
    else:
        first_key = next(iter(rgb_obs.keys())) if rgb_obs else None
        img = rgb_obs[first_key] if first_key else np.zeros((200, 200, 3), dtype=np.uint8)

    if "rgb_gripper" in rgb_obs:
        wrist_img = rgb_obs["rgb_gripper"]
    else:
        wrist_img = np.zeros((84, 84, 3), dtype=np.uint8)

    robot_obs = obs.get("robot_obs", np.zeros(15, dtype=np.float32))
    if robot_obs.shape == (15,):
        tcp_pos = robot_obs[0:3]
        tcp_orn = robot_obs[3:6]
        gripper_action = robot_obs[14:15]
        state = np.concatenate([tcp_pos, tcp_orn, gripper_action])
    else:
        state = np.zeros(7, dtype=np.float32)

    from PIL import Image

    img_pil = Image.fromarray(img).resize((resize_size, resize_size), Image.LANCZOS)
    wrist_img_pil = Image.fromarray(wrist_img).resize((resize_size, resize_size), Image.LANCZOS)
    img = np.array(img_pil, dtype=np.uint8)
    wrist_img = np.array(wrist_img_pil, dtype=np.uint8)

    return img, wrist_img, state.astype(np.float32)


def get_calvin_dataset_path(dataset_type: str = "debug") -> pathlib.Path:
    if dataset_type == "debug":
        return CALVIN_DEBUG_PATH
    elif dataset_type == "D":
        return CALVIN_DD_PATH / "task_D_D"
    elif dataset_type == "ABC":
        return CALVIN_DD_PATH / "task_ABC_D"
    elif dataset_type == "ABCD":
        return CALVIN_DD_PATH / "task_ABCD_D"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_calvin_validation_path(dataset_type: str = "debug") -> pathlib.Path:
    if dataset_type == "debug":
        return CALVIN_DEBUG_PATH / "validation"
    else:
        return get_calvin_dataset_path(dataset_type) / "validation"


def setup_calvin_paths():
    """设置 CALVIN 相关路径到 sys.path。"""
    calvin_root = CALVIN_DATASET_DIR / "calvin" / "calvin"
    calvin_env = calvin_root / "calvin_env"
    paths = [
        str(calvin_root),
        str(calvin_env),
        str(calvin_env / "calvin_env"),
        str(calvin_root / "calvin_models"),
    ]
    for p in paths:
        if pathlib.Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)


# 常量
CALVIN_MAX_STEPS = 360
CALVIN_NUM_SEQUENCES = 1000
CALVIN_DEBUG_NUM_SEQUENCES = 5
