"""
libero 数据加载器。

直接从 Parquet 文件加载 libero 数据集（LeRobot v2.0 格式）。
处理流程：
  1. 读取 Parquet 中的二进制图像 → PIL Image → numpy uint8
  2. 旋转 180°（与评测 main.py 中 [::-1, ::-1] 一致）
  3. resize 256×256 → 224×224（使用 PIL LANCZOS，正方形无需 padding）
  4. 构建 action chunk（未来 N 步 actions）
  5. 加载任务描述作为 prompt
  6. 输出格式兼容 pi0.5 Observation.from_dict

输出 batch 结构：
  {
      "observation": {
          "image": {"base_0_rgb": [B,224,224,3] uint8,
                    "left_wrist_0_rgb": [B,224,224,3] uint8,
                    "right_wrist_0_rgb": [B,224,224,3] uint8},
          "image_mask": {"base_0_rgb": True,
                         "left_wrist_0_rgb": True,
                         "right_wrist_0_rgb": False},
          "state": [B, 8] float32,
      },
      "actions": [B, action_horizon, action_dim] float32,
      "action_mean": [action_dim] float32,
      "action_std": [action_dim] float32,
      "prompt": list[str],  # length B
  }
"""

import io
import json
import logging
import pathlib
from typing import Any, Iterator

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# norm stats
# ---------------------------------------------------------------------------

def load_norm_stats(checkpoint_dir: str | pathlib.Path) -> dict[str, dict[str, np.ndarray]]:
    """
    从 checkpoint 目录加载 normalization statistics。

    Returns:
        dict with keys "state" and "actions", each containing "mean", "std", "q01", "q99"
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    norm_stats_path = checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"

    if not norm_stats_path.exists():
        raise FileNotFoundError(f"norm_stats.json not found at {norm_stats_path}")

    with open(norm_stats_path) as f:
        raw = json.load(f)

    # 兼容 {"norm_stats": {...}} 和直接 {...} 两种格式
    raw = raw.get("norm_stats", raw)

    stats = {}
    for key, val in raw.items():
        stats[key] = {
            k: np.array(v, dtype=np.float32) for k, v in val.items()
        }

    logger.info(f"Loaded norm stats from {norm_stats_path}")
    return stats


# ---------------------------------------------------------------------------
# tasks.jsonl
# ---------------------------------------------------------------------------

def _load_tasks(meta_dir: pathlib.Path) -> dict[int, str]:
    """从 tasks.jsonl 加载任务描述，返回 {task_index: task_str}。"""
    tasks_path = meta_dir / "tasks.jsonl"
    tasks: dict[int, str] = {}
    with open(tasks_path) as f:
        for line in f:
            obj = json.loads(line)
            tasks[obj["task_index"]] = obj["task"]
    return tasks


# ---------------------------------------------------------------------------
# episodes.jsonl
# ---------------------------------------------------------------------------

def _load_episodes(meta_dir: pathlib.Path) -> list[dict]:
    """从 episodes.jsonl 加载 episode 元数据。"""
    episodes_path = meta_dir / "episodes.jsonl"
    episodes: list[dict] = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------

def _decode_image(raw: dict | bytes) -> np.ndarray:
    """
    从 Parquet 的 image 列解码为 numpy uint8 (H, W, C)。
    Parquet 中 image 列存储为 {"bytes": ..., "path": ...} dict。
    """
    if isinstance(raw, dict):
        raw = raw["bytes"]
    pil_img = Image.open(io.BytesIO(raw))
    return np.array(pil_img, dtype=np.uint8)  # (H, W, C) uint8


def _rotate_180(img: np.ndarray) -> np.ndarray:
    """旋转 180°，等价于评测时的 img[::-1, ::-1]。"""
    return np.ascontiguousarray(img[::-1, ::-1])


def _resize_224(img: np.ndarray) -> np.ndarray:
    """
    使用 PIL 将 256×256 图像 resize 到 224×224。
    使用 LANCZOS 重采样。正方形图像不需要 padding。
    """
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((224, 224), Image.LANCZOS)
    return np.array(pil_img, dtype=np.uint8)


def _process_image(raw: dict | bytes) -> np.ndarray:
    """完整图像处理流水线: 解码 → 旋转 180° → resize 224×224。"""
    img = _decode_image(raw)
    img = _rotate_180(img)
    img = _resize_224(img)
    return img


# ---------------------------------------------------------------------------
# data loader
# ---------------------------------------------------------------------------

def create_data_loader(
    dataset_path: str | pathlib.Path,
    batch_size: int = 64,
    num_workers: int = 0,
    action_horizon: int = 10,
    action_dim: int = 7,
    target_action_dim: int | None = None,
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    """
    创建 libero 数据加载器。

    直接从 Parquet 文件加载数据，不依赖 lerobot。
    图像经过旋转 180° + resize 224×224 后作为模型输入。
    输出格式兼容 pi0.5 Observation.from_dict。

    Args:
        dataset_path: 数据集根目录（包含 data/, meta/ 的 LeRobot v2.0 格式）
        batch_size: batch 大小
        num_workers: 未使用（保留接口兼容）
        action_horizon: action chunk 长度
        action_dim: action 维度
        seed: 随机种子

    Yields:
        dict，结构见模块 docstring。
    """
    dataset_path = pathlib.Path(dataset_path)
    meta_dir = dataset_path / "meta"
    data_dir = dataset_path / "data"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据集不存在: {dataset_path}\n"
            f"请确认 data/ 目录下有 Parquet 文件。"
        )

    # 加载元数据
    tasks = _load_tasks(meta_dir)
    episodes = _load_episodes(meta_dir)
    logger.info(f"加载了 {len(episodes)} 个 episode，{len(tasks)} 个任务")

    # 加载 norm stats（用于 action 归一化）
    norm_stats_path = dataset_path / "norm_stats.json"
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            raw_stats = json.load(f)
        # norm_stats.json 结构: {"norm_stats": {"state": {...}, "actions": {...}}}
        stats = raw_stats.get("norm_stats", raw_stats)
        action_mean = np.array(stats["actions"]["mean"], dtype=np.float32)
        action_std = np.array(stats["actions"]["std"], dtype=np.float32)
    else:
        logger.warning(f"norm_stats.json 不存在: {norm_stats_path}，使用零均值和单位标准差")
        action_mean = np.zeros(action_dim, dtype=np.float32)
        action_std = np.ones(action_dim, dtype=np.float32)

    # 构建所有帧的索引: (parquet_path, row_idx, task_index, ep_len)
    # 使用 episodes.jsonl 中的 length 字段，避免读取 parquet 文件
    frame_index_list: list[tuple[pathlib.Path, int, int, int]] = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // 1000  # chunks_size = 1000
        parquet_path = data_dir / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            logger.warning(f"Parquet 文件不存在: {parquet_path}，跳过")
            continue
        task_idx = ep.get("task_index", 0)
        ep_len = ep["length"]
        for row in range(ep_len):
            frame_index_list.append((parquet_path, row, task_idx, ep_len))

    logger.info(f"总帧数: {len(frame_index_list)}")

    rng = np.random.RandomState(seed)

    # 预计算有效帧索引（跳过每个 episode 最后 action_horizon 帧）
    # 使用 episodes.jsonl 中的 length，无需读取 parquet 文件
    valid_indices = [
        i for i, (_, row, _, ep_len) in enumerate(frame_index_list)
        if row + action_horizon <= ep_len
    ]
    logger.info(f"有效帧数: {len(valid_indices)} (跳过 {len(frame_index_list) - len(valid_indices)} 帧)")

    if len(valid_indices) < batch_size:
        logger.warning(f"有效帧数 ({len(valid_indices)}) < batch_size ({batch_size})")

    # Pad actions/action_mean/action_std to target_action_dim once (before loop)
    if target_action_dim is not None and target_action_dim > action_dim:
        pad_width = target_action_dim - action_dim
        action_mean = np.pad(action_mean, (0, pad_width), mode='constant')
        action_std = np.pad(action_std, (0, pad_width), mode='constant')
        action_std[action_dim:] = 1.0  # padded dims: std=1
        effective_action_dim = target_action_dim
    else:
        effective_action_dim = action_dim

    # 预加载当前 parquet 缓存（避免重复读取同一文件）
    _cached_path: pathlib.Path | None = None
    _cached_df: pd.DataFrame | None = None

    def _get_parquet(path: pathlib.Path) -> pd.DataFrame:
        nonlocal _cached_path, _cached_df
        if _cached_path != path:
            _cached_df = pd.read_parquet(path)
            _cached_path = path
        return _cached_df  # type: ignore[return-value]

    while True:
        if len(valid_indices) < batch_size:
            chosen = valid_indices
        else:
            chosen = rng.choice(valid_indices, size=batch_size, replace=False).tolist()

        # 构建 batch
        images = []
        wrist_images = []
        states = []
        actions_list = []
        prompts = []

        for idx in chosen:
            parquet_path, row, task_idx, _ep_len = frame_index_list[idx]
            df = _get_parquet(parquet_path)

            # 图像处理: 解码 → 旋转 180° → resize 224×224
            images.append(_process_image(df["image"].iloc[row]))
            wrist_images.append(_process_image(df["wrist_image"].iloc[row]))

            # State
            state_val = df["state"].iloc[row]
            if isinstance(state_val, np.ndarray):
                states.append(state_val.astype(np.float32))
            else:
                states.append(np.array(state_val, dtype=np.float32))

            # Action chunk: 当前帧开始的 action_horizon 步
            action_chunk = []
            for ah in range(action_horizon):
                arow = row + ah
                if arow < df.shape[0]:
                    a = df["actions"].iloc[arow]
                    if isinstance(a, np.ndarray):
                        action_chunk.append(a[:action_dim].astype(np.float32))
                    else:
                        action_chunk.append(np.array(a[:action_dim], dtype=np.float32))
                else:
                    # 超出 episode 范围，用最后一帧 action 填充
                    action_chunk.append(action_chunk[-1] if action_chunk else np.zeros(action_dim, dtype=np.float32))
            actions_list.append(np.stack(action_chunk, axis=0))  # [action_horizon, action_dim]

            # Prompt
            prompts.append(tasks.get(task_idx, ""))

        # 输出格式兼容 Observation.from_dict
        actions_arr = np.stack(actions_list, axis=0).astype(np.float32)  # [B, ah, action_dim]

        # Pad actions to target_action_dim if needed (e.g., 7 → 32 for pi0.5)
        if effective_action_dim > action_dim:
            actions_arr = np.pad(actions_arr, ((0, 0), (0, 0), (0, effective_action_dim - action_dim)), mode='constant')

        batch = {
            "observation": {
                "image": {
                    "base_0_rgb": np.stack(images, axis=0),              # [B, 224, 224, 3] uint8
                    "left_wrist_0_rgb": np.stack(wrist_images, axis=0),  # [B, 224, 224, 3] uint8
                    "right_wrist_0_rgb": np.zeros_like(np.stack(images, axis=0)),  # padding
                },
                "image_mask": {
                    "base_0_rgb": np.ones(batch_size, dtype=bool),
                    "left_wrist_0_rgb": np.ones(batch_size, dtype=bool),
                    "right_wrist_0_rgb": np.zeros(batch_size, dtype=bool),  # pi0.5 不使用 right wrist
                },
                "state": np.stack(states, axis=0).astype(np.float32),  # [B, 8]
            },
            "actions": actions_arr,      # [B, action_horizon, target_action_dim]
            "action_mean": action_mean,  # [target_action_dim]
            "action_std": action_std,    # [target_action_dim]
            "prompt": prompts,           # list of str, length B
        }

        yield batch


def create_fake_data_loader(
    batch_size: int = 64,
    action_horizon: int = 10,
    action_dim: int = 7,
    state_dim: int = 8,
    num_batches: int = 100,
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    """
    创建假数据加载器（用于 smoke test）。

    Yields:
        与 create_data_loader 相同格式的 dict
    """
    rng = np.random.RandomState(seed)

    for _ in range(num_batches):
        base_img = rng.randint(0, 256, (batch_size, 224, 224, 3), dtype=np.uint8)
        yield {
            "observation": {
                "image": {
                    "base_0_rgb": base_img,
                    "left_wrist_0_rgb": rng.randint(0, 256, (batch_size, 224, 224, 3), dtype=np.uint8),
                    "right_wrist_0_rgb": np.zeros_like(base_img),
                },
                "image_mask": {
                    "base_0_rgb": np.ones(batch_size, dtype=bool),
                    "left_wrist_0_rgb": np.ones(batch_size, dtype=bool),
                    "right_wrist_0_rgb": np.zeros(batch_size, dtype=bool),
                },
                "state": rng.randn(batch_size, state_dim).astype(np.float32),
            },
            "actions": rng.randn(batch_size, action_horizon, action_dim).astype(np.float32),
            "action_mean": np.zeros(action_dim, dtype=np.float32),
            "action_std": np.ones(action_dim, dtype=np.float32),
            "prompt": ["pick up the object"] * batch_size,
        }
