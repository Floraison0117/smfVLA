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
    使用 BILINEAR 重采样（比 LANCZOS 快 3-5 倍）。
    """
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)
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

def _detect_format(dataset_path: pathlib.Path) -> dict[str, str]:
    """
    检测数据集格式（LeRobot v2.0 vs v2.1），返回列名映射。

    v2.0: image, wrist_image, state, actions (images in parquet)
    v2.1: observation.state, action (images in videos/ as mp4 files)

    Returns:
        dict with keys: image, wrist_image, state, actions, format
    """
    # 检查是否有 videos 目录（v2.1 特征）
    videos_dir = dataset_path / "videos"
    has_videos = videos_dir.exists() and any(videos_dir.iterdir())

    # 读取第一个 parquet 文件检测列名
    data_dir = dataset_path / "data"
    first_chunk = data_dir / "chunk-000"
    parquet_files = sorted(first_chunk.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {first_chunk}")

    sample_df = pd.read_parquet(parquet_files[0], columns=None)
    cols = set(sample_df.columns)

    # 检测格式
    has_image_col = "image" in cols
    has_state_v21 = "observation.state" in cols

    if has_image_col:
        # v2.0 format: images in parquet
        logger.info("检测到 LeRobot v2.0 格式（Parquet 图像）")
        return {
            "image": "image",
            "wrist_image": "wrist_image",
            "state": "state",
            "actions": "actions",
            "format": "v2.0",
        }
    elif has_videos and has_state_v21:
        # v2.1 format: images in video files
        logger.info("检测到 LeRobot v2.1 格式（视频图像）")
        return {
            "image": "video:observation.images.front",
            "wrist_image": "video:observation.images.wrist",
            "state": "observation.state",
            "actions": "action",
            "format": "v2.1",
        }
    else:
        # Fallback to v2.0
        logger.warning(f"无法确定数据集格式，回退到 v2.0。列名: {sorted(cols)}")
        return {
            "image": "image",
            "wrist_image": "wrist_image",
            "state": "state",
            "actions": "actions",
            "format": "v2.0",
        }


def _decode_video_frame(video_path: pathlib.Path, frame_idx: int) -> np.ndarray:
    """
    从 mp4 视频文件中解码指定帧（支持 AV1 编码）。

    Args:
        video_path: mp4 文件路径
        frame_idx: 帧索引

    Returns:
        numpy uint8 (H, W, C) 图像
    """
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]

    # seek 到指定帧
    for i, frame in enumerate(container.decode(video=0)):
        if i == frame_idx:
            # 转换为 RGB numpy array
            img = frame.to_ndarray(format="rgb24")
            container.close()
            return img.astype(np.uint8)

    container.close()
    raise RuntimeError(f"无法读取视频帧: {video_path}, frame={frame_idx}")


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
    创建 libero 数据加载器（支持 LeRobot v2.0 和 v2.1 格式）。

    直接从 Parquet 文件加载数据，不依赖 lerobot。
    图像经过旋转 180° + resize 224×224 后作为模型输入。
    输出格式兼容 pi0.5 Observation.from_dict。

    Args:
        dataset_path: 数据集根目录（包含 data/, meta/ 的 LeRobot v2.0/v2.1 格式）
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
    videos_dir = dataset_path / "videos"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据集不存在: {dataset_path}\n"
            f"请确认 data/ 目录下有 Parquet 文件。"
        )

    # 检测数据集格式（v2.0 vs v2.1）
    col_map = _detect_format(dataset_path)

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

    # 检测是否为视频格式
    is_video_format = col_map.get("format") == "v2.1"
    if is_video_format:
        logger.info("使用视频图像格式（v2.1）")
        # 预计算视频路径映射: episode_idx -> (front_video_path, wrist_video_path)
        _video_cache: dict[int, tuple[pathlib.Path, pathlib.Path]] = {}
        for ep in episodes:
            ep_idx = ep["episode_index"]
            chunk_idx = ep_idx // 1000
            front_path = videos_dir / f"chunk-{chunk_idx:03d}" / "observation.images.front" / f"episode_{ep_idx:06d}.mp4"
            wrist_path = videos_dir / f"chunk-{chunk_idx:03d}" / "observation.images.wrist" / f"episode_{ep_idx:06d}.mp4"
            if front_path.exists() and wrist_path.exists():
                _video_cache[ep_idx] = (front_path, wrist_path)
        logger.info(f"视频文件缓存: {len(_video_cache)} episodes")

        # 预计算 parquet -> episode_index 映射
        _parquet_to_episode: dict[pathlib.Path, int] = {}
        for parquet_path, row, task_idx, ep_len in frame_index_list:
            if parquet_path not in _parquet_to_episode:
                # 从文件名解析 episode_index: episode_NNNNNN.parquet
                ep_idx = int(parquet_path.stem.split("_")[1])
                _parquet_to_episode[parquet_path] = ep_idx

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
            if is_video_format:
                # v2.1: 从视频文件解码帧
                ep_idx = _parquet_to_episode.get(parquet_path)
                if ep_idx is not None and ep_idx in _video_cache:
                    front_path, wrist_path = _video_cache[ep_idx]
                    img = _decode_video_frame(front_path, row)
                    wrist_img = _decode_video_frame(wrist_path, row)
                else:
                    # fallback: 使用黑色图像
                    img = np.zeros((256, 256, 3), dtype=np.uint8)
                    wrist_img = np.zeros((256, 256, 3), dtype=np.uint8)
                images.append(_rotate_180(_resize_224(img)))
                wrist_images.append(_rotate_180(_resize_224(wrist_img)))
            else:
                # v2.0: 从 parquet 解码图像
                images.append(_process_image(df[col_map["image"]].iloc[row]))
                wrist_images.append(_process_image(df[col_map["wrist_image"]].iloc[row]))

            # State
            state_val = df[col_map["state"]].iloc[row]
            if isinstance(state_val, np.ndarray):
                states.append(state_val.astype(np.float32))
            else:
                states.append(np.array(state_val, dtype=np.float32))

            # Action chunk: 当前帧开始的 action_horizon 步
            action_chunk = []
            for ah in range(action_horizon):
                arow = row + ah
                if arow < df.shape[0]:
                    a = df[col_map["actions"]].iloc[arow]
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
