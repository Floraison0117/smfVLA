"""
libero data loader for DMF training.

Loads LeRobot v2.0/v2.1 format parquet datasets for the pi0.5 model.
Processing: decode -> rotate 180 -> resize 224x224 -> normalize actions.

Output batch:
  {
      "observation": {
          "image": {"base_0_rgb": [B,224,224,3] uint8, "left_wrist_0_rgb": ..., "right_wrist_0_rgb": ...},
          "image_mask": {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False},
          "state": [B, 8] float32,
      },
      "actions": [B, action_horizon, action_dim] float32,
      "action_mean": [action_dim] float32,
      "action_std": [action_dim] float32,
      "prompt": list[str],
  }
"""

import io
import json
import logging
import pathlib
from functools import lru_cache
from typing import Any, Iterator

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


# ── norm stats ────────────────────────────────────────────────

def load_norm_stats(checkpoint_dir: str | pathlib.Path) -> dict[str, dict[str, np.ndarray]]:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    norm_stats_path = checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"norm_stats.json not found at {norm_stats_path}")
    with open(norm_stats_path) as f:
        raw = json.load(f)
    raw = raw.get("norm_stats", raw)
    stats = {}
    for key, val in raw.items():
        stats[key] = {k: np.array(v, dtype=np.float32) for k, v in val.items()}
    logger.info(f"Loaded norm stats from {norm_stats_path}")
    return stats


# ── metadata loading ──────────────────────────────────────────

def _load_tasks(meta_dir: pathlib.Path) -> dict[int, str]:
    tasks_path = meta_dir / "tasks.jsonl"
    tasks: dict[int, str] = {}
    with open(tasks_path) as f:
        for line in f:
            obj = json.loads(line)
            tasks[obj["task_index"]] = obj["task"]
    return tasks


def _load_episodes(meta_dir: pathlib.Path) -> list[dict]:
    episodes_path = meta_dir / "episodes.jsonl"
    episodes: list[dict] = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


# ── image helpers ─────────────────────────────────────────────

def _decode_image(raw: dict | bytes) -> np.ndarray:
    if isinstance(raw, dict):
        raw = raw["bytes"]
    pil_img = Image.open(io.BytesIO(raw))
    return np.array(pil_img, dtype=np.uint8)


def _rotate_180(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ::-1])


def _resize_224(img: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((224, 224), Image.BILINEAR)
    return np.array(pil_img, dtype=np.uint8)


def _process_image(raw: dict | bytes) -> np.ndarray:
    img = _decode_image(raw)
    img = _rotate_180(img)
    img = _resize_224(img)
    return img


# ── format detection ──────────────────────────────────────────

def _detect_format(dataset_path: pathlib.Path) -> dict[str, str]:
    data_dir = dataset_path / "data"
    first_chunk = data_dir / "chunk-000"
    parquet_files = sorted(first_chunk.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {first_chunk}")

    sample_df = pd.read_parquet(parquet_files[0], columns=None)
    cols = set(sample_df.columns)

    has_image_col = "image" in cols
    has_state_v21 = "observation.state" in cols

    if has_image_col:
        logger.info("Detected LeRobot v2.0 format (images in parquet)")
        return {
            "image": "image", "wrist_image": "wrist_image",
            "state": "state", "actions": "actions", "format": "v2.0",
        }
    elif has_state_v21:
        logger.info("Detected LeRobot v2.1 format (video-based images)")
        return {
            "image": "video:observation.images.front",
            "wrist_image": "video:observation.images.wrist",
            "state": "observation.state", "actions": "action", "format": "v2.1",
        }
    else:
        logger.warning(f"Unknown format, falling back to v2.0. Columns: {sorted(cols)}")
        return {
            "image": "image", "wrist_image": "wrist_image",
            "state": "state", "actions": "actions", "format": "v2.0",
        }


# ── video decoding (v2.1 only) ────────────────────────────────

def _decode_video_frames(video_path: pathlib.Path, frame_indices: set[int]) -> dict[int, np.ndarray]:
    """Decode specific frames from an mp4 video in a single pass using PyAV.

    Opens the video once and iterates through frames, collecting only the requested
    indices. Much faster than per-frame seeking, especially for AV1 software decoding.
    """
    import av
    results: dict[int, np.ndarray] = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        container.seek(0)
        for i, frame in enumerate(container.decode(stream)):
            if i in frame_indices:
                img = frame.to_ndarray(format="rgb24")
                results[i] = img.astype(np.uint8)
            if len(results) == len(frame_indices):
                break
    finally:
        container.close()
    return results


# ── parquet reading (LRU-cached) ──────────────────────────────

@lru_cache(maxsize=32)
def _read_parquet_cached(path_str: str) -> pd.DataFrame:
    """LRU-cached parquet reader.  path_str is used as the cache key."""
    return pd.read_parquet(path_str)


# ── main data loader ──────────────────────────────────────────

def create_data_loader(
    dataset_path: str | pathlib.Path,
    batch_size: int = 64,
    num_workers: int = 0,  # kept for API compatibility; unused
    action_horizon: int = 10,
    action_dim: int = 7,
    target_action_dim: int | None = None,
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    """
    Create a LIBERO data loader.

    Supports LeRobot v2.0 (images embedded in parquet) and v2.1 (images in mp4 video files).
    Yields batches indefinitely with per-epoch shuffling.

    Args:
        dataset_path: LeRobot dataset root (contains data/, meta/).
        batch_size: Number of frames per batch.
        action_horizon: Action chunk length.
        action_dim: Raw action dimension (7 for LIBERO).
        target_action_dim: Pad actions to this dimension (32 for pi0.5).
        seed: Random seed for shuffling.
    """
    dataset_path = pathlib.Path(dataset_path)
    meta_dir = dataset_path / "meta"
    data_dir = dataset_path / "data"
    videos_dir = dataset_path / "videos"

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    col_map = _detect_format(dataset_path)
    is_video_format = col_map["format"] == "v2.1"

    tasks = _load_tasks(meta_dir)
    episodes = _load_episodes(meta_dir)
    logger.info(f"Loaded {len(episodes)} episodes, {len(tasks)} tasks")

    # ── norm stats ──
    norm_stats_path = dataset_path / "norm_stats.json"
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            raw_stats = json.load(f)
        stats = raw_stats.get("norm_stats", raw_stats)
        action_mean = np.array(stats["actions"]["mean"], dtype=np.float32)
        action_std = np.array(stats["actions"]["std"], dtype=np.float32)
    else:
        logger.warning(f"No norm_stats.json at {norm_stats_path}, using identity norm")
        action_mean = np.zeros(action_dim, dtype=np.float32)
        action_std = np.ones(action_dim, dtype=np.float32)

    # ── pad action dim ──
    if target_action_dim is not None and target_action_dim > action_dim:
        pad_width = target_action_dim - action_dim
        action_mean = np.pad(action_mean, (0, pad_width), mode="constant")
        action_std = np.pad(action_std, (0, pad_width), mode="constant")
        action_std[action_dim:] = 1.0
        effective_action_dim = target_action_dim
    else:
        effective_action_dim = action_dim

    # ── build flat frame index ──
    # Each entry: (parquet_path, row_in_parquet, task_index, episode_length)
    frame_entries: list[tuple[pathlib.Path, int, int, int]] = []
    for ep in episodes:
        ep_idx = ep["episode_index"]
        chunk_idx = ep_idx // 1000
        parquet_path = data_dir / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            continue
        task_idx = ep.get("task_index", 0)
        ep_len = ep["length"]
        for row in range(ep_len):
            frame_entries.append((parquet_path, row, task_idx, ep_len))

    # Only keep frames with enough future steps for action_horizon
    valid_indices = [
        i for i, (_, row, _, ep_len) in enumerate(frame_entries)
        if row + action_horizon <= ep_len
    ]
    logger.info(f"Total frames: {len(frame_entries)}, valid: {len(valid_indices)}")
    logger.info(f"Batch size: {batch_size}, steps/epoch: {len(valid_indices) // batch_size}")

    if len(valid_indices) < batch_size:
        logger.warning(f"Valid frames ({len(valid_indices)}) < batch_size ({batch_size})")

    # ── video path index (v2.1 only) ──
    episode_video_paths: dict[int, tuple[pathlib.Path, pathlib.Path]] = {}
    parquet_to_episode: dict[pathlib.Path, int] = {}
    if is_video_format:
        for ep in episodes:
            ep_idx = ep["episode_index"]
            chunk_idx = ep_idx // 1000
            front = videos_dir / f"chunk-{chunk_idx:03d}" / "observation.images.front" / f"episode_{ep_idx:06d}.mp4"
            wrist = videos_dir / f"chunk-{chunk_idx:03d}" / "observation.images.wrist" / f"episode_{ep_idx:06d}.mp4"
            if front.exists() and wrist.exists():
                episode_video_paths[ep_idx] = (front, wrist)
        for pq_path, _, _, _ in frame_entries:
            if pq_path not in parquet_to_episode:
                ep_idx = int(pq_path.stem.split("_")[1])
                parquet_to_episode[pq_path] = ep_idx
        logger.info(f"Video episodes: {len(episode_video_paths)}")

    rng = np.random.RandomState(seed)

    # ── infinite yielding loop ──
    while True:
        # Shuffle valid indices each epoch
        epoch_indices = rng.permutation(valid_indices).tolist()

        for batch_start in range(0, len(epoch_indices), batch_size):
            batch_indices = epoch_indices[batch_start:batch_start + batch_size]
            if len(batch_indices) < batch_size:
                continue  # drop incomplete final batch

            images, wrist_images, states, actions_list, prompts = [], [], [], [], []

            # Group batch indices by parquet file to minimize reads
            by_parquet: dict[pathlib.Path, list[tuple[int, int, int]]] = {}
            for idx in batch_indices:
                pq_path, row, task_idx, _ep_len = frame_entries[idx]
                by_parquet.setdefault(pq_path, []).append((idx, row, task_idx))

            for pq_path, items in by_parquet.items():
                df = _read_parquet_cached(str(pq_path))

                # Pre-decode all needed video frames in a single pass per video
                frame_cache_front: dict[int, np.ndarray] = {}
                frame_cache_wrist: dict[int, np.ndarray] = {}
                if is_video_format:
                    ep_idx = parquet_to_episode[pq_path]
                    front_path, wrist_path = episode_video_paths[ep_idx]
                    needed_rows = {row for _, row, _ in items}
                    frame_cache_front = _decode_video_frames(front_path, needed_rows)
                    frame_cache_wrist = _decode_video_frames(wrist_path, needed_rows)

                for idx, row, task_idx in items:
                    # ── images ──
                    if is_video_format:
                        front_frame = frame_cache_front.get(row)
                        wrist_frame = frame_cache_wrist.get(row)
                        if front_frame is None:
                            raise RuntimeError(f"Frame {row} not found in {front_path}")
                        if wrist_frame is None:
                            raise RuntimeError(f"Frame {row} not found in {wrist_path}")
                        img = _rotate_180(_resize_224(front_frame))
                        wrist_img = _rotate_180(_resize_224(wrist_frame))
                    else:
                        img = _process_image(df[col_map["image"]].iloc[row])
                        wrist_img = _process_image(df[col_map["wrist_image"]].iloc[row])
                    images.append(img)
                    wrist_images.append(wrist_img)

                    # ── state ──
                    state_val = df[col_map["state"]].iloc[row]
                    if isinstance(state_val, np.ndarray):
                        states.append(state_val.astype(np.float32))
                    else:
                        states.append(np.array(state_val, dtype=np.float32))

                    # ── action chunk ──
                    action_chunk = []
                    for ah in range(action_horizon):
                        arow = row + ah
                        a = df[col_map["actions"]].iloc[arow]
                        if isinstance(a, np.ndarray):
                            action_chunk.append(a[:action_dim].astype(np.float32))
                        else:
                            action_chunk.append(np.array(a[:action_dim], dtype=np.float32))
                    actions_list.append(np.stack(action_chunk, axis=0))

                    # ── prompt ──
                    prompts.append(tasks.get(task_idx, ""))

            # ── assemble batch ──
            actions_arr = np.stack(actions_list, axis=0).astype(np.float32)
            if effective_action_dim > action_dim:
                actions_arr = np.pad(
                    actions_arr,
                    ((0, 0), (0, 0), (0, effective_action_dim - action_dim)),
                    mode="constant",
                )

            batch = {
                "observation": {
                    "image": {
                        "base_0_rgb": np.stack(images, axis=0),
                        "left_wrist_0_rgb": np.stack(wrist_images, axis=0),
                        "right_wrist_0_rgb": np.zeros_like(np.stack(images, axis=0)),
                    },
                    "image_mask": {
                        "base_0_rgb": np.ones(batch_size, dtype=bool),
                        "left_wrist_0_rgb": np.ones(batch_size, dtype=bool),
                        "right_wrist_0_rgb": np.zeros(batch_size, dtype=bool),
                    },
                    "state": np.stack(states, axis=0).astype(np.float32),
                },
                "actions": actions_arr,
                "action_mean": action_mean,
                "action_std": action_std,
                "prompt": prompts,
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
    """Fake data loader for smoke testing."""
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
