#!/usr/bin/env python3
"""
CALVIN episode npz → LeRobot v2.0 parquet 转换器（供 DMF data_loader 读）。

输入：一个 CALVIN 数据目录（已解压），含：
  - episode_XXXXXXXX.npz（每帧一个；keys: rgb_static[200,200,3], rgb_gripper[84,84,3],
    robot_obs[15], rel_actions[7]）
  - lang_annotations/auto_lang_ann.npy（语言标注 + info.indx 全局帧范围）
  - ep_start_end_ids.npy（episode 边界 [N,2] 全局帧索引）
输出：LeRobot v2.0 格式（data_loader.py 直接读）：
  - data/chunk-000/episode_NNNNNN.parquet（列 image/wrist_image/state/actions）
  - meta/{tasks.jsonl, episodes.jsonl}
  - norm_stats.json（从 checkpoints/pi05_calvin_corrected 拷贝，7维 CALVIN）

约定对齐 eval (calvin_utils.load_calvin_obs)：
  - state = robot_obs[0:6]（tcp_pos+euler）+ binarize(robot_obs[14])（gripper ACTION ±1）→ 7维
  - actions = rel_actions → 7维
  - 图像预旋转 180° 后存入 parquet（data_loader 加载时会再旋转 180° → 复原为 upright，
    与 eval 不旋转一致）。resize 到 256×256。

用法:
    python convert_calvin_data_to_lerobot.py \\
        --calvin_dir datasets/calvin/calvin/dataset/calvin_debug_dataset/training \\
        --out_dir datasets/calvin_lerobot_debug
    # 真实训练（先解压 subset）:
    #   unzip datasets/calvin_D-D/training/subset_training_011.zip -d /tmp/c011
    #   python convert_calvin_data_to_lerobot.py --calvin_dir /tmp/c011/subset_training_011/training \\
        --out_dir datasets/calvin_lerobot
"""
import argparse
import io
import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def _encode_img(arr_hwc_uint8, size=256):
    """resize 到 size×size，预旋转 180°，编码 PNG bytes。"""
    img = Image.fromarray(arr_hwc_uint8).resize((size, size), Image.LANCZOS)
    img = img.rotate(180)  # == arr[::-1,::-1]；data_loader 加载时会再 rotate180 复原
    buf = io.BytesIO()
    img.save(buf, format="png")
    return buf.getvalue()


def _binarize_gripper(robot_obs):
    """7维 state = robot_obs[0:6] + binarize(robot_obs[14])（gripper ACTION ±1）。"""
    gripper = 1.0 if robot_obs[14] > 0 else -1.0
    return np.concatenate([robot_obs[0:6], [gripper]]).astype(np.float32)


def load_language_map(calvin_dir):
    """返回 {global_frame_idx: language_str}（carry-forward：未标注帧用最近前一条）。"""
    ann_path = pathlib.Path(calvin_dir) / "lang_annotations" / "auto_lang_ann.npy"
    if not ann_path.exists():
        return {}
    ann = np.load(ann_path, allow_pickle=True).item()
    langs = ann["language"]["ann"]
    idxs = ann["info"]["indx"]
    # 按段填充；段内每帧用该语言。段外帧 carry-forward 上一段语言。
    segs = sorted(zip(idxs, langs), key=lambda x: x[0][0])
    frame_to_lang = {}
    carry = ""
    for (s, e), lang in segs:
        if s > 0 and carry:
            pass  # carry 已是上一段
        for f in range(int(s), int(e)):
            frame_to_lang[f] = lang
        carry = lang
    return frame_to_lang, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calvin_dir", required=True, help="CALVIN 训练目录（episode_*.npz + lang_annotations）")
    ap.add_argument("--out_dir", required=True, help="输出 LeRobot 目录")
    ap.add_argument("--norm_stats", default="/root/autodl-tmp/checkpoints/pi05_calvin_corrected/norm_stats.json")
    ap.add_argument("--chunk_size", type=int, default=500,
                    help="把帧切成固定大小的 pseudo-episode（每个 parquet 小，data_loader 读秒级）。"
                         "CALVIN 原始 episode 可达数万帧 → 单 parquet 数 GB，pd.read_parquet 全读会卡 CPU。")
    args = ap.parse_args()

    calvin_dir = pathlib.Path(args.calvin_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_data = out_dir / "data" / "chunk-000"
    out_meta = out_dir / "meta"
    out_data.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    # 收集所有实际存在的帧（按全局索引排序）
    import re as _re
    present = sorted(
        int(m.group(1))
        for m in (_re.match(r"episode_(\d+)\.npz", p.name) for p in calvin_dir.glob("episode_*.npz"))
        if m
    )
    print(f"present frames: {len(present)} (range {present[0]}-{present[-1]})")

    # 语言
    frame_to_lang, segs = load_language_map(calvin_dir)
    task_to_idx = {}
    tasks_rows = []
    def _task_idx(lang):
        if lang not in task_to_idx:
            task_to_idx[lang] = len(task_to_idx)
            tasks_rows.append({"task_index": task_to_idx[lang], "task": lang})
        return task_to_idx[lang]
    for (_, _), lang in segs:
        _task_idx(lang)
    if not task_to_idx:
        _task_idx("")
    print(f"tasks: {len(task_to_idx)}, language segments: {len(segs)}")

    # 切成 chunk_size 的 pseudo-episode
    chunks = [present[i : i + args.chunk_size] for i in range(0, len(present), args.chunk_size)]
    print(f"chunks (episode): {len(chunks)} @ {args.chunk_size} frames each")

    episodes_rows = []
    total_frames = 0
    for ep_idx, chunk in enumerate(chunks):
        rows = []
        for f in chunk:
            npz_path = calvin_dir / f"episode_{f:07d}.npz"
            d = np.load(npz_path)
            rows.append({
                "image": _encode_img(d["rgb_static"]),
                "wrist_image": _encode_img(d["rgb_gripper"]),
                "state": _binarize_gripper(d["robot_obs"]),
                "actions": np.asarray(d["rel_actions"], dtype=np.float32),
            })
        df = pd.DataFrame(rows)
        df.to_parquet(out_data / f"episode_{ep_idx:06d}.parquet", index=False)
        ep_lang = frame_to_lang.get(chunk[0], "")
        episodes_rows.append({"episode_index": ep_idx, "length": len(rows), "task_index": _task_idx(ep_lang)})
        total_frames += len(rows)
        if (ep_idx + 1) % 50 == 0 or ep_idx == len(chunks) - 1:
            print(f"  episode {ep_idx}: {len(rows)} frames (累计 {total_frames})")

    with open(out_meta / "tasks.jsonl", "w") as f:
        for r in tasks_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_meta / "episodes.jsonl", "w") as f:
        for r in episodes_rows:
            f.write(json.dumps(r) + "\n")

    import shutil
    if pathlib.Path(args.norm_stats).exists():
        shutil.copy(args.norm_stats, out_dir / "norm_stats.json")
        print(f"copied norm_stats → {out_dir / 'norm_stats.json'}")

    print(f"\nDONE: {out_dir} | {len(episodes_rows)} episodes, {total_frames} frames")


if __name__ == "__main__":
    main()
