"""eval/ 共享工具：路径设置、结果保存。"""

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

from .constants import OPENPI_DIR, PROJECT_ROOT, LIBERO_DUMMY_ACTION

logger = logging.getLogger(__name__)


def setup_paths():
    """添加评测所需的 sys.path 条目（幂等）。"""
    paths = [
        str(PROJECT_ROOT / "dmf" / "src"),
        str(OPENPI_DIR / "src"),
        str(OPENPI_DIR / "packages" / "openpi-client" / "src"),
    ]
    libero_tp = OPENPI_DIR / "third_party" / "libero"
    if libero_tp.exists() and any(libero_tp.iterdir()):
        paths.append(str(libero_tp))
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


def quat2axisangle(quat):
    """四元数 -> 轴角表示。"""
    quat = np.array(quat)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_result_json(config_dict, task_results, episode_details, all_latencies,
                      total_successes, total_episodes, start_time, end_time):
    """构建结构化结果 JSON。"""
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
