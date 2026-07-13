#!/usr/bin/env python3
"""CALVIN 评测入口（pi0.5 PyTorch，官方 ABCD->D 协议）。

用法:
    python -m eval.calvin.main --model-type pi05 --nfe 1 --mode quick
    python -m eval.calvin.main --model-type pi05 --nfe 10 --mode normal
    python -m eval.calvin.main --model-type pi05 --nfe 10 --mode fullset
"""

import argparse
import logging
import os
import pathlib
import time

import numpy as np
from omegaconf import OmegaConf

from eval.common import setup_paths
from eval.common.utils import build_result_json, save_result_json

# CALVIN 路径（必须在 import calvin_agent 等之前）
from eval.calvin.utils import (
    get_calvin_validation_path,
    setup_calvin_paths,
)
from eval.calvin.runner import (
    EP_LEN,
    Pi05CalvinModel,
    _load_calvin_policy,
    evaluate_sequence,
    make_env,
)
from eval.calvin.protocol import count_success, get_sequences

setup_calvin_paths()
setup_paths()

import calvin_agent

CALVIN_MODELS_ROOT = pathlib.Path(calvin_agent.__file__).resolve().parents[1]
CONF_DIR = CALVIN_MODELS_ROOT / "conf"

# headless rendering
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

PRESETS = {
    "quick": {"dataset": "debug", "num_sequences": 5},
    "normal": {"dataset": "ABCD", "num_sequences": 100},
    "fullset": {"dataset": "ABCD", "num_sequences": 1000},
}


def main():
    parser = argparse.ArgumentParser(description="CALVIN benchmark eval (pi0.5, any NFE)")
    parser.add_argument("--model-type", type=str, default="pi05", choices=["pi05"])
    parser.add_argument("--nfe", type=int, default=1, choices=[1, 2, 4, 10])
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "normal", "fullset"])
    parser.add_argument("--checkpoint", type=str,
                        default="/root/autodl-tmp/checkpoints/pi05_calvin_pt")
    parser.add_argument("--num-sequences", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-egl", action="store_true", default=True)
    parser.add_argument("--no-use-egl", dest="use_egl", action="store_false")
    parser.add_argument("--results-dir", type=str,
                        default="/root/autodl-tmp/eval/results/calvin")
    args = parser.parse_args()

    # 应用 preset
    preset = PRESETS[args.mode]
    dataset = preset["dataset"]
    if args.num_sequences is None:
        args.num_sequences = preset["num_sequences"]

    np.random.seed(args.seed)
    start_time = time.time()

    logger.info(f"CALVIN eval | model={args.model_type} | mode={args.mode} "
                f"| dataset={dataset} | nfe={args.nfe} | seqs={args.num_sequences}")

    # 加载 policy
    policy = _load_calvin_policy(args.nfe, args.checkpoint)
    model = Pi05CalvinModel(policy, replan_steps=args.replan_steps)

    # 环境
    val_path = get_calvin_validation_path(dataset)
    if not (val_path / ".hydra" / "merged_config.yaml").exists():
        raise FileNotFoundError(
            f"Missing scene config: {val_path}/.hydra/merged_config.yaml"
        )
    logger.info(f"Creating CALVIN env (dataset={dataset}, use_egl={args.use_egl})")
    env = make_env(val_path, use_egl=args.use_egl)

    # task_oracle + 语言标注
    import hydra
    task_cfg = OmegaConf.load(CONF_DIR / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(CONF_DIR / "annotations/new_playtable_validation.yaml")

    # 评测序列
    logger.info(f"Generating {args.num_sequences} eval sequences...")
    eval_sequences = get_sequences(args.num_sequences)

    # 主循环
    results = []
    per_sequence = []
    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        r = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, val_annotations
        )
        results.append(r)
        per_sequence.append({
            "seq_idx": i, "success_count": int(r), "chain": list(eval_sequence),
        })
        chain_sr = count_success(results)
        desc = " ".join(f"{k + 1}/5:{v * 100:.1f}%" for k, v in enumerate(chain_sr))
        logger.info(f"[{i + 1}/{len(eval_sequences)}] chain_success={r}/5 | {desc}")

    # 汇总
    end_time = time.time()
    chain_sr = count_success(results)
    avg_seq_len = float(np.mean(results)) if results else 0.0
    sr1_successes = sum(1 for r in results if r >= 1)
    total_sequences = len(results)

    logger.info("=" * 60)
    logger.info(f"CALVIN {dataset}->D | NFE={args.nfe} | {total_sequences} sequences")
    logger.info(f"  Avg chain len: {avg_seq_len:.3f}")
    for k, sr in enumerate(chain_sr):
        logger.info(f"  SR{k + 1}: {sr * 100:.1f}%")
    logger.info("=" * 60)

    # 保存结果
    config_dict = {
        "benchmark": "calvin",
        "task_suite": f"calvin_{dataset}",
        "dataset": dataset,
        "nfe": args.nfe,
        "model_type": args.model_type,
        "checkpoint": str(args.checkpoint),
        "num_sequences": total_sequences,
        "replan_steps": args.replan_steps,
        "ep_len_per_subtask": EP_LEN,
        "seed": args.seed,
        "use_egl": args.use_egl,
    }
    result = build_result_json(
        config_dict,
        task_results={},
        episode_details=per_sequence,
        all_latencies=list(model.latencies_ms),
        total_successes=sr1_successes,
        total_episodes=total_sequences,
        start_time=start_time,
        end_time=end_time,
    )
    result["calvin"] = {
        "avg_successful_seq_len": round(avg_seq_len, 4),
        "chain_sr": {str(k + 1): round(sr, 4) for k, sr in enumerate(chain_sr)},
        "sr1": round(chain_sr[0], 4),
        "sr5": round(chain_sr[4], 4),
    }

    filepath = save_result_json(result, args.results_dir, f"calvin_{dataset}")
    logger.info(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
