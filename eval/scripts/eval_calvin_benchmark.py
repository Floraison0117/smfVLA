#!/usr/bin/env python3
"""
CALVIN ABCD→D 官方长程 benchmark 评测（pi0.5，任意 NFE）。

实现 CALVIN 官方协议（task 链 1→5、task_oracle 成功判定、报 SR1..SR5 + 平均链长），
用我们自己的 load_policy 加载 pi0.5（original 分支），把 policy 包装成
CalvinBaseModel 接口（reset/step），喂给镜像自官方 evaluate_policy.py 的 rollout。

关键事实（已核实）：
- 官方协议不从 validation episodes 取初始状态，而是 get_sequences()（符号化初始条件）
  + get_env_state_for_initial_condition()。不需要 episode_*.npz。
- get_env 只读 dataset_path/.hydra/merged_config.yaml（已从 debug 数据集复制，是 calvin_scene_D）。
- 每个 subtask 最多 EP_LEN=360 步；任务链在首次失败时停止（官方）。

用法（在 calvin_eval 环境下）:
    # smoke (3 条序列, NFE=10)
    python eval_calvin_benchmark.py --dataset ABCD --nfe 10 \\
        --checkpoint /root/autodl-tmp/checkpoints/pi05_calvin_corrected --num-sequences 3
    # 正式
    python eval_calvin_benchmark.py --dataset ABCD --nfe 10 \\
        --checkpoint /root/autodl-tmp/checkpoints/pi05_calvin_corrected --num-sequences 100
"""

import argparse
import collections
import logging
import os
import pathlib
import sys
import time

import numpy as np

# ── 路径设置 ──────────────────────────────────────────────
from calvin_utils import (
    get_calvin_validation_path,
    load_calvin_obs,
    setup_calvin_paths,
)

setup_calvin_paths()

from eval_utils import (
    build_result_json,
    load_policy,
    save_result_json,
    setup_paths,
)

setup_paths()

import hydra
import jax
from omegaconf import OmegaConf

# CALVIN 环境（pybullet，无 torch 依赖）
from calvin_env.envs.play_table_env import get_env  # noqa: E402
# 官方协议的纯函数（vendor，避开 calvin_agent.evaluation 顶层 torch/pyhash import）
from calvin_official_protocol import (  # noqa: E402
    count_success,
    get_env_state_for_initial_condition,
    get_sequences,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# 官方协议常量
EP_LEN = 360  # 每个 subtask 的最大 rollout 步数

# calvin_models 根（conf/ 在其下）
import calvin_agent  # noqa: E402

CALVIN_MODELS_ROOT = pathlib.Path(calvin_agent.__file__).resolve().parents[1]
CONF_DIR = CALVIN_MODELS_ROOT / "conf"


# ── pi0.5 (pi05_calvin) 加载：修正 checkpoint 的两个格式差异 ─────
def _load_calvin_policy(nfe: int, checkpoint_dir: str):
    """
    加载 pi05_calvin（original pi0.5），修正其与标准 openpi checkpoint 的两点差异：
      1) action-expert 头 (action_in/out_proj, time_mlp_in/out) 多包了一层 `projection_params`，
         标准格式在根 → 提升到根。
      2) 无 assets/ 目录，norm_stats.json 在 checkpoint 根 → 直接用 _normalize.load 读取。
    其余完全复刻 create_trained_policy。
    """
    import jax.numpy as jnp  # noqa: F401
    from openpi import transforms as _transforms
    from openpi.models import model as _model
    from openpi.policies import policy as _policy
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    train_config = _config.get_config("pi05_libero")

    # 1) 加载并修正 projection_params 嵌套
    params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
    if "projection_params" in params:
        logger.info("检测到 projection_params 嵌套层，提升 action-expert 头到根")
        merged = {k: v for k, v in params.items() if k != "projection_params"}
        merged.update(params["projection_params"])
        params = merged

    model = train_config.model.load(params)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    # 2) norm_stats 从 checkpoint 根的 norm_stats.json 读
    norm_stats = _normalize.load(checkpoint_dir)
    logger.info(f"Loaded norm_stats: {[(k, v.mean.shape) for k, v in norm_stats.items()]}")

    return _policy.Policy(
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


# ── pi0.5 policy 包装成 CalvinBaseModel 接口 ─────────────
class Pi05CalvinModel:
    """
    把 openpi 的 Policy 包装成 CALVIN 协议要求的 model 接口：
        reset()                 —— 每个 subtask 开始前调用，清空动作缓存
        step(obs, lang) -> action  —— 每步调用，返回 7 维 CALVIN delta 动作

    内部用动作分块（action chunking）：缓存耗尽时重新推理，replan 间隔 = replan_steps。
    """

    def __init__(self, policy, replan_steps: int = 5):
        self.policy = policy
        self.replan_steps = max(1, int(replan_steps))
        self.action_plan = collections.deque()
        self.latencies_ms = []  # 记录每次 policy.infer 耗时

    def reset(self):
        self.action_plan.clear()

    def step(self, obs, goal):
        if not self.action_plan:
            img, wrist_img, state = load_calvin_obs(obs)
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": str(goal),
            }
            t0 = time.monotonic()
            result = self.policy.infer(element)
            self.latencies_ms.append((time.monotonic() - t0) * 1000.0)

            action_chunk = np.asarray(result["actions"])
            if action_chunk.ndim == 3:  # (1, H, D) -> (H, D)
                action_chunk = action_chunk[0]
            # 只缓存 replan_steps 步，耗尽后重新推理
            for a in action_chunk[: self.replan_steps]:
                a = np.asarray(a, dtype=np.float64).copy()
                # CALVIN 环境要求 gripper 动作为离散 ±1（apply_action 会断言），按符号二值化
                a[6] = 1.0 if float(a[6]) > 0 else -1.0
                self.action_plan.append(a)

        return self.action_plan.popleft()


# ── 镜像官方 evaluate_policy.py 的 rollout / evaluate_sequence ──
def rollout(env, model, task_oracle, subtask, val_annotations):
    """对一个 subtask（单条自然语言指令）跑最多 EP_LEN 步，用 task_oracle 判定成功。"""
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for _ in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        # 当前步是否完成了该 subtask
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True
    return False


def evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations):
    """
    评测一条任务链（最多 5 个 subtask）。返回成功完成的 subtask 数（0..5）。
    官方协议：任一 subtask 失败则整条链停止。
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    for subtask in eval_sequence:
        if rollout(env, model, task_oracle, subtask, val_annotations):
            success_counter += 1
        else:
            return success_counter
    return success_counter


def make_env(val_path: pathlib.Path, use_egl: bool = True):
    """
    自包含版的 get_env：读 val_path/.hydra/merged_config.yaml，实例化 PlayTableSimEnv。
    比 CALVIN 自带 get_env 多了对 use_egl 的控制（EGL 渲染不稳时可关掉走软件渲染），
    并删除触觉相机（tacto 库版本不兼容，且 pi0.5 评测只用 rgb_static/rgb_gripper）。
    """
    render_conf = OmegaConf.load(val_path / ".hydra" / "merged_config.yaml")
    if not use_egl:
        try:
            render_conf.env.use_egl = False
        except Exception:
            logger.warning("无法在 merged_config 中设置 use_egl=False，沿用配置默认值")
    # 删除触觉相机（与官方 get_env 的 obs_space 排除机制等价）
    try:
        if "tactile" in render_conf.cameras:
            del render_conf.cameras["tactile"]
            logger.info("已移除 tactile 相机（pi0.5 评测不需要，且 tacto 库版本不兼容）")
    except Exception as e:
        logger.warning(f"移除 tactile 相机失败: {e}")
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(
        render_conf.env, show_gui=False, use_vr=False, use_scene_info=True
    )
    return env


def run_eval(args):
    np.random.seed(args.seed)
    start_time = time.time()

    # ── headless 渲染环境变量 ──
    os.environ.setdefault("DISPLAY", "")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    logger.info(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")

    # ── 加载 pi0.5 policy（original 分支，NFE = num_steps）──
    policy = _load_calvin_policy(args.nfe, args.checkpoint)
    model = Pi05CalvinModel(policy, replan_steps=args.replan_steps)

    # ── 环境 ──
    val_path = get_calvin_validation_path(args.dataset)
    if not (val_path / ".hydra" / "merged_config.yaml").exists():
        raise FileNotFoundError(
            f"缺少场景配置: {val_path}/.hydra/merged_config.yaml。"
            f"请从 debug 数据集复制 .hydra 目录。"
        )
    logger.info(f"实例化 CALVIN env (dataset={args.dataset}, val_path={val_path}, use_egl={args.use_egl})")
    env = make_env(val_path, use_egl=args.use_egl)

    # ── task_oracle + 语言标注 ──
    task_cfg = OmegaConf.load(CONF_DIR / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(CONF_DIR / "annotations/new_playtable_validation.yaml")

    # ── 评测序列（官方符号化生成）──
    logger.info(f"生成 {args.num_sequences} 条评测序列 ...")
    eval_sequences = get_sequences(args.num_sequences)

    # ── 主循环 ──
    results = []
    per_sequence = []
    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        r = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, val_annotations
        )
        results.append(r)
        per_sequence.append(
            {
                "seq_idx": i,
                "success_count": int(r),
                "chain": list(eval_sequence),
            }
        )
        chain_sr = count_success(results)
        desc = " ".join(f"{k + 1}/5:{v * 100:.1f}% |" for k, v in enumerate(chain_sr))
        logger.info(f"[{i + 1}/{len(eval_sequences)}] chain_success={r}/5 | {desc}")

    # ── 汇总指标 ──
    end_time = time.time()
    chain_sr = count_success(results)  # [SR1..SR5]
    avg_seq_len = float(np.mean(results)) if results else 0.0
    sr1_successes = sum(1 for r in results if r >= 1)
    total_sequences = len(results)

    logger.info("=" * 60)
    logger.info(f"CALVIN ABCD→D | NFE={args.nfe} | {total_sequences} sequences | replan={args.replan_steps}")
    logger.info(f"  平均成功链长: {avg_seq_len:.3f}")
    for k, sr in enumerate(chain_sr):
        logger.info(f"  SR{k + 1}: {sr * 100:.1f}%")
    logger.info("=" * 60)

    # ── 保存结果（复用 build_result_json / save_result_json）──
    config_dict = {
        "benchmark": "calvin_ABCD_D",
        "task_suite": f"calvin_{args.dataset}",  # 供 save_result_json 命名用
        "dataset": args.dataset,
        "nfe": args.nfe,
        "model_type": "original",
        "checkpoint": str(args.checkpoint),
        "num_sequences": total_sequences,
        "replan_steps": args.replan_steps,
        "ep_len_per_subtask": EP_LEN,
        "seed": args.seed,
        "use_egl": args.use_egl,
    }
    all_latencies = list(model.latencies_ms)
    result = build_result_json(
        config_dict,
        task_results={},
        episode_details=per_sequence,
        all_latencies=all_latencies,
        total_successes=sr1_successes,
        total_episodes=total_sequences,
        start_time=start_time,
        end_time=end_time,
    )
    # CALVIN 专属指标
    result["calvin"] = {
        "avg_successful_seq_len": round(avg_seq_len, 4),
        "chain_sr": {str(k + 1): round(sr, 4) for k, sr in enumerate(chain_sr)},
        "sr1": round(chain_sr[0], 4),
        "sr5": round(chain_sr[4], 4),
    }

    results_dir = pathlib.Path(args.results_dir)
    filepath = save_result_json(result, str(results_dir), f"calvin_{args.dataset}")
    logger.info(f"Results saved to: {filepath}")
    return result, filepath


def main():
    parser = argparse.ArgumentParser(description="CALVIN ABCD→D benchmark eval (pi0.5, any NFE)")
    parser.add_argument("--dataset", type=str, default="ABCD", choices=["debug", "D", "ABC", "ABCD"])
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/autodl-tmp/checkpoints/pi05_calvin_corrected",
        help="pi0.5 checkpoint 目录（含 params/ + norm_stats.json）",
    )
    parser.add_argument("--nfe", type=int, default=10, choices=[1, 2, 4, 10])
    parser.add_argument("--num-sequences", type=int, default=100, help="评测序列数（官方=1000）")
    parser.add_argument("--replan-steps", type=int, default=5, help="动作分块重规划间隔")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-egl", action=argparse.BooleanOptionalAction, default=True,
                        help="--use-egl / --no-use-egl（EGL 渲染不稳时关闭走软件渲染）")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/root/autodl-tmp/eval/results/calvin",
        help="结果 JSON 保存目录",
    )
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
