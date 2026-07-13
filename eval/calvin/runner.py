"""CALVIN 评测 runner：环境创建、模型包装、rollout。"""

import collections
import logging
import os
import pathlib
import time

import numpy as np

from eval.calvin.utils import (
    get_calvin_validation_path,
    load_calvin_obs,
    setup_calvin_paths,
)
from eval.calvin.protocol import (
    count_success,
    get_env_state_for_initial_condition,
    get_sequences,
)

logger = logging.getLogger(__name__)

EP_LEN = 360


# ── pi0.5 (PyTorch) 加载 ──────────────────────────────────

def _load_calvin_policy(nfe: int, checkpoint_dir: str):
    """加载 pi05_calvin PyTorch checkpoint。"""
    from openpi.policies import policy_config
    from openpi.shared import normalize as _normalize
    from openpi.training import config as _config

    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    train_config = _config.get_config("pi05_libero")
    norm_stats = _normalize.load(checkpoint_dir)
    logger.info(f"Loaded norm_stats: {[(k, v.mean.shape) for k, v in norm_stats.items()]}")

    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        norm_stats=norm_stats,
        sample_kwargs={"num_steps": nfe},
    )
    logger.info("Policy loaded (PyTorch backend)")
    return policy


# ── pi0.5 policy -> CalvinBaseModel 接口 ──────────────────

class Pi05CalvinModel:
    """将 openpi Policy 包装成 CALVIN 协议接口（reset / step）。"""

    def __init__(self, policy, replan_steps: int = 5):
        self.policy = policy
        self.replan_steps = max(1, int(replan_steps))
        self.action_plan = collections.deque()
        self.latencies_ms = []

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
            if action_chunk.ndim == 3:
                action_chunk = action_chunk[0]
            for a in action_chunk[: self.replan_steps]:
                a = np.asarray(a, dtype=np.float64).copy()
                a[6] = 1.0 if float(a[6]) > 0 else -1.0
                self.action_plan.append(a)

        return self.action_plan.popleft()


# ── rollout / evaluate_sequence ───────────────────────────

def rollout(env, model, task_oracle, subtask, val_annotations):
    """对一个 subtask 跑最多 EP_LEN 步。"""
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for _ in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True
    return False


def evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations):
    """评测一条任务链（最多 5 个 subtask）。任一失败则停止。"""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    for subtask in eval_sequence:
        if rollout(env, model, task_oracle, subtask, val_annotations):
            success_counter += 1
        else:
            return success_counter
    return success_counter


# ── 环境创建 ──────────────────────────────────────────────

def make_env(val_path: pathlib.Path, use_egl: bool = True):
    """创建 CALVIN PlayTableSimEnv。"""
    import hydra
    from omegaconf import OmegaConf

    render_conf = OmegaConf.load(val_path / ".hydra" / "merged_config.yaml")
    if not use_egl:
        try:
            render_conf.env.use_egl = False
        except Exception:
            logger.warning("Cannot set use_egl=False in merged_config")
    try:
        if "tactile" in render_conf.cameras:
            del render_conf.cameras["tactile"]
    except Exception:
        pass
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(".")
    env = hydra.utils.instantiate(render_conf.env, show_gui=False, use_vr=False, use_scene_info=True)
    return env
