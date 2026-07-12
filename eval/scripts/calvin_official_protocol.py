#!/usr/bin/env python3
"""
CALVIN 官方长程评测协议的纯函数 vendor（脱离 torch/pyhash 依赖）。

原始实现位于 calvin_agent.evaluation.{multistep_sequences,utils}，但那些模块顶层
`import torch` / `from calvin_agent.models.mcil import MCIL` / `import pyhash`，而 calvin_eval
环境（jax 评测用）没有 torch/pyhash。本模块把评测实际用到的纯 numpy 函数抽出来：

  - get_sequences(num_sequences)            —— 官方符号化任务链生成（多进程改单进程，结果等价）
  - get_env_state_for_initial_condition(ic) —— 符号初始条件 -> (robot_obs[15], scene_obs[24])
  - count_success(results)                  —— 成功计数 -> [SR1..SR5]

唯一与官方不完全一致处：get_env_state_for_initial_condition 里用来定块摆放随机的哈希，
官方用 pyhash.fnv1_32()，这里用纯 python FNV-1 32-bit 替代（仅影响方块位置的微随机，
所有初始条件仍合法，聚合 SR 统计上等价）。任务链生成逻辑、temp_seed(0) 均与官方一致。
"""

import contextlib
import functools
from copy import deepcopy
from itertools import product
from collections import Counter

import numpy as np
from numpy import pi

logger = logging = __import__("logging")


# ── 随机种子上下文（vendor 自 utils.temp_seed）─────────────
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


# ── 哈希（纯 python FNV-1 32-bit，替代 pyhash.fnv1_32）─────
def _fnv1_32(s):
    h = 2166136261  # FNV offset basis
    for b in s.encode("utf-8"):
        h = (h * 16777619) & 0xFFFFFFFF  # FNV prime
        h ^= b
    return h


hasher = _fnv1_32


# ── 成功计数 -> [SR1..SR5]（vendor 自 utils.count_success）──
def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


# ── 符号初始条件 -> (robot_obs, scene_obs)（vendor 自 utils）─
def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = hasher(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


# ── 任务链生成（vendor 自 multistep_sequences）────────────
task_categories = {
    "rotate_red_block_right": 1, "rotate_red_block_left": 1,
    "rotate_blue_block_right": 1, "rotate_blue_block_left": 1,
    "rotate_pink_block_right": 1, "rotate_pink_block_left": 1,
    "push_red_block_right": 1, "push_red_block_left": 1,
    "push_blue_block_right": 1, "push_blue_block_left": 1,
    "push_pink_block_right": 1, "push_pink_block_left": 1,
    "move_slider_left": 2, "move_slider_right": 2,
    "open_drawer": 3, "close_drawer": 3,
    "lift_red_block_table": 4, "lift_red_block_slider": 5, "lift_red_block_drawer": 6,
    "lift_blue_block_table": 4, "lift_blue_block_slider": 5, "lift_blue_block_drawer": 6,
    "lift_pink_block_table": 4, "lift_pink_block_slider": 5, "lift_pink_block_drawer": 6,
    "place_in_slider": 7, "place_in_drawer": 7,
    "turn_on_lightbulb": 8, "turn_off_lightbulb": 8,
    "turn_on_led": 8, "turn_off_led": 8,
    "push_into_drawer": 9,
    "stack_block": 10, "unstack_block": 11,
}

tasks = {
    "rotate_red_block_right": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "rotate_red_block_left": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "rotate_blue_block_right": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "rotate_blue_block_left": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "rotate_pink_block_right": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "rotate_pink_block_left": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "push_red_block_right": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "push_red_block_left": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "push_blue_block_right": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "push_blue_block_left": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "push_pink_block_right": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "push_pink_block_left": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "move_slider_left": [{"condition": {"slider": "right", "grasped": 0}, "effect": {"slider": "left"}}],
    "move_slider_right": [{"condition": {"slider": "left", "grasped": 0}, "effect": {"slider": "right"}}],
    "open_drawer": [{"condition": {"drawer": "closed", "grasped": 0}, "effect": {"drawer": "open"}}],
    "close_drawer": [{"condition": {"drawer": "open", "grasped": 0}, "effect": {"drawer": "closed"}}],
    "lift_red_block_table": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}}],
    "lift_red_block_slider": [
        {"condition": {"red_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}},
        {"condition": {"red_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}},
    ],
    "lift_red_block_drawer": [{"condition": {"red_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}}],
    "lift_blue_block_table": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}}],
    "lift_blue_block_slider": [
        {"condition": {"blue_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}},
        {"condition": {"blue_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}},
    ],
    "lift_blue_block_drawer": [{"condition": {"blue_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}}],
    "lift_pink_block_table": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}}],
    "lift_pink_block_slider": [
        {"condition": {"pink_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}},
        {"condition": {"pink_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}},
    ],
    "lift_pink_block_drawer": [{"condition": {"pink_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}}],
    "place_in_slider": [
        {"condition": {"red_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"red_block": "slider_right", "grasped": 0}},
        {"condition": {"red_block": "grasped", "slider": "left", "grasped": 1}, "effect": {"red_block": "slider_left", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"blue_block": "slider_right", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "slider": "left", "grasped": 1}, "effect": {"blue_block": "slider_left", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"pink_block": "slider_right", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "slider": "left", "grasped": 1}, "effect": {"pink_block": "slider_left", "grasped": 0}},
    ],
    "place_in_drawer": [
        {"condition": {"red_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"red_block": "drawer", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"blue_block": "drawer", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"pink_block": "drawer", "grasped": 0}},
    ],
    "stack_block": [
        {"condition": {"red_block": "grasped", "blue_block": "table", "grasped": 1}, "effect": {"red_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"red_block": "grasped", "pink_block": "table", "grasped": 1}, "effect": {"red_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "red_block": "table", "grasped": 1}, "effect": {"blue_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "pink_block": "table", "grasped": 1}, "effect": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "red_block": "table", "grasped": 1}, "effect": {"pink_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "blue_block": "table", "grasped": 1}, "effect": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}},
    ],
    "unstack_block": [
        {"condition": {"red_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}, "effect": {"red_block": "table", "blue_block": "table"}},
        {"condition": {"red_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}, "effect": {"red_block": "table", "pink_block": "table"}},
        {"condition": {"blue_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}, "effect": {"blue_block": "table", "red_block": "table"}},
        {"condition": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}, "effect": {"blue_block": "table", "pink_block": "table"}},
        {"condition": {"pink_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}, "effect": {"pink_block": "table", "red_block": "table"}},
        {"condition": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}, "effect": {"pink_block": "table", "blue_block": "table"}},
    ],
    "turn_on_lightbulb": [{"condition": {"lightbulb": 0, "grasped": 0}, "effect": {"lightbulb": 1}}],
    "turn_off_lightbulb": [{"condition": {"lightbulb": 1, "grasped": 0}, "effect": {"lightbulb": 0}}],
    "turn_on_led": [{"condition": {"led": 0, "grasped": 0}, "effect": {"led": 1}}],
    "turn_off_led": [{"condition": {"led": 1, "grasped": 0}, "effect": {"led": 0}}],
    "push_into_drawer": [
        {"condition": {"red_block": "table", "blue_block": ["slider_right", "slider_left"], "pink_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"red_block": "drawer", "grasped": 0}},
        {"condition": {"blue_block": "table", "red_block": ["slider_right", "slider_left"], "pink_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"blue_block": "drawer", "grasped": 0}},
        {"condition": {"pink_block": "table", "blue_block": ["slider_right", "slider_left"], "red_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"pink_block": "drawer", "grasped": 0}},
    ],
}


def check_condition(state, condition):
    for k, v in condition.items():
        if isinstance(v, (str, int)):
            if not state[k] == v:
                return False
        elif isinstance(v, list):
            if not state[k] in v:
                return False
        else:
            raise TypeError
    return True


def update_state(state, effect):
    next_state = deepcopy(state)
    for k, v in effect.items():
        next_state[k] = v
    return next_state


def valid_task(curr_state, task):
    next_states = []
    for _task in task:
        if check_condition(curr_state, _task["condition"]):
            next_state = update_state(curr_state, _task["effect"])
            next_states.append(next_state)
    return next_states


def check_sequence(state, seq):
    for task_name in seq:
        states = valid_task(state, tasks[task_name])
        if len(states) != 1:
            return False
        state = states[0]
    categories = [task_categories[name] for name in seq]
    return len(categories) == len(set(categories))


def _get_sequences_for_state(state, num_sequences, i):
    """单进程版 worker（替代官方 get_sequences_for_state2 + ProcessPoolExecutor）。"""
    np.random.seed(i)
    seq_len = 5
    results = []
    task_names = list(tasks.keys())
    while len(results) < num_sequences:
        seq = np.random.choice(task_names, size=seq_len, replace=False)
        if check_sequence(state, seq):
            results.append(seq)
    return results


@functools.lru_cache(maxsize=None)
def get_sequences(num_sequences=1000):
    """
    生成 num_sequences 条 CALVIN 评测序列。
    返回 [(initial_condition_dict, (task1,...,task5)), ...]。
    与官方一致：符号化初始条件 + temp_seed(0) 外层洗牌（多进程改单进程，结果等价）。
    """
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }
    f = lambda l: l.count("table") in [1, 2] and l.count("slider_right") < 2 and l.count("slider_left") < 2
    value_combinations = filter(f, product(*possible_conditions.values()))
    initial_states = [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]

    num_sequences_per_state = list(map(len, np.array_split(range(num_sequences), len(initial_states))))
    logger.info("Start generating evaluation sequences (single-process vendor).")
    with temp_seed(0):
        results = []
        for i, (state, n) in enumerate(zip(initial_states, num_sequences_per_state)):
            for seq in _get_sequences_for_state(state, n, i):
                results.append((state, tuple(seq.tolist())))
        np.random.shuffle(results)
    logger.info(f"Done generating {len(results)} evaluation sequences.")
    return results
