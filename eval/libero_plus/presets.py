"""LIBERO-Plus 评测 presets：quick / normal / fullset。"""

PRESETS = {
    "quick": {
        "suites": ["libero_spatial"],
        "max_tasks": 10,
        "num_episodes": 5,
        "description": "快速验证：libero_spatial, 10 tasks x 5 episodes（~30min）",
    },
    "normal": {
        "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        "max_tasks": None,  # 由采样逻辑决定
        "num_episodes": 5,
        "use_sampling": True,
        "tasks_per_category": 12,
        "description": "日常实验：4 suites, 7 扰动维度均匀采样, 5 ep/task（<10h）",
    },
    "fullset": {
        "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        "max_tasks": None,
        "num_episodes": 50,
        "description": "正式 benchmark：全部 5 suites, 所有 tasks, 50 ep/task",
    },
}

SAFE_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
