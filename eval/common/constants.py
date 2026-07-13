"""eval/ 共享常量。"""

import pathlib

EVAL_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT_ROOT = EVAL_ROOT.parent
OPENPI_DIR = PROJECT_ROOT / "openpi"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
