#!/bin/bash
# 运行 libero 评估（client 端）
# 用法: bash scripts/eval.sh [task_suite_name] [options]
#
# 示例:
#   bash scripts/eval.sh                                          # 使用默认配置
#   bash scripts/eval.sh libero_spatial                           # 指定 task suite
#   bash scripts/eval.sh libero_spatial --num-trials-per-task 10  # 指定 trial 数
#
# 注意: 需要先启动 server (bash scripts/serve_policy.sh)

set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$PROJECT_ROOT/third_party/openpi"
CONDA_ENV="openpi_server"

# ── 默认参数 ──────────────────────────────────────────────
TASK_SUITE="${1:-libero_spatial}"
shift 2>/dev/null || true

# ── 验证路径 ──────────────────────────────────────────────
if [ ! -d "$OPENPI_DIR/examples/libero" ]; then
    echo "错误: openpi libero 示例不存在: $OPENPI_DIR/examples/libero"
    exit 1
fi

# ── 环境变量 ──────────────────────────────────────────────
export PYTHONPATH="$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src:${PYTHONPATH:-}"

# MuJoCo 渲染配置（headless）
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ── 输出目录 ──────────────────────────────────────────────
VIDEO_DIR="$PROJECT_ROOT/results/videos/$TASK_SUITE"
mkdir -p "$VIDEO_DIR"

# ── 运行评估 ──────────────────────────────────────────────
echo "============================================"
echo " smfVLA LIBERO Evaluation"
echo "============================================"
echo "Task suite:  $TASK_SUITE"
echo "Video dir:   $VIDEO_DIR"
echo "Server:      localhost:8000"
echo "============================================"
echo ""

exec /root/miniconda3/envs/$CONDA_ENV/bin/python "$OPENPI_DIR/examples/libero/main.py" \
    --task-suite-name "$TASK_SUITE" \
    --video-out-path "$VIDEO_DIR" \
    --host "localhost" \
    --port 8000 \
    "$@"
