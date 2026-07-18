#!/bin/bash
# SMF-Base 训练启动脚本
# 用法: bash scripts/train.sh [config_path]
#
# 示例:
#   bash scripts/train.sh                              # 使用默认配置
#   bash scripts/train.sh configs/train/smf_base_libero.yaml

set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$PROJECT_ROOT/third_party/openpi"
CONFIG_PATH="${1:-$PROJECT_ROOT/configs/train/smf_base_libero_plus.yaml}"
CONDA_ENV="openpi_server"

# ── 验证路径 ──────────────────────────────────────────────
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

if [ ! -d "$OPENPI_DIR/src" ]; then
    echo "错误: openpi 目录不存在: $OPENPI_DIR"
    exit 1
fi

# ── 环境变量 ──────────────────────────────────────────────
export PYTHONPATH="$PROJECT_ROOT/src:$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src:${PYTHONPATH:-}"
# JAX 编译缓存（避免每次重启重新 JIT 编译）
export JAX_COMPILATION_CACHE_DIR="$PROJECT_ROOT/.jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
# ── JAX 关键环境变量（必须在 python 启动 / import jax 前设置）──────────
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_MAX_SIZE=134217728
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# WANDB_API_KEY: 从环境变量读取，如果未设置则提示用户
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "警告: WANDB_API_KEY 未设置，WandB 记录将被跳过"
    echo "  设置方法: export WANDB_API_KEY=your_key_here"
fi

# ── GPU 检查 ──────────────────────────────────────────────
echo "============================================"
echo " SMF-Base Training"
echo "============================================"
echo "Config:     $CONFIG_PATH"
echo "OpenPI dir: $OPENPI_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "============================================"

echo ""
echo "[GPU 检查]"
/root/miniconda3/envs/$CONDA_ENV/bin/python -c "
import jax
devices = jax.devices()
print(f'JAX backend: {jax.default_backend()}')
print(f'JAX devices: {devices}')
for d in devices:
    assert 'gpu' in str(d).lower() or 'cuda' in str(d).lower(), \
        f'错误: 设备不是 GPU: {d}'
print('✓ GPU 验证通过')
print(f'GPU 内存: {devices[0].memory_stats() if hasattr(devices[0], \"memory_stats\") else \"N/A\"}')
"

# ── 启动训练 ──────────────────────────────────────────────
echo ""
echo "[启动训练...]"
echo "配置文件: $CONFIG_PATH"
echo ""

export PROJECT_ROOT
/root/miniconda3/envs/$CONDA_ENV/bin/python "$PROJECT_ROOT/scripts/run_train.py" "$CONFIG_PATH"
