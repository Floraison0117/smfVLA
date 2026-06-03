#!/bin/bash
# 启动 smfVLA policy server (WebSocket)
# 用法: bash scripts/serve_policy.sh [checkpoint_dir] [port]
#
# 示例:
#   bash scripts/serve_policy.sh                              # 使用默认 checkpoint 和端口
#   bash scripts/serve_policy.sh checkpoints/finetuned/1nfe/  # 使用微调后的 checkpoint
#   bash scripts/serve_policy.sh checkpoints/base/pi05_libero 8001

set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$PROJECT_ROOT/third_party/openpi"

# ── 默认参数 ──────────────────────────────────────────────
CHECKPOINT_DIR="${1:-$PROJECT_ROOT/checkpoints/base/pi05_libero}"
PORT="${2:-8000}"
CONFIG_NAME="pi05_libero"
CONDA_ENV="openpi_server"

# ── 验证路径 ──────────────────────────────────────────────
if [ ! -d "$CHECKPOINT_DIR/params" ]; then
    echo "错误: checkpoint 目录不存在或缺少 params/: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -d "$OPENPI_DIR/src" ]; then
    echo "错误: openpi 目录不存在: $OPENPI_DIR"
    echo "请确保 third_party/openpi 已正确设置"
    exit 1
fi

# ── 环境变量 ──────────────────────────────────────────────
export PYTHONPATH="$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src:$OPENPI_DIR/third_party/libero:${PYTHONPATH:-}"

# ── 启动 server ──────────────────────────────────────────
echo "============================================"
echo " smfVLA Policy Server"
echo "============================================"
echo "Config:      $CONFIG_NAME"
echo "Checkpoint:  $CHECKPOINT_DIR"
echo "Port:        $PORT"
echo "OpenPI dir:  $OPENPI_DIR"
echo "PYTHONPATH:  $PYTHONPATH"
echo "============================================"

# 验证 GPU
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
"

echo ""
echo "[启动 server...]"
exec /root/miniconda3/envs/$CONDA_ENV/bin/python "$OPENPI_DIR/scripts/serve_policy.py" \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$CONFIG_NAME" \
    --policy.dir "$CHECKPOINT_DIR"
