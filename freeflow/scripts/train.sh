#!/bin/bash
# FreeFlow training script
# Usage: bash scripts/train.sh configs/train/freeflow_base_libero.yaml [--resume checkpoint_path]

set -e

# Default config
CONFIG_FILE=${1:-"configs/train/freeflow_base_libero_plus.yaml"}
shift

# Parse optional arguments
RESUME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set paths
PROJECT_ROOT="/root/autodl-tmp/freeflow"
OPENPI_DIR="${PROJECT_ROOT}/third_party/openpi"
DATASET_DIR="${PROJECT_ROOT}/data"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src:${OPENPI_DIR}/third_party/libero"

# ── JAX 关键环境变量（必须在 python 启动 / import jax 前设置）──────────
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_MAX_SIZE=134217728
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Activate conda environment
echo "Activating conda environment: openpi_server"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi_server

# Change to project directory
cd "$PROJECT_ROOT"

echo "=========================================="
echo "FreeFlow Training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
if [ -n "$RESUME" ]; then
    echo "Resume from: $RESUME"
fi
echo "=========================================="

# Run training
python -m freeflow.training.run_train \
    --config "$CONFIG_FILE" \
    ${RESUME:+--resume "$RESUME"}

echo "Training completed!"
