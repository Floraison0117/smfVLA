#!/bin/bash
# DMF training script for π₀.₅ VLA on LIBERO / LIBERO-Plus.
#
# Usage:
#   bash scripts/train.sh                                    # default config
#   bash scripts/train.sh configs/train/dmf_libero_plus.yaml  # specific config
#   bash scripts/train.sh configs/train/dmf_libero_plus.yaml --resume checkpoints/dmf_finetuned/step_10000

set -e

CONFIG_FILE=${1:-"configs/train/dmf_libero_plus.yaml"}
shift

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

PROJECT_ROOT="/root/autodl-tmp/dmf"
OPENPI_DIR="/root/autodl-tmp/openpi"

# Ensure GPU is visible to JAX before Python starts
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_MAX_SIZE=134217728
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50

export PYTHONPATH="${PROJECT_ROOT}/src:${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src"

echo "Activating conda environment: openpi_server"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi_server

cd "$PROJECT_ROOT"

echo "=========================================="
echo "DMF Training"
echo "=========================================="
echo "Config:   $CONFIG_FILE"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "JAX:      $(python -c 'import jax; print(jax.devices()[0])' 2>/dev/null || echo 'unknown')"
[ -n "$RESUME" ] && echo "Resume:   $RESUME"
echo "=========================================="

python scripts/run_train.py "$CONFIG_FILE" ${RESUME:+--resume "$RESUME"}

echo "Training completed!"
