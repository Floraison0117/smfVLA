#!/bin/bash
# SnapFlow training wrapper script

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi_server

# Set project root
PROJECT_ROOT="/root/autodl-tmp/snapflow"
export PROJECT_ROOT="${PROJECT_ROOT}"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/openpi/src:${PROJECT_ROOT}/third_party/openpi/packages/openpi-client/src"

# ── JAX 关键环境变量（必须在 python 启动 / import jax 前设置）──────────
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_MAX_SIZE=134217728
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Change to project root
cd "${PROJECT_ROOT}"

# Run training
python scripts/run_train.py "$@"
