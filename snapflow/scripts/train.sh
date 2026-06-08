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

# ── JAX 内存优化 ─────────────────────────────────────────────────
# 限制 JAX 编译缓存大小，防止 RAM 占用过高
# snapflow/run_train.py 中已设置 jax_compilation_cache_max_size

# Change to project root
cd "${PROJECT_ROOT}"

# Run training
python scripts/run_train.py "$@"
