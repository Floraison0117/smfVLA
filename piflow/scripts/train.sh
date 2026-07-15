#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="/root/autodl-tmp/openpi"

export JAX_PLATFORMS=cuda

PYTHONPATH="${PROJECT_ROOT}/src:${OPENPI_DIR}/src:${OPENPI_DIR}/packages/openpi-client/src"
export PYTHONPATH

CONDA_ENV="openpi_server"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/root/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

CONFIG_FILE="${1:-$PROJECT_ROOT/configs/train/piflow_libero_plus.yaml}"

echo "============================================"
echo "Pi-Flow Training"
echo "Config: ${CONFIG_FILE}"
echo "Python: $(which python)"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "Conda env: ${CONDA_ENV}"
echo "============================================"

RESUME=""
if [[ "${2:-}" == "--resume" ]]; then
    RESUME="--resume ${3}"
fi

python "${SCRIPT_DIR}/run_train.py" "${CONFIG_FILE}" ${RESUME}
