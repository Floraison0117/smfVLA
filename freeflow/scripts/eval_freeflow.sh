#!/bin/bash
# FreeFlow evaluation script for LIBERO-Plus

set -e

# Default values
NFE=${NFE:-1}
PRESET=${PRESET:-quick}
CHECKPOINT=${CHECKPOINT:-"../checkpoints/finetuned/freeflow"}
DATASET=${DATASET:-"libero-plus"}

# Set paths
PROJECT_ROOT="/root/autodl-tmp/freeflow"
EVAL_DIR="/root/autodl-tmp/eval/scripts"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/third_party/openpi/src:${PROJECT_ROOT}/third_party/openpi/packages/openpi-client/src"

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate libero_eval

echo "=========================================="
echo "FreeFlow Evaluation"
echo "=========================================="
echo "Dataset: $DATASET"
echo "NFE: $NFE"
echo "Preset: $PRESET"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

# Run evaluation
if [ "$DATASET" = "libero-plus" ]; then
    python ${EVAL_DIR}/eval_libero_plus.py \
        --preset "$PRESET" \
        --nfe "$NFE" \
        --checkpoint "$CHECKPOINT" \
        --model-type freeflow
else
    python ${EVAL_DIR}/eval_direct.py \
        --preset "$PRESET" \
        --nfe "$NFE" \
        --checkpoint "$CHECKPOINT" \
        --model-type freeflow
fi

echo "Evaluation completed!"
