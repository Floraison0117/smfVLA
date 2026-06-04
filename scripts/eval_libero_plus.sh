#!/bin/bash
# LIBERO-Plus 评测脚本
# 用法:
#   bash scripts/eval_libero_plus.sh                      # 默认 quick preset, 1-NFE
#   bash scripts/eval_libero_plus.sh quick 1              # quick preset, 1-NFE
#   bash scripts/eval_libero_plus.sh full 1 2 4 10        # full preset, 多 NFE
#   bash scripts/eval_libero_plus.sh medium 1 /path/to/ckpt  # 指定 checkpoint
set -e

PROJECT_ROOT="/root/autodl-tmp/smfVLA"
cd "$PROJECT_ROOT"

PYTHON="/root/miniconda3/envs/openpi_server/bin/python"
PRESET="${1:-quick}"
shift || true

# 收集 NFE 值和 checkpoint
NFE_ARGS=""
CHECKPOINT=""
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]] && [[ "$arg" -eq 1 || "$arg" -eq 2 || "$arg" -eq 4 || "$arg" -eq 10 ]]; then
        NFE_ARGS="$NFE_ARGS $arg"
    elif [[ -d "$arg" ]]; then
        CHECKPOINT="$arg"
    fi
done

NFE_ARGS="${NFE_ARGS:-1}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_ROOT/checkpoints/finetuned/smf_base/step_5000}"

echo "============================================"
echo "LIBERO-Plus Evaluation"
echo "Preset: $PRESET"
echo "NFE: $NFE_ARGS"
echo "Checkpoint: $CHECKPOINT"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

$PYTHON scripts/eval_libero_plus.py \
    --preset "$PRESET" \
    --nfe $NFE_ARGS \
    --checkpoint "$CHECKPOINT" \
    --results-dir "$PROJECT_ROOT/results/libero_plus"

echo ""
echo "============================================"
echo "Done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results in: $PROJECT_ROOT/results/libero_plus/"
echo "============================================"
