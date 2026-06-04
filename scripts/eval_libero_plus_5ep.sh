#!/bin/bash
# LIBERO-plus 轻量级评测：pi05-libero checkpoint, 10-NFE 和 1-NFE, 4 suites, 5 episodes/task
#
# 用法:
#   bash scripts/eval_libero_plus_5ep.sh          # 跑 10-NFE + 1-NFE
#   bash scripts/eval_libero_plus_5ep.sh 10       # 只跑 10-NFE
#   bash scripts/eval_libero_plus_5ep.sh 1        # 只跑 1-NFE

set -euo pipefail
cd /root/autodl-tmp/smfVLA

CHECKPOINT="checkpoints/base/pi05_libero"
SUITES="libero_spatial libero_object libero_goal libero_10"
NUM_EPISODES=5
SEED=7
PYTHON="/root/miniconda3/envs/openpi_server/bin/python"

run_nfe() {
    local nfe=$1
    echo "============================================"
    echo "  LIBERO-plus eval: NFE=${nfe}, 5 episodes/task"
    echo "  Checkpoint: ${CHECKPOINT}"
    echo "============================================"

    for suite in $SUITES; do
        echo ""
        echo "--- ${suite} (NFE=${nfe}) ---"
        $PYTHON scripts/eval_libero_plus.py \
            --suite "$suite" \
            --nfe "$nfe" \
            --num-episodes "$NUM_EPISODES" \
            --checkpoint "$CHECKPOINT" \
            --seed "$SEED" 2>&1 | grep -v "DeprecationWarning\|flax\|jax\|linear_util\|scope.py"
    done
}

if [ $# -eq 0 ]; then
    run_nfe 10
    run_nfe 1
else
    run_nfe "$1"
fi

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  Results in: results/libero_plus/"
echo "============================================"
