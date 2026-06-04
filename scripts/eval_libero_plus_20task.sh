#!/bin/bash
# LIBERO-plus 轻量级评测：20 tasks/suite, 5 episodes/task, 1-NFE + 10-NFE
set -euo pipefail
cd /root/autodl-tmp/smfVLA

CHECKPOINT="checkpoints/base/pi05_libero"
SUITES="libero_spatial libero_object libero_goal libero_10"
NUM_EPISODES=5
MAX_TASKS=20
SEED=7
PYTHON="/root/miniconda3/envs/openpi_server/bin/python"

run_nfe() {
    local nfe=$1
    echo "============================================"
    echo "  LIBERO-plus eval: NFE=${nfe}, ${MAX_TASKS} tasks, ${NUM_EPISODES} eps/task"
    echo "============================================"
    for suite in $SUITES; do
        echo ""
        echo "--- ${suite} (NFE=${nfe}) ---"
        $PYTHON scripts/eval_libero_plus.py \
            --suite "$suite" \
            --nfe "$nfe" \
            --num-episodes "$NUM_EPISODES" \
            --max-tasks "$MAX_TASKS" \
            --checkpoint "$CHECKPOINT" \
            --seed "$SEED" 2>&1 | grep -v "DeprecationWarning\|flax\|jax\|linear_util\|scope.py\|robosuite WARNING\|Gym has been\|Please upgrade\|Users of this\|See the migration\|UserWarning"
    done
}

run_nfe 10
run_nfe 1

echo ""
echo "============================================"
echo "  All done! Results in: results/libero_plus/"
echo "============================================"
