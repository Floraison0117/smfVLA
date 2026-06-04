#!/bin/bash
# Evaluate step_3000 on all LIBERO suites, 5 episodes/task
set -e

PROJECT_ROOT="/root/autodl-tmp/smfVLA"
cd "$PROJECT_ROOT"

CKPT="$PROJECT_ROOT/checkpoints/finetuned/smf_curr_v2/step_3000"
NFE=1
NUM_EPISODES=5
RESULTS_DIR="$PROJECT_ROOT/results/eval_step3000_all_suites"
PYTHON="/root/miniconda3/envs/openpi_server/bin/python"
LOG_DIR="$PROJECT_ROOT/logs/eval_step3000_all_suites"
mkdir -p "$LOG_DIR"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10" "libero_90")

echo "============================================"
echo "Step 3000 - All LIBERO Suites Evaluation"
echo "Checkpoint: $CKPT"
echo "Episodes/task: $NUM_EPISODES | NFE: $NFE"
echo "Suites: ${SUITES[*]}"
echo "============================================"

for SUITE in "${SUITES[@]}"; do
    LOG="$LOG_DIR/eval_${SUITE}.log"
    echo ""
    echo "######################################################"
    echo "# Evaluating: $SUITE"
    echo "# Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# Log: $LOG"
    echo "######################################################"

    $PYTHON scripts/eval_direct.py \
        --nfe $NFE \
        --task-suite $SUITE \
        --num-episodes $NUM_EPISODES \
        --checkpoint "$CKPT" \
        --results-dir "$RESULTS_DIR" \
        --no-video > "$LOG" 2>&1

    RATE=$(grep "Total:" "$LOG" | grep -oP '\d+\.\d+%' || echo "UNKNOWN")
    echo ">>> $SUITE done: $RATE at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "============================================"
echo "All suites evaluated!"
echo "Results:"
for SUITE in "${SUITES[@]}"; do
    LOG="$LOG_DIR/eval_${SUITE}.log"
    if [ -f "$LOG" ]; then
        RATE=$(grep "Total:" "$LOG" | grep -oP '\d+\.\d+%' || echo "N/A")
        echo "  $SUITE: $RATE"
    fi
done
echo "============================================"
