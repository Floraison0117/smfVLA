#!/bin/bash
# Evaluate smf_base checkpoints (step 6000-15000) on libero_spatial, 25 episodes/task
set -e

PROJECT_ROOT="/root/autodl-tmp/smfVLA"
cd "$PROJECT_ROOT"

CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/finetuned/smf_base"
SUITE="libero_spatial"
NUM_EPISODES=25
NFE=1
RESULTS_DIR="$PROJECT_ROOT/results/eval_checkpoints_6k_15k"
PYTHON="/root/miniconda3/envs/openpi_server/bin/python"
LOG_DIR="/tmp/eval_logs_6k_15k"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "============================================"
echo "SMF Base Checkpoint Evaluation (6k-15k)"
echo "Suite: $SUITE | Episodes/task: $NUM_EPISODES | NFE: $NFE"
echo "Results dir: $RESULTS_DIR"
echo "Checkpoints: 6000 8000 10000 12000 14000 15000"
echo "============================================"

for STEP in 6000 8000 10000 12000 14000 15000; do
    CKPT="$CHECKPOINT_DIR/step_$STEP"
    LOG="$LOG_DIR/eval_step_${STEP}.log"
    echo ""
    echo "######################################################"
    echo "# Evaluating: step_$STEP"
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
    echo ">>> step_$STEP done: $RATE at $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "============================================"
echo "All checkpoints evaluated!"
echo "Results:"
for STEP in 6000 8000 10000 12000 14000 15000; do
    LOG="$LOG_DIR/eval_step_${STEP}.log"
    if [ -f "$LOG" ]; then
        RATE=$(grep "Total:" "$LOG" | grep -oP '\d+\.\d+%' || echo "N/A")
        echo "  step_$STEP: $RATE"
    fi
done
echo "============================================"
