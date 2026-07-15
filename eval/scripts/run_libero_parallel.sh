#!/bin/bash
# Launch parallel LIBERO-Plus eval workers, then merge results.
#
# Usage:
#   bash eval/scripts/run_libero_parallel.sh                           # defaults
#   bash eval/scripts/run_libero_parallel.sh pi05 1 quick 2
#   bash eval/scripts/run_libero_parallel.sh dmf 10 normal 3
#   bash eval/scripts/run_libero_parallel.sh pi05 10 fullset 4 /path/to/ckpt /path/to/results
#
# Each worker writes to its own log file under logs/libero_plus_parallel/.
# After all workers finish, results are merged automatically.

set -u

MODEL_TYPE="${1:-pi05}"
NFE="${2:-1}"
MODE="${3:-quick}"
NUM_WORKERS="${4:-2}"
CHECKPOINT="${5:-}"
RESULTS_DIR="${6:-}"

PROJECT_ROOT="/root/autodl-tmp"
LOG_DIR="${PROJECT_ROOT}/logs/libero_plus_parallel"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi_server

mkdir -p "$LOG_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)

EXTRA=""
if [ -n "$CHECKPOINT" ]; then
    EXTRA="$EXTRA --checkpoint $CHECKPOINT"
fi
if [ -n "$RESULTS_DIR" ]; then
    EXTRA="$EXTRA --results-dir $RESULTS_DIR"
fi

echo "=== LIBERO-Plus Parallel Eval ==="
echo "  Model:     $MODEL_TYPE"
echo "  NFE:       $NFE"
echo "  Mode:      $MODE"
echo "  Workers:   $NUM_WORKERS"
echo "  Log dir:   $LOG_DIR"
echo "================================="

PIDS=()
for ((i=0; i<NUM_WORKERS; i++)); do
    LOG="${LOG_DIR}/${RUN_TS}_w${i}_of${NUM_WORKERS}.log"
    echo "  Worker $i/$NUM_WORKERS -> $(basename "$LOG")"
    python -m eval.libero_plus.main \
        --model-type "$MODEL_TYPE" \
        --nfe "$NFE" \
        --mode "$MODE" \
        --num-workers "$NUM_WORKERS" \
        --worker-id "$i" \
        $EXTRA \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for all $NUM_WORKERS workers to complete..."
EXIT_CODE=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || EXIT_CODE=$?
done

echo ""
echo "All workers finished (exit=$EXIT_CODE). Merging results..."

python -m eval.libero_plus.main \
    --merge-results \
    --num-workers "$NUM_WORKERS" \
    --model-type "$MODEL_TYPE" \
    --nfe "$NFE" \
    --mode "$MODE" \
    $EXTRA

echo "Done."
