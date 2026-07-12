#!/bin/bash
# Serial LIBERO-Plus quick 1-NFE eval across the 6 FreeFlow finetune checkpoints.
#
# Each checkpoint runs: eval_libero_plus.py --preset quick --nfe 1 --model-type freeflow
#   quick preset = libero_spatial, libero_object, libero_goal, libero_10 (10 tasks x 5 episodes each = 200 episodes)
#
# Runs are STRICTLY SERIAL (one checkpoint at a time, one GPU). One checkpoint failing
# does not abort the others (no `set -e`). Per-checkpoint stdout/stderr -> separate log.
# Summary with exit codes + grand_total_rate -> ${LOG_DIR}/run_<ts>_summary.log

set -u  # unset var = error; intentionally NO `set -e` so the loop survives per-run failures

PROJECT_ROOT="/root/autodl-tmp"
FREEFLOW_ROOT="${PROJECT_ROOT}/freeflow"
EVAL_DIR="${PROJECT_ROOT}/eval/scripts"
CKPT_DIR="${FREEFLOW_ROOT}/checkpoints/finetuned/freeflow"
LOG_DIR="${PROJECT_ROOT}/logs/freeflow_libero_plus_quick"

# PYTHONPATH mirrors freeflow/scripts/eval_freeflow.sh (proven working set)
export PYTHONPATH="${FREEFLOW_ROOT}/src:${FREEFLOW_ROOT}/third_party/openpi/src:${FREEFLOW_ROOT}/third_party/openpi/packages/openpi-client/src"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi_server

mkdir -p "$LOG_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/run_${RUN_TS}_summary.log"

CHECKPOINTS=(step_5000 step_10000 step_15000 step_20000 step_25000 step_30000)

{
  echo "Serial FreeFlow LIBERO-Plus quick 1-NFE eval"
  echo "  started : $(date)"
  echo "  env     : openpi_server"
  echo "  GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "  ckpt dir: ${CKPT_DIR}"
  echo "  log dir : ${LOG_DIR}"
  echo "  checkpoints: ${CHECKPOINTS[*]}"
  echo "=================================================================="
} | tee "$SUMMARY"

for step in "${CHECKPOINTS[@]}"; do
  ckpt="${CKPT_DIR}/${step}"
  log="${LOG_DIR}/${RUN_TS}_${step}.log"

  echo "" | tee -a "$SUMMARY"
  echo "=== [$(date)] START ${step}" | tee -a "$SUMMARY"
  echo "    checkpoint: ${ckpt}" | tee -a "$SUMMARY"
  echo "    log      : ${log}" | tee -a "$SUMMARY"

  python "${EVAL_DIR}/eval_libero_plus.py" \
      --preset quick \
      --nfe 1 \
      --model-type freeflow \
      --checkpoint "$ckpt" \
      > "$log" 2>&1
  rc=$?

  # Pull the grand_total_rate from the combined JSON this run wrote (path is logged on success).
  rate=""
  combined_json=$(grep -oE 'Combined results saved to: .*\.json' "$log" | tail -1 | sed 's/Combined results saved to: //')
  if [ -n "$combined_json" ] && [ -f "$combined_json" ]; then
    rate=$(python -c "import json,sys; print(json.load(open('$combined_json')).get('grand_total_rate','?'))" 2>/dev/null)
  fi

  echo "=== [$(date)] DONE  ${step}  (exit=${rc}, grand_total_rate=${rate:-N/A})" | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
{
  echo "=================================================================="
  echo "All checkpoints finished: $(date)"
} | tee -a "$SUMMARY"
