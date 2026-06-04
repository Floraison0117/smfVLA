#!/bin/bash
# LIBERO-Plus 对比评测：pi05-libero (10-NFE & 1-NFE) vs step_3000 (1-NFE)
# 100 个 episode，相同 seed
set -e

PROJECT_ROOT="/root/autodl-tmp/smfVLA"
cd "$PROJECT_ROOT"

PYTHON="/root/miniconda3/envs/openpi_server/bin/python"
LOG_DIR="$PROJECT_ROOT/logs/eval_libero_plus_comparison"
RESULTS_DIR="$PROJECT_ROOT/results/libero_plus"
mkdir -p "$LOG_DIR"

PI05_CKPT="$PROJECT_ROOT/checkpoints/base/pi05_libero"
SMF_CKPT="$PROJECT_ROOT/checkpoints/finetuned/smf_curr_v2/step_3000"

echo "============================================"
echo "LIBERO-Plus Comparison Evaluation"
echo "100 episodes, same seed=7"
echo "============================================"
echo "1) pi05-libero + NFE=10 (baseline)"
echo "2) pi05-libero + NFE=1"
echo "3) step_3000 + NFE=1 (SMF)"
echo "============================================"

# ── 评测 1: pi05-libero, NFE=10 ─────────────────────────────
LOG1="$LOG_DIR/pi05_nfe10.log"
echo ""
echo "######################################################"
echo "# [1/3] pi05-libero | NFE=10"
echo "# Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "######################################################"

$PYTHON scripts/eval_libero_plus.py \
    --preset quick \
    --nfe 10 \
    --checkpoint "$PI05_CKPT" \
    --seed 7 \
    --results-dir "$RESULTS_DIR" > "$LOG1" 2>&1

RATE1=$(grep "Total:" "$LOG1" | grep -oP '\d+\.\d+%' || echo "UNKNOWN")
echo ">>> pi05-libero NFE=10 done: $RATE1"

# ── 评测 2: pi05-libero, NFE=1 ──────────────────────────────
LOG2="$LOG_DIR/pi05_nfe1.log"
echo ""
echo "######################################################"
echo "# [2/3] pi05-libero | NFE=1"
echo "# Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "######################################################"

$PYTHON scripts/eval_libero_plus.py \
    --preset quick \
    --nfe 1 \
    --checkpoint "$PI05_CKPT" \
    --seed 7 \
    --results-dir "$RESULTS_DIR" > "$LOG2" 2>&1

RATE2=$(grep "Total:" "$LOG2" | grep -oP '\d+\.\d+%' || echo "UNKNOWN")
echo ">>> pi05-libero NFE=1 done: $RATE2"

# ── 评测 3: step_3000, NFE=1 ────────────────────────────────
LOG3="$LOG_DIR/smf_step3000_nfe1.log"
echo ""
echo "######################################################"
echo "# [3/3] step_3000 (SMF) | NFE=1"
echo "# Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "######################################################"

$PYTHON scripts/eval_libero_plus.py \
    --preset quick \
    --nfe 1 \
    --checkpoint "$SMF_CKPT" \
    --seed 7 \
    --results-dir "$RESULTS_DIR" > "$LOG3" 2>&1

RATE3=$(grep "Total:" "$LOG3" | grep -oP '\d+\.\d+%' || echo "UNKNOWN")
echo ">>> step_3000 NFE=1 done: $RATE3"

# ── 汇总 ───────────────────────────────────────────────────
echo ""
echo "============================================"
echo "LIBERO-Plus Comparison Results (100 episodes)"
echo "============================================"
echo "  pi05-libero NFE=10: $RATE1"
echo "  pi05-libero NFE=1:  $RATE2"
echo "  step_3000   NFE=1:  $RATE3"
echo "============================================"
