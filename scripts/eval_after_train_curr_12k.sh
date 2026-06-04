#!/bin/bash
# 监听 smf_curr_12k 训练完成后，自动 eval 4 个 checkpoint
# 用法: bash scripts/eval_after_train_curr_12k.sh

set -e

PROJECT_ROOT="/root/autodl-tmp/smfVLA"
cd "$PROJECT_ROOT"

CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/finetuned/smf_curr_12k"
SUITE="libero_spatial"
NUM_EPISODES=25
NFE=1
RESULTS_DIR="$PROJECT_ROOT/results/eval_curr_12k"
PYTHON="/root/miniconda3/envs/openpi_server/bin/python"
LOG_DIR="/tmp/eval_logs_curr_12k"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

STEPS=(3000 6000 9000 12000)

# ── 等待训练完成 ─────────────────────────────────────────
echo "============================================"
echo " 等待 smf_curr_12k 训练完成..."
echo " 监控 checkpoint 目录: $CHECKPOINT_DIR"
echo " 目标 checkpoints: ${STEPS[*]}"
echo "============================================"

while true; do
    # 检查是否所有 checkpoint 都已存在
    ALL_DONE=true
    for STEP in "${STEPS[@]}"; do
        CKPT="$CHECKPOINT_DIR/step_$STEP/params"
        if [ ! -d "$CKPT" ]; then
            ALL_DONE=false
            break
        fi
    done

    if [ "$ALL_DONE" = true ]; then
        echo ""
        echo "✓ 所有 checkpoint 已就绪！"
        break
    fi

    # 检查训练进程是否还在运行
    if ! pgrep -f "run_train.py.*smf_curr_12k" > /dev/null 2>&1; then
        # 进程不在了，再检查一次 checkpoint
        sleep 5
        ALL_DONE=true
        for STEP in "${STEPS[@]}"; do
            CKPT="$CHECKPOINT_DIR/step_$STEP/params"
            if [ ! -d "$CKPT" ]; then
                ALL_DONE=false
                echo "警告: 训练进程已结束，但 step_$STEP checkpoint 不存在"
            fi
        done
        if [ "$ALL_DONE" = true ]; then
            echo "✓ 训练已完成，所有 checkpoint 就绪"
            break
        else
            echo "✗ 训练异常退出，部分 checkpoint 缺失"
            exit 1
        fi
    fi

    # 显示当前进度
    DONE_COUNT=0
    for STEP in "${STEPS[@]}"; do
        if [ -d "$CHECKPOINT_DIR/step_$STEP/params" ]; then
            DONE_COUNT=$((DONE_COUNT + 1))
        fi
    done
    echo -ne "\r[$(date '+%H:%M:%S')] 已完成 $DONE_COUNT/${#STEPS[@]} 个 checkpoint..."
    sleep 30
done

# ── 开始 Eval ─────────────────────────────────────────────
echo ""
echo "============================================"
echo " 开始 Eval"
echo " Suite: $SUITE | Episodes/task: $NUM_EPISODES | NFE: $NFE"
echo " Checkpoints: ${STEPS[*]}"
echo "============================================"

for STEP in "${STEPS[@]}"; do
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

# ── 汇总 ──────────────────────────────────────────────────
echo ""
echo "============================================"
echo " smf_curr_12k Eval 完成！"
echo " Results:"
for STEP in "${STEPS[@]}"; do
    LOG="$LOG_DIR/eval_step_${STEP}.log"
    if [ -f "$LOG" ]; then
        RATE=$(grep "Total:" "$LOG" | grep -oP '\d+\.\d+%' || echo "N/A")
        echo "  step_$STEP: $RATE"
    fi
done
echo "============================================"
