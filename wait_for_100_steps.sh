#!/bin/bash
# 等待 FreeFlow 训练达到 100 步

CHECKPOINT_DIR="/root/autodl-tmp/freeflow/checkpoints/finetuned/freeflow"
WANDB_FILE="/root/autodl-tmp/freeflow/wandb/run-20260609_083644-egxyes9f/run-egxyes9f.wandb"

echo "等待 FreeFlow 训练达到 100 步..."
echo "开始时间: $(date)"
echo

CHECK_COUNT=0
MAX_CHECKS=100  # 最多检查 100 次 (约 16 分钟)

while [ $CHECK_COUNT -lt $MAX_CHECKS ]; do
    # 检查进程
    PID=$(pgrep -f "freeflow.training.run_train" | head -1)
    if [ -z "$PID" ]; then
        echo "❌ 训练进程已停止"
        exit 1
    fi

    # 检查检查点
    if [ -d "$CHECKPOINT_DIR" ]; then
        STEPS=$(ls -1 $CHECKPOINT_DIR/step_* 2>/dev/null | wc -l)
        if [ $STEPS -gt 0 ]; then
            LATEST_STEP=$(ls -t $CHECKPOINT_DIR/step_* 2>/dev/null | head -1 | grep -o "step_[0-9]*" | grep -o "[0-9]*")
            echo "✅ 发现检查点: step_$LATEST_STEP"
            echo "时间: $(date)"
            echo "运行时间: $(ps -p $PID -o etime= | xargs)"
            exit 0
        fi
    fi

    # 检查 WandB 文件更新
    if [ -f "$WANDB_FILE" ]; then
        LAST_MOD=$(stat -c "%Y" "$WANDB_FILE")
        NOW=$(date +%s)
        DIFF=$((NOW - LAST_MOD))

        if [ $DIFF -gt 300 ]; then
            echo "⚠️  WandB 文件超过 5 分钟未更新"
            echo "进程状态: $(ps -p $PID -o state= | xargs)"
            echo "运行时间: $(ps -p $PID -o etime= | xargs)"
        fi
    fi

    CHECK_COUNT=$((CHECK_COUNT + 1))
    echo "[$CHECK_COUNT/$MAX_CHECKS] 等待中... ($(date +%H:%M:%S))"
    sleep 10
done

echo "⏱️  超时: 100 步检查点未在预期时间内创建"
echo "当前状态:"
bash /root/autodl-tmp/monitor_freeflow.sh
