#!/bin/bash
# Monitor FreeFlow training progress

echo "=== FreeFlow Training Monitor ==="
echo "检查时间: $(date)"
echo

# 检查进程
PID=$(pgrep -f "freeflow.training.run_train" | head -1)
if [ -z "$PID" ]; then
    echo "❌ 训练进程未运行"
    exit 1
fi

echo "✅ 训练进程运行中 (PID: $PID)"
echo "运行时间: $(ps -p $PID -o etime= | xargs)"
echo "CPU: $(ps -p $PID -o %cpu= | xargs)%"
echo "内存: $(ps -p $PID -o rss= | awk '{printf "%.1f GB", $1/1024/1024}')"
echo

# 检查检查点
CHECKPOINT_DIR="/root/autodl-tmp/freeflow/checkpoints/finetuned/freeflow"
if [ -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINTS=$(ls -1 $CHECKPOINT_DIR/step_* 2>/dev/null | wc -l)
    echo "检查点数量: $CHECKPOINTS"
    if [ $CHECKPOINTS -gt 0 ]; then
        echo "最新检查点:"
        ls -lt $CHECKPOINT_DIR/step_* 2>/dev/null | head -1
    fi
else
    echo "检查点目录尚未创建"
fi

echo
echo "等待首次日志 (step 100)..."
