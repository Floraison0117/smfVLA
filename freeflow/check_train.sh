#!/bin/bash
# FreeFlow 训练监控脚本

echo "=== FreeFlow 训练进度 ==="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

echo "--- 资源使用 ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "GPU: %s%%, 显存: %s MB, 温度: %s°C\n", $1, $2, $3}'

echo ""
echo "--- 进程状态 ---"
ps aux | grep -E "python.*freeflow" | grep -v grep | \
    awk '{printf "运行时间: %s, CPU: %s%%, 内存: %s%%\n", $10, $3, $4}'

echo ""
echo "--- WandB 日志 ---"
LATEST_WANDB=$(ls -t /root/autodl-tmp/freeflow/wandb/run-*/files/*.log 2>/dev/null | head -1)
if [ -n "$LATEST_WANDB" ]; then
    echo "最新日志: $LATEST_WANDB"
    tail -5 "$LATEST_WANDB" 2>/dev/null | grep -E "WARNING|ERROR" || echo "无错误或警告"
fi

echo ""
echo "--- Checkpoint ---"
ls -la /root/autodl-tmp/freeflow/checkpoints/finetuned/freeflow/ 2>/dev/null | tail -5 || echo "暂无 checkpoint"
