#!/bin/bash
# 使用 pi05-libero checkpoint 在 LIBERO 4 个 suite 上做 1-NFE eval (除 libero_90)
# 每个 task 5 个 episode

cd /root/autodl-tmp/smfVLA
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi_server

export PYTHONPATH="/root/autodl-tmp/smfVLA/src:$PYTHONPATH"

# Suite 列表
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for suite in "${SUITES[@]}"; do
    echo "============================================================"
    echo "开始评测: $suite (pi05-libero, NFE=10)"
    echo "============================================================"

    python3 scripts/eval_direct.py \
        --checkpoint /root/autodl-tmp/smfVLA/checkpoints/base/pi05_libero \
        --task-suite "$suite" \
        --num-episodes 5 \
        --nfe 1 \
        --results-dir results/pi05_libero_4suites_1nfe

    echo "完成: $suite"
    echo ""
done

echo "所有 4 个 suite 评测完成！"
