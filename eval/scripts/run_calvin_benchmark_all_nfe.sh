#!/bin/bash
# CALVIN benchmark eval for all model types and NFE values.
#
# Usage:
#   bash eval/scripts/run_calvin_benchmark_all_nfe.sh [MODE] [MODEL_TYPES...]
#   bash eval/scripts/run_calvin_benchmark_all_nfe.sh quick pi05 dmf piflow
#   bash eval/scripts/run_calvin_benchmark_all_nfe.sh normal pi05
#   bash eval/scripts/run_calvin_benchmark_all_nfe.sh fullset dmf piflow

set -eu

MODE=${1:-"quick"}
shift || true
MODEL_TYPES=${@:-"pi05 dmf piflow"}
NFE_VALUES="1 2 4 10"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi_server

cd /root/autodl-tmp

for model_type in $MODEL_TYPES; do
    for nfe in $NFE_VALUES; do
        echo "=========================================="
        echo "CALVIN eval: model=$model_type nfe=$nfe mode=$MODE"
        echo "=========================================="
        python -m eval.calvin.main \
            --model-type "$model_type" \
            --nfe "$nfe" \
            --mode "$MODE"
    done
done

echo "All CALVIN evaluations completed!"
