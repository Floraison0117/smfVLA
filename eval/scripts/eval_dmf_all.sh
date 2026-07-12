#!/bin/bash
# Evaluate all DMF checkpoints on LIBERO-Plus quick mode (1-NFE)

CHECKPOINT_DIR="/root/autodl-tmp/checkpoints/dmf_finetuned"
RESULTS_DIR="/root/autodl-tmp/eval/results/dmf"
mkdir -p "$RESULTS_DIR"

# List of all DMF checkpoints
CHECKPOINTS=(
    "step_0005000"
    "step_0010000"
    "step_0015000"
    "step_0020000"
    "step_0025000"
    "step_0030000"
)

echo "========================================"
echo "DMF Evaluation on LIBERO-Plus Quick Mode"
echo "========================================"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Results dir: $RESULTS_DIR"
echo ""

for CKPT in "${CHECKPOINTS[@]}"; do
    CKPT_PATH="$CHECKPOINT_DIR/$CKPT"
    if [ ! -d "$CKPT_PATH" ]; then
        echo "Skipping $CKPT (not found)"
        continue
    fi

    echo "========================================"
    echo "Evaluating: $CKPT"
    echo "========================================"

    conda run -n openpi_server --no-capture-output python eval_libero_plus.py \
        --preset quick \
        --nfe 1 \
        --model-type dmf \
        --checkpoint "$CKPT_PATH" \
        2>&1 | tee "$RESULTS_DIR/${CKPT}_libero_plus_quick.log"

    echo "Completed: $CKPT"
    echo ""
done

echo "========================================"
echo "All evaluations completed!"
echo "========================================"
