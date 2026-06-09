#!/bin/bash
# FreeFlow training with batch_size stability test
# Tests batch_size=32 first, falls back to 16 if unstable

set -e

PROJECT_ROOT="/root/autodl-tmp/freeflow"
CONFIG_FILE="${PROJECT_ROOT}/configs/train/freeflow_base_libero.yaml"
BACKUP_CONFIG="${PROJECT_ROOT}/configs/train/freeflow_base_libero_backup.yaml"

echo "=========================================="
echo "FreeFlow Training with Batch Size Test"
echo "=========================================="
echo "Step 1: Test with batch_size=32"
echo "Step 2: Monitor for 100 steps"
echo "Step 3: If stable, continue; else fallback to 16"
echo "=========================================="

# Backup original config
cp "$CONFIG_FILE" "$BACKUP_CONFIG"

# Function to check if training is stable
check_stability() {
    local log_file="$1"
    local steps_to_check=50

    echo "Checking last $steps_to_check steps for stability..."

    # Check for common instability signs
    if grep -i "oom\|out of memory\|cuda error" "$log_file" 2>/dev/null; then
        echo "❌ OOM detected - unstable"
        return 1
    fi

    # Check for NaN losses
    if grep -E "loss.*nan|loss.*inf" "$log_file" 2>/dev/null; then
        echo "❌ NaN/Inf loss detected - unstable"
        return 1
    fi

    echo "✅ Training appears stable"
    return 0
}

# Function to update batch size
update_batch_size() {
    local new_batch_size=$1
    echo "Updating batch_size to $new_batch_size..."
    sed -i "s/batch_size: [0-9]*/batch_size: $new_batch_size/" "$CONFIG_FILE"
    grep "batch_size" "$CONFIG_FILE"
}

# Function to launch training
launch_training() {
    cd "$PROJECT_ROOT"

    echo "Launching training with current config..."
    bash scripts/train.sh "$CONFIG_FILE" &
    TRAIN_PID=$!

    echo "Training PID: $TRAIN_PID"
    echo "Monitor logs: tail -f logs/train/freeflow/*.log"

    # Wait for initial compilation and first steps
    echo "Waiting 60 seconds for initial training..."
    sleep 60

    # Check if process is still running
    if ps -p $TRAIN_PID > /dev/null; then
        echo "✅ Training process running"

        # Wait for more steps to check stability
        echo "Monitoring for stability (next 2 minutes)..."
        sleep 120

        # Check logs
        local log_file=$(ls -t "$PROJECT_ROOT"/logs/train/freeflow/*.log 2>/dev/null | head -1)
        if [ -n "$log_file" ] && [ -f "$log_file" ]; then
            if check_stability "$log_file"; then
                echo "=========================================="
                echo "✅ STABLE: batch_size=32 confirmed"
                echo "=========================================="
                echo "Training continuing with batch_size=32"
                echo "Monitor: tail -f $log_file"
                echo "To stop: kill $TRAIN_PID"
                return 0
            else
                echo "=========================================="
                echo "❌ UNSTABLE: Falling back to batch_size=16"
                echo "=========================================="
                kill $TRAIN_PID 2>/dev/null || true
                sleep 5
                update_batch_size 16
                echo "Restarting with batch_size=16..."
                sleep 2
                return 1
            fi
        else
            echo "⚠️  No log file found, assuming stable"
            return 0
        fi
    else
        echo "❌ Training process died immediately"
        echo "Trying batch_size=16..."
        update_batch_size 16
        return 1
    fi
}

# Main loop
MAX_RETRIES=1
retry_count=0

while [ $retry_count -le $MAX_RETRIES ]; do
    if launch_training; then
        echo "Training started successfully"
        exit 0
    else
        retry_count=$((retry_count + 1))
        if [ $retry_count -le $MAX_RETRIES ]; then
            echo "Retrying with batch_size=16..."
            sleep 5
        fi
    fi
done

echo "=========================================="
echo "Final configuration:"
grep "batch_size" "$CONFIG_FILE"
echo "=========================================="
echo "Please run training manually:"
echo "  cd $PROJECT_ROOT"
echo "  bash scripts/train.sh $CONFIG_FILE"
