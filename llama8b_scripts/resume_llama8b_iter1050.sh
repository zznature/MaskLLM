#!/bin/bash

# Resume Llama8b training from iteration 1050 while keeping LR schedule trend.
# - Updates the checkpoint tracker to iter 1050
# - Starts training in resume mode (RESUME_FLAG=1)

set -euo pipefail

PROJECT_DIR=$(pwd)
CHECKPOINT_DIR="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8"
TRACKER_FILE="$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"

echo "Setting tracker to iteration 1050: $TRACKER_FILE"
mkdir -p "$CHECKPOINT_DIR"
echo 1050 > "$TRACKER_FILE"
echo "Tracker now points to: $(cat "$TRACKER_FILE")"

echo "Launching resume training (RESUME_FLAG=1)..."
bash "$PROJECT_DIR/llama8b_scripts/llama8b_presparse_training_tp8.sh" 1


