#!/bin/bash

# Purpose: Quick audit of disk/quota and checkpoint sizes for the Llama8b Wanda run
# Usage: Run inside the apptainer via wrapper
#   bash run_maskllm_native.sh llama8b_scripts/check_quota_and_ckpt_usage.sh

set -euo pipefail

PROJECT_DIR=$(pwd)
CKPT_DIR="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8_wanda"
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training_wanda"

echo "[1/5] Filesystem free space (df -h):"
df -h || true
echo

echo "[2/5] User quota (if available):"
if command -v quota >/dev/null 2>&1; then
  quota -s || true
else
  echo "quota command not available; skipping."
fi
echo

echo "[3/5] Checkpoint directory usage: $CKPT_DIR"
if [ -d "$CKPT_DIR" ]; then
  du -sh "$CKPT_DIR" || true
  echo "Top 20 largest files under checkpoint dir (apparent size):"
  # Using apparent size to reflect file system allocation reality for large files
  (cd "$CKPT_DIR" && du -ah --apparent-size 2>/dev/null | sort -h | tail -n 20) || true
else
  echo "Checkpoint dir not found: $CKPT_DIR"
fi
echo

echo "[4/5] Logs directory usage: $LOG_DIR"
if [ -d "$LOG_DIR" ]; then
  du -sh "$LOG_DIR" || true
else
  echo "Logs dir not found: $LOG_DIR"
fi
echo

echo "[5/5] Project root top 30 largest files (may take time):"
du -ah --apparent-size "$PROJECT_DIR" 2>/dev/null | sort -h | tail -n 30 || true
echo

echo "Audit complete."


