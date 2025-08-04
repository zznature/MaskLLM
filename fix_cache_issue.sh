#!/bin/bash

echo "=== MaskLLM Training Cache Issue Diagnostic & Fix ==="
echo "Timestamp: $(date)"
echo

# Get current directory
PROJECT_PATH=$(pwd)
echo "Project Path: $PROJECT_PATH"

# Check current disk usage
echo "=== Disk Usage Analysis ==="
df -h $PROJECT_PATH
echo

# Check if CACHE directory exists and its permissions
echo "=== Cache Directory Analysis ==="
if [ -d "CACHE" ]; then
    echo "CACHE directory exists:"
    ls -la CACHE/
    echo "CACHE directory size:"
    du -sh CACHE/
    echo "CACHE directory permissions:"
    stat CACHE/
else
    echo "CACHE directory does not exist"
fi
echo

# Check available disk space
echo "=== Available Space Check ==="
AVAILABLE_SPACE=$(df $PROJECT_PATH | tail -1 | awk '{print $4}')
echo "Available space: $AVAILABLE_SPACE KB"

# Convert to MB for easier reading
AVAILABLE_MB=$((AVAILABLE_SPACE / 1024))
echo "Available space: $AVAILABLE_MB MB"

if [ $AVAILABLE_MB -lt 1000 ]; then
    echo "⚠️  WARNING: Less than 1GB available space!"
else
    echo "✅ Sufficient disk space available"
fi
echo

# Check if we can write to current directory
echo "=== Write Permission Check ==="
TEST_FILE="test_write_permissions.tmp"
if touch $TEST_FILE 2>/dev/null; then
    echo "✅ Can write to current directory"
    rm $TEST_FILE
else
    echo "❌ Cannot write to current directory"
fi
echo

# Suggest fixes
echo "=== Recommended Solutions ==="
echo "1. Clean up CACHE directory:"
echo "   rm -rf CACHE/*"
echo
echo "2. Use a different cache location with more space:"
echo "   Edit your training script to change:"
echo "   DATA_INDEX_PATH=CACHE"
echo "   to:"
echo "   DATA_INDEX_PATH=/tmp/maskllm_cache"
echo "   or:"
echo "   DATA_INDEX_PATH=\$HOME/maskllm_cache"
echo
echo "3. Create cache directory with proper permissions:"
echo "   mkdir -p /tmp/maskllm_cache"
echo "   chmod 755 /tmp/maskllm_cache"
echo

# Check for existing problematic cache files
echo "=== Checking for Problematic Cache Files ==="
find . -name "*document_index.npy" -size 0 2>/dev/null | head -10
find . -name "*-GPTDataset-*" 2>/dev/null | head -10

echo
echo "=== Quick Fix Command ==="
echo "To quickly fix this issue, run:"
echo "rm -rf CACHE/* && mkdir -p /tmp/maskllm_cache"
echo "Then edit your training script to use: DATA_INDEX_PATH=/tmp/maskllm_cache" 