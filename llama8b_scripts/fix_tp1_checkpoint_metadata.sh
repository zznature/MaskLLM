#!/bin/bash

# 修复TP=1 checkpoint元数据
# 为评测创建必需的latest_checkpointed_iteration.txt文件

set -e

PROJECT_DIR="/data/home/zdhs0054/zzhou/MaskLLM"
TP1_CHECKPOINT_DIR="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp1"
ITERATION_DIR="$TP1_CHECKPOINT_DIR/iter_0002000"

echo "🔧 修复TP=1 checkpoint元数据文件"

# 检查目录是否存在
if [ ! -d "$ITERATION_DIR" ]; then
    echo "❌ TP=1 checkpoint目录不存在: $ITERATION_DIR"
    exit 1
fi

# 创建latest_checkpointed_iteration.txt
echo "📝 创建latest_checkpointed_iteration.txt文件"
echo "2000" > "$TP1_CHECKPOINT_DIR/latest_checkpointed_iteration.txt"

# 验证文件创建
if [ -f "$TP1_CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "✅ 创建成功: $TP1_CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
    echo "   内容: $(cat $TP1_CHECKPOINT_DIR/latest_checkpointed_iteration.txt)"
else
    echo "❌ 创建失败"
    exit 1
fi

# 检查checkpoint文件结构
echo ""
echo "📋 验证checkpoint结构:"
ls -la "$TP1_CHECKPOINT_DIR/"
echo ""
echo "📂 iter_0002000内容:"
ls -la "$ITERATION_DIR/"

echo "✅ TP=1 checkpoint元数据修复完成"
