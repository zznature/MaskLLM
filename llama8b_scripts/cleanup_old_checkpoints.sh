#!/bin/bash

# Checkpoint自动清理脚本
# 只保留最新N个checkpoint，删除旧的以节省磁盘空间

# 配置
CHECKPOINT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8_fp32"
KEEP_LATEST=3  # 保留最新的N个checkpoint

echo "🗑️  Checkpoint清理脚本"
echo "目标目录: $CHECKPOINT_DIR"
echo "保留策略: 最新 $KEEP_LATEST 个checkpoint"
echo ""

# 检查目录是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ 错误: Checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

# 进入checkpoint目录
cd "$CHECKPOINT_DIR"

# 查找所有iter_*目录（checkpoint目录）
CHECKPOINTS=($(ls -d iter_* 2>/dev/null | sort -V))
TOTAL_CHECKPOINTS=${#CHECKPOINTS[@]}

echo "📊 发现 $TOTAL_CHECKPOINTS 个checkpoint目录"

if [ $TOTAL_CHECKPOINTS -le $KEEP_LATEST ]; then
    echo "✅ Checkpoint数量 ($TOTAL_CHECKPOINTS) <= 保留数量 ($KEEP_LATEST)"
    echo "无需清理"
    exit 0
fi

# 计算要删除的数量
DELETE_COUNT=$((TOTAL_CHECKPOINTS - KEEP_LATEST))
echo "🔍 需要删除 $DELETE_COUNT 个旧checkpoint"
echo ""

# 列出要删除的checkpoint
echo "将要删除的checkpoint:"
for i in $(seq 0 $((DELETE_COUNT - 1))); do
    CKPT="${CHECKPOINTS[$i]}"
    SIZE=$(du -sh "$CKPT" 2>/dev/null | cut -f1)
    echo "  - $CKPT (大小: $SIZE)"
done

echo ""
echo "将要保留的checkpoint:"
for i in $(seq $DELETE_COUNT $((TOTAL_CHECKPOINTS - 1))); do
    CKPT="${CHECKPOINTS[$i]}"
    SIZE=$(du -sh "$CKPT" 2>/dev/null | cut -f1)
    echo "  ✅ $CKPT (大小: $SIZE)"
done

echo ""
read -p "确认删除以上旧checkpoint? (y/N): " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "❌ 取消删除"
    exit 0
fi

echo ""
echo "🗑️  开始删除..."

# 删除旧checkpoint
DELETED=0
FREED_SPACE=0
for i in $(seq 0 $((DELETE_COUNT - 1))); do
    CKPT="${CHECKPOINTS[$i]}"
    SIZE_KB=$(du -sk "$CKPT" 2>/dev/null | cut -f1)
    
    echo "删除: $CKPT"
    rm -rf "$CKPT"
    
    if [ $? -eq 0 ]; then
        DELETED=$((DELETED + 1))
        FREED_SPACE=$((FREED_SPACE + SIZE_KB))
    else
        echo "⚠️  删除失败: $CKPT"
    fi
done

# 转换为GB
FREED_GB=$(echo "scale=2; $FREED_SPACE / 1024 / 1024" | bc)

echo ""
echo "🎉 清理完成!"
echo "  - 删除数量: $DELETED 个checkpoint"
echo "  - 释放空间: ${FREED_GB}GB"
echo "  - 保留数量: $KEEP_LATEST 个checkpoint"

# 显示当前磁盘使用情况
echo ""
echo "📊 当前磁盘使用情况:"
df -h "$CHECKPOINT_DIR" | tail -1

echo ""
echo "✅ Checkpoint目录清理完成"

