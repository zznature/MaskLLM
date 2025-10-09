#!/bin/bash

# 磁盘空间监控脚本
# 实时监控checkpoint目录的磁盘使用情况

CHECKPOINT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8_fp32"
LOGS_DIR="/data/home/zdhs0054/zzhou/MaskLLM/output/logs/llama8b_presparse_training_fp32"

echo "💾 磁盘空间监控"
echo "==============================================="
echo ""

# 1. 整体磁盘使用情况
echo "📊 整体磁盘使用情况:"
df -h /data | grep -E "Filesystem|/data"
echo ""

# 2. Checkpoint目录大小
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "📁 FP32 Checkpoint目录: $CHECKPOINT_DIR"
    CKPT_SIZE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
    echo "  总大小: $CKPT_SIZE"
    
    # 统计checkpoint数量
    CKPT_COUNT=$(ls -d "$CHECKPOINT_DIR"/iter_* 2>/dev/null | wc -l)
    echo "  Checkpoint数量: $CKPT_COUNT 个"
    
    if [ $CKPT_COUNT -gt 0 ]; then
        echo ""
        echo "  最新5个checkpoint:"
        ls -dt "$CHECKPOINT_DIR"/iter_* 2>/dev/null | head -5 | while read ckpt; do
            SIZE=$(du -sh "$ckpt" 2>/dev/null | cut -f1)
            BASENAME=$(basename "$ckpt")
            echo "    - $BASENAME: $SIZE"
        done
    fi
else
    echo "📁 FP32 Checkpoint目录不存在"
fi
echo ""

# 3. 日志目录大小
if [ -d "$LOGS_DIR" ]; then
    echo "📝 FP32 日志目录: $LOGS_DIR"
    LOG_SIZE=$(du -sh "$LOGS_DIR" 2>/dev/null | cut -f1)
    echo "  总大小: $LOG_SIZE"
    
    LOG_COUNT=$(ls "$LOGS_DIR"/*.log 2>/dev/null | wc -l)
    echo "  日志文件数: $LOG_COUNT 个"
else
    echo "📝 FP32 日志目录不存在"
fi
echo ""

# 4. bf16 checkpoint目录（对比）
BF16_CKPT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8"
if [ -d "$BF16_CKPT_DIR" ]; then
    echo "📁 bf16 Checkpoint目录 (对比): $BF16_CKPT_DIR"
    BF16_SIZE=$(du -sh "$BF16_CKPT_DIR" 2>/dev/null | cut -f1)
    echo "  总大小: $BF16_SIZE"
    
    BF16_COUNT=$(ls -d "$BF16_CKPT_DIR"/iter_* 2>/dev/null | wc -l)
    echo "  Checkpoint数量: $BF16_COUNT 个"
fi
echo ""

# 5. 整个output目录大小
OUTPUT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/output"
if [ -d "$OUTPUT_DIR" ]; then
    echo "📦 Output总目录: $OUTPUT_DIR"
    OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "  总大小: $OUTPUT_SIZE"
    echo ""
    echo "  子目录大小："
    du -sh "$OUTPUT_DIR"/*/ 2>/dev/null | sort -rh | head -10
fi
echo ""

# 6. 空间预警
echo "⚠️  空间预警检查:"
AVAIL_GB=$(df /data | tail -1 | awk '{print $4}')
AVAIL_GB_NUM=$(echo "scale=0; $AVAIL_GB / 1024 / 1024" | bc 2>/dev/null)

if [ -z "$AVAIL_GB_NUM" ]; then
    # 如果bc失败，直接从df -h获取
    AVAIL_GB_NUM=$(df -h /data | tail -1 | awk '{print $4}' | sed 's/G//')
fi

echo "  可用空间: ${AVAIL_GB_NUM}GB"

if [ "$AVAIL_GB_NUM" -lt 100 ]; then
    echo "  🔴 警告: 可用空间 < 100GB，建议立即清理！"
    echo ""
    echo "  建议操作："
    echo "    1. 运行清理脚本: bash llama8b_scripts/cleanup_old_checkpoints.sh"
    echo "    2. 手动删除不需要的checkpoint"
    echo "    3. 压缩或转移旧日志文件"
elif [ "$AVAIL_GB_NUM" -lt 200 ]; then
    echo "  🟡 注意: 可用空间 < 200GB，建议定期清理"
    echo ""
    echo "  建议操作："
    echo "    - 定期运行: bash llama8b_scripts/cleanup_old_checkpoints.sh"
else
    echo "  🟢 正常: 可用空间充足 (>${AVAIL_GB_NUM}GB)"
fi
echo ""

# 7. 预估checkpoint增长
if [ -d "$CHECKPOINT_DIR" ] && [ $CKPT_COUNT -gt 0 ]; then
    echo "📈 Checkpoint空间预估:"
    
    # 计算平均checkpoint大小
    TOTAL_CKPT_SIZE=$(du -sk "$CHECKPOINT_DIR"/iter_* 2>/dev/null | awk '{sum+=$1} END {print sum}')
    AVG_CKPT_SIZE_GB=$(echo "scale=1; $TOTAL_CKPT_SIZE / $CKPT_COUNT / 1024 / 1024" | bc)
    
    echo "  当前checkpoint数量: $CKPT_COUNT"
    echo "  平均checkpoint大小: ${AVG_CKPT_SIZE_GB}GB"
    
    # 预估2000次迭代需要的空间（每50次保存）
    TOTAL_CKPTS=40  # 2000 / 50
    REMAINING_CKPTS=$((TOTAL_CKPTS - CKPT_COUNT))
    
    if [ $REMAINING_CKPTS -gt 0 ]; then
        EST_ADDITIONAL_GB=$(echo "scale=1; $AVG_CKPT_SIZE_GB * $REMAINING_CKPTS" | bc)
        echo "  预计还需保存: $REMAINING_CKPTS 个checkpoint"
        echo "  预计还需空间: ${EST_ADDITIONAL_GB}GB"
        
        if [ $(echo "$EST_ADDITIONAL_GB > $AVAIL_GB_NUM" | bc) -eq 1 ]; then
            echo "  🔴 警告: 剩余空间不足以完成训练！"
            echo "  建议: 减少保存频率或启用定期清理"
        else
            echo "  🟢 空间充足，可以完成训练"
        fi
    fi
fi

echo ""
echo "==============================================="
echo "监控完成"

