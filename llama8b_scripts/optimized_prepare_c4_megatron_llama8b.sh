#!/bin/bash
# C4数据集转换为Llama8b Megatron格式 - 优化版
# 基于成功测试结果进行性能和稳定性优化

echo "🚀 Llama8b C4数据预处理 - 优化版"
echo "====================================="

# 配置参数
TOTAL_FILES=20
WORKERS=32  # 从64减少到32，避免资源竞争
START_FILE=${1:-0}    # 允许指定起始文件（默认0）
END_FILE=${2:-19}     # 允许指定结束文件（默认19）

echo "📋 处理配置:"
echo "  - 起始文件: ${START_FILE}"
echo "  - 结束文件: ${END_FILE}"  
echo "  - 并行workers: ${WORKERS}"
echo "  - Tokenizer: Llama8bTokenizer"
echo "  - 词汇表大小: 119,696"

# 创建输出目录
mkdir -p assets/data/c4_llama8b_pretokenized

# 统计变量
SUCCESS_COUNT=0
FAILED_COUNT=0
START_TIME=$(date +%s)

# 处理C4数据集文件
for i in $(seq -f "%05g" $START_FILE $END_FILE); do
    echo ""
    echo "📄 处理文件 ${i} ($(($i - $START_FILE + 1))/$(($END_FILE - $START_FILE + 1)))..."
    
    INPUT_FILE="./assets/data/c4/en/c4-train.${i}-of-01024.json"
    OUTPUT_PREFIX="assets/data/c4_llama8b_pretokenized/c4_llama8b_${i}"
    
    # 检查输入文件是否存在
    if [ ! -f "$INPUT_FILE" ]; then
        echo "  ❌ 输入文件不存在: $INPUT_FILE"
        ((FAILED_COUNT++))
        continue
    fi
    
    # 检查输出文件是否已存在
    if [ -f "${OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
        echo "  ⏭️  输出文件已存在，跳过: ${OUTPUT_PREFIX}"
        ((SUCCESS_COUNT++))
        continue
    fi
    
    echo "  🔄 正在处理: $INPUT_FILE"
    
    # 执行预处理
    python3 tools/preprocess_data.py \
        --input "$INPUT_FILE" \
        --output-prefix "$OUTPUT_PREFIX" \
        --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
        --tokenizer-type Llama8bTokenizer \
        --tokenizer-model ./assets/checkpoints/Llama8b \
        --append-eod \
        --workers $WORKERS
    
    if [ $? -eq 0 ]; then
        echo "  ✅ 成功完成: ${i}"
        ((SUCCESS_COUNT++))
        
        # 验证输出文件
        if [ -f "${OUTPUT_PREFIX}_text_document.bin" ] && [ -f "${OUTPUT_PREFIX}_text_document.idx" ]; then
            BIN_SIZE=$(ls -lh "${OUTPUT_PREFIX}_text_document.bin" | awk '{print $5}')
            echo "  📊 输出文件大小: ${BIN_SIZE}"
        else
            echo "  ⚠️  警告: 输出文件可能不完整"
        fi
    else
        echo "  ❌ 处理失败: ${i}"
        ((FAILED_COUNT++))
        
        # 清理可能的不完整文件
        rm -f "${OUTPUT_PREFIX}_text_document.bin" "${OUTPUT_PREFIX}_text_document.idx" 2>/dev/null
    fi
    
    # 显示进度
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    PROGRESS=$(($i - $START_FILE + 1))
    TOTAL=$(($END_FILE - $START_FILE + 1))
    
    if [ $PROGRESS -gt 0 ]; then
        AVG_TIME=$((ELAPSED / PROGRESS))
        REMAINING=$(($TOTAL - $PROGRESS))
        ETA=$((REMAINING * AVG_TIME))
        
        echo "  📈 进度: ${PROGRESS}/${TOTAL} ($(($PROGRESS * 100 / $TOTAL))%)"
        echo "  ⏱️  已用时: ${ELAPSED}s, 预计剩余: ${ETA}s"
    fi
done

# 最终统计
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "🎯 处理完成统计"
echo "================"
echo "✅ 成功: $SUCCESS_COUNT 个文件"
echo "❌ 失败: $FAILED_COUNT 个文件"
echo "⏱️  总用时: ${TOTAL_TIME}s"

if [ $SUCCESS_COUNT -gt 0 ]; then
    AVG_TIME_PER_FILE=$((TOTAL_TIME / SUCCESS_COUNT))
    echo "📊 平均每文件用时: ${AVG_TIME_PER_FILE}s"
fi

# 列出生成的文件
echo ""
echo "📁 生成的文件:"
ls -lh assets/data/c4_llama8b_pretokenized/ | head -10

if [ $FAILED_COUNT -eq 0 ]; then
    echo ""
    echo "🎉 所有文件处理成功！"
    echo "✅ C4数据集已成功转换为Llama8b Megatron格式"
else
    echo ""
    echo "⚠️  有 $FAILED_COUNT 个文件处理失败，请检查错误信息"
    exit 1
fi
