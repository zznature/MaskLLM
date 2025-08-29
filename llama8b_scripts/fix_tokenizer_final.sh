#!/bin/bash
# Llama8b tokenizer最终修复方案
echo "🔧 Llama8b tokenizer最终修复方案"

# 确保输出目录存在
mkdir -p assets/data/c4_llama8b_pretokenized

echo "📋 问题分析:"
echo "  - Llama8b使用自定义tokenizer结构"
echo "  - 不是标准的tokenizer.model文件"
echo "  - 需要指向包含tokenizer.py的目录"

# 方案: 移除tokenizer-model参数，仅使用vocab-file
echo "🚀 尝试简化配置（仅使用vocab-file）..."

python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_00000 \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama2Tokenizer \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "🎉 简化配置成功！问题已解决"
    echo "✅ 最终工作配置:"
    echo "   --tokenizer-type Llama2Tokenizer"
    echo "   --vocab-file ./assets/checkpoints/Llama8b/vocab.txt"
    echo "   (移除--tokenizer-model参数)"
else
    echo "❌ 简化配置失败，尝试替代方案..."
    
    # 替代方案：检查其他可能的配置
    echo "🔍 检查Llama2Tokenizer是否需要tokenizer-model参数..."
    
    # 尝试使用tokenizer_config.json
    python3 tools/preprocess_data.py \
        --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
        --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_00000 \
        --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
        --tokenizer-type Llama2Tokenizer \
        --tokenizer-model ./assets/checkpoints/Llama8b/tokenizer_config.json \
        --append-eod \
        --workers 1
    
    if [ $? -eq 0 ]; then
        echo "🎉 替代配置成功！"
    else
        echo "❌ 需要深入分析tokenizer实现"
        exit 1
    fi
fi
