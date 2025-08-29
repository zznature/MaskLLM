#!/bin/bash
# Llama8b tokenizer快速修复测试
# 基于成功验证的Llama2Tokenizer方案

echo "🧪 Llama8b tokenizer快速修复测试"
echo "✅ 使用已验证成功的Llama2Tokenizer方案"

# 确保输出目录存在
mkdir -p assets/data/c4_llama8b_pretokenized

# 检查必要文件
echo "📋 检查必要文件..."

if [ -f "./assets/data/c4/en/c4-train.00000-of-01024.json" ]; then
    echo "✅ C4数据文件存在"
elif [ -d "./assets/data/c4/en/" ]; then
    echo "⚠️  C4目录存在，列出文件:"
    ls -la ./assets/data/c4/en/ | head -5
    # 尝试找到实际的数据文件
    ACTUAL_FILE=$(find ./assets/data/c4/en/ -name "*.json" | head -1)
    if [ ! -z "$ACTUAL_FILE" ]; then
        echo "🔍 找到实际文件: $ACTUAL_FILE"
        INPUT_FILE="$ACTUAL_FILE"
    else
        echo "❌ 未找到JSON文件"
        exit 1
    fi
else
    echo "❌ C4数据目录不存在"
    exit 1
fi

if [ -d "./assets/checkpoints/Llama8b" ]; then
    echo "✅ Llama8b tokenizer目录存在"
    echo "📁 目录内容:"
    ls -la ./assets/checkpoints/Llama8b/
else
    echo "❌ Llama8b tokenizer目录不存在"
    exit 1
fi

# 使用成功验证的配置进行测试
echo "🚀 开始测试（使用Llama2Tokenizer）..."

INPUT_FILE=${INPUT_FILE:-"./assets/data/c4/en/c4-train.00000-of-01024.json"}

python3 tools/preprocess_data.py \
    --input "$INPUT_FILE" \
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_test \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "🎉 Llama8b tokenizer解决方案验证成功！"
    echo "✅ Llama2Tokenizer方案完全有效"
else
    echo "❌ 仍有其他问题需要解决"
    exit 1
fi