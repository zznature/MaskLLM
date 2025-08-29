#!/bin/bash
# 使用正确的Llama8bTokenizer进行测试

echo "🧪 测试正确的Llama8bTokenizer配置"
echo "📋 关键发现:"
echo "  - Llama2Tokenizer需要.model文件（SentencePiece）"
echo "  - Llama8b使用自定义tokenizer.py实现"
echo "  - 应该使用Llama8bTokenizer类型"

# 确保输出目录存在
mkdir -p assets/data/c4_llama8b_pretokenized

# 使用正确的Llama8bTokenizer
echo "🚀 使用Llama8bTokenizer测试..."

python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_00000 \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "🎉 Llama8bTokenizer成功！"
    echo "✅ 找到了正确的tokenizer类型"
else
    echo "❌ Llama8bTokenizer失败，检查注册问题"
    
    # 检查Llama8bTokenizer是否在arguments.py中注册
    echo "🔍 检查tokenizer注册状态..."
    if grep -q "Llama8bTokenizer" megatron/arguments.py; then
        echo "✅ Llama8bTokenizer已在arguments.py中注册"
    else
        echo "❌ Llama8bTokenizer未在arguments.py中注册"
        echo "💡 需要添加到choices列表中"
    fi
    
    # 检查实现文件
    if [ -f "megatron/tokenizer/llama8b_tokenizer.py" ]; then
        echo "✅ llama8b_tokenizer.py实现文件存在"
    else
        echo "❌ llama8b_tokenizer.py实现文件缺失"
    fi
fi
