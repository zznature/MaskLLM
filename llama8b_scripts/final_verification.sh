#!/bin/bash
# Llama8b tokenizer最终验证

echo "🎯 Llama8b tokenizer最终验证"
echo "📋 解决方案总结:"
echo "  ❌ Llama2Tokenizer: 需要.model文件（Llama8b没有）"
echo "  ✅ Llama8bTokenizer: 专为Llama8b设计"

# 检查必要组件
echo "🔍 检查必要组件..."

# 1. 检查arguments.py注册
if grep -q "Llama8bTokenizer" megatron/arguments.py; then
    echo "  ✅ Llama8bTokenizer已在arguments.py中注册"
else
    echo "  ❌ 需要添加Llama8bTokenizer到arguments.py"
    exit 1
fi

# 2. 检查实现文件
if [ -f "megatron/tokenizer/llama8b_tokenizer.py" ]; then
    echo "  ✅ llama8b_tokenizer.py实现文件存在"
else
    echo "  ❌ llama8b_tokenizer.py实现文件缺失"
    exit 1
fi

# 3. 检查Llama8b模型文件
if [ -d "./assets/checkpoints/Llama8b" ]; then
    echo "  ✅ Llama8b模型目录存在"
    if [ -f "./assets/checkpoints/Llama8b/vocab.txt" ]; then
        echo "  ✅ vocab.txt文件存在"
    else
        echo "  ❌ vocab.txt文件缺失"
        exit 1
    fi
else
    echo "  ❌ Llama8b模型目录不存在"
    exit 1
fi

# 4. 测试最终配置
echo "🚀 测试最终配置..."
mkdir -p assets/data/c4_llama8b_pretokenized

python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_final_test \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "🎉 Llama8b tokenizer最终验证成功！"
    echo "✅ 所有问题已解决，可以运行完整批处理"
else
    echo "❌ 仍有问题，需要进一步调试"
    exit 1
fi
