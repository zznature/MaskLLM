#!/bin/bash
# 验证Llama8bTokenizer修正效果

echo "🔧 验证Llama8bTokenizer修正效果"

# 1. 检查choices列表
echo "📋 1. 检查arguments.py中的注册..."
if grep -A 10 "choices=\[" megatron/arguments.py | grep -q "Llama8bTokenizer"; then
    echo "  ✅ Llama8bTokenizer已在arguments.py中注册"
else
    echo "  ❌ Llama8bTokenizer未在arguments.py中注册"
    exit 1
fi

# 2. 检查help信息
echo "📋 2. 检查命令行help信息..."
if python3 tools/preprocess_data.py --help 2>&1 | grep -q "Llama8bTokenizer"; then
    echo "  ✅ Llama8bTokenizer出现在help信息中"
else
    echo "  ❌ Llama8bTokenizer未出现在help信息中"
    exit 1
fi

# 3. 快速tokenizer测试
echo "📋 3. 快速tokenizer构建测试..."
mkdir -p assets/data/c4_llama8b_pretokenized

python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/verify_test \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "🎉 修正成功！Llama8bTokenizer现在工作正常"
    echo "✅ 可以运行完整的C4数据预处理"
else
    echo "❌ 仍有其他问题需要解决"
    exit 1
fi
