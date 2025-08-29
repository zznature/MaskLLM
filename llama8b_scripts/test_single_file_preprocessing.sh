#!/bin/bash
# 测试单个文件的C4预处理 - 使用AutoTokenizer
echo "🧪 测试单个文件C4预处理（使用AutoTokenizer）"

# 确保输出目录存在
mkdir -p assets/data/c4_llama8b_pretokenized

# 测试单个文件处理
echo "Processing ./assets/data/c4/en/c4-train.00000-of-01024.json"

python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_00000 \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type AutoTokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 8

echo "✅ 单文件测试完成"
