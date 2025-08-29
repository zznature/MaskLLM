#!/bin/bash
# 基于PHASE3_PROGRESS_SUMMARY.md Phase 3.2成功经验的修复测试
# 使用Llama2Tokenizer替代AutoTokenizer

echo "🧪 测试Llama2Tokenizer修复方案（基于Phase 3.2成功经验）"
echo "📋 参考: PHASE3_PROGRESS_SUMMARY.md 第333行和343行"

# 确保输出目录存在
mkdir -p assets/data/c4_llama8b_pretokenized

# 测试单个文件 - 使用Phase 3.2验证成功的配置
echo "Processing ./assets/data/c4/en/c4-train.00000-of-01024.json"

python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_00000 \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 8

if [ $? -eq 0 ]; then
    echo "✅ Llama2Tokenizer修复方案测试成功！"
    echo "🎉 基于Phase 3.2成功经验的解决方案有效"
else
    echo "❌ Llama2Tokenizer测试失败"
    exit 1
fi
