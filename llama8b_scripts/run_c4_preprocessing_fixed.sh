#!/bin/bash
# Llama8b C4数据预处理 - 基于Phase 3.2成功经验
# 参考: PHASE3_PROGRESS_SUMMARY.md - NullTokenizer解决方案

echo "🚀 Llama8b C4数据预处理 (基于Phase 3.2成功经验)"
echo "📋 使用NullTokenizer替代Llama8bTokenizer"

# 确保输出目录存在
mkdir -p assets/data/c4_llama8b_pretokenized

# 基于Phase 3.2成功配置的数据预处理
for i in $(printf "%05d" {0..19}); do
    echo "Processing ./assets/data/c4/en/c4-train.${i}-of-01024.json"
    
    python3 tools/preprocess_data.py \
        --input "./assets/data/c4/en/c4-train.${i}-of-01024.json" \
        --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_${i} \
        --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
        --tokenizer-type NullTokenizer \
        --vocab-size 119696 \
        --append-eod \
        --workers 64 || {
        echo "❌ 处理文件 ${i} 失败"
        exit 1
    }
done

echo "✅ Llama8b C4数据预处理完成！"
