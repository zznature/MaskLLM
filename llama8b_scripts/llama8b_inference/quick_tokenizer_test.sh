#!/bin/bash

# 快速测试tokenizer修复
# 用于验证修复后的推理功能

echo "🔧 快速tokenizer修复测试"
echo "=========================="

MODEL_DIR="../../output/checkpoints/llama8b_hf_maskllm_c4"

# Step 1: 修复tokenizer配置
echo "修复tokenizer配置..."
bash run_maskllm_native.sh "../fix_tokenizer_config.py" "$MODEL_DIR"

# Step 2: 快速推理测试
echo ""
echo "运行快速推理测试..."
bash run_maskllm_native.sh "./test_llama8b_sparse_inference.py" \
    --model "$MODEL_DIR" \
    --prompt "What is MaskLLM?" \
    --max-new-tokens 50 \
    --temperature 0.7 \
    --use-gpu \
    --dtype float16

echo ""
echo "✅ 快速测试完成！"
