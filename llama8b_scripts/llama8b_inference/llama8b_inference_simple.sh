#!/bin/bash

# 简化的 Llama8b 稀疏模型推理脚本
# 用于快速交互式测试

SCRIPT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/llama8b_inference"
MODEL_DIR="../../output/checkpoints/llama8b_hf_maskllm_c4"

echo "🚀 Llama8b 稀疏模型快速推理"
echo "模型: $MODEL_DIR"
echo ""

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: $0 \"your prompt here\""
    echo ""
    echo "示例:"
    echo "  $0 \"What is MaskLLM?\""
    echo "  $0 \"Write a Python function to sort a list\""
    echo ""
    exit 1
fi

PROMPT="$*"

echo "💬 提示词: $PROMPT"
echo "⚡ 开始推理..."
echo ""

# 运行推理
bash run_maskllm_native.sh "$SCRIPT_DIR/test_llama8b_sparse_inference.py" \
    --model "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --max-new-tokens 256 \
    --temperature 0.8 \
    --use-gpu \
    --dtype float16

echo ""
echo "✅ 推理完成!"
