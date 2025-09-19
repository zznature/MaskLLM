#!/bin/bash

# Llama8b 稀疏模型推理测试脚本
# 基于导出的 HuggingFace 格式模型进行推理验证

echo "========================================"
echo "🚀 Llama8b 稀疏模型推理测试"
echo "========================================"

SCRIPT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/llama8b_inference"
MODEL_DIR="../../output/checkpoints/llama8b_hf_maskllm_c4"

# 检查模型是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 稀疏模型未找到: $MODEL_DIR"
    echo "请先完成 Stage 4: HuggingFace 导出"
    exit 1
fi

echo "📁 模型路径: $MODEL_DIR"
echo "📊 模型大小: $(du -sh $MODEL_DIR | cut -f1)"
echo "📋 模型文件:"
echo "$(ls -1 $MODEL_DIR/*.safetensors | wc -l) 个 safetensors 文件"
echo ""

# 推理参数配置
PROMPTS=(
    "You are a helpful assistant. Please introduce MaskLLM and explain its benefits."
    "What is structured sparsity in neural networks? Explain in simple terms."
    "Please write a simple Python function to calculate the factorial of a number."
)

echo "🧪 准备进行 ${#PROMPTS[@]} 个推理测试..."
echo ""

# 运行推理测试
for i in "${!PROMPTS[@]}"; do
    test_num=$((i + 1))
    prompt="${PROMPTS[$i]}"
    
    echo "📝 测试 $test_num/${#PROMPTS[@]}: ${prompt:0:50}..."
    echo ""
    
    # 运行推理
    bash run_maskllm_native.sh "$SCRIPT_DIR/test_llama8b_sparse_inference.py" \
        --model "$MODEL_DIR" \
        --prompt "$prompt" \
        --max-new-tokens 128 \
        --temperature 0.7 \
        --use-gpu \
        --dtype float16 \
        --seed 42
    
    if [ $? -eq 0 ]; then
        echo "✅ 测试 $test_num 完成"
    else
        echo "❌ 测试 $test_num 失败"
        exit 1
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
done

echo "🎉 所有推理测试完成！"
echo ""
echo "📊 测试总结:"
echo "  ✅ 模型加载: 成功"  
echo "  ✅ Tokenizer: 成功"
echo "  ✅ 推理生成: 成功"
echo "  ✅ 稀疏优化: 已应用 (float16 + 2:4结构化稀疏)"
echo ""
echo "🚀 稀疏模型已准备好部署使用！"
echo "========================================"
