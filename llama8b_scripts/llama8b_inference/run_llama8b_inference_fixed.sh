#!/bin/bash

# Llama8b 稀疏模型推理测试 - 包含tokenizer修复
# 自动修复tokenizer配置问题并运行推理测试

echo "🔧 Llama8b 稀疏模型推理测试 (包含自动修复)"
echo "========================================"

SCRIPT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/llama8b_inference"
PARENT_SCRIPT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts"
MODEL_DIR="../../output/checkpoints/llama8b_hf_maskllm_c4"

# 检查模型是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 稀疏模型未找到: $MODEL_DIR"
    echo "请先完成 Stage 4: HuggingFace 导出"
    exit 1
fi

echo "📁 模型路径: $MODEL_DIR"
echo "📊 模型大小: $(du -sh $MODEL_DIR | cut -f1)"
echo ""

# Step 1: 修复tokenizer配置
echo "🔧 Step 1: 修复tokenizer配置..."
echo "----------------------------------------"

bash run_maskllm_native.sh "$PARENT_SCRIPT_DIR/fix_tokenizer_config.py" "$MODEL_DIR"

if [ $? -ne 0 ]; then
    echo "❌ tokenizer配置修复失败"
    exit 1
fi

echo "✅ tokenizer配置修复完成"
echo ""

# Step 2: 运行推理测试
echo "🧪 Step 2: 运行推理测试..."
echo "----------------------------------------"

# 测试用例
TEST_PROMPTS=(
    "You are a helpful assistant. Please introduce MaskLLM and explain its benefits."
    "What is structured sparsity in neural networks?"
    "Write a Python function to calculate fibonacci numbers."
)

success_count=0
total_tests=${#TEST_PROMPTS[@]}

for i in "${!TEST_PROMPTS[@]}"; do
    test_num=$((i + 1))
    prompt="${TEST_PROMPTS[$i]}"
    
    echo "📝 测试 $test_num/$total_tests:"
    echo "提示词: ${prompt:0:60}..."
    echo ""
    
    # 运行推理
    bash run_maskllm_native.sh "$SCRIPT_DIR/test_llama8b_sparse_inference.py" \
        --model "$MODEL_DIR" \
        --prompt "$prompt" \
        --max-new-tokens 100 \
        --temperature 0.7 \
        --use-gpu \
        --dtype float16 \
        --seed 42
    
    if [ $? -eq 0 ]; then
        echo "✅ 测试 $test_num 成功"
        success_count=$((success_count + 1))
    else
        echo "❌ 测试 $test_num 失败"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
done

# 显示最终结果
echo "🎉 推理测试完成！"
echo ""
echo "📊 测试总结:"
echo "  成功测试: $success_count/$total_tests"

if [ $success_count -eq $total_tests ]; then
    echo "  ✅ 所有测试通过!"
    echo "  ✅ Llama8b 稀疏模型推理功能正常"
    echo "  ✅ tokenizer 配置已修复"
    echo "  ✅ 模型可以正常部署使用"
    echo ""
    echo "🚀 稀疏模型已准备好进行生产部署！"
else
    echo "  ⚠️ 部分测试失败，请检查错误信息"
fi

echo "========================================"
