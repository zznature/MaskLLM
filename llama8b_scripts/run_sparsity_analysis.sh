#!/bin/bash
#
# 分析Llama8b-MaskLLM FP16模型的稀疏率分布
#

echo "=== Llama8b-MaskLLM 稀疏率分析 ==="
echo "分析MaskLLM训练后的权重稀疏分布情况"
echo

# 确保在正确的目录
cd /data/home/zdhs0054/zzhou/MaskLLM

# 设置Python路径
export PYTHONPATH=/data/home/zdhs0054/zzhou/MaskLLM:$PYTHONPATH

# 检查模型文件是否存在
FP32_FILE="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng.pt"
FP16_FILE="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt"

echo "检查模型文件..."

# 检查FP32文件
if [ -f "$FP32_FILE" ]; then
    echo "✓ 找到FP32文件: $FP32_FILE"
    echo "  文件大小: $(du -h "$FP32_FILE" | cut -f1)"
    HAS_FP32=true
else
    echo "⚠️ FP32文件不存在: $FP32_FILE"
    HAS_FP32=false
fi

# 检查FP16文件
if [ -f "$FP16_FILE" ]; then
    echo "✓ 找到FP16文件: $FP16_FILE"
    echo "  文件大小: $(du -h "$FP16_FILE" | cut -f1)"
    HAS_FP16=true
else
    echo "⚠️ FP16文件不存在: $FP16_FILE"
    echo "  建议先运行: bash run_maskllm_native.sh llama8b_scripts/run_fp16_conversion.sh"
    HAS_FP16=false
fi

# 检查是否至少有一个文件
if [ "$HAS_FP32" = false ] && [ "$HAS_FP16" = false ]; then
    echo "❌ 没有找到任何模型文件，无法进行分析"
    exit 1
fi

echo

echo "开始稀疏率分析 (预计需要5-10分钟)..."
echo "这将显示："
echo "- 整体模型稀疏率对比"
echo "- 各层稀疏率分布"
echo "- 权重稀疏模式文本分析"
echo "- 具体权重数值样本"
echo "- FP32/FP16对比分析"
echo

# 运行稀疏率分析脚本 (同时分析两个模型)
echo "运行命令:"
echo "python3 analyze_sparsity.py --fp32 \"$FP32_FILE\" --fp16 \"$FP16_FILE\" --detailed-layers 2"
echo

python3 /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/analyze_sparsity.py \
    --fp32 "$FP32_FILE" \
    --fp16 "$FP16_FILE" \
    --detailed-layers 2

# 检查分析结果
if [ $? -eq 0 ]; then
    echo
    echo "✅ 稀疏率分析完成!"
    echo
    echo "分析说明:"
    echo "- 稀疏率 = 接近0的参数数量 / 总参数数量"
    echo "- 高稀疏率表明模型经过了有效的剪枝"
    echo "- 稀疏分布可能不均匀，注意力权重通常保留较多"
    echo "- MaskLLM训练应该产生结构化的稀疏模式"
else
    echo "❌ 稀疏率分析失败!"
    exit 1
fi

echo
echo "分析完成！"
