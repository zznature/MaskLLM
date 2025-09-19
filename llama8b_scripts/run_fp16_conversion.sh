#!/bin/bash
#
# Llama8b-MaskLLM FP32→FP16 转换脚本执行器
#

echo "=== Llama8b-MaskLLM FP32→FP16 权重转换 ==="
echo "将32.65GB的FP32模型转换为16.33GB的FP16模型"
echo

# 确保在正确的目录
cd /data/home/zdhs0054/zzhou/MaskLLM

# 设置Python路径
export PYTHONPATH=/data/home/zdhs0054/zzhou/MaskLLM:$PYTHONPATH

# 检查输入文件是否存在
INPUT_FILE="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng.pt"
OUTPUT_FILE="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt"

echo "检查输入文件..."
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 输入文件不存在: $INPUT_FILE"
    echo "请确认检查点文件路径是否正确"
    exit 1
fi

echo "✓ 找到输入文件: $INPUT_FILE"
echo "文件大小: $(du -h "$INPUT_FILE" | cut -f1)"
echo

echo "开始转换 (预计需要5-10分钟)..."
echo

# 运行转换脚本
python3 /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/convert_fp32_to_fp16.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE"

# 检查转换结果
if [ $? -eq 0 ] && [ -f "$OUTPUT_FILE" ]; then
    echo
    echo "=== 转换结果 ==="
    echo "原始文件 (FP32): $(du -h "$INPUT_FILE" | cut -f1)"
    echo "转换后 (FP16): $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo
    echo "✅ 转换成功!"
    echo "FP16模型文件: $OUTPUT_FILE"
    echo
    echo "使用建议:"
    echo "1. 推理时内存占用减半"
    echo "2. 在支持Tensor Core的GPU上速度更快"
    echo "3. 精度损失极小，适合大多数任务"
else
    echo "❌ 转换失败!"
    exit 1
fi

echo "转换完成！"
