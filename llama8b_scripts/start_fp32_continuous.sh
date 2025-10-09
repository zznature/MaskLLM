#!/bin/bash

# 快速启动 FP32 连续训练脚本

set -e

echo "🚀 启动 FP32 连续训练..."
echo ""
echo "检查环境..."

# 检查是否在 Apptainer 内
if [ -z "$SINGULARITY_CONTAINER" ] && [ -z "$APPTAINER_CONTAINER" ]; then
    echo "❌ 错误: 必须在 Apptainer 容器内运行"
    echo ""
    echo "请先执行:"
    echo "  bash run_apptainer_extended_libs.sh"
    echo ""
    exit 1
fi

echo "✅ 在容器内"

# 创建必要的目录
mkdir -p /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8_fp32_continuous
mkdir -p /data/home/zdhs0054/zzhou/MaskLLM/output/logs/llama8b_presparse_training_fp32_continuous

echo "✅ 目录已准备"
echo ""
echo "启动训练脚本..."
echo ""

# 启动训练
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_continuous_from_iter1310.sh

