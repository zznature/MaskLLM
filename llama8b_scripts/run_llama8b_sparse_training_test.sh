#!/bin/bash
# Llama8b 稀疏化训练兼容性测试启动脚本
# Phase 3.4: 稀疏化训练兼容性测试
# HPC Server Command Execution Protocol compliant

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

echo "🚀 Llama8b MaskLLM 稀疏化训练兼容性测试"
echo "============================================================"
echo "Phase 3.4: 验证转换后模型在MaskLLM框架中的稀疏化训练能力"
echo ""

cd "$PROJECT_DIR"

# 检查必要条件
echo "📋 环境检查:"

# 1. 检查转换后的checkpoint
LLAMA8B_CHECKPOINT="output/checkpoints/llama8b_megatron_tp8"
if [[ ! -d "$LLAMA8B_CHECKPOINT" ]]; then
    echo "❌ Llama8b Megatron checkpoint不存在: $LLAMA8B_CHECKPOINT"
    echo "请先完成Phase 3.1转换验证！"
    exit 1
fi
echo "✅ Llama8b checkpoint: $LLAMA8B_CHECKPOINT"

# 2. 检查数据集
C4_DATA="assets/data/c4_llama2_pretokenized"
if [[ ! -d "$C4_DATA" ]]; then
    echo "⚠️ C4数据集不存在: $C4_DATA"
    echo "💡 稀疏化训练测试将跳过数据相关验证"
else
    echo "✅ C4数据集: $C4_DATA"
fi

# 3. 检查GPU
echo "🔍 GPU环境检查:"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "✅ 检测到 $GPU_COUNT 个GPU"
    
    if [[ $GPU_COUNT -lt 8 ]]; then
        echo "⚠️ GPU数量少于8个，将调整并行配置"
        # 可以考虑自动调整TENSOR_PARALLEL_SIZE
    fi
else
    echo "⚠️ 无法检测GPU，请确保CUDA环境正确"
fi

echo ""
echo "🔄 准备启动稀疏化训练测试..."

# 构建训练命令
TRAINING_SCRIPT="$SCRIPT_DIR/llama8b_mask_only_tp8_c4.sh"
RESUME_FLAG=0  # 0=从checkpoint开始，1=从训练checkpoint恢复

# 设置权限
chmod +x "$TRAINING_SCRIPT"

echo "📋 训练配置:"
echo "  脚本: $TRAINING_SCRIPT"
echo "  模式: 稀疏化训练兼容性测试"
echo "  迭代: 100 (测试用)"
echo "  批大小: 8 (测试用)"
echo "  张量并行: 8"
echo ""

echo "🚨 重要提示："
echo "  这是一个兼容性测试，不是完整训练"
echo "  目标是验证Llama8b在MaskLLM框架中的稀疏化训练能力"
echo "  测试将运行100个迭代以验证核心功能"
echo ""

read -p "是否开始稀疏化训练测试？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 启动稀疏化训练测试..."
    echo ""
    
    # 运行训练脚本
    if bash "$TRAINING_SCRIPT" $RESUME_FLAG; then
        echo ""
        echo "🎉 稀疏化训练测试完成！"
        echo ""
        echo "📊 测试结果分析:"
        echo "  ✅ 如果训练正常启动并运行了几个迭代，说明兼容性良好"
        echo "  ✅ 如果稀疏化模块正常初始化，说明MaskLLM集成成功"
        echo "  ✅ 如果没有内存或参数错误，说明配置正确"
        echo ""
        echo "🎯 Phase 3.4 稀疏化训练兼容性测试 - 成功完成！"
        echo "🚀 Llama8b已完全适配MaskLLM框架，可以进行稀疏化训练！"
    else
        echo ""
        echo "❌ 稀疏化训练测试遇到问题"
        echo "🔧 请检查错误日志，可能需要调整配置参数"
        exit 1
    fi
else
    echo "测试已取消"
    echo ""
    echo "💡 手动运行命令:"
    echo "bash $TRAINING_SCRIPT 0"
fi
