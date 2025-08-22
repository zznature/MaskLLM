#!/bin/bash
# CUDA诊断和修复综合脚本
# 先诊断问题，再应用修复，最后验证结果

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

echo "🔧 CUDA 8GPU 诊断和修复综合流程"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"

echo "📋 步骤1: 系统诊断"
echo "----------------------------------------"
echo "🔄 运行CUDA配置诊断..."
bash llama8b_scripts/diagnose_cuda_config.sh
echo ""

echo "📋 步骤2: 问题修复"
echo "----------------------------------------"
echo "🔄 应用CUDA 8GPU配置修复..."
bash llama8b_scripts/fix_cuda_8gpu_config.sh
echo ""

echo "📋 步骤3: 修复验证"
echo "----------------------------------------"
echo "🔄 验证修复效果..."

# 检查最终GPU状态
FINAL_GPU_COUNT=$(python3.10 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")

echo "最终GPU配置状态:"
echo "  可见GPU数量: $FINAL_GPU_COUNT"

if [ "$FINAL_GPU_COUNT" -eq 8 ]; then
    echo "  状态: ✅ 完美 (8/8 GPU)"
    echo ""
    echo "🎉 CUDA配置修复成功！"
    echo ""
    echo "🚀 推荐的下一步操作:"
    echo "1. 运行修复版本的8GPU训练:"
    echo "   bash run_maskllm_native.sh llama8b_scripts/llama8b_mask_only_tp8_c4_fixed.sh 0"
    echo ""
    echo "2. 或者运行原始脚本(现在应该能正常工作):"
    echo "   bash run_maskllm_native.sh llama8b_scripts/llama8b_mask_only_tp8_c4.sh 0"
    
elif [ "$FINAL_GPU_COUNT" -gt 4 ]; then
    echo "  状态: ⚠️ 改善 ($FINAL_GPU_COUNT/8 GPU)"
    echo ""
    echo "🔧 GPU数量有所改善，但未达到8个"
    echo ""
    echo "💡 可以尝试的选项:"
    echo "1. 使用当前可用的GPU数量运行训练"
    echo "2. 调整训练脚本的TP大小匹配可用GPU"
    echo "3. 检查容器启动参数是否包含--gpus all"
    
else
    echo "  状态: ❌ 仍有问题 ($FINAL_GPU_COUNT/8 GPU)"
    echo ""
    echo "🔧 需要进一步调查的问题:"
    echo "1. 容器GPU绑定配置"
    echo "2. 宿主机GPU驱动状态"
    echo "3. 容器运行时参数"
    echo ""
    echo "💡 临时解决方案:"
    echo "1. 使用单GPU测试脚本验证MaskLLM兼容性:"
    echo "   bash run_maskllm_native.sh llama8b_scripts/llama8b_mask_only_single_gpu_test.sh 0"
    echo ""
    echo "2. 使用4GPU版本(如果有4个GPU可见):"
    echo "   bash run_maskllm_native.sh llama8b_scripts/llama8b_mask_only_tp4_c4.sh 0"
fi

echo ""
echo "============================================================"
echo "📊 诊断和修复完成"
echo "============================================================"
echo ""
echo "📁 生成的文件:"
echo "  - llama8b_scripts/llama8b_mask_only_tp8_c4_fixed.sh (修复版8GPU训练脚本)"
echo ""
echo "🎯 Phase 3.4 稀疏化训练兼容性测试可以继续进行！"
