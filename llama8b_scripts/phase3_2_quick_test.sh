#!/bin/bash

# Phase 3.2: 快速测试脚本
# 提供简单的命令让用户逐个执行

echo "🚀 Phase 3.2: 快速模型加载测试"
echo "========================================="
echo ""

echo "🔧 修复说明:"
echo "已修复 run_maskllm_native.sh 中的Python脚本识别问题"
echo "现在 'test' 和 'phase' 关键词的脚本会被正确识别为测试脚本"
echo ""

echo "📋 请按顺序执行以下命令:"
echo ""

echo "1️⃣ 检查CUDA环境 (应该显示8个GPU):"
echo "bash run_maskllm_native.sh -c 'python3.10 -c \"
import torch
print(f\"CUDA设备数: {torch.cuda.device_count()}\")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f\"设备{i}: {torch.cuda.get_device_name(i)}\")
\"'"
echo ""

echo "2️⃣ 验证转换模型文件结构:"
echo "ls -la output/checkpoints/llama8b_megatron_tp8/iter_0000001/ | grep mp_rank | wc -l"
echo "# 应该显示8 (表示8个TP分片)"
echo ""

echo "3️⃣ 快速Megatron导入测试:"
echo "bash run_maskllm_native.sh -c 'python3.10 -c \"
import sys
sys.path.insert(0, \\\"/data/home/zdhs0054/zzhou/MaskLLM\\\")
from megatron.arguments import parse_args
print(\\\"✅ Megatron导入成功\\\")
\"'"
echo ""

echo "4️⃣ 运行完整测试 (修复后):"
echo "bash run_maskllm_native.sh llama8b_scripts/simple_load_test.py"
echo ""

echo "5️⃣ 如果步骤4成功，运行完整加载测试:"
echo "bash run_maskllm_native.sh llama8b_scripts/test_model_loading_phase3_2.py"
echo ""

echo "🎯 预期结果:"
echo "  ✅ 步骤1: 显示8个GPU设备"
echo "  ✅ 步骤2: 显示数字8 (8个TP分片)"
echo "  ✅ 步骤3: 显示 '✅ Megatron导入成功'"
echo "  ✅ 步骤4: 完整诊断测试通过"
echo "  ✅ 步骤5: 模型加载测试成功"
echo ""

echo "⚠️  如果遇到问题，请提供具体的错误输出。"
