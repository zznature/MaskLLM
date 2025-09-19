#!/bin/bash
# Llama8b SafeTensors兼容性修复批处理脚本

set -e

echo "🔧 Llama8b SafeTensors兼容性修复工具"
echo "======================================"

# 检查并修复环境
echo "第一步: 环境检查和修复建议"
bash run_maskllm_native.sh llama8b_scripts/fix_safetensors_compatibility.py

echo ""
echo "第二步: 选择修复方案"
echo ""
echo "方案A: 安装缺失依赖 (推荐)"
echo "  pip install safetensors"
echo "  pip install transformers==4.36.2"
echo ""
echo "方案B: 使用PyTorch原生加载器 (备用)"
echo "  bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/pytorch_native_inference.py --prompt '测试'"
echo ""
echo "请选择修复方案 [A/B]:"
