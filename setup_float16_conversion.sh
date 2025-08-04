#!/bin/bash

# 设置Float16转换相关的脚本权限
# 根据HPC规则，集中所有命令到一个脚本

echo "=== 设置MaskLLM Float16转换脚本权限 ==="
echo ""

echo "设置脚本可执行权限..."
chmod +x convert_checkpoint_to_hf.sh
chmod +x generate_precomputed_mask.sh
chmod +x quick_convert_to_hf.sh
chmod +x convert_hf_to_float16.py
chmod +x scripts/tools/convert_llama2_7b_tp4_to_tp1.sh

echo "✅ 权限设置完成"
echo ""

echo "=== Float16转换脚本清单 ==="
echo ""
echo "主要脚本："
echo "  📄 quick_convert_to_hf.sh           - 一键转换脚本 (推荐)"
echo "  📄 convert_checkpoint_to_hf.sh      - 完整转换流程"
echo "  📄 convert_hf_to_float16.py         - Float16转换工具"
echo "  📄 generate_precomputed_mask.sh     - 生成预计算mask (可选)"
echo ""
echo "辅助脚本："
echo "  📄 scripts/tools/convert_llama2_7b_tp4_to_tp1.sh - TP转换"
echo ""

echo "=== 使用方法 ==="
echo ""
echo "1️⃣ 一键转换 (推荐)："
echo "   bash run_maskllm_native.sh quick_convert_to_hf.sh"
echo ""
echo "2️⃣ 分步转换："
echo "   bash run_maskllm_native.sh convert_checkpoint_to_hf.sh"
echo ""
echo "3️⃣ 生成预计算mask："
echo "   bash run_maskllm_native.sh generate_precomputed_mask.sh"
echo ""

echo "=== Float16转换特性 ==="
echo "  ✅ 模型大小减半 (约7GB → 3.5GB)"
echo "  ✅ 推理速度提升"
echo "  ✅ 显存占用减少"
echo "  ✅ 自动文件大小对比"
echo "  ✅ 安全的safetensors格式"
echo "  ✅ 完整的错误检查"
echo ""

echo "准备就绪! 可以开始转换了 🚀" 