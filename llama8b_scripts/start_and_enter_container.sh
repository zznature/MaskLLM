#!/bin/bash

echo "=== 启动并进入MaskLLM容器 ==="

# 确保在正确的目录
cd /data/home/zdhs0054/zzhou/MaskLLM

echo "1. 当前目录: $(pwd)"

echo "2. 检查run_maskllm_native.sh是否存在:"
if [ -f "run_maskllm_native.sh" ]; then
    echo "✓ 找到run_maskllm_native.sh"
    ls -la run_maskllm_native.sh
else
    echo "✗ 未找到run_maskllm_native.sh"
    exit 1
fi

echo ""
echo "3. 启动容器环境..."
echo "执行命令: bash run_maskllm_native.sh"
echo ""

# 启动容器
bash run_maskllm_native.sh
