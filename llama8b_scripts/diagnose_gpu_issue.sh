#!/bin/bash

# GPU配置诊断脚本
echo "🔍 GPU配置诊断"
echo "============================================================"

# 检查CUDA环境
echo "📋 CUDA环境检查:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"
echo "WORLD_SIZE: ${WORLD_SIZE:-未设置}"
echo ""

# 检查容器内GPU
echo "📊 容器内GPU检查:"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi可用"
    nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader,nounits
    echo ""
    echo "GPU数量: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"
else
    echo "❌ nvidia-smi不可用"
fi

echo ""

# 检查PyTorch GPU识别
echo "🐍 PyTorch GPU识别:"
python3 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  内存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
else:
    print('CUDA不可用')
"

echo ""

# 检查NCCL配置
echo "🔗 NCCL环境变量:"
env | grep -E "(NCCL|UCX|UCC)" | sort

echo ""

# 检查当前torchrun进程
echo "🏃‍♂️ 当前torchrun进程:"
ps aux | grep torchrun | grep -v grep || echo "无torchrun进程"

echo ""

# 检查GPU占用
echo "💾 GPU内存占用:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
fi

echo ""
echo "============================================================"
echo "诊断完成"
