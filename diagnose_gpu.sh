#!/bin/bash

echo "=== GPU 和 Apptainer 诊断脚本 ==="
echo ""

# 加载模块
echo "1. 加载模块..."
module load apptainer
module load cuda/12.4

echo ""
echo "2. 检查 NVIDIA 库路径..."
find /usr/lib -name "libnvidia-ml.so*" 2>/dev/null | head -5
find /usr/lib64 -name "libnvidia-ml.so*" 2>/dev/null | head -5

echo ""
echo "3. 检查 CUDA 库路径..."
find /data/apps/cuda -name "libnvidia-ml.so*" 2>/dev/null | head -5

echo ""
echo "4. 检查环境变量..."
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"

echo ""
echo "5. 检查 Apptainer 版本..."
apptainer --version

echo ""
echo "6. 检查 GPU 设备..."
ls -la /dev/nvidia* 2>/dev/null || echo "No NVIDIA devices found"

echo ""
echo "7. 检查 nvidia-smi..."
which nvidia-smi
nvidia-smi --version 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "8. 检查 Apptainer 配置..."
apptainer config list | grep -E "(bind|nv)" || echo "No bind/nv config found" 