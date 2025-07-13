#!/bin/bash

# CUDA库文件诊断脚本
# 检查系统中可用的CUDA库文件，帮助选择最佳解决方案

echo "=== CUDA库文件诊断 ==="
echo "检查系统中可用的CUDA库文件"
echo ""

# 检查系统CUDA安装
echo "1. 检查系统CUDA安装:"
if [ -d "/data/apps/cuda/12.4" ]; then
    echo "✅ 系统CUDA 12.4 已安装: /data/apps/cuda/12.4"
    
    echo "   CUDA库文件:"
    ls -la /data/apps/cuda/12.4/lib64/ | grep -E "(libcupti|libcudnn|libcublas|libcurand)" | head -10
    
    echo "   CUPTI库文件:"
    if [ -d "/data/apps/cuda/12.4/extras/CUPTI/lib64" ]; then
        ls -la /data/apps/cuda/12.4/extras/CUPTI/lib64/ | grep libcupti
    else
        echo "❌ CUPTI目录不存在"
    fi
    
    echo "   Targets库文件:"
    if [ -d "/data/apps/cuda/12.4/targets/x86_64-linux/lib" ]; then
        ls -la /data/apps/cuda/12.4/targets/x86_64-linux/lib/ | grep -E "(libcupti|libcudnn)" | head -5
    else
        echo "❌ Targets目录不存在"
    fi
else
    echo "❌ 系统CUDA 12.4 未安装"
fi

echo ""

# 检查系统库文件
echo "2. 检查系统库文件:"
echo "   /usr/lib/x86_64-linux-gnu/ 中的CUDA相关库:"
ls -la /usr/lib/x86_64-linux-gnu/ | grep -E "(libcupti|libcudnn|libcublas|libcurand)" | head -10

echo ""

# 检查conda环境中的库文件
echo "3. 检查conda环境中的库文件:"
CONDA_PREFIX="/data/home/zdhs0054/jpu/software/miniconda3/envs/maskllm"
if [ -d "$CONDA_PREFIX/lib" ]; then
    echo "✅ Conda环境库目录存在: $CONDA_PREFIX/lib"
    ls -la $CONDA_PREFIX/lib/ | grep -E "(libcupti|libcudnn|libcublas|libcurand)" | head -10
else
    echo "❌ Conda环境库目录不存在"
fi

echo ""

# 检查容器内的库文件
echo "4. 检查容器内的库文件:"
echo "   容器内 /usr/local/cuda/lib64/ 中的库文件:"
apptainer exec ./pytorch_24.01-py3.sif ls -la /usr/local/cuda/lib64/ | grep -E "(libcupti|libcudnn|libcublas|libcurand)" | head -10

echo ""

# 检查容器内PyTorch库文件
echo "5. 检查容器内PyTorch库文件:"
echo "   容器内PyTorch库目录:"
apptainer exec ./pytorch_24.01-py3.sif find /usr/local/lib/python3.10/dist-packages/torch/lib/ -name "*.so*" | head -10

echo ""

# 测试库文件可用性
echo "6. 测试库文件可用性:"
echo "   测试 libcupti.so.12:"
if [ -f "/data/apps/cuda/12.4/lib64/libcupti.so.12" ]; then
    echo "✅ 系统中有 libcupti.so.12"
elif [ -f "/data/apps/cuda/12.4/lib64/libcupti.so" ]; then
    echo "⚠️  系统中有 libcupti.so，但没有 libcupti.so.12"
else
    echo "❌ 系统中没有 libcupti.so"
fi

echo "   测试 libcudnn.so.8:"
if [ -f "/data/apps/cuda/12.4/lib64/libcudnn.so.8" ]; then
    echo "✅ 系统中有 libcudnn.so.8"
elif [ -f "/data/apps/cuda/12.4/lib64/libcudnn.so" ]; then
    echo "⚠️  系统中有 libcudnn.so，但没有 libcudnn.so.8"
else
    echo "❌ 系统中没有 libcudnn.so"
fi

echo ""

# 推荐方案
echo "=== 推荐方案 ==="
echo "基于诊断结果，推荐以下方案（按优先级排序）:"
echo ""
echo "方案A (推荐): 扩展库路径绑定"
echo "  - 优点: 保持容器原生环境，绑定更多CUDA库路径"
echo "  - 适用: 系统中有完整的CUDA库文件"
echo "  - 命令: bash run_apptainer_extended_libs.sh"
echo ""
echo "方案B: 使用系统CUDA环境"
echo "  - 优点: 完全使用系统CUDA，避免容器内CUDA问题"
echo "  - 适用: 系统CUDA环境完整且兼容"
echo "  - 命令: bash run_apptainer_system_cuda.sh"
echo ""
echo "方案C: 创建符号链接"
echo "  - 优点: 简单直接，解决特定库文件缺失"
echo "  - 适用: 有基础库文件但缺少特定版本"
echo "  - 命令: bash run_apptainer_symlinks.sh"
echo ""
echo "方案D: 降级PyTorch版本"
echo "  - 优点: 使用更兼容的PyTorch版本"
echo "  - 适用: 其他方案都失败时的备选"
echo "  - 命令: bash run_apptainer_downgrade_pytorch.sh"
echo ""
echo "建议先尝试方案A，如果失败再尝试方案B，以此类推。" 