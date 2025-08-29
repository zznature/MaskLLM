#!/bin/bash

# 综合修复应用脚本
# 一键应用CUDA路径修复 + UCC符号劫持

echo "🔧 应用综合环境修复..."

# 增强UCC劫持库
ENHANCED_LIB="/data/home/zdhs0054/zzhou/MaskLLM/comprehensive_fix/libenhanced_ucc_hijack.so"
if [ -f "$ENHANCED_LIB" ]; then
    export LD_PRELOAD="$ENHANCED_LIB:$LD_PRELOAD"
    echo "✅ 增强UCC劫持库已加载"
else
    echo "❌ 增强劫持库不存在: $ENHANCED_LIB"
    exit 1
fi

# 修复的CUDA库路径
export LD_LIBRARY_PATH="/opt/conda_libs:/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib:/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"

# HPC-X冲突清理
unset HPCX_DIR HPCX_HOME HPCX_ROOT OMPI_HOME MPI_HOME
export OMPI_MCA_pml="^ucx"
export UCX_TLS="^all"
export UCC_TLS="nccl"

# 8GPU配置
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WORLD_SIZE="8"
export TORCH_DISTRIBUTED_BACKEND="nccl"
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# Python路径
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:$(pwd)/.ext_pkgs:$(pwd)"

echo "✅ 综合环境修复已应用"
echo "🚀 PyTorch和8GPU分布式训练环境已就绪"
[ENHANCED_UCC] 增强UCC劫持库已加载
[ENHANCED_UCC] 包含100+符号，解决所有UCC相关问题
