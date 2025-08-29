#!/bin/bash

# 容器环境设置脚本 - 在容器内sourcing使用
# 用法: source container_environment_setup.sh

echo "🔧 设置容器环境变量..."

# LD_LIBRARY_PATH设置
export LD_LIBRARY_PATH="/opt/conda_libs:/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib:/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"

# PYTHONPATH设置
CURRENT_DIR="$(pwd)"
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:${CURRENT_DIR}/.ext_pkgs:${CURRENT_DIR}"

# UCC冲突禁用
export NCCL_UCX_DISABLE=1
export TORCH_UCC_DISABLE=1
export UCC_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo "✅ 环境变量设置完成"
echo "📦 PYTHONPATH包含.ext_pkgs: $(echo $PYTHONPATH | grep -o '.ext_pkgs' | wc -l)个路径"
echo "🐳 LD_LIBRARY_PATH包含容器库: $(echo $LD_LIBRARY_PATH | grep -o 'conda_libs' | wc -l)个路径"
