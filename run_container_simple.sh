#!/bin/bash

# 简化的 NVIDIA PyTorch NGC 容器启动脚本
# 使用容器内的原生环境运行 MaskLLM，混合方案保留必要库路径

echo "=== 启动 NVIDIA PyTorch NGC 容器 (混合库路径原生环境) ==="

# 加载必要的模块
module load apptainer
module load cuda/12.4

# 设置 GPU 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置系统限制
ulimit -l unlimited
ulimit -s 67108864

# 获取 conda 环境路径（用于库文件）
CONDA_PREFIX="/data/home/zdhs0054/jpu/software/miniconda3/envs/maskllm"

# 检查是否提供了脚本参数
if [ $# -eq 0 ]; then
    echo "启动容器交互模式..."
    echo "容器内已预装:"
    echo "  - Python 3.10.12"
    echo "  - PyTorch 2.2.0a0+81ea7a4"
    echo "  - transformer_engine 1.2"
    echo "  - CUDA 12.3.2.001"
    echo "  - cuDNN 8.9.7.29"
    echo "  - NCCL 2.19.4"
    echo "  - torchrun: /usr/local/bin/torchrun"
    echo "  - 混合库路径: 容器内 + 外部必要库"
    echo ""
    echo "在容器内可以运行:"
    echo "  bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4.sh SparseGPT"
    echo ""
    
    # 启动容器，混合方案：原生环境 + 必要库路径
    apptainer run --nv \
        --ipc \
        --writable-tmpfs \
        --bind $HOME:$HOME \
        --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
        --bind /data/apps/cuda/12.4/lib64:/usr/local/cuda/lib64 \
        --env PATH="/usr/local/bin:/usr/bin:/usr/sbin:/sbin" \
        --env PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages" \
        --env LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib" \
        --env CONDA_DEFAULT_ENV="" \
        --env CONDA_PREFIX="" \
        --env CONDA_PYTHON_EXE="" \
        --env CONDA_EXE="" \
        --env CONDA_PROMPT_MODIFIER="" \
        --env CONDA_SHLVL="" \
        --env CONDA_ENVS_PATH="" \
        --env CONDA_PKGS_DIRS="" \
        ./pytorch_24.01-py3.sif
else
    # 直接运行指定的脚本
    SCRIPT_PATH="$1"
    shift
    
    echo "直接运行脚本: $SCRIPT_PATH"
    echo "参数: $@"
    echo ""
    echo "容器环境信息:"
    echo "  - Python 3.10.12"
    echo "  - PyTorch 2.2.0a0+81ea7a4"
    echo "  - transformer_engine 1.2"
    echo "  - CUDA 12.3.2.001"
    echo "  - torchrun: /usr/local/bin/torchrun"
    echo "  - 混合库路径: 容器内 + 外部必要库"
    echo ""
    
    # 在容器内直接运行脚本，使用混合方案
    apptainer exec --nv \
        --ipc \
        --writable-tmpfs \
        --bind $HOME:$HOME \
        --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
        --bind /data/apps/cuda/12.4/lib64:/usr/local/cuda/lib64 \
        --env PATH="/usr/local/bin:/usr/bin:/usr/sbin:/sbin" \
        --env PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages" \
        --env LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib" \
        --env CONDA_DEFAULT_ENV="" \
        --env CONDA_PREFIX="" \
        --env CONDA_PYTHON_EXE="" \
        --env CONDA_EXE="" \
        --env CONDA_PROMPT_MODIFIER="" \
        --env CONDA_SHLVL="" \
        --env CONDA_ENVS_PATH="" \
        --env CONDA_PKGS_DIRS="" \
        ./pytorch_24.01-py3.sif \
        bash "$SCRIPT_PATH" "$@"
fi 