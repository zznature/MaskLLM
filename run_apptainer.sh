#!/bin/bash

# 最终修复版 - 基于诊断结果
# 专门解决 Apptainer 1.3.2 的 NVIDIA 库绑定问题
# 使用容器内的原生环境，混合方案保留必要库路径
# 修复库文件缺失问题 (cuDNN, cuPTI, NCCL)

# 加载模块
module load apptainer
module load cuda/12.4

# 设置 GPU 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置系统限制
ulimit -l unlimited
ulimit -s 67108864

# 获取 conda 环境路径（用于库文件）
CONDA_PREFIX="/data/home/zdhs0054/jpu/software/miniconda3/envs/maskllm"

echo "=== 使用容器原生环境方案 (混合库路径) ==="
echo "NVIDIA 库路径: /usr/lib/x86_64-linux-gnu/"
echo "CUDA 路径: /data/apps/cuda/12.4/"
echo "容器内 Python: /usr/bin/python3.10 (原生环境)"
echo "容器内 PyTorch: 2.2.0a0+81ea7a4"
echo "容器内 transformer_engine: 1.2"
echo "容器内 torchrun: /usr/local/bin/torchrun"
echo "混合库路径: 容器内 + 外部必要库"
echo ""

echo "Starting container with native environment and hybrid library paths..."

# 方案6混合方案：使用容器内环境但保留必要的库路径
# 1. 使用容器内原生PATH和PYTHONPATH
# 2. 混合LD_LIBRARY_PATH：容器内 + 外部必要库
# 3. 清空conda环境变量，避免Python解释器冲突
# 4. 确保cuDNN等库文件可用
apptainer run --nv \
    --ipc \
    --writable-tmpfs \
    --bind $HOME:$HOME \
    --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    --bind /data/apps/cuda/12.4/lib64:/usr/local/cuda/lib64 \
    --bind /data/apps/cuda/12.4/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib \
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