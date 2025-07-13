#!/bin/bash

# 系统CUDA环境方案 - 使用系统CUDA而不是容器内CUDA
# 解决库文件缺失问题

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

echo "=== 系统CUDA环境方案 ==="
echo "使用系统CUDA环境，容器内只使用Python和PyTorch"
echo ""

echo "Starting container with system CUDA environment..."

# 系统CUDA环境方案：使用系统CUDA，容器内只使用Python和PyTorch
apptainer run --nv \
    --ipc \
    --writable-tmpfs \
    --bind $HOME:$HOME \
    --bind /data/apps/cuda/12.4:/usr/local/cuda \
    --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    --env PATH="/usr/local/bin:/usr/bin:/usr/sbin:/sbin" \
    --env PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages" \
    --env LD_LIBRARY_PATH="/data/apps/cuda/12.4/lib64:/data/apps/cuda/12.4/extras/CUPTI/lib64:/data/apps/cuda/12.4/targets/x86_64-linux/lib:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib" \
    --env CUDA_HOME="/data/apps/cuda/12.4" \
    --env CUDA_ROOT="/data/apps/cuda/12.4" \
    --env CONDA_DEFAULT_ENV="" \
    --env CONDA_PREFIX="" \
    --env CONDA_PYTHON_EXE="" \
    --env CONDA_EXE="" \
    --env CONDA_PROMPT_MODIFIER="" \
    --env CONDA_SHLVL="" \
    --env CONDA_ENVS_PATH="" \
    --env CONDA_PKGS_DIRS="" \
    ./pytorch_24.01-py3.sif 