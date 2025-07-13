#!/bin/bash

# 扩展库路径方案 - 绑定更多CUDA相关库
# 解决 libcupti.so.12、libcudnn.so.8、libnccl.so.2 和 nvcc 等工具缺失问题

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

echo "=== 扩展库路径方案 ==="
echo "绑定更多CUDA相关库路径到容器内"
echo "解决 libcupti.so.12、libcudnn.so.8、libnccl.so.2 和 nvcc 等工具缺失问题"
echo ""

echo "Starting container with extended library paths..."

# 扩展库路径方案：绑定更多CUDA相关库
# 注意：不覆盖容器内已有的CUDA库，只补充缺失的库
apptainer run --nv \
    --ipc \
    --writable-tmpfs \
    --bind $HOME:$HOME \
    --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    --bind /data/apps/cuda/12.4/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib \
    --bind /data/apps/cuda/12.4/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64 \
    --bind /data/apps/cuda/12.4/nsight-systems-2023.4.4/target-linux-x64:/opt/nsight-systems-libs \
    --bind /data/apps/cuda/12.4/nsight-compute-2024.1.0/host/target-linux-x64:/opt/nsight-compute-libs \
    --bind /data/apps/cuda/12.4/nvvm/lib64:/usr/local/cuda/nvvm/lib64 \
    --bind /data/apps/cuda/12.4/nvvm/libdevice:/usr/local/cuda/nvvm/libdevice \
    --bind /data/apps/cuda/12.4/bin:/usr/local/cuda/bin \
    --bind /data/apps/cuda/12.4/include:/usr/local/cuda/include \
    --bind ${CONDA_PREFIX}/lib:/opt/conda_libs \
    --bind ${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nccl/lib:/opt/nccl_libs \
    --env PATH="/usr/local/bin:/usr/bin:/usr/sbin:/sbin:/usr/local/cuda/bin" \
    --env PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages" \
    --env LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/opt/conda_libs:/opt/nsight-systems-libs:/opt/nsight-compute-libs:/opt/nccl_libs" \
    --env CUDA_HOME="/usr/local/cuda" \
    --env CUDA_ROOT="/usr/local/cuda" \
    --env CONDA_DEFAULT_ENV="" \
    --env CONDA_PREFIX="" \
    --env CONDA_PYTHON_EXE="" \
    --env CONDA_EXE="" \
    --env CONDA_PROMPT_MODIFIER="" \
    --env CONDA_SHLVL="" \
    --env CONDA_ENVS_PATH="" \
    --env CONDA_PKGS_DIRS="" \
    ./pytorch_24.01-py3.sif 