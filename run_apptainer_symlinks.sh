#!/bin/bash

# 符号链接方案 - 在容器内创建缺失库文件的符号链接
# 解决 libcupti.so.12 等库文件缺失问题

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

echo "=== 符号链接方案 ==="
echo "在容器内创建缺失库文件的符号链接"
echo ""

echo "Starting container with symbolic links..."

# 符号链接方案：在容器内创建缺失库文件的符号链接
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
    ./pytorch_24.01-py3.sif \
    bash -c '
        echo "=== 创建符号链接 ==="
        
        # 检查并创建 libcupti.so.12 符号链接
        if [ ! -f "/usr/local/cuda/lib64/libcupti.so.12" ] && [ -f "/usr/local/cuda/lib64/libcupti.so" ]; then
            echo "创建 libcupti.so.12 符号链接..."
            ln -sf /usr/local/cuda/lib64/libcupti.so /usr/local/cuda/lib64/libcupti.so.12
        fi
        
        # 检查并创建其他可能的缺失库文件符号链接
        for lib in libcudnn.so.8 libcublas.so.12 libcurand.so.10; do
            if [ ! -f "/usr/local/cuda/lib64/$lib" ] && [ -f "/usr/local/cuda/lib64/${lib%.*}" ]; then
                echo "创建 $lib 符号链接..."
                ln -sf /usr/local/cuda/lib64/${lib%.*} /usr/local/cuda/lib64/$lib
            fi
        done
        
        echo "符号链接创建完成"
        echo ""
        
        # 启动交互式shell
        exec bash
    ' 