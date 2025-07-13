#!/bin/bash

# 用户目录安装版本 - 解决根文件系统空间不足问题
# 将Python包安装到用户目录而不是系统目录

# 加载模块
module load apptainer
module load cuda/12.4

# 设置 GPU 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置系统限制
ulimit -l unlimited
ulimit -s 67108864

# 设置用户Python环境
export PYTHONUSERBASE="$HOME/.local"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

echo "=== 用户目录安装版本 ==="
echo "用户Python目录: $PYTHONUSERBASE"
echo "NVIDIA 库路径: /usr/lib/x86_64-linux-gnu/"
echo "CUDA 路径: /data/apps/cuda/12.4/"
echo ""

echo "Starting container with user directory package installation..."

# 关键修复：使用用户目录安装包
apptainer run --nv \
    --ipc \
    --writable-tmpfs \
    --bind $HOME:$HOME \
    --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    --bind /data/apps/cuda/12.4/lib64:/usr/local/cuda/lib64 \
    --bind /data/apps/cuda/12.4/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib \
    --env PYTHONUSERBASE="$HOME/.local" \
    --env PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH" \
    --env PATH="$HOME/.local/bin:$PATH" \
    --env LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
    ../pytorch_24.01-py3.sif 