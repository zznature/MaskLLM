#!/bin/bash

# 大临时空间版本 - 解决磁盘空间不足问题
# 专门解决 No space left on device 错误

# 加载模块
module load apptainer
module load cuda/12.4

# 设置 GPU 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置系统限制
ulimit -l unlimited
ulimit -s 67108864

# 清理和设置容器内环境变量（避免重复）
export PYTHONPATH="/usr/local/lib/python3.10/site-packages"
export PATH="/usr/local/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# 设置更大的临时目录
export TMPDIR="/tmp/large_tmp_$$"
mkdir -p $TMPDIR

echo "=== 大临时空间版本 ==="
echo "临时目录: $TMPDIR"
echo "NVIDIA 库路径: /usr/lib/x86_64-linux-gnu/"
echo "CUDA 路径: /data/apps/cuda/12.4/"
echo ""

echo "Starting container with large temporary space..."

# 关键修复：使用更大的临时空间和正确的绑定
apptainer run --nv \
    --ipc \
    --writable-tmpfs \
    --bind $HOME:$HOME \
    --bind $TMPDIR:/tmp \
    --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    --bind /data/apps/cuda/12.4/lib64:/usr/local/cuda/lib64 \
    --bind /data/apps/cuda/12.4/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib \
    --env PYTHONPATH="/usr/local/lib/python3.10/site-packages" \
    --env PATH="/usr/local/bin:/usr/bin:$PATH" \
    --env LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
    --env TMPDIR="/tmp" \
    ../pytorch_24.01-py3.sif

# 清理临时目录
rm -rf $TMPDIR 