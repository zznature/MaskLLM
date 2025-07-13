#!/bin/bash

# 降级PyTorch方案 - 使用更兼容的PyTorch版本
# 避免对特定CUDA库的依赖

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

echo "=== 降级PyTorch方案 ==="
echo "使用更兼容的PyTorch版本，避免库文件依赖问题"
echo ""

echo "Starting container with downgraded PyTorch..."

# 降级PyTorch方案：使用更兼容的PyTorch版本
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
        echo "=== 降级PyTorch ==="
        
        # 检查当前PyTorch版本
        python3.10 -c "import torch; print(\"当前PyTorch版本:\", torch.__version__)"
        
        echo "安装更兼容的PyTorch版本..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
        
        echo "验证新版本..."
        python3.10 -c "import torch; print(\"新PyTorch版本:\", torch.__version__); print(\"CUDA 可用:\", torch.cuda.is_available())"
        
        echo "PyTorch降级完成"
        echo ""
        
        # 启动交互式shell
        exec bash
    ' 