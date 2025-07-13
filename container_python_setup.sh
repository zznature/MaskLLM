#!/bin/bash

# 容器内 Python 包管理脚本
# 用于在 Apptainer 容器内安装和管理 Python 包

echo "=== 容器内 Python 环境管理 ==="

# 检查容器内 Python 环境
echo "1. 检查 Python 环境..."
apptainer exec --nv pytorch_24.01-py3.sif python3 --version
apptainer exec --nv pytorch_24.01-py3.sif which python3
apptainer exec --nv pytorch_24.01-py3.sif which pip3

echo ""
echo "2. 检查已安装的包..."
apptainer exec --nv pytorch_24.01-py3.sif pip3 list | head -10

echo ""
echo "3. 检查 PyTorch 和 CUDA..."
apptainer exec --nv pytorch_24.01-py3.sif python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

echo ""
echo "4. 安装 Transformer Engine v1.2.1..."
echo "注意：这将在容器内安装包"

# 在容器内安装 Transformer Engine
apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu \
    --bind /data/apps/cuda/12.4/lib64:/usr/local/cuda/lib64 \
    --env PYTHONPATH="/usr/local/lib/python3.10/site-packages:$PYTHONPATH" \
    --env PATH="/usr/local/bin:/usr/bin:$PATH" \
    --env LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
    pytorch_24.01-py3.sif \
    bash -c "
        cd $HOME/zzhou/TransformerEngine
        export CUDNN_INCLUDE_DIR=/usr/include
        export CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so
        pip3 install --use-pep517 -e .
    "

echo ""
echo "5. 验证安装..."
apptainer exec --nv pytorch_24.01-py3.sif python3 -c "
try:
    import transformer_engine as te
    print('✓ Transformer Engine 安装成功')
    print(f'版本: {te.__version__}')
except ImportError as e:
    print(f'✗ Transformer Engine 安装失败: {e}')
" 