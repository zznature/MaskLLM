#!/bin/bash

# 检查 NVIDIA PyTorch NGC 容器内的原生环境
echo "=== 检查 NVIDIA PyTorch NGC 容器原生环境 ==="

echo ""
echo "1. 检查 Python 版本和路径:"
which python
python --version
which python3
python3 --version

echo ""
echo "2. 检查系统 Python 路径:"
python -c "import sys; print('Python 路径:'); [print(p) for p in sys.path]"

echo ""
echo "3. 检查 PyTorch:"
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo ""
echo "4. 检查 transformer_engine:"
python -c "import transformer_engine; print('transformer_engine 版本:', transformer_engine.__version__)" 2>/dev/null || echo "transformer_engine 不可用"

echo ""
echo "5. 检查 CUDA 环境:"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
ls -la /usr/local/cuda* 2>/dev/null || echo "CUDA 目录不存在"

echo ""
echo "6. 检查已安装的包:"
pip list | grep -E "(torch|transformer|nvidia)" || echo "没有找到相关包"

echo ""
echo "=== 环境检查完成 ===" 