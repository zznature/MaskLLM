#!/bin/bash

# 容器内初始化脚本
# 在容器启动时自动设置所有必要的环境

echo "=== 容器初始化开始 ==="

# 创建 cuPTI 符号链接
echo "设置 cuPTI 库..."
cd /usr/local/cuda-12.3/NsightSystems-cli-2023.4.1/target-linux-x64/
if [ ! -L libcupti.so.12 ]; then
    ln -sf libcupti.so.12.3 libcupti.so.12
    echo "✓ 创建 cuPTI 符号链接: libcupti.so.12 -> libcupti.so.12.3"
else
    echo "✓ cuPTI 符号链接已存在"
fi
chmod +x libcupti.so.12.3

# 激活 conda 环境
echo "激活 conda 环境..."
source /data/home/zdhs0054/jpu/software/miniconda3/etc/profile.d/conda.sh
conda activate maskllm

# 验证环境设置
echo ""
echo "=== 环境验证 ==="
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo ""
echo "=== 库文件检查 ==="
echo "cuPTI 库:"
ls -la /usr/local/cuda-12.3/NsightSystems-cli-2023.4.1/target-linux-x64/libcupti.so.12* 2>/dev/null || echo "✗ cuPTI 库未找到"

echo ""
echo "NCCL 库:"
ls -la $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null && echo "✓ NCCL 库存在" || echo "✗ NCCL 库未找到"

echo ""
echo "cuDNN 库:"
ls -la $CONDA_PREFIX/lib/libcudnn.so.8 2>/dev/null && echo "✓ cuDNN 库存在" || echo "✗ cuDNN 库未找到"

echo ""
echo "=== PyTorch 测试 ==="
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null && echo "✓ PyTorch 导入成功！" || echo "✗ PyTorch 导入失败"

echo ""
echo "=== torchrun 测试 ==="
torchrun --version 2>/dev/null && echo "✓ torchrun 可用！" || echo "✗ torchrun 不可用"

echo ""
echo "=== 容器初始化完成 ==="
echo "现在可以运行 MaskLLM 脚本了" 