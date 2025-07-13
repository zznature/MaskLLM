#!/bin/bash

# MaskLLM 容器环境设置脚本
# 解决库文件缺失问题 (cuDNN, cuPTI, NCCL)
# 使用容器内 Python 3.10 而不是 conda Python 3.12

echo "=== MaskLLM 容器环境设置 (使用 Python 3.10) ==="

# 检查是否在容器内
if [ ! -f /.singularity.d/startscript ]; then
    echo "警告: 此脚本应在容器内运行"
    echo "请先启动容器: bash run_apptainer.sh"
    exit 1
fi

# 设置 Python 环境使用容器内的 Python 3.10
echo "设置 Python 环境使用容器内的 Python 3.10..."
export PYTHONPATH="/usr/local/lib/python3.10/site-packages:/usr/lib/python3.10/site-packages:$PYTHONPATH"
export PATH="/usr/local/bin:/usr/bin:$PATH"

# 激活 conda 环境（但主要使用容器内的 Python 3.10）
echo "激活 conda 环境..."
source /data/home/zdhs0054/jpu/software/miniconda3/etc/profile.d/conda.sh
conda activate maskllm

# 设置完整的库路径
echo "设置库路径..."
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/NsightSystems-cli-2023.4.1/target-linux-x64:/usr/local/cuda/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 创建 cuPTI 符号链接
echo "设置 cuPTI 库..."
cd /usr/local/cuda-12.3/NsightSystems-cli-2023.4.1/target-linux-x64/
if [ ! -L libcupti.so.12 ]; then
    ln -sf libcupti.so.12.3 libcupti.so.12
    echo "创建 cuPTI 符号链接: libcupti.so.12 -> libcupti.so.12.3"
fi
chmod +x libcupti.so.12.3

# 验证环境设置
echo ""
echo "=== 环境变量设置完成 ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "PYTHONPATH: $PYTHONPATH"

echo ""
echo "=== Python 环境检查 ==="
echo "当前 Python 版本:"
python3.10 --version 2>/dev/null || echo "Python 3.10 不可用"
python --version 2>/dev/null || echo "默认 Python 不可用"

echo ""
echo "=== Transformer Engine 检查 ==="
echo "检查 Python 3.10 中的 transformer_engine:"
python3.10 -c "import transformer_engine; print('✓ Python 3.10 中找到 transformer_engine')" 2>/dev/null || echo "✗ Python 3.10 中没有 transformer_engine"

echo "检查默认 Python 中的 transformer_engine:"
python -c "import transformer_engine; print('✓ 默认 Python 中找到 transformer_engine')" 2>/dev/null || echo "✗ 默认 Python 中没有 transformer_engine"

echo ""
echo "=== 库文件检查 ==="
echo "cuPTI 库:"
ls -la /usr/local/cuda-12.3/NsightSystems-cli-2023.4.1/target-linux-x64/libcupti.so.12* 2>/dev/null || echo "cuPTI 库未找到"

echo ""
echo "NCCL 库:"
ls -la $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null || echo "NCCL 库未找到"

echo ""
echo "cuDNN 库:"
ls -la $CONDA_PREFIX/lib/libcudnn.so.8 2>/dev/null || echo "cuDNN 库未找到"

echo ""
echo "=== PyTorch 测试 ==="
python3.10 -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('CUDA 版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null && echo "✓ PyTorch 导入成功！" || echo "✗ PyTorch 导入失败，请检查环境配置"

echo ""
echo "=== 环境设置完成 ==="
echo "现在可以运行 MaskLLM 脚本了"
echo "建议使用: python3.10 而不是 python 来运行脚本" 