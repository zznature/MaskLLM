#!/bin/bash

# 快速验证 NVIDIA PyTorch NGC 容器环境
# 检查 MaskLLM 运行所需的所有组件

echo "=== 验证容器环境 (MaskLLM 专用) ==="

# 检查是否在容器内
if [ ! -f /.singularity.d/startscript ]; then
    echo "警告: 此脚本应在容器内运行"
    echo "请先启动容器: bash run_container_simple.sh"
    exit 1
fi

echo "✅ 容器环境检查开始..."

# 1. 检查 Python 环境
echo ""
echo "1. Python 环境:"
python --version
which python

# 2. 检查 PyTorch
echo ""
echo "2. PyTorch 检查:"
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'当前 GPU: {torch.cuda.current_device()}')
    print(f'GPU 名称: {torch.cuda.get_device_name()}')
"

# 3. 检查 transformer_engine
echo ""
echo "3. Transformer Engine 检查:"
python -c "
import transformer_engine as te
print(f'Transformer Engine 版本: {te.__version__}')
print('✅ transformer_engine 可用')
" 2>/dev/null || echo "❌ transformer_engine 不可用"

# 4. 检查 CUDA 环境
echo ""
echo "4. CUDA 环境检查:"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VERSION: $CUDA_VERSION"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 5. 检查关键库文件
echo ""
echo "5. 关键库文件检查:"
echo "cuDNN:"
ls -la /usr/local/cuda/lib64/libcudnn* 2>/dev/null | head -3 || echo "cuDNN 库未找到"

echo ""
echo "NCCL:"
ls -la /usr/local/cuda/lib64/libnccl* 2>/dev/null | head -3 || echo "NCCL 库未找到"

# 6. 检查 torchrun
echo ""
echo "6. torchrun 检查:"
torchrun --version 2>/dev/null && echo "✅ torchrun 可用" || echo "❌ torchrun 不可用"

# 7. 检查 MaskLLM 脚本
echo ""
echo "7. MaskLLM 脚本检查:"
if [ -f "scripts/oneshot/run_llama2_7b_prune_tp4.sh" ]; then
    echo "✅ MaskLLM 脚本存在"
    echo "脚本路径: scripts/oneshot/run_llama2_7b_prune_tp4.sh"
else
    echo "❌ MaskLLM 脚本不存在"
fi

# 8. 检查配置文件
echo ""
echo "8. 配置文件检查:"
if [ -f "scripts/oneshot/llama2_7b_prune_tp4.yaml" ]; then
    echo "✅ 配置文件存在"
    echo "配置文件: scripts/oneshot/llama2_7b_prune_tp4.yaml"
else
    echo "❌ 配置文件不存在"
fi

# 9. 检查 tasks 目录
echo ""
echo "9. tasks 目录检查:"
if [ -f "tasks/main.py" ]; then
    echo "✅ tasks/main.py 存在"
else
    echo "❌ tasks/main.py 不存在"
fi

echo ""
echo "=== 环境验证完成 ==="
echo ""
echo "如果所有检查都通过，可以运行:"
echo "bash scripts/oneshot/run_llama2_7b_prune_tp4.sh SparseGPT"
echo ""
echo "或者使用简化启动脚本:"
echo "bash run_container_simple.sh scripts/oneshot/run_llama2_7b_prune_tp4.sh SparseGPT" 