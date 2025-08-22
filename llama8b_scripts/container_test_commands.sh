#!/bin/bash

echo "🔍 容器内环境检查："
echo "Python: $(python3.10 --version)"
echo "当前目录: $(pwd)"

echo ""
echo "🔍 检查transformer_engine："
python3.10 -c "
try:
    import transformer_engine as te
    print(f'✅ transformer_engine可用: {te.__version__}')
    print(f'路径: {te.__file__}')
except ImportError as e:
    print(f'❌ transformer_engine不可用: {e}')
"

echo ""
echo "🔍 检查PyTorch CUDA："
python3.10 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA设备数: {torch.cuda.device_count()}')
"

echo ""
echo "🔍 设置环境并运行测试："

# 设置环境变量
export PYTHONPATH="/data/home/zdhs0054/zzhou/MaskLLM:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "PYTHONPATH已设置"
echo "CUDA_VISIBLE_DEVICES已设置为8个GPU"

echo ""
echo "🧪 运行模型加载测试："
python3.10 llama8b_scripts/test_load_args_only.py

