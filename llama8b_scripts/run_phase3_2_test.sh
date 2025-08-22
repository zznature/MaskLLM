#!/bin/bash

# Phase 3.2: 转换后模型加载测试执行脚本
# 确保在正确环境中运行模型加载测试

echo "🚀 Phase 3.2: 转换后模型加载测试"
echo "==========================================="

# 检查当前目录
if [[ ! -f "tools/checkpoint/util.py" ]]; then
    echo "❌ 错误: 请从MaskLLM根目录运行此脚本"
    exit 1
fi

# 检查转换后的模型是否存在
CHECKPOINT_DIR="output/checkpoints/llama8b_megatron_tp8"
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "❌ 错误: 转换后的模型目录不存在: $CHECKPOINT_DIR"
    echo "请先运行Phase 3.1转换验证"
    exit 1
fi

echo "✅ 找到转换后的模型: $CHECKPOINT_DIR"

# 设置Python路径
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:$(pwd)/.ext_pkgs:$(pwd)"

echo "🔧 设置环境变量:"
echo "  PYTHONPATH: $PYTHONPATH"

# 检查基础Python环境
echo ""
echo "🔍 验证Python环境:"
python3.10 -c "
import sys
print(f'Python版本: {sys.version}')

try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f'CUDA设备数: {device_count}')
        for i in range(device_count):
            print(f'  设备{i}: {torch.cuda.get_device_name(i)}')
    else:
        print('⚠️ CUDA不可用')
except ImportError as e:
    print(f'PyTorch导入失败: {e}')

try:
    import transformers
    print(f'Transformers版本: {transformers.__version__}')
except ImportError as e:
    print(f'Transformers导入失败: {e}')

# 检查CUDA环境变量
import os
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
print(f'CUDA_VISIBLE_DEVICES: {cuda_visible}')
"

echo ""
echo "🔧 检查NVIDIA设备:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | head -8
else
    echo "⚠️ nvidia-smi命令不可用"
fi

echo ""
echo "🧪 开始执行模型加载测试..."
echo ""

# 执行测试
if python3.10 llama8b_scripts/test_model_loading_phase3_2.py; then
    echo ""
    echo "🎉 Phase 3.2测试完成！"
    
    # 显示测试结果文件
    if [[ -f "llama8b_scripts/phase3_2_results.log" ]]; then
        echo ""
        echo "📋 测试结果已保存到: llama8b_scripts/phase3_2_results.log"
    fi
    
    echo ""
    echo "✅ 模型加载测试成功，可以进行Phase 3.3推理一致性验证"
    exit 0
else
    echo ""
    echo "❌ Phase 3.2测试失败"
    echo "请检查错误信息并进行调试"
    exit 1
fi
