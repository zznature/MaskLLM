#!/bin/bash

# PyTorch版本替换方案
# 安装没有HPC-X依赖的PyTorch版本

echo "🔧 PyTorch版本替换方案"
echo "⚠️ 注意: 这会替换当前的PyTorch版本"

read -p "是否继续替换PyTorch版本? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏭️ 跳过PyTorch替换"
    exit 0
fi

echo "🔧 备份当前PyTorch..."
BACKUP_DIR="/tmp/pytorch_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -d "/usr/local/lib/python3.10/dist-packages/torch" ]; then
    cp -r "/usr/local/lib/python3.10/dist-packages/torch" "$BACKUP_DIR/"
    echo "✅ PyTorch已备份到: $BACKUP_DIR"
fi

echo "🔧 安装标准PyTorch版本..."

# 使用清华镜像源安装标准PyTorch
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --force-reinstall --no-deps

if [ $? -eq 0 ]; then
    echo "✅ 标准PyTorch安装成功"
    
    # 测试新PyTorch
    python3.10 -c "
import torch
print(f'新PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
"
    
    echo "💡 如果有问题，可以恢复备份:"
    echo "   rm -rf /usr/local/lib/python3.10/dist-packages/torch"
    echo "   cp -r $BACKUP_DIR/torch /usr/local/lib/python3.10/dist-packages/"
    
else
    echo "❌ PyTorch替换失败"
    echo "💡 恢复备份..."
    if [ -d "$BACKUP_DIR/torch" ]; then
        rm -rf "/usr/local/lib/python3.10/dist-packages/torch"
        cp -r "$BACKUP_DIR/torch" "/usr/local/lib/python3.10/dist-packages/"
        echo "✅ 已恢复原PyTorch版本"
    fi
fi
