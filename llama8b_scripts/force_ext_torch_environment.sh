#!/bin/bash

# 强制使用.ext_pkgs中的PyTorch环境设置脚本
# 完全绕过系统PyTorch，避免HPC-X冲突

echo "🔧 强制设置.ext_pkgs PyTorch环境..."

# 获取当前目录
CURRENT_DIR="$(pwd)"
EXT_PKGS_DIR="${CURRENT_DIR}/.ext_pkgs"

# 检查.ext_pkgs是否存在
if [ ! -d "$EXT_PKGS_DIR" ]; then
    echo "❌ 错误: .ext_pkgs目录不存在: $EXT_PKGS_DIR"
    exit 1
fi

# 检查PyTorch是否在.ext_pkgs中
if [ ! -d "$EXT_PKGS_DIR/torch" ]; then
    echo "❌ 错误: PyTorch不在.ext_pkgs中: $EXT_PKGS_DIR/torch"
    exit 1
fi

echo "✅ 发现.ext_pkgs PyTorch: $EXT_PKGS_DIR/torch"

# 完全重设PYTHONPATH，优先.ext_pkgs
echo "🔧 重设PYTHONPATH，优先.ext_pkgs..."
export PYTHONPATH="$EXT_PKGS_DIR:$CURRENT_DIR"

# 清理所有可能导致冲突的环境变量
echo "🧹 清理冲突环境变量..."
unset LD_LIBRARY_PATH
unset LD_PRELOAD

# 重建LD_LIBRARY_PATH，只包含必要路径
echo "🔧 重建LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

# 如果有CUDA，添加CUDA路径
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    echo "✅ 添加CUDA库路径"
fi

# 设置CUDA环境（如果存在）
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export CUDA_ROOT="/usr/local/cuda"
    export PATH="/usr/local/cuda/bin:$PATH"
    echo "✅ 设置CUDA环境"
fi

# 强制禁用所有HPC相关组件
echo "🚫 强制禁用HPC组件..."
export OMPI_MCA_pml=^ucx,^hcoll
export OMPI_MCA_btl=^openib,^uct
export UCX_TLS=^ud,^dc,^rc
export UCC_TLS=^ucp,^sharp
export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_IB_DISABLE=1

# 设置PyTorch相关环境变量
export TORCHDYNAMO_DISABLE=1
export TORCH_SHOW_CPP_STACKTRACES=1

echo "📋 最终环境配置:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  CUDA_HOME: $CUDA_HOME"

# 验证PyTorch
echo ""
echo "🔍 验证PyTorch环境..."
python3.10 -c "
import sys
print('Python路径:', sys.executable)
print('PYTHONPATH前3个:', sys.path[:3])

try:
    import torch
    print('✅ PyTorch导入成功')
    print('PyTorch版本:', torch.__version__)
    print('PyTorch路径:', torch.__file__)
    print('CUDA可用:', torch.cuda.is_available())
    
    # 测试基本操作
    x = torch.randn(2, 2)
    y = torch.mm(x, x)
    print('✅ 张量运算成功')
    
except Exception as e:
    print('❌ PyTorch测试失败:', e)
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 .ext_pkgs PyTorch环境设置成功！"
    echo "💡 现在可以运行需要PyTorch的脚本了"
else
    echo "❌ 环境设置失败"
    exit 1
fi
