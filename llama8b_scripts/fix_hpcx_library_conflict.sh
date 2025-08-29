#!/bin/bash

# HPC-X库冲突修复脚本
# 解决 /opt/hpcx/ucc/lib/libucc.so.1 符号冲突问题

echo "🔧 开始修复HPC-X库冲突问题..."

# 1. 备份当前LD_LIBRARY_PATH
echo "📋 当前LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH"
echo ""

# 2. 识别冲突库路径
CONFLICT_PATHS=(
    "/opt/hpcx/ucc/lib"
    "/opt/hpcx/ucx/lib"
    "/opt/hpcx/ompi/lib"
    "/opt/hpcx/hcoll/lib"
)

echo "🔍 检查冲突库路径:"
for path in "${CONFLICT_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "  ❌ 发现冲突路径: $path"
        if [ -f "$path/libucc.so.1" ]; then
            echo "    - 包含冲突库: libucc.so.1"
        fi
    else
        echo "  ✅ 路径不存在: $path"
    fi
done
echo ""

# 3. 清理LD_LIBRARY_PATH中的HPC-X路径
echo "🧹 清理LD_LIBRARY_PATH中的HPC-X路径..."
CLEAN_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

for path in "${CONFLICT_PATHS[@]}"; do
    CLEAN_LD_LIBRARY_PATH=$(echo "$CLEAN_LD_LIBRARY_PATH" | sed "s|:*$path:*||g" | sed 's|^:||g' | sed 's|:$||g' | sed 's|::*|:|g')
done

export LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH"
echo "✅ 清理后的LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH"
echo ""

# 4. 设置替代的通信库路径
echo "🔄 设置替代通信库路径..."

# 优先使用容器内的NCCL库
NCCL_LIB_PATH="/opt/nccl_libs"
if [ -d "$NCCL_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$NCCL_LIB_PATH:$LD_LIBRARY_PATH"
    echo "✅ 添加NCCL路径: $NCCL_LIB_PATH"
fi

# 确保CUDA库优先
CUDA_LIBS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda/compat/lib"
    "/usr/local/nvidia/lib64"
    "/usr/local/nvidia/lib"
)

for cuda_lib in "${CUDA_LIBS[@]}"; do
    if [ -d "$cuda_lib" ]; then
        export LD_LIBRARY_PATH="$cuda_lib:$LD_LIBRARY_PATH"
        echo "✅ 优先CUDA路径: $cuda_lib"
    fi
done

# 清理重复路径
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//')

echo ""
echo "🎯 最终LD_LIBRARY_PATH:"
echo "$LD_LIBRARY_PATH"
echo ""

# 5. 设置环境变量以禁用有问题的通信库
echo "🚫 设置环境变量禁用冲突组件..."
export OMPI_MCA_pml=^ucx  # 禁用UCX作为OpenMPI的点对点消息传递层
export OMPI_MCA_btl=^openib  # 禁用InfiniBand BTL
export UCX_TLS=^ud  # 禁用UCX的不可靠数据报传输
export UCC_TLS=^ucp  # 禁用UCC的UCP传输层

echo "✅ 禁用配置:"
echo "  OMPI_MCA_pml=$OMPI_MCA_pml"
echo "  OMPI_MCA_btl=$OMPI_MCA_btl"
echo "  UCX_TLS=$UCX_TLS"
echo "  UCC_TLS=$UCC_TLS"
echo ""

# 6. 验证修复效果
echo "🔍 验证修复效果..."

echo "验证Python可用性:"
if python3.10 -c "print('Python OK')" 2>/dev/null; then
    echo "✅ Python3.10 正常"
else
    echo "❌ Python3.10 仍有问题"
fi

echo ""
echo "验证PyTorch导入:"
if python3.10 -c "import torch; print('PyTorch版本:', torch.__version__)" 2>/dev/null; then
    echo "✅ PyTorch 导入成功"
else
    echo "❌ PyTorch 仍无法导入"
    echo "详细错误信息:"
    python3.10 -c "import torch" 2>&1 | head -5
fi

echo ""
echo "验证torchrun:"
if /usr/local/bin/torchrun --version 2>/dev/null; then
    echo "✅ torchrun 正常"
else
    echo "❌ torchrun 仍有问题"
    echo "详细错误信息:"
    /usr/local/bin/torchrun --version 2>&1 | head -3
fi

echo ""
echo "🎉 HPC-X库冲突修复完成!"
echo ""
echo "💡 使用方法:"
echo "  在执行任何需要PyTorch的脚本前，先运行此修复脚本:"
echo "  source llama8b_scripts/fix_hpcx_library_conflict.sh"
echo "  然后再运行目标脚本"
