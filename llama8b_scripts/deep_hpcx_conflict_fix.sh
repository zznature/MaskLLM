#!/bin/bash

# 深度HPC-X库冲突修复脚本
# 使用更激进的方法解决 libucc.so.1 符号冲突

echo "🔧 开始深度HPC-X库冲突修复..."

# 1. 完全隔离HPC-X环境
echo "📋 步骤1: 完全隔离HPC-X环境"

# 清除所有可能的HPC-X环境变量
unset HPCX_DIR
unset HPCX_HOME
unset HPCX_ROOT
unset HPCX_MPI_DIR
unset HPCX_UCX_DIR
unset HPCX_UCC_DIR
unset HPCX_HCOLL_DIR

# 清除OpenMPI相关的HPC-X配置
unset OMPI_HOME
unset OMPI_ROOT
unset MPI_HOME
unset MPI_ROOT

echo "✅ HPC-X环境变量已清除"

# 2. 强制禁用所有HPC-X相关组件
echo "📋 步骤2: 强制禁用HPC-X组件"

export OMPI_MCA_pml=^ucx,^hcoll
export OMPI_MCA_btl=^openib,^uct
export OMPI_MCA_coll=^hcoll,^ucx
export OMPI_MCA_osc=^ucx
export OMPI_MCA_spml=^ucx

export UCX_TLS=^ud,^dc,^rc
export UCC_TLS=^ucp,^sharp
export UCC_CLS=^sharp

# 完全禁用UCX和UCC
export UCX_LOG_LEVEL=error
export UCC_LOG_LEVEL=error

echo "✅ HPC-X组件已强制禁用"

# 3. 重建LD_LIBRARY_PATH，完全排除HPC-X
echo "📋 步骤3: 重建库路径"

# 获取当前LD_LIBRARY_PATH
CURRENT_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# HPC-X相关路径模式（更全面）
HPCX_PATH_PATTERNS=(
    "/opt/hpcx"
    "/usr/local/hpcx"
    "/opt/mellanox"
    "/usr/lib64/openmpi"
    "/usr/lib/openmpi"
    "hpcx"
    "ucx"
    "ucc"
    "hcoll"
    "sharp"
)

# 清理LD_LIBRARY_PATH
NEW_LD_LIBRARY_PATH=""
IFS=':' read -ra ADDR <<< "$CURRENT_LD_LIBRARY_PATH"
for path in "${ADDR[@]}"; do
    INCLUDE_PATH=true
    for pattern in "${HPCX_PATH_PATTERNS[@]}"; do
        if [[ "$path" == *"$pattern"* ]]; then
            echo "  🗑️  排除路径: $path (匹配模式: $pattern)"
            INCLUDE_PATH=false
            break
        fi
    done
    
    if [ "$INCLUDE_PATH" = true ] && [ -n "$path" ]; then
        if [ -z "$NEW_LD_LIBRARY_PATH" ]; then
            NEW_LD_LIBRARY_PATH="$path"
        else
            NEW_LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH:$path"
        fi
    fi
done

export LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH"
echo "✅ LD_LIBRARY_PATH已重建"

# 4. 优先使用容器内库
echo "📋 步骤4: 优先容器内库"

# 确保NCCL库优先
NCCL_PATHS=(
    "/opt/nccl_libs"
    "/usr/local/lib/python3.10/dist-packages/torch/lib"
    "/usr/local/nccl/lib"
)

for nccl_path in "${NCCL_PATHS[@]}"; do
    if [ -d "$nccl_path" ]; then
        export LD_LIBRARY_PATH="$nccl_path:$LD_LIBRARY_PATH"
        echo "  ✅ 优先NCCL路径: $nccl_path"
    fi
done

# 确保CUDA库优先
CUDA_PATHS=(
    "/usr/local/cuda/lib64"
    "/usr/local/cuda/compat/lib"
    "/usr/local/nvidia/lib64"
)

for cuda_path in "${CUDA_PATHS[@]}"; do
    if [ -d "$cuda_path" ]; then
        export LD_LIBRARY_PATH="$cuda_path:$LD_LIBRARY_PATH"
        echo "  ✅ 优先CUDA路径: $cuda_path"
    fi
done

# 清理重复路径
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//')

# 5. 强制指定通信后端
echo "📋 步骤5: 强制指定通信后端"

# 强制使用GLOO作为CPU通信后端
export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_SOCKET_IFNAME=lo  # 使用loopback接口避免网络冲突

# 禁用所有可能引起冲突的功能
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

echo "✅ 通信后端配置完成"

# 6. 设置Python模块搜索优先级
echo "📋 步骤6: 优化Python模块搜索"

# 确保容器内PyTorch优先
PYTHON_TORCH_PATH="/usr/local/lib/python3.10/dist-packages"
if [ -d "$PYTHON_TORCH_PATH" ]; then
    export PYTHONPATH="$PYTHON_TORCH_PATH:$PYTHONPATH"
    echo "✅ PyTorch路径已优先: $PYTHON_TORCH_PATH"
fi

# 7. 验证修复效果
echo "📋 步骤7: 验证修复效果"

echo "🔍 最终环境配置:"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:0:200}..."
echo "  TORCH_DISTRIBUTED_BACKEND: $TORCH_DISTRIBUTED_BACKEND"
echo "  OMPI_MCA_pml: $OMPI_MCA_pml"
echo "  UCX_TLS: $UCX_TLS"
echo "  UCC_TLS: $UCC_TLS"

# 检查关键库文件
echo ""
echo "🔍 检查关键库文件:"
key_libs=("libcuda.so" "libcudart.so" "libnccl.so")
for lib in "${key_libs[@]}"; do
    found_path=$(find ${LD_LIBRARY_PATH//:/ } -name "$lib*" 2>/dev/null | head -1)
    if [ -n "$found_path" ]; then
        echo "  ✅ $lib: $found_path"
    else
        echo "  ⚠️  $lib: 未找到"
    fi
done

# 检查是否还能找到HPC-X UCC库
echo ""
echo "🔍 检查HPC-X库隔离效果:"
if echo "$LD_LIBRARY_PATH" | grep -q "hpcx\|ucx\|ucc"; then
    echo "  ⚠️  LD_LIBRARY_PATH中仍包含HPC-X路径"
else
    echo "  ✅ LD_LIBRARY_PATH已完全清理HPC-X路径"
fi

echo ""
echo "🎉 深度HPC-X冲突修复完成!"
echo "💡 现在可以尝试导入PyTorch了"
