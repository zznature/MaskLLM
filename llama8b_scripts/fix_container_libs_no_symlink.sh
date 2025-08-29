#!/bin/bash

# 容器内库依赖修复脚本 - 无符号链接方案
# 通过LD_LIBRARY_PATH直接指向库文件，避免权限问题

echo "🔧 开始修复容器内库依赖问题 (无符号链接方案)"
echo "时间: $(date)"
echo "策略: 通过LD_LIBRARY_PATH直接访问库文件"
echo "=" * 60

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# 1. 确认关键库文件位置
echo "📋 步骤1: 确认关键库文件位置"
CONDA_LIBS_DIR="/opt/conda_libs"

# 检查cuDNN
CUDNN_LIB="$CONDA_LIBS_DIR/libcudnn.so.8.9.7"
if [ -f "$CUDNN_LIB" ]; then
    echo "✅ cuDNN: $CUDNN_LIB"
else
    echo "❌ cuDNN未找到: $CUDNN_LIB"
    exit 1
fi

# 检查cuPTI
CUPTI_DIRS=(
    "$CONDA_LIBS_DIR/python3.10/site-packages/nvidia/cuda_cupti/lib"
    "$CONDA_LIBS_DIR/python3.10/site-packages/triton/backends/nvidia/lib/cupti"
)

CUPTI_DIR=""
for dir in "${CUPTI_DIRS[@]}"; do
    if [ -f "$dir/libcupti.so.12" ]; then
        CUPTI_DIR="$dir"
        echo "✅ cuPTI: $dir/libcupti.so.12"
        break
    fi
done

if [ -z "$CUPTI_DIR" ]; then
    echo "❌ cuPTI未找到"
    exit 1
fi

# 检查NCCL
NCCL_DIR="$CONDA_LIBS_DIR/python3.10/site-packages/nvidia/nccl/lib"
if [ -f "$NCCL_DIR/libnccl.so.2" ]; then
    echo "✅ NCCL: $NCCL_DIR/libnccl.so.2"
else
    echo "❌ NCCL未找到: $NCCL_DIR/libnccl.so.2"
    exit 1
fi
echo ""

# 2. 构建完整的LD_LIBRARY_PATH
echo "📋 步骤2: 构建优化的LD_LIBRARY_PATH"

# 清除可能的HPC-X路径
clean_ld_library_path() {
    local path="$1"
    echo "$path" | tr ':' '\n' | grep -v -E "(hpcx|ucx|ucc)" | tr '\n' ':' | sed 's/:$//'
}

# 获取清理后的原始路径
CLEAN_ORIGINAL_PATH=""
if [ -n "$LD_LIBRARY_PATH" ]; then
    CLEAN_ORIGINAL_PATH=$(clean_ld_library_path "$LD_LIBRARY_PATH")
fi

# 构建新的LD_LIBRARY_PATH，优先级从高到低
NEW_LD_LIBRARY_PATH=""

# 1. 最高优先级：我们需要的特定库目录
PRIORITY_PATHS=(
    "$CONDA_LIBS_DIR"                    # cuDNN主目录
    "$CUPTI_DIR"                         # cuPTI目录
    "$NCCL_DIR"                          # NCCL目录
    "/usr/local/cuda/lib64"              # CUDA标准库
    "/usr/local/lib/python3.10/dist-packages/torch/lib"  # PyTorch库
)

for path in "${PRIORITY_PATHS[@]}"; do
    if [ -d "$path" ]; then
        if [ -z "$NEW_LD_LIBRARY_PATH" ]; then
            NEW_LD_LIBRARY_PATH="$path"
        else
            NEW_LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH:$path"
        fi
        echo "  ✅ 优先路径: $path"
    fi
done

# 2. 添加清理后的原始路径
if [ -n "$CLEAN_ORIGINAL_PATH" ]; then
    NEW_LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH:$CLEAN_ORIGINAL_PATH"
fi

export LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH"
echo "✅ LD_LIBRARY_PATH已重建"
echo ""

# 3. 设置其他必要环境变量
echo "📋 步骤3: 设置环境变量"
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"

# 强制禁用HPC-X组件
export OMPI_MCA_pml=^ucx,^hcoll
export OMPI_MCA_btl=^openib,^uct
export UCX_TLS=^ud,^dc,^rc
export UCC_TLS=^ucp,^sharp
export TORCH_DISTRIBUTED_BACKEND=gloo

echo "✅ 环境变量设置完成"
echo ""

# 4. 验证库文件可访问性
echo "📋 步骤4: 验证库文件可访问性"

test_library() {
    local lib_name="$1"
    echo "🔍 测试 $lib_name:"
    
    # 使用ldconfig -p 模拟库搜索
    local found=false
    IFS=':' read -ra ADDR <<< "$LD_LIBRARY_PATH"
    for path in "${ADDR[@]}"; do
        if [ -f "$path/$lib_name" ]; then
            echo "  ✅ 找到: $path/$lib_name"
            ls -la "$path/$lib_name"
            found=true
            break
        fi
    done
    
    if [ "$found" = false ]; then
        echo "  ❌ 未找到: $lib_name"
        return 1
    fi
    return 0
}

# 测试关键库
CRITICAL_LIBS=(
    "libcudnn.so.8.9.7"
    "libcupti.so.12"
    "libnccl.so.2"
)

ALL_LIBS_FOUND=true
for lib in "${CRITICAL_LIBS[@]}"; do
    if ! test_library "$lib"; then
        ALL_LIBS_FOUND=false
    fi
done
echo ""

# 5. 测试PyTorch导入
echo "📋 步骤5: 测试PyTorch导入"

if [ "$ALL_LIBS_FOUND" = true ]; then
    python3.10 -c "
import os
print('🔍 LD_LIBRARY_PATH验证:')
paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
for i, path in enumerate(paths[:8]):
    if path:
        print(f'  {i+1}. {path}')
print('')

print('🔍 关键库文件检查:')
import ctypes.util
key_libs = ['cudnn', 'cupti', 'nccl']
for lib in key_libs:
    lib_path = ctypes.util.find_library(lib)
    if lib_path:
        print(f'  ✅ {lib}: {lib_path}')
    else:
        print(f'  ❌ {lib}: 未找到')
print('')

try:
    print('🚀 开始PyTorch导入测试...')
    import torch
    print('✅ PyTorch导入成功!')
    print('版本:', torch.__version__)
    print('文件路径:', torch.__file__)
    
    print('🔍 CUDA测试:')
    cuda_available = torch.cuda.is_available()
    print('CUDA可用:', cuda_available)
    
    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            print('CUDA设备数:', device_count)
            if device_count > 0:
                print('当前设备:', torch.cuda.current_device())
                print('设备名称:', torch.cuda.get_device_name())
        except Exception as e:
            print('设备信息获取异常:', str(e))
    
    print('🔍 基本功能测试:')
    x = torch.randn(3, 3)
    y = torch.mm(x, x)
    print('CPU张量运算: ✅')
    
    # 测试分布式
    try:
        import torch.distributed as dist
        print('分布式模块: ✅')
    except Exception as e:
        print('分布式模块异常:', str(e))
    
    print('')
    print('🎉 PyTorch完全正常!')
    
except Exception as e:
    print('❌ PyTorch导入失败:', str(e))
    print('')
    print('🔍 详细错误信息:')
    import traceback
    traceback.print_exc()
    exit(1)
"
else
    echo "❌ 关键库文件验证失败，跳过PyTorch测试"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 容器内库依赖修复完全成功!"
    echo ""
    echo "🔧 环境已优化:"
    echo "- ✅ cuDNN库可访问"
    echo "- ✅ cuPTI库可访问" 
    echo "- ✅ NCCL库可访问"
    echo "- ✅ PyTorch完全正常"
    echo ""
    echo "💡 现在可以测试相关组件:"
    echo ""
    echo "1️⃣ transformer_engine测试:"
    echo "   python3.10 -c 'import transformer_engine as te; print(\"transformer_engine:\", te.__version__)'"
    echo ""
    echo "2️⃣ megatron.tokenizer测试:"
    echo "   python3.10 -c 'from megatron.tokenizer import build_tokenizer; print(\"megatron.tokenizer: 可用\")'"
    echo ""
    echo "3️⃣ C4数据转换测试:"
    echo "   bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh 0 0"
    echo ""
    echo "🚀 环境完全就绪，可以开始所有MaskLLM任务!"
else
    echo ""
    echo "❌ 修复失败，请检查错误信息"
    exit 1
fi
