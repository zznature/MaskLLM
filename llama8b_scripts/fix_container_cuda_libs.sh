#!/bin/bash

# 容器内CUDA库修复脚本 
# 基于Project_Tasks.md中的已知问题解决方案

echo "🔧 开始修复容器内CUDA库依赖问题"
echo "时间: $(date)"
echo "参考: Project_Tasks.md 中的已知解决方案"
echo "=" * 60

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# 1. 检查CONDA_PREFIX环境变量
echo "📋 步骤1: 检查CONDA环境"
if [ -z "$CONDA_PREFIX" ]; then
    echo "⚠️  CONDA_PREFIX未设置，尝试自动检测..."
    # 在NGC容器中，conda通常在这些位置
    POSSIBLE_CONDA_PATHS=(
        "/opt/conda"
        "/usr/local/conda"
        "/opt/miniconda3"
    )
    
    for path in "${POSSIBLE_CONDA_PATHS[@]}"; do
        if [ -d "$path" ]; then
            export CONDA_PREFIX="$path"
            echo "✅ 自动设置CONDA_PREFIX: $CONDA_PREFIX"
            break
        fi
    done
    
    if [ -z "$CONDA_PREFIX" ]; then
        echo "❌ 无法找到conda环境，请手动设置CONDA_PREFIX"
        exit 1
    fi
else
    echo "✅ CONDA_PREFIX已设置: $CONDA_PREFIX"
fi
echo ""

# 2. 检查关键库文件位置
echo "📋 步骤2: 检查关键库文件"

# cuDNN库检查
echo "🔍 检查cuDNN库:"
CUDNN_LOCATIONS=(
    "$CONDA_PREFIX/lib/libcudnn.so.8.9.7"
    "/usr/local/cuda/lib64/libcudnn.so.8.9.7" 
    "/opt/conda/lib/libcudnn.so.8.9.7"
)

CUDNN_SOURCE=""
for location in "${CUDNN_LOCATIONS[@]}"; do
    if [ -f "$location" ]; then
        CUDNN_SOURCE="$location"
        echo "  ✅ 发现cuDNN源文件: $location"
        break
    fi
done

if [ -z "$CUDNN_SOURCE" ]; then
    echo "  ❌ 未找到cuDNN源文件"
    echo "  尝试查找所有libcudnn文件:"
    find /opt /usr/local /conda -name "*libcudnn*" 2>/dev/null | head -10
    exit 1
fi

# cuPTI库检查  
echo "🔍 检查cuPTI库:"
CUPTI_LOCATIONS=(
    "/usr/local/cuda-12.3/NsightSystems-cli-2023.4.1/target-linux-x64/libcupti.so.12.4"
    "/opt/nsight-systems-libs/libcupti.so.12.4"
    "/opt/nsight-compute-libs/libcupti.so.12.4"
)

CUPTI_SOURCE=""
for location in "${CUPTI_LOCATIONS[@]}"; do
    if [ -f "$location" ]; then
        CUPTI_SOURCE="$location"
        echo "  ✅ 发现cuPTI源文件: $location"
        break
    fi
done

if [ -z "$CUPTI_SOURCE" ]; then
    echo "  ⚠️  未找到cuPTI源文件，搜索所有libcupti文件:"
    find /opt /usr/local -name "*libcupti*" 2>/dev/null | head -5
fi

# NCCL库检查
echo "🔍 检查NCCL库:"
NCCL_LOCATIONS=(
    "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib"
    "/opt/nccl_libs"
)

NCCL_SOURCE=""
for location in "${NCCL_LOCATIONS[@]}"; do
    if [ -d "$location" ] && [ -f "$location/libnccl.so.2" ]; then
        NCCL_SOURCE="$location"
        echo "  ✅ 发现NCCL库目录: $location"
        break
    fi
done

if [ -z "$NCCL_SOURCE" ]; then
    echo "  ⚠️  未找到NCCL库，搜索libnccl文件:"
    find /opt /usr/local /conda -name "*libnccl*" 2>/dev/null | head -5
fi
echo ""

# 3. 创建符号链接 - cuDNN
echo "📋 步骤3: 创建cuDNN符号链接"
CUDA_LIB64="/usr/local/cuda/lib64"

# 确保目录存在
if [ ! -d "$CUDA_LIB64" ]; then
    echo "⚠️  $CUDA_LIB64 不存在，创建目录"
    mkdir -p "$CUDA_LIB64" || {
        echo "❌ 无法创建目录，权限不足"
        exit 1
    }
fi

# cuDNN主库
if [ ! -f "$CUDA_LIB64/libcudnn.so.8" ]; then
    if [ -n "$CUDNN_SOURCE" ]; then
        ln -sf "$CUDNN_SOURCE" "$CUDA_LIB64/libcudnn.so.8"
        echo "✅ 创建 libcudnn.so.8 -> $CUDNN_SOURCE"
    else
        echo "❌ 无法创建cuDNN链接，源文件不存在"
    fi
else
    echo "✅ libcudnn.so.8 已存在"
fi

# cuDNN其他库
CUDNN_LIBS=(
    "libcudnn_cnn_infer.so.8"
    "libcudnn_cnn_train.so.8" 
    "libcudnn_ops_infer.so.8"
    "libcudnn_ops_train.so.8"
    "libcudnn_adv_infer.so.8"
    "libcudnn_adv_train.so.8"
)

CUDNN_BASE_DIR=$(dirname "$CUDNN_SOURCE")
for lib in "${CUDNN_LIBS[@]}"; do
    source_lib="$CUDNN_BASE_DIR/${lib}.9.7"
    target_lib="$CUDA_LIB64/$lib"
    
    if [ ! -f "$target_lib" ]; then
        if [ -f "$source_lib" ]; then
            ln -sf "$source_lib" "$target_lib"
            echo "✅ 创建 $lib -> $source_lib"
        else
            echo "⚠️  未找到 $source_lib"
        fi
    else
        echo "✅ $lib 已存在"
    fi
done
echo ""

# 4. 创建符号链接 - cuPTI
echo "📋 步骤4: 创建cuPTI符号链接" 
if [ -n "$CUPTI_SOURCE" ]; then
    if [ ! -f "$CUDA_LIB64/libcupti.so.12" ]; then
        ln -sf "$CUPTI_SOURCE" "$CUDA_LIB64/libcupti.so.12"
        echo "✅ 创建 libcupti.so.12 -> $CUPTI_SOURCE"
    else
        echo "✅ libcupti.so.12 已存在"
    fi
else
    echo "⚠️  跳过cuPTI链接创建"
fi
echo ""

# 5. 设置NCCL库路径
echo "📋 步骤5: 设置NCCL库路径"
if [ -n "$NCCL_SOURCE" ]; then
    if [ ! -d "/opt/nccl_libs" ]; then
        ln -sf "$NCCL_SOURCE" "/opt/nccl_libs"
        echo "✅ 创建NCCL库链接: /opt/nccl_libs -> $NCCL_SOURCE"
    else
        echo "✅ /opt/nccl_libs 已存在"
    fi
else
    echo "⚠️  跳过NCCL库设置"
fi
echo ""

# 6. 设置环境变量
echo "📋 步骤6: 设置环境变量"
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda" 
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/opt/nccl_libs:$LD_LIBRARY_PATH"

echo "✅ CUDA_HOME: $CUDA_HOME"
echo "✅ CUDA_ROOT: $CUDA_ROOT"
echo "✅ LD_LIBRARY_PATH已更新"
echo ""

# 7. 验证修复结果
echo "📋 步骤7: 验证修复结果"
echo "🔍 检查关键库文件:"
CRITICAL_LIBS=(
    "/usr/local/cuda/lib64/libcudnn.so.8"
    "/usr/local/cuda/lib64/libcupti.so.12"
    "/opt/nccl_libs/libnccl.so.2"
)

for lib in "${CRITICAL_LIBS[@]}"; do
    if [ -f "$lib" ] || [ -L "$lib" ]; then
        echo "  ✅ $lib"
        ls -la "$lib"
    else
        echo "  ❌ $lib 缺失"
    fi
done
echo ""

# 8. 测试PyTorch导入
echo "📋 步骤8: 测试PyTorch导入"
python3.10 -c "
try:
    import torch
    print('✅ PyTorch导入成功!')
    print('版本:', torch.__version__)
    print('CUDA可用:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA设备数:', torch.cuda.device_count())
        print('当前设备:', torch.cuda.current_device())
except Exception as e:
    print('❌ PyTorch导入失败:', str(e))
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 容器内CUDA库修复成功!"
    echo "💡 现在可以运行C4数据转换任务:"
    echo "bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh 0 0"
else
    echo "❌ 修复失败，请检查错误信息"
    exit 1
fi
