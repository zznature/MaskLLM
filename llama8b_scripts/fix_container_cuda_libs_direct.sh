#!/bin/bash

# 容器内CUDA库直接修复脚本 
# 基于实际检测结果，不依赖conda环境

echo "🔧 开始修复容器内CUDA库依赖问题 (直接方式)"
echo "时间: $(date)"
echo "基于检测结果: 库文件位于 /opt/conda_libs/"
echo "=" * 60

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# 1. 设置库文件源路径（基于检测结果）
echo "📋 步骤1: 设置库文件源路径"
CONDA_LIBS_DIR="/opt/conda_libs"
CUDA_LIB64="/usr/local/cuda/lib64"

if [ ! -d "$CONDA_LIBS_DIR" ]; then
    echo "❌ 错误: $CONDA_LIBS_DIR 目录不存在"
    exit 1
fi

echo "✅ 库文件源目录: $CONDA_LIBS_DIR"
echo "✅ 目标目录: $CUDA_LIB64"

# 确保目标目录存在
mkdir -p "$CUDA_LIB64" 2>/dev/null
echo ""

# 2. 创建cuDNN符号链接
echo "📋 步骤2: 创建cuDNN符号链接"

# cuDNN主库
CUDNN_SOURCE="$CONDA_LIBS_DIR/libcudnn.so.8.9.7"
CUDNN_TARGET="$CUDA_LIB64/libcudnn.so.8"

if [ -f "$CUDNN_SOURCE" ]; then
    if [ ! -f "$CUDNN_TARGET" ] && [ ! -L "$CUDNN_TARGET" ]; then
        ln -sf "$CUDNN_SOURCE" "$CUDNN_TARGET"
        echo "✅ 创建 libcudnn.so.8 -> $CUDNN_SOURCE"
    else
        echo "✅ libcudnn.so.8 已存在"
    fi
else
    echo "❌ 未找到cuDNN源文件: $CUDNN_SOURCE"
fi

# 其他cuDNN库 (基于检测结果)
CUDNN_LIBS=(
    "libcudnn_cnn_infer.so.8.9.7:libcudnn_cnn_infer.so.8"
    "libcudnn_cnn_train.so.8.9.7:libcudnn_cnn_train.so.8"
    "libcudnn_ops_infer.so.8.9.7:libcudnn_ops_infer.so.8"
    "libcudnn_ops_train.so.8.9.7:libcudnn_ops_train.so.8"
    "libcudnn_adv_train.so.8.9.7:libcudnn_adv_train.so.8"
)

for lib_pair in "${CUDNN_LIBS[@]}"; do
    IFS=':' read -r source_name target_name <<< "$lib_pair"
    source_file="$CONDA_LIBS_DIR/$source_name"
    target_file="$CUDA_LIB64/$target_name"
    
    if [ -f "$source_file" ]; then
        if [ ! -f "$target_file" ] && [ ! -L "$target_file" ]; then
            ln -sf "$source_file" "$target_file"
            echo "✅ 创建 $target_name -> $source_file"
        else
            echo "✅ $target_name 已存在"
        fi
    else
        echo "⚠️  未找到 $source_file"
    fi
done
echo ""

# 3. 创建cuPTI符号链接
echo "📋 步骤3: 创建cuPTI符号链接"

# 基于检测结果的cuPTI位置
CUPTI_LOCATIONS=(
    "$CONDA_LIBS_DIR/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"
    "$CONDA_LIBS_DIR/python3.10/site-packages/triton/backends/nvidia/lib/cupti/libcupti.so.12"
)

CUPTI_SOURCE=""
for location in "${CUPTI_LOCATIONS[@]}"; do
    if [ -f "$location" ]; then
        CUPTI_SOURCE="$location"
        echo "✅ 发现cuPTI: $location"
        break
    fi
done

if [ -n "$CUPTI_SOURCE" ]; then
    CUPTI_TARGET="$CUDA_LIB64/libcupti.so.12"
    if [ ! -f "$CUPTI_TARGET" ] && [ ! -L "$CUPTI_TARGET" ]; then
        ln -sf "$CUPTI_SOURCE" "$CUPTI_TARGET"
        echo "✅ 创建 libcupti.so.12 -> $CUPTI_SOURCE"
    else
        echo "✅ libcupti.so.12 已存在"
    fi
else
    echo "⚠️  未找到cuPTI源文件"
fi
echo ""

# 4. 设置NCCL库路径
echo "📋 步骤4: 设置NCCL库路径"

# 搜索NCCL库的实际位置
NCCL_SEARCH_DIRS=(
    "$CONDA_LIBS_DIR/python3.10/site-packages/nvidia/nccl/lib"
    "$CONDA_LIBS_DIR/lib"
    "/usr/local/lib/python3.10/dist-packages/torch/lib"
)

NCCL_SOURCE=""
for dir in "${NCCL_SEARCH_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -f "$dir/libnccl.so.2" ]; then
        NCCL_SOURCE="$dir"
        echo "✅ 发现NCCL库: $dir"
        break
    fi
done

if [ -n "$NCCL_SOURCE" ]; then
    if [ ! -d "/opt/nccl_libs" ] && [ ! -L "/opt/nccl_libs" ]; then
        ln -sf "$NCCL_SOURCE" "/opt/nccl_libs"
        echo "✅ 创建NCCL库链接: /opt/nccl_libs -> $NCCL_SOURCE"
    else
        echo "✅ /opt/nccl_libs 已存在"
    fi
else
    echo "⚠️  未找到NCCL库，尝试使用PyTorch内置NCCL"
fi
echo ""

# 5. 设置环境变量
echo "📋 步骤5: 设置环境变量"
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"

# 构建LD_LIBRARY_PATH，优先我们的库
NEW_LD_LIBRARY_PATH="/usr/local/cuda/lib64"

# 添加conda_libs目录
if [ -d "$CONDA_LIBS_DIR" ]; then
    NEW_LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH:$CONDA_LIBS_DIR"
fi

# 添加NCCL路径
if [ -d "/opt/nccl_libs" ]; then
    NEW_LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH:/opt/nccl_libs"
fi

# 添加PyTorch库路径
PYTORCH_LIB="/usr/local/lib/python3.10/dist-packages/torch/lib"
if [ -d "$PYTORCH_LIB" ]; then
    NEW_LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH:$PYTORCH_LIB"
fi

# 保持原有的其他路径
if [ -n "$LD_LIBRARY_PATH" ]; then
    NEW_LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
fi

export LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH"

echo "✅ CUDA_HOME: $CUDA_HOME"
echo "✅ CUDA_ROOT: $CUDA_ROOT"
echo "✅ LD_LIBRARY_PATH已设置"
echo ""

# 6. 验证关键库文件
echo "📋 步骤6: 验证关键库文件"
CRITICAL_LIBS=(
    "/usr/local/cuda/lib64/libcudnn.so.8"
    "/usr/local/cuda/lib64/libcupti.so.12"
)

echo "🔍 检查符号链接:"
for lib in "${CRITICAL_LIBS[@]}"; do
    if [ -L "$lib" ]; then
        target=$(readlink "$lib")
        echo "  ✅ $lib -> $target"
        if [ -f "$target" ]; then
            echo "     目标文件存在"
        else
            echo "     ❌ 目标文件不存在"
        fi
    elif [ -f "$lib" ]; then
        echo "  ✅ $lib (常规文件)"
    else
        echo "  ❌ $lib 不存在"
    fi
done
echo ""

# 7. 测试PyTorch导入
echo "📋 步骤7: 测试PyTorch导入"
python3.10 -c "
import os
print('LD_LIBRARY_PATH前5个路径:')
for i, path in enumerate(os.environ.get('LD_LIBRARY_PATH', '').split(':')[:5]):
    print(f'  {i+1}. {path}')
print('')

try:
    import torch
    print('✅ PyTorch导入成功!')
    print('版本:', torch.__version__)
    print('PyTorch路径:', torch.__file__)
    print('CUDA可用:', torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print('CUDA设备数:', torch.cuda.device_count())
        try:
            print('当前设备:', torch.cuda.current_device())
            print('设备名称:', torch.cuda.get_device_name())
        except:
            print('设备信息获取失败（正常，在某些环境中）')
    
    # 测试基本运算
    x = torch.randn(2, 2)
    y = torch.mm(x, x)
    print('✅ 基本张量运算成功')
    
    # 测试分布式
    try:
        import torch.distributed as dist
        print('✅ torch.distributed导入成功')
    except Exception as e:
        print('⚠️ torch.distributed导入问题:', str(e))
    
except Exception as e:
    print('❌ PyTorch导入失败:', str(e))
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 容器内CUDA库修复成功!"
    echo ""
    echo "💡 现在可以测试相关任务:"
    echo "1. transformer_engine测试:"
    echo "   python3.10 -c 'import transformer_engine; print(\"transformer_engine可用\")'"
    echo ""
    echo "2. megatron.tokenizer测试:"
    echo "   python3.10 -c 'from megatron.tokenizer import build_tokenizer; print(\"megatron.tokenizer可用\")'"
    echo ""
    echo "3. C4数据转换测试:"
    echo "   bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh 0 0"
else
    echo ""
    echo "❌ 修复失败，请检查错误信息"
    exit 1
fi
