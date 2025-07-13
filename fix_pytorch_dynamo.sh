#!/bin/bash

# 解决 PyTorch Dynamo 编译器的 CUDA 库问题
echo "=== 修复 PyTorch Dynamo 编译器 CUDA 库问题 ==="

# 设置路径
CUDA_COMPAT_LIB="/usr/local/cuda/compat/lib"
CUDA_LIB64="/usr/local/cuda/lib64"

echo "检查 CUDA 兼容性库目录..."
if [ -d "$CUDA_COMPAT_LIB" ]; then
    echo "✓ 找到 CUDA 兼容性库目录: $CUDA_COMPAT_LIB"
    ls -la "$CUDA_COMPAT_LIB"/libcuda*
else
    echo "✗ 未找到 CUDA 兼容性库目录: $CUDA_COMPAT_LIB"
fi

echo ""
echo "检查 CUDA 库目录..."
if [ -d "$CUDA_LIB64" ]; then
    echo "✓ 找到 CUDA 库目录: $CUDA_LIB64"
    ls -la "$CUDA_LIB64"/libcuda*
else
    echo "✗ 未找到 CUDA 库目录: $CUDA_LIB64"
fi

echo ""
echo "=== 创建 libcuda.so 符号链接 ==="

# 创建 libcuda.so 符号链接
if [ ! -f "$CUDA_LIB64/libcuda.so" ]; then
    if [ -f "$CUDA_COMPAT_LIB/libcuda.so.1" ]; then
        ln -sf "$CUDA_COMPAT_LIB/libcuda.so.1" "$CUDA_LIB64/libcuda.so"
        echo "✓ 创建 libcuda.so -> libcuda.so.1"
    elif [ -f "$CUDA_COMPAT_LIB/libcuda.so" ]; then
        ln -sf "$CUDA_COMPAT_LIB/libcuda.so" "$CUDA_LIB64/libcuda.so"
        echo "✓ 创建 libcuda.so -> libcuda.so (compat)"
    else
        echo "✗ 未找到 libcuda.so.1 或 libcuda.so"
        echo "尝试在其他位置查找..."
        
        # 在其他可能的位置查找
        possible_paths=(
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
            "/usr/lib/x86_64-linux-gnu/libcuda.so"
            "/usr/local/lib/libcuda.so.1"
            "/usr/local/lib/libcuda.so"
        )
        
        for path in "${possible_paths[@]}"; do
            if [ -f "$path" ]; then
                echo "✓ 找到 libcuda: $path"
                ln -sf "$path" "$CUDA_LIB64/libcuda.so"
                echo "✓ 创建 libcuda.so -> $path"
                break
            fi
        done
    fi
else
    echo "✓ libcuda.so 已存在"
fi

echo ""
echo "=== 验证符号链接 ==="
if [ -L "$CUDA_LIB64/libcuda.so" ]; then
    echo "✓ libcuda.so 是符号链接"
    ls -la "$CUDA_LIB64/libcuda.so"
else
    echo "✗ libcuda.so 不是符号链接或不存在"
fi

echo ""
echo "=== 设置 PyTorch Dynamo 配置 ==="

# 设置环境变量来禁用 Dynamo 编译器（如果仍有问题）
export TORCHDYNAMO_DISABLE=1
echo "设置 TORCHDYNAMO_DISABLE=1 (禁用 Dynamo 编译器)"

# 或者设置回退到 eager 模式
export TORCHDYNAMO_VERBOSE=1
echo "设置 TORCHDYNAMO_VERBOSE=1 (启用详细日志)"

echo ""
echo "=== 测试 PyTorch CUDA 可用性 ==="
python3.10 -c "
import torch
print('PyTorch 版本:', torch.__version__)
print('CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA 版本:', torch.version.cuda)
    print('GPU 数量:', torch.cuda.device_count())
    print('当前 GPU:', torch.cuda.current_device())
    print('GPU 名称:', torch.cuda.get_device_name(0))
"

echo ""
echo "修复完成！"
echo "如果仍有问题，可以尝试："
echo "1. 设置 TORCHDYNAMO_DISABLE=1 完全禁用 Dynamo 编译器"
echo "2. 设置 torch._dynamo.config.suppress_errors = True 抑制错误并回退到 eager 模式" 