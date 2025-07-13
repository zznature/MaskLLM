#!/bin/bash

# CUDA 库符号链接设置脚本
# 在容器内运行此脚本来创建必要的 cuDNN 和 cuPTI 符号链接

echo "=== 设置 CUDA 库符号链接 ==="

# 检查库路径
CONDA_LIBS="/opt/conda_libs"
CUDA_LIB64="/usr/local/cuda/lib64"
NSIGHT_SYSTEMS_LIBS="/opt/nsight-systems-libs"
NSIGHT_COMPUTE_LIBS="/opt/nsight-compute-libs"

echo "检查库路径:"
echo "  conda 库路径: $CONDA_LIBS"
echo "  CUDA 库路径: $CUDA_LIB64"
echo "  nsight-systems 库路径: $NSIGHT_SYSTEMS_LIBS"
echo "  nsight-compute 库路径: $NSIGHT_COMPUTE_LIBS"

echo ""
echo "=== 设置 cuDNN 符号链接 ==="

# 创建 cuDNN 主库符号链接
if [ ! -f "$CUDA_LIB64/libcudnn.so.8" ]; then
    if [ -f "$CONDA_LIBS/libcudnn.so.8.9.7" ]; then
        ln -sf "$CONDA_LIBS/libcudnn.so.8.9.7" "$CUDA_LIB64/libcudnn.so.8"
        echo "✓ 创建 libcudnn.so.8 -> libcudnn.so.8.9.7"
    else
        echo "✗ 未找到 $CONDA_LIBS/libcudnn.so.8.9.7"
    fi
else
    echo "✓ libcudnn.so.8 已存在"
fi

# 创建其他 cuDNN 相关符号链接
cudnn_libs=(
    "libcudnn_cnn_infer.so.8"
    "libcudnn_cnn_train.so.8"
    "libcudnn_ops_infer.so.8"
    "libcudnn_ops_train.so.8"
    "libcudnn_adv_infer.so.8"
    "libcudnn_adv_train.so.8"
)

for lib in "${cudnn_libs[@]}"; do
    base_lib=$(echo "$lib" | sed 's/\.so\.8$/.so.8.9.7/')
    if [ ! -f "$CUDA_LIB64/$lib" ]; then
        if [ -f "$CONDA_LIBS/$base_lib" ]; then
            ln -sf "$CONDA_LIBS/$base_lib" "$CUDA_LIB64/$lib"
            echo "✓ 创建 $lib -> $base_lib"
        else
            echo "✗ 未找到 $CONDA_LIBS/$base_lib"
        fi
    else
        echo "✓ $lib 已存在"
    fi
done

echo ""
echo "=== 设置 cuPTI 符号链接 ==="

# 创建 cuPTI 主库符号链接
if [ ! -f "$CUDA_LIB64/libcupti.so.12" ]; then
    # 优先使用 nsight-systems 中的库
    if [ -f "$NSIGHT_SYSTEMS_LIBS/libcupti.so.12.4" ]; then
        ln -sf "$NSIGHT_SYSTEMS_LIBS/libcupti.so.12.4" "$CUDA_LIB64/libcupti.so.12"
        echo "✓ 创建 libcupti.so.12 -> libcupti.so.12.4 (nsight-systems)"
    elif [ -f "$NSIGHT_COMPUTE_LIBS/libcupti.so.12.4" ]; then
        ln -sf "$NSIGHT_COMPUTE_LIBS/libcupti.so.12.4" "$CUDA_LIB64/libcupti.so.12"
        echo "✓ 创建 libcupti.so.12 -> libcupti.so.12.4 (nsight-compute)"
    else
        echo "✗ 未找到 libcupti.so.12.4"
    fi
else
    echo "✓ libcupti.so.12 已存在"
fi

echo ""
echo "=== 验证库文件 ==="
echo "检查 libcudnn.so.8:"
if [ -f "$CUDA_LIB64/libcudnn.so.8" ]; then
    ls -la "$CUDA_LIB64/libcudnn.so.8"
    echo "✓ libcudnn.so.8 可用"
else
    echo "✗ libcudnn.so.8 不可用"
fi

echo ""
echo "检查 libcupti.so.12:"
if [ -f "$CUDA_LIB64/libcupti.so.12" ]; then
    ls -la "$CUDA_LIB64/libcupti.so.12"
    echo "✓ libcupti.so.12 可用"
else
    echo "✗ libcupti.so.12 不可用"
fi

echo ""
echo "CUDA 库符号链接设置完成！" 