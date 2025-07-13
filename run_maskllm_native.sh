#!/bin/bash

# MaskLLM 原生环境运行脚本
# 使用容器内的原生环境运行 MaskLLM，混合方案保留必要库路径
# 确保使用容器内的 Python、PyTorch 和 transformer_engine

echo "=== MaskLLM 原生环境运行脚本 (混合库路径) ==="
echo "使用容器内原生环境:"
echo "  - Python: /usr/bin/python3.10"
echo "  - PyTorch: 2.2.0a0+81ea7a4"
echo "  - transformer_engine: 1.2"
echo "  - torchrun: /usr/local/bin/torchrun"
echo "  - 混合库路径: 容器内 + 外部必要库"
echo ""

# 设置 HuggingFace 镜像环境变量
echo "=== 设置 HuggingFace 镜像环境变量 ==="
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"
echo "HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"
echo ""

# 设置 PyTorch Dynamo 编译器禁用选项
echo "=== 设置 PyTorch Dynamo 编译器选项 ==="
export TORCHDYNAMO_DISABLE=1
export TORCHDYNAMO_VERBOSE=1
echo "TORCHDYNAMO_DISABLE=$TORCHDYNAMO_DISABLE"
echo "TORCHDYNAMO_VERBOSE=$TORCHDYNAMO_VERBOSE"
echo ""

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <脚本路径> [剪枝方法] [其他参数...]"
    echo "示例1 (剪枝脚本): $0 scripts/oneshot/run_llama2_7b_prune_tp4.sh SparseGPT"
    echo "示例2 (转换脚本): $0 scripts/tools/convert_llama2_7b_hf_to_megatron_tp4.sh"
    exit 1
fi

SCRIPT_PATH="$1"
shift

# 检查是否是剪枝脚本（包含 "prune" 关键字）
if [[ "$SCRIPT_PATH" == *"prune"* ]]; then
    if [ $# -lt 1 ]; then
        echo "错误: 剪枝脚本需要指定剪枝方法"
        echo "用法: $0 <剪枝脚本路径> <剪枝方法> [其他参数...]"
        echo "示例: $0 scripts/oneshot/run_llama2_7b_prune_tp4.sh SparseGPT"
        exit 1
    fi
    PRUNING_METHOD="$1"
    shift
    echo "运行剪枝脚本: $SCRIPT_PATH"
    echo "剪枝方法: $PRUNING_METHOD"
    echo "其他参数: $@"
else
    echo "运行脚本: $SCRIPT_PATH"
    echo "参数: $@"
fi

echo ""

# 获取 conda 环境路径（用于库文件）
CONDA_PREFIX="/data/home/zdhs0054/jpu/software/miniconda3/envs/maskllm"

# 设置环境变量，使用混合方案
export PATH="/usr/local/bin:/usr/bin:/usr/sbin:/sbin:/usr/local/cuda/bin"
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib:/opt/nsight-systems-libs:/opt/nsight-compute-libs:/opt/nccl_libs"
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"

# 清空 conda 环境变量，避免冲突
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset CONDA_PYTHON_EXE
unset CONDA_EXE
unset CONDA_PROMPT_MODIFIER
unset CONDA_SHLVL
unset CONDA_ENVS_PATH
unset CONDA_PKGS_DIRS

echo "环境变量设置完成:"
echo "  PATH: $PATH"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

# 设置 cuDNN 符号链接
echo "=== 设置 cuDNN 符号链接 ==="
CONDA_LIBS="/opt/conda_libs"
CUDA_LIB64="/usr/local/cuda/lib64"

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

echo "cuDNN 符号链接设置完成！"
echo ""

# 设置 cuPTI 符号链接
echo "=== 设置 cuPTI 符号链接 ==="
NSIGHT_SYSTEMS_LIBS="/opt/nsight-systems-libs"
NSIGHT_COMPUTE_LIBS="/opt/nsight-compute-libs"

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

echo "cuPTI 符号链接设置完成！"
echo ""

# 设置 CUDA 库符号链接
echo "=== 设置 CUDA 库符号链接 ==="
CUDA_COMPAT_LIB="/usr/local/cuda/compat/lib"
CUDA_LIB64="/usr/local/cuda/lib64"

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
    fi
else
    echo "✓ libcuda.so 已存在"
fi

# 创建其他可能需要的 CUDA 库符号链接
cuda_libs=(
    "libcudart.so"
    "libcublas.so"
    "libcublasLt.so"
    "libcurand.so"
    "libcufft.so"
    "libcusparse.so"
    "libcusolver.so"
    "libnvrtc.so"
    "libnvToolsExt.so"
)

for lib in "${cuda_libs[@]}"; do
    base_lib=$(echo "$lib" | sed 's/\.so$/.so.12/')
    if [ ! -f "$CUDA_LIB64/$lib" ]; then
        if [ -f "$CUDA_LIB64/$base_lib" ]; then
            ln -sf "$base_lib" "$CUDA_LIB64/$lib"
            echo "✓ 创建 $lib -> $base_lib"
        else
            echo "✗ 未找到 $CUDA_LIB64/$base_lib"
        fi
    else
        echo "✓ $lib 已存在"
    fi
done

echo "CUDA 库符号链接设置完成！"
echo ""

# 验证环境
echo "验证环境..."
python3.10 --version
which python3.10
python3.10 -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"
python3.10 -c "import transformer_engine; print('transformer_engine 可用')"
/usr/local/bin/torchrun --version

# 验证 CUDA 工具链
echo ""
echo "验证 CUDA 工具链..."
which nvcc
nvcc --version
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_ROOT: $CUDA_ROOT"
echo ""

echo "开始运行脚本..."
echo ""

# 运行脚本
if [[ "$SCRIPT_PATH" == *"prune"* ]]; then
    # 剪枝脚本：传递剪枝方法和其他参数
    bash "$SCRIPT_PATH" "$PRUNING_METHOD" "$@"
else
    # 其他脚本：直接传递所有参数
    bash "$SCRIPT_PATH" "$@"
fi 