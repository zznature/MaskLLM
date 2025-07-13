#!/bin/bash

# 简化的 MaskLLM 运行脚本
# 使用 NVIDIA PyTorch NGC 容器内的原生环境

echo "=== 运行 MaskLLM (简化版) ==="

# 检查是否在容器内
if [ ! -f /.singularity.d/startscript ]; then
    echo "警告: 此脚本应在容器内运行"
    echo "请先启动容器: bash run_container_simple.sh"
    exit 1
fi

echo "检查容器内原生环境..."

# 检查 Python 环境
echo "Python 版本:"
python --version

# 检查 PyTorch
echo ""
echo "检查 PyTorch:"
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available())" || {
    echo "PyTorch 不可用"
    exit 1
}

# 检查 transformer_engine
echo ""
echo "检查 transformer_engine:"
python -c "import transformer_engine; print('transformer_engine 可用')" 2>/dev/null || {
    echo "transformer_engine 不可用，尝试安装..."
    pip install transformer-engine || {
        echo "安装 transformer_engine 失败"
        exit 1
    }
}

echo ""
echo "=== 开始运行 MaskLLM 任务 ==="

# 运行原始的 MaskLLM 脚本
if [ $# -eq 0 ]; then
    echo "用法: $0 <脚本路径> [参数...]"
    echo "示例: $0 scripts/oneshot/run_llama2_7b_prune_tp4.sh SparseGPT"
    exit 1
fi

SCRIPT_PATH="$1"
shift

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 脚本文件 $SCRIPT_PATH 不存在"
    exit 1
fi

echo "运行脚本: $SCRIPT_PATH"
echo "参数: $@"

# 直接运行脚本，使用容器内的原生 Python
bash "$SCRIPT_PATH" "$@"

echo ""
echo "=== MaskLLM 任务完成 ===" 