#!/bin/bash

# MaskLLM 运行脚本 - 使用 Python 3.10
# 确保使用容器内的 Python 3.10 而不是 conda Python 3.12

echo "=== MaskLLM 运行脚本 (Python 3.10) ==="

# 检查是否在容器内
if [ ! -f /.singularity.d/startscript ]; then
    echo "警告: 此脚本应在容器内运行"
    echo "请先启动容器: bash run_apptainer.sh"
    exit 1
fi

# 设置 Python 环境使用容器内的 Python 3.10
export PYTHONPATH="/usr/local/lib/python3.10/site-packages:/usr/lib/python3.10/site-packages:$PYTHONPATH"
export PATH="/usr/local/bin:/usr/bin:$PATH"

# 检查 Python 3.10 是否可用
if ! command -v python3.10 &> /dev/null; then
    echo "错误: Python 3.10 不可用"
    exit 1
fi

echo "使用 Python 版本:"
python3.10 --version

echo ""
echo "检查 transformer_engine:"
python3.10 -c "import transformer_engine; print('✓ transformer_engine 可用')" 2>/dev/null || {
    echo "✗ transformer_engine 不可用"
    echo "请检查 Python 3.10 环境中是否正确安装了 transformer_engine"
    exit 1
}

echo ""
echo "检查 PyTorch:"
python3.10 -c "import torch; print('✓ PyTorch 可用, 版本:', torch.__version__)" 2>/dev/null || {
    echo "✗ PyTorch 不可用"
    exit 1
}

echo ""
echo "=== 开始运行 MaskLLM 任务 ==="

# 运行原始的 MaskLLM 脚本，但使用 python3.10
# 这里需要根据您的具体任务来修改
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

# 修改脚本中的 python 调用为 python3.10
# 这里我们创建一个临时脚本
TEMP_SCRIPT=$(mktemp)
sed 's/python /python3.10 /g' "$SCRIPT_PATH" > "$TEMP_SCRIPT"
chmod +x "$TEMP_SCRIPT"

# 运行修改后的脚本
bash "$TEMP_SCRIPT" "$@"

# 清理临时文件
rm -f "$TEMP_SCRIPT"

echo ""
echo "=== MaskLLM 任务完成 ===" 