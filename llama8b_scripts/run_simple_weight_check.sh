#!/bin/bash
# 简化权重检查调用脚本
# 专门诊断Megatron结构问题
# HPC Server Command Execution Protocol compliant

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

echo "🔍 简化权重检查 - Megatron结构问题诊断"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"

# 构建Python命令
PYTHON_SCRIPT="$SCRIPT_DIR/simple_weight_check.py"
PYTHON_CMD=("python3.10" "$PYTHON_SCRIPT")

echo "🔄 开始权重结构诊断..."
echo ""

# 执行检查
if "${PYTHON_CMD[@]}"; then
    echo ""
    echo "✅ 权重检查完成！"
else
    echo ""
    echo "❌ 权重检查遇到问题"
    exit 1
fi
