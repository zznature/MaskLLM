#!/bin/bash

# Purpose: Safely ensure HuggingFace Transformers is installed using CN mirrors
# Usage: chmod +x fix_transformers_safe_commands.sh && ./fix_transformers_safe_commands.sh

set -euo pipefail

echo "=== Transformers 安全修复脚本（使用镜像） ==="

MIRROR_PYPI_URL=${MIRROR_PYPI_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
export PIP_INDEX_URL="${MIRROR_PYPI_URL}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "0. 镜像设置："
echo "  - PIP_INDEX_URL=${PIP_INDEX_URL}"
echo "  - HF_ENDPOINT=${HF_ENDPOINT}"

echo "1. 检查磁盘空间："
df -h /usr/local ~

echo "2. 检查pip状态："
which pip3 || true
which pip || true

echo "3. 检查pip版本（使用python -m pip）："
python3 -m pip --version || python -m pip --version || true

echo "4. 清理pip缓存："
python3 -m pip cache purge || true

echo "5. 检查transformers是否已安装："
python3 -c "import transformers; print('transformers已安装，版本:', transformers.__version__)" 2>/dev/null || echo "transformers未安装"

echo "6. 如果未安装，则使用清华镜像安装到用户目录："
if ! python3 -c "import transformers" 2>/dev/null; then
  echo "正在安装 transformers ..."
  python3 -m pip install -i "${MIRROR_PYPI_URL}" --no-cache-dir --default-timeout 120 --user transformers
else
  echo "transformers已存在，跳过安装"
fi

echo "7. 验证安装："
python3 -c "import transformers; print('transformers版本:', transformers.__version__)" 2>/dev/null || echo "transformers安装失败"

echo "8. 测试基本功能："
python3 -c "from transformers import AutoTokenizer; print('AutoTokenizer导入成功')" 2>/dev/null || echo "AutoTokenizer导入失败"

echo "=== 安全修复完成 ==="


