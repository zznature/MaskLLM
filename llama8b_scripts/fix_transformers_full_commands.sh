#!/bin/bash

# Purpose: Thoroughly reinstall HuggingFace Transformers using CN mirrors
# Usage: chmod +x fix_transformers_full_commands.sh && ./fix_transformers_full_commands.sh

set -euo pipefail

echo "=== Transformers 彻底修复脚本（使用镜像） ==="

MIRROR_PYPI_URL=${MIRROR_PYPI_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
export PIP_INDEX_URL="${MIRROR_PYPI_URL}"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "0. 镜像设置："
echo "  - PIP_INDEX_URL=${PIP_INDEX_URL}"
echo "  - HF_ENDPOINT=${HF_ENDPOINT}"

echo "1. 强制卸载transformers（如果存在）："
pip3 uninstall transformers -y 2>/dev/null || echo "transformers未安装或已卸载"
pip uninstall transformers -y 2>/dev/null || echo "transformers未安装或已卸载"

echo "2. 清理可能的残留文件："
find /usr/local/lib/python3.10/dist-packages -name "*transformers*" -type d 2>/dev/null | xargs rm -rf 2>/dev/null || echo "无残留目录"
find /usr/local/lib/python3.10/dist-packages -name "*transformers*" -type f 2>/dev/null | xargs rm -f 2>/dev/null || echo "无残留文件"

echo "3. 清理用户目录下的残留："
find ~/.local/lib/python3.10/site-packages -name "*transformers*" -type d 2>/dev/null | xargs rm -rf 2>/dev/null || echo "无用户目录残留"
find ~/.local/lib/python3.10/site-packages -name "*transformers*" -type f 2>/dev/null | xargs rm -f 2>/dev/null || echo "无用户目录残留"

echo "4. 升级pip（使用清华镜像）："
python3 -m pip install -i "${MIRROR_PYPI_URL}" --upgrade pip

echo "5. 重新安装transformers到用户目录（使用清华镜像）："
echo "正在安装 transformers[torch] ..."
python3 -m pip install -i "${MIRROR_PYPI_URL}" --no-cache-dir --default-timeout 180 --force-reinstall --user "transformers[torch]"

echo "6. 验证安装："
python3 -c "import transformers; print('transformers版本:', transformers.__version__)" 2>/dev/null || echo "安装失败，请检查错误信息"

echo "7. 测试基本功能："
python3 -c "from transformers import AutoTokenizer; print('AutoTokenizer导入成功')" 2>/dev/null || echo "AutoTokenizer导入失败"

echo "=== 修复完成 ==="


