#!/bin/bash

echo "=== Transformers 安全修复脚本 ==="
echo ""

echo "1. 检查磁盘空间："
df -h /usr/local
echo ""

echo "2. 检查pip状态："
which pip3
which pip
echo ""

echo "3. 尝试使用python -m pip（避免权限问题）："
echo "检查pip版本："
python3 -m pip --version
echo ""

echo "4. 清理pip缓存（释放空间）："
python3 -m pip cache purge
echo ""

echo "5. 检查transformers是否已安装："
python3 -c "import transformers; print('transformers已安装，版本:', transformers.__version__)" 2>/dev/null || echo "transformers未安装"
echo ""

echo "6. 如果transformers未安装，尝试安装："
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "正在安装transformers..."
    python3 -m pip install transformers --no-cache-dir --user
else
    echo "transformers已存在，跳过安装"
fi
echo ""

echo "7. 验证安装："
python3 -c "import transformers; print('transformers版本:', transformers.__version__)" 2>/dev/null || echo "transformers安装失败"
echo ""

echo "8. 测试基本功能："
python3 -c "from transformers import AutoTokenizer; print('AutoTokenizer导入成功')" 2>/dev/null || echo "AutoTokenizer导入失败"
echo ""

echo "=== 安全修复完成 ===" 