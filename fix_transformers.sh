#!/bin/bash

echo "=== Transformers 彻底修复脚本 ==="
echo ""

echo "1. 强制卸载transformers（如果存在）："
pip3 uninstall transformers -y 2>/dev/null || echo "transformers未安装或已卸载"
pip uninstall transformers -y 2>/dev/null || echo "transformers未安装或已卸载"
echo ""

echo "2. 清理可能的残留文件："
find /usr/local/lib/python3.10/dist-packages -name "*transformers*" -type d 2>/dev/null | xargs rm -rf 2>/dev/null || echo "无残留目录"
find /usr/local/lib/python3.10/dist-packages -name "*transformers*" -type f 2>/dev/null | xargs rm -f 2>/dev/null || echo "无残留文件"
echo ""

echo "3. 清理用户目录下的残留："
find ~/.local/lib/python3.10/site-packages -name "*transformers*" -type d 2>/dev/null | xargs rm -rf 2>/dev/null || echo "无用户目录残留"
find ~/.local/lib/python3.10/site-packages -name "*transformers*" -type f 2>/dev/null | xargs rm -f 2>/dev/null || echo "无用户目录残留"
echo ""

echo "4. 升级pip："
python3 -m pip install --upgrade pip
echo ""

echo "5. 重新安装transformers："
echo "正在安装transformers[torch]..."
pip3 install transformers[torch] --no-cache-dir --force-reinstall
echo ""

echo "6. 验证安装："
echo "检查transformers版本："
python3 -c "import transformers; print('transformers版本:', transformers.__version__)" 2>/dev/null || echo "安装失败，请检查错误信息"
echo ""

echo "7. 测试基本功能："
python3 -c "from transformers import AutoTokenizer; print('AutoTokenizer导入成功')" 2>/dev/null || echo "AutoTokenizer导入失败"
echo ""

echo "=== 修复完成 ===" 