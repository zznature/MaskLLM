#!/bin/bash
# 调查preprocess_data.py的参数定义来源

echo "🔍 调查preprocess_data.py的参数定义来源"
echo "========================================"

# 1. 检查preprocess_data.py中的tokenizer相关代码
echo "📋 1. 检查preprocess_data.py中的tokenizer定义..."
echo "  🔍 搜索tokenizer-type定义："
grep -n "tokenizer.*type" tools/preprocess_data.py | head -10

echo ""
echo "  🔍 搜索choices定义："
grep -n "choices.*=" tools/preprocess_data.py | head -5

echo ""
echo "  🔍 搜索add_argument调用："
grep -n "add_argument.*tokenizer" tools/preprocess_data.py | head -5

# 2. 检查是否有其他arguments文件
echo ""
echo "📋 2. 查找所有包含tokenizer定义的文件..."
find . -name "*.py" -exec grep -l "tokenizer.*type.*choices" {} \; 2>/dev/null | head -10

# 3. 检查preprocess_data.py的导入
echo ""
echo "📋 3. 检查preprocess_data.py的模块导入..."
echo "  🔍 前20行（查看导入）："
head -20 tools/preprocess_data.py

# 4. 查找直接定义tokenizer choices的位置
echo ""
echo "📋 4. 搜索所有包含Llama2Tokenizer但不包含Llama8bTokenizer的文件..."
grep -r "Llama2Tokenizer" . --include="*.py" | grep -v "Llama8bTokenizer" | head -5

echo ""
echo "🎯 调查完成。请执行此脚本并分享结果！"
