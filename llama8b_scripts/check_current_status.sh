#!/bin/bash
# 检查当前tokenizer配置状态

echo "🔍 检查当前tokenizer配置状态"
echo "================================"

# 1. 检查prepare_c4_megatron_llama8b.sh当前使用什么tokenizer
echo "📋 1. 当前prepare_c4_megatron_llama8b.sh配置："
grep -A 2 -B 2 "tokenizer-type" llama8b_scripts/prepare_c4_megatron_llama8b.sh

# 2. 检查tools/preprocess_data.py是否已修复
echo ""
echo "📋 2. tools/preprocess_data.py中的tokenizer choices："
grep -A 6 "choices=\[" tools/preprocess_data.py

# 3. 检查help输出
echo ""
echo "📋 3. 当前可用的tokenizer类型："
python3 tools/preprocess_data.py --help 2>&1 | grep -A 1 "tokenizer-type"

echo ""
echo "🎯 状态检查完成！"
