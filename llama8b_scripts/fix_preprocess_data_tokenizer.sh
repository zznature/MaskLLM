#!/bin/bash
# 直接修复tools/preprocess_data.py中的tokenizer定义

echo "🔧 修复preprocess_data.py中的tokenizer定义"
echo "==========================================="

# 1. 备份原文件
echo "📋 1. 备份原文件..."
cp tools/preprocess_data.py tools/preprocess_data.py.backup
echo "  ✅ 已备份到 tools/preprocess_data.py.backup"

# 2. 显示当前的choices定义
echo ""
echo "📋 2. 当前的choices定义："
grep -A 4 -B 2 "choices=\[" tools/preprocess_data.py

# 3. 添加Llama8bTokenizer到choices列表
echo ""
echo "📋 3. 添加Llama8bTokenizer..."
sed -i "/choices=\[/,/\]/s/'Llama2Tokenizer',/'Llama2Tokenizer',\n                                'Llama8bTokenizer',/" tools/preprocess_data.py
echo "  ✅ 已添加Llama8bTokenizer到choices列表"

# 4. 验证修改结果
echo ""
echo "📋 4. 验证修改结果："
grep -A 6 -B 2 "choices=\[" tools/preprocess_data.py

# 5. 测试修改后的tokenizer选项
echo ""
echo "📋 5. 测试修改后的tokenizer选项..."
if python3 tools/preprocess_data.py --help 2>&1 | grep -q "Llama8bTokenizer"; then
    echo "  ✅ Llama8bTokenizer现在出现在help中"
else
    echo "  ❌ 修改失败，恢复备份文件"
    cp tools/preprocess_data.py.backup tools/preprocess_data.py
    exit 1
fi

# 6. 测试实际使用
echo ""
echo "📋 6. 测试Llama8bTokenizer实际使用..."
mkdir -p assets/data/c4_llama8b_pretokenized

python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/final_test \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 成功！Llama8bTokenizer现在完全工作"
    echo "✅ 可以运行完整的C4数据预处理："
    echo "   source container_environment_setup.sh && bash llama8b_scripts/prepare_c4_megatron_llama8b.sh"
else
    echo ""
    echo "❌ 仍有其他问题，需要进一步调试"
    echo "📋 检查错误输出并分析具体问题"
fi

echo ""
echo "🎯 修复脚本执行完成！"
