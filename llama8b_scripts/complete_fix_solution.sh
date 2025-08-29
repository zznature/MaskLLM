#!/bin/bash
# 完整修复方案：修复preprocess_data.py并更新配置

echo "🔧 完整修复方案"
echo "==============="

# 步骤1: 修复tools/preprocess_data.py中的tokenizer定义
echo "📋 步骤1: 修复tools/preprocess_data.py..."

# 备份原文件
cp tools/preprocess_data.py tools/preprocess_data.py.backup

# 检查是否已包含Llama8bTokenizer
if grep -q "Llama8bTokenizer" tools/preprocess_data.py; then
    echo "  ✅ Llama8bTokenizer已在preprocess_data.py中"
else
    echo "  🔧 添加Llama8bTokenizer到preprocess_data.py..."
    # 在Llama2Tokenizer后添加Llama8bTokenizer
    sed -i "/choices=\[/,/\]/s/'Llama2Tokenizer',/'Llama2Tokenizer',\n                                'Llama8bTokenizer',/" tools/preprocess_data.py
    echo "  ✅ 已添加Llama8bTokenizer"
fi

# 步骤2: 更新prepare_c4_megatron_llama8b.sh使用正确的tokenizer
echo ""
echo "📋 步骤2: 更新prepare_c4_megatron_llama8b.sh配置..."

# 恢复使用Llama8bTokenizer
sed -i 's/--tokenizer-type AutoTokenizer/--tokenizer-type Llama8bTokenizer/' llama8b_scripts/prepare_c4_megatron_llama8b.sh
echo "  ✅ 已更新为使用Llama8bTokenizer"

# 步骤3: 验证修复结果
echo ""
echo "📋 步骤3: 验证修复结果..."

# 检查help输出
if python3 tools/preprocess_data.py --help 2>&1 | grep -q "Llama8bTokenizer"; then
    echo "  ✅ Llama8bTokenizer现在出现在help中"
else
    echo "  ❌ 修复失败"
    exit 1
fi

# 步骤4: 测试单文件处理
echo ""
echo "📋 步骤4: 测试单文件处理..."
mkdir -p assets/data/c4_llama8b_pretokenized

python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/complete_fix_test \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 完整修复成功！"
    echo "✅ 现在可以运行完整的C4数据预处理："
    echo "   source container_environment_setup.sh && bash llama8b_scripts/prepare_c4_megatron_llama8b.sh"
else
    echo ""
    echo "❌ 仍有问题，需要进一步调试"
    echo "📋 请检查上面的错误信息"
fi

echo ""
echo "🎯 完整修复方案执行完成！"
