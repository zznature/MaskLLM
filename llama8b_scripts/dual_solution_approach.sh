#!/bin/bash
# 双重解决方案：调查+修复

echo "🎯 Llama8b Tokenizer双重解决方案"
echo "================================="

# 方案A: 调查参数定义来源
echo "📋 方案A: 调查preprocess_data.py参数来源..."

# 1. 找到tokenizer定义的确切位置
echo "  🔍 1. 查找tokenizer choices定义："
grep -rn "choices.*=.*\[.*Llama2Tokenizer" . --include="*.py" | head -3

# 2. 检查preprocess_data.py的参数解析
echo ""
echo "  🔍 2. 检查preprocess_data.py参数解析部分："
grep -A 20 -B 5 "tokenizer.*type" tools/preprocess_data.py

# 方案B: 备用解决方案（使用AutoTokenizer）
echo ""
echo "📋 方案B: 备用解决方案测试..."

# 3. 测试AutoTokenizer是否可用
echo "  🧪 3. 测试AutoTokenizer方案："
mkdir -p assets/data/c4_llama8b_pretokenized

echo "正在测试AutoTokenizer..."
python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/autotokenizer_test \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type AutoTokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "✅ AutoTokenizer方案成功！"
    echo "💡 建议：更新prepare_c4_megatron_llama8b.sh使用AutoTokenizer"
    
    # 自动更新配置文件
    echo "  🔧 自动更新配置文件..."
    sed -i 's/--tokenizer-type Llama8bTokenizer/--tokenizer-type AutoTokenizer/' llama8b_scripts/prepare_c4_megatron_llama8b.sh
    echo "  ✅ 已更新prepare_c4_megatron_llama8b.sh使用AutoTokenizer"
    
    echo ""
    echo "🎉 解决方案找到！现在可以运行："
    echo "   source container_environment_setup.sh && bash llama8b_scripts/prepare_c4_megatron_llama8b.sh"
    
else
    echo "❌ AutoTokenizer方案失败"
    echo "📋 需要进一步调查preprocess_data.py的参数定义"
fi

echo ""
echo "🎯 双重方案执行完成！"
