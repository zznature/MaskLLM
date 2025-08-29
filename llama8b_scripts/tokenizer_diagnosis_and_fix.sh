#!/bin/bash

# Llama8bTokenizer Diagnosis and Fix Script
# Purpose: Diagnose why Llama8bTokenizer is not being recognized and apply fixes

echo "🔧 Llama8bTokenizer诊断和修复脚本"
echo "======================================"

# Step 1: Check current arguments.py status
echo "📋 1. 检查arguments.py当前状态..."
if grep -q "Llama8bTokenizer" megatron/arguments.py; then
    echo "  ✅ Llama8bTokenizer已在arguments.py中注册"
    grep -A 5 -B 5 "Llama8bTokenizer" megatron/arguments.py
else
    echo "  ❌ Llama8bTokenizer未在arguments.py中找到"
    echo "  🔧 正在添加..."
    # Add Llama8bTokenizer to the choices list
    sed -i "/Llama2Tokenizer/a\\                                'Llama8bTokenizer'," megatron/arguments.py
    echo "  ✅ 已添加Llama8bTokenizer"
fi

# Step 2: Clear Python cache
echo ""
echo "📋 2. 清除Python缓存..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✅ Python缓存已清除"

# Step 3: Verify tokenizer implementation exists
echo ""
echo "📋 3. 验证tokenizer实现文件..."
if [ -f "megatron/tokenizer/llama8b_tokenizer.py" ]; then
    echo "  ✅ llama8b_tokenizer.py存在"
    echo "  📄 文件大小: $(wc -l < megatron/tokenizer/llama8b_tokenizer.py) 行"
else
    echo "  ❌ llama8b_tokenizer.py不存在"
fi

# Step 4: Check if module can be imported
echo ""
echo "📋 4. 测试模块导入..."
export PYTHONDONTWRITEBYTECODE=1
python3 -c "
try:
    import sys
    sys.path.append('.')
    from megatron.tokenizer.llama8b_tokenizer import _Llama8bTokenizer
    print('  ✅ _Llama8bTokenizer类导入成功')
except ImportError as e:
    print(f'  ❌ _Llama8bTokenizer导入失败: {e}')

try:
    from megatron.arguments import parse_args
    print('  ✅ arguments模块导入成功')
except ImportError as e:
    print(f'  ❌ arguments模块导入失败: {e}')
"

# Step 5: Test help output
echo ""
echo "📋 5. 检查命令行help输出..."
python3 tools/preprocess_data.py --help | grep -A 2 -B 2 "tokenizer-type" | head -10

# Step 6: Test if Llama8bTokenizer is now available
echo ""
echo "📋 6. 测试Llama8bTokenizer可用性..."
python3 tools/preprocess_data.py --tokenizer-type 2>&1 | tail -5

# Step 7: If still failing, try backup approach
echo ""
echo "📋 7. 如果仍然失败，准备备用方案..."
echo "备用方案1: 使用AutoTokenizer"
echo "备用方案2: 直接修改preprocess_data.py"

# Step 8: Create test command
echo ""
echo "📋 8. 准备测试命令..."
echo "如果Llama8bTokenizer可用，将执行以下命令:"
echo "python3 tools/preprocess_data.py \\"
echo "    --input \"./assets/data/c4/en/c4-train.00000-of-01024.json\" \\"
echo "    --output-prefix assets/data/c4_llama8b_pretokenized/test \\"
echo "    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \\"
echo "    --tokenizer-type Llama8bTokenizer \\"
echo "    --tokenizer-model ./assets/checkpoints/Llama8b \\"
echo "    --append-eod --workers 1"

echo ""
echo "🎯 脚本执行完成。请运行此脚本并分享结果！"
