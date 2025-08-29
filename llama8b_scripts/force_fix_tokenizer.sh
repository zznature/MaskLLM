#!/bin/bash
# 强制修复Llama8bTokenizer注册问题

echo "🔧 强制修复Llama8bTokenizer注册问题"

# 1. 确认当前arguments.py状态
echo "📋 1. 检查arguments.py状态..."
if grep -q "Llama8bTokenizer" megatron/arguments.py; then
    echo "  ✅ Llama8bTokenizer已在arguments.py中"
else
    echo "  ❌ 需要重新添加Llama8bTokenizer"
    # 重新添加
    sed -i '/Llama2Tokenizer/a\                                '\''Llama8bTokenizer'\'',' megatron/arguments.py
    echo "  ✅ 已重新添加Llama8bTokenizer"
fi

# 2. 清除Python缓存
echo "📋 2. 清除Python缓存..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  ✅ Python缓存已清除"

# 3. 强制重新导入模块测试
echo "📋 3. 强制重新导入测试..."
export PYTHONDONTWRITEBYTECODE=1  # 禁止生成.pyc文件

python3 -c "
import sys
import importlib

# 清除已导入的模块
modules_to_clear = [mod for mod in sys.modules.keys() if 'megatron' in mod]
for mod in modules_to_clear:
    del sys.modules[mod]

# 重新导入
from megatron.arguments import parse_args
print('✅ 模块重新导入成功')
"

# 4. 测试tokenizer选项
echo "📋 4. 测试tokenizer选项..."
if python3 tools/preprocess_data.py --help 2>&1 | grep -q "Llama8bTokenizer"; then
    echo "  ✅ Llama8bTokenizer现在出现在help中"
else
    echo "  ❌ Llama8bTokenizer仍未出现，尝试替代方案"
    echo "  💡 使用AutoTokenizer作为备用方案"
    
    # 创建备用配置文件
    cat > llama8b_scripts/backup_config.sh << 'EOF'
# 备用配置：使用AutoTokenizer
python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/backup_test \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type AutoTokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1
EOF
    echo "  📁 已创建备用配置文件"
    exit 1
fi

# 5. 最终测试
echo "📋 5. 执行最终测试..."
mkdir -p assets/data/c4_llama8b_pretokenized

python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/force_fix_test \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "🎉 强制修复成功！Llama8bTokenizer现在正常工作"
else
    echo "❌ 强制修复失败，需要使用备用方案"
    echo "💡 请尝试运行: bash llama8b_scripts/backup_config.sh"
    exit 1
fi
