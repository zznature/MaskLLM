#!/bin/bash
# 测试多种tokenizer配置方案

echo "🧪 测试多种Llama8b tokenizer配置方案"

# 确保输出目录存在
mkdir -p assets/data/c4_llama8b_pretokenized

# 方案1: 指向llama8b子目录（已修改）
echo "📋 方案1: 使用llama8b子目录"
python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_test1 \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b/llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "✅ 方案1成功！"
    exit 0
fi

echo "❌ 方案1失败，尝试方案2..."

# 方案2: 直接使用主目录（恢复原始配置）
echo "📋 方案2: 使用主目录"
python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_test2 \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "✅ 方案2成功！"
    exit 0
fi

echo "❌ 方案2失败，尝试方案3..."

# 方案3: 使用tokenizer_config.json路径
echo "📋 方案3: 尝试简化路径"
python3 tools/preprocess_data.py \
    --input "./assets/data/c4/en/c4-train.00000-of-01024.json" \
    --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_test3 \
    --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ./assets/checkpoints/Llama8b/tokenizer_config.json \
    --append-eod \
    --workers 1

if [ $? -eq 0 ]; then
    echo "✅ 方案3成功！"
    exit 0
fi

echo "❌ 所有方案都失败，需要进一步分析"

# 尝试获取更多调试信息
echo "🔍 尝试直接测试tokenizer构建..."
python3 -c "
import sys
sys.path.append('.')
from megatron.tokenizer.tokenizer import build_tokenizer

class MockArgs:
    def __init__(self):
        self.tokenizer_type = 'Llama2Tokenizer'
        self.tokenizer_model = './assets/checkpoints/Llama8b/llama8b'
        self.vocab_file = './assets/checkpoints/Llama8b/vocab.txt'
        self.rank = 0

try:
    args = MockArgs()
    tokenizer = build_tokenizer(args)
    print(f'✅ Tokenizer构建成功: {type(tokenizer)}')
except Exception as e:
    print(f'❌ Tokenizer构建失败: {e}')
"
