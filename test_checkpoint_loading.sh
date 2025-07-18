#!/bin/bash

echo "=== 测试Checkpoint加载 ==="
echo ""

# 正确的checkpoint路径
CHECKPOINT_PATH="output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000"

echo "1. 验证checkpoint路径..."
echo "路径: $CHECKPOINT_PATH"
if [ -d "$CHECKPOINT_PATH" ]; then
    echo "✅ 目录存在"
else
    echo "❌ 目录不存在"
    exit 1
fi
echo ""

echo "2. 检查checkpoint文件..."
for rank in 0 1 2 3; do
    file="$CHECKPOINT_PATH/mp_rank_0$rank/model_optim_rng.pt"
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "✅ mp_rank_0$rank: $size"
    else
        echo "❌ mp_rank_0$rank: 文件不存在"
    fi
done
echo ""

echo "3. 检查元数据文件..."
metadata_file="$CHECKPOINT_PATH/latest_checkpointed_iteration.txt"
if [ -f "$metadata_file" ]; then
    echo "✅ 元数据文件存在: $(cat $metadata_file)"
else
    echo "❌ 元数据文件不存在，创建中..."
    echo "2000" > "$metadata_file"
    echo "✅ 已创建元数据文件"
fi
echo ""

echo "4. 尝试不同的路径格式..."
echo "原始路径: $CHECKPOINT_PATH"
echo "父目录: $(dirname $CHECKPOINT_PATH)"
echo ""

echo "5. 建议的解决方案..."
echo "如果仍然出现路径重复错误，请尝试以下路径之一："
echo ""
echo "选项1: 使用父目录"
echo "bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2_fixed.sh $(dirname $CHECKPOINT_PATH) 7b 4 sparse"
echo ""
echo "选项2: 使用绝对路径"
echo "bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2_fixed.sh $(pwd)/$CHECKPOINT_PATH 7b 4 sparse"
echo ""
echo "选项3: 使用相对路径（从项目根目录）"
echo "bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2_fixed.sh $CHECKPOINT_PATH 7b 4 sparse"
echo ""

echo "=== 测试完成 ===" 