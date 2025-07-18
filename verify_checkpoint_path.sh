#!/bin/bash

echo "=== 验证Checkpoint路径 ==="
echo ""

# 正确的checkpoint路径
CORRECT_PATH="output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000"

echo "1. 检查正确的checkpoint路径..."
if [ -d "$CORRECT_PATH" ]; then
    echo "✅ 找到checkpoint目录: $CORRECT_PATH"
    ls -la "$CORRECT_PATH"
    echo ""
    
    # 检查各个rank目录
    for rank in 0 1 2 3; do
        rank_dir="$CORRECT_PATH/mp_rank_0$rank"
        if [ -d "$rank_dir" ]; then
            echo "mp_rank_0$rank 目录内容:"
            ls -la "$rank_dir"
            echo ""
        fi
    done
else
    echo "❌ checkpoint目录不存在: $CORRECT_PATH"
    exit 1
fi

echo "2. 检查元数据文件..."
if [ -f "$CORRECT_PATH/latest_checkpointed_iteration.txt" ]; then
    echo "✅ 元数据文件存在"
    echo "内容: $(cat $CORRECT_PATH/latest_checkpointed_iteration.txt)"
else
    echo "❌ 元数据文件不存在，创建中..."
    echo "2000" > "$CORRECT_PATH/latest_checkpointed_iteration.txt"
    echo "✅ 已创建元数据文件"
fi
echo ""

echo "3. 验证checkpoint文件完整性..."
all_files_exist=true
for rank in 0 1 2 3; do
    checkpoint_file="$CORRECT_PATH/mp_rank_0$rank/model_optim_rng.pt"
    if [ -f "$checkpoint_file" ]; then
        file_size=$(du -h "$checkpoint_file" | cut -f1)
        echo "✅ mp_rank_0$rank: $checkpoint_file ($file_size)"
    else
        echo "❌ mp_rank_0$rank: $checkpoint_file 不存在"
        all_files_exist=false
    fi
done
echo ""

if [ "$all_files_exist" = true ]; then
    echo "4. 所有checkpoint文件都存在，路径正确！"
    echo ""
    echo "正确的评估命令："
    echo "bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2_fixed.sh $CORRECT_PATH 7b 4 sparse"
    echo ""
    echo "或者使用修复版评估脚本："
    echo "bash run_evaluation.sh $CORRECT_PATH 7b 4 sparse"
else
    echo "4. ❌ 部分checkpoint文件缺失，请检查训练是否完成"
fi

echo ""
echo "=== 验证完成 ===" 