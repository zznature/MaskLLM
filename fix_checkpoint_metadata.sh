#!/bin/bash

echo "=== 修复Checkpoint元数据问题 ==="
echo ""

CHECKPOINT_DIR="output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000"

echo "1. 检查checkpoint目录结构..."
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "找到checkpoint目录: $CHECKPOINT_DIR"
    ls -la "$CHECKPOINT_DIR"
    echo ""
    
    # 检查各个rank目录
    for rank in 0 1 2 3; do
        rank_dir="$CHECKPOINT_DIR/mp_rank_0$rank"
        if [ -d "$rank_dir" ]; then
            echo "mp_rank_0$rank 目录内容:"
            ls -la "$rank_dir"
            echo ""
        fi
    done
else
    echo "错误: checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

echo "2. 创建缺失的元数据文件..."
# 创建latest_checkpointed_iteration.txt文件
echo "2000" > "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
echo "已创建: $CHECKPOINT_DIR/latest_checkpointed_iteration.txt"
echo ""

echo "3. 检查评估脚本参数..."
echo "当前评估脚本使用的参数:"
echo "- 检查点路径: $CHECKPOINT_DIR"
echo "- 模型大小: 7b"
echo "- GPU数量: 4"
echo "- 稀疏类型: sparse"
echo ""

echo "4. 验证修复结果..."
if [ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "✅ 元数据文件已创建"
    echo "内容: $(cat $CHECKPOINT_DIR/latest_checkpointed_iteration.txt)"
else
    echo "❌ 元数据文件创建失败"
fi
echo ""

echo "5. 建议的评估命令..."
echo "现在可以重新运行评估:"
echo "bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2.sh $CHECKPOINT_DIR 7b 4 sparse"
echo ""

echo "=== 修复完成 ===" 