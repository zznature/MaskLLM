#!/bin/bash

# 验证并从 iter 1235 启动 FP32 训练

echo "🔍 验证配置..."
echo ""

# 检查 checkpoint
CHECKPOINT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8/iter_0001235"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ iter_0001235 checkpoint 不存在！"
    exit 1
fi

echo "✅ iter_0001235 checkpoint 存在"

# 检查所有 rank 文件
RANK_FILES=$(ls "$CHECKPOINT_DIR"/mp_rank_*/model_optim_rng.pt 2>/dev/null | wc -l)
if [ "$RANK_FILES" -ne 8 ]; then
    echo "❌ Checkpoint 不完整: 只找到 $RANK_FILES/8 个 rank"
    exit 1
fi

echo "✅ 所有 8 个 rank 文件完整"

# 检查脚本配置
SCRIPT_PATH="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/llama8b_presparse_training_tp8_fp32_continuous_from_iter1310.sh"
RESUME_ITER=$(grep "^RESUME_ITER=" "$SCRIPT_PATH" | head -1 | cut -d'=' -f2)

echo ""
echo "📋 脚本配置:"
echo "  RESUME_ITER: $RESUME_ITER"

if [ "$RESUME_ITER" != "1235" ]; then
    echo "⚠️  警告: RESUME_ITER 不是 1235，而是 $RESUME_ITER"
    echo "  请手动检查脚本"
fi

echo ""
echo "🎯 预期训练状态:"
echo "  起点: iter 1236 (日志中显示)"
echo "  LR: ~2.9e-5 (从 3.89e-5 cosine decay)"
echo "  Reg loss: ~8695"
echo "  Temperature: ~2.46"
echo ""

# 显示将要启动的命令
echo "📝 启动命令:"
echo "  bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_continuous_from_iter1310.sh"
echo ""

read -p "确认从 iter 1235 启动？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 启动训练..."
    echo ""
    bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_continuous_from_iter1310.sh
else
    echo "❌ 取消启动"
    exit 0
fi

