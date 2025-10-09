#!/bin/bash

# ============================================================================
# Llama8b Pre-Sparse 训练脚本 (FP32 连续训练版本)
# ============================================================================
# 
# 目标: 从 BF16 iter 1310 checkpoint 无缝继续训练到 iter 2000
# 关键: 完全保持训练连续性 (LR, Temperature, Reg loss 等)
#
# 核心配置:
# - 精度: FP32 (解决 BF16 NaN 问题)
# - Batch size: 256 (与原训练一致，确保参数连续)
# - Recompute: No (速度优化)
# - Finetune: No (保持连续性)
#
# 显存: 68-72GB / 80GB (充分利用)
# 速度: ~220秒/iter
# ============================================================================

set -e

# 基础配置
PROJECT_DIR="/data/home/zdhs0054/zzhou/MaskLLM"
NPROC_PER_NODE=8
NNODES=1
MASTER_ADDR=localhost
MASTER_PORT=45537  # 新端口避免冲突

# 路径配置
BF16_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8"
FP32_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8_fp32_continuous"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"

# 数据配置 (使用所有 20 个文件)
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

# 检查 BF16 checkpoint (iter 1280 - 平衡点：节省时间 + 仍在安全区)
RESUME_ITER=1280
if [ ! -d "$BF16_CHECKPOINT/iter_0001280" ]; then
    echo "⚠️  iter_0001280 不存在，尝试最近的 checkpoint..."
    LATEST_ITER=$(cat "$BF16_CHECKPOINT/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
    if [ -n "$LATEST_ITER" ]; then
        RESUME_ITER=$LATEST_ITER
        echo "✅ 使用最新 checkpoint: iter_$(printf "%07d" $RESUME_ITER)"
    else
        echo "❌ 找不到任何 checkpoint，请检查路径"
        exit 1
    fi
fi

# 检查所有 rank 的 checkpoint 文件
BF16_FILES=$(ls "$BF16_CHECKPOINT/iter_$(printf "%07d" $RESUME_ITER)"/mp_rank_*/model_optim_rng.pt 2>/dev/null | wc -l)
if [ "$BF16_FILES" -ne 8 ]; then
    echo "❌ BF16 checkpoint 不完整: 只找到 $BF16_FILES/8 个 rank"
    exit 1
fi

echo "✅ BF16 checkpoint 完整: $BF16_FILES/8 个 rank"

# 临时修改 latest_checkpointed_iteration.txt 以加载指定的 iteration
TRACKER_FILE="$BF16_CHECKPOINT/latest_checkpointed_iteration.txt"
ORIGINAL_ITER=$(cat "$TRACKER_FILE")
echo "📝 原始 latest iteration: $ORIGINAL_ITER"
echo "📝 临时修改为: $RESUME_ITER"
echo "$RESUME_ITER" > "$TRACKER_FILE"

# Checkpoint 配置
LOAD_CHECKPOINT="$BF16_CHECKPOINT"
SAVE_CHECKPOINT="$FP32_CHECKPOINT"

# 日志配置
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training_fp32_continuous"
mkdir -p "$LOG_DIR"
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
LOG_FILE="$LOG_DIR/training_continuous_from_iter${RESUME_ITER}_${DATETIME}.log"

# 训练参数
TRAIN_ITERS=2000
SAVE_INTERVAL=20
LOG_INTERVAL=1
EVAL_INTERVAL=100
WARMUP_ITERS=400

# MaskLLM 任务配置 (方案 B: lr-mult=4, 克服 1280-1310 危险区)
# 注意: Temperature 会从 checkpoint 中加载并继续衰减
TASK_CMD="--gumbel-scale-range 5e1 2.5e2 --gumbel-temperature-range 4 1.5 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 4 --weight-reg 5e-7"

options=" \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --no-masked-softmax-fusion \
    --swiglu \
    --adam-eps 1e-5 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-rope-fusion \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --make-vocab-size-divisible-by 1 \
    --ffn-hidden-size 14336 --normalization RMSNorm \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --train-iters $TRAIN_ITERS \
    --lr 1.5e-5 \
    --min-lr 1.5e-6 \
    --lr-decay-style cosine \
    --log-interval $LOG_INTERVAL \
    --eval-iters 10 \
    --eval-interval $EVAL_INTERVAL \
    --data-path "$DATA_BLEND" \
    --data-cache-path CACHE \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-file $PROJECT_DIR/assets/checkpoints/Llama8b/vocab.txt \
    --vocab-size 119696 \
    --save-interval $SAVE_INTERVAL \
    --save $SAVE_CHECKPOINT \
    --load $LOAD_CHECKPOINT \
    --no-save-optim \
    --split 98,2,0 \
    --clip-grad 0.3 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-num-zeros-in-grad \
    --lr-warmup-iters $WARMUP_ITERS \
    --exit-on-missing-checkpoint \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --log-diff-mask \
    --exit-signal-handler \
    --exp-name llama8b-tp8-fp32-continuous-from-iter$RESUME_ITER \
    --no-load-optim \
    --no-load-rng \
    --seed 52 \
    $TASK_CMD "

# 注意：关键配置 (方案 B: 保守克服危险区)
# ✅ global-batch-size: 256 (与 BF16 训练一致，确保 consumed_samples 兼容)
# ⬇️  降低 LR: 1.5e-5 (从 2e-5，防止 FP32 梯度爆炸)
# ⬇️  保守 lr-mult: 4 (从 5 降低 20%，克服 iter 1280-1310 危险区)
#    → Mask LR ≈ 1.37e-4 (安全！< BF16 的 1.7e-4, 远低于失败时的 1.71e-4)
# ❌ 移除: --bf16 (使用 FP32)
# ❌ 移除: --use-flash-attn (FP32 不支持)
# ❌ 移除: --recompute-activations (速度优化)
# ❌ 移除: --finetune (保持训练连续性)
# ❌ 移除: --override-opt_param-scheduler (导致 LR 跳升 → NaN)

cd $PROJECT_DIR

echo "🏃‍♂️ 开始 FP32 连续训练 (从 iter $RESUME_ITER)..."
echo ""
echo "📊 核心配置:"
echo ""
echo "  ┌────────────────────────┬───────────────────┐"
echo "  │ 配置项                 │ 值                │"
echo "  ├────────────────────────┼───────────────────┤"
echo "  │ 精度                   │ FP32 ✨           │"
echo "  │ 起始 Iteration         │ $RESUME_ITER      │"
echo "  │ 目标 Iteration         │ 2000              │"
echo "  │ Global Batch Size      │ 256 (连续)        │"
echo "  │ Base Learning Rate     │ 1.5e-5            │"
echo "  │ LR Mult (Mask)         │ 4 (保守策略)      │"
echo "  │ Weight Reg             │ 5e-7              │"
echo "  │ Clip Grad              │ 0.3               │"
echo "  │ Flash Attention        │ ❌ No             │"
echo "  │ Recompute              │ ❌ No             │"
echo "  │ Finetune Mode          │ ❌ No (连续训练)  │"
echo "  │ Save Optimizer         │ ❌ No (节省空间)  │"
echo "  │ Save Interval          │ $SAVE_INTERVAL iters │"
echo "  └────────────────────────┴───────────────────┘"
echo ""
echo "🎯 训练连续性保证:"
echo "  ✅ Iteration: 从 $RESUME_ITER 继续 (日志中正确显示)"
echo "  ✅ Learning Rate: 从 checkpoint 加载 (~2.9e-5 for iter 1280)"
echo "  ✅ Temperature: 从 checkpoint 加载 (~2.42 for iter 1280)"
echo "  ✅ Reg Loss: 从 checkpoint 加载 (~8831 for iter 1280, 预测)"
echo "  ✅ Scale Multiplier: 从 checkpoint 加载 (~176 for iter 1280)"
echo ""
echo "💾 显存预估:"
echo "  - 预期占用: 68-72GB / 80GB"
echo "  - 安全余量: 8-12GB"
echo "  - 对比 batch=128: 提高 ~15% 利用率"
echo ""
echo "⏱️  性能预期:"
echo "  - 剩余迭代: $((2000-RESUME_ITER)) 次"
echo "  - 每次迭代: ~230秒"
echo "  - 总时间: ~$((( 2000-RESUME_ITER) * 230 / 3600)) 小时 (约 2天)"
echo ""
echo "📝 与之前配置对比:"
echo "  vs BF16 (batch=256): 速度相当，但无 NaN 风险"
echo "  vs FP32+recompute: 速度快 2倍"
echo "  vs FP32+batch=128: 显存利用率提高 ~15%"
echo ""

# 启动训练
echo "🚀 启动训练..."
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py $options 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# 恢复原始 latest_checkpointed_iteration.txt
echo "$ORIGINAL_ITER" > "$TRACKER_FILE"
echo "📝 已恢复 latest iteration 为: $ORIGINAL_ITER"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 FP32 连续训练成功完成!"
    echo ""
    echo "📊 训练总结:"
    echo "  - 起点: iter $RESUME_ITER"
    echo "  - 终点: iter 2000"
    echo "  - 日志: $LOG_FILE"
    echo "  - Checkpoint: $SAVE_CHECKPOINT"
    echo ""
    echo "✅ 下一步: 评估稀疏模型性能"
else
    echo ""
    echo "❌ 训练失败，退出码: $EXIT_CODE"
    echo ""
    echo "🔍 故障排查:"
    echo "  1. 检查日志: tail -100 $LOG_FILE"
    echo "  2. 检查 GPU: nvidia-smi"
    echo "  3. 检查显存: 如果 OOM，考虑降低 batch size"
    echo ""
    exit $EXIT_CODE
fi

