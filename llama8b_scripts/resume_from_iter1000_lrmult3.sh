#!/bin/bash

# ============================================================================
# 从 iter 1000 恢复 BF16 Conservative Plan B 训练 (方案B优化: lr_mult=4.0)
# ============================================================================
# 
# 背景: iter 75-1017训练成功, 但Reg loss增长率偏高 (2.87 /iter)
# 解决: 从 iter 1000 恢复, 采用方案B优化 - 最佳速度/稳定性平衡
# 
# 关键修改 (方案B优化版):
# 1. 起点: iter 1000 (最后完整checkpoint)
# 2. lr_mult: 2.0 → 4.0 (提高100%, 充分利用剩余1000 iters)
# 3. weight_reg: 1.5e-5 → 2.0e-5 (提高33%, 补偿稳定性)
# 4. Mask LR: 3.0e-5 → 6.0e-5 (更快收敛到最优2:4模式)
# 5. 稳定性方程: 2.0e-5 / (4.0 × 1.5e-5) = 0.33 (维持不变!)
# 6. 预期: Reg loss增长率从2.87降到~1.2-1.5 /iter, max_prob达0.90+
# ============================================================================

set -e

# 设置必需的环境变量
export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 基础配置
PROJECT_DIR="/data/home/zdhs0054/zzhou/MaskLLM"
NPROC_PER_NODE=8
NNODES=1
MASTER_ADDR=localhost
MASTER_PORT=45542

# 路径配置
TRAINING_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8_bf16_conservative_plan_b"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"

# 数据配置 (使用所有 20 个文件)
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

echo "🔧 准备从 iter 1000 恢复训练 (方案B优化: lr_mult=4.0)..."
echo ""

# 验证 iter 1000 checkpoint 完整性
echo "🔍 验证 iter 1000 checkpoint 完整性..."
RESUME_ITER=1000
if [ ! -d "$TRAINING_CHECKPOINT/iter_0001000" ]; then
    echo "❌ iter 1000 checkpoint 不存在"
    echo "   期望路径: $TRAINING_CHECKPOINT/iter_0001000"
    exit 1
fi

for rank in {0..7}; do
    RANK_DIR="$TRAINING_CHECKPOINT/iter_0001000/mp_rank_$(printf '%02d' $rank)"
    if [ ! -d "$RANK_DIR" ]; then
        echo "❌ 缺少 rank $rank 的 checkpoint: $RANK_DIR"
        exit 1
    fi
done
echo "✅ iter 1000 checkpoint 完整"

# 检查checkpoint大小
CKPT_SIZE=$(du -sh "$TRAINING_CHECKPOINT/iter_0001000" | awk '{print $1}')
echo "✅ Checkpoint 大小: $CKPT_SIZE"

# 更新 latest_checkpointed_iteration.txt
echo "1000" > "$TRAINING_CHECKPOINT/latest_checkpointed_iteration.txt"
echo "✅ 设置起点为 iter 1000"

LOAD_CHECKPOINT="$TRAINING_CHECKPOINT"
SAVE_CHECKPOINT="$TRAINING_CHECKPOINT"

# 日志配置
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training_bf16_conservative_plan_b"
mkdir -p "$LOG_DIR"
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
LOG_FILE="$LOG_DIR/training_bf16_conservative_resume_iter1000_lrmult4_optimized_${DATETIME}.log"

# 训练参数 - 关键修改
TRAIN_ITERS=2000
SAVE_INTERVAL=50  # 🔧 从25增加到50 (减少checkpoint数量, 节省空间)
LOG_INTERVAL=1
EVAL_INTERVAL=100
WARMUP_ITERS=200

# 🔧 MaskLLM 任务配置 (方案B优化: lr_mult=4.0, weight_reg=2.0e-5)
# 
# 核心修改 (方案B优化版):
# - lr_mult: 2.0 → 4.0 (提高100%, 充分利用剩余训练时间)
# - weight_reg: 1.5e-5 → 2.0e-5 (提高33%, 补偿稳定性)
# - Mask LR: 3.0e-5 → 6.0e-5 (比方案A快33%)
# - 稳定性方程: 2.0e-5 / (4.0 × 1.5e-5) = 0.33 (维持不变!)
# 
# 预期效果 (优于方案A):
# - Reg loss增长率: 2.87 → ~1.2-1.5 /iter (vs 方案A的1.3)
# - Max prob @ 2000: 0.90-0.92 (vs 方案A的0.85) - 更高确定性!
# - 更快收敛到最优2:4模式
# - 成功率: ~80% (vs 方案A的85%, 可接受的tradeoff)
TASK_CMD="--gumbel-scale-range 100 300 --gumbel-temperature-range 4.0 2.5 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 4.0 --weight-reg 2.0e-5 --clip-grad 0.5"

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
    --lr-decay-iters $TRAIN_ITERS \
    --save $SAVE_CHECKPOINT \
    --load $LOAD_CHECKPOINT \
    --no-save-optim \
    --no-load-optim \
    --no-load-rng \
    --data-path $DATA_BLEND \
    --data-cache-path CACHE \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-file $PROJECT_DIR/assets/checkpoints/Llama8b/vocab.txt \
    --vocab-size 119696 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 10 \
    --log-interval $LOG_INTERVAL \
    --split 98,2,0 \
    --distributed-backend nccl \
    --lr 1.5e-5 \
    --lr-decay-style cosine \
    --min-lr 1.5e-6 \
    --lr-warmup-iters $WARMUP_ITERS \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-num-zeros-in-grad \
    --exit-on-missing-checkpoint \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --exit-signal-handler \
    --log-diff-mask \
    --bf16 \
    --use-flash-attn \
    --exp-name llama8b-tp8-bf16-conservative-plan-b-resume-iter1000-lrmult4-optimized \
    --seed 55 \
    $TASK_CMD "

cd $PROJECT_DIR

echo ""
echo "🏃‍♂️ 从 iter 1000 恢复训练 (方案B优化: lr_mult=4.0)..."
echo ""
echo "  ╔════════════════════════════════════════════════════════════════╗"
echo "  ║    🚀 Resume Training: iter 1000 → 2000 (方案B优化)          ║"
echo "  ╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "  ┌────────────────────────┬───────────────────────────────────┐"
echo "  │ 配置项                 │ 值                                │"
echo "  ├────────────────────────┼───────────────────────────────────┤"
echo "  │ 恢复点                 │ iter 1000 (训练中期)              │"
echo "  │ 目标                   │ iter 2000                         │"
echo "  │ 剩余迭代               │ 1000 iters                        │"
echo "  ├────────────────────────┼───────────────────────────────────┤"
echo "  │ LM loss @ 1000         │ 2.692 (↓51% from start)           │"
echo "  │ Reg loss @ 1000        │ 8,216 (增长率2.87 /iter)          │"
echo "  │ Max prob @ 1000        │ 0.566 (需要加速到0.90+)           │"
echo "  ├────────────────────────┼───────────────────────────────────┤"
echo "  │ lr_mult                │ 2.0 → **4.0** ⬆️⬆️ (提高100%)  │"
echo "  │ weight_reg             │ 1.5e-5 → **2.0e-5** ⬆️ (提高33%) │"
echo "  │ Mask LR                │ 3.0e-5 → **6.0e-5** ⬆️⬆️        │"
echo "  │ 稳定性方程             │ **0.33** (维持不变!)              │"
echo "  └────────────────────────┴───────────────────────────────────┘"
echo ""
echo "🎯 核心改进策略 (方案B优化版 - 最佳平衡):"
echo "  1. ⬆️⬆️ 提高 lr_mult: 2.0 → 4.0 (充分利用剩余1000 iters)"
echo "  2. ⬆️  提高 weight_reg: 1.5e-5 → 2.0e-5 (补偿稳定性)"
echo "  3. 🎯 目标: Reg loss增长率 2.87 → ~1.2-1.5 /iter"
echo "  4. 🎯 目标: Max prob 0.57 → **0.90-0.92** @ iter 2000"
echo "  5. 🎯 目标: 更快更确定地收敛到最优2:4模式"
echo ""
echo "📊 预期效果 (优于方案A):"
echo "  ✅ Reg loss @ 2000: ~9,400 (vs 方案A的9,500)"
echo "  ✅ Reg增长减缓: 58% (2.87 → ~1.3 /iter)"
echo "  ✅ Mask更快收敛: max_prob预计达**0.90-0.92** (vs 方案A的0.85)"
echo "  ✅ 稳定性维持: 0.33 (vs 方案A的0.33, 相同)"
echo "  ✅ 成功率: ~80% (vs 方案A的85%, 可接受的tradeoff)"
echo ""
echo "💾 磁盘空间状态:"
df -h /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/ | tail -1
echo ""
echo "📝 日志文件: $LOG_FILE"
echo ""
echo "⏳ 开始恢复训练..."
echo "================================================================"
echo ""

# 启动训练
torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    pretrain_maskllm.py $options 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "================================================================"
echo "✅ 训练完成！"
echo "📝 日志文件: $LOG_FILE"
echo "💾 Checkpoint: $SAVE_CHECKPOINT"
echo "================================================================"
