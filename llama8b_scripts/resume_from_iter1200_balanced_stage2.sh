#!/bin/bash

# ============================================================================
# Llama8b BF16 Balanced Training - Stage 2 (iter 1200 → 2000)
# 目标: 在稳定性=0.5的前提下加速mask学习与PPL下降
# 方案: lr_mult=4.0, weight_reg=3.0e-5, scale 100→300, temp 4.0→2.5
# ============================================================================

set -e

# 基础环境
export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 集群/分布式配置
PROJECT_DIR="/data/home/zdhs0054/zzhou/MaskLLM"
NPROC_PER_NODE=8
NNODES=1
MASTER_ADDR=localhost
MASTER_PORT=45544

# 路径配置
TRAINING_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8_bf16_conservative_plan_b"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"

# 数据配置 (使用 20 个文件)
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

echo "🔧 阶段2: 从 iter 1200 恢复训练 (稳定性=0.5, 加速收敛)"
echo ""

# 恢复点检查
RESUME_ITER=1200
if [ ! -d "$TRAINING_CHECKPOINT/iter_0001200" ]; then
    echo "❌ iter 1200 checkpoint 不存在"
    echo "   期望路径: $TRAINING_CHECKPOINT/iter_0001200"
    exit 1
fi

echo "🔍 验证 iter 1200 checkpoint 完整性..."
for rank in {0..7}; do
    RANK_DIR="$TRAINING_CHECKPOINT/iter_0001200/mp_rank_$(printf '%02d' $rank)"
    if [ ! -d "$RANK_DIR" ]; then
        echo "❌ 缺少 rank $rank 的 checkpoint: $RANK_DIR"
        exit 1
    fi
done
echo "✅ iter 1200 checkpoint 完整"

# 备份并设置 latest_checkpointed_iteration.txt
LATEST_FILE="$TRAINING_CHECKPOINT/latest_checkpointed_iteration.txt"
BACKUP_FILE="$TRAINING_CHECKPOINT/latest_checkpointed_iteration.txt.backup_stage2_iter1200"
if [ -f "$LATEST_FILE" ]; then
    cp "$LATEST_FILE" "$BACKUP_FILE"
    echo "✅ 已备份 latest_checkpointed_iteration.txt"
fi
echo "$RESUME_ITER" > "$LATEST_FILE"
echo "✅ 临时设置 checkpoint iteration 为 $RESUME_ITER"

cleanup() {
    if [ -f "$BACKUP_FILE" ]; then
        mv "$BACKUP_FILE" "$LATEST_FILE"
        echo "✅ 已恢复 latest_checkpointed_iteration.txt"
    fi
}
trap cleanup EXIT

LOAD_CHECKPOINT="$TRAINING_CHECKPOINT"
SAVE_CHECKPOINT="$TRAINING_CHECKPOINT"
EXTRA_CMD="--no-load-optim --no-load-rng --no-save-optim"

# 日志
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training_bf16_balanced_two_stage"
mkdir -p "$LOG_DIR"
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
LOG_FILE="$LOG_DIR/stage2_resume_iter1200_to_2000_${DATETIME}.log"

# 训练区间与间隔
TRAIN_ITERS=2000
SAVE_INTERVAL=50  # 阶段2可提高间隔
EVAL_INTERVAL=100
LOG_INTERVAL=1
WARMUP_ITERS=200

# 阶段2 MaskLLM 核心参数 (stability = 3.0e-5 / (4.0 × 1.5e-5) = 0.5)
TASK_CMD="--gumbel-scale-range 100 300 \
          --gumbel-temperature-range 4.0 2.5 \
          --N 2 --M 4 \
          --mask-only \
          --prior-strength 3.0 \
          --lr-mult 4.0 \
          --weight-reg 3.0e-5 \
          --clip-grad 0.5"

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
    $EXTRA_CMD \
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
    --exp-name llama8b-tp8-bf16-balanced-stage2-iter1200-2000 \
    --seed 55 \
    $TASK_CMD "

cd $PROJECT_DIR

echo "  ╔════════════════════════════════════════════════════════════════╗"
echo "  ║    🚀 Stage 2: iter 1200 → 2000 (Stability = 0.5)            ║"
echo "  ╚════════════════════════════════════════════════════════════════╝"
echo "📝 日志: $LOG_FILE"
echo "📌 参数: lr_mult=4.0, weight_reg=3.0e-5, scale 100→300, temp 4.0→2.5"
echo "⚠️ 监控: reg增速≤1.5/iter, grad norm≤0.08, max_prob持续上升"
echo "================================================================"

torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    pretrain_maskllm.py $options 2>&1 | tee -a "$LOG_FILE"

echo "================================================================"
echo "✅ 阶段2完成！"
echo "📝 日志: $LOG_FILE"
echo "================================================================"


