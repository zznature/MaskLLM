#!/bin/bash

# ============================================================================
# 从 iter 75 恢复 BF16 Conservative Plan B 训练
# ============================================================================
# 
# 问题: iter 100 保存失败 (磁盘空间不足, checkpoint不完整)
# 解决: 从 iter 75 恢复, 并优化checkpoint保存策略
# 
# 修改:
# 1. 删除不完整的 iter 100 checkpoint
# 2. 恢复 latest_checkpointed_iteration.txt 为 75
# 3. 增加 save_interval 到 50 (减少checkpoint数量)
# 4. 添加 --no-save-optim (节省空间)
# 5. 添加 --no-load-optim 和 --no-load-rng (从checkpoint恢复时跳过optimizer)
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

echo "🔧 准备恢复训练..."
echo ""

# 1. 删除不完整的 iter 100 checkpoint
if [ -d "$TRAINING_CHECKPOINT/iter_0000100" ]; then
    echo "⚠️  删除不完整的 iter 100 checkpoint..."
    rm -rf "$TRAINING_CHECKPOINT/iter_0000100"
    echo "✅ 已删除"
fi

# 2. 恢复 latest_checkpointed_iteration.txt
echo "75" > "$TRAINING_CHECKPOINT/latest_checkpointed_iteration.txt"
echo "✅ 恢复 latest_checkpointed_iteration.txt 为 75"

# 3. 验证 iter 75 checkpoint 完整性
echo ""
echo "🔍 验证 iter 75 checkpoint 完整性..."
for rank in {0..7}; do
    RANK_DIR="$TRAINING_CHECKPOINT/iter_0000075/mp_rank_$(printf '%02d' $rank)"
    if [ ! -d "$RANK_DIR" ]; then
        echo "❌ 缺少 rank $rank 的 checkpoint: $RANK_DIR"
        exit 1
    fi
done
echo "✅ iter 75 checkpoint 完整 (53GB)"

LOAD_CHECKPOINT="$TRAINING_CHECKPOINT"
SAVE_CHECKPOINT="$TRAINING_CHECKPOINT"

# 日志配置
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training_bf16_conservative_plan_b"
mkdir -p "$LOG_DIR"
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
LOG_FILE="$LOG_DIR/training_bf16_conservative_resume_iter75_${DATETIME}.log"

# 训练参数 - 关键修改
TRAIN_ITERS=2000
SAVE_INTERVAL=50  # 🔧 从25增加到50 (减少checkpoint数量, 节省空间)
LOG_INTERVAL=1
EVAL_INTERVAL=100
WARMUP_ITERS=200

# 🔧 MaskLLM 任务配置 (与原配置相同)
TASK_CMD="--gumbel-scale-range 100 300 --gumbel-temperature-range 4.0 2.5 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 2.0 --weight-reg 1.5e-5 --clip-grad 0.5"

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
    --exp-name llama8b-tp8-bf16-conservative-plan-b-resume \
    --seed 55 \
    $TASK_CMD "

cd $PROJECT_DIR

echo ""
echo "🏃‍♂️ 从 iter 75 恢复 BF16 Conservative Plan B 训练..."
echo ""
echo "  ╔════════════════════════════════════════════════════════════════╗"
echo "  ║    🔄 Resume Training: 从 iter 75 恢复                        ║"
echo "  ╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "  ┌────────────────────────┬───────────────────────────────────┐"
echo "  │ 配置项                 │ 值                                │"
echo "  ├────────────────────────┼───────────────────────────────────┤"
echo "  │ 恢复点                 │ iter 75 (最后完整checkpoint)      │"
echo "  │ 目标                   │ iter 2000                         │"
echo "  │ 剩余迭代               │ 1925 iters                        │"
echo "  ├────────────────────────┼───────────────────────────────────┤"
echo "  │ Save Interval          │ 50 (原25, 减少checkpoint数)       │"
echo "  │ 节省空间               │ ~50% checkpoint数量               │"
echo "  │ --no-save-optim        │ ✅ 启用 (节省optimizer空间)       │"
echo "  └────────────────────────┴───────────────────────────────────┘"
echo ""
echo "⚠️  重要修改 (应对磁盘空间问题):"
echo "  1. Save interval: 25 → 50 (减少checkpoint数量)"
echo "  2. 已启用 --no-save-optim (节省空间)"
echo "  3. 已启用 --no-load-optim 和 --no-load-rng (从checkpoint恢复)"
echo "  4. 删除了不完整的 iter 100 checkpoint"
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
