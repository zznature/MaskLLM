#!/bin/bash

# Llama8b 预稀疏模型训练脚本
# 基于oneshot剪枝产生的稀疏模型进行后续训练
# 使用MaskLLM框架进行稀疏化训练

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45532" # 避免与剪枝脚本端口冲突
NNODES=1
NPROC_PER_NODE=8
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE))

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 参数解析
RESUME_FLAG=${1:-0}  # 0=从预稀疏checkpoint开始，1=从训练checkpoint恢复
EXTRA_CMD=$2

# 稀疏模型配置
SPARSEMETHOD="SparseGPT"
SPARSITY=0.5
PATTERN="nmprune"
EXCLUDE=0
BASE_NAME="llama8b-tp8"
SPARSE_NAME="${BASE_NAME}.sparse.${PATTERN}.sp${SPARSITY}${SPARSEMETHOD}.ex${EXCLUDE}"

PROJECT_DIR=$(pwd)
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

# 路径配置
PRESPARSE_CHECKPOINT="$PROJECT_DIR/output/oneshot_pruning/checkpoint/${SPARSE_NAME}"
TRAINING_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"
DATA_PATH="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized/c4_llama8b"

# 检查预稀疏checkpoint
if [ ! -d "$PRESPARSE_CHECKPOINT" ]; then
    echo "❌ 错误: 预稀疏checkpoint不存在: $PRESPARSE_CHECKPOINT"
    echo "请先运行稀疏化脚本："
    echo "bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh"
    exit 1
fi

# 检查数据
if [ ! -f "${DATA_PATH}_text_document.bin" ]; then
    echo "❌ 错误: 预处理数据不存在: ${DATA_PATH}_text_document.bin"
    echo "请先运行数据预处理："
    echo "bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh"
    exit 1
fi

# 根据resume标志设置checkpoint路径
if [ $RESUME_FLAG -eq 1 ]; then
    LOAD_CHECKPOINT="$TRAINING_CHECKPOINT"
    echo "🔄 恢复训练模式: 从训练checkpoint恢复"
else
    LOAD_CHECKPOINT="$PRESPARSE_CHECKPOINT"
    echo "🚀 初始训练模式: 从预稀疏checkpoint开始"
fi

SAVE_CHECKPOINT="$TRAINING_CHECKPOINT"
mkdir -p "$SAVE_CHECKPOINT"

echo "🎯 Llama8b预稀疏模型训练配置:"
echo "  - 预稀疏checkpoint: $PRESPARSE_CHECKPOINT"
echo "  - 加载checkpoint: $LOAD_CHECKPOINT"
echo "  - 保存checkpoint: $SAVE_CHECKPOINT"
echo "  - 数据路径: $DATA_PATH"
echo "  - 稀疏率: ${SPARSITY} (50%)"
echo "  - 稀疏模式: ${PATTERN} (2:4结构化)"

# 训练参数
TRAIN_ITERS=10000
SAVE_INTERVAL=1000
EVAL_INTERVAL=500
LOG_INTERVAL=50

# 创建日志目录
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_${DATETIME}.log"

echo "📝 日志文件: $LOG_FILE"

options=" \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters $TRAIN_ITERS \
    --save $SAVE_CHECKPOINT \
    --load $LOAD_CHECKPOINT \
    --data-path $DATA_PATH \
    --vocab-file $PROJECT_DIR/assets/checkpoints/Llama8b/vocab.txt \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-size 119696 \
    --split 949,50,1 \
    --distributed-backend nccl \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction 0.01 \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 10 \
    --bf16 \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --swiglu \
    --normalization RMSNorm \
    --num-query-groups 32 \
    --group-query-attention \
    --ffn-hidden-size 14336 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --make-vocab-size-divisible-by 1 \
    --no-masked-softmax-fusion \
    --exit-on-missing-checkpoint \
    --no-load-optim \
    --no-load-rng \
    --mask-only \
    --sparse-mode \
    --sparsity $SPARSITY \
    --sparse-pattern $PATTERN \
    --mask-learning-rate 0.1 \
    --mask-decay-rate 0.98 \
    --mask-update-freq 100 \
    --structured-sparsity \
    $EXTRA_CMD "

cd $PROJECT_DIR

echo "🏃‍♂️ 开始预稀疏模型训练..."
echo "📊 训练配置:"
echo "  - 训练迭代: $TRAIN_ITERS"
echo "  - 保存间隔: $SAVE_INTERVAL"
echo "  - 评估间隔: $EVAL_INTERVAL"
echo "  - 批大小: 16 (global)"
echo "  - 学习率: 1e-5"
echo "  - Mask学习率: 0.1"

# 启动训练
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_gpt.py $options 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 Llama8b预稀疏模型训练完成!"
    echo "📁 训练checkpoint保存在: $SAVE_CHECKPOINT"
    echo "📝 完整日志: $LOG_FILE"
    echo ""
    echo "📊 训练总结:"
    echo "  - 预稀疏模型: ✅"
    echo "  - 稀疏化训练: ✅"
    echo "  - Mask优化: ✅"
    echo "  - 结构化稀疏: ✅ (2:4)"
    echo ""
    echo "下一步: 评估稀疏模型性能"
    echo "bash run_maskllm_native.sh llama8b_scripts/evaluate_sparse_model.sh"
else
    echo ""
    echo "❌ 训练失败，退出码: $EXIT_CODE"
    echo "📝 请检查日志: $LOG_FILE"
    echo ""
    echo "🔧 可能的解决方案:"
    echo "  1. 检查GPU内存是否充足"
    echo "  2. 调整批大小参数"
    echo "  3. 检查checkpoint完整性"
    echo "  4. 验证数据路径"
    exit $EXIT_CODE
fi
