#!/bin/bash

# Llama8b预稀疏模型稀疏化训练脚本
# 使用SparseGPT生成的预稀疏模型作为起始点进行进一步的稀疏化训练

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45531"
NNODES=1
NPROC_PER_NODE=8
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE))

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 配置参数
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
PROJECT_DIR=$(pwd)

# 预稀疏模型路径（来自任务1.3的输出）
SPARSE_PATTERN="nmprune"
SPARSITY="0.5"
SPARSE_METHOD="SparseGPT"
BASE_NAME="llama8b-tp8"
PRESPARSE_NAME="${BASE_NAME}.sparse.${SPARSE_PATTERN}.sp${SPARSITY}${SPARSE_METHOD}.ex0"
PRESPARSE_CHECKPOINT="$PROJECT_DIR/output/oneshot_pruning/checkpoint/${PRESPARSE_NAME}"

# 训练数据路径（来自任务1.4的输出）
TRAIN_DATA="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized/c4_llama8b"

# 训练输出路径
TRAIN_OUTPUT="$PROJECT_DIR/output/sparse_training/llama8b_tp8_presparse_${DATETIME}"
mkdir -p $TRAIN_OUTPUT

# Llama8b tokenizer配置
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"
VOCAB_SIZE=119696

echo "🚀 开始Llama8b预稀疏模型训练..."
echo "📋 训练配置:"
echo "   - 预稀疏模型: ${PRESPARSE_CHECKPOINT}"
echo "   - 训练数据: ${TRAIN_DATA}"
echo "   - 输出路径: ${TRAIN_OUTPUT}"
echo "   - 张量并行: 8"
echo "   - 词汇表大小: ${VOCAB_SIZE}"

# 检查预稀疏模型是否存在
if [ ! -d "${PRESPARSE_CHECKPOINT}" ]; then
    echo "❌ 错误: 预稀疏模型不存在: ${PRESPARSE_CHECKPOINT}"
    echo "💡 请先运行任务1.3生成预稀疏模型:"
    echo "   bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh SparseGPT"
    exit 1
fi

# 检查训练数据是否存在
if [ ! -f "${TRAIN_DATA}_text_document.bin" ]; then
    echo "❌ 错误: 训练数据不存在: ${TRAIN_DATA}_text_document.bin"
    echo "💡 请先运行任务1.4生成训练数据:"
    echo "   bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh"
    exit 1
fi

options=" \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 2 \
    --global-batch-size 256 \
    --train-iters 5000 \
    --log-interval 10 \
    --eval-iters 50 \
    --eval-interval 500 \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --vocab-size ${VOCAB_SIZE} \
    --make-vocab-size-divisible-by 1 \
    --ffn-hidden-size 11008 \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --disable-bias-linear \
    --no-position-embedding \
    --use-flash-attn \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --weight-decay 0.01 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000 \
    --bf16 \
    --no-masked-softmax-fusion \
    --split 99,1,0 \
    --data-path ${TRAIN_DATA} \
    --save-interval 1000 \
    --save ${TRAIN_OUTPUT} \
    --load ${PRESPARSE_CHECKPOINT} \
    --dataloader-type cyclic \
    --no-load-optim \
    --no-load-rng \
    --num-query-groups 32 \
    --group-query-attention \
    --mask-only \
    --save-separate-masked-params \
    --sparse-learning \
    --sparse-pattern nmprune \
    --sparsity 0.5 \
    --prunen 2 \
    --prunem 4 \
    --mask-update-method magnitude \
    --mask-update-interval 100 \
    --mask-lr 1e-3"

echo "🔄 启动训练进程..."

cd $PROJECT_DIR

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    pretrain_maskllm.py $options

echo "✅ Llama8b预稀疏模型训练完成!"
echo "📁 输出路径: ${TRAIN_OUTPUT}"
