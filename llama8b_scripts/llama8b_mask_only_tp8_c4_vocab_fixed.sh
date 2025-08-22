#!/usr/bin/bash
# Llama8b MaskLLM 稀疏化训练脚本 - 词汇表大小修复版本
# 修复vocab_size不匹配问题

# 强制设置8GPU可见性
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置NCCL环境避免分布式冲突
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo "🚀 Llama8b MaskLLM 词汇表修复版本"
echo "📋 修复: 使用原始vocab_size=119696"
echo ""

# Get Data Blend
C4_HOME=assets/data/c4_llama2_pretokenized
DATA_BLEND=""
for i in {00000..00019}; do # 1/20
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama2_${i}_text_document"
done

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45526"

# Device Configs
NNODES=1
NPROC_PER_NODE=8
export WORLD_SIZE=8
resume=$1

# Task Configs
TAG="llama8b-tp8-mask-vocab-fixed"
DATA_INDEX_PATH=CACHE
PROJECT_PATH=$(pwd)
OUTPUT_PATH="$PROJECT_PATH/output"

# Transformer Configs - 关键修复: 使用原始词汇表大小
HIDEN_SIZE=4096
NUM_LAYERS=32
NUM_ATTN_HEADS=32
SEQ_LENGTH=4096
VOCAB_SIZE=119696  # 使用原始大小，匹配checkpoint

# Training Configs
TOKENIZER_TYPE="NullTokenizer"
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
LR=5e-5
MIN_LR=5e-6
TRAIN_ITERS=100
WARMUP_ITERS=0
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8

# intervals
SAVE_INTERVALS=50
LOG_INTERVALS=5
EVAL_INTERVALS=25
EVAL_ITERS=5

# Set Training configs
CKPT_SUBDIR="$OUTPUT_PATH/checkpoints/$TAG/train_iters_$TRAIN_ITERS"
if [ $resume -eq 0 ]; then
    LOAD="$PROJECT_PATH/output/checkpoints/llama8b_megatron_tp8" 
    EXTRA_CMD="--no-load-optim --no-load-rng --finetune --enable-partial-load " 
else
    LOAD="$CKPT_SUBDIR/ckpt"
    EXTRA_CMD=""
fi  

TASK_CMD=" --gumbel-scale-range 1e2 5e2 --gumbel-temperature-range 4 0.05 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 10 --weight-reg 1e-5 "

cd $PROJECT_PATH; mkdir -p $CKPT_SUBDIR/ckpt; mkdir -p $CKPT_SUBDIR/logs

OPTIONS=" \
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
--tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
--pipeline-model-parallel-size $PIPELINE_PARALLEL_SIZE \
--num-layers $NUM_LAYERS  \
--hidden-size $HIDEN_SIZE \
--num-attention-heads $NUM_ATTN_HEADS \
--seq-length $SEQ_LENGTH \
--max-position-embeddings $SEQ_LENGTH \
--vocab-size $VOCAB_SIZE \
--ffn-hidden-size 11008 --normalization RMSNorm \
--micro-batch-size $MICRO_BATCH_SIZE \
--global-batch-size $GLOBAL_BATCH_SIZE \
--train-iters $TRAIN_ITERS   \
--lr $LR \
--min-lr $MIN_LR \
--lr-decay-style cosine \
--log-interval $LOG_INTERVALS \
--eval-iters $EVAL_ITERS \
--eval-interval $EVAL_INTERVALS \
--data-path \"$DATA_BLEND\"  \
--data-cache-path $DATA_INDEX_PATH \
--tokenizer-type $TOKENIZER_TYPE \
--save-interval $SAVE_INTERVALS \
--save $CKPT_SUBDIR/ckpt \
--load $LOAD \
--split 98,2,0 \
--clip-grad 1.0 \
--weight-decay 0.1 \
--adam-beta1 0.9 \
--adam-beta2 0.95 \
--init-method-std 0.014  \
--log-num-zeros-in-grad \
--lr-warmup-iters $WARMUP_ITERS \
--exit-on-missing-checkpoint \
--no-gradient-accumulation-fusion \
--no-async-tensor-model-parallel-allreduce \
--use-flash-attn \
--bf16 \
--log-diff-mask \
--exit-signal-handler \
--exp-name $TAG \
${EXTRA_CMD} ${TASK_CMD}"

echo "开始词汇表修复版本稀疏化训练..."
echo "使用vocab_size=$VOCAB_SIZE (原始大小，匹配checkpoint)"
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py ${OPTIONS}
