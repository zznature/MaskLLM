#!/usr/bin/bash
# Llama8b MaskLLM 单GPU稀疏化训练测试脚本
# 避免分布式问题，专注于MaskLLM兼容性验证
# Phase 3.4: 稀疏化训练兼容性测试

# Get Data Blend
C4_HOME=assets/data/c4_llama2_pretokenized
DATA_BLEND=""
for i in {00000..00009}; do # 减少数据用于测试
    DATA_BLEND="${DATA_BLEND} 0.1 ${C4_HOME}/c4_llama2_${i}_text_document"
done
echo $DATA_BLEND 

# Single GPU配置
export CUDA_VISIBLE_DEVICES=0  # 只使用第一个GPU
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45525"

# Device Configs
NNODES=1
NPROC_PER_NODE=1  # 单GPU
export WORLD_SIZE=1
resume=$1

# Task Configs
TAG="llama8b-single-gpu-mask-test"
DATA_INDEX_PATH=CACHE
PROJECT_PATH=$(pwd)
OUTPUT_PATH="$PROJECT_PATH/output"

# Transformer Configs - Llama8b specific
HIDEN_SIZE=4096
NUM_LAYERS=32
NUM_ATTN_HEADS=32
SEQ_LENGTH=2048  # 减少序列长度节省内存
VOCAB_SIZE=119696

# Training Configs
TOKENIZER_TYPE="NullTokenizer"

TENSOR_PARALLEL_SIZE=1  # 单GPU，无并行
PIPELINE_PARALLEL_SIZE=1
LR=5e-5
MIN_LR=5e-6
TRAIN_ITERS=20  # 极少迭代，仅用于兼容性验证
WARMUP_ITERS=0
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1

# intervals
SAVE_INTERVALS=10
LOG_INTERVALS=1
EVAL_INTERVALS=10
EVAL_ITERS=2

# Set Training configs
CKPT_SUBDIR="$OUTPUT_PATH/checkpoints/$TAG/train_iters_$TRAIN_ITERS"
if [ $resume -eq 0 ]; then
    # 使用TP=8 checkpoint，enable-partial-load会自动处理
    LOAD="$PROJECT_PATH/output/checkpoints/llama8b_megatron_tp8" 
    EXTRA_CMD="--no-load-optim --no-load-rng --finetune --enable-partial-load " 
else
    LOAD="$CKPT_SUBDIR/ckpt"
    EXTRA_CMD=""
fi  

# MaskLLM稀疏化参数
TASK_CMD=" --gumbel-scale-range 1e2 5e2 --gumbel-temperature-range 4 0.05 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 10 --weight-reg 1e-5 "

cd $PROJECT_PATH; mkdir -p $CKPT_SUBDIR/ckpt; mkdir -p $CKPT_SUBDIR/logs; echo Start Llama8b Single GPU MaskLLM Test

echo "🚀 Llama8b MaskLLM单GPU兼容性测试"
echo "🎯 目标: 验证MaskLLM稀疏化组件与Llama8b的兼容性"
echo "📋 配置信息:"
echo "  模型: Llama8b (119696 vocab)"
echo "  并行: TP=1 (单GPU)"
echo "  序列长度: $SEQ_LENGTH (减少内存使用)"
echo "  训练: $TRAIN_ITERS iterations (仅兼容性测试)"
echo "  Checkpoint: $LOAD"
echo ""

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
--make-vocab-size-divisible-by 128 \
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
--data-path "$DATA_BLEND"  \
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

export CUDA_DEVICE_MAX_CONNECTIONS=1;

# 单GPU运行，无需torchrun
python pretrain_maskllm.py ${OPTIONS}
