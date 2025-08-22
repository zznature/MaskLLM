#!/usr/bin/bash
# Llama8b MaskLLM 稀疏化训练脚本
# 基于 llama2_7b_mask_only_tp8_c4.sh 适配
# Phase 3.4: 稀疏化训练兼容性测试

# Get Data Blend
# multilingual datasets
C4_HOME=assets/data/c4_llama2_pretokenized
DATA_BLEND=""
for i in {00000..00019}; do # 1/20
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama2_${i}_text_document"
done
# . ./assets/c4-blend.sh # check this file for more detials about the training data
echo $DATA_BLEND 

export MASTER_ADDR="127.0.0.1" # select the master address
export MASTER_PORT="45523" # select a different port from llama2

# Device Configs
NNODES=1 # number of nodes. 
NPROC_PER_NODE=8 # number of gpus (processes) per node
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE)) # number of gpus we have in total
resume=$1 # resume from checkpoint

# Task Configs
TAG="llama8b-tp8-mask-only-c4-test" # this will be the name of output folder
DATA_INDEX_PATH=CACHE # path to the cache folder. Will generate if not exists
PROJECT_PATH=$(pwd)
OUTPUT_PATH="$PROJECT_PATH/output"

# Transformer Configs - Llama8b specific
HIDEN_SIZE=4096 # hidden size (same as Llama2-7B)
NUM_LAYERS=32 # number of layers (same as Llama2-7B)
NUM_ATTN_HEADS=32 # number of attention heads (same as Llama2-7B)
SEQ_LENGTH=4096 # sequence length
VOCAB_SIZE=119696 # Llama8b large vocabulary size

# Training Configs - Llama8b specific
# Use Llama8b tokenizer configuration
TOKENIZER_TYPE="NullTokenizer" # Use NullTokenizer for large vocab (as verified in Phase 3.2)
# TOKENIZER_MODEL 不需要指定，因为使用NullTokenizer

TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
LR=5e-5
MIN_LR=5e-6
TRAIN_ITERS=100 # 减少训练迭代用于测试 (原来是2000)
WARMUP_ITERS=0
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8 # 减少到8用于测试 (原来是256)

# intervals
SAVE_INTERVALS=50 # 更频繁保存用于测试
LOG_INTERVALS=5   # 更频繁日志用于测试
EVAL_INTERVALS=25 # 更频繁评估用于测试
EVAL_ITERS=5      # 减少评估迭代

# Set Training configs
CKPT_SUBDIR="$OUTPUT_PATH/checkpoints/$TAG/train_iters_$TRAIN_ITERS"
if [ $resume -eq 0 ]; then
    # Use our converted Llama8b checkpoint
    LOAD="$PROJECT_PATH/output/checkpoints/llama8b_megatron_tp8" 
    EXTRA_CMD="--no-load-optim --no-load-rng --finetune --enable-partial-load " 
else
    LOAD="$CKPT_SUBDIR/ckpt"
    EXTRA_CMD=""
fi  

# MaskLLM稀疏化参数 (与Llama2相同)
TASK_CMD=" --gumbel-scale-range 1e2 5e2 --gumbel-temperature-range 4 0.05 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 10 --weight-reg 1e-5 "

cd $PROJECT_PATH; mkdir -p $CKPT_SUBDIR/ckpt; mkdir -p $CKPT_SUBDIR/logs; export WANDB_API_KEY=$WANDB_API_KEY; echo Start Llama8b MaskLLM Training

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

echo "🚀 开始Llama8b MaskLLM稀疏化训练测试..."
echo "📋 配置信息:"
echo "  模型: Llama8b (119696 vocab)"
echo "  并行: TP=$TENSOR_PARALLEL_SIZE, PP=$PIPELINE_PARALLEL_SIZE"
echo "  训练: $TRAIN_ITERS iterations, batch=$GLOBAL_BATCH_SIZE"
echo "  数据: C4 dataset"
echo "  Checkpoint: $LOAD"
echo ""

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py ${OPTIONS}
