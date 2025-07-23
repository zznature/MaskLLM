#!/usr/bin/bash

# WikiText-103 Incremental Training Script
# Updated for new C4-format preprocessed dataset: assets/data/wikitext103_llama2_pretokenized/
# Optimized sequence length and batch size for WikiText-103 characteristics

# Get Data Blend
# wikitext-103 dataset (updated path - processed with C4 format)
WIKI_HOME=assets/data/wikitext103_llama2_pretokenized
DATA_BLEND="1.0 ${WIKI_HOME}/wiki_train_llama2_text_document"
echo $DATA_BLEND 

export MASTER_ADDR="127.0.0.1" # select the master address
export MASTER_PORT="45522" # select the port - CHANGED TO MATCH C4 SCRIPT

# # NCCL Configuration for better reliability
# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=1800  # 30 minutes instead of 10
# export NCCL_SOCKET_IFNAME=lo  # Force using loopback interface for single-node
# export NCCL_P2P_DISABLE=1     # Disable GPU direct P2P transfers
# export NCCL_IB_DISABLE=1      # Disable InfiniBand
# export NCCL_NET_GDR_LEVEL=0   # Disable GPU Direct RDMA
# export NCCL_TREE_THRESHOLD=0  # Force ring algorithm
# export NCCL_ALGO=Ring         # Use ring algorithm instead of tree
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Reduce memory fragmentation

# Device Configs
NNODES=1 # number of nodes. 
NPROC_PER_NODE=4 # number of gpus (processes) per node
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE)) # number of gpus we have in total
resume=$1 # resume from checkpoint

# Task Configs
TAG="llama2-7b-tp4-mask-only-wikitext103-c4format" # this will be the name of output folder
DATA_INDEX_PATH=CACHE # path to the cache folder. Will generate if not exists
PROJECT_PATH=$(pwd)
OUTPUT_PATH="$PROJECT_PATH/output"

# Transformer Configs
HIDEN_SIZE=4096 # hidden size
NUM_LAYERS=32 # number of layers
NUM_ATTN_HEADS=32 # number of attention heads
SEQ_LENGTH=2048 # sequence length (optimized for WikiText-103: most docs < 2048 tokens)

# Training Configs
TOKENIZER_MODEL="$PROJECT_PATH/assets/checkpoints/llama2_7b_hf/tokenizer.model" # path to the tokenizer model

TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=1
LR=2e-5
MIN_LR=2e-6
TRAIN_ITERS=800 # number of iterations to train for incremental training (increased for WikiText-103)
WARMUP_ITERS=80 # warmup iterations (10% of TRAIN_ITERS)
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=64 # Reduced for WikiText-103 (smaller dataset, avoid overfitting)

# intervals
SAVE_INTERVALS=200
LOG_INTERVALS=10
EVAL_INTERVALS=40
EVAL_ITERS=5

# Set Training configs
CKPT_SUBDIR="$OUTPUT_PATH/checkpoints/$TAG/train_iters_$TRAIN_ITERS"
if [ $resume -eq 0 ]; then
    # Load from the checkpoint trained on C4
    LOAD="$PROJECT_PATH/output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt" 
    EXTRA_CMD="--no-load-optim --no-load-rng --finetune --enable-partial-load " 
else
    LOAD="$CKPT_SUBDIR/ckpt"
    EXTRA_CMD=""
fi  

# According to Project_Tasks.md, we need to adjust the gumbel temperature range
TASK_CMD=" --gumbel-scale-range 1e2 5e2 --gumbel-temperature-range 2 0.05 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 10 --weight-reg 1e-5 "

cd $PROJECT_PATH; mkdir -p $CKPT_SUBDIR/ckpt; mkdir -p $CKPT_SUBDIR/logs; export WANDB_API_KEY=$WANDB_API_KEY; echo Start Training

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
--make-vocab-size-divisible-by 1 \
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
--tokenizer-type Llama2Tokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
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

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py ${OPTIONS} 