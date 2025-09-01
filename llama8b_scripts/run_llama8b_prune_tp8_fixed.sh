#!/bin/bash

# Llama8b 稀疏化准备脚本 (datasets错误修复版)
# 基于SparseGPT方法对Llama8b进行2:4结构化稀疏化
# 修复: 跳过WikiText2在线数据集，专注稀疏化校准

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45531" # select the port (避免与其他训练冲突)
NNODES=1 # number of nodes
NPROC_PER_NODE=8 # number of gpus (processes) per node
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE)) # number of gpus we have in total

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

SPARSEMETHOD=$1 # SparseGPT, Magnitude, Wanda
EXTRA_CMD=$2

# 如果没有提供稀疏化方法，默认使用SparseGPT
if [ -z "$SPARSEMETHOD" ]; then
    SPARSEMETHOD="SparseGPT"
    echo "使用默认稀疏化方法: $SPARSEMETHOD"
fi

# 🔧 修复: 使用不依赖在线数据集的任务配置
export TASK='c4'  # 改为使用C4任务，避免WikiText2在线下载
export SPARSITY=0.5
export PATTERN='nmprune'
export EXCLUDE=0

NSAMPLES=128
BASE_NAME="llama8b-tp8"
NAME="${BASE_NAME}.sparse.${PATTERN}.sp${SPARSITY}${SPARSEMETHOD}.ex${EXCLUDE}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
PROJECT_DIR=$(pwd)
LOG_DIR=$PROJECT_DIR/output/oneshot_pruning
mkdir -p $LOG_DIR

# Llama8b特定配置
CHECKPOINT_LOAD_DIR="$PROJECT_DIR/output/checkpoints/llama8b_megatron_tp8"
CHECKPOINT_SAVE_DIR="$PROJECT_DIR/output/oneshot_pruning/checkpoint/${NAME}"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"

# 检查输入checkpoint是否存在
if [ ! -d "$CHECKPOINT_LOAD_DIR" ]; then
    echo "❌ 错误: 输入checkpoint目录不存在: $CHECKPOINT_LOAD_DIR"
    echo "请先运行Phase 3.1模型转换："
    echo "bash scripts/tools/convert_llama8b_hf_to_megatron_simple.sh 8"
    exit 1
fi

# 检查tokenizer是否存在
if [ ! -d "$TOKENIZER_MODEL" ]; then
    echo "❌ 错误: Tokenizer目录不存在: $TOKENIZER_MODEL"
    echo "请检查Llama8b tokenizer路径"
    exit 1
fi

# 创建输出目录
mkdir -p "$CHECKPOINT_SAVE_DIR"

echo "🚀 Llama8b稀疏化准备开始 (datasets修复版)"
echo "📊 配置信息:"
echo "  - 稀疏化方法: $SPARSEMETHOD"
echo "  - 稀疏率: $SPARSITY (50%)"
echo "  - 稀疏模式: $PATTERN (2:4结构化)"
echo "  - 任务类型: $TASK (避免在线数据集)"
echo "  - 输入checkpoint: $CHECKPOINT_LOAD_DIR"
echo "  - 输出checkpoint: $CHECKPOINT_SAVE_DIR"
echo "  - Tokenizer: $TOKENIZER_MODEL"

TASK_NAME="PRUNE-C4"  # 修改任务名称
options=" \
    ${mag_options} \
    --task ${TASK_NAME} \
    --valid-data ${VALID_DATA} \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --use-rotary-position-embeddings \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --train-iters 1000 \
    --log-interval 10 \
    --overlapping-eval 4096 \
    --eval-iters 10 \
    --eval-interval 500 \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --vocab-size 119696 \
    --make-vocab-size-divisible-by 1 \
    --ffn-hidden-size 14336 --normalization RMSNorm \
    --split 99,1,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --num-query-groups 32 \
    --group-query-attention \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --exit-on-missing-checkpoint \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --log-interval 100 \
    --eval-iters 32 \
    --eval-interval 2000 \
    --data-path "None" \
    --save-interval 20000 \
    --save ${CHECKPOINT_SAVE_DIR} \
    --load ${CHECKPOINT_LOAD_DIR} \
    --hessian-compute \
    --sparse-pattern ${PATTERN} \
    --sparse-method ${SPARSEMETHOD} \
    --sparsity ${SPARSITY} \
    --row-b -1 \
    --col-b 128 \
    --prunen 2 \
    --prunem 4 \
    --hessian-samples $NSAMPLES \
    --exclude-layers-from-prune ${EXCLUDE} ${EXTRA_CMD} "

cd $PROJECT_DIR; export CUDA_DEVICE_MAX_CONNECTIONS=1; 

echo "🏃‍♂️ 开始执行稀疏化 (修复版)..."
echo "📝 日志将保存到: $LOG_DIR"
echo "🔧 修复: 使用C4任务避免WikiText2在线下载问题"

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tasks/main.py $options

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 Llama8b稀疏化完成!"
    echo "📁 稀疏模型保存在: $CHECKPOINT_SAVE_DIR"
    echo "📊 稀疏化信息:"
    echo "  - 方法: $SPARSEMETHOD"
    echo "  - 稀疏率: 50% (2:4结构化)"
    echo "  - 模型名: $NAME"
    echo ""
    echo "✅ datasets问题绕过成功"
    echo "下一步: 使用稀疏模型进行训练"
    echo "bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh"
else
    echo "❌ 稀疏化失败，退出码: $EXIT_CODE"
    echo "📝 请检查日志文件: $LOG_DIR"
    exit $EXIT_CODE
fi
