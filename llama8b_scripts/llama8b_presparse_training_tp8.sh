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
# 🔧 采用与llama2验证脚本相同的数据配置格式
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

# 检查预稀疏checkpoint
if [ ! -d "$PRESPARSE_CHECKPOINT" ]; then
    echo "❌ 错误: 预稀疏checkpoint不存在: $PRESPARSE_CHECKPOINT"
    echo "请先运行稀疏化脚本："
    echo "bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh"
    exit 1
fi

# 检查数据文件 (采用与llama2相同的检查方式)
SAMPLE_FILE="${C4_HOME}/c4_llama8b_00000_text_document.bin"
if [ ! -f "$SAMPLE_FILE" ]; then
    echo "❌ 错误: 预处理数据不存在: $SAMPLE_FILE"
    echo "请先运行数据预处理："
    echo "bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh"
    exit 1
fi

# 统计可用数据文件
AVAILABLE_FILES=$(echo "$DATA_BLEND" | wc -w)
AVAILABLE_FILES=$((AVAILABLE_FILES / 2))  # 每个文件对应weight+path两个字段
echo "📊 发现 $AVAILABLE_FILES 个可用的预处理数据文件"

# 根据resume标志设置checkpoint路径和兼容性参数
if [ $RESUME_FLAG -eq 1 ]; then
    LOAD_CHECKPOINT="$TRAINING_CHECKPOINT"
    EXTRA_CMD="$EXTRA_CMD"  # 恢复训练不需要特殊参数
    echo "🔄 恢复训练模式: 从训练checkpoint恢复"
else
    LOAD_CHECKPOINT="$PRESPARSE_CHECKPOINT"
    EXTRA_CMD="$EXTRA_CMD --no-load-optim --no-load-rng --finetune --enable-partial-load"  # 🔑 关键兼容性参数
    echo "🚀 初始训练模式: 从预稀疏checkpoint开始"
    echo "🔑 使用兼容性参数: --finetune --enable-partial-load"
fi

SAVE_CHECKPOINT="$TRAINING_CHECKPOINT"
mkdir -p "$SAVE_CHECKPOINT"

echo "🎯 Llama8b预稀疏模型训练配置:"
echo "  - 预稀疏checkpoint: $PRESPARSE_CHECKPOINT"
echo "  - 加载checkpoint: $LOAD_CHECKPOINT"  
echo "  - 保存checkpoint: $SAVE_CHECKPOINT"
echo "  - 数据路径: $DATA_BLEND"
echo "  - 稀疏率: ${SPARSITY} (50%)"
echo "  - 稀疏模式: ${PATTERN} (2:4结构化)"

# 训练参数 (与验证的llama2脚本保持一致)
TRAIN_ITERS=2000
SAVE_INTERVAL=100 # 更频繁保存以防数值问题
EVAL_INTERVAL=50 # 更频繁评估以监控稳定性
LOG_INTERVAL=2 # 更频繁日志以及时发现问题
WARMUP_ITERS=400 # 增加warmup以稳定初期训练

# 创建日志目录
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_${DATETIME}.log"

echo "📝 日志文件: $LOG_FILE"

# 🔧 采用与验证的llama2脚本相同的参数配置
TASK_CMD="--gumbel-scale-range 5e1 2.5e2 --gumbel-temperature-range 4 0.05 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 10 --weight-reg 2e-6"

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
    --lr 2e-5 \
    --min-lr 2e-6 \
    --lr-decay-style cosine \
    --log-interval $LOG_INTERVAL \
    --eval-iters 10 \
    --eval-interval $EVAL_INTERVAL \
    --data-path "$DATA_BLEND" \
    --data-cache-path CACHE \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-file $PROJECT_DIR/assets/checkpoints/Llama8b/vocab.txt \
    --vocab-size 119696 \
    --save-interval $SAVE_INTERVAL \
    --save $SAVE_CHECKPOINT \
    --load $LOAD_CHECKPOINT \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-num-zeros-in-grad \
    --lr-warmup-iters $WARMUP_ITERS \
    --exit-on-missing-checkpoint \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --use-flash-attn \
    --bf16 \
    --log-diff-mask \
    --exit-signal-handler \
    --exp-name llama8b-tp8-mask-only-c4 \
    $EXTRA_CMD $TASK_CMD "

cd $PROJECT_DIR

echo "🏃‍♂️ 开始预稀疏模型训练..."
echo "📊 训练配置 (采用验证的llama2配置):"
echo "  - 训练迭代: $TRAIN_ITERS"
echo "  - 保存间隔: $SAVE_INTERVAL"
echo "  - 评估间隔: $EVAL_INTERVAL"
echo "  - 批大小: 256 (global), 1 (micro)"
echo "  - 学习率: 2e-5 → 2e-6"
echo "  - 🔑 兼容性: --finetune --enable-partial-load"

# 启动训练 (使用正确的MaskLLM预训练器)
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py $options 2>&1 | tee "$LOG_FILE"

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
