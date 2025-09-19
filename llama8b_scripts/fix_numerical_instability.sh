#!/bin/bash

# 修复Llama8b训练的数值不稳定问题
# 基于日志分析的梯度爆炸和正则化损失异常

echo "🔧 修复Llama8b训练数值不稳定问题"
echo "=========================================="

echo "📊 问题诊断结果:"
echo "  ❌ reg loss异常高: 4.761145E+03"
echo "  ❌ scale_multiplier过大: 269.800"
echo "  ❌ 在第850步出现NaN梯度"
echo "  ✅ 兼容性已解决: 成功运行到第850步"

echo ""
echo "🎯 解决方案: 保守化超参数配置"
echo ""

# 创建保守的训练配置
cat > llama8b_scripts/llama8b_presparse_training_conservative.sh << 'EOF'
#!/bin/bash

# Llama8b预稀疏模型训练 - 数值稳定版本
# 基于错误分析的保守化配置

DATETIME=$(date +%y-%m-%d_%H-%M-%S)
PROJECT_DIR=$(pwd)
RESUME_FLAG=${1:-0}

# 基础配置
MASTER_ADDR="127.0.0.1"
MASTER_PORT="45534"  # 避免端口冲突
NNODES=1
NPROC_PER_NODE=8
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# 检查点配置
SPARSE_NAME="llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0"
PRESPARSE_CHECKPOINT="$PROJECT_DIR/output/oneshot_pruning/checkpoint/${SPARSE_NAME}"
TRAINING_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_conservative_tp8"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"

# 数据配置 (与验证脚本保持一致)
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

# 检查prerequisite
echo "🔍 检查前置条件..."
if [ ! -d "$PRESPARSE_CHECKPOINT" ]; then
    echo "❌ 预稀疏checkpoint不存在: $PRESPARSE_CHECKPOINT"
    exit 1
fi

SAMPLE_FILE="${C4_HOME}/c4_llama8b_00000_text_document.bin"
if [ ! -f "$SAMPLE_FILE" ]; then
    echo "❌ 预处理数据不存在: $SAMPLE_FILE"
    exit 1
fi

echo "✅ 前置条件检查通过"

# 根据resume设置兼容性参数
if [ $RESUME_FLAG -eq 1 ]; then
    LOAD_CHECKPOINT="$TRAINING_CHECKPOINT"
    EXTRA_CMD=""
    echo "🔄 恢复训练模式"
else
    LOAD_CHECKPOINT="$PRESPARSE_CHECKPOINT"
    EXTRA_CMD="--no-load-optim --no-load-rng --finetune --enable-partial-load"
    echo "🚀 初始训练模式 (使用兼容性参数)"
fi

SAVE_CHECKPOINT="$TRAINING_CHECKPOINT"
mkdir -p "$SAVE_CHECKPOINT"

# 🔧 保守化训练参数配置
TRAIN_ITERS=2000
SAVE_INTERVAL=200    # 更频繁保存以防数值问题
EVAL_INTERVAL=50     # 更频繁评估以监控稳定性
LOG_INTERVAL=5       # 更频繁日志以及时发现问题
WARMUP_ITERS=50      # 增加warmup以稳定初期训练

# 🔧 保守化稀疏参数 (减少数值激进性)
TASK_CMD="--gumbel-scale-range 1e1 1e2 --gumbel-temperature-range 2 0.1 --N 2 --M 4 --mask-only --prior-strength 1.0 --lr-mult 5 --weight-reg 1e-6"

# 日志配置
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_conservative_training"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_${DATETIME}.log"

echo ""
echo "🎯 保守化配置:"
echo "  - 学习率: 1e-5 (降低10x)"
echo "  - Weight正则: 1e-6 (降低10x)"  
echo "  - Gumbel缩放: 1e1→1e2 (降低10x)"
echo "  - LR倍数: 5 (降低50%)"
echo "  - Prior强度: 1.0 (降低66%)"
echo "  - Warmup: 50步 (增加稳定性)"
echo ""

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
    --global-batch-size 128 \
    --train-iters $TRAIN_ITERS \
    --lr 1e-5 \
    --min-lr 1e-7 \
    --lr-decay-style cosine \
    --log-interval $LOG_INTERVAL \
    --eval-iters 5 \
    --eval-interval $EVAL_INTERVAL \
    --data-path \"$DATA_BLEND\" \
    --data-cache-path CACHE \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-file $PROJECT_DIR/assets/checkpoints/Llama8b/vocab.txt \
    --vocab-size 119696 \
    --save-interval $SAVE_INTERVAL \
    --save $SAVE_CHECKPOINT \
    --load $LOAD_CHECKPOINT \
    --split 98,2,0 \
    --clip-grad 0.5 \
    --weight-decay 0.05 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.01 \
    --log-num-zeros-in-grad \
    --lr-warmup-iters $WARMUP_ITERS \
    --exit-on-missing-checkpoint \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --use-flash-attn \
    --bf16 \
    --log-diff-mask \
    --exit-signal-handler \
    --exp-name llama8b-tp8-mask-conservative \
    $EXTRA_CMD $TASK_CMD "

cd $PROJECT_DIR

echo "🏃‍♂️ 启动保守化训练..."
echo "📊 关键改进:"
echo "  - 降低学习率避免梯度爆炸"
echo "  - 减少正则化强度"
echo "  - 温和的Gumbel参数"
echo "  - 更频繁的检查点保存"
echo "  - 增强的数值监控"

# 设置CUDA环境以提高数值稳定性
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo ""
echo "🚀 启动训练..."
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py $options 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成!"
    echo "📊 最终checkpoint: $SAVE_CHECKPOINT"
else
    echo "❌ 训练失败 (Exit Code: $EXIT_CODE)"
    echo "📝 检查日志: $LOG_FILE"
fi

echo "📝 训练日志: $LOG_FILE"
EOF

chmod +x llama8b_scripts/llama8b_presparse_training_conservative.sh

echo "✅ 创建保守化训练脚本: llama8b_presparse_training_conservative.sh"

echo ""
echo "🔧 关键改进对比:"
echo ""
printf "| 参数 | 原配置 | 保守配置 | 改进说明 |\n"
printf "|------|--------|----------|----------|\n"
printf "| **学习率** | 5e-5 | 1e-5 | 降低5x，避免梯度爆炸 |\n"
printf "| **Weight正则** | 1e-5 | 1e-6 | 降低10x，减少reg loss |\n"
printf "| **Gumbel缩放** | 1e2-5e2 | 1e1-1e2 | 降低10x，数值更稳定 |\n"
printf "| **LR倍数** | 10 | 5 | 降低50%%，mask学习更温和 |\n"
printf "| **Prior强度** | 3.0 | 1.0 | 降低66%%，稀疏约束更温和 |\n"
printf "| **批大小** | 256 | 128 | 降低50%%，内存更安全 |\n"
printf "| **梯度裁剪** | 1.0 | 0.5 | 更严格的梯度控制 |\n"
printf "| **权重衰减** | 0.1 | 0.05 | 降低50%%，减少正则化压力 |\n"

echo ""
echo "🧪 测试命令:"
echo "bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_conservative.sh"

echo ""
echo "📈 预期改进:"
echo "  ✅ reg loss控制在合理范围"
echo "  ✅ 梯度范数稳定不爆炸"
echo "  ✅ temperature退火更平滑"
echo "  ✅ 训练过程数值稳定"
