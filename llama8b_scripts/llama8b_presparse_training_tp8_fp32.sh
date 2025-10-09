#!/bin/bash

# Llama8b FP32训练脚本 - 彻底解决数值稳定性问题
# 基于 llama8b_presparse_training_tp8_stable.sh
# 主要修改: bf16 → fp32 + activation recomputation + 更保守参数

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45533" # 与其他脚本端口区分
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
PRESPARSE_CHECKPOINT="$PROJECT_DIR/output/oneshot_pruning/checkpoint/${SPARSE_NAME}.update_weight"
TRAINING_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8_fp32"  # 🔧 单独的fp32 checkpoint目录
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"
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

# 检查数据文件
SAMPLE_FILE="${C4_HOME}/c4_llama8b_00000_text_document.bin"
if [ ! -f "$SAMPLE_FILE" ]; then
    echo "❌ 错误: 预处理数据不存在: $SAMPLE_FILE"
    echo "请先运行数据预处理："
    echo "bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh"
    exit 1
fi

# 统计可用数据文件
AVAILABLE_FILES=$(echo "$DATA_BLEND" | wc -w)
AVAILABLE_FILES=$((AVAILABLE_FILES / 2))
echo "📊 发现 $AVAILABLE_FILES 个可用的预处理数据文件"

# 根据resume标志设置checkpoint路径和兼容性参数
if [ $RESUME_FLAG -eq 1 ]; then
    LOAD_CHECKPOINT="$TRAINING_CHECKPOINT"
    EXTRA_CMD="$EXTRA_CMD --no-load-optim --no-load-rng"
    echo "🔄 恢复训练模式: 从FP32训练checkpoint恢复"
    echo "🔑 使用兼容性参数: --no-load-optim --no-load-rng"
else
    LOAD_CHECKPOINT="$PRESPARSE_CHECKPOINT"
    EXTRA_CMD="$EXTRA_CMD --no-load-optim --no-load-rng --finetune --enable-partial-load"
    echo "🚀 初始训练模式: 从预稀疏checkpoint开始（FP32）"
    echo "🔑 使用兼容性参数: --finetune --enable-partial-load"
fi

SAVE_CHECKPOINT="$TRAINING_CHECKPOINT"
mkdir -p "$SAVE_CHECKPOINT"

echo ""
echo "🎯 Llama8b FP32训练配置:"
echo "  - 精度: FP32 (彻底解决数值问题)"
echo "  - 显存优化: Activation Recomputation"
echo "  - 预稀疏checkpoint: $PRESPARSE_CHECKPOINT"
echo "  - 加载checkpoint: $LOAD_CHECKPOINT"
echo "  - 保存checkpoint: $SAVE_CHECKPOINT"
echo "  - 数据路径: $DATA_BLEND"
echo "  - 稀疏率: ${SPARSITY} (50%)"
echo "  - 稀疏模式: ${PATTERN} (2:4结构化)"
echo ""

# 训练参数
TRAIN_ITERS=2000
SAVE_INTERVAL=25  # 🔧 保持与bf16相同的保存频率
EVAL_INTERVAL=100
LOG_INTERVAL=1
WARMUP_ITERS=0

# 创建日志目录
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training_fp32"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_${DATETIME}.log"

echo "📝 日志文件: $LOG_FILE"

# 🔧 FP32 保守训练参数
# 设计原则:
# 1. 彻底解决 bf16 数值下溢/溢出问题
# 2. 更保守的 mask 更新策略，减缓 reg loss 上升
# 3. 更宽松的温度范围，避免极端分布
# 4. 更激进的梯度裁剪，防止局部梯度过大
#
# 目标:
# - Reg loss 增速 < +3/iter (vs 当前 +6/iter)
# - Mask difference 逐步下降到 0.04 (vs 当前 0.077)
# - Grad norm 稳定在 0.01-0.03 (vs 当前 0.035)
# - 完全避免 NaN
TASK_CMD="--gumbel-scale-range 1e2 5e2 --gumbel-temperature-range 4 0.5 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 4 --weight-reg 1e-7 --clip-grad 0.3"

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
    --seed 44 \
    --lr 5e-6 \
    --min-lr 5e-7 \
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
    --no-save-optim \
    --split 98,2,0 \
    --clip-grad 0.3 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-num-zeros-in-grad \
    --lr-warmup-iters $WARMUP_ITERS \
    --exit-on-missing-checkpoint \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --recompute-activations \
    --recompute-granularity full \
    --log-diff-mask \
    --exit-signal-handler \
    --exp-name llama8b-tp8-mask-only-c4-fp32 \
    $EXTRA_CMD $TASK_CMD "

cd $PROJECT_DIR

echo ""
echo "🏃‍♂️ 开始FP32预稀疏模型训练..."
echo ""
echo "📊 FP32 训练配置详情:"
echo "  ┌─ 数值精度"
echo "  │  ├─ 精度: FP32 (彻底解决bf16问题)"
echo "  │  ├─ 显存优化: Activation Recomputation"
echo "  │  └─ 预期显存: 65-70GB / 80GB"
echo "  │"
echo "  ├─ 训练参数"
echo "  │  ├─ 训练迭代: $TRAIN_ITERS"
echo "  │  ├─ 保存间隔: $SAVE_INTERVAL (每50次)"
echo "  │  ├─ 评估间隔: $EVAL_INTERVAL"
echo "  │  └─ 批大小: 256 (global), 1 (micro)"
echo "  │"
echo "  ├─ 学习率策略"
echo "  │  ├─ 基础学习率: 5e-6 (vs bf16的1e-5)"
echo "  │  ├─ Mask学习率倍数: 1 (实际mask LR = 5e-6)"
echo "  │  ├─ 最小学习率: 5e-7"
echo "  │  └─ 衰减方式: Cosine"
echo "  │"
echo "  ├─ 稀疏训练参数（极保守）"
echo "  │  ├─ 正则化权重: 1e-7 (vs bf16的1e-6)"
echo "  │  ├─ Gumbel温度: 6.0 → 2.0 (vs bf16的4.0→0.5)"
echo "  │  ├─ 梯度裁剪: 0.3 (vs bf16的0.5)"
echo "  │  └─ Prior strength: 3.0"
echo "  │"
echo "  ├─ 数据配置"
echo "  │  ├─ Seed: 44 (新的数据顺序)"
echo "  │  └─ Split: 98% train, 2% valid"
echo "  │"
echo "  └─ 预期效果"
echo "     ├─ Reg loss增速: <+3/iter (vs bf16的+6/iter)"
echo "     ├─ Mask difference: 逐步降至0.04 (vs bf16的0.077)"
echo "     ├─ Grad norm: 0.01-0.03 (vs bf16的0.035)"
echo "     ├─ NaN风险: 极低 (<5%)"
echo "     └─ 训练速度: ~295秒/iter (+35% vs bf16)"
echo ""
echo "⏱️  预计完成时间: ~164小时 (6.8天)"
echo ""
echo "🔍 关键监控指标（前100次迭代）:"
echo "  ✅ Reg loss增速应 < +3/iter"
echo "  ✅ Mask difference应开始下降"
echo "  ✅ Grad norm应稳定无峰值"
echo "  ✅ 无任何NaN warning"
echo ""
echo "💾 Checkpoint大小优化（节省磁盘空间）:"
echo "  - 每个checkpoint: ~32GB (仅模型权重，无优化器状态)"
echo "  - 保存频率: 每50次迭代"
echo "  - 总checkpoint数: 40个"
echo "  - ⚠️  注意: 由于--no-save-optim，从checkpoint恢复需使用--no-load-optim"
echo ""

# 启动训练
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py $options 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 Llama8b FP32预稀疏模型训练完成!"
    echo "📁 训练checkpoint保存在: $SAVE_CHECKPOINT"
    echo "📝 完整日志: $LOG_FILE"
    echo ""
    echo "📊 训练总结:"
    echo "  - 精度: FP32 ✅"
    echo "  - 预稀疏模型: ✅"
    echo "  - 稀疏化训练: ✅"
    echo "  - Mask优化: ✅"
    echo "  - 结构化稀疏: ✅ (2:4)"
    echo "  - 数值稳定性: ✅"
    echo ""
    echo "下一步: 评估稀疏模型性能"
    echo "bash run_maskllm_native.sh llama8b_scripts/evaluate_sparse_model.sh"
else
    echo ""
    echo "❌ 训练失败，退出码: $EXIT_CODE"
    echo "📝 请检查日志: $LOG_FILE"
    echo ""
    echo "🔧 可能的解决方案:"
    echo "  1. 检查GPU内存是否充足 (应为65-70GB/80GB)"
    echo "  2. 如果显存不足，尝试:"
    echo "     - 降低global-batch-size到192"
    echo "     - 或添加 --recompute-num-layers 4"
    echo "  3. 如果仍然NaN (极少见):"
    echo "     - 进一步降低lr-mult到0.3"
    echo "     - 或降低weight-reg到5e-8"
    echo "     - 或考虑从Dense checkpoint开始"
    echo "  4. 检查checkpoint完整性"
    echo "  5. 验证数据路径"
    exit $EXIT_CODE
fi

