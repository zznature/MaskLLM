#!/bin/bash

# 修复学习率checkpoint不匹配错误
# 错误: class input value 2e-05 and checkpoint value 5e-05 for learning rate do not match

echo "🔧 修复学习率checkpoint不匹配问题"
echo "=========================================="

PROJECT_DIR=$(pwd)
CHECKPOINT_DIR="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8"

echo "📊 错误分析:"
echo "  ❌ 当前脚本LR: 2e-05"
echo "  ❌ Checkpoint LR: 5e-05"
echo "  ❌ Megatron要求完全匹配"

echo ""
echo "🎯 解决方案选择:"
echo ""

# 检查checkpoint是否存在
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "📁 发现已有checkpoint: $CHECKPOINT_DIR"
    
    # 检查最新的checkpoint文件
    LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "iter_*" -type d | sort -V | tail -1)
    if [ -n "$LATEST_CHECKPOINT" ]; then
        ITER_NUM=$(basename "$LATEST_CHECKPOINT" | sed 's/iter_//')
        echo "📊 最新checkpoint: 第${ITER_NUM}步"
        echo ""
        
        echo "🔄 可选解决方案:"
        echo ""
        echo "1️⃣  【推荐】删除checkpoint，重新训练"
        echo "   - 清理不一致的状态"
        echo "   - 使用新的保守配置从头开始"
        echo "   - 命令: bash llama8b_scripts/fix_lr_checkpoint_mismatch.sh clean"
        echo ""
        echo "2️⃣  恢复到匹配的学习率继续训练"  
        echo "   - 使用5e-05学习率匹配checkpoint"
        echo "   - 从第${ITER_NUM}步继续，但可能仍有数值不稳定"
        echo "   - 命令: bash llama8b_scripts/fix_lr_checkpoint_mismatch.sh continue"
        echo ""
        echo "3️⃣  跳过优化器状态，只加载模型权重"
        echo "   - 保留训练进度但重置优化器状态"
        echo "   - 可以使用新学习率，但会有轻微性能损失"
        echo "   - 命令: bash llama8b_scripts/fix_lr_checkpoint_mismatch.sh skip_optim"
    else
        echo "⚠️  checkpoint目录存在但无有效iteration文件"
    fi
else
    echo "✅ 无现有checkpoint，可以正常开始训练"
fi

# 处理用户选择
ACTION=${1:-"help"}

case "$ACTION" in
    "clean")
        echo ""
        echo "🧹 执行方案1: 清理checkpoint重新开始"
        echo "----------------------------------------"
        
        if [ -d "$CHECKPOINT_DIR" ]; then
            echo "📁 备份现有checkpoint..."
            BACKUP_DIR="$PROJECT_DIR/output/checkpoints/llama8b_backup_$(date +%y%m%d_%H%M%S)"
            mv "$CHECKPOINT_DIR" "$BACKUP_DIR"
            echo "✅ Checkpoint已备份到: $BACKUP_DIR"
            echo ""
            echo "🚀 现在可以使用任何学习率配置重新开始训练："
            echo ""
            echo "🔧 保守配置 (推荐):"
            echo "bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_conservative.sh"
            echo ""
            echo "🔧 对齐Llama3配置:"
            echo "bash run_maskllm_native.sh llama8b_scripts/llama8b_aligned_with_llama3.sh"
        else
            echo "⚠️  Checkpoint目录不存在，无需清理"
        fi
        ;;
        
    "continue")
        echo ""
        echo "🔄 执行方案2: 匹配checkpoint学习率继续训练"
        echo "--------------------------------------------"
        
        # 创建匹配5e-05学习率的继续训练脚本
        cat > llama8b_scripts/continue_with_original_lr.sh << 'EOF'
#!/bin/bash

# 继续训练 - 匹配checkpoint的5e-05学习率
# 用于从第500步恢复训练

RESUME_FLAG=1  # 设置为恢复模式

echo "🔄 使用原始5e-05学习率恢复训练"
echo "从第500步继续..."

# 调用原始配置的训练脚本
bash llama8b_scripts/llama8b_presparse_training_tp8.sh $RESUME_FLAG
EOF
        
        chmod +x llama8b_scripts/continue_with_original_lr.sh
        echo "✅ 创建继续训练脚本: continue_with_original_lr.sh"
        echo ""
        echo "⚠️  注意: 这将使用可能不稳定的原始5e-05学习率"
        echo "🚀 执行命令:"
        echo "bash run_maskllm_native.sh llama8b_scripts/continue_with_original_lr.sh"
        ;;
        
    "skip_optim")
        echo ""
        echo "⚡ 执行方案3: 跳过优化器状态，使用新学习率"
        echo "---------------------------------------------"
        
        # 创建跳过优化器的训练脚本
        cat > llama8b_scripts/resume_skip_optimizer.sh << 'EOF'
#!/bin/bash

# 从模型checkpoint恢复但跳过优化器状态
# 这样可以使用新的学习率配置

DATETIME=$(date +%y-%m-%d_%H-%M-%S)
PROJECT_DIR=$(pwd)

echo "⚡ 跳过优化器状态恢复训练"
echo "使用新的保守学习率配置"

# 基础配置
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45536"
NNODES=1
NPROC_PER_NODE=8

# 路径配置
TRAINING_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"

# 数据配置
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

# 🔑 关键: 跳过优化器状态，只加载模型权重
EXTRA_CMD="--no-load-optim --no-load-rng"

# 保守学习率配置
LR=2e-5
MIN_LR=2e-6
TRAIN_ITERS=2000
SAVE_INTERVAL=200
EVAL_INTERVAL=50
LOG_INTERVAL=10
WARMUP_ITERS=400

# 保守稀疏参数
TASK_CMD="--gumbel-scale-range 1e1 1e2 --gumbel-temperature-range 2 0.1 --N 2 --M 4 --mask-only --prior-strength 1.0 --lr-mult 5 --weight-reg 1e-6"

# 日志配置  
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_resume_skip_optim"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_${DATETIME}.log"

echo ""
echo "🎯 配置总结:"
echo "  📂 从checkpoint加载: $TRAINING_CHECKPOINT" 
echo "  ⚡ 跳过优化器状态: --no-load-optim --no-load-rng"
echo "  📚 新学习率: $LR → $MIN_LR"
echo "  🔧 保守稀疏参数: 降低数值激进性"

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
--lr $LR \
--min-lr $MIN_LR \
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
--save $TRAINING_CHECKPOINT \
--load $TRAINING_CHECKPOINT \
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
--exp-name llama8b-tp8-resume-skip-optim \
$EXTRA_CMD $TASK_CMD"

cd $PROJECT_DIR

echo ""
echo "🚀 启动恢复训练 (跳过优化器状态)..."

export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py $options 2>&1 | tee "$LOG_FILE"

echo ""
echo "📝 训练日志: $LOG_FILE"
EOF

        chmod +x llama8b_scripts/resume_skip_optimizer.sh
        echo "✅ 创建跳过优化器脚本: resume_skip_optimizer.sh"
        echo ""
        echo "💡 优势: 保留模型训练进度，但可以使用新的保守学习率"
        echo "🚀 执行命令:"
        echo "bash run_maskllm_native.sh llama8b_scripts/resume_skip_optimizer.sh"
        ;;
        
    *)
        echo ""
        echo "❓ 使用方法:"
        echo "bash llama8b_scripts/fix_lr_checkpoint_mismatch.sh [clean|continue|skip_optim]"
        echo ""
        echo "📋 选项说明:"
        echo "  clean      - 删除checkpoint重新开始 (推荐)"
        echo "  continue   - 使用原始5e-05学习率继续"
        echo "  skip_optim - 跳过优化器状态使用新学习率"
        ;;
esac

echo ""
echo "🎯 推荐方案: clean (重新开始)"
echo "理由: 避免数值不稳定，使用经过优化的保守配置"
