#!/bin/bash

# Llama8b 稳定化训练脚本 - 从 iter 1125 恢复
# 目标: 平滑渡过 iter 1100-1200 NaN 危险期
# 策略: 放慢 temperature 下降，降低 mask LR，增强梯度裁剪

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45532"
NNODES=1
NPROC_PER_NODE=8
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE))

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

PROJECT_DIR=$(pwd)
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

# 路径配置
TRAINING_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"

# 🔧 使用与之前相同的数据配置（保证一致性）
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00009..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.1 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

# 检查 checkpoint
if [ ! -d "$TRAINING_CHECKPOINT/iter_0001125" ]; then
    echo "❌ 错误: iter_0001125 checkpoint 不存在"
    echo "可用的 checkpoint:"
    ls -d "$TRAINING_CHECKPOINT"/iter_* 2>/dev/null | tail -5
    exit 1
fi

echo "✅ 找到 iter 1125 checkpoint"
echo ""
echo "🎯 BF16 稳定化训练配置:"
echo "  ┌─ 恢复信息"
echo "  │  ├─ 起点: iter 1125 (temp=1.864, mask_diff=0.112)"
echo "  │  ├─ 目标: iter 2000 (剩余 875 iterations)"
echo "  │  ├─ 优化器: 重新初始化 (--no-load-optim)"
echo "  │  └─ RNG: 重新初始化 (--no-load-rng, seed=48)"
echo "  │"
echo "  ├─ 稳定化策略"
echo "  │  ├─ Temperature: 4→1.0 (vs 之前 4→0.2, 更温和)"
echo "  │  ├─ Mask LR mult: 8 (vs 之前 10, 更慢)"
echo "  │  ├─ Gradient clip: 0.5 (vs 之前 1.0, 更激进)"
echo "  │  └─ Weight reg: 1e-6 (vs 之前 2e-6, 降低压力)"
echo "  │"
echo "  └─ 危险期应对 (iter 1125-1200)"
echo "     ├─ 每 10 次迭代保存 checkpoint"
echo "     ├─ 密切监控 grad norm (<0.10)"
echo "     ├─ 新 seed 避免相同数值陷阱"
echo "     └─ 预计通过危险期时间: ~5小时"
echo ""

# 训练参数
TRAIN_ITERS=2000
SAVE_INTERVAL=10  # 🔧 更频繁保存以快速定位问题
EVAL_INTERVAL=100
LOG_INTERVAL=1
WARMUP_ITERS=400  # 🔑 必须与原始训练一致，否则 scheduler 检查会失败

# 日志配置
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training_resume"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_stable_resume_${DATETIME}.log"

echo "📝 日志文件: $LOG_FILE"
echo ""

# 🔧 稳定化参数配置
# 
# 核心修改（vs 之前配置）:
# 1. Temperature 终点: 0.2 → 1.0 (避免过度确定性)
# 2. Mask LR mult: 10 → 8 (稍慢一点)
# 3. Gradient clip: 1.0 → 0.5 (更激进裁剪)
# 4. Weight reg: 2e-6 → 1e-6 (降低 reg loss 压力)
# 5. Seed: 46 → 48 (避免相同数值路径)
#
# 预期效果:
# - Temperature 更平滑下降: 1.864 → 1.0 (vs 1.864 → 0.2)
# - Mask 更稳定收敛: grad norm 峰值 < 0.10
# - Reg loss 增速: +6.6 → +4.5/iter
# - NaN 风险: 30% → <10%
TASK_CMD="--gumbel-scale-range 5e1 2.5e2 --gumbel-temperature-range 4 1.0 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 8 --weight-reg 1e-6"

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
    --save $TRAINING_CHECKPOINT \
    --load $TRAINING_CHECKPOINT \
    --split 98,2,0 \
    --clip-grad 0.5 \
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
    --exp-name llama8b-tp8-stable-resume-iter1125 \
    --no-save-optim \
    --no-load-optim \
    --no-load-rng \
    --override-opt_param-scheduler \
    --seed 48 \
    $TASK_CMD "

cd $PROJECT_DIR

echo "🏃‍♂️ 开始稳定化训练..."
echo ""
echo "📊 详细配置:"
echo "  ┌─ 数值精度"
echo "  │  ├─ 精度: BF16 (最佳性能)"
echo "  │  ├─ Flash Attention: ✅"
echo "  │  └─ 显存: ~53GB/80GB"
echo "  │"
echo "  ├─ 训练参数"
echo "  │  ├─ 起点: iter 1125"
echo "  │  ├─ 目标: iter 2000 (剩余 875)"
echo "  │  ├─ 保存间隔: 每 10 次"
echo "  │  └─ 批大小: 256"
echo "  │"
echo "  ├─ 学习率"
echo "  │  ├─ Base LR: 2e-5"
echo "  │  ├─ Mask LR mult: 8 (实际 mask LR = 1.6e-4)"
echo "  │  └─ Min LR: 2e-6"
echo "  │"
echo "  ├─ 稀疏训练（稳定化）"
echo "  │  ├─ Temperature: 当前 1.864 → 目标 1.0"
echo "  │  ├─ Weight reg: 1e-6 (vs 之前 2e-6)"
echo "  │  ├─ Gradient clip: 0.5 (vs 之前 1.0)"
echo "  │  └─ Prior strength: 3.0"
echo "  │"
echo "  └─ 数据配置"
echo "     ├─ Seed: 48 (新, vs 之前 46)"
echo "     ├─ Blend: 11 files (00009-00019)"
echo "     └─ Split: 98% train, 2% valid"
echo ""
echo "🎯 危险期监控 (iter 1125-1200):"
echo "  ✅ Grad norm 应保持 < 0.10"
echo "  ✅ Reg loss 增速应 < +5/iter"
echo "  ✅ Mask difference 应继续下降"
echo "  ✅ 无任何 NaN warning"
echo ""
echo "⏱️  预计时间:"
echo "  - 危险期 (iter 1125-1200): ~4.5小时"
echo "  - 完整训练 (iter 1125-2000): ~53.5小时 (2.2天)"
echo ""

# 启动训练
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py $options 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 Llama8b 稳定化训练完成!"
    echo "📁 Checkpoint: $TRAINING_CHECKPOINT"
    echo "📝 日志: $LOG_FILE"
    echo ""
    echo "📊 训练成功！关键指标:"
    echo "  ✅ 平滑渡过 iter 1100-1200 危险期"
    echo "  ✅ Mask 持续优化"
    echo "  ✅ Reg loss 控制在可接受范围"
    echo "  ✅ 无 NaN 错误"
    echo ""
    echo "🔍 建议检查:"
    echo "  1. Mask difference 最终值（期待 0.07-0.08）"
    echo "  2. Reg loss 最终值（期待 <13000）"
    echo "  3. Temperature 最终值（期待 ~1.0）"
    echo ""
    echo "下一步: 评估稀疏模型性能"
    echo "bash run_maskllm_native.sh llama8b_scripts/evaluate_sparse_model.sh"
else
    echo ""
    echo "❌ 训练失败，退出码: $EXIT_CODE"
    echo "📝 日志: $LOG_FILE"
    echo ""
    
    # 检查是否是 NaN 错误
    if grep -q "NaN" "$LOG_FILE"; then
        LAST_ITER=$(grep "iteration.*consumed samples" "$LOG_FILE" | tail -1 | awk '{print $2}' | cut -d/ -f1)
        echo "🔴 检测到 NaN 错误！"
        echo "   - 崩溃位置: iter $LAST_ITER"
        echo ""
        
        if [ $LAST_ITER -lt 1200 ]; then
            echo "🔧 NaN 仍在危险期 (iter 1125-1200)，建议方案:"
            echo "  选项 1 (推荐): 进一步放慢参数"
            echo "    --gumbel-temperature-range 4 1.5"
            echo "    --lr-mult 5"
            echo "    --clip-grad 0.3"
            echo ""
            echo "  选项 2 (激进): 切换到 FP32"
            echo "    使用 llama8b_presparse_training_tp8_fp32_adjusted.sh"
        else
            echo "✅ 已成功渡过危险期！"
            echo "   - 从最近的 checkpoint 继续即可"
        fi
    else
        echo "🔧 可能的解决方案:"
        echo "  1. 检查 GPU 内存"
        echo "  2. 验证 checkpoint 完整性"
        echo "  3. 检查数据路径"
    fi
    
    exit $EXIT_CODE
fi

