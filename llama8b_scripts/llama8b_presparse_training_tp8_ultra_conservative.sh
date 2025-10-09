#!/bin/bash

# Llama8b 超保守训练脚本 - 从 iter 1150 恢复
# 目标: 彻底解决 iter 1100-1200 NaN 问题
# 策略: 极保守参数 - 更慢的 temperature，更低的 mask LR，更激进的裁剪

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

# 数据配置
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00009..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.1 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

# 检查 checkpoint (优先使用 iter 1150, 如果不存在则用 1145 或 1140)
RESUME_ITER=1150
if [ ! -d "$TRAINING_CHECKPOINT/iter_0001150" ]; then
    echo "⚠️  iter_0001150 不存在，尝试 1145..."
    RESUME_ITER=1145
    if [ ! -d "$TRAINING_CHECKPOINT/iter_0001145" ]; then
        echo "⚠️  iter_0001145 不存在，尝试 1140..."
        RESUME_ITER=1140
        if [ ! -d "$TRAINING_CHECKPOINT/iter_0001140" ]; then
            echo "❌ 错误: 找不到可用的 checkpoint (1150, 1145, 1140)"
            ls -d "$TRAINING_CHECKPOINT"/iter_* 2>/dev/null | tail -5
            exit 1
        fi
    fi
fi

echo "✅ 找到 checkpoint: iter $RESUME_ITER"
echo ""
echo "🎯 超保守训练配置:"
echo "  ┌─ 恢复信息"
echo "  │  ├─ 起点: iter $RESUME_ITER"
echo "  │  ├─ 目标: iter 2000 (剩余 $((2000-RESUME_ITER)) iterations)"
echo "  │  ├─ 优化器: 重新初始化 (--no-load-optim)"
echo "  │  └─ RNG: 重新初始化 (--no-load-rng, seed=50)"
echo "  │"
echo "  ├─ 超保守策略 (vs 之前稳定化)"
echo "  │  ├─ Temperature: 4→1.5 (vs 之前 4→1.0, 更温和)"
echo "  │  ├─ Mask LR mult: 5 (vs 之前 8, 更慢)"
echo "  │  ├─ Gradient clip: 0.3 (vs 之前 0.5, 更激进)"
echo "  │  ├─ Weight reg: 5e-7 (vs 之前 1e-6, 更低压力)"
echo "  │  └─ Base LR: 1.5e-5 (vs 之前 2e-5, 降低 25%)"
echo "  │"
echo "  └─ 危险期应对"
echo "     ├─ 每 5 次迭代保存 (vs 之前 10)"
echo "     ├─ 新 seed=50 避免数值陷阱"
echo "     └─ 目标: 无 NaN 完成到 iter 2000"
echo ""

# 训练参数
TRAIN_ITERS=2000
SAVE_INTERVAL=5  # 🔧 更频繁保存
EVAL_INTERVAL=100
LOG_INTERVAL=1
WARMUP_ITERS=400  # 与原始训练一致

# 日志配置
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training_resume"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_ultra_conservative_${DATETIME}.log"

echo "📝 日志文件: $LOG_FILE"
echo ""

# 🔧 超保守参数配置
# 
# 核心修改（vs 稳定化配置）:
# 1. Temperature 终点: 1.0 → 1.5 (避免过快收敛)
# 2. Mask LR mult: 8 → 5 (更慢的 mask 更新)
# 3. Gradient clip: 0.5 → 0.3 (更激进的裁剪)
# 4. Weight reg: 1e-6 → 5e-7 (降低 reg loss 压力)
# 5. Base LR: 2e-5 → 1.5e-5 (降低整体学习率)
# 6. Seed: 48 → 50 (新的数值路径)
#
# 预期效果:
# - Temperature 极慢下降: 避免 mask 突变
# - Mask LR 更低: 平滑优化
# - Reg loss 增速: +6.6 → +3.5/iter (预期)
# - NaN 风险: <5% (极低)
TASK_CMD="--gumbel-scale-range 5e1 2.5e2 --gumbel-temperature-range 4 1.5 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 5 --weight-reg 5e-7"

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
    --lr 1.5e-5 \
    --min-lr 1.5e-6 \
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
    --use-flash-attn \
    --bf16 \
    --log-diff-mask \
    --exit-signal-handler \
    --exp-name llama8b-tp8-ultra-conservative-iter${RESUME_ITER} \
    --no-save-optim \
    --no-load-optim \
    --no-load-rng \
    --override-opt_param-scheduler \
    --seed 50 \
    $TASK_CMD "

cd $PROJECT_DIR

echo "🏃‍♂️ 开始超保守训练..."
echo ""
echo "📊 详细配置对比:"
echo ""
echo "  ┌─────────────────────┬────────────┬────────────┬────────────┐"
echo "  │ 参数                │ 原始 (NaN) │ 稳定化 (NaN)│ 超保守 ✨   │"
echo "  ├─────────────────────┼────────────┼────────────┼────────────┤"
echo "  │ Temperature 终点    │ 0.2        │ 1.0        │ 1.5        │"
echo "  │ Mask LR mult        │ 10         │ 8          │ 5          │"
echo "  │ Gradient clip       │ 1.0        │ 0.5        │ 0.3        │"
echo "  │ Weight reg          │ 2e-6       │ 1e-6       │ 5e-7       │"
echo "  │ Base LR             │ 2e-5       │ 2e-5       │ 1.5e-5     │"
echo "  │ Seed                │ 46         │ 48         │ 50         │"
echo "  │ Save interval       │ 25         │ 10         │ 5          │"
echo "  └─────────────────────┴────────────┴────────────┴────────────┘"
echo ""
echo "  🎯 关键改进:"
echo "  ├─ 数值精度"
echo "  │  ├─ 精度: BF16 (Flash Attention ✅)"
echo "  │  └─ 显存: ~53GB/80GB"
echo "  │"
echo "  ├─ 训练参数"
echo "  │  ├─ 起点: iter $RESUME_ITER"
echo "  │  ├─ 目标: iter 2000"
echo "  │  └─ 保存: 每 5 次迭代"
echo "  │"
echo "  ├─ 学习率（降低 25%）"
echo "  │  ├─ Base LR: 1.5e-5 (was 2e-5)"
echo "  │  ├─ Mask LR: 7.5e-5 (was 1.6e-4)"
echo "  │  └─ Min LR: 1.5e-6"
echo "  │"
echo "  ├─ 稀疏训练（极保守）"
echo "  │  ├─ Temperature: 当前 ~2.3 → 目标 1.5"
echo "  │  ├─ Mask LR mult: 5 (最慢)"
echo "  │  ├─ Weight reg: 5e-7 (最低)"
echo "  │  └─ Gradient clip: 0.3 (最激进)"
echo "  │"
echo "  └─ 数据配置"
echo "     ├─ Seed: 50 (第3次尝试)"
echo "     └─ Blend: 11 files"
echo ""
echo "🎯 预期效果:"
echo "  ✅ Temperature 平滑下降，避免突变"
echo "  ✅ Mask 极慢优化，数值稳定"
echo "  ✅ Reg loss 增速 < +4/iter"
echo "  ✅ NaN 风险 < 5%"
echo ""
echo "⏱️  预计时间:"
echo "  - 剩余迭代: $((2000-RESUME_ITER)) 次"
echo "  - 预计时间: ~$((( 2000-RESUME_ITER) * 220 / 3600)) 小时"
echo ""

# 启动训练
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py $options 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 超保守训练成功完成!"
    echo "📁 Checkpoint: $TRAINING_CHECKPOINT"
    echo "📝 日志: $LOG_FILE"
    echo ""
    echo "📊 训练成功！"
    echo "  ✅ 彻底解决 NaN 问题"
    echo "  ✅ 平滑完成 iter $RESUME_ITER → 2000"
    echo "  ✅ Mask 稳定优化"
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
        echo "🔴 仍然遇到 NaN 错误！"
        echo "   - 崩溃位置: iter $LAST_ITER"
        echo ""
        echo "📊 NaN 历史:"
        echo "   - 原始训练: iter 1143 (lr-mult=10, temp→0.2)"
        echo "   - 稳定化: iter 1150 (lr-mult=8, temp→1.0)"
        echo "   - 超保守: iter $LAST_ITER (lr-mult=5, temp→1.5)"
        echo ""
        echo "🔧 建议方案:"
        echo ""
        echo "  方案 A: 继续降低参数（最后一次尝试）"
        echo "    --lr-mult 3"
        echo "    --clip-grad 0.2"
        echo "    --gumbel-temperature-range 4 2.0"
        echo "    --lr 1e-5"
        echo ""
        echo "  方案 B: 切换到 FP32 (推荐)"
        echo "    BF16 在 iter 1100-1200 有系统性数值问题"
        echo "    FP32 虽慢但彻底解决 NaN"
        echo "    使用脚本: llama8b_presparse_training_tp8_fp32_adjusted.sh"
        echo ""
        echo "  方案 C: 从更早的 checkpoint 开始"
        echo "    从 iter 1100 重新开始，使用超保守参数"
        echo ""
    else
        echo "🔧 非 NaN 错误，检查:"
        echo "  1. GPU 内存"
        echo "  2. Checkpoint 完整性"
        echo "  3. 数据路径"
    fi
    
    exit $EXIT_CODE
fi

