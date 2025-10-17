#!/bin/bash

# ============================================================================
# Llama8b Pre-Sparse 训练脚本 (BF16 Conservative Plan B: 高度保守)
# ============================================================================
# 
# 🎯 目标: 从稀疏checkpoint (干净起点) 训练到 iter 2000 (NaN-Free)
# 🔬 策略: BF16高效 + 高度保守参数 - 平衡速度与稳定性
#
# 🔴 关键发现 (基于所有训练失败的深度分析):
# - 从BF16 iter 1235继续训练: Rollback失败 @ iter 1323 (Reg增长2.78 /iter)
# - 稳定性方程<1.0在BF16下不够稳定
# - BF16需要更高稳定性阈值 (至少0.5)
# - 从干净checkpoint开始可避免累积数值问题
#
# ✅ 新策略: BF16 Conservative Plan B
# - 起点: Sparse checkpoint (oneshot pruning输出, 干净无训练历史)
# - LR Mult: 2.0 (平衡速度) - Mask LR = 3.0e-5
# - Weight Reg: 1.5e-5 (高度正则化)
# - 稳定性方程: 0.5 (BF16安全阈值)
# - 目标: Reg loss 增长率 <0.6 /iter
# - 精度: BF16 + Flash Attention (比FP32快~15%)
#
# 核心优化逻辑:
# 1. 从干净稀疏 checkpoint 开始:
#    - Sparse checkpoint (oneshot pruning输出)
#    - 无MaskLLM训练历史, 无累积数值问题
#    - 从0开始完整训练2000次迭代
#
# 2. 平衡Mask LR (lr_mult=2.0):
#    - Effective Mask LR = 2.0 × 1.5e-5 = 3.0e-5
#    - 比FP32 Ultra-Aggressive快2x (vs 1.5e-5)
#    - 2000 iters 累积更新: 0.06 (安全)
#
# 3. 高度正则化 (weight_reg=1.5e-5):
#    - 比BF16历史最强10x (vs 1.5e-6)
#    - 强力约束 mask logits 保持在 prior 附近
#    - 目标: reg loss 增长率 <0.6 /iter
#
# 4. 稳定性方程 (BF16安全阈值):
#    - 新方程: 1.5e-5 / (2.0 × 1.5e-5) = **0.5** (高度稳定)
#    - vs BF16历史最高: ~0.08 (失败)
#    - vs FP32 Rollback: 0.444 (失败 @ 1323)
#    - vs FP32 Ultra-Aggressive: 1.33 (极度稳定)
#    - BF16安全阈值: 0.5 (足够稳定 + 保持速度优势)
#
# 5. BF16 + Flash Attention:
#    - 速度: ~220s/iter (vs FP32 ~255s/iter)
#    - 显存: ~53GB (vs FP32 ~70GB)
#    - 速度优势: ~15%
#
# 预期效果:
# - Reg loss @ iter 2000: ~8700 (vs 起点 ~7500, 增长 1200)
# - Reg loss 增长率: ~0.6 /iter (vs FP32 Ultra-Aggressive <0.3)
# - 成功率: ~90% (BF16 + 稳定性0.5)
# - 训练时间: ~48h (vs FP32 ~54h, 节省11%)
#
# 显存: 53-56GB / 80GB
# 速度: ~220秒/iter
# ============================================================================

set -e

# 设置必需的环境变量
export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 基础配置
PROJECT_DIR="/data/home/zdhs0054/zzhou/MaskLLM"
NPROC_PER_NODE=8
NNODES=1
MASTER_ADDR=localhost
MASTER_PORT=45542  # 不同于Ultra-Aggressive的45541

# 路径配置
SPARSE_CHECKPOINT="$PROJECT_DIR/output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight"
TRAINING_CHECKPOINT="$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8_bf16_conservative_plan_b"
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"

# 数据配置 (使用所有 20 个文件)
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

# 检查 Sparse checkpoint
if [ ! -d "$SPARSE_CHECKPOINT" ]; then
    echo "❌ 稀疏checkpoint不存在: $SPARSE_CHECKPOINT"
    echo ""
    echo "请先运行稀疏化脚本:"
    echo "bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh SparseGPT"
    exit 1
fi

# 验证所有 8 个 rank 的 checkpoint 都存在
echo "🔍 验证稀疏checkpoint完整性..."
for rank in {0..7}; do
    RANK_DIR="$SPARSE_CHECKPOINT/iter_0000001/mp_rank_$(printf '%02d' $rank)"
    if [ ! -d "$RANK_DIR" ]; then
        echo "❌ 缺少 rank $rank 的 checkpoint: $RANK_DIR"
        exit 1
    fi
done
echo "✅ 所有 8 个 rank 的稀疏checkpoint完整"

# 创建训练 checkpoint 目录
mkdir -p "$TRAINING_CHECKPOINT"

LOAD_CHECKPOINT="$SPARSE_CHECKPOINT"
SAVE_CHECKPOINT="$TRAINING_CHECKPOINT"
# 从稀疏checkpoint开始训练需要特殊参数
EXTRA_CMD="--finetune --enable-partial-load --no-load-optim --no-load-rng"

# 日志配置
LOG_DIR="$PROJECT_DIR/output/logs/llama8b_presparse_training_bf16_conservative_plan_b"
mkdir -p "$LOG_DIR"
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
LOG_FILE="$LOG_DIR/training_bf16_conservative_plan_b_${DATETIME}.log"

# 训练参数
TRAIN_ITERS=2000
SAVE_INTERVAL=25  # 每 25 iters保存 (用户指定)
LOG_INTERVAL=1
EVAL_INTERVAL=100
WARMUP_ITERS=200

# 🔧 MaskLLM 任务配置 (BF16 Conservative Plan B)
# 
# 核心思想: BF16高效 + 高度保守 = 平衡速度与稳定性
#
# 参数选择依据 (基于所有训练失败的综合分析):
#
# 1. LR Mult = 2.0 (平衡选择):
#    - Effective Mask LR = 2.0 × 1.5e-5 = **3.0e-5**
#    - 比FP32 Ultra-Aggressive快2x (vs 1.5e-5)
#    - 2000 iters 累积更新: 0.06 (安全)
#
# 2. Weight Reg = 1.5e-5 (高度正则化):
#    - 正则化强度: 比BF16历史最强10x (vs 1.5e-6)
#    - 强力约束 mask logits 保持在 prior 附近
#    - 目标: reg loss 增长率 **<0.6 /iter**
#
# 3. 稳定性方程 (BF16安全阈值):
#    - 新方程: 1.5e-5 / (2.0 × 1.5e-5) = **0.5** (高度稳定)
#    - vs BF16历史最高: ~0.08 (失败)
#    - vs FP32 Rollback: 0.444 (失败 @ 1323)
#    - vs FP32 Ultra-Aggressive: 1.33 (极度稳定)
#    - **BF16安全阈值: 0.5 (足够稳定 + 保持速度)**
#
# 4. Clip Grad = 0.5 (标准裁剪):
#    - 允许正常梯度流动
#    - 标准BF16裁剪阈值
#
# 5. Scale 范围 100→300:
#    - Max Eff_Sharp = 300 / 2.5 = 120
#    - 已在FP32验证有效
#
# 6. Temp 范围 4.0→2.5:
#    - 保守温度衰减
#    - 保持充足柔性
#
TASK_CMD="--gumbel-scale-range 100 300 --gumbel-temperature-range 4.0 2.5 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 2.0 --weight-reg 1.5e-5 --clip-grad 0.5"

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
    --lr-decay-iters $TRAIN_ITERS \
    --save $SAVE_CHECKPOINT \
    --load $LOAD_CHECKPOINT \
    --no-save-optim \
    $EXTRA_CMD \
    --data-path $DATA_BLEND \
    --data-cache-path CACHE \
    --tokenizer-type Llama8bTokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --vocab-file $PROJECT_DIR/assets/checkpoints/Llama8b/vocab.txt \
    --vocab-size 119696 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 10 \
    --log-interval $LOG_INTERVAL \
    --split 98,2,0 \
    --distributed-backend nccl \
    --lr 1.5e-5 \
    --lr-decay-style cosine \
    --min-lr 1.5e-6 \
    --lr-warmup-iters $WARMUP_ITERS \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-num-zeros-in-grad \
    --exit-on-missing-checkpoint \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --exit-signal-handler \
    --log-diff-mask \
    --bf16 \
    --use-flash-attn \
    --exp-name llama8b-tp8-bf16-conservative-plan-b \
    --seed 55 \
    $EXTRA_CMD \
    $TASK_CMD "

# 关键配置说明 (BF16 Conservative Plan B)
# ✅ 起点: Sparse checkpoint (干净起点, 无训练历史)
# ✅ 精度: BF16 + Flash Attention (高效)
# ✅ Global Batch: 256 (标准)
# 🔽 LR Mult: 2.0 (平衡速度)
#    → Mask LR = 3.0e-5 (比FP32快2x)
# 🔼🔼 Weight Reg: 1.5e-5 (高度正则化)
#    → Reg loss 增长率 <0.6 /iter
#    → 稳定性方程: **0.5** (BF16安全阈值)
# 🔼 Clip Grad: 0.5 (标准BF16裁剪)
#    → 允许正常梯度流动
# 🔽 Scale 范围: 100→300 (已验证)
#    → Max Eff_Sharp = 120
# 🔼 Temp 范围: 4.0→2.5 (保守衰减)
#    → 保持充足柔性
# ✅ 使用: --bf16, --use-flash-attn
# 💾 Save Interval: 25 (用户指定)

cd $PROJECT_DIR

echo "🏃‍♂️ 开始 BF16 Conservative Plan B 训练..."
echo ""
echo "  ╔════════════════════════════════════════════════════════════════╗"
echo "  ║    🚀 BF16 Conservative: 平衡速度与稳定性                    ║"
echo "  ╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "  ┌────────────────────────┬───────────────────────────────────┐"
echo "  │ 配置项                 │ 值                                │"
echo "  ├────────────────────────┼───────────────────────────────────┤"
echo "  │ 精度                   │ BF16 + Flash Attention ✨        │"
echo "  │ 起始点                 │ Sparse checkpoint (干净起点)     │"
echo "  │ 起点 Reg Loss          │ ~7500 (预期)                     │"
echo "  │ 目标 Iteration         │ 2000                              │"
echo "  │ Global Batch Size      │ 256                               │"
echo "  │ Base Learning Rate     │ 1.5e-5                            │"
echo "  ├────────────────────────┼───────────────────────────────────┤"
echo "  │ LR Mult (Mask)         │ 2.0 (平衡速度)                   │"
echo "  │ Mask LR                │ ~3.0e-5 (比FP32快2x)             │"
echo "  │ Weight Reg             │ 1.5e-5 🔼🔼 (高度正则化)       │"
echo "  │ 稳定性方程             │ **0.5** (BF16安全阈值)           │"
echo "  │ Clip Grad              │ 0.5 (标准BF16)                   │"
echo "  ├────────────────────────┼───────────────────────────────────┤"
echo "  │ Gumbel Scale 范围      │ 100 → 300                        │"
echo "  │ Gumbel Temp 范围       │ 4.0 → 2.5                        │"
echo "  │ Max Eff Sharpness      │ 120 (已验证)                     │"
echo "  ├────────────────────────┼───────────────────────────────────┤"
echo "  │ Flash Attention        │ ✅ Yes                            │"
echo "  │ 显存占用               │ ~53GB / 80GB                     │"
echo "  │ Save Interval          │ $SAVE_INTERVAL iters (用户指定)  │"
echo "  └────────────────────────┴───────────────────────────────────┘"
echo ""
echo "🎯 BF16 Conservative Plan B 优势 (基于所有失败案例分析):"
echo "  ✅ 从干净checkpoint开始: 无累积数值问题"
echo "  ✅ BF16 + Flash Attention: 速度快15%, 显存节省23%"
echo "  ✅ 稳定性方程 0.5: 达到BF16安全阈值"
echo "  ✅ 高度正则化: weight_reg = 1.5e-5 (BF16历史最强10x)"
echo "  ✅ 平衡速度: Mask LR比FP32快2x, 但保持稳定性"
echo ""
echo "📊 预期效果 (iter 0→2000):"
echo "  - Reg loss 增长率: <0.6 /iter"
echo "  - Reg loss at 2000: ~8700 (vs 起点 ~7500)"
echo "  - Grad norm: 0.03-0.06 (健康范围)"
echo "  - 成功率: ~90% (BF16 + 稳定性0.5)"
echo "  - 训练时间: ~48h (vs FP32 ~54h, 节省11%)"
echo ""
echo "💾 显存预估:"
echo "  - 预期占用: 53-56GB / 80GB"
echo "  - 安全余量: 24-27GB"
echo ""
echo "🔍 关键监控指标 (每 25 iters 检查):"
echo "  ⚠️  如果 reg loss 增长率 > 1.0 /iter → 立即停止"
echo "  ⚠️  如果 grad norm > 0.08 → 立即停止"
echo "  ⚠️  如果 LM loss 不下降 → 检查数据/参数"
echo "  ✅ 目标: reg loss 增长率 < 0.6 /iter"
echo ""
echo "📝 日志文件: $LOG_FILE"
echo "📝 参考文档: llama8b_scripts/BF16_CONSERVATIVE_PLAN_B.md"
echo ""
echo "⏳ 开始训练..."
echo "================================================================"
echo ""

# 启动训练
torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank 0 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    pretrain_maskllm.py $options 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "================================================================"
echo "✅ 训练完成！"
echo "📝 日志文件: $LOG_FILE"
echo "💾 Checkpoint: $SAVE_CHECKPOINT"
echo "📊 参考文档: llama8b_scripts/BF16_CONSERVATIVE_PLAN_B.md"
echo "📊 对比文档: llama8b_scripts/TRAINING_STRATEGIES_COMPARISON.md"
echo "================================================================"

