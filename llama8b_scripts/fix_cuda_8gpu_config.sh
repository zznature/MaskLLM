#!/bin/bash
# CUDA 8GPU配置修复脚本
# 解决容器内只显示4卡的问题，确保8卡可见

set -euo pipefail

echo "🔧 CUDA 8GPU配置修复"
echo "============================================================"
echo "目标: 确保容器内可见全部8个GPU设备"
echo ""

echo "📋 1. 当前GPU状态检查:"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "宿主机nvidia-smi:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    HOST_GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "宿主机GPU总数: $HOST_GPU_COUNT"
else
    echo "❌ nvidia-smi不可用"
    HOST_GPU_COUNT=0
fi
echo ""

echo "当前PyTorch可见GPU:"
TORCH_GPU_COUNT=$(python3.10 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
echo "PyTorch GPU数量: $TORCH_GPU_COUNT"
echo ""

echo "📋 2. 环境变量配置修复:"
echo "----------------------------------------"

# 清除可能限制GPU可见性的环境变量
echo "清除限制性环境变量:"
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
echo "✓ 已清除 CUDA_VISIBLE_DEVICES"

# 设置8GPU可见
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "✓ 设置 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

# 确保CUDA路径正确
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda" 
export CUDA_DEVICE_MAX_CONNECTIONS=1
echo "✓ 设置 CUDA 环境变量"
echo ""

echo "📋 3. GPU设备文件检查:"
echo "----------------------------------------"
echo "GPU设备文件:"
ls -la /dev/nvidia* 2>/dev/null | head -10 || echo "❌ 未找到GPU设备文件"
echo ""

echo "nvidia-uvm设备:"
ls -la /dev/nvidia-uvm* 2>/dev/null || echo "❌ 未找到nvidia-uvm设备"
echo ""

echo "📋 4. 修复后GPU可见性验证:"
echo "----------------------------------------"
python3.10 -c "
import torch
import os
print(f'环境变量 CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"未设置\")}')
print(f'PyTorch CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'PyTorch可见GPU数量: {gpu_count}')
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory//1024//1024//1024}GB)')
else:
    print('❌ PyTorch仍无法访问CUDA')
"
echo ""

echo "📋 5. NCCL配置检查和修复:"
echo "----------------------------------------"
# 设置NCCL环境变量避免分布式问题
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
echo "✓ 设置 NCCL 环境变量避免冲突"

python3.10 -c "
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print(f'NCCL版本: {torch.cuda.nccl.version()}')
        print('测试GPU间通信...')
        for i in range(min(torch.cuda.device_count(), 8)):
            torch.cuda.set_device(i)
            test_tensor = torch.cuda.FloatTensor([i])
            print(f'GPU {i}: 通信测试通过 ✓')
    else:
        print('GPU数量不足，跳过NCCL测试')
except Exception as e:
    print(f'NCCL测试失败: {e}')
"
echo ""

echo "📋 6. 生成修复后的训练脚本:"
echo "----------------------------------------"

# 创建修复版本的训练脚本
cat > llama8b_scripts/llama8b_mask_only_tp8_c4_fixed.sh << 'EOF'
#!/usr/bin/bash
# Llama8b MaskLLM 稀疏化训练脚本 - CUDA 8GPU修复版本
# 解决GPU可见性问题

# 强制设置8GPU可见性
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 设置NCCL环境避免分布式冲突
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 清除可能的GPU限制
unset CUDA_LAUNCH_BLOCKING 2>/dev/null || true

echo "🚀 Llama8b MaskLLM 8GPU 修复版本"
echo "📋 GPU配置: $CUDA_VISIBLE_DEVICES"
echo ""

# Get Data Blend
C4_HOME=assets/data/c4_llama2_pretokenized
DATA_BLEND=""
for i in {00000..00019}; do # 1/20
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama2_${i}_text_document"
done

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45526"

# Device Configs
NNODES=1
NPROC_PER_NODE=8
export WORLD_SIZE=8
resume=$1

# Task Configs
TAG="llama8b-tp8-mask-fixed"
DATA_INDEX_PATH=CACHE
PROJECT_PATH=$(pwd)
OUTPUT_PATH="$PROJECT_PATH/output"

# Transformer Configs
HIDEN_SIZE=4096
NUM_LAYERS=32
NUM_ATTN_HEADS=32
SEQ_LENGTH=4096
VOCAB_SIZE=119696

# Training Configs
TOKENIZER_TYPE="NullTokenizer"
TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
LR=5e-5
MIN_LR=5e-6
TRAIN_ITERS=100
WARMUP_ITERS=0
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8

# intervals
SAVE_INTERVALS=50
LOG_INTERVALS=5
EVAL_INTERVALS=25
EVAL_ITERS=5

# Set Training configs
CKPT_SUBDIR="$OUTPUT_PATH/checkpoints/$TAG/train_iters_$TRAIN_ITERS"
if [ $resume -eq 0 ]; then
    LOAD="$PROJECT_PATH/output/checkpoints/llama8b_megatron_tp8" 
    EXTRA_CMD="--no-load-optim --no-load-rng --finetune --enable-partial-load " 
else
    LOAD="$CKPT_SUBDIR/ckpt"
    EXTRA_CMD=""
fi  

TASK_CMD=" --gumbel-scale-range 1e2 5e2 --gumbel-temperature-range 4 0.05 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 10 --weight-reg 1e-5 "

cd $PROJECT_PATH; mkdir -p $CKPT_SUBDIR/ckpt; mkdir -p $CKPT_SUBDIR/logs

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

echo "开始8GPU稀疏化训练..."
torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT pretrain_maskllm.py ${OPTIONS}
EOF

chmod +x llama8b_scripts/llama8b_mask_only_tp8_c4_fixed.sh
echo "✓ 创建修复版本训练脚本: llama8b_mask_only_tp8_c4_fixed.sh"
echo ""

echo "📋 7. 验证修复效果:"
echo "----------------------------------------"
FIXED_GPU_COUNT=$(python3.10 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
echo "修复后PyTorch GPU数量: $FIXED_GPU_COUNT"

if [ "$FIXED_GPU_COUNT" -eq 8 ]; then
    echo "✅ GPU配置修复成功！"
    echo ""
    echo "🚀 可以运行修复版本训练:"
    echo "bash run_maskllm_native.sh llama8b_scripts/llama8b_mask_only_tp8_c4_fixed.sh 0"
elif [ "$FIXED_GPU_COUNT" -gt 4 ]; then
    echo "⚠️ GPU数量增加但未达到8个，可能需要进一步配置"
    echo "当前可用GPU: $FIXED_GPU_COUNT"
else
    echo "❌ GPU配置仍有问题，需要检查容器启动参数"
    echo ""
    echo "🔧 可能的解决方案:"
    echo "1. 检查容器是否使用了--gpus all参数"
    echo "2. 验证宿主机GPU驱动状态"
    echo "3. 检查容器运行时配置"
fi
echo ""

echo "============================================================"
echo "🎯 修复总结"
echo "============================================================"
echo "已执行的修复操作:"
echo "  ✓ 清除GPU可见性限制"
echo "  ✓ 设置8GPU环境变量"
echo "  ✓ 配置NCCL防冲突参数"
echo "  ✓ 创建修复版本训练脚本"
echo ""
echo "下一步: 使用修复版本脚本进行8GPU训练测试"
