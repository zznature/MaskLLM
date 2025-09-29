#!/bin/bash

# Llama8b WIKITEXT2 Perplexity Evaluation Script
# Based on evaluate_llama2_wikitext2.sh with Llama8b adaptations

# export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
LOAD=$1 # path to the model
TP=$2   # tensor parallel size (1, 2, 4, 8)
MODE=$3 # dense, sparse

# Normalize load directory so Megatron can locate latest_checkpointed_iteration.txt
if [ -f "$LOAD/latest_checkpointed_iteration.txt" ]; then
    echo "🔍 Detected checkpoint metadata in $LOAD"
elif [ -f "$(dirname "$LOAD")/latest_checkpointed_iteration.txt" ]; then
    echo "🔄 latest_checkpointed_iteration.txt found in parent directory. Adjusting --load to parent."
    LOAD="$(dirname "$LOAD")"
else
    echo "❌ latest_checkpointed_iteration.txt not found in $LOAD or its parent."
    echo "   Please ensure the checkpoint directory structure is intact."
    exit 1
fi

echo "🚀 Llama8b WIKITEXT2 Evaluation"
echo "Model Path: $LOAD"
echo "Tensor Parallel: $TP"
echo "Mode: $MODE"
echo ""

PROJECT_DIR=$(pwd) # change this to the path of your maskllm project
OUTPUT="$PROJECT_DIR/output"

# Llama8b model configuration
HIDDEN_SIZE=4096 # hidden size (8b model)
NUM_LAYERS=32 # number of layers
NUM_ATTN_HEADS=32 # number of attention heads
FFN_HIDDEN_SIZE=14336 # feedforward hidden size
SEQ_LENGTH=4096 # sequence length

# Llama8b tokenizer configuration
TOKENIZER_MODEL="$PROJECT_DIR/assets/checkpoints/Llama8b"
TOKENIZER_PYTHON="$PROJECT_DIR/assets/checkpoints/Llama8b/llama8b/tokenizer.py"
VOCAB_FILE="$PROJECT_DIR/assets/checkpoints/Llama8b/vocab.txt"

# Check if tokenizer files exist
if [ ! -f "$TOKENIZER_PYTHON" ]; then
    echo "❌ Llama8b tokenizer Python module not found: $TOKENIZER_PYTHON"
    echo "Please ensure the original Llama8b model is available in assets/checkpoints/Llama8b/llama8b/"
    exit 1
fi

if [ ! -f "$VOCAB_FILE" ]; then
    echo "❌ Llama8b vocab file not found: $VOCAB_FILE"
    echo "Please ensure the original Llama8b model is available in assets/checkpoints/Llama8b/"
    exit 1
fi

echo "✅ Llama8b tokenizer files verified:"
echo "   - Python module: $TOKENIZER_PYTHON"
echo "   - Vocab file: $VOCAB_FILE"

# Set sparsity mode
# Note: For checkpoints from Stage 2+ (sparsity already applied to weights),
#       we load in DENSE mode since sparsity is "burned into" the weights
if [ "$MODE" == "dense" ]; then
   MASK_OPTIONS=" "
   echo "🔹 Running in DENSE mode"
elif [ "$MODE" == "sparse" ]; then
   # For Stage 2+ checkpoints: sparsity already applied, load as dense
   MASK_OPTIONS=" "
   echo "🔸 Running in SPARSE mode (loading sparse weights as dense)"
else
   MASK_OPTIONS=" "
   echo "🔹 Running in DEFAULT (dense) mode"
fi

# Validate tensor parallel size
if [[ ! "$TP" =~ ^[1248]$ ]]; then
    echo "⚠️ Warning: TP=$TP may not be optimal. Recommended values: 1, 2, 4, 8"
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1;

OPTIONS=" \
   --task WIKITEXT2 \
   --use-flash-attn \
   --untie-embeddings-and-output-weights \
   --disable-bias-linear \
   --no-position-embedding \
   --no-masked-softmax-fusion \
   --use-rotary-position-embeddings \
   --swiglu \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --tensor-model-parallel-size $TP \
   --pipeline-model-parallel-size 1 \
   --overlapping-eval $SEQ_LENGTH \
   --num-layers $NUM_LAYERS \
   --hidden-size $HIDDEN_SIZE \
   --num-attention-heads $NUM_ATTN_HEADS \
   --seq-length $SEQ_LENGTH \
   --max-position-embeddings $SEQ_LENGTH \
   --micro-batch-size 1 \
   --global-batch-size 256 \
   --train-iters 1 \
   --lr-decay-iters 1 \
   --lr 1.0e-4 \
   --min-lr 1.0e-5 \
   --lr-decay-style cosine \
   --log-interval 100 \
   --tokenizer-type Llama8bTokenizer \
   --tokenizer-model ${TOKENIZER_MODEL} \
   --make-vocab-size-divisible-by 1 \
   --ffn-hidden-size $FFN_HIDDEN_SIZE --normalization RMSNorm \
   --data-path None \
   --bf16 \
   --no-save-optim --no-save-rng \
   --no-load-optim --no-load-rng \
   --exit-on-missing-checkpoint \
   --load ${LOAD} \
   --hidden-dropout 0.0 --attention-dropout 0.0 \
   $MASK_OPTIONS"

echo "📁 Project Directory: $PROJECT_DIR"
echo "📋 Starting evaluation with the following configuration:"
echo "   - Hidden Size: $HIDDEN_SIZE"
echo "   - Layers: $NUM_LAYERS"
echo "   - Attention Heads: $NUM_ATTN_HEADS"
echo "   - FFN Hidden Size: $FFN_HIDDEN_SIZE"
echo "   - Sequence Length: $SEQ_LENGTH"
echo "   - Tensor Parallel: $TP"
echo "   - Tokenizer: Llama8bTokenizer"
echo ""

cd $PROJECT_DIR; 

export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45531" # select the port (avoid conflict with training)
NNODES=1 # number of nodes
NPROC_PER_NODE=${TP} # number of gpus (processes) per node
export WORLD_SIZE=$(($NNODES * $NPROC_PER_NODE)) # number of gpus we have in total

echo "🔥 Running torchrun with $NPROC_PER_NODE processes..."

# Use the correct torchrun path (available in container environment) 
TORCHRUN_CMD="/usr/local/bin/torchrun"
if [ ! -f "$TORCHRUN_CMD" ]; then
    TORCHRUN_CMD="torchrun"  # fallback
fi

echo "Command: $TORCHRUN_CMD --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tasks/main.py"
echo ""

$TORCHRUN_CMD --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT tasks/main.py ${OPTIONS} ${MASK_OPTIONS}

echo ""
echo "✅ Llama8b WIKITEXT2 evaluation completed!"
