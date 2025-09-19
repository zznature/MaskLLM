#!/bin/bash

# Convert Llama8b checkpoint from TP=8 to TP=1
# This script converts tensor parallel checkpoints to single device format for HF export

echo "========================================"
echo "Llama8b TP=8 to TP=1 Conversion Script"
echo "========================================"

PROJECT_DIR=$(pwd)

# Configuration
TP=1
TP8_DIR=$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp8
TP1_DIR=$PROJECT_DIR/output/checkpoints/llama8b_maskllm_training_tp1
TOKENIZER_MODEL=assets/checkpoints/Llama8b/tokenizer.py

# Check if input directory exists
if [ ! -d "$TP8_DIR" ]; then
    echo "ERROR: TP=8 checkpoint directory not found: $TP8_DIR"
    echo "Please ensure the sparse checkpoint has been created first."
    exit 1
fi

echo "Input TP=8 directory: $TP8_DIR"
echo "Output TP=1 directory: $TP1_DIR"

# Create output directory
mkdir -p "$TP1_DIR"

# Conversion options
OPTIONS=" \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --target-tensor-parallel-size ${TP} \
    --load-dir ${TP8_DIR} \
    --save-dir ${TP1_DIR} \
    --megatron-path ${PROJECT_DIR}"

echo ""
echo "Starting TP=8 to TP=1 conversion..."
echo "This process may take several minutes..."
echo ""

# Install required packages and run conversion
pip install transformers wandb accelerate tqdm --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Run the conversion utility
cd $PROJECT_DIR/tools/checkpoint
python util.py $OPTIONS

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ TP=8 to TP=1 conversion completed successfully!"
    echo "✅ Converted checkpoint saved to: $TP1_DIR"
    echo "✅ Ready for HuggingFace export"
else
    echo ""
    echo "❌ TP=8 to TP=1 conversion failed!"
    echo "Please check the error messages above and verify:"
    echo "  1. Input TP=8 checkpoint exists and is valid"
    echo "  2. All required dependencies are installed"
    echo "  3. Sufficient disk space for conversion output"
    exit 1
fi

echo "========================================"
echo "TP conversion completed"
echo "========================================"
