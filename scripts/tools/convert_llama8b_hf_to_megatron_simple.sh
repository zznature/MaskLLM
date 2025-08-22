#!/bin/bash

# Simple Llama8b HF to Megatron conversion script
# Based on successful llama2/llama3 conversion patterns

PROJECT_DIR=$(pwd)

# Configuration
TP=${1:-8}  # Default to 8, can be overridden by first argument
HF_FORMAT_DIR=$PROJECT_DIR/assets/checkpoints/Llama8b
MEGATRON_FORMAT_DIR=$PROJECT_DIR/output/checkpoints/llama8b_megatron_tp$TP

# Use existing tokenizer files from Llama8b
TOKENIZER_MODEL=$HF_FORMAT_DIR  # Directory containing tokenizer files

echo "=== Llama8b Simple Conversion ==="
echo "Tensor Parallel Size: $TP"
echo "HF Format Dir: $HF_FORMAT_DIR"
echo "Megatron Format Dir: $MEGATRON_FORMAT_DIR"
echo "Tokenizer Model: $TOKENIZER_MODEL"
echo ""

# Validate input directory
if [[ ! -d "$HF_FORMAT_DIR" ]]; then
    echo "Error: HF format directory not found: $HF_FORMAT_DIR"
    exit 1
fi

if [[ ! -f "$HF_FORMAT_DIR/config.json" ]]; then
    echo "Error: config.json not found in $HF_FORMAT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$MEGATRON_FORMAT_DIR"

# Use the same pattern as successful llama2/llama3 conversions
OPTIONS=" \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size ${TP} \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL}"

echo "Conversion command:"
echo "python util.py $OPTIONS"
echo ""

# Change to checkpoint tools directory and run conversion
cd $PROJECT_DIR/tools/checkpoint

# Set environment for the conversion
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:$PROJECT_DIR/.ext_pkgs"

echo "Starting conversion..."
python3.10 util.py $OPTIONS

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Conversion completed successfully!"
    
    # Copy tokenizer files (following the pattern of successful scripts)
    echo "Copying tokenizer files..."
    cp -r $HF_FORMAT_DIR/vocab.txt $MEGATRON_FORMAT_DIR/ 2>/dev/null || true
    cp -r $HF_FORMAT_DIR/tokenizer.json $MEGATRON_FORMAT_DIR/ 2>/dev/null || true
    cp -r $HF_FORMAT_DIR/llama8b/ $MEGATRON_FORMAT_DIR/ 2>/dev/null || true
    
    echo ""
    echo "Output directory contents:"
    ls -la "$MEGATRON_FORMAT_DIR"
    
    echo ""
    echo "🎉 Llama8b conversion to TP=$TP completed!"
    echo "Output: $MEGATRON_FORMAT_DIR"
else
    echo "❌ Conversion failed"
    exit 1
fi
