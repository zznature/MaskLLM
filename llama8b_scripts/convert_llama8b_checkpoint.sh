#!/bin/bash

# Llama8b Checkpoint Conversion Script
# This script converts the trained llama8b checkpoint by trimming learnable sparsity components
# and converting weights to float16 format

echo "========================================"
echo "Llama8b Checkpoint Conversion Script"
echo "========================================"

# Set paths
SCRIPT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts"
CHECKPOINT_DIR="output/checkpoints/llama8b_maskllm_training_tp8/iter_0002000"
CONVERSION_SCRIPT="$SCRIPT_DIR/tool_trim_learnable_sparsity_llama8b.py"

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Please verify the checkpoint path and try again."
    exit 1
fi

# Check if conversion script exists
if [ ! -f "$CONVERSION_SCRIPT" ]; then
    echo "ERROR: Conversion script not found: $CONVERSION_SCRIPT"
    exit 1
fi

echo "Input checkpoint directory: $CHECKPOINT_DIR"
echo "Using conversion script: $CONVERSION_SCRIPT"
echo ""

# Run the conversion
echo "Starting checkpoint conversion..."
python "$CONVERSION_SCRIPT" --ckpt_dir "$CHECKPOINT_DIR"

# Check if conversion was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Checkpoint conversion completed successfully!"
    echo "✅ Converted checkpoint saved to: output/checkpoints/llama8b_maskllm_training_tp8/release"
    echo "✅ Latest checkpoint pointer updated to 'release'"
    echo ""
    echo "The converted checkpoint includes:"
    echo "  - Winner masks only (removed diff_mask.gate, mask_options)"
    echo "  - All weights converted to float16 format"
    echo "  - Optimized for inference usage"
else
    echo ""
    echo "❌ Checkpoint conversion failed!"
    echo "Please check the error messages above and verify:"
    echo "  1. Checkpoint directory exists and contains valid files"
    echo "  2. All required dependencies are available"
    echo "  3. Sufficient disk space for conversion output"
    exit 1
fi

echo "========================================"
echo "Conversion process completed"
echo "========================================"
