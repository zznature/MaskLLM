#!/bin/bash

# Export Llama8b sparse checkpoint to HuggingFace format
# This script converts Megatron TP=1 checkpoint to HF format

echo "========================================"
echo "Llama8b HuggingFace Export Script"
echo "========================================"

PROJECT_DIR=$(pwd)

# Configuration paths
HF_TEMPLATE_DIR="assets/checkpoints/Llama8b"
MEGATRON_CKPT_FILE="output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng.pt"
OUTPUT_HF_DIR="output/checkpoints/llama8b_hf_maskllm_c4"

# Check if required files exist
if [ ! -d "$HF_TEMPLATE_DIR" ]; then
    echo "ERROR: HF template directory not found: $HF_TEMPLATE_DIR"
    echo "Please ensure the original Llama8b model is available."
    exit 1
fi

if [ ! -f "$MEGATRON_CKPT_FILE" ]; then
    echo "ERROR: Megatron checkpoint file not found: $MEGATRON_CKPT_FILE"
    echo "Please run TP=8 to TP=1 conversion first."
    exit 1
fi

echo "HF template directory: $HF_TEMPLATE_DIR"
echo "Megatron checkpoint file: $MEGATRON_CKPT_FILE"
echo "Output HF directory: $OUTPUT_HF_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_HF_DIR"

echo "Starting HuggingFace export..."

# Run the export script
python tool_export_to_hf.py \
    --hf_ckpt "$HF_TEMPLATE_DIR" \
    --megatron_ckpt "$MEGATRON_CKPT_FILE" \
    --save_ckpt "$OUTPUT_HF_DIR"

# Check if export was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ HuggingFace export completed successfully!"
    echo ""
    
    # Copy necessary tokenizer files
    echo "Copying tokenizer and configuration files..."
    cp "$HF_TEMPLATE_DIR"/tokenizer* "$OUTPUT_HF_DIR/" 2>/dev/null || true
    cp "$HF_TEMPLATE_DIR"/vocab.txt "$OUTPUT_HF_DIR/" 2>/dev/null || true
    cp "$HF_TEMPLATE_DIR"/config.json "$OUTPUT_HF_DIR/" 2>/dev/null || true
    cp "$HF_TEMPLATE_DIR"/special_tokens_map.json "$OUTPUT_HF_DIR/" 2>/dev/null || true
    
    echo "✅ Tokenizer files copied"
    echo ""
    echo "Exported HF model structure:"
    ls -la "$OUTPUT_HF_DIR"
    echo ""
    echo "✅ Llama8b sparse model exported to: $OUTPUT_HF_DIR"
    echo "✅ Ready for evaluation and deployment"
    
else
    echo ""
    echo "❌ HuggingFace export failed!"
    echo "Please check the error messages above and verify:"
    echo "  1. Megatron checkpoint file is valid"
    echo "  2. HF template directory contains required files"
    echo "  3. Sufficient disk space for output"
    exit 1
fi

echo "========================================"
echo "HuggingFace export completed"
echo "========================================"
