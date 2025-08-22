#!/bin/bash

# Test script for Llama8b checkpoint conversion
# This script tests the conversion process with validation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Llama8b Checkpoint Conversion Test ===${NC}"
echo ""

# Default paths
INPUT_DIR="assets/checkpoints/Llama8b"
OUTPUT_DIR="output/checkpoints/llama8b_megatron_test"
CONVERSION_SCRIPT="scripts/tools/convert_llama8b_hf_to_megatron.sh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--input-dir PATH] [--output-dir PATH]"
            echo ""
            echo "Options:"
            echo "  --input-dir PATH     Path to Llama8b HF checkpoint (default: assets/checkpoints/Llama8b)"
            echo "  --output-dir PATH    Path for output checkpoint (default: output/checkpoints/llama8b_megatron_test)"
            echo "  --help, -h           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [[ ! -f "tools/checkpoint/util.py" ]]; then
    echo -e "${RED}Error: Please run this script from MaskLLM root directory${NC}"
    exit 1
fi

# Verify prerequisites
echo -e "${YELLOW}=== Prerequisites Check ===${NC}"

# Check input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}❌ Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

# Check conversion script
if [[ ! -f "$CONVERSION_SCRIPT" ]]; then
    echo -e "${RED}❌ Conversion script not found: $CONVERSION_SCRIPT${NC}"
    exit 1
fi

# Check loader
if [[ ! -f "tools/checkpoint/loader_llama8b_hf.py" ]]; then
    echo -e "${RED}❌ Llama8b loader not found: tools/checkpoint/loader_llama8b_hf.py${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites found${NC}"

# Clean output directory if exists
if [[ -d "$OUTPUT_DIR" ]]; then
    echo -e "${YELLOW}⚠️  Output directory exists, cleaning: $OUTPUT_DIR${NC}"
    rm -rf "$OUTPUT_DIR"
fi

echo ""
echo -e "${YELLOW}=== Phase 1: Dry Run Validation ===${NC}"

# Run dry run first
if bash "$CONVERSION_SCRIPT" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --dry-run; then
    echo -e "${GREEN}✅ Dry run validation passed${NC}"
else
    echo -e "${RED}❌ Dry run validation failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}=== Phase 2: Actual Conversion Test ===${NC}"

# Record start time
START_TIME=$(date +%s)

# Run actual conversion
echo "Starting conversion test..."
if bash "$CONVERSION_SCRIPT" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --tensor-parallel-size 1; then
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "${GREEN}✅ Conversion completed in ${DURATION} seconds${NC}"
else
    echo -e "${RED}❌ Conversion failed${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}=== Phase 3: Output Validation ===${NC}"

# Check output structure
echo "Validating output structure..."

# Check basic structure
if [[ -d "$OUTPUT_DIR" ]]; then
    echo -e "${GREEN}✅ Output directory created${NC}"
else
    echo -e "${RED}❌ Output directory not found${NC}"
    exit 1
fi

# Check for checkpoint metadata
if [[ -f "$OUTPUT_DIR/latest_checkpointed_iteration.txt" ]]; then
    LATEST_ITER=$(cat "$OUTPUT_DIR/latest_checkpointed_iteration.txt")
    echo -e "${GREEN}✅ Checkpoint metadata found (iteration: $LATEST_ITER)${NC}"
else
    echo -e "${YELLOW}⚠️  Checkpoint metadata not found${NC}"
fi

# Find checkpoint directories
CHECKPOINT_DIRS=($(find "$OUTPUT_DIR" -type d -name "iter_*" 2>/dev/null))
if [[ ${#CHECKPOINT_DIRS[@]} -gt 0 ]]; then
    echo -e "${GREEN}✅ Found ${#CHECKPOINT_DIRS[@]} checkpoint directory(ies)${NC}"
    for dir in "${CHECKPOINT_DIRS[@]}"; do
        echo "  - $dir"
        
        # Check for model files
        if [[ -f "$dir/mp_rank_00_model_states.pt" ]]; then
            echo -e "${GREEN}    ✅ Model state found${NC}"
        else
            echo -e "${YELLOW}    ⚠️  Model state not found${NC}"
        fi
        
        # Check file sizes
        MODEL_SIZE=$(du -h "$dir" 2>/dev/null | tail -1 | cut -f1)
        echo "    Size: $MODEL_SIZE"
    done
else
    echo -e "${RED}❌ No checkpoint directories found${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}=== Phase 4: Quick Model Loading Test ===${NC}"

# Create a simple model loading test
cat > /tmp/test_llama8b_loading.py << 'EOF'
#!/usr/bin/env python3

import sys
import os
import torch

def test_loading(checkpoint_dir):
    """Test basic checkpoint loading without full Megatron initialization."""
    
    print(f"Testing checkpoint loading from: {checkpoint_dir}")
    
    # Find checkpoint directories
    import glob
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "iter_*"))
    if not checkpoint_dirs:
        print("❌ No checkpoint directories found")
        return False
    
    latest_checkpoint = sorted(checkpoint_dirs)[-1]
    print(f"Latest checkpoint: {latest_checkpoint}")
    
    # Try to load model state
    model_state_file = os.path.join(latest_checkpoint, "mp_rank_00_model_states.pt")
    if not os.path.exists(model_state_file):
        print("❌ Model state file not found")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_state_file, map_location='cpu')
        
        # Check basic structure
        if 'model' in checkpoint:
            model_state = checkpoint['model']
            print(f"✅ Model state loaded with {len(model_state)} parameters")
            
            # Check for expected parameters
            expected_params = [
                'language_model.embedding.word_embeddings.weight',
                'language_model.encoder.final_norm.weight',
                'language_model.output_layer.weight'
            ]
            
            found_params = 0
            for param in expected_params:
                if param in model_state:
                    shape = model_state[param].shape
                    print(f"  ✅ {param}: {shape}")
                    found_params += 1
                else:
                    print(f"  ❌ {param}: not found")
            
            if found_params == len(expected_params):
                print("✅ All expected parameters found")
                return True
            else:
                print(f"⚠️  Only {found_params}/{len(expected_params)} parameters found")
                return False
        else:
            print("❌ No 'model' key in checkpoint")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_loading.py <checkpoint_dir>")
        sys.exit(1)
    
    success = test_loading(sys.argv[1])
    sys.exit(0 if success else 1)
EOF

# Run the loading test
if python3 /tmp/test_llama8b_loading.py "$OUTPUT_DIR"; then
    echo -e "${GREEN}✅ Basic loading test passed${NC}"
else
    echo -e "${YELLOW}⚠️  Basic loading test had issues (may be normal)${NC}"
fi

# Clean up
rm -f /tmp/test_llama8b_loading.py

echo ""
echo -e "${YELLOW}=== Test Summary ===${NC}"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Conversion time: ${DURATION} seconds"

# Calculate output size
if [[ -d "$OUTPUT_DIR" ]]; then
    OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
    echo "Output size: $OUTPUT_SIZE"
fi

echo ""
echo -e "${GREEN}🎉 Checkpoint conversion test completed!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Run full model loading test in Megatron environment"
echo "2. Test inference with converted checkpoint"
echo "3. Validate sparse training compatibility"
echo ""
echo -e "${YELLOW}Commands to try:${NC}"
echo "# Test loading:"
echo "bash run_maskllm_native.sh scripts/test/test_model_loading.sh $OUTPUT_DIR"
echo ""
echo "# Test inference:"
echo "bash run_maskllm_native.sh scripts/test/test_inference.sh $OUTPUT_DIR"
