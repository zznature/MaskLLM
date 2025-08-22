#!/bin/bash

# Convert Llama8b HuggingFace checkpoint to Megatron format
# This script handles the complete conversion pipeline for Llama8b models

set -e

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Llama8b HuggingFace to Megatron Conversion ===${NC}"
echo ""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input-dir PATH          Path to HuggingFace Llama8b checkpoint directory"
    echo "                           (default: assets/checkpoints/Llama8b)"
    echo "  --output-dir PATH         Path to save Megatron checkpoint"
    echo "                           (default: output/checkpoints/llama8b_megatron)"
    echo "  --tensor-parallel-size N  Tensor parallel size for conversion (default: 1)"
    echo "  --pipeline-parallel-size N Pipeline parallel size (default: 1)"
    echo "  --target-precision STR    Target precision: fp32, fp16, bf16 (default: fp16)"
    echo "  --dry-run                Run validation only, don't convert"
    echo "  --help                   Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  LLAMA8B_INPUT_DIR        Override default input directory"
    echo "  LLAMA8B_OUTPUT_DIR       Override default output directory"
    echo ""
    echo "Examples:"
    echo "  # Basic conversion with default settings"
    echo "  $0"
    echo ""
    echo "  # Convert with tensor parallelism"
    echo "  $0 --tensor-parallel-size 4"
    echo ""
    echo "  # Custom paths"
    echo "  $0 --input-dir /path/to/llama8b --output-dir /path/to/output"
}

# Parse command line arguments
INPUT_DIR="${LLAMA8B_INPUT_DIR:-assets/checkpoints/Llama8b}"
OUTPUT_DIR="${LLAMA8B_OUTPUT_DIR:-output/checkpoints/llama8b_megatron}"
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
TARGET_PRECISION="fp16"
DRY_RUN=false

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
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --pipeline-parallel-size)
            PIPELINE_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --target-precision)
            TARGET_PRECISION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Validate inputs
echo -e "${YELLOW}=== Configuration Validation ===${NC}"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Pipeline parallel size: $PIPELINE_PARALLEL_SIZE"
echo "Target precision: $TARGET_PRECISION"
echo "Dry run: $DRY_RUN"
echo ""

# Check if running from correct directory
if [[ ! -f "tools/checkpoint/util.py" ]]; then
    echo -e "${RED}Error: Please run this script from MaskLLM root directory${NC}"
    exit 1
fi

# Check input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}Error: Input directory does not exist: $INPUT_DIR${NC}"
    exit 1
fi

# Check required files in input directory
REQUIRED_FILES=("config.json" "pytorch_model.bin" "vocab.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$INPUT_DIR/$file" ]]; then
        echo -e "${RED}Error: Required file not found: $INPUT_DIR/$file${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ Input validation passed${NC}"

# Check tokenizer structure
if [[ -d "$INPUT_DIR/llama8b" && -f "$INPUT_DIR/llama8b/tokenizer.py" ]]; then
    echo -e "${GREEN}✅ Llama8b tokenizer structure detected${NC}"
else
    echo -e "${YELLOW}⚠️  Llama8b tokenizer structure not found, will use fallback${NC}"
fi

# Validate model configuration
echo -e "${YELLOW}=== Model Configuration Check ===${NC}"
VOCAB_SIZE=$(python3 -c "
import json
with open('$INPUT_DIR/config.json') as f:
    config = json.load(f)
print(config.get('vocab_size', 'unknown'))
")

if [[ "$VOCAB_SIZE" == "119696" ]]; then
    echo -e "${GREEN}✅ Vocabulary size confirmed: $VOCAB_SIZE${NC}"
elif [[ "$VOCAB_SIZE" =~ ^[0-9]+$ ]]; then
    echo -e "${YELLOW}⚠️  Unexpected vocabulary size: $VOCAB_SIZE (expected 119696)${NC}"
else
    echo -e "${RED}❌ Could not determine vocabulary size${NC}"
    exit 1
fi

# If dry run, stop here
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${GREEN}✅ Dry run completed successfully${NC}"
    echo "Run without --dry-run to perform actual conversion"
    exit 0
fi

# Create output directory
echo -e "${YELLOW}=== Preparing Output Directory ===${NC}"
mkdir -p "$OUTPUT_DIR"
echo "Created output directory: $OUTPUT_DIR"

# Prepare conversion command
CONVERSION_CMD="python3.10 tools/checkpoint/util.py"
CONVERSION_CMD+=" --model-type GPT"
CONVERSION_CMD+=" --loader llama8b_hf"
CONVERSION_CMD+=" --saver megatron"
CONVERSION_CMD+=" --load-dir $INPUT_DIR"
CONVERSION_CMD+=" --save-dir $OUTPUT_DIR"
CONVERSION_CMD+=" --tokenizer-model $INPUT_DIR"

# Add parallel settings if specified
if [[ "$TENSOR_PARALLEL_SIZE" -gt 1 ]]; then
    CONVERSION_CMD+=" --target-tensor-parallel-size $TENSOR_PARALLEL_SIZE"
fi

if [[ "$PIPELINE_PARALLEL_SIZE" -gt 1 ]]; then
    CONVERSION_CMD+=" --target-pipeline-parallel-size $PIPELINE_PARALLEL_SIZE"
fi

# Note: Precision is set in the loader, not as command line arguments
# The fp16 setting is handled in load_args_from_checkpoint() function
echo "Target precision: $TARGET_PRECISION (configured in loader)"

echo -e "${YELLOW}=== Starting Conversion ===${NC}"
echo "Conversion command:"
echo "$CONVERSION_CMD"
echo ""

# Record start time
START_TIME=$(date +%s)

# Add megatron path to conversion command
CONVERSION_CMD+=" --megatron-path $(pwd)"

echo -e "${BLUE}Running checkpoint conversion...${NC}"

# Use run_maskllm_native.sh to ensure proper environment
# Create a temporary script to run the conversion command
TEMP_SCRIPT="$(mktemp)"
cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
$CONVERSION_CMD
EOF
chmod +x "$TEMP_SCRIPT"

echo "Using run_maskllm_native.sh for proper environment setup..."
if bash run_maskllm_native.sh "$TEMP_SCRIPT"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}✅ Conversion completed successfully!${NC}"
    echo "Duration: ${DURATION} seconds"
    echo "Output saved to: $OUTPUT_DIR"
    
    # Display output structure
    echo ""
    echo -e "${YELLOW}=== Output Structure ===${NC}"
    if [[ -d "$OUTPUT_DIR" ]]; then
        ls -la "$OUTPUT_DIR"
        
        # Check for expected files
        if [[ -f "$OUTPUT_DIR/latest_checkpointed_iteration.txt" ]]; then
            echo -e "${GREEN}✅ Checkpoint metadata found${NC}"
        fi
        
        # Show checkpoint directory structure if exists
        CHECKPOINT_DIRS=($(find "$OUTPUT_DIR" -type d -name "iter_*" 2>/dev/null | head -3))
        if [[ ${#CHECKPOINT_DIRS[@]} -gt 0 ]]; then
            echo ""
            echo "Checkpoint directories:"
            for dir in "${CHECKPOINT_DIRS[@]}"; do
                echo "  $dir"
                ls -la "$dir" | head -5
                if [[ $(ls -la "$dir" | wc -l) -gt 6 ]]; then
                    echo "  ..."
                fi
            done
        fi
    fi
    
    echo ""
    echo -e "${GREEN}🎉 Llama8b checkpoint conversion completed!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Test model loading with the converted checkpoint"
    echo "2. Run inference validation"
    echo "3. Test sparse training compatibility"
    echo ""
    echo -e "${YELLOW}Usage example:${NC}"
    echo "python pretrain_maskllm.py --load $OUTPUT_DIR [other args]"
    
else
    echo ""
    echo -e "${RED}❌ Conversion failed${NC}"
    echo "Check the error messages above for troubleshooting"
    rm -f "$TEMP_SCRIPT"
    exit 1
fi

# Clean up temporary script
rm -f "$TEMP_SCRIPT"
