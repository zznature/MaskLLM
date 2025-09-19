#!/bin/bash
# Llama8b HuggingFace Format Inference Launch Script

set -e  # Exit on any error

# Configuration
MODEL_PATH="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4"
VOCAB_PATH="/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b/vocab.txt"
INFERENCE_SCRIPT="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/llama8b_inference/llama8b_hf_inference_simple.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}   Llama8b HF Inference Launcher${NC}"
echo -e "${BLUE}================================${NC}"

# Function to show usage
show_usage() {
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model PATH      Model path (default: $MODEL_PATH)"
    echo "  -v, --vocab PATH      Vocab file path (default: $VOCAB_PATH)"
    echo "  -d, --device DEVICE   Device to use (auto/cuda/cpu, default: auto)"
    echo "  -p, --prompt TEXT     Single prompt for non-interactive mode"
    echo "  -l, --max_length N    Maximum generation length (default: 512)"
    echo "  -t, --temperature F   Temperature for generation (default: 0.7)"
    echo "  --demo                Run with demo prompts"
    echo "  --test                Run quick test mode"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Interactive mode"
    echo "  $0 --demo                            # Demo mode with sample prompts"
    echo "  $0 --test                            # Quick test mode"
    echo "  $0 -p \"山东最高的山是什么山？\"         # Single prompt"
    echo "  $0 -d cuda -t 0.8 -l 256            # Custom parameters"
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    # Check model path
    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${RED}❌ Model directory not found: $MODEL_PATH${NC}"
        exit 1
    fi
    
    # Check vocab file
    if [ ! -f "$VOCAB_PATH" ]; then
        echo -e "${RED}❌ Vocab file not found: $VOCAB_PATH${NC}"
        exit 1
    fi
    
    # Check inference script
    if [ ! -f "$INFERENCE_SCRIPT" ]; then
        echo -e "${RED}❌ Inference script not found: $INFERENCE_SCRIPT${NC}"
        exit 1
    fi
    
    # Check tokenizer
    TOKENIZER_PATH="/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b/llama8b/tokenizer.py"
    if [ ! -f "$TOKENIZER_PATH" ]; then
        echo -e "${RED}❌ Tokenizer not found: $TOKENIZER_PATH${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites check passed${NC}"
}

# Function to show model info
show_model_info() {
    echo -e "${BLUE}📋 Model Information:${NC}"
    echo "   Model Path: $MODEL_PATH"
    echo "   Vocab Path: $VOCAB_PATH"
    echo "   Device: $DEVICE"
    
    # Show model directory size
    if [ -d "$MODEL_PATH" ]; then
        MODEL_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
        echo "   Model Size: $MODEL_SIZE"
    fi
    
    # Show config info if available
    CONFIG_FILE="$MODEL_PATH/config.json"
    if [ -f "$CONFIG_FILE" ]; then
        echo "   Configuration:"
        python3 -c "
import json
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = json.load(f)
    print(f'     - Layers: {config.get(\"num_hidden_layers\", \"Unknown\")}')
    print(f'     - Hidden Size: {config.get(\"hidden_size\", \"Unknown\")}')
    print(f'     - Vocab Size: {config.get(\"vocab_size\", \"Unknown\")}')
    print(f'     - Data Type: {config.get(\"torch_dtype\", \"Unknown\")}')
except Exception as e:
    print('     - Error reading config:', str(e))
" 2>/dev/null
    fi
    echo ""
}

# Function to run demo prompts
run_demo() {
    echo -e "${YELLOW}🎯 Running demo mode with sample prompts...${NC}"
    
    DEMO_PROMPTS=(
        "山东最高的山是什么山？"
        "请介绍一下人工智能的发展历史"
        "如何制作一道简单的家常菜？"
        "解释一下什么是机器学习"
    )
    
    for i in "${!DEMO_PROMPTS[@]}"; do
        echo -e "\n${BLUE}Demo ${i+1}/${#DEMO_PROMPTS[@]}:${NC}"
        echo "Prompt: ${DEMO_PROMPTS[$i]}"
        echo "---"
        
        python3 "$INFERENCE_SCRIPT" \
            --model_path "$MODEL_PATH" \
            --vocab_path "$VOCAB_PATH" \
            --device "$DEVICE" \
            --prompt "${DEMO_PROMPTS[$i]}" \
            --max_length 256 \
            --temperature 0.7
        
        echo ""
        read -p "Press Enter to continue to next demo (or Ctrl+C to exit)..."
    done
}

# Function to run quick test
run_test() {
    echo -e "${YELLOW}🧪 Running quick test mode...${NC}"
    
    python3 "$INFERENCE_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --vocab_path "$VOCAB_PATH" \
        --device "$DEVICE" \
        --prompt "你好" \
        --max_length 50 \
        --temperature 0.7
}

# Parse command line arguments
MODEL_PATH_ARG=""
VOCAB_PATH_ARG=""
DEVICE="auto"
PROMPT=""
MAX_LENGTH=""
TEMPERATURE=""
DEMO_MODE=false
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH_ARG="$2"
            shift 2
            ;;
        -v|--vocab)
            VOCAB_PATH_ARG="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -l|--max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --demo)
            DEMO_MODE=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Use provided paths if specified
if [ -n "$MODEL_PATH_ARG" ]; then
    MODEL_PATH="$MODEL_PATH_ARG"
fi
if [ -n "$VOCAB_PATH_ARG" ]; then
    VOCAB_PATH="$VOCAB_PATH_ARG"
fi

# Check prerequisites
check_prerequisites

# Show model information
show_model_info

# Prepare inference command
CMD="python3 \"$INFERENCE_SCRIPT\" --model_path \"$MODEL_PATH\" --vocab_path \"$VOCAB_PATH\" --device \"$DEVICE\""

if [ -n "$PROMPT" ]; then
    CMD="$CMD --prompt \"$PROMPT\""
fi
if [ -n "$MAX_LENGTH" ]; then
    CMD="$CMD --max_length $MAX_LENGTH"
fi
if [ -n "$TEMPERATURE" ]; then
    CMD="$CMD --temperature $TEMPERATURE"
fi

# Execute based on mode
if [ "$DEMO_MODE" = true ]; then
    run_demo
elif [ "$TEST_MODE" = true ]; then
    run_test
else
    echo -e "${GREEN}🚀 Starting Llama8b inference...${NC}"
    echo "Command: $CMD"
    echo ""
    
    eval $CMD
fi

echo -e "\n${GREEN}🏁 Inference session completed!${NC}"
