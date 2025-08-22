#!/bin/bash

# Test script for Llama8b tokenizer integration
# This script tests the Llama8b tokenizer integration with Megatron

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Llama8b Tokenizer Integration Test ===${NC}"
echo ""

# Set default paths
TOKENIZER_PATH="assets/checkpoints/Llama8b"
TEST_SCRIPT="llama8b_scripts/test_llama8b_tokenizer_integration.py"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tokenizer-path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--tokenizer-path PATH]"
            echo ""
            echo "Options:"
            echo "  --tokenizer-path PATH    Path to Llama8b tokenizer directory (default: assets/checkpoints/Llama8b)"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [[ ! -f "megatron/tokenizer/tokenizer.py" ]]; then
    echo -e "${RED}Error: Please run this script from MaskLLM root directory${NC}"
    exit 1
fi

# Check if tokenizer path exists
if [[ ! -d "$TOKENIZER_PATH" ]]; then
    echo -e "${RED}Error: Tokenizer path does not exist: $TOKENIZER_PATH${NC}"
    echo -e "${YELLOW}Make sure you have downloaded the Llama8b model to the correct location${NC}"
    exit 1
fi

# Check essential files
VOCAB_FILE="$TOKENIZER_PATH/vocab.txt"
LLAMA8B_VOCAB_FILE="$TOKENIZER_PATH/llama8b/vocab.txt"
TOKENIZER_MODULE="$TOKENIZER_PATH/llama8b/tokenizer.py"

if [[ ! -f "$VOCAB_FILE" ]] && [[ ! -f "$LLAMA8B_VOCAB_FILE" ]]; then
    echo -e "${RED}Error: vocab.txt not found in $TOKENIZER_PATH or $TOKENIZER_PATH/llama8b/${NC}"
    exit 1
fi

if [[ ! -f "$TOKENIZER_MODULE" ]]; then
    echo -e "${RED}Error: tokenizer.py not found at $TOKENIZER_MODULE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Tokenizer files verified${NC}"
echo "  Tokenizer path: $TOKENIZER_PATH"
if [[ -f "$VOCAB_FILE" ]]; then
    echo "  Vocab file: $VOCAB_FILE"
    VOCAB_SIZE=$(wc -l < "$VOCAB_FILE" 2>/dev/null || echo "unknown")
    echo "  Vocab size: $VOCAB_SIZE"
elif [[ -f "$LLAMA8B_VOCAB_FILE" ]]; then
    echo "  Vocab file: $LLAMA8B_VOCAB_FILE"
    VOCAB_SIZE=$(wc -l < "$LLAMA8B_VOCAB_FILE" 2>/dev/null || echo "unknown")
    echo "  Vocab size: $VOCAB_SIZE"
fi
echo ""

# Run the test
echo -e "${GREEN}Running tokenizer integration test...${NC}"
echo ""

python3.10 "$TEST_SCRIPT" --tokenizer-path "$TOKENIZER_PATH"
TEST_EXIT_CODE=$?

echo ""
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}🎉 Tokenizer integration test completed successfully!${NC}"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Run phase 2: Checkpoint conversion"
    echo "2. Test model loading with the new tokenizer"
    echo "3. Test sparsity training compatibility"
else
    echo -e "${RED}❌ Tokenizer integration test failed${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "1. Check that all required files are present"
    echo "2. Verify Python environment has required packages"
    echo "3. Check tokenizer implementation for compatibility issues"
fi

exit $TEST_EXIT_CODE
