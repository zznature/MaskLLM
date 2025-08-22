#!/bin/bash

# Test script for large vocabulary compatibility
# This script tests the Llama8b large vocabulary (119696) compatibility with Megatron

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Llama8b Large Vocabulary Compatibility Test ===${NC}"
echo ""

# Set default paths
TOKENIZER_PATH="assets/checkpoints/Llama8b"
TEST_SCRIPT="llama8b_scripts/test_large_vocab_compatibility.py"

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

echo -e "${GREEN}Testing large vocabulary compatibility...${NC}"
echo "  Tokenizer path: $TOKENIZER_PATH"
echo "  Expected vocab size: ~119,696"
echo ""

# Run the test using the correct python path
python3.10 "$TEST_SCRIPT" --tokenizer-path "$TOKENIZER_PATH"
TEST_EXIT_CODE=$?

echo ""
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}🎉 Large vocabulary compatibility test completed successfully!${NC}"
    echo ""
    echo -e "${GREEN}Key achievements:${NC}"
    echo "✅ Large vocabulary (119,696) correctly handled"
    echo "✅ Vocabulary padding works with different tensor parallel sizes"
    echo "✅ Memory usage is acceptable"
    echo "✅ Edge cases handled properly"
    echo "✅ Megatron interface compatibility verified"
    echo ""
    echo -e "${GREEN}Phase 1 Complete! 🎉${NC}"
    echo "All tokenizer integration tasks completed successfully."
    echo ""
    echo -e "${GREEN}Ready for Phase 2: Checkpoint Conversion${NC}"
else
    echo -e "${RED}❌ Large vocabulary compatibility test failed${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo "1. Check memory availability (large vocab requires more memory)"
    echo "2. Verify tokenizer files are complete and not corrupted"
    echo "3. Check for any tensor parallel configuration issues"
    echo "4. Review error messages above for specific failure points"
fi

exit $TEST_EXIT_CODE
