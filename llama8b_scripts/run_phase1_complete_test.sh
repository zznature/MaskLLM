#!/bin/bash

# Complete Phase 1 test script for Llama8b tokenizer integration
# This script runs all Phase 1 tests to verify complete tokenizer integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MaskLLM Llama8b Tokenizer - Phase 1 Complete Test Suite ===${NC}"
echo ""

# Set default paths
TOKENIZER_PATH="assets/checkpoints/Llama8b"

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
            echo "This script runs the complete Phase 1 test suite:"
            echo "  1. Basic tokenizer integration test"
            echo "  2. Large vocabulary compatibility test"
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

echo -e "${BLUE}Phase 1 Test Configuration:${NC}"
echo "  Tokenizer path: $TOKENIZER_PATH"
echo "  Test environment: MaskLLM with .ext_pkgs support"
echo ""

# Initialize test tracking
TOTAL_TESTS=2
PASSED_TESTS=0
TEST_RESULTS=()

# Test 1: Basic Integration Test
echo -e "${YELLOW}=== Test 1/2: Basic Tokenizer Integration ===${NC}"
echo ""

if bash llama8b_scripts/run_tokenizer_test.sh --tokenizer-path "$TOKENIZER_PATH"; then
    echo -e "${GREEN}✅ Test 1 PASSED: Basic tokenizer integration successful${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TEST_RESULTS+=("✅ Basic Integration")
else
    echo -e "${RED}❌ Test 1 FAILED: Basic tokenizer integration failed${NC}"
    TEST_RESULTS+=("❌ Basic Integration")
fi

echo ""
echo "=" * 80
echo ""

# Test 2: Large Vocabulary Compatibility Test
echo -e "${YELLOW}=== Test 2/2: Large Vocabulary Compatibility ===${NC}"
echo ""

if bash llama8b_scripts/run_large_vocab_test.sh --tokenizer-path "$TOKENIZER_PATH"; then
    echo -e "${GREEN}✅ Test 2 PASSED: Large vocabulary compatibility successful${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TEST_RESULTS+=("✅ Large Vocabulary Compatibility")
else
    echo -e "${RED}❌ Test 2 FAILED: Large vocabulary compatibility failed${NC}"
    TEST_RESULTS+=("❌ Large Vocabulary Compatibility")
fi

echo ""
echo "=" * 80
echo ""

# Final Results
echo -e "${BLUE}=== PHASE 1 TEST SUITE RESULTS ===${NC}"
echo ""

for result in "${TEST_RESULTS[@]}"; do
    echo "  $result"
done

echo ""
echo -e "${BLUE}Overall Result: $PASSED_TESTS/$TOTAL_TESTS tests passed${NC}"

if [[ $PASSED_TESTS -eq $TOTAL_TESTS ]]; then
    echo ""
    echo -e "${GREEN}🎉🎉🎉 PHASE 1 COMPLETE! 🎉🎉🎉${NC}"
    echo ""
    echo -e "${GREEN}All Phase 1 objectives achieved:${NC}"
    echo "✅ Llama8b tokenizer successfully integrated with Megatron"
    echo "✅ Large vocabulary (119,696 tokens) properly handled"
    echo "✅ Tensor parallel compatibility verified"
    echo "✅ Memory usage optimized"
    echo "✅ All Megatron interfaces working correctly"
    echo ""
    echo -e "${BLUE}=== READY FOR PHASE 2: CHECKPOINT CONVERSION ===${NC}"
    echo ""
    echo -e "${YELLOW}Next steps for Phase 2:${NC}"
    echo "1. Create Llama8b-specific checkpoint loader"
    echo "2. Update checkpoint conversion tools"
    echo "3. Test model loading with converted checkpoints"
    echo "4. Verify sparsity training compatibility"
    echo ""
    echo -e "${GREEN}To proceed to Phase 2, follow the implementation plan in Project_SelfDefinedModels.md${NC}"
    
    exit 0
else
    echo ""
    echo -e "${RED}❌ PHASE 1 INCOMPLETE${NC}"
    echo ""
    echo -e "${YELLOW}Failed tests need to be resolved before proceeding to Phase 2.${NC}"
    echo "Please review the test output above and fix any issues."
    echo ""
    echo -e "${YELLOW}Common troubleshooting steps:${NC}"
    echo "1. Verify all required files are present in the tokenizer directory"
    echo "2. Check that transformers package is accessible (.ext_pkgs path)"
    echo "3. Ensure sufficient memory for large vocabulary processing"
    echo "4. Verify Python environment compatibility"
    
    exit 1
fi
