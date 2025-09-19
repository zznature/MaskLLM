#!/bin/bash
# Llama8b SafeTensors FP32 to FP16 Conversion Script
# This script checks and converts SafeTensors model files from FP32 to FP16

set -e  # Exit on any error

# Configuration
MODEL_DIR="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4"
SCRIPTS_DIR="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts"
LOG_FILE="${SCRIPTS_DIR}/fp16_conversion_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Llama8b SafeTensors FP16 Conversion  ${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to log with timestamp
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}❌ Model directory not found: $MODEL_DIR${NC}"
    exit 1
fi

log "${GREEN}🚀 Starting SafeTensors FP16 conversion process${NC}"
log "📁 Model directory: $MODEL_DIR"
log "📝 Log file: $LOG_FILE"

# Step 1: Check model datatypes
log "${YELLOW}📊 Step 1: Checking model datatypes...${NC}"
echo "Running datatype analysis..."

if python3 "${SCRIPTS_DIR}/check_model_datatypes.py" 2>&1 | tee -a "$LOG_FILE"; then
    log "${GREEN}✅ Datatype check completed successfully${NC}"
else
    log "${RED}❌ Datatype check failed${NC}"
    exit 1
fi

# Check if conversion is needed based on analysis result
if [ -f "${SCRIPTS_DIR}/model_dtype_analysis.json" ]; then
    RECOMMENDATION=$(python3 -c "
import json
with open('${SCRIPTS_DIR}/model_dtype_analysis.json', 'r') as f:
    data = json.load(f)
print(data.get('recommendation', 'unknown'))
")
    
    if [ "$RECOMMENDATION" = "convert_to_fp16" ]; then
        log "${YELLOW}🔄 Step 2: FP32 detected, starting conversion to FP16...${NC}"
        echo ""
        echo "=== CONVERSION PROCESS ==="
        echo "This will:"
        echo "• Convert all FP32 tensors to FP16"
        echo "• Update model configuration"
        echo "• Create backups of original files"
        echo "• Reduce model size by ~50%"
        echo ""
        
        # Perform conversion
        if python3 "${SCRIPTS_DIR}/convert_safetensors_fp32_to_fp16.py" -i "$MODEL_DIR" 2>&1 | tee -a "$LOG_FILE"; then
            log "${GREEN}🎉 Conversion completed successfully!${NC}"
            
            # Show before/after comparison
            echo ""
            echo "=== CONVERSION RESULTS ==="
            if [ -f "${SCRIPTS_DIR}/conversion_report.json" ]; then
                python3 -c "
import json
with open('${SCRIPTS_DIR}/conversion_report.json', 'r') as f:
    report = json.load(f)
stats = report['conversion_stats']
print(f'Files converted: {stats[\"files_converted\"]}')
print(f'Tensors converted: {stats[\"tensors_converted\"]}')
print(f'Original size: {stats[\"original_size\"]/(1024**3):.2f} GB')
print(f'New size: {stats[\"new_size\"]/(1024**3):.2f} GB')
print(f'Space saved: {(stats[\"original_size\"]-stats[\"new_size\"])/(1024**3):.2f} GB')
print(f'Compression ratio: {(1-stats[\"new_size\"]/stats[\"original_size\"])*100:.1f}%')
"
            fi
        else
            log "${RED}❌ Conversion failed${NC}"
            exit 1
        fi
        
    elif [ "$RECOMMENDATION" = "already_optimized" ]; then
        log "${GREEN}✅ Model is already optimized (FP16), no conversion needed${NC}"
        echo "Current model is already in FP16 format."
        
    else
        log "${YELLOW}⚠️  Unknown recommendation: $RECOMMENDATION${NC}"
        echo "Unable to determine if conversion is needed."
    fi
else
    log "${RED}❌ Analysis result file not found${NC}"
    exit 1
fi

# Step 3: Final verification
log "${YELLOW}🔍 Step 3: Final verification...${NC}"

# Check final model size and type
if [ -f "$MODEL_DIR/config.json" ]; then
    TORCH_DTYPE=$(python3 -c "
import json
with open('$MODEL_DIR/config.json', 'r') as f:
    config = json.load(f)
print(config.get('torch_dtype', 'unknown'))
")
    log "📋 Final model torch_dtype: $TORCH_DTYPE"
fi

# Calculate total model size
TOTAL_SIZE=$(du -sb "$MODEL_DIR" | cut -f1)
TOTAL_SIZE_GB=$(python3 -c "print(f'{$TOTAL_SIZE/(1024**3):.2f}')")
log "📊 Final model directory size: ${TOTAL_SIZE_GB} GB"

log "${GREEN}🏁 Process completed successfully!${NC}"
echo ""
echo "=== SUMMARY ==="
echo "• Log file: $LOG_FILE"
echo "• Analysis report: ${SCRIPTS_DIR}/model_dtype_analysis.json"
echo "• Conversion report: ${SCRIPTS_DIR}/conversion_report.json (if converted)"
echo "• Model directory: $MODEL_DIR"
echo "• Final model size: ${TOTAL_SIZE_GB} GB"

echo ""
echo -e "${GREEN}✨ All operations completed successfully!${NC}"
