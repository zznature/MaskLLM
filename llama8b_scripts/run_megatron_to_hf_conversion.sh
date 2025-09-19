#!/bin/bash
# Llama8b Megatron→HuggingFace 转换启动脚本

set -e  # Exit on any error

# 配置路径
MEGATRON_CHECKPOINT="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt"
OUTPUT_DIR="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4_from_megatron"
REFERENCE_MODEL="/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b"
CONVERSION_SCRIPT="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/convert_megatron_to_hf.py"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}  Llama8b Megatron→HF 转换工具  ${NC}"
echo -e "${BLUE}=================================${NC}"

# 检查先决条件
echo -e "${YELLOW}🔍 检查先决条件...${NC}"

# 检查Python环境
python3 -c "
import sys
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
    
    from tqdm import tqdm
    print('✅ tqdm: 可用')
    
except ImportError as e:
    print(f'❌ 缺少依赖: {e}')
    sys.exit(1)
" || {
    echo -e "${RED}❌ Python环境检查失败${NC}"
    exit 1
}

# 检查输入文件
echo -e "${YELLOW}📁 检查输入文件...${NC}"

if [ ! -f "$MEGATRON_CHECKPOINT" ]; then
    echo -e "${RED}❌ Megatron检查点不存在: $MEGATRON_CHECKPOINT${NC}"
    exit 1
else
    SIZE=$(du -sh "$MEGATRON_CHECKPOINT" | cut -f1)
    echo -e "${GREEN}✅ Megatron检查点: $SIZE${NC}"
fi

if [ ! -d "$REFERENCE_MODEL" ]; then
    echo -e "${RED}❌ 参考模型目录不存在: $REFERENCE_MODEL${NC}"
    exit 1
else
    SIZE=$(du -sh "$REFERENCE_MODEL" | cut -f1)
    echo -e "${GREEN}✅ 参考模型: $SIZE${NC}"
fi

if [ ! -f "$CONVERSION_SCRIPT" ]; then
    echo -e "${RED}❌ 转换脚本不存在: $CONVERSION_SCRIPT${NC}"
    exit 1
else
    echo -e "${GREEN}✅ 转换脚本: 就绪${NC}"
fi

# 检查磁盘空间
echo -e "${YELLOW}💾 检查磁盘空间...${NC}"
OUTPUT_DIR_PARENT=$(dirname "$OUTPUT_DIR")
AVAILABLE_SPACE=$(df -h "$OUTPUT_DIR_PARENT" | tail -1 | awk '{print $4}')
echo -e "${GREEN}✅ 可用空间: $AVAILABLE_SPACE${NC}"

# 显示转换计划
echo ""
echo -e "${BLUE}📋 转换计划:${NC}"
echo "  输入: Megatron FP16 检查点 (17GB)"
echo "  输出: HuggingFace单文件格式 (~17GB)"
echo "  格式: pytorch_model.bin + 配置文件"
echo "  兼容: transformers 4.40.0"
echo ""

# 确认执行
echo -e "${YELLOW}⚠️  此操作将创建约17GB的新文件${NC}"
echo -e "${YELLOW}📁 输出目录: $OUTPUT_DIR${NC}"

# 检查输出目录是否存在
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}⚠️  输出目录已存在，将被覆盖${NC}"
fi

echo ""
read -p "是否继续执行转换？[y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}转换已取消${NC}"
    exit 0
fi

# 开始转换
echo ""
echo -e "${GREEN}🚀 开始Megatron→HuggingFace转换...${NC}"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 执行转换
bash run_maskllm_native.sh "$CONVERSION_SCRIPT" \
    --megatron_checkpoint "$MEGATRON_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --reference_model "$REFERENCE_MODEL"

# 检查转换结果
if [ $? -eq 0 ]; then
    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo -e "${GREEN}🎉 转换成功完成!${NC}"
    echo -e "${GREEN}⏱️  耗时: ${MINUTES}分${SECONDS}秒${NC}"
    
    # 验证输出
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        echo -e "${BLUE}📁 输出内容验证:${NC}"
        
        # 检查关键文件
        KEY_FILES=("pytorch_model.bin" "config.json" "vocab.txt")
        for file in "${KEY_FILES[@]}"; do
            if [ -f "$OUTPUT_DIR/$file" ]; then
                SIZE=$(du -sh "$OUTPUT_DIR/$file" | cut -f1)
                echo -e "${GREEN}  ✅ $file ($SIZE)${NC}"
            else
                echo -e "${RED}  ❌ $file (缺失)${NC}"
            fi
        done
        
        # 检查tokenizer目录
        if [ -d "$OUTPUT_DIR/llama8b" ]; then
            echo -e "${GREEN}  ✅ llama8b/ (tokenizer目录)${NC}"
        else
            echo -e "${RED}  ❌ llama8b/ (缺失)${NC}"
        fi
        
        # 显示总大小
        TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
        echo -e "${BLUE}💾 总大小: $TOTAL_SIZE${NC}"
        
        echo ""
        echo -e "${GREEN}🎯 现在可以使用转换后的模型了:${NC}"
        echo ""
        echo "# 测试模型加载"
        echo "bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/llama8b_hf_inference_simple.py \\"
        echo "    --model_path $OUTPUT_DIR \\"
        echo "    --prompt \"山东最高的山是什么山？\""
        echo ""
    else
        echo -e "${RED}❌ 输出目录未创建${NC}"
    fi
    
else
    echo ""
    echo -e "${RED}❌ 转换失败${NC}"
    echo -e "${YELLOW}💡 请检查错误信息并重试${NC}"
    exit 1
fi
