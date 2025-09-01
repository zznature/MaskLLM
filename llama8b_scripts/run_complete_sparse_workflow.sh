#!/bin/bash

# Llama8b完整稀疏化工作流程自动化脚本
# 从模型转换到稀疏化训练的一键式解决方案

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

# 配置参数
SPARSE_METHOD=${1:-"SparseGPT"}  # SparseGPT, Magnitude, Wanda
RESUME_MODE=${2:-"0"}            # 0=完整流程，1=仅稀疏化，2=仅训练
DATA_FILES=${3:-"0-19"}          # 处理的C4文件范围

echo "🚀 Llama8b完整稀疏化工作流程"
echo "============================================================"
echo "配置参数:"
echo "  - 稀疏化方法: $SPARSE_METHOD"
echo "  - 执行模式: $RESUME_MODE (0=完整, 1=仅稀疏化, 2=仅训练)"
echo "  - 数据范围: C4文件 $DATA_FILES"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查函数
check_file() {
    if [ ! -f "$1" ]; then
        log_error "文件不存在: $1"
        return 1
    fi
    return 0
}

check_dir() {
    if [ ! -d "$1" ]; then
        log_error "目录不存在: $1"
        return 1
    fi
    return 0
}

# Phase 0: 环境检查
log_info "Phase 0: 环境检查"
echo "----------------------------------------"

# 检查关键脚本
REQUIRED_SCRIPTS=(
    "scripts/tools/convert_llama8b_hf_to_megatron_simple.sh"
    "llama8b_scripts/prepare_c4_megatron_llama8b.sh"
    "llama8b_scripts/run_llama8b_prune_tp8.sh"
    "llama8b_scripts/llama8b_presparse_training_tp8.sh"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if check_file "$script"; then
        log_success "脚本检查通过: $script"
    else
        exit 1
    fi
done

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    log_success "检测到 $GPU_COUNT 个GPU"
    if [[ $GPU_COUNT -lt 8 ]]; then
        log_warning "GPU数量少于8个，可能需要调整并行配置"
    fi
else
    log_warning "无法检测GPU，请确保CUDA环境正确"
fi

echo ""

# Phase 1: 模型转换 (如果需要)
if [[ $RESUME_MODE -eq 0 ]]; then
    log_info "Phase 1: 模型转换 (HF → Megatron)"
    echo "----------------------------------------"
    
    MEGATRON_CHECKPOINT="output/checkpoints/llama8b_megatron_tp8"
    
    if [ -d "$MEGATRON_CHECKPOINT" ]; then
        log_warning "Megatron checkpoint已存在，跳过转换"
    else
        log_info "开始模型转换..."
        if bash scripts/tools/convert_llama8b_hf_to_megatron_simple.sh 8; then
            log_success "模型转换完成"
        else
            log_error "模型转换失败"
            exit 1
        fi
    fi
    
    if check_dir "$MEGATRON_CHECKPOINT"; then
        log_success "Phase 1 完成: Megatron格式模型准备就绪"
    else
        exit 1
    fi
    echo ""
fi

# Phase 2: 数据预处理 (如果需要)
if [[ $RESUME_MODE -eq 0 ]]; then
    log_info "Phase 2: 数据预处理 (C4 → Llama8b Tokenized)"
    echo "----------------------------------------"
    
    DATA_PATH="assets/data/c4_llama8b_pretokenized"
    SAMPLE_FILE="${DATA_PATH}/c4_llama8b_00000_text_document.bin"
    
    if [ -f "$SAMPLE_FILE" ]; then
        log_warning "预处理数据已存在，跳过数据预处理"
    else
        log_info "开始数据预处理 (预计12-15分钟)..."
        log_info "处理C4文件范围: $DATA_FILES"
        
        if bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh; then
            log_success "数据预处理完成"
        else
            log_error "数据预处理失败"
            exit 1
        fi
    fi
    
    if check_file "$SAMPLE_FILE"; then
        log_success "Phase 2 完成: C4数据预处理就绪"
        
        # 显示数据统计
        DATA_COUNT=$(ls -1 "${DATA_PATH}"/c4_llama8b_*_text_document.bin 2>/dev/null | wc -l)
        log_info "已处理 $DATA_COUNT 个数据文件"
    else
        exit 1
    fi
    echo ""
fi

# Phase 3: 稀疏化准备 (如果需要)
if [[ $RESUME_MODE -le 1 ]]; then
    log_info "Phase 3: 稀疏化准备 (SparseGPT剪枝)"
    echo "----------------------------------------"
    
    SPARSE_CHECKPOINT="output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5${SPARSE_METHOD}.ex0"
    
    if [ -d "$SPARSE_CHECKPOINT" ]; then
        log_warning "稀疏化checkpoint已存在，跳过稀疏化"
    else
        log_info "开始稀疏化准备 (预计30-60分钟)..."
        log_info "使用方法: $SPARSE_METHOD"
        log_info "目标稀疏率: 50% (2:4结构化)"
        
        if bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh "$SPARSE_METHOD"; then
            log_success "稀疏化准备完成"
        else
            log_error "稀疏化准备失败"
            exit 1
        fi
    fi
    
    if check_dir "$SPARSE_CHECKPOINT"; then
        log_success "Phase 3 完成: 预稀疏模型准备就绪"
        log_info "稀疏checkpoint: $SPARSE_CHECKPOINT"
    else
        exit 1
    fi
    echo ""
fi

# Phase 4: 稀疏化训练 (如果需要)
if [[ $RESUME_MODE -le 2 ]]; then
    log_info "Phase 4: 稀疏化训练 (MaskLLM)"
    echo "----------------------------------------"
    
    TRAINING_CHECKPOINT="output/checkpoints/llama8b_maskllm_training_tp8"
    
    # 检查是否需要恢复训练
    TRAINING_MODE=0
    if [ -d "$TRAINING_CHECKPOINT" ]; then
        log_warning "发现已有训练checkpoint"
        read -p "是否从已有checkpoint恢复训练？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            TRAINING_MODE=1
            log_info "将从训练checkpoint恢复"
        else
            log_info "将从预稀疏checkpoint重新开始"
        fi
    fi
    
    log_info "开始稀疏化训练 (预计6-12小时)..."
    log_info "训练配置:"
    echo "  - 迭代次数: 10,000"
    echo "  - 批大小: 16 (global)"
    echo "  - 学习率: 1e-5 (模型), 0.1 (mask)"
    echo "  - 保存间隔: 1,000迭代"
    echo "  - 评估间隔: 500迭代"
    
    if bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh "$TRAINING_MODE"; then
        log_success "稀疏化训练完成"
    else
        log_error "稀疏化训练失败"
        exit 1
    fi
    
    if check_dir "$TRAINING_CHECKPOINT"; then
        log_success "Phase 4 完成: 稀疏化训练成功"
    else
        exit 1
    fi
    echo ""
fi

# 工作流程完成总结
echo "============================================================"
log_success "🎉 Llama8b稀疏化工作流程完成!"
echo "============================================================"

echo ""
echo "📊 结果总结:"
if [[ $RESUME_MODE -eq 0 ]]; then
    echo "✅ Phase 1: 模型转换完成"
    echo "✅ Phase 2: 数据预处理完成"
fi
if [[ $RESUME_MODE -le 1 ]]; then
    echo "✅ Phase 3: 稀疏化准备完成"
fi
if [[ $RESUME_MODE -le 2 ]]; then
    echo "✅ Phase 4: 稀疏化训练完成"
fi

echo ""
echo "📁 重要输出:"
echo "  - Megatron模型: output/checkpoints/llama8b_megatron_tp8/"
echo "  - 预处理数据: assets/data/c4_llama8b_pretokenized/"
echo "  - 预稀疏模型: output/oneshot_pruning/checkpoint/"
echo "  - 训练结果: output/checkpoints/llama8b_maskllm_training_tp8/"

echo ""
echo "🔄 下一步建议:"
echo "  1. 评估稀疏模型性能"
echo "  2. 调优训练超参数"
echo "  3. 部署模型到推理环境"

echo ""
echo "📚 参考文档:"
echo "  - 详细指南: llama8b_scripts/PHASE3_5_SPARSE_WORKFLOW_GUIDE.md"
echo "  - 进度记录: llama8b_scripts/PHASE3_PROGRESS_SUMMARY.md"
echo "  - 项目文档: Project_SelfDefinedModels.md"

echo ""
log_success "工作流程执行完毕! 🚀"
