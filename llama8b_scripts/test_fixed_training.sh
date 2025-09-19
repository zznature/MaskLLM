#!/bin/bash

# 测试修复后的Llama8b稀疏训练脚本
# 使用与验证的llama2脚本相同的成功模式

echo "🧪 测试修复后的Llama8b稀疏训练"
echo "=========================================="

PROJECT_DIR=$(pwd)

echo "📋 修复总结:"
echo "  ✅ 兼容性问题: 添加--finetune --enable-partial-load"
echo "  ✅ 数据格式: 采用验证的DATA_BLEND格式"
echo "  ✅ 训练参数: 使用验证的llama2超参数"
echo "  ✅ 稀疏参数: 使用验证的Gumbel配置"

echo ""
echo "🔍 关键修复验证:"

# 1. 检查预稀疏checkpoint
SPARSE_CHECKPOINT="output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0"
if [ -d "$SPARSE_CHECKPOINT" ]; then
    echo "✅ 预稀疏checkpoint存在"
else
    echo "❌ 需要先运行稀疏化脚本"
    echo "bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh SparseGPT"
    exit 1
fi

# 2. 检查数据文件
C4_HOME="assets/data/c4_llama8b_pretokenized"
SAMPLE_FILE="${C4_HOME}/c4_llama8b_00000_text_document.bin"
if [ -f "$SAMPLE_FILE" ]; then
    echo "✅ C4预处理数据存在"
    echo "📊 数据文件数量: $(ls ${C4_HOME}/c4_llama8b_*_text_document.bin 2>/dev/null | wc -l)"
else
    echo "❌ 需要先运行数据预处理"
    exit 1
fi

echo ""
echo "🚀 启动测试训练 (短时间验证):"
echo "----------------------------------------"

# 创建测试用的短期训练脚本
cat > llama8b_scripts/test_training_short.sh << 'EOF'
#!/bin/bash

# 短期测试训练 - 只训练10步验证兼容性
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="45533"
NNODES=1
NPROC_PER_NODE=8

RESUME_FLAG=0
EXTRA_CMD="--train-iters 10 --save-interval 5 --eval-interval 5"

# 调用主训练脚本但覆盖为短期测试
bash llama8b_scripts/llama8b_presparse_training_tp8.sh $RESUME_FLAG "$EXTRA_CMD"
EOF

chmod +x llama8b_scripts/test_training_short.sh

echo ""
echo "📝 测试命令:"
echo "bash run_maskllm_native.sh llama8b_scripts/test_training_short.sh"
echo ""
echo "⏱️  预期结果:"
echo "  ✅ 成功加载OneShot checkpoint"
echo "  ✅ 自动初始化diff_mask参数" 
echo "  ✅ 运行10步训练验证兼容性"
echo "  ✅ 不再出现Missing key错误"

echo ""
echo "🎯 如果测试成功，则运行完整训练:"
echo "bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh"
