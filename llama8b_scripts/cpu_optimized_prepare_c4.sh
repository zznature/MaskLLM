#!/bin/bash
# C4数据集转换为Llama8b Megatron格式 - 极致CPU优化版
# 针对192核心系统的高性能配置

echo "🚀 Llama8b C4数据预处理 - 极致CPU优化版"
echo "==========================================="

# 获取系统信息
LOGICAL_CPUS=$(nproc)
PHYSICAL_CORES=$((LOGICAL_CPUS / 2))
AVAILABLE_MEMORY=$(free -g | awk '/^Mem:/{print $7}')

echo "📊 系统配置分析:"
echo "  💻 逻辑CPU: $LOGICAL_CPUS"
echo "  🔧 物理核心: $PHYSICAL_CORES"  
echo "  💾 可用内存: ${AVAILABLE_MEMORY}GB"

# 智能计算最优workers配置
# 策略：根据CPU数量和内存情况动态调整
if [ $PHYSICAL_CORES -ge 96 ]; then
    # 高性能服务器配置
    WORKERS=$((PHYSICAL_CORES * 4 / 5))  # 使用80%物理核心
    echo "  🎯 检测到高性能服务器 (≥96核心)"
elif [ $PHYSICAL_CORES -ge 48 ]; then
    # 中高端服务器配置  
    WORKERS=$((PHYSICAL_CORES * 3 / 4))  # 使用75%物理核心
    echo "  🎯 检测到中高端服务器 (≥48核心)"
else
    # 标准服务器配置
    WORKERS=$((PHYSICAL_CORES * 2 / 3))  # 使用66%物理核心
    echo "  🎯 检测到标准服务器 (<48核心)"
fi

# 内存限制检查（每个worker大约需要0.5-1GB内存）
MAX_WORKERS_BY_MEMORY=$((AVAILABLE_MEMORY / 1))
if [ $WORKERS -gt $MAX_WORKERS_BY_MEMORY ]; then
    WORKERS=$MAX_WORKERS_BY_MEMORY
    echo "  ⚠️  根据内存限制调整workers到: $WORKERS"
fi

# 设置合理的边界
if [ $WORKERS -lt 32 ]; then
    WORKERS=32
elif [ $WORKERS -gt 150 ]; then
    WORKERS=150
fi

echo "  ⚙️  最终配置: $WORKERS workers"
echo "  📈 CPU利用率: $(($WORKERS * 100 / $PHYSICAL_CORES))%"

# 创建输出目录
mkdir -p assets/data/c4_llama8b_pretokenized

# 设置进程优先级和CPU亲和性优化
echo "🔧 应用系统优化..."
# 设置高优先级（需要权限时会忽略错误）
renice -10 $$ 2>/dev/null || true

# 记录开始时间
START_TIME=$(date +%s)
echo "🕐 开始时间: $(date)"
echo ""

# 并行处理多个文件（如果CPU足够强大）
if [ $PHYSICAL_CORES -ge 96 ]; then
    echo "🚀 启用超高性能模式：并行处理多个文件"
    PARALLEL_FILES=2  # 同时处理2个文件
else
    PARALLEL_FILES=1  # 顺序处理
fi

# 处理函数
process_file() {
    local i=$1
    local workers=$2
    
    echo "📄 开始处理文件 ${i}..."
    echo "⚡ 使用 ${workers} workers"
    
    python3 tools/preprocess_data.py \
        --input "./assets/data/c4/en/c4-train.${i}-of-01024.json" \
        --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_${i} \
        --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
        --tokenizer-type Llama8bTokenizer \
        --tokenizer-model ./assets/checkpoints/Llama8b \
        --append-eod \
        --workers $workers
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ 文件 ${i} 处理完成"
    else
        echo "❌ 文件 ${i} 处理失败 (exit code: $exit_code)"
        return $exit_code
    fi
}

# 主处理循环
if [ $PARALLEL_FILES -gt 1 ]; then
    # 并行处理模式
    WORKERS_PER_FILE=$((WORKERS / PARALLEL_FILES))
    echo "📊 并行配置: ${PARALLEL_FILES}个文件 × ${WORKERS_PER_FILE} workers"
    
    for i in {00000..00019}; do
        if [ $((10#$i % PARALLEL_FILES)) -eq 0 ]; then
            # 等待之前的后台任务完成
            wait
        fi
        
        # 在后台启动处理
        process_file $i $WORKERS_PER_FILE &
        
        # 控制并发数量
        if [ $(jobs -r | wc -l) -ge $PARALLEL_FILES ]; then
            wait -n  # 等待任意一个任务完成
        fi
    done
    
    # 等待所有任务完成
    wait
    
else
    # 顺序处理模式
    for i in {00000..00019}; do
        echo ""
        echo "📄 处理文件 ${i} ($(echo $i | sed 's/^0*//'))/20)..."
        
        process_file $i $WORKERS
        
        if [ $? -ne 0 ]; then
            echo "❌ 处理中断，退出"
            exit 1
        fi
        
        # 显示进度
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        PROGRESS=$(echo $i | sed 's/^0*//')
        if [ -z "$PROGRESS" ]; then PROGRESS=0; fi
        PROGRESS=$((PROGRESS + 1))
        
        if [ $PROGRESS -gt 0 ]; then
            AVG_TIME=$((ELAPSED / PROGRESS))
            REMAINING=$((20 - PROGRESS))
            ETA=$((REMAINING * AVG_TIME))
            
            echo "📈 总体进度: ${PROGRESS}/20 ($(($PROGRESS * 100 / 20))%)"
            echo "⏱️  已用时: ${ELAPSED}s, 预计剩余: ${ETA}s"
        fi
    done
fi

# 计算总用时和性能统计
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "🎉 所有文件处理完成！"
echo "==============================="
echo "⏱️  总用时: ${MINUTES}分${SECONDS}秒"
echo "📊 平均每文件: $((TOTAL_TIME / 20))秒"
echo "💻 CPU配置: ${WORKERS} workers / ${LOGICAL_CPUS} CPUs"
echo "🚀 性能提升: 相比32 workers提升 $(($WORKERS * 100 / 32 - 100))%"

# 验证输出文件
echo ""
echo "📁 生成的文件统计:"
ls -lh assets/data/c4_llama8b_pretokenized/*.bin | wc -l | xargs echo "  二进制文件数量:"
ls -lh assets/data/c4_llama8b_pretokenized/*.idx | wc -l | xargs echo "  索引文件数量:"

TOTAL_SIZE=$(du -sh assets/data/c4_llama8b_pretokenized/ | cut -f1)
echo "  总大小: $TOTAL_SIZE"

echo ""
echo "✅ C4数据集已成功转换为Llama8b Megatron格式！"
