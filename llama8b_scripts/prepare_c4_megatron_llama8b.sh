# C4数据集转换为Llama8b Megatron格式 - CPU优化版
# 系统配置: 192 逻辑CPU (96物理核心) Intel Xeon Platinum 8558
# 优化workers数量以充分利用CPU资源

echo "🚀 Llama8b C4数据预处理 - CPU优化版"
echo "======================================="
echo "📊 系统信息: $(nproc) 逻辑CPU, $(nproc --ignore=2) 物理核心"
echo "⚡ 优化配置: 高性能并行处理"

# 动态计算最优workers数量
LOGICAL_CPUS=$(nproc)
PHYSICAL_CORES=$((LOGICAL_CPUS / 2))

# 设置workers为物理核心数的75% (避免资源饱和)
WORKERS=$((PHYSICAL_CORES * 3 / 4))

# 确保workers在合理范围内 (最小32, 最大128)
if [ $WORKERS -lt 32 ]; then
    WORKERS=32
elif [ $WORKERS -gt 128 ]; then
    WORKERS=128
fi

echo "💻 计算结果: 逻辑CPU=${LOGICAL_CPUS}, 物理核心=${PHYSICAL_CORES}"
echo "⚙️  优化配置: workers=${WORKERS} (物理核心的75%)"

# 创建输出目录
mkdir -p assets/data/c4_llama8b_pretokenized

# 记录开始时间
START_TIME=$(date +%s)
echo "🕐 开始时间: $(date)"

# 处理C4数据集文件 - 默认处理前20个文件 (00000-00019)
for i in {00000..00019}; do
       echo ""
       echo "📄 处理文件 ${i} ($(echo $i | sed 's/^0*//')}/20)..."
       echo "⚡ 使用 ${WORKERS} 个workers进行并行处理"
       
       python3 tools/preprocess_data.py \
              --input "./assets/data/c4/en/c4-train.${i}-of-01024.json" \
              --output-prefix assets/data/c4_llama8b_pretokenized/c4_llama8b_${i} \
              --vocab-file ./assets/checkpoints/Llama8b/vocab.txt \
              --tokenizer-type Llama8bTokenizer \
              --tokenizer-model ./assets/checkpoints/Llama8b \
              --append-eod \
              --workers $WORKERS
       
       if [ $? -eq 0 ]; then
           echo "✅ 文件 ${i} 处理完成"
       else
           echo "❌ 文件 ${i} 处理失败"
           exit 1
       fi
done

# 计算总用时
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "🎉 所有文件处理完成！"
echo "⏱️  总用时: ${MINUTES}分${SECONDS}秒"
echo "📊 平均每文件: $((TOTAL_TIME / 20))秒"
echo "💻 CPU优化效果: 使用${WORKERS}个workers充分利用${LOGICAL_CPUS}个CPU"
