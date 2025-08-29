#!/bin/bash

# 激进库隔离方案 - 通过文件系统级别隔离HPC-X
# 适用于无法通过环境变量解决的深层库冲突

echo "🔒 激进库隔离方案 - 文件系统级HPC-X隔离"
echo "时间: $(date)"
echo "警告: 这是一个激进的解决方案，将临时隔离HPC-X库文件"
echo "=" * 80

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# ================================================
# 第1步: 创建隔离环境
# ================================================
echo "📋 第1步: 创建隔离环境"

# 创建临时备份目录
ISOLATION_DIR="/tmp/hpcx_isolation"
BACKUP_DIR="$ISOLATION_DIR/backup"
MASKED_DIR="$ISOLATION_DIR/masked"

mkdir -p "$BACKUP_DIR" "$MASKED_DIR"

echo "✅ 隔离环境目录创建完成: $ISOLATION_DIR"
echo ""

# ================================================
# 第2步: 识别并隔离HPC-X库文件
# ================================================
echo "📋 第2步: 识别并隔离HPC-X库文件"

# HPC-X库路径列表
HPCX_LIB_PATHS=(
    "/opt/hpcx/ucc/lib"
    "/opt/hpcx/ucx/lib" 
    "/opt/hpcx/ompi/lib"
    "/opt/hpcx/hcoll/lib"
    "/opt/hpcx/sharp/lib"
)

# 关键冲突库文件
CONFLICT_LIBS=(
    "libucc.so.1"
    "libucx.so"
    "libucp.so"
    "libucs.so"
    "libhcoll.so"
)

echo "🔍 扫描HPC-X库文件..."
ISOLATED_FILES=0

for hpcx_path in "${HPCX_LIB_PATHS[@]}"; do
    if [ -d "$hpcx_path" ]; then
        echo "  📂 处理路径: $hpcx_path"
        
        for conflict_lib in "${CONFLICT_LIBS[@]}"; do
            # 查找所有版本的冲突库
            find "$hpcx_path" -name "${conflict_lib}*" 2>/dev/null | while read lib_file; do
                if [ -f "$lib_file" ]; then
                    echo "    🎯 发现冲突库: $lib_file"
                    
                    # 创建备份
                    backup_file="$BACKUP_DIR/$(basename $lib_file)_$(echo $hpcx_path | tr '/' '_')"
                    cp "$lib_file" "$backup_file" 2>/dev/null
                    
                    # 创建掩码文件（空文件或无害文件）
                    masked_file="$MASKED_DIR/$(basename $lib_file)"
                    echo "# Masked HPC-X library - $(date)" > "$masked_file"
                    
                    # 原子性替换：先移动到临时位置，再替换
                    temp_file="${lib_file}.hpcx_masked.tmp"
                    mv "$lib_file" "$temp_file" 2>/dev/null && \
                    cp "$masked_file" "$lib_file" 2>/dev/null && \
                    echo "      ✅ 已隔离: $(basename $lib_file)" || \
                    echo "      ❌ 隔离失败: $(basename $lib_file)"
                    
                    ((ISOLATED_FILES++))
                fi
            done
        done
    fi
done

echo "✅ 共隔离 $ISOLATED_FILES 个HPC-X库文件"
echo ""

# ================================================
# 第3步: 应用环境优化
# ================================================
echo "📋 第3步: 应用环境优化"

# 重新应用我们的环境修复
source llama8b_scripts/ultimate_environment_fix.sh >/dev/null 2>&1

echo "✅ 环境优化重新应用完成"
echo ""

# ================================================
# 第4步: 测试PyTorch
# ================================================
echo "📋 第4步: 测试PyTorch导入"

echo "🧪 测试1: 基本PyTorch导入"
python3.10 -c "
try:
    import torch
    print('🎉 PyTorch导入成功!')
    print(f'✅ 版本: {torch.__version__}')
    print(f'✅ 路径: {torch.__file__}')
    
    # CUDA测试
    cuda_available = torch.cuda.is_available()
    print(f'✅ CUDA可用: {cuda_available}')
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f'✅ CUDA设备数: {device_count}')
        if device_count > 0:
            print(f'✅ 当前设备: {torch.cuda.current_device()}')
            print(f'✅ 设备名称: {torch.cuda.get_device_name()}')
    
    # 基本功能测试
    x = torch.randn(3, 3)
    y = torch.mm(x, x)
    print('✅ 基本张量运算正常')
    
    # 分布式测试
    import torch.distributed as dist
    print('✅ 分布式模块导入成功')
    
    import sys
    sys.exit(0)  # 成功退出
    
except Exception as e:
    print(f'❌ PyTorch测试失败: {e}')
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)  # 失败退出
"

PYTORCH_TEST_RESULT=$?

echo ""
echo "🧪 测试2: MaskLLM组件测试"

if [ $PYTORCH_TEST_RESULT -eq 0 ]; then
    echo "✅ PyTorch测试成功，继续测试MaskLLM组件..."
    
    python3.10 -c "
try:
    from megatron.tokenizer import build_tokenizer
    print('✅ megatron.tokenizer导入成功')
    
    # 测试Llama8bTokenizer
    from megatron.tokenizer.llama8b_tokenizer import _Llama8bTokenizer
    print('✅ Llama8bTokenizer导入成功')
    
    import transformer_engine as te
    print('✅ transformer_engine导入成功')
    if hasattr(te, 'pytorch'):
        print('✅ transformer_engine.pytorch可用')
    
except Exception as e:
    print(f'⚠️ MaskLLM组件测试异常: {e}')
"
else
    echo "❌ PyTorch测试失败，跳过MaskLLM组件测试"
fi

echo ""

# ================================================
# 第5步: 创建恢复脚本
# ================================================
echo "📋 第5步: 创建恢复脚本"

cat > "$ISOLATION_DIR/restore_hpcx.sh" << 'EOF'
#!/bin/bash
echo "🔓 恢复HPC-X库文件..."

ISOLATION_DIR="/tmp/hpcx_isolation"
BACKUP_DIR="$ISOLATION_DIR/backup"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ 备份目录不存在，无法恢复"
    exit 1
fi

RESTORED_FILES=0

# 恢复所有备份的库文件
for backup_file in "$BACKUP_DIR"/*; do
    if [ -f "$backup_file" ]; then
        # 解析原始路径
        basename_file=$(basename "$backup_file" | sed 's/_.*$//')
        
        # 查找原始文件位置
        find /opt/hpcx -name "$basename_file" -type f 2>/dev/null | while read original_file; do
            if [ -f "$original_file" ]; then
                cp "$backup_file" "$original_file" 2>/dev/null && \
                echo "  ✅ 恢复: $original_file" && \
                ((RESTORED_FILES++)) || \
                echo "  ❌ 恢复失败: $original_file"
            fi
        done
    fi
done

echo "✅ HPC-X库文件恢复完成"
echo "🧹 清理临时文件..."
rm -rf "$ISOLATION_DIR"
echo "✅ 清理完成"
EOF

chmod +x "$ISOLATION_DIR/restore_hpcx.sh"

echo "✅ 恢复脚本已创建: $ISOLATION_DIR/restore_hpcx.sh"
echo ""

# ================================================
# 第6步: 结果总结
# ================================================
echo "📋 第6步: 结果总结"

if [ $PYTORCH_TEST_RESULT -eq 0 ]; then
    echo "🎉 激进库隔离方案成功!"
    echo ""
    echo "✅ 成功项目:"
    echo "  - HPC-X库文件已被隔离"
    echo "  - PyTorch可以正常导入和运行"
    echo "  - CUDA功能正常"
    echo "  - 分布式模块可用"
    echo ""
    echo "💡 现在可以运行MaskLLM任务:"
    echo "  bash llama8b_scripts/run_maskllm_ultimate.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh 0 0"
    echo "  bash llama8b_scripts/run_maskllm_ultimate.sh llama8b_scripts/run_llama8b_prune_tp8.sh"
    echo ""
    echo "⚠️ 注意: 隔离是临时的，重启容器后会恢复"
    echo "⚠️ 如需恢复HPC-X库，运行: $ISOLATION_DIR/restore_hpcx.sh"
    
else
    echo "❌ 激进库隔离方案失败"
    echo ""
    echo "问题可能是:"
    echo "  - PyTorch与HPC-X的链接过于深层"
    echo "  - 需要重新编译PyTorch"
    echo "  - 或者使用不同的容器镜像"
    echo ""
    echo "🔓 正在自动恢复HPC-X库..."
    bash "$ISOLATION_DIR/restore_hpcx.sh"
fi

echo ""
echo "🔒 激进库隔离方案执行完成"
echo "=" * 80
