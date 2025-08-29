#!/bin/bash

# 深度PyTorch诊断脚本 - 彻底分析HPC-X冲突根源
# 从系统级、编译级、运行时级全面诊断

echo "🔬 深度PyTorch诊断 - HPC-X冲突根源分析"
echo "时间: $(date)"
echo "=" * 80

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# ================================================
# 第1步: 系统级库配置诊断
# ================================================
echo "📋 第1步: 系统级库配置诊断"

echo "🔍 检查系统库配置文件:"
if [ -f "/etc/ld.so.conf" ]; then
    echo "  📄 /etc/ld.so.conf:"
    cat /etc/ld.so.conf | grep -v "^#" | grep -v "^$"
fi

echo ""
echo "🔍 检查ld.so.conf.d目录:"
if [ -d "/etc/ld.so.conf.d" ]; then
    for conf_file in /etc/ld.so.conf.d/*.conf; do
        if [ -f "$conf_file" ]; then
            echo "  📄 $conf_file:"
            cat "$conf_file" | grep -v "^#" | grep -v "^$" | head -10
        fi
    done
fi

echo ""
echo "🔍 使用ldconfig检查库缓存:"
echo "  HPC-X相关库缓存:"
ldconfig -p | grep -i "hpcx\|ucx\|ucc\|hcoll" | head -10

echo ""

# ================================================
# 第2步: PyTorch二进制文件分析
# ================================================
echo "📋 第2步: PyTorch二进制文件分析"

PYTORCH_SO="/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_python.so"
PYTORCH_CPU_SO="/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cpu.so"

echo "🔍 PyTorch核心SO文件分析:"
for so_file in "$PYTORCH_SO" "$PYTORCH_CPU_SO"; do
    if [ -f "$so_file" ]; then
        echo "  📄 分析: $so_file"
        
        echo "    🔍 RPATH/RUNPATH:"
        if command -v readelf >/dev/null 2>&1; then
            readelf -d "$so_file" | grep -E "(RPATH|RUNPATH)" || echo "    ⚠️ 无RPATH/RUNPATH"
        else
            echo "    ⚠️ readelf不可用"
        fi
        
        echo "    🔍 依赖的HPC-X库:"
        if command -v ldd >/dev/null 2>&1; then
            ldd "$so_file" 2>/dev/null | grep -i "hpcx\|ucx\|ucc\|hcoll" | head -5 || echo "    ✅ 无直接HPC-X依赖"
        fi
        
        echo "    🔍 符号表中的HPC-X引用:"
        if command -v nm >/dev/null 2>&1; then
            nm -D "$so_file" 2>/dev/null | grep -i "ucx\|ucc\|hcoll" | head -3 || echo "    ✅ 符号表中无HPC-X引用"
        fi
        echo ""
    fi
done

# ================================================
# 第3步: 运行时动态链接诊断
# ================================================
echo "📋 第3步: 运行时动态链接诊断"

echo "🔍 当前LD_LIBRARY_PATH路径验证:"
IFS=':' read -ra ADDR <<< "$LD_LIBRARY_PATH"
for i, path in "${ADDR[@]}"; do
    if [[ "$path" == *"hpcx"* ]] || [[ "$path" == *"ucx"* ]] || [[ "$path" == *"ucc"* ]]; then
        echo "  ❌ 发现HPC-X路径: $path"
    elif [ -d "$path" ]; then
        echo "  ✅ 清洁路径: $path"
    elif [ -n "$path" ]; then
        echo "  ⚠️ 路径不存在: $path"
    fi
done

echo ""
echo "🔍 检查LD_PRELOAD:"
if [ -n "$LD_PRELOAD" ]; then
    echo "  LD_PRELOAD已设置:"
    IFS=':' read -ra PRELOAD_LIBS <<< "$LD_PRELOAD"
    for lib in "${PRELOAD_LIBS[@]}"; do
        if [ -f "$lib" ]; then
            echo "    ✅ $lib"
        else
            echo "    ❌ $lib (文件不存在)"
        fi
    done
else
    echo "  ⚠️ LD_PRELOAD未设置"
fi

echo ""

# ================================================
# 第4步: 库搜索路径跟踪
# ================================================
echo "📋 第4步: 库搜索路径跟踪"

echo "🔍 使用strace跟踪PyTorch导入过程..."
echo "  注意: 这将显示实际的库搜索过程"

# 创建最小的PyTorch导入测试
cat > /tmp/minimal_torch_test.py << 'EOF'
try:
    import torch
    print("PyTorch导入成功")
except Exception as e:
    print(f"PyTorch导入失败: {e}")
EOF

echo "  🔍 跟踪库文件打开过程:"
if command -v strace >/dev/null 2>&1; then
    strace -e trace=openat,open -f python3.10 /tmp/minimal_torch_test.py 2>&1 | \
    grep -E "(hpcx|ucx|ucc|hcoll)" | head -10 || echo "    ✅ 未检测到HPC-X库访问"
else
    echo "    ⚠️ strace不可用，跳过跟踪"
fi

echo ""

# ================================================
# 第5步: 环境变量冲突诊断
# ================================================
echo "📋 第5步: 环境变量冲突诊断"

echo "🔍 检查可能的冲突环境变量:"
SUSPECT_VARS=$(env | grep -i "hpcx\|ucx\|ucc\|hcoll\|ompi\|mpi" | head -20)
if [ -n "$SUSPECT_VARS" ]; then
    echo "  ⚠️ 发现可疑环境变量:"
    echo "$SUSPECT_VARS" | while read var; do
        echo "    $var"
    done
else
    echo "  ✅ 未发现明显的冲突环境变量"
fi

echo ""

# ================================================
# 第6步: 替代PyTorch策略测试
# ================================================
echo "📋 第6步: 替代PyTorch策略测试"

echo "🔍 测试直接库隔离策略:"

# 策略1: 使用chroot-like库隔离
echo "  策略1: 临时移除HPC-X库文件访问权限"
HPCX_LIB_PATHS=(
    "/opt/hpcx/ucc/lib"
    "/opt/hpcx/ucx/lib"
    "/opt/hpcx/ompi/lib"
    "/opt/hpcx/hcoll/lib"
)

# 备份当前权限并临时修改
BACKUP_DIR="/tmp/hpcx_perms_backup"
mkdir -p "$BACKUP_DIR"

echo "  🔒 临时隔离HPC-X库文件..."
for hpcx_path in "${HPCX_LIB_PATHS[@]}"; do
    if [ -d "$hpcx_path" ]; then
        echo "    处理: $hpcx_path"
        
        # 记录原始权限
        stat -c "%a" "$hpcx_path" > "$BACKUP_DIR/$(basename $hpcx_path)_perm" 2>/dev/null
        
        # 临时移除读取权限 (如果有权限的话)
        chmod 000 "$hpcx_path" 2>/dev/null && echo "      ✅ 权限已临时修改" || echo "      ⚠️ 权限修改失败"
    fi
done

echo ""
echo "  🧪 测试PyTorch导入 (HPC-X库被隔离):"
python3.10 -c "
try:
    import torch
    print('🎉 PyTorch导入成功 (库隔离策略有效)!')
    print(f'版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'设备数: {torch.cuda.device_count()}')
except Exception as e:
    print(f'❌ PyTorch导入仍然失败: {e}')
"

# 恢复权限
echo ""
echo "  🔓 恢复HPC-X库文件权限..."
for hpcx_path in "${HPCX_LIB_PATHS[@]}"; do
    if [ -d "$hpcx_path" ]; then
        backup_file="$BACKUP_DIR/$(basename $hpcx_path)_perm"
        if [ -f "$backup_file" ]; then
            orig_perm=$(cat "$backup_file")
            chmod "$orig_perm" "$hpcx_path" 2>/dev/null && echo "    ✅ 恢复权限: $hpcx_path" || echo "    ⚠️ 恢复权限失败: $hpcx_path"
        fi
    fi
done

rm -rf "$BACKUP_DIR"

echo ""

# ================================================
# 第7步: 解决方案建议
# ================================================
echo "📋 第7步: 解决方案建议"

echo "🎯 基于诊断结果的解决方案:"

echo ""
echo "1️⃣ 如果发现RPATH/RUNPATH问题:"
echo "   - 需要使用patchelf修改PyTorch二进制文件"
echo "   - 或者重新编译PyTorch"

echo ""
echo "2️⃣ 如果发现系统库缓存问题:"
echo "   - 需要重新生成ldconfig缓存"
echo "   - 或者使用LD_LIBRARY_PATH优先级"

echo ""
echo "3️⃣ 如果库隔离策略有效:"
echo "   - 可以考虑永久性移除或重命名HPC-X库"
echo "   - 或者使用容器级别的库隔离"

echo ""
echo "4️⃣ 如果问题持续存在:"
echo "   - 可能需要使用不同的PyTorch版本"
echo "   - 或者完全重新构建容器环境"

echo ""
echo "🔬 诊断完成！"
echo "=" * 80

rm -f /tmp/minimal_torch_test.py
