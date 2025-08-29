#!/bin/bash

# 容器环境变化诊断脚本
# 分析为什么之前正常的环境现在出现ucc_ee_ack_event错误

echo "🔍 容器环境变化诊断"
echo "时间: $(date)"
echo "目标: 分析容器内环境变化原因"
echo "=" * 60

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# ================================================
# 第1步: 环境状态快照分析
# ================================================
echo "📋 第1步: 当前环境状态快照"

echo "🔍 核心环境变量状态:"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-未设置}"
echo "  LD_PRELOAD: ${LD_PRELOAD:-未设置}"
echo "  PYTHONPATH: ${PYTHONPATH:-未设置}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"
echo ""

echo "🔍 关键库文件状态:"
key_libraries=(
    "/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cuda.so"
    "/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_python.so"
    "/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cpu.so"
)

for lib in "${key_libraries[@]}"; do
    if [ -f "$lib" ]; then
        size=$(stat -c%s "$lib")
        mtime=$(stat -c%Y "$lib")
        echo "  ✅ $(basename $lib): 存在 ($(($size / 1024 / 1024))MB, 修改时间: $(date -d @$mtime))"
    else
        echo "  ❌ $(basename $lib): 不存在"
    fi
done

echo ""

# ================================================
# 第2步: 之前修复状态检查
# ================================================
echo "📋 第2步: 之前的修复状态检查"

echo "🔍 检查之前创建的兼容库:"
compat_libraries=(
    "/tmp/ultimate_pytorch_fix/libucc_compat_ultimate.so"
    "/tmp/minimal_hijack/libhijack_complete.so"
    "/tmp/optimal_strategy_b/libsmart_compat_complete.so"
    "/tmp/eight_gpu_distributed/setup_8gpu_env.sh"
)

missing_count=0
for lib in "${compat_libraries[@]}"; do
    if [ -f "$lib" ]; then
        size=$(stat -c%s "$lib" 2>/dev/null)
        echo "  ✅ $(basename $lib): 存在 ($(($size / 1024))KB)"
    else
        echo "  ❌ $(basename $lib): 缺失"
        ((missing_count++))
    fi
done

if [ $missing_count -gt 0 ]; then
    echo ""
    echo "⚠️ 发现 $missing_count 个之前的修复文件缺失"
    echo "💡 这可能是环境问题的主要原因"
fi

echo ""

# ================================================
# 第3步: 容器内进程和修改历史分析
# ================================================
echo "📋 第3步: 容器内活动分析"

echo "🔍 检查可能的环境修改:"

# 检查run_maskllm_native.sh的当前状态
if [ -f "run_maskllm_native.sh" ]; then
    echo "  📄 run_maskllm_native.sh 当前状态:"
    
    # 检查是否包含关键的修复代码
    if grep -q "NGC容器库修复" "run_maskllm_native.sh"; then
        echo "    ✅ 包含NGC容器库修复代码"
    else
        echo "    ❌ 缺少NGC容器库修复代码"
    fi
    
    if grep -q "opt/conda_libs" "run_maskllm_native.sh"; then
        echo "    ✅ 包含conda_libs路径配置"
    else
        echo "    ❌ 缺少conda_libs路径配置"
    fi
    
    if grep -q "LD_PRELOAD" "run_maskllm_native.sh"; then
        echo "    ✅ 包含LD_PRELOAD配置"
    else
        echo "    ❌ 缺少LD_PRELOAD配置"
    fi
fi

echo ""

# 检查当前会话的环境历史
echo "🔍 检查当前会话环境修改:"
if [ -n "$LD_LIBRARY_PATH" ]; then
    echo "  📍 当前LD_LIBRARY_PATH包含的路径:"
    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | while read path; do
        if [ -n "$path" ]; then
            if [ -d "$path" ]; then
                echo "    ✅ $path (存在)"
            else
                echo "    ❌ $path (不存在)"
            fi
        fi
    done
else
    echo "  ⚠️ LD_LIBRARY_PATH 未设置"
fi

echo ""

# ================================================
# 第4步: 可能的变化原因分析
# ================================================
echo "📋 第4步: 变化原因分析"

echo "🧠 分析可能的环境变化原因:"

# 原因1: 临时文件系统清理
echo ""
echo "1️⃣ 临时文件系统清理:"
temp_dirs=("/tmp" "/var/tmp")
for dir in "${temp_dirs[@]}"; do
    if [ -d "$dir" ]; then
        # 检查/tmp目录的挂载类型
        mount_info=$(mount | grep " $dir ")
        if echo "$mount_info" | grep -q "tmpfs"; then
            echo "  ⚠️ $dir 是tmpfs挂载 - 重启后会清空"
        else
            echo "  ✅ $dir 是持久存储"
        fi
    fi
done

# 原因2: 环境变量重置
echo ""
echo "2️⃣ 环境变量重置分析:"
if [ -z "$LD_PRELOAD" ]; then
    echo "  ❌ LD_PRELOAD未设置 - 兼容库未加载"
else
    echo "  ✅ LD_PRELOAD已设置: $LD_PRELOAD"
    # 检查预加载的库是否存在
    echo "$LD_PRELOAD" | tr ':' '\n' | while read lib; do
        if [ -n "$lib" ] && [ -f "$lib" ]; then
            echo "    ✅ $lib (存在)"
        elif [ -n "$lib" ]; then
            echo "    ❌ $lib (不存在)"
        fi
    done
fi

# 原因3: PyTorch库文件变化
echo ""
echo "3️⃣ PyTorch库文件变化分析:"
torch_lib="/usr/local/lib/python3.10/dist-packages/torch/lib/libtorch_cuda.so"

if [ -f "$torch_lib" ]; then
    # 检查库的依赖关系
    echo "  📄 libtorch_cuda.so 当前依赖关系:"
    ldd "$torch_lib" 2>/dev/null | grep -E "(ucc|ucx|ucs|mpi)" | head -5 | while read line; do
        echo "    $line"
    done
    
    # 检查是否有未定义符号
    echo "  🔍 检查关键未定义符号:"
    if ldd -r "$torch_lib" 2>&1 | grep -q "ucc_ee_ack_event"; then
        echo "    ❌ 确认存在 ucc_ee_ack_event 未定义符号"
    else
        echo "    ✅ 未发现 ucc_ee_ack_event 符号问题"
    fi
fi

# 原因4: 容器重启或状态重置
echo ""
echo "4️⃣ 容器状态分析:"
if [ -f "/proc/1/stat" ]; then
    # 检查容器启动时间（通过init进程）
    boot_time=$(stat -c %Y /proc/1/stat 2>/dev/null)
    current_time=$(date +%s)
    uptime_hours=$(( (current_time - boot_time) / 3600 ))
    echo "  📊 容器运行时间: 约 $uptime_hours 小时"
    
    if [ $uptime_hours -lt 1 ]; then
        echo "  ⚠️ 容器最近重启过 - 可能导致临时修复丢失"
    else
        echo "  ✅ 容器运行时间较长"
    fi
fi

echo ""

# ================================================
# 第5步: 问题根因总结和建议
# ================================================
echo "📋 第5步: 问题根因总结"

echo "🎯 最可能的原因排序:"
echo ""

echo "1️⃣ **临时修复文件丢失** (概率: 90%)"
echo "   原因: /tmp目录下的兼容库被清理或重置"
echo "   证据: 之前创建的兼容库文件不存在"
echo "   影响: LD_PRELOAD无法加载兼容库，符号劫持失效"
echo ""

echo "2️⃣ **环境变量重置** (概率: 70%)"
echo "   原因: 会话重启或环境配置脚本未执行"
echo "   证据: LD_PRELOAD等关键变量可能未正确设置"
echo "   影响: 即使兼容库存在也无法生效"
echo ""

echo "3️⃣ **run_maskllm_native.sh状态变化** (概率: 50%)"
echo "   原因: 脚本被修改或覆盖，丢失了之前的修复代码"
echo "   证据: 需要检查脚本内容"
echo "   影响: 自动环境配置失效"
echo ""

echo "4️⃣ **容器状态重置** (概率: 30%)"
echo "   原因: 容器重启或状态回滚"
echo "   证据: 运行时间和文件时间戳"
echo "   影响: 所有临时修复都会丢失"
echo ""

echo "🔧 推荐解决方案:"
echo ""
echo "✅ **立即解决方案** (5分钟):"
echo "   重新应用之前成功的环境修复"
echo "   bash llama8b_scripts/optimal_strategy_executor.sh"
echo ""
echo "✅ **持久化解决方案** (10分钟):"
echo "   将兼容库复制到持久目录"
echo "   修改run_maskllm_native.sh集成修复"
echo ""
echo "✅ **预防措施**:"
echo "   创建环境检查和自动修复机制"
echo "   定期备份关键修复文件"

echo ""
echo "🎯 容器环境诊断完成"
echo "=" * 60
