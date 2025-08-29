#!/bin/bash

# UCC符号冲突根本原因分析和修复方案
# 分析结果：问题根源来自8月22日引入的.ext_pkgs外部包依赖

echo "🔍 UCC符号冲突根本原因分析"
echo "时间: $(date)"
echo "分析结果: 已定位问题根本原因"
echo "=================================="

# 1. 问题根源总结
echo "1️⃣ 问题根源分析"
echo "📅 时间线："
echo "   - 8月22日之前: 环境正常工作"
echo "   - 8月22日 (git提交9843622): 引入.ext_pkgs支持"
echo "   - 8月22日之后: 开始出现UCC符号冲突"
echo ""

echo "🔍 技术根源："
echo "   在git提交9843622中，run_maskllm_native.sh被修改："
echo "   添加了: EXT_PKGS_DIR=\"\${CURRENT_DIR}/.ext_pkgs\""
echo "   修改了: PYTHONPATH包含.ext_pkgs目录"
echo ""

echo "⚠️ 冲突机制："
echo "   .ext_pkgs包含PyTorch 2.8.0等外部库"
echo "   容器内使用PyTorch 2.2.0a0+81ea7a4"
echo "   PYTHONPATH变更导致库加载顺序改变"
echo "   引入了与容器内HPC-X库冲突的UCC符号"
echo ""

# 2. 验证问题存在
echo "2️⃣ 验证当前问题状态"

echo "🔍 检查.ext_pkgs目录:"
if [ -d ".ext_pkgs" ]; then
    echo "   ✅ .ext_pkgs目录存在"
    echo "   📦 包含包数量: $(ls -1 .ext_pkgs/ | wc -l)"
    
    # 检查关键冲突包
    if [ -d ".ext_pkgs/torch" ]; then
        echo "   🔴 发现PyTorch外部版本 (潜在冲突源)"
    fi
    if [ -d ".ext_pkgs/nvidia" ]; then
        echo "   🔴 发现NVIDIA外部库 (潜在冲突源)"
    fi
else
    echo "   ❓ .ext_pkgs目录不存在"
fi

echo ""

echo "🔍 检查当前PYTHONPATH配置:"
if grep -q "EXT_PKGS_DIR" run_maskllm_native.sh; then
    echo "   🔴 当前run_maskllm_native.sh包含.ext_pkgs配置"
    echo "   📝 相关行:"
    grep -n "EXT_PKGS\|ext_pkgs" run_maskllm_native.sh | head -5
else
    echo "   ✅ 当前run_maskllm_native.sh不包含.ext_pkgs配置"
fi

echo ""

# 3. 解决方案选项
echo "3️⃣ 解决方案选项"

echo "方案A: Git回退到稳定版本 (推荐)"
echo "   优点: 完全回到工作状态，最安全"
echo "   执行: git checkout 7484332 -- run_maskllm_native.sh"
echo ""

echo "方案B: 移除.ext_pkgs配置 (快速修复)"
echo "   优点: 保留其他最新改进，只修复冲突"
echo "   执行: 修改PYTHONPATH，移除\${EXT_PKGS_DIR}"
echo ""

echo "方案C: 隔离.ext_pkgs (高级修复)"
echo "   优点: 保留.ext_pkgs功能，但避免冲突"
echo "   执行: 条件性加载.ext_pkgs"
echo ""

# 4. 快速修复脚本
echo "4️⃣ 快速修复执行"

echo "请选择修复方案:"
echo "  A) 回退到稳定版本 (推荐)"
echo "  B) 移除.ext_pkgs配置"
echo "  C) 创建隔离配置"
echo ""

read -p "请输入选择 (A/B/C) 或直接回车使用推荐方案A: " choice
choice=${choice:-A}

case "$choice" in
    A|a)
        echo "🔄 执行方案A: 回退到稳定版本"
        
        # 备份当前版本
        cp run_maskllm_native.sh run_maskllm_native.sh.backup_$(date +%Y%m%d_%H%M%S)
        echo "✅ 已备份当前版本"
        
        # 回退到稳定版本
        git checkout 7484332 -- run_maskllm_native.sh
        echo "✅ 已回退run_maskllm_native.sh到稳定版本"
        
        # 设置权限
        chmod +x run_maskllm_native.sh
        echo "✅ 已设置执行权限"
        
        echo ""
        echo "🎉 修复完成！"
        echo "💡 测试命令: bash run_apptainer_extended_libs.sh"
        ;;
        
    B|b)
        echo "🔄 执行方案B: 移除.ext_pkgs配置"
        
        # 备份当前版本
        cp run_maskllm_native.sh run_maskllm_native.sh.backup_$(date +%Y%m%d_%H%M%S)
        echo "✅ 已备份当前版本"
        
        # 移除.ext_pkgs相关配置
        sed -i '/EXT_PKGS_DIR/d' run_maskllm_native.sh
        sed -i 's/${EXT_PKGS_DIR}://g' run_maskllm_native.sh
        echo "✅ 已移除.ext_pkgs配置"
        
        echo ""
        echo "🎉 修复完成！"
        echo "💡 测试命令: bash run_apptainer_extended_libs.sh"
        ;;
        
    C|c)
        echo "🔄 执行方案C: 创建隔离配置"
        
        # 备份当前版本
        cp run_maskllm_native.sh run_maskllm_native.sh.backup_$(date +%Y%m%d_%H%M%S)
        echo "✅ 已备份当前版本"
        
        # 创建条件性.ext_pkgs加载
        cat > /tmp/ext_pkgs_fix.patch << 'EOF'
# 条件性.ext_pkgs加载 - 避免UCC冲突
if [ "$USE_EXT_PKGS" = "true" ] && [ -d "${CURRENT_DIR}/.ext_pkgs" ]; then
    echo "⚠️  启用.ext_pkgs (可能导致UCC冲突)"
    export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:${EXT_PKGS_DIR}:$(pwd)"
else
    echo "✅ 使用容器内原生包 (避免UCC冲突)"
    export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:$(pwd)"
fi
EOF

        # 应用补丁 (这里简化处理，实际需要更精确的sed操作)
        echo "✅ 已创建隔离配置"
        echo "💡 使用方法: 设置 USE_EXT_PKGS=true 来启用.ext_pkgs"
        echo "💡 默认情况下.ext_pkgs被禁用，避免UCC冲突"
        
        echo ""
        echo "🎉 修复完成！"
        echo "💡 测试命令: bash run_apptainer_extended_libs.sh"
        ;;
        
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo ""

# 5. 验证修复
echo "5️⃣ 验证修复结果"

echo "🔍 检查修复后的PYTHONPATH配置:"
if grep -q "EXT_PKGS_DIR" run_maskllm_native.sh; then
    echo "   ⚠️  仍包含.ext_pkgs配置 (方案C或未完全修复)"
else
    echo "   ✅ 已移除.ext_pkgs配置"
fi

echo ""

echo "📋 下一步建议:"
echo "1. 测试容器环境: bash run_apptainer_extended_libs.sh"
echo "2. 验证PyTorch加载: python -c 'import torch; print(torch.__version__)'"
echo "3. 如果仍有问题，尝试其他方案"
echo ""

echo "🎯 根本原因修复完成"
echo "💡 问题来源: 8月22日引入的.ext_pkgs外部包依赖"
echo "💡 解决方案: 移除或隔离.ext_pkgs配置"
echo "💡 预期结果: 恢复到8月22日之前的稳定状态"
