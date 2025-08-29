#!/bin/bash

# 替代方案评估脚本
# 重新分析是否需要库劫持，探索更简单的解决方案

echo "🤔 替代方案评估"
echo "时间: $(date)"
echo "目标: 评估是否真的需要库劫持方案"
echo "=" * 50

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# ================================================
# 第1步: 重新分析问题本质
# ================================================
echo "📋 第1步: 重新分析问题本质"

echo "🔍 问题回顾:"
echo "  当前错误: ucc_ee_ack_event 未定义符号"
echo "  错误位置: PyTorch导入时"
echo "  历史状态: 之前曾经正常工作"
echo ""

echo "💡 关键思考:"
echo "  Q1: 之前是如何工作的？"
echo "  Q2: 是否存在更简单的解决方案？"
echo "  Q3: 库劫持是否必要？"
echo ""

# ================================================
# 第2步: 检查run_maskllm_native.sh的历史修复
# ================================================
echo "📋 第2步: 检查run_maskllm_native.sh的历史修复"

if [ -f "run_maskllm_native.sh" ]; then
    echo "🔍 分析run_maskllm_native.sh当前状态:"
    
    # 检查是否包含NGC容器修复
    if grep -q "NGC容器库修复" "run_maskllm_native.sh"; then
        echo "  ✅ 包含NGC容器库修复代码"
        
        # 检查具体的修复内容
        echo "  🔍 检查具体修复内容:"
        
        if grep -q "opt/conda_libs" "run_maskllm_native.sh"; then
            echo "    ✅ 包含conda_libs路径配置"
        else
            echo "    ❌ 缺少conda_libs路径配置"
        fi
        
        if grep -q "LD_LIBRARY_PATH.*opt/conda_libs" "run_maskllm_native.sh"; then
            echo "    ✅ LD_LIBRARY_PATH包含正确路径"
        else
            echo "    ❌ LD_LIBRARY_PATH配置可能不正确"
        fi
        
        # 检查HPC-X相关配置
        if grep -q "HPC.*清理\|HPCX.*清理" "run_maskllm_native.sh"; then
            echo "    ✅ 包含HPC-X清理配置"
        else
            echo "    ❌ 缺少HPC-X清理配置"
        fi
        
    else
        echo "  ❌ 缺少NGC容器库修复代码"
    fi
else
    echo "  ❌ run_maskllm_native.sh不存在"
fi

echo ""

# ================================================
# 第3步: 测试简单的环境修复
# ================================================
echo "📋 第3步: 测试简单的环境修复 (不使用库劫持)"

echo "🧪 尝试1: 仅通过环境变量修复"

# 备份当前环境
ORIGINAL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
ORIGINAL_LD_PRELOAD="$LD_PRELOAD"

# 尝试简单的环境修复
echo "🔧 应用简单环境修复..."

# 设置正确的CUDA库路径 (基于诊断结果)
CORRECT_NGC_PATHS="/opt/conda_libs:/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib:/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib"
CORRECT_CUDA_PATHS="/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib"
CORRECT_SYSTEM_PATHS="/usr/local/cuda/compat/lib:/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"

export LD_LIBRARY_PATH="$CORRECT_NGC_PATHS:$CORRECT_CUDA_PATHS:$CORRECT_SYSTEM_PATHS"

# 清理可能的HPC-X冲突变量
unset HPCX_DIR HPCX_HOME HPCX_ROOT
unset OMPI_HOME MPI_HOME

# 设置禁用HPC-X的环境变量
export OMPI_MCA_pml="^ucx"
export UCX_TLS="^all"
export UCC_TLS="nccl"

echo "✅ 简单环境修复已应用"
echo "  LD_LIBRARY_PATH: 已更新为正确的NGC路径"
echo "  HPC-X变量: 已清理"
echo "  冲突禁用: 已设置"

# 测试简单修复效果
echo ""
echo "🧪 测试简单修复效果..."

python3.10 -c "
print('🔍 测试简单环境修复...')
try:
    import torch
    print('✅ PyTorch导入成功')
    print(f'  版本: {torch.__version__}')
    
    if torch.cuda.is_available():
        print(f'✅ CUDA可用: {torch.cuda.device_count()}个GPU')
    else:
        print('⚠️ CUDA不可用')
    
    # 测试分布式
    import torch.distributed as dist
    print('✅ 分布式模块导入成功')
    
    print('\\n🎉 简单环境修复成功！无需库劫持')
    
except ImportError as e:
    if 'ucc_ee_ack_event' in str(e):
        print('❌ 仍有ucc_ee_ack_event符号问题')
        print('💡 需要库劫持方案')
        exit(1)
    elif 'libcudnn' in str(e) or 'libcupti' in str(e):
        print('❌ CUDA库路径问题')
        print('💡 需要进一步修复路径配置')
        exit(2)
    else:
        print(f'❌ 其他导入错误: {e}')
        exit(3)
except Exception as e:
    print(f'❌ 意外错误: {e}')
    exit(4)
"

SIMPLE_FIX_RESULT=$?

# 恢复原始环境 (避免影响后续测试)
export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"
export LD_PRELOAD="$ORIGINAL_LD_PRELOAD"

echo ""

# ================================================
# 第4步: 尝试运行run_maskllm_native.sh
# ================================================
echo "📋 第4步: 测试run_maskllm_native.sh修复能力"

if [ $SIMPLE_FIX_RESULT -ne 0 ]; then
    echo "🧪 尝试2: 使用run_maskllm_native.sh"
    
    # 创建一个简单的测试脚本
    cat > /tmp/simple_pytorch_test.py << 'EOF'
#!/usr/bin/env python3.10

print("🔍 通过run_maskllm_native.sh测试PyTorch...")

try:
    import torch
    print("✅ PyTorch导入成功")
    print(f"  版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.device_count()}个GPU")
    else:
        print("⚠️ CUDA不可用")
    
    import torch.distributed as dist
    print("✅ 分布式模块导入成功")
    
    print("\n🎉 run_maskllm_native.sh修复成功！")
    
except ImportError as e:
    if 'ucc_ee_ack_event' in str(e):
        print("❌ 仍有ucc_ee_ack_event符号问题")
        print("💡 run_maskllm_native.sh修复不够")
    else:
        print(f"❌ 其他导入错误: {e}")
except Exception as e:
    print(f"❌ 意外错误: {e}")
EOF
    
    chmod +x /tmp/simple_pytorch_test.py
    
    echo "🔧 通过run_maskllm_native.sh测试..."
    
    # 使用run_maskllm_native.sh运行测试
    bash run_maskllm_native.sh /tmp/simple_pytorch_test.py 2>&1 | tail -20
    
    RUN_MASKLLM_RESULT=${PIPESTATUS[0]}
    
    # 清理测试文件
    rm -f /tmp/simple_pytorch_test.py
    
else
    echo "✅ 简单环境修复已成功，无需测试run_maskllm_native.sh"
    RUN_MASKLLM_RESULT=0
fi

echo ""

# ================================================
# 第5步: 方案推荐
# ================================================
echo "📋 第5步: 方案推荐"

echo "🎯 测试结果分析:"

if [ $SIMPLE_FIX_RESULT -eq 0 ]; then
    echo "  ✅ 简单环境修复成功"
    echo "  💡 结论: 无需复杂的库劫持方案"
    echo ""
    echo "🚀 推荐方案: 简单环境修复"
    echo ""
    echo "📋 具体操作:"
    echo "1. 修复run_maskllm_native.sh中的环境配置"
    echo "2. 确保正确的CUDA库路径"
    echo "3. 清理HPC-X冲突变量"
    echo ""
    echo "💡 优势:"
    echo "  - 简单快速 (2-3分钟)"
    echo "  - 无需编译库文件"
    echo "  - 维护成本低"
    echo "  - 风险小"

elif [ $RUN_MASKLLM_RESULT -eq 0 ]; then
    echo "  ✅ run_maskllm_native.sh修复成功"
    echo "  💡 结论: 现有脚本可能只需要小幅调整"
    echo ""
    echo "🚀 推荐方案: 增强run_maskllm_native.sh"
    echo ""
    echo "📋 具体操作:"
    echo "1. 分析run_maskllm_native.sh中的修复逻辑"
    echo "2. 确保修复逻辑能够自动应用"
    echo "3. 可能只需要调整环境变量设置"

else
    echo "  ❌ 简单修复和run_maskllm_native.sh都失败"
    echo "  💡 结论: 确实需要库劫持方案"
    echo ""
    echo "🚀 推荐方案: 库劫持方案"
    echo ""
    echo "📋 具体操作:"
    echo "1. 运行精准库劫持修复"
    echo "2. 创建必要的兼容库"
    echo "3. 配置LD_PRELOAD"
fi

echo ""

# ================================================
# 第6步: 创建简单修复脚本 (如果可行)
# ================================================

if [ $SIMPLE_FIX_RESULT -eq 0 ]; then
    echo "📋 第6步: 创建简单修复脚本"
    
    SIMPLE_FIX_SCRIPT="llama8b_scripts/simple_environment_fix.sh"
    
    cat > "$SIMPLE_FIX_SCRIPT" << 'EOF'
#!/bin/bash

# 简单环境修复脚本
# 仅通过环境变量修复，无需库劫持

echo "🔧 应用简单环境修复"
echo "时间: $(date)"

# 设置正确的CUDA库路径
CORRECT_NGC_PATHS="/opt/conda_libs:/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib:/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib"
CORRECT_CUDA_PATHS="/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib"
CORRECT_SYSTEM_PATHS="/usr/local/cuda/compat/lib:/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"

export LD_LIBRARY_PATH="$CORRECT_NGC_PATHS:$CORRECT_CUDA_PATHS:$CORRECT_SYSTEM_PATHS"

# 清理HPC-X冲突
unset HPCX_DIR HPCX_HOME HPCX_ROOT
unset OMPI_HOME MPI_HOME

# 禁用冲突组件
export OMPI_MCA_pml="^ucx"
export UCX_TLS="^all"
export UCC_TLS="nccl"

# 8GPU配置
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WORLD_SIZE="8"
export TORCH_DISTRIBUTED_BACKEND="nccl"

# Python路径
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:$(pwd)/.ext_pkgs:$(pwd)"

echo "✅ 简单环境修复完成"
echo "💡 无需库劫持，仅使用环境变量修复"
EOF

    chmod +x "$SIMPLE_FIX_SCRIPT"
    
    echo "✅ 简单修复脚本已创建: $SIMPLE_FIX_SCRIPT"
    echo ""
    echo "🚀 立即使用:"
    echo "   source $SIMPLE_FIX_SCRIPT"
    echo "   python3.10 your_script.py"
fi

echo ""
echo "🎯 替代方案评估完成"
echo "=" * 50
