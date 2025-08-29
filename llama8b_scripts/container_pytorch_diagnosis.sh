#!/bin/bash

# 容器内PyTorch诊断和修复脚本
# 请在Apptainer容器内运行此脚本

echo "🚀 开始容器内PyTorch环境诊断和修复"
echo "时间: $(date)"
echo "=" * 60

# 1. 环境信息检查
echo "📋 步骤1: 环境信息检查"
echo "容器标识: $APPTAINER_NAME $SINGULARITY_NAME"
echo "Python版本: $(python3.10 --version)"
echo "当前目录: $(pwd)"
echo "用户: $(whoami)"
echo ""

# 2. 检查当前PYTHONPATH和LD_LIBRARY_PATH
echo "📋 步骤2: 检查关键环境变量"
echo "PYTHONPATH: $PYTHONPATH"
echo ""
echo "LD_LIBRARY_PATH前10个路径:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -10
echo ""

# 3. 检查HPC-X库存在情况
echo "📋 步骤3: 检查HPC-X库"
if [ -f "/opt/hpcx/ucc/lib/libucc.so.1" ]; then
    echo "❌ 发现HPC-X冲突库: /opt/hpcx/ucc/lib/libucc.so.1"
    echo "文件信息:"
    ls -la /opt/hpcx/ucc/lib/libucc.so.1
else
    echo "✅ 未发现HPC-X冲突库"
fi
echo ""

# 4. 检查容器内PyTorch位置
echo "📋 步骤4: 检查容器内PyTorch"
echo "搜索容器内PyTorch位置:"
find /usr/local/lib/python3.10/dist-packages/ -name "torch" -type d 2>/dev/null || echo "未找到"
find /usr/local/lib/python3.10/site-packages/ -name "torch" -type d 2>/dev/null || echo "未找到"
echo ""

# 5. 测试基础Python导入
echo "📋 步骤5: 测试Python基础模块导入"
echo "测试sys导入:"
python3.10 -c "import sys; print('✅ sys导入成功')" 2>&1

echo "测试os导入:"
python3.10 -c "import os; print('✅ os导入成功')" 2>&1

echo "测试numpy导入:"
python3.10 -c "import numpy; print('✅ numpy导入成功')" 2>&1
echo ""

# 6. 测试PyTorch导入 - 详细错误信息
echo "📋 步骤6: 测试PyTorch导入 (详细错误)"
echo "尝试导入PyTorch:"
python3.10 -c "import torch; print('✅ PyTorch导入成功, 版本:', torch.__version__)" 2>&1
echo ""

# 7. 如果PyTorch导入失败，尝试强制环境修复
echo "📋 步骤7: 应用环境修复"

# 清除所有HPC-X相关环境变量
unset HPCX_DIR HPCX_HOME HPCX_ROOT HPCX_MPI_DIR HPCX_UCX_DIR HPCX_UCC_DIR HPCX_HCOLL_DIR
unset OMPI_HOME OMPI_ROOT MPI_HOME MPI_ROOT

# 强力禁用HPC-X组件
export OMPI_MCA_pml=^ucx,^hcoll
export OMPI_MCA_btl=^openib,^uct
export OMPI_MCA_coll=^hcoll,^ucx
export OMPI_MCA_osc=^ucx
export OMPI_MCA_spml=^ucx
export UCX_TLS=^ud,^dc,^rc
export UCC_TLS=^ucp,^sharp
export UCC_CLS=^sharp
export UCX_LOG_LEVEL=error
export UCC_LOG_LEVEL=error

# 重建LD_LIBRARY_PATH - 排除HPC-X路径
echo "重建LD_LIBRARY_PATH..."
CLEAN_LD_LIBRARY_PATH=""
IFS=':' read -ra ADDR <<< "$LD_LIBRARY_PATH"
for path in "${ADDR[@]}"; do
    if [[ "$path" != *"hpcx"* ]] && [[ "$path" != *"ucx"* ]] && [[ "$path" != *"ucc"* ]]; then
        if [ -z "$CLEAN_LD_LIBRARY_PATH" ]; then
            CLEAN_LD_LIBRARY_PATH="$path"
        else
            CLEAN_LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH:$path"
        fi
    else
        echo "排除HPC-X路径: $path"
    fi
done

# 优先容器内库
PRIORITY_PATHS=(
    "/usr/local/lib/python3.10/dist-packages/torch/lib"
    "/usr/local/cuda/lib64"
    "/usr/local/cuda/compat/lib"
    "/opt/nccl_libs"
)

for priority_path in "${PRIORITY_PATHS[@]}"; do
    if [ -d "$priority_path" ]; then
        CLEAN_LD_LIBRARY_PATH="$priority_path:$CLEAN_LD_LIBRARY_PATH"
        echo "优先添加: $priority_path"
    fi
done

export LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH"
echo "✅ LD_LIBRARY_PATH已重建"
echo ""

# 8. 重新测试PyTorch
echo "📋 步骤8: 修复后重新测试PyTorch"
python3.10 -c "
import sys
print('PYTHONPATH前5个:', sys.path[:5])
print('')

try:
    import torch
    print('✅ PyTorch导入成功!')
    print('版本:', torch.__version__)
    print('路径:', torch.__file__)
    print('CUDA可用:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA设备数:', torch.cuda.device_count())
    
    # 测试基本运算
    x = torch.randn(2, 2)
    y = torch.mm(x, x)
    print('✅ 张量运算测试成功')
    
    # 测试分布式
    import torch.distributed as dist
    print('✅ torch.distributed导入成功')
    
except Exception as e:
    print('❌ PyTorch测试失败:', str(e))
    import traceback
    traceback.print_exc()
" 2>&1
echo ""

# 9. 测试transformer_engine
echo "📋 步骤9: 测试transformer_engine"
python3.10 -c "
try:
    import transformer_engine as te
    print('✅ transformer_engine导入成功')
    print('版本:', getattr(te, '__version__', 'unknown'))
    print('路径:', te.__file__)
    if hasattr(te, 'pytorch'):
        print('✅ transformer_engine.pytorch可用')
    else:
        print('⚠️ transformer_engine.pytorch不可用')
except Exception as e:
    print('❌ transformer_engine导入失败:', str(e))
" 2>&1
echo ""

# 10. 测试数据预处理工具
echo "📋 步骤10: 测试数据预处理工具导入"
python3.10 -c "
try:
    from megatron.tokenizer import build_tokenizer
    print('✅ megatron.tokenizer导入成功')
except Exception as e:
    print('❌ megatron.tokenizer导入失败:', str(e))
" 2>&1
echo ""

# 11. 最终建议
echo "📋 步骤11: 诊断完成"
echo "=" * 60
echo "如果PyTorch导入成功，可以尝试运行:"
echo "bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh 0 0"
echo ""
echo "如果仍有问题，请将以上完整输出发送给我分析。"
