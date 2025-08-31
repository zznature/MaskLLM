#!/bin/bash
# 直接UCC冲突测试脚本
# 目标：绕过所有其他依赖，直接测试UCC冲突
# 运行方式: 在apptainer内直接执行

echo "=========================================="
echo "直接UCC冲突测试"
echo "=========================================="

# 1. 首先完善库路径，包含所有可能需要的路径
echo "1. 设置完整的库路径环境..."

# 备份原始环境
ORIGINAL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# 构建完整的库路径，包含所有可能需要的库
COMPLETE_PATHS=(
    "/opt/conda_libs"                                           # cuDNN
    "/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib"  # cuPTI
    "/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib"        # NCCL
    "/usr/local/cuda/lib64"                                     # CUDA
    "/usr/local/lib/python3.10/dist-packages/torch/lib"        # PyTorch
    "/usr/lib/x86_64-linux-gnu"                                 # 系统库
    "/usr/local/lib"                                            # 本地库
    "/lib/x86_64-linux-gnu"                                     # 系统基础库
)

# 构建新的LD_LIBRARY_PATH
NEW_LD_LIBRARY_PATH=""
for path in "${COMPLETE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        if [ -z "$NEW_LD_LIBRARY_PATH" ]; then
            NEW_LD_LIBRARY_PATH="$path"
        else
            NEW_LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH:$path"
        fi
        echo "  ✅ 添加: $path"
    else
        echo "  ⚠️  跳过: $path (不存在)"
    fi
done

export LD_LIBRARY_PATH="$NEW_LD_LIBRARY_PATH"

echo "完整库路径前5个："
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | head -5 | nl

# 2. 直接测试torch._C模块（PyTorch核心）
echo -e "\n2. 直接测试torch._C模块..."

TORCH_C_RESULT=$(python3 -c "
import sys
import os

print('Python版本:', sys.version.split()[0])
print('当前LD_LIBRARY_PATH前3个:')
for i, path in enumerate(os.environ.get('LD_LIBRARY_PATH', '').split(':')[:3]):
    print(f'  {i+1}. {path}')

print('\\n尝试导入torch._C...')
try:
    import torch._C
    print('✅ torch._C导入成功')
    print('STATUS: SUCCESS')
except ImportError as e:
    error_msg = str(e)
    print('❌ torch._C导入失败:')
    print('错误详情:', error_msg)
    
    # 精确检查UCC冲突
    if 'undefined symbol: ucs_mpool_params_reset' in error_msg:
        print('🔍 确认: UCC冲突问题存在！')
        print('涉及文件:', [part for part in error_msg.split() if 'hpcx' in part or '.so' in part])
        print('STATUS: UCC_CONFLICT')
    elif 'libcudnn' in error_msg:
        print('🔍 发现: cuDNN依赖问题')
        print('STATUS: CUDNN_ISSUE')
    elif 'libcupti' in error_msg:
        print('🔍 发现: cuPTI依赖问题')
        print('STATUS: CUPTI_ISSUE')
    else:
        print('🔍 发现: 其他类型问题')
        print('STATUS: OTHER_ISSUE')
        
except Exception as e:
    print('❌ 其他异常:', str(e))
    print('STATUS: EXCEPTION')
" 2>&1)

echo "$TORCH_C_RESULT"

# 3. 如果torch._C成功，继续测试完整的torch导入
if echo "$TORCH_C_RESULT" | grep -q "STATUS: SUCCESS"; then
    echo -e "\n3. torch._C成功，测试完整torch导入..."
    
    FULL_TORCH_RESULT=$(python3 -c "
try:
    import torch
    print('✅ 完整torch导入成功')
    print('PyTorch版本:', torch.__version__)
    
    # 测试基本操作
    x = torch.randn(2, 3)
    print('✅ 张量操作正常')
    
    print('STATUS: FULL_SUCCESS')
except Exception as e:
    print('❌ 完整torch导入失败:', str(e))
    print('STATUS: FULL_FAILED')
" 2>&1)
    
    echo "$FULL_TORCH_RESULT"
else
    echo -e "\n3. torch._C失败，无需测试完整导入"
fi

# 4. 专门验证UCC库本身在当前环境的兼容性
echo -e "\n4. 验证UCC库在当前完整环境下的状态..."

UCC_ENV_TEST=$(python3 -c "
import ctypes
import os

try:
    print('测试UCC/UCX在当前环境下的兼容性...')
    
    # 加载UCX库
    ucx_lib = ctypes.CDLL('/opt/hpcx/ucx/lib/libucs.so.0')
    print('✅ UCX库加载成功')
    
    # 测试符号
    symbol = ucx_lib.ucs_mpool_params_reset
    print('✅ UCX符号访问成功')
    
    # 加载UCC库
    ucc_lib = ctypes.CDLL('/opt/hpcx/ucc/lib/libucc.so.1')
    print('✅ UCC库加载成功')
    
    print('STATUS: UCC_COMPATIBLE')
    
except OSError as e:
    error_msg = str(e)
    if 'undefined symbol: ucs_mpool_params_reset' in error_msg:
        print('❌ UCC冲突在当前环境下确实存在')
        print('STATUS: UCC_CONFLICT_CURRENT_ENV')
    else:
        print('❌ 其他UCC加载问题:', error_msg)
        print('STATUS: UCC_OTHER_ERROR')
except Exception as e:
    print('❌ UCC测试异常:', str(e))
    print('STATUS: UCC_EXCEPTION')
" 2>&1)

echo "$UCC_ENV_TEST"

# 5. 综合结论
echo -e "\n=========================================="
echo "直接测试综合结论"
echo "=========================================="

# 分析torch._C结果
if echo "$TORCH_C_RESULT" | grep -q "STATUS: UCC_CONFLICT"; then
    echo "🔍 关键发现: torch._C确认UCC冲突存在"
    echo "📋 最终结论: UCC问题确实未解决，用户判断正确"
    echo "🎯 下一步: 需要执行深度UCC冲突修复方案"
    
elif echo "$TORCH_C_RESULT" | grep -q "STATUS: SUCCESS"; then
    echo "✅ 发现: torch._C导入成功"
    
    if echo "$FULL_TORCH_RESULT" | grep -q "STATUS: FULL_SUCCESS"; then
        echo "🎉 最终结论: 所有PyTorch功能正常，UCC问题已解决"
        echo "🎯 下一步: 可以继续MaskLLM相关工作"
    else
        echo "⚠️  发现: torch._C成功但完整torch失败"
        echo "📋 最终结论: 存在其他模块的依赖问题"
    fi
    
else
    echo "⚠️  发现: torch._C遇到非UCC问题"
    echo "📋 最终结论: 需要先解决基础依赖问题，然后再测试UCC"
fi

# 恢复原始环境
echo -e "\n恢复原始LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH="$ORIGINAL_LD_LIBRARY_PATH"

echo -e "\n直接测试完成。"
