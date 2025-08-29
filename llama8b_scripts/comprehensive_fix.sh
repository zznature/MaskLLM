#!/bin/bash

# 一站式综合修复脚本
# 解决PYTHONPATH传递和cuDNN库问题

echo "🔧 一站式综合修复"
echo "时间: $(date)"
echo "目标: 修复PYTHONPATH传递和cuDNN库问题"
echo "=================================="

# 1. 设置LD_LIBRARY_PATH（解决cuDNN问题）
echo "1️⃣ 修复cuDNN库配置"
export LD_LIBRARY_PATH="/opt/conda_libs:/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib:/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"
echo "✅ LD_LIBRARY_PATH已设置"

# 2. 设置PYTHONPATH（解决transformers导入问题）
echo "2️⃣ 修复PYTHONPATH配置"
CURRENT_DIR="$(pwd)"
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:${CURRENT_DIR}/.ext_pkgs:${CURRENT_DIR}"
echo "✅ PYTHONPATH已设置"

# 3. 设置UCC冲突禁用
echo "3️⃣ 禁用UCC冲突"
export NCCL_UCX_DISABLE=1
export TORCH_UCC_DISABLE=1
export UCC_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
echo "✅ UCC冲突禁用设置完成"

# 4. 验证修复效果
echo "4️⃣ 验证修复效果"

echo "验证PYTHONPATH..."
python3 -c "
import sys
print('Python sys.path前8个:')
for i, path in enumerate(sys.path[:8]):
    print(f'  {i+1}: {path}')

ext_pkgs_paths = [p for p in sys.path if '.ext_pkgs' in p]
if ext_pkgs_paths:
    print(f'\\n✅ 找到.ext_pkgs路径: {len(ext_pkgs_paths)}个')
    for path in ext_pkgs_paths:
        print(f'  📦 {path}')
else:
    print('\\n❌ 未找到.ext_pkgs路径')
"

echo ""
echo "验证transformers导入..."
python3 -c "
try:
    import transformers
    print('✅ transformers导入成功')
    print('✅ 版本:', transformers.__version__)
    print('✅ 路径:', transformers.__file__)
    
    if '.ext_pkgs' in transformers.__file__:
        print('📦 确认来自.ext_pkgs')
    else:
        print('🐳 来自容器内')
        
except ImportError as e:
    print('❌ transformers导入失败:', e)
"

echo ""
echo "验证PyTorch导入..."
python3 -c "
try:
    import torch
    print('✅ PyTorch导入成功')
    print('✅ 版本:', torch.__version__)
    
    # 测试基本功能
    x = torch.randn(2, 2)
    y = torch.mm(x, x)
    print('✅ 张量运算正常')
    
    if 'usr/local' in torch.__file__:
        print('🐳 确认使用容器内PyTorch')
        
except Exception as e:
    if 'ucc' in str(e).lower():
        print('❌ UCC冲突仍存在:', e)
    elif 'cudnn' in str(e).lower():
        print('❌ cuDNN问题仍存在:', e)
    else:
        print('❌ PyTorch导入失败:', e)
"

echo ""
echo "验证megatron.tokenizer..."
python3 -c "
try:
    from megatron.tokenizer import build_tokenizer
    print('✅ megatron.tokenizer导入成功')
except Exception as e:
    print('❌ megatron.tokenizer导入失败:', e)
"

echo ""
echo "🎯 综合修复验证完成"
echo ""

# 5. 生成环境设置脚本
echo "5️⃣ 生成环境设置脚本"

ENV_SCRIPT="container_environment_setup.sh"
cat > "$ENV_SCRIPT" << 'EOF'
#!/bin/bash

# 容器环境设置脚本 - 在容器内sourcing使用
# 用法: source container_environment_setup.sh

echo "🔧 设置容器环境变量..."

# LD_LIBRARY_PATH设置
export LD_LIBRARY_PATH="/opt/conda_libs:/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib:/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"

# PYTHONPATH设置
CURRENT_DIR="$(pwd)"
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:${CURRENT_DIR}/.ext_pkgs:${CURRENT_DIR}"

# UCC冲突禁用
export NCCL_UCX_DISABLE=1
export TORCH_UCC_DISABLE=1
export UCC_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo "✅ 环境变量设置完成"
echo "📦 PYTHONPATH包含.ext_pkgs: $(echo $PYTHONPATH | grep -o '.ext_pkgs' | wc -l)个路径"
echo "🐳 LD_LIBRARY_PATH包含容器库: $(echo $LD_LIBRARY_PATH | grep -o 'conda_libs' | wc -l)个路径"
EOF

chmod +x "$ENV_SCRIPT"

echo "✅ 环境设置脚本已生成: $ENV_SCRIPT"
echo ""

# 6. 使用建议
echo "6️⃣ 使用建议"
echo "=================================="
echo ""
echo "🚀 立即测试（当前shell有效）:"
echo "  python3 -c \"import transformers, torch; print('所有包导入成功!')\""
echo ""
echo "💾 持久化设置（下次进入容器时使用）:"
echo "  source container_environment_setup.sh"
echo ""
echo "🔧 集成到run_maskllm_native.sh:"
echo "  可以将环境设置集成到run_maskllm_native.sh中"
echo ""
echo "🧪 运行完整测试:"
echo "  bash llama8b_scripts/test_intelligent_environment.sh"
echo ""
echo "🎯 综合修复完成！"
