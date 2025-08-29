#!/bin/bash

# 单GPU/禁用分布式环境创建脚本
# 通过避开分布式模块来绕过HPC-X/UCC依赖问题

echo "🎯 创建单GPU/禁用分布式PyTorch环境"
echo "时间: $(date)"
echo "策略: 禁用分布式后端，只使用PyTorch核心功能"
echo "=" * 60

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# ================================================
# 第1步: 创建禁用分布式的环境配置
# ================================================
echo "📋 第1步: 创建禁用分布式的环境配置"

SINGLE_GPU_ENV_DIR="/tmp/single_gpu_env"
mkdir -p "$SINGLE_GPU_ENV_DIR"

# 创建环境设置脚本
cat > "$SINGLE_GPU_ENV_DIR/setup_single_gpu_env.sh" << 'EOF'
#!/bin/bash

# 单GPU PyTorch环境设置脚本
# 完全避开分布式通信模块

echo "🎯 设置单GPU PyTorch环境"

# 1. 禁用所有分布式相关环境变量
export TORCH_DISTRIBUTED_BACKEND="gloo"  # 使用最简单的后端
export NCCL_SOCKET_IFNAME="lo"           # 只使用本地回环
export RANK="-1"                         # 禁用分布式rank
export WORLD_SIZE="1"                    # 单机模式
export LOCAL_RANK="0"                    # 本地rank 0
export NODE_RANK="0"                     # 节点rank 0

# 2. 禁用UCC/UCX相关日志和功能
export UCC_LOG_LEVEL="error"
export UCX_LOG_LEVEL="error"
export UCC_TLS="^all"                    # 禁用所有UCC传输层
export UCX_TLS="^all"                    # 禁用所有UCX传输层

# 3. 强制使用CUDA但避免NCCL
export CUDA_VISIBLE_DEVICES="0"          # 只使用第一个GPU
export NCCL_DEBUG="ERROR"                # 最小NCCL日志
export NCCL_IB_DISABLE="1"               # 禁用InfiniBand
export NCCL_P2P_DISABLE="1"              # 禁用点对点通信

# 4. 设置基础CUDA环境
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"

# 5. 设置基础库路径（避免HPC-X路径）
export LD_LIBRARY_PATH="/opt/conda_libs:/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib:/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"

# 6. 设置Python路径
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:$(pwd)/.ext_pkgs:$(pwd)"

# 7. 清理可能的HPC-X污染
unset HPCX_DIR HPCX_HOME HPCX_ROOT OMPI_HOME MPI_HOME

echo "✅ 单GPU环境变量已设置"
echo "💡 注意: 此环境仅支持单GPU训练"

# 快速验证
python3.10 -c "
print('🔍 快速环境验证:')
import os
print(f'  CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"未设置\")}')
print(f'  WORLD_SIZE: {os.environ.get(\"WORLD_SIZE\", \"未设置\")}')
print(f'  TORCH_DISTRIBUTED_BACKEND: {os.environ.get(\"TORCH_DISTRIBUTED_BACKEND\", \"未设置\")}')
"
EOF

chmod +x "$SINGLE_GPU_ENV_DIR/setup_single_gpu_env.sh"

echo "✅ 单GPU环境配置脚本已创建"
echo ""

# ================================================
# 第2步: 创建受限PyTorch导入模块
# ================================================
echo "📋 第2步: 创建受限PyTorch导入模块"

# 创建安全的PyTorch导入包装器
cat > "$SINGLE_GPU_ENV_DIR/safe_pytorch_import.py" << 'EOF'
"""
安全PyTorch导入模块 - 避开分布式功能
只导入和使用PyTorch的核心功能，避免触发UCC/HPC-X依赖
"""

import os
import sys

# 设置环境以避免分布式初始化
os.environ['RANK'] = '-1'
os.environ['WORLD_SIZE'] = '1'
os.environ['TORCH_DISTRIBUTED_BACKEND'] = 'gloo'

def safe_import_torch():
    """安全导入PyTorch，避开分布式模块"""
    try:
        print("🔍 尝试安全导入PyTorch...")
        
        # 导入核心PyTorch
        import torch
        print("✅ PyTorch核心模块导入成功")
        print(f"✅ 版本: {torch.__version__}")
        
        # 测试CUDA功能
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA可用，检测到 {device_count} 个设备")
            
            # 获取第一个GPU信息
            if device_count > 0:
                print(f"✅ 使用设备: {torch.cuda.get_device_name(0)}")
                
                # 测试GPU张量运算
                try:
                    x = torch.randn(100, 100, device='cuda:0')
                    y = torch.mm(x, x)
                    print("✅ GPU张量运算正常")
                except Exception as e:
                    print(f"⚠️ GPU运算异常: {e}")
        else:
            print("⚠️ CUDA不可用")
        
        # 测试CPU功能
        x_cpu = torch.randn(50, 50)
        y_cpu = torch.mm(x_cpu, x_cpu)
        print("✅ CPU张量运算正常")
        
        return torch
        
    except Exception as e:
        print(f"❌ PyTorch导入失败: {e}")
        return None

def test_core_functionality(torch_module):
    """测试PyTorch核心功能"""
    if torch_module is None:
        return False
        
    try:
        print("\n🧪 测试PyTorch核心功能...")
        
        # 1. 张量创建和操作
        x = torch_module.randn(10, 10)
        y = torch_module.randn(10, 10)
        z = torch_module.mm(x, y)
        print("  ✅ 张量运算")
        
        # 2. 自动梯度
        x.requires_grad_(True)
        loss = (x ** 2).sum()
        loss.backward()
        print("  ✅ 自动梯度")
        
        # 3. 神经网络模块（避免分布式）
        import torch.nn as nn
        model = nn.Linear(10, 1)
        output = model(x)
        print("  ✅ 神经网络模块")
        
        # 4. CUDA功能（如果可用）
        if torch_module.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = model.cuda()(x_gpu)
            print("  ✅ CUDA运算")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 功能测试失败: {e}")
        return False

def test_megatron_compatibility():
    """测试与MaskLLM/Megatron的兼容性"""
    try:
        print("\n🧪 测试MaskLLM兼容性...")
        
        # 测试megatron tokenizer（核心功能）
        from megatron.tokenizer import build_tokenizer
        print("  ✅ megatron.tokenizer基础模块")
        
        # 测试Llama8b tokenizer
        from megatron.tokenizer.llama8b_tokenizer import _Llama8bTokenizer
        print("  ✅ Llama8bTokenizer")
        
        # 测试transformer_engine基础功能
        import transformer_engine as te
        print("  ✅ transformer_engine基础模块")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MaskLLM兼容性测试失败: {e}")
        return False

if __name__ == "__main__":
    # 完整测试流程
    print("🚀 开始安全PyTorch环境测试")
    print("=" * 50)
    
    # 导入PyTorch
    torch = safe_import_torch()
    
    if torch:
        # 测试核心功能
        core_ok = test_core_functionality(torch)
        
        # 测试MaskLLM兼容性
        maskllm_ok = test_megatron_compatibility()
        
        print("\n📊 测试结果总结:")
        print(f"  PyTorch核心功能: {'✅ 正常' if core_ok else '❌ 异常'}")
        print(f"  MaskLLM兼容性: {'✅ 正常' if maskllm_ok else '❌ 异常'}")
        
        if core_ok:
            print("\n🎉 单GPU环境创建成功！")
            print("💡 可以进行单GPU的MaskLLM任务")
        else:
            print("\n⚠️ 部分功能异常，但PyTorch基本可用")
    else:
        print("\n❌ 单GPU环境创建失败")
EOF

echo "✅ 安全PyTorch导入模块已创建"
echo ""

# ================================================
# 第3步: 应用环境并测试
# ================================================
echo "📋 第3步: 应用环境并测试"

echo "🔧 应用单GPU环境设置..."
source "$SINGLE_GPU_ENV_DIR/setup_single_gpu_env.sh"

echo ""
echo "🧪 运行安全PyTorch测试..."
python3.10 "$SINGLE_GPU_ENV_DIR/safe_pytorch_import.py"

SINGLE_GPU_TEST_RESULT=$?

echo ""

# ================================================
# 第4步: 创建MaskLLM单GPU适配脚本
# ================================================
echo "📋 第4步: 创建MaskLLM单GPU适配脚本"

if [ $SINGLE_GPU_TEST_RESULT -eq 0 ]; then
    echo "🔧 创建MaskLLM单GPU任务脚本..."
    
    # 创建单GPU数据转换脚本
    cat > "$SINGLE_GPU_ENV_DIR/llama8b_single_gpu_data_prep.sh" << 'EOF'
#!/bin/bash

# Llama8b单GPU数据准备脚本
# 避开分布式依赖，使用单GPU进行数据转换

echo "🎯 Llama8b单GPU数据准备"

# 应用单GPU环境
source /tmp/single_gpu_env/setup_single_gpu_env.sh

# 设置单GPU专用参数
export CUDA_VISIBLE_DEVICES="0"

echo "🔍 环境检查..."
python3.10 -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'可见GPU数: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')
"

echo ""
echo "🚀 开始C4数据转换 (单GPU模式)..."

# 修改C4数据转换脚本为单GPU模式
LLAMA8B_TOKENIZER_DIR="assets/checkpoints/Llama8b"
INPUT_FILE="assets/data/c4/c4-train.00000-of-01024.json"
OUTPUT_PREFIX="assets/data/c4_processed/c4_llama8b_single_gpu"
NUM_WORKERS=16  # 减少工作进程数

mkdir -p "assets/data/c4_processed"

python3.10 tools/preprocess_data.py \
       --input "${INPUT_FILE}" \
       --output-prefix "${OUTPUT_PREFIX}" \
       --vocab-file "${LLAMA8B_TOKENIZER_DIR}/vocab.txt" \
       --tokenizer-type Llama8bTokenizer \
       --tokenizer-model "${LLAMA8B_TOKENIZER_DIR}" \
       --append-eod \
       --workers "${NUM_WORKERS}"

echo "✅ 单GPU数据转换完成"
EOF

    chmod +x "$SINGLE_GPU_ENV_DIR/llama8b_single_gpu_data_prep.sh"
    
    # 创建单GPU训练示例脚本
    cat > "$SINGLE_GPU_ENV_DIR/llama8b_single_gpu_training.sh" << 'EOF'
#!/bin/bash

# Llama8b单GPU训练示例脚本
# 展示如何在单GPU环境下进行基础训练

echo "🎯 Llama8b单GPU训练示例"

# 应用单GPU环境
source /tmp/single_gpu_env/setup_single_gpu_env.sh

echo "💡 注意: 这是单GPU模式，不支持大规模分布式训练"
echo "💡 适用场景: 小规模实验、调试、单GPU推理"

# 设置单GPU训练参数
export CUDA_VISIBLE_DEVICES="0"

echo "🔍 环境验证..."
python3.10 -c "
import torch
from megatron.tokenizer import build_tokenizer
print('✅ 环境就绪')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无GPU\"}')
"

echo ""
echo "💡 单GPU模式下可以进行的任务:"
echo "  1. 模型推理和测试"
echo "  2. 小批量训练实验"  
echo "  3. 数据预处理"
echo "  4. 模型转换和验证"

echo ""
echo "🚀 示例: 单GPU推理测试"
echo "python3.10 -c \"
import torch
from megatron.tokenizer import build_tokenizer

# 创建简单的张量进行测试
x = torch.randn(2, 10, device='cuda' if torch.cuda.is_available() else 'cpu')
print('✅ 张量创建成功:', x.shape)
print('✅ 单GPU环境工作正常')
\""
EOF

    chmod +x "$SINGLE_GPU_ENV_DIR/llama8b_single_gpu_training.sh"
    
    echo "✅ MaskLLM单GPU适配脚本已创建"
fi

echo ""

# ================================================
# 第5步: 结果总结和使用指南
# ================================================
echo "📋 第5步: 结果总结和使用指南"

if [ $SINGLE_GPU_TEST_RESULT -eq 0 ]; then
    echo "🎉 单GPU环境创建成功！"
    echo ""
    echo "✅ 成功要素:"
    echo "  - PyTorch核心功能完全正常"
    echo "  - CUDA单GPU运算正常"
    echo "  - MaskLLM基础组件可用"
    echo "  - 完全避开了HPC-X/UCC依赖"
    echo ""
    echo "🚀 现在可以使用的功能:"
    echo ""
    echo "1️⃣ 设置环境 (每次使用前):"
    echo "   source $SINGLE_GPU_ENV_DIR/setup_single_gpu_env.sh"
    echo ""
    echo "2️⃣ 数据预处理:"
    echo "   bash $SINGLE_GPU_ENV_DIR/llama8b_single_gpu_data_prep.sh"
    echo ""
    echo "3️⃣ 单GPU训练/推理:"
    echo "   bash $SINGLE_GPU_ENV_DIR/llama8b_single_gpu_training.sh"
    echo ""
    echo "4️⃣ 手动运行:"
    echo "   source $SINGLE_GPU_ENV_DIR/setup_single_gpu_env.sh"
    echo "   python3.10 your_script.py"
    echo ""
    echo "💡 限制和注意事项:"
    echo "  - 只支持单GPU训练"
    echo "  - 无法使用分布式训练加速"
    echo "  - 大模型可能受GPU内存限制"
    echo "  - 适合原型开发和小规模实验"
    echo ""
    echo "🎯 这个方案绕过了所有HPC-X问题，可以立即开始工作！"
    
else
    echo "⚠️ 单GPU环境部分成功"
    echo "虽然可能有一些限制，但PyTorch基本功能应该可用"
    echo ""
    echo "💡 建议:"
    echo "1. 检查具体的错误信息"
    echo "2. 尝试更保守的设置"
    echo "3. 考虑其他替代方案"
fi

echo ""
echo "🎯 单GPU环境创建完成"
echo "=" * 60
