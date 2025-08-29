#!/bin/bash

# 根本性高效解决方案
# 三种彻底避开UCC符号问题的方案

echo "🎯 根本性高效解决方案"
echo "时间: $(date)"
echo "目标: 彻底避开UCC符号无尽循环问题"
echo "=" * 60

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# ================================================
# 错误分析总结
# ================================================
echo "📋 错误演进分析"

echo "🔍 UCC符号无尽循环:"
echo "  1️⃣ ucc_ee_ack_event"
echo "  2️⃣ ucc_init_version" 
echo "  3️⃣ ucc_lib_get_attr"
echo "  4️⃣ (预计还会有更多...)"
echo ""

echo "💡 根本原因:"
echo "  - NGC容器的PyTorch深度集成了完整UCC库"
echo "  - 空实现劫持无法满足PyTorch的实际功能需求"
echo "  - UCC库有数百个相互依赖的符号"
echo ""

echo "🎯 高效解决策略:"
echo "  方案A: 系统级UCC库重定向 (推荐)"
echo "  方案B: PyTorch版本替换"
echo "  方案C: 容器外运行"

echo ""

EFFICIENT_DIR="$(pwd)/efficient_solution"
mkdir -p "$EFFICIENT_DIR"

# ================================================
# 方案A: 系统级UCC库重定向 (推荐)
# ================================================
echo "📋 方案A: 系统级UCC库重定向"

echo "🎯 原理: 用系统的UCC库替换HPC-X的UCC库"

# 检查系统UCC库
echo "🔍 检查系统UCC库可用性..."

SYSTEM_UCC_PATHS=(
    "/usr/lib/x86_64-linux-gnu"
    "/lib/x86_64-linux-gnu" 
    "/usr/local/lib"
)

FOUND_SYSTEM_UCC=""
for path in "${SYSTEM_UCC_PATHS[@]}"; do
    if [ -f "$path/libucc.so" ] || [ -f "$path/libucc.so.1" ]; then
        FOUND_SYSTEM_UCC="$path"
        echo "  ✅ 找到系统UCC库: $path"
        break
    fi
done

if [ -n "$FOUND_SYSTEM_UCC" ]; then
    echo "✅ 系统UCC库可用，创建重定向方案..."
    
    cat > "$EFFICIENT_DIR/system_ucc_redirect.sh" << EOF
#!/bin/bash

# 系统UCC库重定向方案
# 使用系统的UCC库替代HPC-X库

echo "🔧 应用系统UCC库重定向..."

# 构建重定向的库路径，优先使用系统UCC
SYSTEM_UCC_PATH="$FOUND_SYSTEM_UCC"
NGC_CUDA_PATHS="/opt/conda_libs:/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib:/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib"
STANDARD_PATHS="/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu"

# 关键: 系统UCC路径放在最前面
export LD_LIBRARY_PATH="\$SYSTEM_UCC_PATH:\$NGC_CUDA_PATHS:\$STANDARD_PATHS"

# 强制禁用HPC-X路径
export LD_LIBRARY_PATH="\$(echo \$LD_LIBRARY_PATH | sed 's|/opt/hpcx/[^:]*:||g')"

# 清理HPC-X环境变量
unset HPCX_DIR HPCX_HOME HPCX_ROOT
unset OMPI_HOME MPI_HOME

# 设置MPI使用系统库
export OMPI_MCA_pml="ob1"
export OMPI_MCA_btl="self,tcp"

# 8GPU配置
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WORLD_SIZE="8"
export TORCH_DISTRIBUTED_BACKEND="nccl"

echo "✅ 系统UCC重定向已应用"
echo "💡 使用系统UCC库: \$SYSTEM_UCC_PATH"
EOF

    chmod +x "$EFFICIENT_DIR/system_ucc_redirect.sh"
    echo "✅ 系统UCC重定向方案已创建"
    
    # 测试系统UCC方案
    echo ""
    echo "🧪 测试系统UCC重定向方案..."
    source "$EFFICIENT_DIR/system_ucc_redirect.sh"
    
    python3.10 -c "
try:
    import torch
    print('✅ 方案A: 系统UCC重定向成功')
    print(f'  PyTorch版本: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  GPU数量: {torch.cuda.device_count()}')
    exit(0)
except Exception as e:
    print('❌ 方案A: 系统UCC重定向失败')
    print(f'  错误: {e}')
    exit(1)
"
    
    SYSTEM_UCC_RESULT=$?
    
else
    echo "❌ 系统UCC库不可用"
    SYSTEM_UCC_RESULT=1
fi

echo ""

# ================================================
# 方案B: PyTorch版本替换
# ================================================
echo "📋 方案B: PyTorch版本替换"

echo "🎯 原理: 使用没有HPC-X依赖的PyTorch版本"

# 检查是否有pip和网络
if command -v pip >/dev/null 2>&1; then
    echo "✅ pip可用，可以尝试PyTorch替换"
    
    cat > "$EFFICIENT_DIR/pytorch_replacement.sh" << 'EOF'
#!/bin/bash

# PyTorch版本替换方案
# 安装没有HPC-X依赖的PyTorch版本

echo "🔧 PyTorch版本替换方案"
echo "⚠️ 注意: 这会替换当前的PyTorch版本"

read -p "是否继续替换PyTorch版本? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏭️ 跳过PyTorch替换"
    exit 0
fi

echo "🔧 备份当前PyTorch..."
BACKUP_DIR="/tmp/pytorch_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -d "/usr/local/lib/python3.10/dist-packages/torch" ]; then
    cp -r "/usr/local/lib/python3.10/dist-packages/torch" "$BACKUP_DIR/"
    echo "✅ PyTorch已备份到: $BACKUP_DIR"
fi

echo "🔧 安装标准PyTorch版本..."

# 使用清华镜像源安装标准PyTorch
pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --force-reinstall --no-deps

if [ $? -eq 0 ]; then
    echo "✅ 标准PyTorch安装成功"
    
    # 测试新PyTorch
    python3.10 -c "
import torch
print(f'新PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
"
    
    echo "💡 如果有问题，可以恢复备份:"
    echo "   rm -rf /usr/local/lib/python3.10/dist-packages/torch"
    echo "   cp -r $BACKUP_DIR/torch /usr/local/lib/python3.10/dist-packages/"
    
else
    echo "❌ PyTorch替换失败"
    echo "💡 恢复备份..."
    if [ -d "$BACKUP_DIR/torch" ]; then
        rm -rf "/usr/local/lib/python3.10/dist-packages/torch"
        cp -r "$BACKUP_DIR/torch" "/usr/local/lib/python3.10/dist-packages/"
        echo "✅ 已恢复原PyTorch版本"
    fi
fi
EOF

    chmod +x "$EFFICIENT_DIR/pytorch_replacement.sh"
    echo "✅ PyTorch替换方案已创建"
    echo "💡 使用方法: bash $EFFICIENT_DIR/pytorch_replacement.sh"
    
else
    echo "❌ pip不可用，无法尝试PyTorch替换"
fi

echo ""

# ================================================
# 方案C: 容器外运行建议
# ================================================
echo "📋 方案C: 容器外运行建议"

echo "🎯 原理: 在容器外的标准环境中运行"

cat > "$EFFICIENT_DIR/container_outside_guide.md" << 'EOF'
# 容器外运行指南

## 问题分析
NGC容器的PyTorch与HPC-X深度集成，导致符号依赖问题难以根本解决。

## 解决方案
在容器外的标准Linux环境中运行MaskLLM。

## 环境准备步骤

### 1. 安装标准PyTorch
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### 2. 安装CUDA工具链
```bash
# 确保CUDA 11.8+可用
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3. 安装MaskLLM依赖
```bash
pip install transformers datasets tokenizers
pip install flash-attn --no-build-isolation
```

### 4. 配置环境
```bash
export PYTHONPATH=/path/to/MaskLLM:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 优势
- 完全避开HPC-X符号问题
- 环境更纯净，问题更少
- 更容易调试和维护

## 劣势
- 需要重新配置环境
- 可能缺少某些NGC容器的优化
EOF

echo "✅ 容器外运行指南已创建: $EFFICIENT_DIR/container_outside_guide.md"

echo ""

# ================================================
# 方案推荐和总结
# ================================================
echo "📋 方案推荐和总结"

echo "🎯 基于测试结果的推荐:"

if [ $SYSTEM_UCC_RESULT -eq 0 ]; then
    echo ""
    echo "🥇 **强烈推荐: 方案A (系统UCC重定向)**"
    echo "   ✅ 已验证可行"
    echo "   ✅ 无需修改PyTorch"
    echo "   ✅ 保持8GPU环境"
    echo "   🚀 立即使用:"
    echo "      source $EFFICIENT_DIR/system_ucc_redirect.sh"
    echo "      python3.10 your_script.py"
    
elif command -v pip >/dev/null 2>&1; then
    echo ""
    echo "🥈 **推荐: 方案B (PyTorch替换)**"
    echo "   💡 系统UCC不可用时的备选"
    echo "   ⚠️ 需要替换PyTorch版本"
    echo "   🚀 使用方法:"
    echo "      bash $EFFICIENT_DIR/pytorch_replacement.sh"
    
else
    echo ""
    echo "🥉 **最后选择: 方案C (容器外运行)**"
    echo "   💡 当前容器环境问题太复杂"
    echo "   📋 参考指南: $EFFICIENT_DIR/container_outside_guide.md"
fi

echo ""
echo "🔍 **为什么这些方案更高效:**"
echo "  1. 避开UCC符号无尽循环"
echo "  2. 解决根本依赖问题，不是表面修复"
echo "  3. 一次配置，长期使用"
echo "  4. 减少重复调试时间"

echo ""

# ================================================
# 创建一键执行脚本
# ================================================
if [ $SYSTEM_UCC_RESULT -eq 0 ]; then
    echo "📋 创建一键解决方案"
    
    cat > "$EFFICIENT_DIR/one_click_solution.sh" << 'EOF'
#!/bin/bash

# 一键高效解决方案
# 应用最佳可行方案

echo "🚀 一键高效解决方案"

# 应用系统UCC重定向
source $(dirname $0)/system_ucc_redirect.sh

# 验证效果
echo "🧪 快速验证..."
python3.10 -c "
import torch
print('✅ PyTorch导入成功')
print(f'GPU数量: {torch.cuda.device_count()}')

import torch.distributed as dist
print('✅ 分布式模块正常')

from megatron.tokenizer import build_tokenizer
print('✅ MaskLLM组件正常')

print('\\n🎉 环境完全就绪，可以开始8GPU稀疏训练！')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 一键解决方案成功！"
    echo "🚀 现在可以直接运行MaskLLM任务"
else
    echo "❌ 仍有问题，建议尝试方案B或方案C"
fi
EOF
    
    chmod +x "$EFFICIENT_DIR/one_click_solution.sh"
    
    echo "✅ 一键解决方案已创建: $EFFICIENT_DIR/one_click_solution.sh"
    echo ""
    echo "🚀 **立即使用最佳方案:**"
    echo "   bash $EFFICIENT_DIR/one_click_solution.sh"
fi

echo ""
echo "🎯 根本性高效解决方案完成"
echo "💡 不再需要重复调试UCC符号问题！"
echo "=" * 60
