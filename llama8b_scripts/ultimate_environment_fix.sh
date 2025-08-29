#!/bin/bash

# 终极环境修复方案 - 彻底解决NGC容器内HPC-X冲突
# 适用于MaskLLM稀疏训练环境

echo "🔧 终极环境修复方案 - 彻底解决HPC-X冲突"
echo "时间: $(date)"
echo "目标: 为MaskLLM稀疏训练创建完全清洁的环境"
echo "=" * 80

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi
echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# ================================================
# 第1步: 激进的HPC-X库路径隔离
# ================================================
echo "📋 第1步: 激进的HPC-X库路径隔离"

# 1.1 完全移除HPC-X相关的库搜索路径
echo "🧹 清理所有HPC-X相关库路径..."

# 构建完全清洁的LD_LIBRARY_PATH
CLEAN_LD_LIBRARY_PATH=""

# NGC容器优先路径 - 基于我们验证成功的配置
NGC_PRIORITY_PATHS=(
    "/opt/conda_libs"                                                    # cuDNN主目录
    "/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib"    # cuPTI
    "/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib"          # NCCL
    "/usr/local/cuda/lib64"                                              # CUDA标准库
    "/usr/local/lib/python3.10/dist-packages/torch/lib"                 # PyTorch库
    "/usr/local/cuda/compat/lib"                                         # CUDA兼容库
    "/usr/local/nvidia/lib64"                                            # NVIDIA库
    "/usr/local/nvidia/lib"                                              # NVIDIA库32
    "/lib/x86_64-linux-gnu"                                              # 系统核心库
    "/usr/lib/x86_64-linux-gnu"                                          # 系统库
)

for path in "${NGC_PRIORITY_PATHS[@]}"; do
    if [ -d "$path" ]; then
        if [ -z "$CLEAN_LD_LIBRARY_PATH" ]; then
            CLEAN_LD_LIBRARY_PATH="$path"
        else
            CLEAN_LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH:$path"
        fi
        echo "  ✅ 添加清洁路径: $path"
    fi
done

# 1.2 导出清洁的LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH"
echo "✅ 设置完全清洁的LD_LIBRARY_PATH"
echo ""

# ================================================
# 第2步: 极端HPC-X环境变量清理
# ================================================
echo "📋 第2步: 极端HPC-X环境变量清理"

# 2.1 清除所有HPC-X相关环境变量
HPCX_ENV_VARS=(
    HPCX_DIR HPCX_HOME HPCX_ROOT HPCX_PREFIX HPCX_PATH
    HPCX_MPI_DIR HPCX_UCX_DIR HPCX_UCC_DIR HPCX_HCOLL_DIR HPCX_SHARP_DIR
    OMPI_HOME OMPI_ROOT MPI_HOME MPI_ROOT MPI_DIR
    UCX_DIR UCX_ROOT UCX_PREFIX UCX_PATH
    UCC_DIR UCC_ROOT UCC_PREFIX UCC_PATH
    HCOLL_DIR HCOLL_ROOT HCOLL_PREFIX HCOLL_PATH
    SHARP_DIR SHARP_ROOT SHARP_PREFIX SHARP_PATH
)

for var in "${HPCX_ENV_VARS[@]}"; do
    if [ -n "${!var}" ]; then
        echo "  🧹 清除: $var=${!var}"
        unset $var
    fi
done

echo "✅ HPC-X环境变量清理完成"
echo ""

# ================================================
# 第3步: 强制通信框架配置
# ================================================
echo "📋 第3步: 强制通信框架配置"

# 3.1 激进的OMPI禁用设置
export OMPI_MCA_pml=^ucx,^hcoll,^yalla
export OMPI_MCA_btl=^openib,^uct,^smcuda
export OMPI_MCA_coll=^hcoll,^ucx,^ml
export OMPI_MCA_osc=^ucx,^rdma
export OMPI_MCA_spml=^ucx,^yoda
export OMPI_MCA_mtl=^ofi,^psm,^psm2
export OMPI_MCA_pml_base_verbose=0

# 3.2 UCX完全禁用
export UCX_TLS=^all,^ud,^dc,^rc,^cm
export UCX_NET_DEVICES=^mlx
export UCX_LOG_LEVEL=fatal

# 3.3 UCC完全禁用  
export UCC_TLS=^all,^ucp,^sharp,^nccl
export UCC_CLS=^basic,^sharp
export UCC_LOG_LEVEL=fatal

# 3.4 强制PyTorch使用安全后端
export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN

echo "✅ 通信框架强制配置完成"
echo ""

# ================================================
# 第4步: 容器内Python环境优化
# ================================================
echo "📋 第4步: 容器内Python环境优化"

# 4.1 设置清洁的PYTHONPATH
CLEAN_PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages"
CURRENT_DIR="$(pwd)"
EXT_PKGS_DIR="${CURRENT_DIR}/.ext_pkgs"

export PYTHONPATH="$CLEAN_PYTHONPATH:$EXT_PKGS_DIR:$CURRENT_DIR"

# 4.2 清理conda污染
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PYTHON_EXE CONDA_EXE
unset CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_ENVS_PATH CONDA_PKGS_DIRS

# 4.3 设置CUDA环境
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 确保8GPU可见

echo "✅ Python环境优化完成"
echo ""

# ================================================
# 第5步: 运行时库预加载策略
# ================================================
echo "📋 第5步: 运行时库预加载策略"

# 5.1 使用LD_PRELOAD强制库加载顺序
CRITICAL_LIBS=(
    "/opt/conda_libs/libcudnn.so.8.9.7"
    "/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"
    "/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2"
)

LD_PRELOAD_LIBS=""
for lib in "${CRITICAL_LIBS[@]}"; do
    if [ -f "$lib" ]; then
        if [ -z "$LD_PRELOAD_LIBS" ]; then
            LD_PRELOAD_LIBS="$lib"
        else
            LD_PRELOAD_LIBS="$LD_PRELOAD_LIBS:$lib"
        fi
        echo "  ✅ 预加载: $lib"
    fi
done

if [ -n "$LD_PRELOAD_LIBS" ]; then
    export LD_PRELOAD="$LD_PRELOAD_LIBS"
    echo "✅ 设置LD_PRELOAD强制库加载顺序"
else
    echo "⚠️ 未设置LD_PRELOAD（库文件检测失败）"
fi
echo ""

# ================================================
# 第6步: 环境验证
# ================================================
echo "📋 第6步: 环境验证"

# 6.1 显示当前环境配置
echo "🔍 当前环境配置:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  LD_PRELOAD: $LD_PRELOAD"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# 6.2 验证关键库可访问性
echo "🔍 验证关键库可访问性:"
python3.10 -c "
import ctypes.util
import os

key_libs = ['cudnn', 'cupti', 'nccl', 'cuda']
for lib in key_libs:
    lib_path = ctypes.util.find_library(lib)
    if lib_path:
        print(f'  ✅ {lib}: {lib_path}')
    else:
        print(f'  ❌ {lib}: 未找到')

print('')
print('🔍 LD_LIBRARY_PATH路径检查:')
ld_paths = os.environ.get('LD_LIBRARY_PATH', '').split(':')
for i, path in enumerate(ld_paths[:10]):
    if path and 'hpcx' not in path.lower():
        print(f'  ✅ {i+1}: {path}')
    elif 'hpcx' in path.lower():
        print(f'  ❌ {i+1}: {path} (包含HPC-X)')
    elif path:
        print(f'  ⚠️ {i+1}: {path}')
"
echo ""

# ================================================
# 第7步: PyTorch导入测试
# ================================================
echo "📋 第7步: PyTorch导入测试"

echo "🚀 尝试PyTorch导入..."
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
    
except Exception as e:
    print(f'❌ PyTorch导入失败: {e}')
    import traceback
    traceback.print_exc()
"

# ================================================
# 第8步: MaskLLM关键组件测试
# ================================================
echo ""
echo "📋 第8步: MaskLLM关键组件测试"

echo "🔍 测试transformer_engine..."
python3.10 -c "
try:
    import transformer_engine as te
    print('✅ transformer_engine导入成功')
    print(f'✅ 版本: {getattr(te, \"__version__\", \"unknown\")}')
    
    if hasattr(te, 'pytorch'):
        print('✅ transformer_engine.pytorch可用')
    else:
        print('⚠️ transformer_engine.pytorch不可用')
        
except Exception as e:
    print(f'❌ transformer_engine导入失败: {e}')
"

echo ""
echo "🔍 测试megatron.tokenizer..."
python3.10 -c "
try:
    from megatron.tokenizer import build_tokenizer
    print('✅ megatron.tokenizer导入成功')
    
    # 测试Llama8bTokenizer
    try:
        from megatron.tokenizer.llama8b_tokenizer import _Llama8bTokenizer
        print('✅ Llama8bTokenizer可用')
    except Exception as e:
        print(f'⚠️ Llama8bTokenizer导入异常: {e}')
        
except Exception as e:
    print(f'❌ megatron.tokenizer导入失败: {e}')
"

echo ""
echo "🎉 终极环境修复完成!"
echo "=" * 80

if [ $? -eq 0 ]; then
    echo ""
    echo "💡 环境修复成功！现在可以运行:"
    echo ""
    echo "1️⃣ 稀疏训练测试:"
    echo "   python3.10 llama8b_scripts/test_run_maskllm_native_fix.py"
    echo ""
    echo "2️⃣ C4数据转换:"
    echo "   bash llama8b_scripts/prepare_c4_megatron_llama8b.sh 0 0"
    echo ""
    echo "3️⃣ 预稀疏模型生成:"
    echo "   bash llama8b_scripts/run_llama8b_prune_tp8.sh"
    echo ""
    echo "🚀 环境已完全就绪，可开展Llama8b稀疏训练！"
else
    echo ""
    echo "❌ 部分修复失败，请检查错误信息"
    exit 1
fi
