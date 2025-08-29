#!/bin/bash

# MaskLLM 终极环境运行脚本
# 集成终极环境修复方案，彻底解决HPC-X冲突
# 专为NGC容器内稀疏训练优化

echo "🚀 MaskLLM 终极环境运行脚本"
echo "版本: Ultimate Environment Fix"
echo "目标: 彻底解决HPC-X冲突，为稀疏训练创建完美环境"
echo "=" * 80

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <脚本路径> [参数...]"
    echo ""
    echo "训练脚本示例:"
    echo "  $0 scripts/learnable_sparsity/llama2_7b_mask_only_tp4_c4.sh 0"
    echo "  $0 llama8b_scripts/llama8b_mask_only_tp8_c4_fixed.sh 0"
    echo ""
    echo "数据转换示例:"
    echo "  $0 llama8b_scripts/prepare_c4_megatron_llama8b.sh 0 0"
    echo ""
    echo "预稀疏模型生成:"
    echo "  $0 llama8b_scripts/run_llama8b_prune_tp8.sh"
    echo ""
    exit 1
fi

SCRIPT_PATH="$1"
shift

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    echo "启动容器: bash run_apptainer_extended_libs.sh"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo "运行脚本: $SCRIPT_PATH"
echo "参数: $@"
echo ""

# ================================================
# 应用终极环境修复
# ================================================
echo "📋 应用终极环境修复方案..."

# 1. 激进的HPC-X库路径隔离
CLEAN_LD_LIBRARY_PATH=""
NGC_PRIORITY_PATHS=(
    "/opt/conda_libs"                                                    # cuDNN主目录
    "/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib"    # cuPTI
    "/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib"          # NCCL
    "/usr/local/cuda/lib64"                                              # CUDA标准库
    "/usr/local/lib/python3.10/dist-packages/torch/lib"                 # PyTorch库
    "/usr/local/cuda/compat/lib"                                         # CUDA兼容库
    "/usr/local/nvidia/lib64"                                            # NVIDIA库
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
    fi
done

export LD_LIBRARY_PATH="$CLEAN_LD_LIBRARY_PATH"

# 2. 极端HPC-X环境变量清理
HPCX_ENV_VARS=(
    HPCX_DIR HPCX_HOME HPCX_ROOT HPCX_PREFIX
    HPCX_MPI_DIR HPCX_UCX_DIR HPCX_UCC_DIR HPCX_HCOLL_DIR HPCX_SHARP_DIR
    OMPI_HOME OMPI_ROOT MPI_HOME MPI_ROOT MPI_DIR
    UCX_DIR UCX_ROOT UCX_PREFIX UCC_DIR UCC_ROOT HCOLL_DIR SHARP_DIR
)

for var in "${HPCX_ENV_VARS[@]}"; do
    unset $var
done

# 3. 强制通信框架配置
export OMPI_MCA_pml=^ucx,^hcoll,^yalla
export OMPI_MCA_btl=^openib,^uct,^smcuda
export OMPI_MCA_coll=^hcoll,^ucx,^ml
export OMPI_MCA_osc=^ucx,^rdma
export OMPI_MCA_spml=^ucx,^yoda

export UCX_TLS=^all,^ud,^dc,^rc,^cm
export UCX_NET_DEVICES=^mlx
export UCX_LOG_LEVEL=fatal

export UCC_TLS=^all,^ucp,^sharp,^nccl
export UCC_CLS=^basic,^sharp
export UCC_LOG_LEVEL=fatal

export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0

# 4. 容器内Python环境优化
CURRENT_DIR="$(pwd)"
EXT_PKGS_DIR="${CURRENT_DIR}/.ext_pkgs"
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:$EXT_PKGS_DIR:$CURRENT_DIR"

# 清理conda污染
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PYTHON_EXE CONDA_EXE
unset CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_ENVS_PATH CONDA_PKGS_DIRS

# 5. CUDA环境
export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 6. LD_PRELOAD强制库加载顺序
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
    fi
done

if [ -n "$LD_PRELOAD_LIBS" ]; then
    export LD_PRELOAD="$LD_PRELOAD_LIBS"
fi

# 7. 其他优化设置
export PATH="/usr/local/bin:/usr/bin:/usr/sbin:/sbin:/usr/local/cuda/bin"
export HF_ENDPOINT=https://hf-mirror.com
export TORCHDYNAMO_DISABLE=1

echo "✅ 终极环境修复已应用"
echo ""

# ================================================
# 快速环境验证
# ================================================
echo "📋 快速环境验证..."

# 验证PyTorch
echo "🔍 验证PyTorch..."
if python3.10 -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>/dev/null; then
    echo "✅ PyTorch导入成功"
else
    echo "❌ PyTorch导入失败，但继续执行..."
fi

# 验证CUDA设备
echo "🔍 验证CUDA设备..."
if python3.10 -c "import torch; print('CUDA设备数:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null; then
    echo "✅ CUDA设备检测完成"
else
    echo "⚠️ CUDA设备检测异常，但继续执行..."
fi

echo ""

# ================================================
# 执行目标脚本
# ================================================
echo "🚀 开始执行目标脚本..."
echo "脚本: $SCRIPT_PATH"
echo "参数: $@"
echo "时间: $(date)"
echo ""

# 检测脚本类型并执行
if [[ "$SCRIPT_PATH" == *.py ]]; then
    # Python脚本
    echo "执行Python脚本..."
    python3.10 "$SCRIPT_PATH" "$@"
else
    # Bash脚本
    echo "执行Bash脚本..."
    bash "$SCRIPT_PATH" "$@"
fi

# 执行结果
EXIT_CODE=$?
echo ""
echo "脚本执行完成 (退出码: $EXIT_CODE)"
echo "时间: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 执行成功!"
else
    echo "❌ 执行失败 (退出码: $EXIT_CODE)"
fi

exit $EXIT_CODE
