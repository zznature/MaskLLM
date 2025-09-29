#!/bin/bash

# Minimal MaskLLM native launcher
# - keeps core environment alignment with container defaults
# - skips heavy environment validation to accelerate workflow

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script> [args...]"
    exit 1
fi

SCRIPT_PATH="$1"
shift

PROJECT_ROOT="$(pwd)"

echo "=== MaskLLM native (simple) ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Target script: ${SCRIPT_PATH}"

# HuggingFace mirrors / timeout
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=300

# PyTorch dynamo controls
export TORCHDYNAMO_DISABLE=1
export TORCHDYNAMO_VERBOSE=1

# UCX / UCC / NCCL compatibility toggles (mirrors full launcher defaults)
export OMPI_MCA_pml="^ucx,^hcoll,^ucc"
export OMPI_MCA_btl="^openib,^uct,^ucx"
export OMPI_MCA_coll="^hcoll,^ucx,^ucc"
export OMPI_MCA_osc="^ucx,^ucc"
export OMPI_MCA_spml="^ucx,^ucc"
export UCX_TLS="^ud,^dc,^rc,^tcp,^shm"
export UCC_TLS="^ucp,^sharp,^tcp,^shm,^cuda"
export UCC_CLS="^sharp,^basic"
export UCX_LOG_LEVEL=error
export UCC_LOG_LEVEL=error
export NCCL_UCX_DISABLE=1
export TORCH_UCC_DISABLE=1
export UCC_DISABLE=1
export UCX_DISABLE=1
export TORCH_DISTRIBUTED_BACKEND=gloo
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=ring
export PYTORCH_CUDA_ALLOC_CONF=backend:native

# Base PATH / PYTHONPATH
export PATH="/usr/local/bin:/usr/bin:/usr/sbin:/sbin:/usr/local/cuda/bin"
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/usr/local/lib/python3.10/site-packages:${PROJECT_ROOT}/.ext_pkgs:${PROJECT_ROOT}"

# Construct LD_LIBRARY_PATH from known directories (include if present)
declare -a LD_PATHS
add_ld_path() {
    local dir="$1"
    if [ -d "$dir" ]; then
        LD_PATHS+=("$dir")
    fi
}

add_ld_path "${PROJECT_ROOT}/ucx_ucc_install_optimized/lib"
add_ld_path "/opt/conda_libs"
add_ld_path "/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib"
add_ld_path "/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib"
add_ld_path "/usr/local/cuda/lib64"
add_ld_path "/usr/local/lib/python3.10/dist-packages/torch/lib"
add_ld_path "/usr/local/cuda/compat/lib"
add_ld_path "/usr/local/nvidia/lib"
add_ld_path "/usr/local/nvidia/lib64"
add_ld_path "/usr/lib/x86_64-linux-gnu"

if [ ${#LD_PATHS[@]} -gt 0 ]; then
    export LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PATHS[*]}")"
fi

export CUDA_HOME="/usr/local/cuda"
export CUDA_ROOT="/usr/local/cuda"

# Clean conda markers to avoid interference
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PYTHON_EXE CONDA_EXE \
      CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_ENVS_PATH CONDA_PKGS_DIRS

echo "Environment ready. PATH=${PATH}"

# Execute target script
if [[ "$SCRIPT_PATH" == *.py ]]; then
    python3 "$SCRIPT_PATH" "$@"
else
    bash "$SCRIPT_PATH" "$@"
fi

