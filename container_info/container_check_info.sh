check_container_info.sh 脚本容器信息结果

=== 检查 pytorch_24.01-py3.sif 容器信息 ===

1. 文件基本信息:
-rwxrwxr-x 1 zdhs0054 zdhs0054 9.5G Jun 14 08:19 pytorch_24.01-py3.sif

2. 文件修改时间:
  File: pytorch_24.01-py3.sif
  Size: 10135441408     Blocks: 19795792   IO Block: 1048576 regular file
Device: 33h/51d Inode: 122629164   Links: 1
Access: (0775/-rwxrwxr-x)  Uid: (10054/zdhs0054)   Gid: (10054/zdhs0054)
Access: 2025-07-06 00:15:21.740957411 +0000
Modify: 2025-06-14 08:19:14.905669192 +0000
Change: 2025-06-14 08:19:14.905669192 +0000
 Birth: -

3. 容器标签信息:
com.nvidia.build.id: 80741402
com.nvidia.build.ref: 3a8f39e58d71996b362a9358b971d42d695351fd
com.nvidia.cublas.version: 12.3.4.1
com.nvidia.cuda.version: 9.0
com.nvidia.cudnn.version: 8.9.7.29+cuda12.2
com.nvidia.cufft.version: 11.0.12.1
com.nvidia.curand.version: 10.3.4.107
com.nvidia.cusolver.version: 11.5.4.101
com.nvidia.cusparse.version: 12.2.0.103
com.nvidia.cutensor.version: 2.0.0.7
com.nvidia.nccl.version: 2.19.4
com.nvidia.npp.version: 12.2.3.2
com.nvidia.nsightcompute.version: 2023.3.1.1
com.nvidia.nsightsystems.version: 2023.4.1.97
com.nvidia.nvjpeg.version: 12.3.0.81
com.nvidia.pytorch.version: 2.2.0a0+81ea7a4
com.nvidia.tensorrt.version: 8.6.1.6+cuda12.0.1.011
com.nvidia.tensorrtoss.version: 23.11
com.nvidia.volumes.needed: nvidia_driver
org.label-schema.build-arch: amd64
org.label-schema.build-date: Saturday_14_June_2025_16:17:37_CST
org.label-schema.schema-version: 1.0
org.label-schema.usage.apptainer.version: 1.3.2-dirty
org.label-schema.usage.singularity.deffile.bootstrap: docker
org.label-schema.usage.singularity.deffile.from: nvcr.io/nvidia/pytorch:24.01-py3
org.opencontainers.image.ref.name: ubuntu
org.opencontainers.image.version: 22.04

4. 容器环境变量:
=== /.singularity.d/env/10-docker2singularity.sh ===
#!/bin/sh
export PATH="/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin"
export CUDA_VERSION="${CUDA_VERSION:-"12.3.2.001"}"
export CUDA_DRIVER_VERSION="${CUDA_DRIVER_VERSION:-"545.23.08"}"
export CUDA_CACHE_DISABLE="${CUDA_CACHE_DISABLE:-"1"}"
export NVIDIA_REQUIRE_JETPACK_HOST_MOUNTS="${NVIDIA_REQUIRE_JETPACK_HOST_MOUNTS:-""}"
export _CUDA_COMPAT_PATH="${_CUDA_COMPAT_PATH:-"/usr/local/cuda/compat"}"
export ENV="${ENV:-"/etc/shinit_v2"}"
export BASH_ENV="${BASH_ENV:-"/etc/bash.bashrc"}"
export SHELL="${SHELL:-"/bin/bash"}"
export NVIDIA_REQUIRE_CUDA="${NVIDIA_REQUIRE_CUDA:-"cuda>=9.0"}"
export NCCL_VERSION="${NCCL_VERSION:-"2.19.4"}"
export CUBLAS_VERSION="${CUBLAS_VERSION:-"12.3.4.1"}"
export CUFFT_VERSION="${CUFFT_VERSION:-"11.0.12.1"}"
export CURAND_VERSION="${CURAND_VERSION:-"10.3.4.107"}"
export CUSPARSE_VERSION="${CUSPARSE_VERSION:-"12.2.0.103"}"
export CUSOLVER_VERSION="${CUSOLVER_VERSION:-"11.5.4.101"}"
export CUTENSOR_VERSION="${CUTENSOR_VERSION:-"2.0.0.7"}"
export NPP_VERSION="${NPP_VERSION:-"12.2.3.2"}"
export NVJPEG_VERSION="${NVJPEG_VERSION:-"12.3.0.81"}"
export CUDNN_VERSION="${CUDNN_VERSION:-"8.9.7.29+cuda12.2"}"
export TRT_VERSION="${TRT_VERSION:-"8.6.1.6+cuda12.0.1.011"}"
export TRTOSS_VERSION="${TRTOSS_VERSION:-"23.11"}"
export NSIGHT_SYSTEMS_VERSION="${NSIGHT_SYSTEMS_VERSION:-"2023.4.1.97"}"
export NSIGHT_COMPUTE_VERSION="${NSIGHT_COMPUTE_VERSION:-"2023.3.1.1"}"
export DALI_VERSION="${DALI_VERSION:-"1.33.0"}"
export DALI_BUILD="${DALI_BUILD:-"11414174"}"
export POLYGRAPHY_VERSION="${POLYGRAPHY_VERSION:-"0.49.1"}"
export TRANSFORMER_ENGINE_VERSION="${TRANSFORMER_ENGINE_VERSION:-"1.2"}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-"/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"}"
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-"all"}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-"compute,utility,video"}"
export NVIDIA_PRODUCT_NAME="${NVIDIA_PRODUCT_NAME:-"PyTorch"}"
export GDRCOPY_VERSION="${GDRCOPY_VERSION:-"2.3"}"
export HPCX_VERSION="${HPCX_VERSION:-"2.16rc4"}"
export MOFED_VERSION="${MOFED_VERSION:-"5.4-rdmacore39.0"}"
export OPENUCX_VERSION="${OPENUCX_VERSION:-"1.15.0"}"
export OPENMPI_VERSION="${OPENMPI_VERSION:-"4.1.5rc2"}"
export RDMACORE_VERSION="${RDMACORE_VERSION:-"39.0"}"
export OPAL_PREFIX="${OPAL_PREFIX:-"/opt/hpcx/ompi"}"
export OMPI_MCA_coll_hcoll_enable="${OMPI_MCA_coll_hcoll_enable:-"0"}"
export LIBRARY_PATH="${LIBRARY_PATH:-"/usr/local/cuda/lib64/stubs:"}"
export PYTORCH_BUILD_VERSION="${PYTORCH_BUILD_VERSION:-"2.2.0a0+81ea7a4"}"
export PYTORCH_VERSION="${PYTORCH_VERSION:-"2.2.0a0+81ea7a4"}"
export PYTORCH_BUILD_NUMBER="${PYTORCH_BUILD_NUMBER:-"0"}"
export NVIDIA_PYTORCH_VERSION="${NVIDIA_PYTORCH_VERSION:-"24.01"}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-"python"}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-"utf-8"}"
export LC_ALL="${LC_ALL:-"C.UTF-8"}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-"100"}"
export NVM_DIR="${NVM_DIR:-"/usr/local/nvm"}"
export JUPYTER_PORT="${JUPYTER_PORT:-"8888"}"
export TENSORBOARD_PORT="${TENSORBOARD_PORT:-"6006"}"
export UCC_CL_BASIC_TLS="${UCC_CL_BASIC_TLS:-"^sharp"}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-"5.2 6.0 6.1 7.0 7.2 7.5 8.0 8.6 8.7 9.0+PTX"}"
export PYTORCH_HOME="${PYTORCH_HOME:-"/opt/pytorch/pytorch"}"
export CUDA_HOME="${CUDA_HOME:-"/usr/local/cuda"}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-"1"}"
export USE_EXPERIMENTAL_CUDNN_V8_API="${USE_EXPERIMENTAL_CUDNN_V8_API:-"1"}"
export COCOAPI_VERSION="${COCOAPI_VERSION:-"2.0+nv0.8.0"}"
export TORCH_CUDNN_V8_API_ENABLED="${TORCH_CUDNN_V8_API_ENABLED:-"1"}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-"LAZY"}"
export NVIDIA_BUILD_ID="${NVIDIA_BUILD_ID:-"80741402"}"

=== /.singularity.d/env/90-environment.sh ===
#!/bin/sh
# Copyright (c) Contributors to the Apptainer project, established as
#   Apptainer a Series of LF Projects LLC.
#   For website terms of use, trademark policy, privacy policy and other
#   project policies see https://lfprojects.org/policies
# Copyright (c) 2018-2021, Sylabs Inc. All rights reserved.
# This software is licensed under a 3-clause BSD license. Please consult
# https://github.com/apptainer/apptainer/blob/main/LICENSE.md regarding your
# rights to use or distribute this software.

# Custom environment shell code should follow


5. 容器运行脚本:
#!/bin/sh
OCI_ENTRYPOINT='"/opt/nvidia/nvidia_entrypoint.sh"'
OCI_CMD=''

# When SINGULARITY_NO_EVAL set, use OCI compatible behavior that does
# not evaluate resolved CMD / ENTRYPOINT / ARGS through the shell, and
# does not modify expected quoting behavior of args.
if [ -n "$SINGULARITY_NO_EVAL" ]; then
    # ENTRYPOINT only - run entrypoint plus args
    if [ -z "$OCI_CMD" ] && [ -n "$OCI_ENTRYPOINT" ]; then
        set -- '/opt/nvidia/nvidia_entrypoint.sh' "$@"

        exec "$@"
    fi

    # CMD only - run CMD or override with args
    if [ -n "$OCI_CMD" ] && [ -z "$OCI_ENTRYPOINT" ]; then
        exec "$@"
    fi

    # ENTRYPOINT and CMD - run ENTRYPOINT with CMD as default args
    # override with user provided args
    if [ $# -gt 0 ]; then
        set -- '/opt/nvidia/nvidia_entrypoint.sh' "$@"

        else
        
        set -- '/opt/nvidia/nvidia_entrypoint.sh' "$@"

    fi
    exec "$@"
fi

# Standard Apptainer behavior evaluates CMD / ENTRYPOINT / ARGS
# combination through shell before exec, and requires special quoting
# due to concatenation of CMDLINE_ARGS.
CMDLINE_ARGS=""
# prepare command line arguments for evaluation
for arg in "$@"; do
        CMDLINE_ARGS="${CMDLINE_ARGS} \"$arg\""
done

# ENTRYPOINT only - run entrypoint plus args
if [ -z "$OCI_CMD" ] && [ -n "$OCI_ENTRYPOINT" ]; then
    if [ $# -gt 0 ]; then
        SINGULARITY_OCI_RUN="${OCI_ENTRYPOINT} ${CMDLINE_ARGS}"
    else
        SINGULARITY_OCI_RUN="${OCI_ENTRYPOINT}"
    fi
fi

# CMD only - run CMD or override with args
if [ -n "$OCI_CMD" ] && [ -z "$OCI_ENTRYPOINT" ]; then
    if [ $# -gt 0 ]; then
        SINGULARITY_OCI_RUN="${CMDLINE_ARGS}"
    else
        SINGULARITY_OCI_RUN="${OCI_CMD}"
    fi
fi

# ENTRYPOINT and CMD - run ENTRYPOINT with CMD as default args
# override with user provided args
if [ $# -gt 0 ]; then
    SINGULARITY_OCI_RUN="${OCI_ENTRYPOINT} ${CMDLINE_ARGS}"
else
    SINGULARITY_OCI_RUN="${OCI_ENTRYPOINT} ${OCI_CMD}"
fi

# Evaluate shell expressions first and set arguments accordingly,
# then execute final command as first container process
eval "set ${SINGULARITY_OCI_RUN}"
exec "$@"

6. 容器启动脚本:
#!/bin/sh
# Copyright (c) Contributors to the Apptainer project, established as
#   Apptainer a Series of LF Projects LLC.
#   For website terms of use, trademark policy, privacy policy and other
#   project policies see https://lfprojects.org/policies
# Copyright (c) 2018-2021, Sylabs Inc. All rights reserved.
# This software is licensed under a 3-clause BSD license. Please consult the
# LICENSE.md file distributed with the sources of this project regarding your
# rights to use or distribute this software.

7. 容器测试脚本:

8. 容器定义文件:
bootstrap: docker
from: nvcr.io/nvidia/pytorch:24.01-py3

9. 检查容器内的 Python 环境:
Python 3.10.12

10. 检查容器内的 PyTorch:
PyTorch 版本: 2.2.0a0+81ea7a4

11. 检查容器内的 transformer_engine:
transformer_engine 可用

=== 容器信息检查完成 ===