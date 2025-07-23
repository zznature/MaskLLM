# 项目任务

## HPC服务器节点配置

登录节点: hd02-gpfs-quorum-02
GPU计算节点: hd02-gpu1-0056,hd02-gpu1-0024,hd02-gpu1-0017,hd02-gpu1-0029
登录方法: `ssh hd02-gpu1-0017`, optional `srun --jobid=4879 --pty bash -i`

### 加载模块
```bash
source /etc/profile.d/modules.sh
module load apptainer 
module load cuda/12.4
```

### 容器信息
`pytorch_24.01-py3.sif` 是 NVIDIA 官方的 PyTorch NGC 容器，包含完整的 MaskLLM 运行环境：

**预装组件：**
- Python 3.10.12
- PyTorch 2.2.0a0+81ea7a4
- transformer_engine 1.2
- CUDA 12.3.2.001
- cuDNN 8.9.7.29
- NCCL 2.19.4
- TensorRT 8.6.1.6
- torchrun: /usr/local/bin/torchrun

### 启动容器

#### 混合库路径原生环境（推荐）
```bash
bash run_apptainer_extended_libs.sh
```

#### 运行 MaskLLM 脚本（推荐）
```bash
bash run_container_simple.sh scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
```

### 环境验证
进入容器后，运行环境验证脚本：
```bash
bash verify_container_env.sh
```

### 运行 MaskLLM 任务

#### 方法1：使用原生环境脚本（推荐）
```bash
# 在容器内运行
bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4_fixed.sh hessian
```

#### 方法2：直接运行（已修复torchrun路径）
```bash
# 在容器内运行
bash scripts/oneshot/run_llama2_7b_prune_tp4_fixed.sh hessian
```

#### 方法3：手动运行
```bash
# 在容器内，使用容器内的torchrun
/usr/local/bin/torchrun --nproc_per_node=4 --master_port=29500 tasks/main.py --config scripts/oneshot/llama2_7b_prune_tp4.yaml --pruning_method hessian
```

### 故障排除

#### 环境问题
1. **验证容器环境**：
   ```bash
   bash verify_container_env.sh
   ```

2. **检查容器信息**：
   ```bash
   bash check_container_info.sh
   ```

#### 常见问题
1. **torchrun 路径问题**：已修复，使用容器内的 `/usr/local/bin/torchrun`
2. **PyTorch C扩展问题**：已通过混合库路径方案解决
3. **transformer_engine 问题**：容器已预装 transformer_engine 1.2
4. **CUDA 问题**：容器已配置完整的 CUDA 12.3.2.001 环境
5. **cuDNN 库问题**：已通过混合库路径解决

#### 环境验证命令
```bash
# 检查 Python 环境
python --version
which python

# 检查 PyTorch
python -c "import torch; print('PyTorch 版本:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available())"

# 检查 transformer_engine
python -c "import transformer_engine; print('transformer_engine 可用')"

# 检查 torchrun
/usr/local/bin/torchrun --version
```

### 环境变量设置
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/NsightSystems-cli-2023.4.1/target-linux-x64:/usr/local/cuda/lib64:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### 故障排除

#### 库文件缺失问题
如果遇到库文件缺失问题，请检查：
1. **cuDNN**: `$CONDA_PREFIX/lib/libcudnn.so.8.9.7` → 自动创建符号链接 `libcudnn.so.8`
2. **cuPTI**: `/usr/local/cuda-12.3/NsightSystems-cli-2023.4.1/target-linux-x64/libcupti.so.12`
3. **NCCL**: `$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2`

#### CUDA 库问题解决方案

**cuDNN 问题**: `ImportError: libcudnn.so.8: cannot open shared object file: No such file or directory`

**cuPTI 问题**: `ImportError: libcupti.so.12: cannot open shared object file: No such file or directory`

**NCCL 问题**: `ImportError: libnccl.so.2: cannot open shared object file: No such file or directory`

**nvcc 问题**: `FileNotFoundError: [Errno 2] No such file or directory: '/data/apps/cuda/12.4/bin/nvcc'`

**解决方案**: 
1. **自动解决**: 使用 `run_maskllm_native.sh` 脚本，会自动创建 cuDNN、cuPTI 和 NCCL 符号链接，并绑定 CUDA 工具链
2. **手动解决**: 在容器内运行 `bash setup_cuda_libs.sh`

**cuDNN 符号链接创建**:
- `libcudnn.so.8` → `libcudnn.so.8.9.7`
- `libcudnn_cnn_infer.so.8` → `libcudnn_cnn_infer.so.8.9.7`
- `libcudnn_cnn_train.so.8` → `libcudnn_cnn_train.so.8.9.7`
- 等等...

**cuPTI 符号链接创建**:
- `libcupti.so.12` → `libcupti.so.12.4` (来自 nsight-systems 或 nsight-compute)

**NCCL 库路径**:
- 直接绑定: `${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nccl/lib` → `/opt/nccl_libs`

**CUDA 工具链绑定**:
- 绑定: `/data/apps/cuda/12.4/bin` → `/usr/local/cuda/bin`
- 绑定: `/data/apps/cuda/12.4/include` → `/usr/local/cuda/include`
- 环境变量: `CUDA_HOME=/usr/local/cuda`, `CUDA_ROOT=/usr/local/cuda`


### 推荐工作流程
1. **启动容器**：`bash run_apptainer_extended_libs.sh`
2. **验证环境**：`bash verify_container_env.sh`
3. **运行任务**：`bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4_fixed.sh SparseGPT`

**注意**: 
- `run_maskllm_native.sh` 会自动设置 cuDNN、cuPTI 和 NCCL 符号链接，并绑定 CUDA 工具链
- 使用 `run_llama2_7b_prune_tp4_fixed.sh` 解决检查点问题

### 当前状态总结

#### 已解决的问题 ✅
1. **环境配置**：成功配置 NVIDIA PyTorch NGC 容器环境
2. **CUDA 库问题**：
   - cuDNN: `libcudnn.so.8` → `libcudnn.so.8.9.7` ✅
   - cuPTI: `libcupti.so.12` → `libcupti.so.12.4` ✅
   - NCCL: 直接绑定 `/opt/nccl_libs` ✅
3. **CUDA 工具链**：nvcc 编译器可用 ✅
4. **检查点问题**：使用现有 `llama2_7b_hf` 检查点 ✅
5. **MaskLLM 启动**：成功启动并开始构建模型 ✅

#### 当前进展
- ✅ 容器环境完全配置
- ✅ 所有 CUDA 库和工具可用
- ✅ MaskLLM 成功启动
- ✅ 模型构建开始


### 解决方案说明
- ✅ **使用容器内原生环境**：Python、PyTorch、transformer_engine
- ✅ **混合库路径**：容器内 + 外部必要库（cuDNN等）
- ✅ **使用容器内torchrun**：直接使用 `/usr/local/bin/torchrun`
- ✅ **避免Python解释器冲突**：清空conda环境变量
- ✅ **自动解决cuDNN问题**：自动创建符号链接 `libcudnn.so.8` → `libcudnn.so.8.9.7`
- ✅ **自动解决cuPTI问题**：自动创建符号链接 `libcupti.so.12` → `libcupti.so.12.4`
- ✅ **自动解决NCCL问题**：直接绑定 NCCL 库路径 `/opt/nccl_libs`
- ✅ **自动解决nvcc问题**：绑定 CUDA 工具链 `/usr/local/cuda/bin`
- ✅ **解决库文件缺失问题**：确保cuDNN、cuPTI、NCCL等库和工具可用

## 开展稀疏训练

```bash
bash run_maskllm_native.sh scripts/learnable_sparsity/llama2_7b_mask_only_tp4_c4.sh 0
```

## 评测训练完成后的checkpoint

容器内先加载环境，然后测评 PPL：
checkpoint: `output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000`

```bash
bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2.sh output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt/iter_0002000 7b 4 sparse
```

## 性能提升-增量训练

提升模型的在 wikitext2 的 perplexity 测试成绩.
- 在`output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000`的基础上，继续稀疏训练。
- 数据集为 wikitext-103-v1 (wikitext-103-v1 has the same preprocessing and tokenization format as the wikitext2 test set).

### 预处理数据集
下载数据集, 保存到 `assets/data/wikitext-103/`.
pretokenize数据集(基于 llama2-7b tokenizer),保存到 `assets/data/wikitext-103/pretokenized/`.
脚本保存到`scripts/data/pretokenize_wikitext-103_llama2-7b.sh`.
- [] `pretokenize_wikitext-103_llama2-7b.sh`需要将处理任务改到在 CPU 上执行，不需要调用 GPU。

Seq_length: 4096
total_iters: 200

### 开展增量训练
Continue further mask learning, maintain the sparsity of the model(N=2, M=4).

1. The gumbel-temperature-range start from 2, and end with 0.05.
2. The gumbel-scale-range start from 1e2, and end with 5e2.
3. The weight-reg start from 1e-5, and end with 1e-5.

```bash
bash run_maskllm_native.sh scripts/incremental/llama2_7b_mask_only_wikitext103_tp4.sh 0
```
0 表示不resume(start from llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000)，1 表示resume。