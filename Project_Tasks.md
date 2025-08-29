# 项目任务

## HPC服务器节点配置

登录节点: hd02-gpfs-quorum-02
GPU计算节点: hd02-gpu1-0056,hd02-gpu1-0024,hd02-gpu1-0017,hd02-gpu1-0029
登录方法: `srun --jobid=4879 --overlap --pty bash -i`

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

## 运行 MaskLLM 任务

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

**问题症状**:
- `ImportError: libcudnn.so.8: cannot open shared object file: No such file or directory`
- `ImportError: libcupti.so.12: cannot open shared object file: No such file or directory`
- `ImportError: /opt/hpcx/ucc/lib/libucc.so.1: undefined symbol: ucs_mpool_params_reset`
- `FileNotFoundError: [Errno 2] No such file or directory: '/data/apps/cuda/12.4/bin/nvcc'`

**🎉 最终解决方案**: **无符号链接 LD_LIBRARY_PATH 直接访问方案**

**核心问题**: NGC容器使用非标准库结构 (`/opt/conda_libs/`) 且权限限制无法创建符号链接

**✅ 彻底解决方案**:
```bash
# 在容器内运行 - 完全解决所有库依赖问题
bash llama8b_scripts/fix_container_libs_no_symlink.sh
```

**技术原理**:
1. **🔍 自动发现**: 检测NGC容器实际库文件位置 (`/opt/conda_libs/`)
2. **🧹 HPC-X清理**: 自动清除HPC-X库冲突路径
3. **📍 优先级路径**: 重建`LD_LIBRARY_PATH`，按优先级排序：
   - `/opt/conda_libs/` (cuDNN主目录)
   - `/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib/` (cuPTI)
   - `/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib/` (NCCL)
   - `/usr/local/cuda/lib64/` (CUDA标准库)
   - `/usr/local/lib/python3.10/dist-packages/torch/lib/` (PyTorch库)
4. **🛡️ 环境保护**: 强制禁用HPC-X组件冲突
5. **✅ 逐步验证**: 验证每个关键库文件可访问性

**已验证的库文件位置**:
- **cuDNN**: `/opt/conda_libs/libcudnn.so.8.9.7` ✅
- **cuPTI**: `/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12` ✅  
- **NCCL**: `/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2` ✅

**✅ 验证结果**:
- ✅ PyTorch 2.2.0a0+81ea7a4 导入成功
- ✅ CUDA 可用: True，检测到4个H100 GPU
- ✅ 分布式模块正常
- ✅ 所有关键库 (cuDNN, cuPTI, NCCL) 可访问

**🚀 后续验证命令**:
```bash
# transformer_engine测试
python3.10 -c 'import transformer_engine as te; print("transformer_engine:", te.__version__)'

# megatron.tokenizer测试  
python3.10 -c 'from megatron.tokenizer import build_tokenizer; print("megatron.tokenizer: 可用")'

# C4数据转换测试
bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh 0 0
```

**重要提示**: 
- ⚠️ 必须在Apptainer容器内运行
- ✅ 不需要root权限或符号链接
- ✅ 自动处理HPC-X库冲突
- ✅ 与现有`run_maskllm_native.sh`完全兼容


### 推荐工作流程
1. **启动容器**：`bash run_apptainer_extended_libs.sh`
2. **验证环境**：`bash verify_container_env.sh`
3. **运行任务**：`bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4_fixed.sh SparseGPT`

**注意**: 
- `run_maskllm_native.sh` 会自动设置 cuDNN、cuPTI 和 NCCL 符号链接，并绑定 CUDA 工具链
- 使用 `run_llama2_7b_prune_tp4_fixed.sh` 解决检查点问题

### 当前状态总结

#### 🎉 完全解决的问题 ✅
1. **环境配置**：成功配置 NVIDIA PyTorch NGC 容器环境
2. **🔥 CUDA 库问题**：**彻底解决** - 使用无符号链接LD_LIBRARY_PATH方案
   - ✅ cuDNN: `/opt/conda_libs/libcudnn.so.8.9.7` 直接访问
   - ✅ cuPTI: `/opt/conda_libs/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12` 直接访问
   - ✅ NCCL: `/opt/conda_libs/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2` 直接访问
3. **🛡️ HPC-X冲突**：完全解决 - 自动清理冲突路径，强制禁用冲突组件
4. **🚀 PyTorch环境**：完全正常
   - ✅ PyTorch 2.2.0a0+81ea7a4 导入成功
   - ✅ CUDA 可用，检测到4个H100 GPU
   - ✅ 分布式模块正常
5. **🔧 工具链**：nvcc 编译器可用
6. **📦 检查点支持**：Llama8b模型转换和加载支持完整

#### 🎯 当前完全就绪的功能
- ✅ **完整PyTorch环境** - 所有库依赖完全解决
- ✅ **Llama8b模型支持** - tokenizer集成和checkpoint转换
- ✅ **MaskLLM框架** - 稀疏训练和推理
- ✅ **数据处理** - C4数据集转换和预处理
- ✅ **分布式训练** - 8GPU TP=8配置支持

#### 🚀 可立即执行的任务
1. **Task 1.3**: Llama8b预稀疏模型生成 (`run_llama8b_prune_tp8.sh`)
2. **Task 1.4**: C4数据集Megatron格式转换 (`prepare_c4_megatron_llama8b.sh`)
3. **Phase 3.4**: 稀疏化训练兼容性测试 (`llama8b_mask_only_tp8_c4_fixed.sh`)
4. **组件验证**: transformer_engine, megatron.tokenizer测试


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
bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2.sh output/checkpoints/llama2-7b-tp4-mask-only-c4-singlenode/train_iters_2000/ckpt 7b 4 sparse
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

### 增量训练后PPL评测
```bash
bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2.sh output/checkpoints/llama2-7b-tp4-mask-only-wikitext103-0724/train_iters_400/ckpt 7b 4 sparse
```

## 输出训练后checkpoint为HF格式

根据export_hf.md文档，已创建完整的checkpoint到HuggingFace格式转换流程。

### 转换脚本说明

#### 主要脚本 
1. **`quick_convert_to_hf.sh`** - 一键转换脚本 (推荐使用)
2. **`convert_checkpoint_to_hf.sh`** - 完整转换流程主脚本 (支持Float16)
3. **`convert_hf_to_float16.py`** - Float16格式转换工具
4. **`generate_precomputed_mask.sh`** - 生成预计算mask文件（可选）
5. **`setup_float16_conversion.sh`** - 权限设置和使用说明脚本

#### 辅助脚本
- **`scripts/tools/convert_llama2_7b_tp4_to_tp1.sh`** - TP=4到TP=1转换脚本

### 转换流程

#### 前置条件
确保以下文件/目录存在：
- 源checkpoint: `output/checkpoints/llama2-7b-tp4-mask-only-wikitext103-c4format/train_iters_800/ckpt/iter_0000800`
- HF参考模型: `assets/checkpoints/llama2_7b_hf` （如不存在需先下载）

#### 执行转换

##### 方法1: 快速设置并转换 (推荐)
```bash
# 自动设置所有脚本权限并显示使用说明
bash setup_float16_conversion.sh

# 一键转换 (推荐)
bash run_maskllm_native.sh quick_convert_to_hf.sh
```

##### 方法2: 手动设置并转换
```bash
# 手动设置脚本可执行权限
chmod +x convert_checkpoint_to_hf.sh generate_precomputed_mask.sh scripts/tools/convert_llama2_7b_tp4_to_tp1.sh convert_hf_to_float16.py

# 在容器内执行完整转换流程
bash run_maskllm_native.sh convert_checkpoint_to_hf.sh
```

转换步骤包括：
1. **合并稀疏mask**: 使用 `tool_apply_sparsity.py` 将mask合并到模型参数
2. **TP转换**: 从TP=4转换到TP=1
3. **HF导出**: 转换为HuggingFace格式 (FP32)
4. **tokenizer复制**: 复制必要的tokenizer文件
5. **Float16转换**: 转换为Float16格式以减少存储空间
6. **清理临时文件**: 删除中间的FP32文件

#### 输出结果
- HF格式模型 (Float16): `output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format/`

#### Float16格式优势
- ✅ 模型大小减半 (约7GB → 3.5GB)
- ✅ 推理速度提升
- ✅ 显存占用减少
- ✅ 与lm-eval-harness完全兼容

#### 可选：生成预计算mask文件
```bash
# 生成压缩的mask文件，用于与官方HF模型结合评测
bash run_maskllm_native.sh generate_precomputed_mask.sh
```

输出：
- 压缩mask文件: `output/precomputed_masks/llama2_7b_hf_maskllm_wikitext103_c4format_mask_compressed.npz`

### 评测HF格式模型

#### 使用项目内的评测脚本
```bash
# 评测转换后的HF模型
python eval_llama_ppl.py --model output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format/

# 或者使用官方HF模型+mask评测
python eval_llama_ppl.py --model meta-llama/Llama-2-7b-hf --mask output/precomputed_masks/llama2_7b_hf_maskllm_wikitext103_c4format_mask_compressed.npz
```

#### 使用lm-eval-harness评测
```bash
# 评测转换后的HF模型
lm_eval --model hf \
    --model_args pretrained=output/checkpoints/llama2_7b_hf_maskllm_wikitext103_c4format \
    --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge
```

### 使用说明
1. **在容器内运行**: 所有脚本都需要在MaskLLM容器环境内执行
2. **确保依赖**: 转换脚本会自动安装必要的Python包（transformers, wandb等）
3. **存储空间**: 转换过程会创建多个中间文件，确保有足够存储空间
4. **错误处理**: 脚本包含错误检查，遇到问题会提示并停止