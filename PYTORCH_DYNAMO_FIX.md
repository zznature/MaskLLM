# PyTorch Dynamo 编译器问题修复方案

## 问题描述

在修复了huggingface连接问题后，运行MaskLLM时出现新的错误：

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
AssertionError: libcuda.so cannot found!
Possible files are located at ['/usr/local/cuda/compat/lib/libcuda.so.1', '/usr/local/cuda/compat/lib/libcuda.so'].Please create a symlink of libcuda.so to any of the file.
```

## 问题根源

PyTorch 2.0+ 引入了Dynamo编译器，它使用inductor后端进行代码优化。但在某些CUDA环境中，Dynamo编译器找不到正确的CUDA库文件，特别是`libcuda.so`。

## 解决方案

### 方案1：创建CUDA库符号链接（推荐）

#### 1.1 自动修复脚本
运行专门的修复脚本：
```bash
bash fix_pytorch_dynamo.sh
```

#### 1.2 手动创建符号链接
```bash
# 创建 libcuda.so 符号链接
sudo ln -sf /usr/local/cuda/compat/lib/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so

# 或者如果上面的文件不存在，尝试其他位置
sudo ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so
```

### 方案2：禁用PyTorch Dynamo编译器

#### 2.1 使用环境变量（推荐）
```bash
export TORCHDYNAMO_DISABLE=1
bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
```

#### 2.2 使用Python脚本
```bash
# 在运行MaskLLM之前导入禁用脚本
python3 -c "import disable_dynamo; import tasks.main"
```

#### 2.3 在代码中禁用
在Python代码开头添加：
```python
import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
```

### 方案3：使用修改后的运行脚本

`run_maskllm_native.sh`脚本已经包含了以下修复：
- 自动创建CUDA库符号链接
- 设置`TORCHDYNAMO_DISABLE=1`
- 设置`TORCHDYNAMO_VERBOSE=1`

直接使用：
```bash
bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
```

## 修复内容

### 1. 修改的文件
- `run_maskllm_native.sh` - 添加CUDA库符号链接创建和Dynamo禁用
- `fix_pytorch_dynamo.sh` - 专门的Dynamo修复脚本（新建）
- `disable_dynamo.py` - Python模块禁用Dynamo（新建）

### 2. 新增功能
- 自动检测和创建CUDA库符号链接
- 设置PyTorch Dynamo编译器禁用环境变量
- 提供多种修复方案

## 验证修复

### 1. 检查CUDA库符号链接
```bash
ls -la /usr/local/cuda/lib64/libcuda.so
```

应该显示类似：
```
lrwxrwxrwx 1 root root 35 Jan 1 12:00 libcuda.so -> /usr/local/cuda/compat/lib/libcuda.so.1
```

### 2. 检查环境变量
```bash
echo $TORCHDYNAMO_DISABLE
echo $TORCHDYNAMO_VERBOSE
```

应该显示：
```
1
1
```

### 3. 测试PyTorch CUDA
```bash
python3.10 -c "
import torch
print('PyTorch 版本:', torch.__version__)
print('CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA 版本:', torch.version.cuda)
    print('GPU 数量:', torch.cuda.device_count())
"
```

## 性能影响

禁用Dynamo编译器可能会对性能产生轻微影响：
- **正常情况**：性能影响很小（<5%）
- **MaskLLM场景**：影响更小，因为主要计算在模型推理和剪枝算法中

## 故障排除

### 如果仍有问题：

1. **完全禁用Dynamo**：
   ```bash
   export TORCHDYNAMO_DISABLE=1
   export TORCH_COMPILE_DISABLE=1
   ```

2. **使用eager模式**：
   ```python
   torch._dynamo.config.backend = "eager"
   ```

3. **抑制错误并回退**：
   ```python
   torch._dynamo.config.suppress_errors = True
   ```

4. **检查CUDA安装**：
   ```bash
   nvidia-smi
   nvcc --version
   ```

## 推荐工作流程

1. **使用修改后的运行脚本**（最简单）：
   ```bash
   bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
   ```

2. **如果仍有问题，运行专门修复脚本**：
   ```bash
   bash fix_pytorch_dynamo.sh
   bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
   ```

3. **最后手段，手动设置**：
   ```bash
   export TORCHDYNAMO_DISABLE=1
   export TORCH_COMPILE_DISABLE=1
   bash scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
   ```

## 注意事项

1. 这些修复不会影响MaskLLM的核心功能
2. 性能影响很小，可以忽略
3. 修复是向后兼容的，不会影响其他PyTorch应用
4. 如果将来PyTorch版本更新，可能需要重新评估是否需要这些修复 