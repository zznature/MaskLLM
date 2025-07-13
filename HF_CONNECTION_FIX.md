# HuggingFace 连接问题修复方案

## 问题描述

在运行 MaskLLM 的 Hessian 剪枝任务时，出现连接 huggingface.co 超时的问题：

```
MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /api/datasets/allenai/c4/tree/...
```

## 问题根源

代码中多处尝试从 huggingface.co 下载 C4 数据集，但由于网络连接问题导致超时。

## 解决方案

### 1. 修改数据集加载代码

将以下文件中的 `datasets.load_dataset('allenai/c4', ...)` 调用修改为使用本地已下载的数据集：

#### 修改的文件：
- `tasks/pruning/datasets.py` - `_build_c4_dataset()` 函数
- `tasks/pruning/sparsity/utils/datautils.py` - `get_c4()` 函数  
- `tasks/latency/datasets.py` - `_build_c4_dataset()` 函数
- `tasks/zeroshot_gpt/datasets.py` - `_build_c4_dataset()` 函数
- `eval_llama_ppl.py` - `get_c4()` 函数

#### 修改内容：
```python
# 原来的代码
traindata = datasets.load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, cache_dir="assets/cache", split='train')

# 修改后的代码
import json
c4_file_path = "./assets/data/c4/en/c4-train.00000-of-01024.json"
with open(c4_file_path, 'r', encoding='utf-8') as f:
    traindata = []
    for line in f:
        if line.strip():
            traindata.append(json.loads(line))
```

### 2. 设置 HuggingFace 镜像环境变量

在 `run_maskllm_native.sh` 脚本中添加了以下环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
```

### 3. 创建独立的环境变量设置脚本

创建了 `set_hf_mirror.sh` 脚本，可以单独设置 HuggingFace 镜像环境变量。

## 本地数据集状态

C4 数据集已经下载到 `./assets/data/c4/` 目录，包含：
- `en/c4-train.00000-of-01024.json` (786MB)
- `en/c4-train.00001-of-01024.json` (783MB)
- ... 共20个文件

## 使用方法

### 方法1：使用修改后的运行脚本（推荐）
```bash
bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
```

### 方法2：手动设置环境变量
```bash
source set_hf_mirror.sh
bash scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
```

### 方法3：直接在容器中设置
```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
bash scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
```

## 验证修复

修复后，代码将：
1. 直接从本地 `./assets/data/c4/en/c4-train.00000-of-01024.json` 文件加载数据
2. 不再尝试连接 huggingface.co
3. 避免网络连接超时问题

## 注意事项

1. 确保 `./assets/data/c4/en/c4-train.00000-of-01024.json` 文件存在
2. 如果验证数据集文件不存在，代码会自动使用训练数据的前1100个样本作为验证集
3. 所有修改都保持了原有的数据处理逻辑，只是改变了数据源

## 测试建议

运行以下命令测试修复效果：
```bash
bash run_maskllm_native.sh scripts/oneshot/run_llama2_7b_prune_tp4.sh hessian
```

应该看到类似以下输出，而不是连接错误：
```
Loaded 364868 samples from local C4 dataset
calculating layerwise Hessian
token 1 / 128
token 2 / 128
...
``` 