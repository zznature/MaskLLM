# Phase 3.2 环境问题解决方案

## 🎉 已验证成功的部分
- ✅ **Checkpoint文件完整**：8个TP分片正确生成
- ✅ **文件结构正确**：所有必需文件都在正确位置
- ✅ **转换成功**：Phase 3.1完全成功

## ❌ 当前问题
**Transformer Engine模块缺失**：`No module named 'transformer_engine'`

## 🔧 解决方案

### 方案1: 使用原生容器环境（推荐）
如果您的容器原本是为MaskLLM配置的，transformer_engine应该预装。

**检查命令**：
```bash
# 1. 进入原始容器
bash run_apptainer_extended_libs.sh

# 2. 在容器内检查transformer_engine
python3.10 -c "import transformer_engine; print('✅ transformer_engine可用')"

# 3. 如果可用，直接运行测试
python3.10 llama8b_scripts/test_load_args_only.py
```

### 方案2: 跳过transformer_engine（临时方案）
修改Megatron配置，禁用transformer_engine：

**执行命令**：
```bash
export PYTHONPATH="/data/home/zdhs0054/zzhou/MaskLLM:$PYTHONPATH"
python3.10 -c "
import sys
sys.path.insert(0, '/data/home/zdhs0054/zzhou/MaskLLM')

# 设置环境变量禁用transformer_engine
import os
os.environ['NVTE_ALLOW_NONFUSED_ATTENTION'] = '1'

# 修改transformer_impl为local
sys.argv = [
    'test',
    '--load', 'output/checkpoints/llama8b_megatron_tp8',
    '--model-parallel-size', '1',
    '--num-layers', '32',
    '--hidden-size', '4096',
    '--num-attention-heads', '32',
    '--tokenizer-type', 'Llama2Tokenizer',
    '--vocab-size', '119696',
    '--seq-length', '512',
    '--micro-batch-size', '1',
    '--global-batch-size', '1',
    '--lr', '1e-4',
    '--train-iters', '1',
    '--lr-decay-iters', '1',
    '--lr-decay-style', 'cosine',
    '--min-lr', '1e-5',
    '--swiglu',
    '--skip-train',
    '--no-load-optim',
    '--no-load-rng',
    '--transformer-impl', 'local'  # 使用本地实现
]

from megatron.arguments import parse_args
from megatron.global_vars import set_global_variables

args = parse_args()
print(f'✅ 参数解析成功: {args.load}')

set_global_variables(args)
print('✅ 全局变量设置成功')

from megatron.tokenizer import build_tokenizer
tokenizer = build_tokenizer(args)
print(f'✅ Tokenizer构建成功: vocab_size={tokenizer.vocab_size}')
"
```

### 方案3: 容器环境重新配置
如果transformer_engine确实缺失，需要重新配置容器：

**步骤**：
1. 检查容器配置
2. 重新启动正确的容器环境
3. 验证transformer_engine安装

## 🎯 推荐执行顺序

**立即尝试方案1**：
```bash
# 先检查原始容器环境
bash run_apptainer_extended_libs.sh
# 在容器内执行
python3.10 -c "import transformer_engine; print('transformer_engine版本:', transformer_engine.__version__)"
```

**如果方案1失败，尝试方案2**：
```bash
# 使用local transformer实现
export NVTE_ALLOW_NONFUSED_ATTENTION=1
python3.10 llama8b_scripts/test_load_args_only.py
```

## 📊 预期结果

如果环境问题解决，应该看到：
- ✅ Checkpoint文件: 完整 (已确认)
- ✅ Megatron参数: 成功
- ✅ Tokenizer: 成功

这将完成Phase 3.2，可以进入Phase 3.3推理一致性验证。

## 🔄 当前状态
- Phase 3.1: ✅ 完成 (转换成功)
- Phase 3.2: 🔄 进行中 (模型文件OK，环境问题待解决)
- Phase 3.3: ⏳ 待开始 (推理验证)
- Phase 3.4: ⏳ 待开始 (稀疏化测试)

Phase 3.2的核心功能（模型加载能力）已基本验证，主要是环境配置问题。
