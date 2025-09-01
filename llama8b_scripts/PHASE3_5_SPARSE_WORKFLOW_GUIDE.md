# Llama8b稀疏化工作流程指南

## 概述 🎯

本指南详细说明了如何使用Llama8b进行完整的稀疏化训练工作流程，从模型转换到最终的稀疏模型部署。

## 工作流程图 📊

```
Phase 3.1: 模型转换
    ↓
Phase 3.4: 数据预处理 
    ↓
Phase 3.5: 稀疏化准备
    ↓
Phase 3.6: 稀疏化训练
    ↓
Phase 3.7: 模型评估
```

## 详细步骤 🚀

### Step 1: 环境准备 🛠️ ✅

```bash
# 1. 进入Apptainer容器 ✅
bash run_apptainer_extended_libs.sh

# 2. 设置环境变量 ✅
source container_environment_setup.sh

# 3. 安装缺失的Python包 ✅
bash llama8b_scripts/install_missing_packages.sh
```

#### 1.1 依赖包安装状态 ✅

**已安装包** (在.ext_pkgs/):
- ✅ `transformers==4.40`
- ✅ `accelerate` 
- ✅ `datasets`
- ✅ `SentencePiece`
- ✅ `wandb`
- ✅ `tqdm`
- ✅ `ninja`
- ✅ `tensorboardx==2.6`
- ✅ `pulp`
- ✅ `timm`
- ✅ `einops`
- ✅ `nltk`
- ✅ `pydantic==1.10.8`

**导入方法**: 已通过`run_maskllm_native.sh`自动配置PYTHONPATH ✅

### Step 2: 模型转换 (Phase 3.1) 🔄

**目标**: 将HuggingFace格式的Llama8b转换为Megatron格式

```bash
# 执行模型转换
bash scripts/tools/convert_llama8b_hf_to_megatron_simple.sh 8

# 验证转换结果
ls -la output/checkpoints/llama8b_megatron_tp8/
```

**预期输出**: `output/checkpoints/llama8b_megatron_tp8/` 目录包含8个分片的模型文件

### Step 3: 数据预处理 (Phase 3.4) 📝

**目标**: 将C4数据集转换为Megatron格式并使用Llama8b tokenizer

```bash
# 执行数据预处理 (处理前20个文件，约12-15分钟)
bash run_maskllm_native.sh llama8b_scripts/prepare_c4_megatron_llama8b.sh

# 验证数据结果
ls -la assets/data/c4_llama8b_pretokenized/
```

**预期输出**: 
- `c4_llama8b_00000_text_document.bin/idx` 等文件
- 总共40个文件 (20个.bin + 20个.idx)

### Step 4: 稀疏化准备 (Phase 3.5) ✂️ ✅

**目标**: 使用SparseGPT对Llama8b进行一次性剪枝，生成50%稀疏的起始点

#### 4.1 执行稀疏化 ✅

```bash
# 使用默认SparseGPT方法 ✅ 已完成
bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh SparseGPT

# 或指定其他方法
bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh Magnitude
bash run_maskllm_native.sh llama8b_scripts/run_llama8b_prune_tp8.sh Wanda
```

#### 4.2 稀疏化配置详情 ✅

- **稀疏率**: 50% (2:4结构化稀疏) ✅
- **Hessian样本**: 128个 ✅
- **并行配置**: 8 GPU (tensor parallel) ✅
- **实际时间**: 完成 ✅
- **质量验证**: 困惑度 8.0133 ✅

**✅ 实际输出**: `output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0/`

#### 4.3 稀疏化结果验证 ✅

```
🎉 Llama8b稀疏化完成!
📁 稀疏模型保存在: /data/home/zdhs0054/zzhou/MaskLLM/output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0
📊 稀疏化信息:
  - 方法: SparseGPT ✅
  - 稀疏率: 50% (2:4结构化) ✅
  - 模型名: llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0 ✅
  - 验证困惑度: 8.0133 ✅
```

### Step 5: 稀疏化训练 (Phase 3.6) 🏋️‍♂️

**目标**: 基于预稀疏模型进行MaskLLM框架的稀疏化训练

#### 5.1 初始训练

```bash
# 从预稀疏checkpoint开始训练
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 0
```

#### 5.2 恢复训练

```bash
# 从训练checkpoint恢复
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 1
```

#### 5.3 训练配置详情

- **训练迭代**: 10,000
- **批大小**: 16 (global), 2 (micro)
- **学习率**: 1e-5 (模型参数), 0.1 (mask参数)
- **保存间隔**: 1,000迭代
- **评估间隔**: 500迭代
- **预计时间**: 6-12小时

**预期输出**: `output/checkpoints/llama8b_maskllm_training_tp8/`

### Step 6: 模型评估 📊

**目标**: 评估稀疏模型的性能和稀疏度

```bash
# 评估稀疏模型 (待实现)
bash run_maskllm_native.sh llama8b_scripts/evaluate_sparse_model.sh
```

## 故障排除 🔧

### 常见问题与解决方案

#### 1. 环境问题

**问题**: UCC符号冲突
```
ImportError: /opt/hpcx/ucc/lib/libucc.so.1: undefined symbol: ucc_ee_ack_event
```

**解决**: 已通过`run_maskllm_native.sh`自动处理，包含UCC禁用和智能PYTHONPATH配置

#### 2. Tokenizer问题

**问题**: `invalid choice: 'Llama8bTokenizer'`

**解决**: 已修复`tools/preprocess_data.py`和`megatron/arguments.py`，添加了Llama8b tokenizer支持

#### 3. 内存问题

**问题**: GPU内存不足

**解决方案**:
```bash
# 调整批大小
--micro-batch-size 1
--global-batch-size 8

# 启用梯度检查点
--checkpoint-activations
```

#### 4. 性能优化

**建议配置**:
- **CPU核心**: 使用72个workers进行数据预处理
- **GPU配置**: 8 GPU tensor parallel
- **内存**: 建议32GB+ GPU内存

## 性能基准 📈

### 数据预处理性能
- **文件数量**: 20个C4文件
- **处理时间**: 12-15分钟 (72 workers)
- **输出大小**: ~10GB预tokenized数据

### 稀疏化性能 ✅
- **稀疏率**: 50% ✅ (已达成)
- **结构**: 2:4结构化稀疏 ✅
- **质量验证**: 困惑度8.0133 ✅
- **性能保持**: 验证通过 ✅

### 训练性能
- **吞吐量**: ~2000 tokens/second/GPU (预期)
- **收敛**: 8000-10000迭代 (预期)
- **最终perplexity**: 待测试

## 检查点管理 💾

### 重要检查点路径

```
项目根目录/
├── assets/checkpoints/Llama8b/              # 原始tokenizer
├── output/checkpoints/llama8b_megatron_tp8/ # 转换后模型
├── output/oneshot_pruning/checkpoint/       # 预稀疏模型
└── output/checkpoints/llama8b_maskllm_training_tp8/ # 训练结果
```

### 备份建议

```bash
# 备份重要检查点
tar -czf llama8b_checkpoints_backup.tar.gz output/checkpoints/
tar -czf llama8b_sparse_backup.tar.gz output/oneshot_pruning/
```

## 下一步计划 🎯

1. **完善评估脚本**: 实现comprehensive模型评估
2. **性能优化**: 调优训练超参数
3. **部署准备**: 转换为推理格式
4. **文档完善**: 添加更多使用样例

## 技术支持 📞

如遇到问题，请检查：
1. 日志文件: `output/logs/`
2. 环境配置: `container_environment_setup.sh`
3. 进度文档: `llama8b_scripts/PHASE3_PROGRESS_SUMMARY.md`

---

**更新时间**: 2024年12月
**版本**: Phase 3.5
**状态**: 🎉 **Phase 3.5稀疏化准备完成！预稀疏模型已生成**

## 最新进展总结 🎉

### ✅ **Phase 3.5 里程碑达成**:
1. **环境准备**: 所有Python依赖包已安装到.ext_pkgs/ ✅
2. **稀疏化执行**: SparseGPT方法成功完成 ✅
3. **质量验证**: 困惑度8.0133，质量保持良好 ✅
4. **模型输出**: 预稀疏模型已保存到指定路径 ✅

### 🚀 **下一步操作**:
开始Phase 3.6稀疏化训练:
```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8.sh 0
```

### 📊 **技术成就**:
- ✅ 成功将119,696词汇表Llama8b模型稀疏化
- ✅ 2:4结构化稀疏模式完美实现
- ✅ SparseGPT方法验证其在大词汇表模型上的有效性
- ✅ 完整的自动化工作流程建立
