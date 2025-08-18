# MaskLLM处理自定义架构模型工作方案

## 项目背景
Llama8b是从别的模型转换过来的自定义模型，需要在MaskLLM框架中进行适配以支持稀疏化训练。

GPU node: hd02-gpu1-0017
使用 Git 分支: `H100_Llama8b`

## 0. 模型推理

```bash
bash run_maskllm_native.sh llama8b_scripts/run_llama8b_infer.sh
```

## 1. 模型分析

### 1.1 Llama8b模型特点
- **架构**: LlamaForCausalLM（标准Llama架构）
- **参数配置**: 32层，4096隐藏维度，32个注意力头
- **特殊性**: 
  - 使用自定义Llama8bTokenizer
  - 词汇表大小119696（远大于标准Llama的32000）
  - 自定义的tokenizer实现，包含特殊的分词逻辑

### 1.2 现有MaskLLM架构分析
- 基于Megatron框架，支持模型并行和流水线并行
- 具有完整的稀疏化训练支持（learnable_sparsity模块）
- 现有checkpoint转换系统（tools/checkpoint/）
- 已支持标准Llama2模型的HF到Megatron转换

## 2. 技术挑战

### 2.1 主要挑战
1. **自定义Tokenizer适配**: Llama8bTokenizer需要在Megatron框架中正确注册和使用
2. **非标准词汇表大小**: 119696的vocab_size需要特殊处理
3. **权重文件转换**: 需要从HF格式转换到Megatron格式
4. **稀疏化适配**: 确保稀疏化训练正常工作

### 2.2 依赖关系
- Transformers库版本兼容性
- 自定义tokenizer的正确加载

## 3. 实施方案

### 3.1 阶段一：Tokenizer适配

#### 任务1.1: 创建Llama8b Tokenizer注册器
- **文件**: `megatron/tokenizer/llama8b_tokenizer.py`
- **功能**: 
  - 包装现有的Llama8bTokenizer
  - 实现Megatron tokenizer接口
  - 处理特殊token和编码逻辑

#### 任务1.2: 更新tokenizer工厂
- **文件**: `megatron/tokenizer/tokenizer.py`
- **功能**: 添加"Llama8bTokenizer"类型支持

#### 任务1.3: 测试tokenizer功能
- 创建独立测试脚本验证tokenizer加载和编码

### 3.2 阶段二：Checkpoint转换器开发

#### 任务2.1: 创建Llama8b专用loader
- **文件**: `tools/checkpoint/loader_llama8b_hf.py`
- **基于**: `loader_llama2_hf.py`进行修改
- **主要变更**:
  - 支持119696词汇表大小
  - 正确加载Llama8bTokenizer
  - 处理自定义配置参数

#### 任务2.2: 更新checkpoint转换工具
- **文件**: `tools/checkpoint/util.py`
- **功能**: 添加llama8b loader选项

#### 任务2.3: 创建转换脚本
- **文件**: `scripts/tools/convert_llama8b_hf_to_megatron.sh`
- **功能**: 自动化转换流程

### 3.3 阶段三：模型加载和测试

#### 任务3.1: 测试模型加载
- 使用转换后的checkpoint加载模型
- 验证模型结构和参数正确性
- 检查内存使用情况

#### 任务3.2: 稀疏化适配测试
- 使用`pretrain_maskllm.py`测试稀疏化功能
- 验证mask应用和梯度计算
- 检查sparse linear层正确创建

#### 任务3.3: 推理测试
- 使用`eval_llama_ppl.py`测试模型推理
- 计算困惑度验证模型正确性
- 测试不同batch size和序列长度

### 3.4 阶段四：优化和文档

#### 任务4.1: 性能优化
- 内存使用优化
- 加载速度优化
- 并行化参数调优

#### 任务4.2: 完善文档和示例
- 创建使用说明文档
- 提供完整的训练示例
- 记录已知问题和解决方案

## 4. 具体实施步骤

### 4.1 准备工作
1. **环境检查**
   ```bash
   # 检查依赖版本
   python -c "import transformers; print(transformers.__version__)"
   python -c "import torch; print(torch.__version__)"
   ```

2. **权重文件验证**
   
   ```bash
   shasum -a 256 "/Llama8b/pytorch_model.bin" | cat
   ```
   output:15ab6f1b6e6ae05e289e66f2532cd178db74b00c788fab4f7817238df20e73c1

   ```bash
   # 确认权重文件完整性
   ls -la assets/checkpoints/Llama8b/
   python -c "
   from transformers import AutoTokenizer, AutoModelForCausalLM
   tokenizer = AutoTokenizer.from_pretrained('assets/checkpoints/Llama8b/')
   model = AutoModelForCausalLM.from_pretrained('assets/checkpoints/Llama8b/')
   print('Model loaded successfully')
   "
   ```

### 4.2 开发顺序
1. **Tokenizer适配** → **Checkpoint转换** → **模型测试** → **稀疏化测试**
2. 每个阶段完成后进行验证测试
3. 遇到问题及时调整方案

### 4.3 验证标准
- **Tokenizer**: 编码解码一致性，特殊token处理正确
- **转换**: 权重数值一致性，结构对应正确
- **推理**: 困惑度合理，输出文本质量正常
- **稀疏化**: mask正确应用，训练loss正常下降

## 5. 风险评估和缓解

### 5.1 主要风险
1. **自定义tokenizer兼容性问题**
   - 缓解: 提前进行小规模测试
   - 备选: 考虑tokenizer转换方案

2. **大词汇表内存问题**
   - 缓解: 使用gradient checkpointing
   - 备选: 考虑词汇表裁剪

3. **权重转换精度问题**
   - 缓解: 逐层验证权重一致性
   - 备选: 使用更高精度进行转换

### 5.2 应急方案
- 如果直接转换困难，考虑先转换为标准Llama格式
- 如果性能问题严重，考虑模型量化
- 如果稀疏化不兼容，先进行dense训练验证

## 6. 成功标准

### 6.1 功能性标准
- [x] Llama8b模型能在MaskLLM框架中正确加载
- [x] Tokenizer功能完全正常
- [x] 稀疏化训练可以正常进行
- [x] 推理结果与原始HF模型一致

### 6.2 性能标准
- [x] 内存使用合理（不超过同规模标准Llama 20%）
- [x] 训练速度可接受（不低于标准Llama 80%）
- [x] 模型精度保持（困惑度差异<5%）

## 7. 后续扩展

### 7.1 其他模型支持
基于Llama8b的经验，可以扩展支持：
- 其他自定义tokenizer的模型
- 非标准架构的转换模型
- 多语言或领域特定模型

### 7.2 工具链完善
- 自动化转换工具
- 模型验证工具套件
- 性能基准测试工具