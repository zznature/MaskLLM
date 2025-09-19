# Llama3-8B vs Llama8b配置对比分析

## 🎯 关键发现

**重要**: Llama3-8B使用**相同的"激进"参数**但训练成功，说明问题**不是参数过激**，而是其他原因！

## 📊 详细配置对比

### 1. 架构参数对比

| 参数 | Llama3-8B | Llama8b | 状态 | 分析 |
|------|-----------|---------|------|------|
| **FFN Hidden Size** | `14336` | `14336` | ✅ 相同 | 架构匹配 |
| **Hidden Size** | `4096` | `4096` | ✅ 相同 | 架构匹配 |
| **Num Layers** | `32` | `32` | ✅ 相同 | 架构匹配 |
| **Attention Heads** | `32` | `32` | ✅ 相同 | 架构匹配 |
| **Vocab Size** | 默认 | `119696` | ❓ 不同 | Llama8b扩展词汇表 |

### 2. 训练超参数对比

| 参数 | Llama3-8B | Llama8b | 状态 | 分析 |
|------|-----------|---------|------|------|
| **学习率** | `5e-5` | `5e-5` | ✅ 相同 | 非问题根源 |
| **最小学习率** | `5e-6` | `5e-6` | ✅ 相同 | 非问题根源 |
| **Global Batch** | `256` | `256` | ✅ 相同 | 非问题根源 |
| **Micro Batch** | `1` | `1` | ✅ 相同 | 非问题根源 |
| **梯度裁剪** | `1.0` | `1.0` | ✅ 相同 | 非问题根源 |
| **权重衰减** | `0.1` | `0.1` | ✅ 相同 | 非问题根源 |
| **训练迭代** | `2000` | `2000` | ✅ 相同 | 非问题根源 |

### 3. 稀疏化参数对比

| 参数 | Llama3-8B | Llama8b | 状态 | 分析 |
|------|-----------|---------|------|------|
| **Gumbel Scale** | `1e2 5e2` | `1e2 5e2` | ✅ 相同 | 非问题根源 |
| **Gumbel Temp** | `4 0.05` | `4 0.05` | ✅ 相同 | 非问题根源 |
| **Prior Strength** | `3.0` | `3.0` | ✅ 相同 | 非问题根源 |
| **LR Mult** | `10` | `10` | ✅ 相同 | 非问题根源 |
| **Weight Reg** | `1e-5` | `1e-5` | ✅ 相同 | 非问题根源 |
| **N:M Sparsity** | `2:4` | `2:4` | ✅ 相同 | 非问题根源 |

## 🔍 关键差异分析

### 差异1: Tokenizer类型 ⚠️ **可能是关键**

```bash
# Llama3-8B (成功)
--tokenizer-type AutoTokenizer
--tokenizer-model ${TOKENIZER_MODEL}  # 标准HF目录

# Llama8b (失败)  
--tokenizer-type Llama8bTokenizer
--tokenizer-model ${TOKENIZER_MODEL}
--vocab-file ${PROJECT_DIR}/assets/checkpoints/Llama8b/vocab.txt
```

**分析**: 
- Llama3使用标准的`AutoTokenizer`
- Llama8b使用自定义的`Llama8bTokenizer`
- 自定义tokenizer可能有数值稳定性问题

### 差异2: 保存频率 ⚠️ **影响调试**

```bash
# Llama3-8B
SAVE_INTERVALS=100  # 100步保存一次

# Llama8b  
SAVE_INTERVALS=500  # 500步保存一次
```

**分析**: 
- Llama3更频繁保存，能在850步前捕获稳定状态
- 我们500步保存，错过了问题发生前的状态

### 差异3: 架构特殊参数 ❓ **待确认影响**

```bash
# Llama3-8B 独有
--group-query-attention \
--num-query-groups 8 \
--rotary-base 500000 \
```

**分析**: Llama3有特殊的GQA配置，但Llama8b不一定需要

## 🎯 问题根因推测

### 最可能的原因排序:

1. **🥇 Tokenizer问题** (80%可能性)
   - 自定义`Llama8bTokenizer`可能有数值稳定性bug
   - 词汇表扩展到119696可能导致embedding层不稳定
   - 建议: 尝试`AutoTokenizer`

2. **🥈 数据预处理差异** (15%可能性)  
   - Llama3使用`c4-blend-llama3.sh`
   - Llama8b使用手动构建的data_blend
   - 数据格式或预处理细节可能不同

3. **🥉 模型初始化差异** (5%可能性)
   - OneShot稀疏化结果在不同模型间可能有差异
   - Llama8b的稀疏权重分布可能更容易不稳定

## 🔬 验证策略

### 策略1: Tokenizer对齐测试 (优先)
```bash
# 测试AutoTokenizer是否解决问题
bash run_maskllm_native.sh llama8b_scripts/llama8b_aligned_with_llama3.sh 0 1
```

### 策略2: 保存频率对齐
- 改为100步保存，及时捕获稳定状态
- 在出现问题前获得可用checkpoint

### 策略3: 分阶段调试
1. 先运行50步，验证初期稳定性
2. 如果稳定，逐步增加到100、200、500步
3. 定位具体失败点

## 📊 预期结果

如果是Tokenizer问题：
- ✅ 使用AutoTokenizer应该能解决数值不稳定
- ✅ reg loss会在合理范围内
- ✅ 梯度范数保持稳定

如果不是Tokenizer问题：
- 🔍 需要进一步分析数据预处理和模型初始化差异
- 🔍 可能需要调整vocab_size或embedding初始化策略

## 🚀 立即行动

1. **优先测试**: AutoTokenizer对齐
2. **备选方案**: 保守化参数配置
3. **深度调试**: 如果仍失败，需要详细分析OneShot checkpoint的权重分布

**核心假设**: Llama3-8B成功说明参数配置本身没问题，问题在于模型特定的实现细节！
