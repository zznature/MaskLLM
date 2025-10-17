# 数据集样本耗尽错误修复

## 🔴 错误信息

```
AssertionError: no samples left to consume: 335360, 257290
```

## 🔍 根本原因

### **问题分析**

```python
consumed_samples: 335360  # iter 1310 已消耗的样本数
total_samples: 257290     # 当前数据集配置的总样本数
```

**冲突**:
- 从 iter 1310 恢复时，训练器期望从 sample 335360 继续
- 但当前数据集配置只有 257290 个样本（使用 11 个文件，batch=128）
- 335360 > 257290 → 没有样本可以继续消费

### **为什么会发生？**

原因 1: **Batch Size 改变**
```
原始训练 (iter 0-1310):
  Data files: 00009-00019 (11 files)
  Global batch size: 256
  Total samples = 11 files × samples_per_file / 256
  = 257290 × 2 = 514580 samples (for batch=256)

新配置 (iter 1310+):
  Data files: 00009-00019 (11 files, 相同)
  Global batch size: 128 (减半!)
  Total samples = 257290 (for batch=128)
  
Consumed samples at iter 1310:
  1310 × 256 = 335360 samples
  
335360 > 257290 → 错误!
```

原因 2: **数据文件数量**
```
原始训练可能使用了更多文件 (00000-00019, 20个文件)
恢复训练只配置了 11 个文件 (00009-00019)
```

---

## ✅ 解决方案

### **方案 1: 增加数据文件 (推荐 ⭐⭐⭐⭐⭐)**

**修改脚本中的数据配置**:

```bash
# 当前配置 (11 个文件)
for i in {00009..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.1 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

# 修改为使用所有 20 个文件
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done
```

**效果**:
```
文件数: 11 → 20
权重: 0.1 → 0.05 (保持总和为 1.0)
Total samples: 257290 → ~468000 (足够!)
335360 < 468000 ✅
```

---

### **方案 2: 重置 iteration 计数 (不推荐)**

使用 `--finetune` 标志重置 iteration 和 consumed_samples。

**问题**:
- 会从 iter 0 重新开始计数
- 学习率调度会重置
- 不推荐

---

### **方案 3: 调整样本起点 (复杂)**

手动修改 checkpoint 中的 consumed_samples。

**问题**:
- 需要修改 checkpoint 文件
- 容易出错
- 不推荐

---

## 🔧 **立即修复**

### **修改脚本**

修改 `llama8b_presparse_training_tp8_fp32_fast_from_iter1310.sh`:

```bash
# 找到第 24-27 行
# 当前:
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00009..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.1 ${C4_HOME}/c4_llama8b_${i}_text_document"
done

# 修改为:
C4_HOME="$PROJECT_DIR/assets/data/c4_llama8b_pretokenized"
DATA_BLEND=""
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done
```

### **验证数据文件**

```bash
# 检查是否所有文件都存在
ls -lh assets/data/c4_llama8b_pretokenized/c4_llama8b_000{00..19}_text_document.{bin,idx}

# 应该看到 40 个文件 (20 × 2: .bin + .idx)
```

---

## 📊 **数据配置对比**

### **原始 BF16 训练**

```bash
# 可能的配置 (推测)
for i in {00000..00019}; do
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done
Global batch size: 256
Total samples: ~514580
Iter 1310 consumed: 1310 × 256 = 335360 ✅
```

### **当前 FP32 配置 (错误)**

```bash
for i in {00009..00019}; do  # 只有 11 个文件
    DATA_BLEND="${DATA_BLEND} 0.1 ${C4_HOME}/c4_llama8b_${i}_text_document"
done
Global batch size: 128
Total samples: 257290
Iter 1310 consumed: 335360 ❌ (超出!)
```

### **修复后的配置**

```bash
for i in {00000..00019}; do  # 所有 20 个文件
    DATA_BLEND="${DATA_BLEND} 0.05 ${C4_HOME}/c4_llama8b_${i}_text_document"
done
Global batch size: 128
Total samples: ~468000
Iter 1310 consumed: 335360 ✅
剩余: 468000 - 335360 = 132640 samples
可训练: 132640 / 128 = 1036 iters
1310 + 1036 = 2346 > 2000 ✅ (足够到 iter 2000)
```

---

## ⚠️ **重要说明**

### **数据一致性**

修改数据配置后，训练数据会略有不同：
```
原始: 文件 00009-00019 (11个)
新配置: 文件 00000-00019 (20个)

影响:
- 数据分布略有变化
- 但由于使用相同的 seed 和 shuffle，影响很小
- 模型已经训练到 iter 1310，影响微乎其微
```

### **为什么原始训练能跑？**

可能的原因：
1. 原始训练脚本使用了所有 20 个文件
2. 或者使用了更多的数据文件
3. 超保守训练脚本可能复制了不完整的配置

---

## 📝 **修复步骤**

1. **修改脚本**:
   ```bash
   vim llama8b_scripts/llama8b_presparse_training_tp8_fp32_fast_from_iter1310.sh
   
   # 修改第 24-27 行
   # {00009..00019} → {00000..00019}
   # 0.1 → 0.05
   ```

2. **验证数据文件**:
   ```bash
   ls assets/data/c4_llama8b_pretokenized/c4_llama8b_*.{bin,idx} | wc -l
   # 应该输出 40 (20个文件 × 2)
   ```

3. **重新启动训练**:
   ```bash
   bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_fast_from_iter1310.sh
   ```

---

## ✅ **预期结果**

修复后:
```
> finished creating GPT datasets ...
> BlendedDataset length: 468000 (or similar)
loading checkpoint from .../iter_0001310
successfully loaded checkpoint
```

**不会再看到**:
```
AssertionError: no samples left to consume
```

---

**修复时间**: 2分钟  
**影响**: 数据配置  
**风险**: 极低（数据略有变化但影响微小）

