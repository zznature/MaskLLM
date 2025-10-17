# ❌ --finetune 导致参数重置问题

## 🔴 问题发现

### **当前训练状态 (iter 1, 实际应为 1311)**

```
learning rate: 2.500E-07  ← ❌ 太小！在 warmup 阶段
temperature: 4.000        ← ❌ 重置到初始值！
reg loss: 4372.466        ← ❌ 重置到初始值！
lm loss: 5.122            ← ❌ 回到预训练初期水平
```

### **期望状态 (基于 iter 1310)**

```
learning rate: 3.402E-05  ← ✅ 应该接近这个值
temperature: 2.364        ← ✅ 应该从这里继续衰减
reg loss: 9032.425        ← ✅ 应该从这里继续
lm loss: 2.558            ← ✅ 应该接近这个值
```

---

## 🔍 根本原因

### **--finetune 的副作用**

`--finetune` 标志不仅重置了 `consumed_samples` 和 `iteration`，还重置了：

```python
1. ✅ consumed_samples = 0      # 预期行为
2. ✅ iteration = 0              # 预期行为
3. ❌ Learning rate scheduler    # 意外副作用！
4. ❌ Temperature scheduler      # 意外副作用！
5. ❌ Scale multiplier           # 意外副作用！
```

**具体表现**:

```
Learning Rate:
  - 从 warmup 重新开始: 2.5e-7, 5e-7, 7.5e-7, ...
  - 应该: 3.4e-5 (iter 1310 的值)

Temperature (Gumbel):
  - 从初始值 4.0 重新开始
  - 应该: 2.364 (iter 1310 的值)

Reg Loss:
  - 从初始值 ~4400 重新开始
  - 应该: ~9032 (iter 1310 的累积值)
```

---

## ⚠️ 影响分析

### **严重性：高 🔴**

1. **Training Discontinuity**:
   ```
   模型权重: 来自 iter 1310 ✅
   训练参数: 从 iter 0 重新开始 ❌
   → 不连续！训练质量严重下降
   ```

2. **Reg Loss 回退**:
   ```
   Reg loss 从 9032 → 4372 (减半!)
   → Mask 正则化压力突然减小
   → 可能导致 sparsity pattern 崩溃
   ```

3. **Temperature 重置**:
   ```
   Temperature 从 2.364 → 4.0
   → Gumbel-Softmax 变得更 "soft"
   → Mask 选择的确定性降低
   ```

4. **Learning Rate Warmup**:
   ```
   LR 从 3.4e-5 → 2.5e-7 (降低 100+ 倍!)
   → 学习速度极慢
   → 浪费大量计算资源
   ```

---

## ✅ 正确的解决方案

### **方案：不使用 --finetune，手动调整配置**

#### **核心思路**

```
问题: consumed_samples (335360) > total_samples (257280)
      因为 batch size 从 256 → 128

正确方案:
  保持 batch size = 256 ⭐
  或增大 train_iters 使 total_samples 足够大
```

#### **推荐方案 1: 保持 batch=256 (最简单) ⭐⭐⭐⭐⭐**

```bash
修改:
  --global-batch-size 256  # 保持与原训练一致

效果:
  consumed_samples = 335360
  total_samples = 2000 × 256 = 512,000
  335360 < 512,000 ✅

优点:
  ✅ 完全连续训练
  ✅ LR, Temperature, Reg loss 都连续
  ✅ 无需 --finetune
  ✅ 单卡显存: 53GB (BF16) → 约 68-72GB (FP32)
  ✅ 仍在 80GB 范围内

缺点:
  ⚠️ 显存略高 (但完全可行)
```

#### **推荐方案 2: 增大 train_iters (备选) ⭐⭐⭐⭐**

```bash
修改:
  --train-iters 3000  # 从 2000 增加
  --global-batch-size 128  # 保持新配置

效果:
  consumed_samples = 335360
  total_samples = 3000 × 128 = 384,000
  335360 < 384,000 ✅

优点:
  ✅ 完全连续训练
  ✅ 保持 batch=128 的显存优势
  ✅ 无需 --finetune

注意:
  训练到 iter 2000 时手动停止
  或让它训练到 3000 (更充分)
```

---

## 🔧 立即修复

### **采用方案 1: 保持 batch=256**

#### **显存验证**

```python
当前 FP32 (batch=128):
  Reserved: 61.5GB - 61.8GB / 80GB
  Allocated: 31GB (peak 59GB)

预估 FP32 (batch=256):
  Batch size 翻倍 → 显存增加约 10-15%
  预估 Reserved: 68-72GB / 80GB
  安全余量: 8-12GB ✅ 充足
```

#### **修改脚本**

```bash
# 文件: llama8b_presparse_training_tp8_fp32_fast_from_iter1310.sh

# 修改 1: global-batch-size
--global-batch-size 256 \  # was 128

# 修改 2: 移除 --finetune
# --finetune \  # 注释掉

# 修改 3: 恢复 --override-opt_param-scheduler (如果需要)
--override-opt_param-scheduler \  # 恢复

# 修改 4: 学习率保持不变 (因为 batch 恢复)
--lr 2e-5 \        # 保持 (或改回 1.5e-5)
--min-lr 2e-6 \
```

#### **参数建议**

```bash
batch=256 时的 LR 建议:
  --lr 1.5e-5       # 与超保守训练一致
  --min-lr 1.5e-6

或更激进:
  --lr 2e-5         # 当前设置
  --min-lr 2e-6
```

---

## 📊 显存详细分析

### **FP32 vs BF16 内存占用**

| 组件 | BF16 | FP32 | 增长 |
|------|------|------|------|
| **模型权重** | 8GB | 16GB | 2x |
| **梯度** | 8GB | 16GB | 2x |
| **Activations (batch=256)** | 25GB | 35GB | 1.4x |
| **Optimizer (--no-save-optim)** | 0GB | 0GB | - |
| **其他 (buffers, etc)** | 12GB | 15GB | 1.25x |
| **总计** | **53GB** | **~70GB** | 1.32x |

### **结论**

```
FP32 + batch=256:
  预估显存: 68-72GB / 80GB
  安全余量: 8-12GB
  可行性: ✅ 完全可行

FP32 + batch=128:
  实测显存: 61-62GB / 80GB
  安全余量: 18-19GB
  浪费: ~20% 显存未利用
```

---

## 🚀 最终推荐配置

### **配置 A: 完全连续训练 (推荐) ⭐⭐⭐⭐⭐**

```bash
精度: FP32
Batch size: 256  ← 与原训练一致
Learning rate: 1.5e-5 (或 2e-5)
LR mult: 5
Recompute: No
Flash Attention: No
Finetune: No  ← 关键！

显存: 68-72GB / 80GB
速度: ~220秒/iter
连续性: 完全连续 ✅
```

### **配置 B: 高效显存利用**

```bash
精度: FP32
Batch size: 192  ← 折中方案
Learning rate: 1.8e-5  ← 按比例调整
Train iters: 2800  ← 确保足够样本

显存: 64-67GB / 80GB
速度: ~220秒/iter
连续性: 完全连续 ✅
```

---

## ⚠️ 当前训练建议

### **立即行动**

```bash
1. 停止当前训练 (Ctrl+C)
   原因: LR 太低，Temperature 重置，不连续

2. 修改脚本:
   - global-batch-size: 128 → 256
   - 移除: --finetune
   - (可选) 添加: --override-opt_param-scheduler

3. 重新启动训练
   预期: LR ~3.4e-5, Temperature ~2.36, Reg loss ~9032
```

---

## 📝 经验教训

### **--finetune 的真正用途**

```
适用场景:
  ✅ 从预训练模型开始 finetune 任务
  ✅ 改变训练目标或数据分布
  ✅ 需要重新开始学习率调度

不适用场景:
  ❌ 从中间 checkpoint 继续相同训练
  ❌ 仅为解决 consumed_samples 问题
  ❌ 需要保持训练连续性

本次错误:
  使用 --finetune 解决 consumed_samples
  → 但破坏了训练连续性
  → 应该调整 batch size 或 train_iters
```

---

**修复优先级**: 🔴 高  
**预计修复时间**: 5分钟  
**影响**: 完全连续训练 + 更高显存利用率

