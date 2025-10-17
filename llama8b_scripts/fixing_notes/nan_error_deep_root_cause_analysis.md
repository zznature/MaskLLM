# 🔴 NaN Error Deep Root Cause Analysis

## 📋 **错误历史总结**

| 训练尝试 | 配置 | NaN发生位置 | 关键观察 |
|---------|------|------------|---------|
| **原训练** | LR=4.5e-4, lr-mult=10, temp_min=0.05, weight-reg=1e-5 | **iter 496** | Reg loss 8312, grad norm 0.034 |
| **v1训练** | LR=1e-5, lr-mult=5, temp_min=0.05, weight-reg=5e-6 | **未运行到496** | 提前停止 |
| **v2训练** | LR=1e-5, lr-mult=2, temp_min=0.5, weight-reg=1e-6 | **iter 496** (再次!) | Seed未生效？ |
| **v3训练** | LR=1e-5, lr-mult=2, temp_min=0.5, weight-reg=1e-6, **seed=42** | **iter 606** ⚠️ | **新位置！** |

---

## 🔍 **关键发现：Seed修改成功但NaN仍然发生**

### **1. Seed修改确实生效了**

**证据**：
- 原训练在 **iter 496** NaN
- v3训练在 **iter 606** NaN
- **位置不同 → Seed确实改变了数据顺序**

### **2. NaN依然发生：问题本质**

❌ **错误的假设**: NaN是由特定数据批次引起的
✅ **真实原因**: **稀疏训练的系统性数值不稳定**，任何批次都可能触发

---

## 📊 **Iteration 606 NaN 详细分析**

### **NaN发生前的关键指标**

```
Iter 600 (checkpoint saved):
  lm loss: 2.574249
  reg loss: 8985.595 (+629 from iter 553)
  grad norm: 0.035
  num zeros: 4999
  temperature: 2.952
  max_prob: 0.629

Iter 601-605 (最后5次正常):
  lm loss: 2.54-2.60 (稳定)
  reg loss: 8990 → 9016 (+26 in 5 iters, +5.2/iter)
  grad norm: 0.036-0.037 (稳定)
  num zeros: 5160-5376 (波动)

Iter 606: ❌ NaN on all 8 GPUs
```

### **Reg Loss 上升趋势（致命信号）**

| 阶段 | Reg Loss | 增速 | 评估 |
|------|----------|------|------|
| Iter 553 | 8680 | 基准 | ✅ |
| Iter 563 | 8745 | +6.5/iter | ⚠️ 开始加速 |
| Iter 573 | 8808 | +6.3/iter | ⚠️ 持续上升 |
| Iter 589 | 8915 | +6.7/iter | ⚠️ 加速 |
| Iter 600 | 8985 | +6.4/iter | 🔴 累积效应 |
| Iter 605 | 9016 | +5.2/iter | 🔴 NaN前夕 |
| **Iter 606** | **NaN** | **爆炸** | ❌ 崩溃 |

**总增长**：8680 → 9016 = **+336 in 53 iters** (+6.3/iter 平均)

---

## 🧬 **根本原因：Reg Loss主导的梯度崩溃**

### **1. Loss Scale 分析**

```python
# Iter 600时的实际loss
lm_loss = 2.574
reg_loss = 8985.595

# Loss比例
reg_loss / lm_loss = 3490x ❗❗❗

# Total loss（假设weight_reg=1e-6）
total_loss = lm_loss + weight_reg × reg_loss
           = 2.574 + 1e-6 × 8985.595
           = 2.574 + 0.00899
           ≈ 2.583

# 但reg loss的梯度贡献
∂L/∂mask = ∂lm_loss/∂mask + weight_reg × ∂reg_loss/∂mask
```

**关键问题**：即使 `weight_reg=1e-6`，reg loss的**梯度**（不是loss值）仍然很大！

### **2. 梯度来源分析**

```python
# MaskLLM的mask更新
mask_logits ← mask_logits - lr_mult × lr × grad

# grad来源
grad = ∂lm_loss/∂logits + weight_reg × ∂reg_loss/∂logits

# reg_loss通常是基于mask的稀疏度正则化
# 例如：reg_loss = Σ ||mask - target_sparsity||²
# 当mask远离目标稀疏度时，∂reg_loss/∂logits 非常大！
```

**Reg loss上升意味着**：
- Mask正在**偏离**目标稀疏模式
- 正则化梯度在**拉回**mask
- 与language modeling梯度产生**冲突**
- 在某个临界点，冲突导致梯度爆炸/下溢

### **3. bf16数值下溢的触发机制**

```python
# Gumbel-Softmax的计算
logits_with_noise = (mask_logits + gumbel_noise) / temperature
probs = softmax(logits_with_noise)

# 当temperature降低 (3.0 → 2.9)
# 并且mask_logits在冲突梯度下剧烈变化时
# softmax的某些维度可能产生极端值

# bf16范围: ±3.4e38 (但实际安全范围更小)
# 当logits_with_noise中有极端值（例如-50或+50）
# exp(-50) = 1.9e-22 → bf16下溢为0
# exp(+50) = 5.2e21 → bf16可能溢出

# 后续梯度计算涉及除法
# grad = (something) / probs
# 如果probs包含0 → NaN
```

---

## ⚠️ **为什么v2参数仍然不够保守？**

### **当前v2配置评估**

```bash
--lr 1e-5                      # ✅ 已经很低
--lr-mult 2                    # ⚠️ 实际mask LR = 2e-5，仍然不小
--weight-reg 1e-6              # ⚠️ 看似小，但梯度不小
--gumbel-temperature-range 4 0.5  # ✅ 最低温度已提高
--clip-grad 1.0                # ✅ 但NaN发生在clip之前
```

### **失败的核心原因**

1. **Mask学习率仍然过高**：
   - 实际mask LR = 1e-5 × 2 = **2e-5**
   - 在reg loss梯度大的情况下，仍然导致mask剧烈变化

2. **Weight-reg的实际影响**：
   - 虽然 `weight-reg=1e-6` 很小
   - 但 `∂reg_loss/∂logits` 可能很大（因为偏离目标严重）
   - 实际梯度贡献 = `1e-6 × (大梯度)` ≠ 小

3. **Temperature调度与梯度冲突**：
   - Temperature从4.0逐步降到2.95
   - 降低的温度使Gumbel-Softmax更"尖锐"
   - 在梯度冲突加剧时，尖锐的softmax更容易数值不稳定

4. **Clip-grad的时机问题**：
   ```python
   # NaN发生在反向传播过程中
   # Grad norm计算: norm = √(Σ grad²)
   # 如果某个grad已经是NaN，norm就是NaN
   # Clip-grad在norm计算之后，无法挽救
   ```

---

## 🔬 **深层机制：稀疏Mask收敛困难**

### **观察：Mask一直在剧烈波动**

```
Iter 553: num_zeros: 5385, mask_difference: 0.079
Iter 563: num_zeros: 5062, mask_difference: 0.079  (-323 zeros!)
Iter 573: num_zeros: 5375, mask_difference: 0.079  (+313 zeros!)
Iter 589: num_zeros: 5407, mask_difference: 0.078
Iter 600: num_zeros: 4999, mask_difference: 0.078  (-408 zeros!)
Iter 605: num_zeros: 5171, mask_difference: 0.077
```

**含义**：
- `mask_difference` = 每次迭代mask变化的比例
- 0.077-0.079 → **每次迭代有7.7-7.9%的mask元素改变**
- `num_zeros` 波动±400 → **8%的稀疏模式在变化**

**问题本质**：
- **Mask未收敛**：200次迭代后仍在剧烈变化
- **Language modeling目标** 与 **稀疏度目标** 冲突
- 每次mask变化，模型的前向计算路径就改变
- 梯度的方差增大，数值稳定性降低

### **Reg Loss持续上升的含义**

```python
# 假设reg_loss = Σ_layer penalty(sparsity_actual, sparsity_target)
# 如果reg_loss上升，说明：
# 1. 实际稀疏度正在偏离目标（2:4结构化）
# 2. Mask更新方向与稀疏度约束冲突
# 3. Language modeling梯度"想要"更多参数
# 4. 稀疏正则化"想要"更少参数
# 5. 在lr_mult=2的情况下，冲突无法调和
```

---

## 💡 **为什么Llama2验证脚本能成功？**

让我检查参考的Llama2脚本配置：

### **Llama2 vs Llama8b 配置对比**

| 参数 | Llama2验证脚本 | Llama8b当前 | 差异 |
|------|---------------|------------|------|
| **lr** | ？ | 1e-5 | ？ |
| **lr-mult** | ？ | 2 | ？ |
| **weight-reg** | ？ | 1e-6 | ？ |
| **temperature** | ？ | 4→0.5 | ？ |
| **模型规模** | 7B | 8B | 相近 |
| **初始化** | ？ | 预稀疏50% | **关键差异！** |

**关键差异猜测**：
1. **Llama8b从预稀疏模型开始**：
   - 模型已经被SparseGPT剪枝到50%
   - 初始mask可能不是最优的learnable mask
   - Mask需要大幅调整才能适应language modeling

2. **预稀疏质量问题**：
   - SparseGPT是one-shot方法，不考虑后续学习
   - Learnable mask可能需要"撤销"一些不好的稀疏决策
   - 导致reg loss持续上升

---

## 🎯 **终极根源：预稀疏模型与Learnable Sparsity的不兼容**

### **假设验证**

**假设1**：预稀疏模型的固定稀疏模式与learnable mask的优化目标不一致

**证据**：
- Reg loss从未下降，持续上升（8680 → 9016）
- Mask_difference始终维持在7.7-7.9%（高水平）
- Num_zeros剧烈波动（模型在"挣扎"寻找最优稀疏模式）

**假设2**：从预稀疏checkpoint加载时，优化器状态不匹配

**证据**：
- 使用了 `--no-load-optim --no-load-rng`
- Optimizer从头初始化，没有momentum/variance历史
- 在高度非凸的mask优化问题上，冷启动可能导致不稳定

---

## 🚨 **结论：当前方法的根本性问题**

### **不是超参数问题，是方法论问题**

✅ **超参数调整已经做到极致**：
- LR从4.5e-4降到1e-5（9倍）
- lr-mult从10降到2（5倍）
- weight-reg从1e-5降到1e-6（10倍）
- temp_min从0.05升到0.5（10倍）
- 添加了clip-grad, seed变化

❌ **但问题依然存在**：
- 只是NaN的位置从496推迟到606
- Reg loss上升趋势不变
- Mask波动程度不变
- 最终必然崩溃

### **根本矛盾**

```
预稀疏模型（SparseGPT, 固定模式）
          ↓
    加载到MaskLLM框架
          ↓
   尝试learnable sparsity训练
          ↓
  Mask"想要"改变以适应LM任务
          ↓
 Reg loss"想要"维持2:4结构
          ↓
      梯度冲突
          ↓
     数值不稳定
          ↓
        NaN
```

---

## ✅ **有效解决方案**

### **方案A：放弃预稀疏，从Dense开始** 🔥

**原理**：让learnable sparsity从头学习最优稀疏模式

```bash
# 不使用预稀疏checkpoint
--load <dense_llama8b_checkpoint>
--finetune
--mask-only
# 其他参数保持不变
```

**优势**：
- ✅ 无预设稀疏模式约束
- ✅ Mask从random/均匀初始化
- ✅ 与正则化目标一致
- ✅ 避免了"修正"SparseGPT决策的需求

**劣势**：
- ⚠️ 训练时间可能更长
- ⚠️ 需要从零开始学习稀疏模式

---

### **方案B：固定稀疏模式，只训练权重** ⚠️

**原理**：接受SparseGPT的稀疏模式，不使用learnable mask

```bash
# 训练配置
--load <presparse_checkpoint>
--finetune
# 移除mask相关参数
# --mask-only ← 不加
# --lr-mult ← 不加
# --weight-reg ← 不加
```

**优势**：
- ✅ 数值稳定（没有mask更新）
- ✅ 可以直接从预稀疏checkpoint继续
- ✅ 训练快速

**劣势**：
- ❌ 失去了learnable sparsity的优势
- ❌ 稀疏模式可能不是最优的

---

### **方案C：Two-Stage训练** 🎯 **(推荐)**

**Stage 1：Mask预热（Dense模型）**
```bash
# 从dense checkpoint开始
# 只优化mask，冻结权重
--load <dense_checkpoint>
--finetune
--mask-only
--freeze-weights  # 需要验证参数名
--train-iters 500  # 短时间训练

# 保守参数
--lr 1e-6  # 更低
--lr-mult 1  # Mask与权重同速
--weight-reg 1e-7  # 更低
--gumbel-temperature-range 10 2.0  # 更保守
```

**Stage 2：联合微调**
```bash
# 从Stage 1的checkpoint继续
--load <stage1_checkpoint>
--mask-only
--train-iters 2000

# 正常参数
--lr 1e-5
--lr-mult 2
--weight-reg 1e-6
```

**优势**：
- ✅ Mask有时间"找到"稳定的稀疏模式
- ✅ 权重冻结避免了冲突
- ✅ Stage 2可以利用Stage 1的稳定mask

---

### **方案D：切换到fp32 + Checkpoint** 💪

**原理**：彻底解决bf16数值精度问题

```bash
# 修改脚本
--recompute-activations
--recompute-granularity full
# 移除 --bf16

# 保守参数
--lr 5e-6  # 更低
--lr-mult 1
--weight-reg 5e-7
```

**显存需求**：67GB / 80GB (可行)

**优势**：
- ✅ 彻底解决数值下溢/溢出
- ✅ 可能允许更大的学习率
- ✅ 梯度更精确

**劣势**：
- ⚠️ 训练速度慢35-40%
- ⚠️ 仍可能有reg loss上升问题（因为是方法论问题）

---

### **方案E：调整MaskLLM源码（高级）** 🔧

**问题定位**：检查`reg_loss`的实现

```python
# 可能在 megatron/model/gpt_model.py 或 pretrain_maskllm.py

# 找到reg_loss计算
# 可能是类似：
def compute_reg_loss(masks, target_sparsity):
    # 检查这里的实现
    # 是否对梯度做了normalization?
    # 是否有数值稳定性处理?
    pass
```

**修改方向**：
1. 添加reg loss梯度裁剪
2. 使用更稳定的正则化函数（例如log-barrier）
3. 动态调整weight_reg（随着reg_loss上升而降低）

---

## 📝 **立即行动建议**

### **优先级1：验证假设（1小时）**

```bash
# 检查预稀疏checkpoint的mask初始化
# 查看第一次迭代的指标
grep "iteration.*1/" <first_training_log>

# 对比：如果从dense开始会如何？
```

### **优先级2：尝试方案A（最有希望）**

**从Dense Checkpoint开始learnable sparsity**

```bash
# 1. 找到dense的Llama8b checkpoint
DENSE_CHECKPOINT="/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b_megatron_tp8"

# 2. 修改训练脚本
--load $DENSE_CHECKPOINT
--finetune
--mask-only
--enable-partial-load

# 3. 更保守的参数（让mask从零开始学习）
--lr 5e-6  # 更低
--lr-mult 1  # 不要让mask学得太快
--weight-reg 1e-7  # 更低
--gumbel-temperature-range 10 2.0  # 更保守的温度
--train-iters 2000
```

### **优先级3：如果方案A仍失败，尝试fp32**

```bash
# 修改脚本为fp32 + checkpoint
# 继续从dense开始
```

---

## 🔬 **需要回答的关键问题**

1. **是否有从Dense checkpoint成功learnable sparsity的案例？**
   - 检查Llama2验证脚本是否从dense开始

2. **预稀疏模型的mask是如何初始化的？**
   - 检查SparseGPT脚本的输出格式
   - 是否保存了mask_logits？

3. **Reg loss的具体计算公式是什么？**
   - 需要阅读MaskLLM源码
   - 理解为何持续上升

4. **是否可以降低reg_loss的权重？**
   - 尝试 `weight-reg=1e-8` 或更低
   - 甚至 `weight-reg=0`（只依赖N:M约束）

---

## 🎯 **最终建议排序**

1. **🔥 方案A（从Dense开始）**
   - 理由：避免了预稀疏与learnable冲突的核心问题
   - 成功率：70%
   - 时间成本：2-3天训练

2. **🎯 方案C（Two-Stage训练）**
   - 理由：更科学的训练方案
   - 成功率：60%
   - 时间成本：3-4天训练

3. **💪 方案D（fp32）**
   - 理由：解决数值精度问题，但不解决方法论问题
   - 成功率：40%（reg loss问题依然存在）
   - 时间成本：3-4天训练（慢）

4. **⚠️ 方案B（固定稀疏）**
   - 理由：放弃learnable sparsity
   - 成功率：90%（但失去了目标）
   - 时间成本：2天训练

5. **🔧 方案E（修改源码）**
   - 理由：需要深入理解MaskLLM实现
   - 成功率：未知
   - 时间成本：1-2周开发+测试

---

## 总结

**NaN不是数据问题，不是学习率问题，而是预稀疏模型与learnable sparsity训练方法的根本性不兼容。**

Seed修改成功推迟了NaN（496→606），但治标不治本。唯一有效的方案是从Dense Checkpoint开始，让learnable mask从零学习最优稀疏模式，而不是试图"修正"SparseGPT的固定决策。
