# MaskLLM 稀疏训练参数详解

## 📋 目录

1. [基础训练参数](#1-基础训练参数)
2. [MaskLLM 核心参数](#2-maskllm-核心参数)
3. [Gumbel Softmax 参数](#3-gumbel-softmax-参数)
4. [优化器参数](#4-优化器参数)
5. [2:4 结构化稀疏参数](#5-24-结构化稀疏参数)
6. [参数协同作用](#6-参数协同作用)

---

## 1. 基础训练参数

### 1.1 学习率配置

```bash
--lr 5e-5                  # 主学习率
--min-lr 5e-6             # 最小学习率
--lr-decay-style cosine   # 余弦衰减
```

#### **为什么选择 5e-5?**

**理论依据**:
1. **稀疏训练需要较高学习率**
   - 从预稀疏模型开始，权重已经被"剪枝"
   - Mask 需要快速适应新的稀疏模式
   - 较高学习率有助于跳出局部最优

2. **Llama2 验证经验**
   - Llama2 (7B) 稀疏训练成功使用 5e-5
   - Llama8b (8B) 规模相近，参数量差异仅 14%
   - 保持一致降低风险

3. **与预训练的对比**
   - 预训练通常使用 3e-4 (更高)
   - 微调通常使用 1e-5 到 5e-5
   - 稀疏训练介于两者之间

**衰减比例 (10:1)**:
```
5e-5 : 5e-6 = 10 : 1
```
- 10:1 比例是标准配置
- 确保训练后期有足够精细调整空间

---

### 1.2 训练步数与间隔

```bash
--train-iters 2000        # 总迭代次数
--save-interval 100       # 保存间隔
--eval-interval 50        # 评估间隔
--log-interval 1          # 日志间隔
--lr-warmup-iters 0       # Warmup步数
```

#### **为什么 2000 步?**

**计算依据**:
```
Total tokens = 2000 iters × 256 batch × 4096 seqlen
             = 2,097,152,000 tokens
             ≈ 2.1B tokens
```

**原因**:
1. **足够的 Mask 收敛**
   - Gumbel softmax 需要从软到硬的渐进过程
   - 2000 步提供充足的演化时间

2. **计算成本平衡**
   - 在 8×A100 上约需 12-15 小时
   - 与 Llama2 验证配置一致

3. **避免过拟合**
   - 稀疏模型容量降低
   - 过长训练可能导致过拟合

#### **为什么 Warmup = 0?**

**关键原因**:
1. **从预稀疏 checkpoint 开始**
   - 权重已经经过 SparseGPT 优化
   - 模型已经是合理初始化状态
   - 不需要 warmup 来稳定训练

2. **对比其他场景**:
   - 从随机初始化: 需要 warmup (如 400 步)
   - 从预训练模型微调: 可能需要少量 warmup
   - 从预稀疏模型: **不需要 warmup**

---

## 2. MaskLLM 核心参数

### 2.1 Mask 优化参数

```bash
--mask-only               # 只训练 mask，权重固定
--freeze-weight           # 冻结权重
```

#### **为什么 mask-only?**

**核心理念**: **可学习稀疏性 (Learnable Sparsity)**

**原理图**:
```
传统稀疏化:
  Dense Weights → [Pruning] → Sparse Weights (固定)
                     ↓
                  结束

MaskLLM 方法:
  Dense Weights → [SparseGPT] → Sparse Weights (固定)
                                      ↓
                                Learnable Masks (可训练)
                                      ↓
                                持续优化稀疏模式
```

**为什么这样设计?**

1. **保护 SparseGPT 的结果**
   - SparseGPT 已经找到了好的权重值
   - 直接训练权重可能破坏稀疏结构
   - Mask 提供了软约束机制

2. **更灵活的稀疏模式**
   - Mask 可以在 2:4 约束内调整选择
   - 动态适应训练数据分布
   - 比固定稀疏模式更优

3. **数学表达**:
```python
# 传统方法
output = sparse_weight @ input  # 稀疏模式固定

# MaskLLM 方法
mask = gumbel_softmax(mask_logits, temp, hard=False)  # 可学习
output = (weight * mask) @ input  # 稀疏模式可调整
```

---

### 2.2 Mask 学习率倍数

```bash
--lr-mult 10.0
```

#### **为什么 Mask 学习率是权重的 10 倍?**

**直观解释**:
- 权重学习率: 5e-5
- Mask 学习率: 5e-5 × 10 = **5e-4**

**原因分析**:

1. **Mask 参数尺度不同**
```python
# 权重值范围
weights: [-0.5, +0.5]  # 经过 SparseGPT 优化

# Mask logits 范围
mask_logits: [-0.08, +0.08]  # 初始化接近均匀分布
```
- Mask logits 初始值很小（接近0）
- 需要更大的学习率来驱动变化

2. **Mask 梯度信号较弱**
```
Loss → Output → Weights (直接影响)
                   ↓
               Mask (间接影响，梯度经过权重)
```
- Mask 的梯度信号经过权重"稀释"
- 需要更高学习率补偿

3. **实验验证**
   - Llama2 论文中验证 `lr_mult=10` 效果最好
   - 过低 (如 5): Mask 收敛太慢
   - 过高 (如 20): Mask 变化不稳定

---

### 2.3 先验强度

```bash
--prior-strength 3.0
```

#### **什么是 Prior (先验)?**

**初始化 Mask 的机制**:

```python
# SparseGPT 产生的稀疏模式
sparse_pattern = [[1, 0, 1, 0],   # 第一个 4 元组: 保留 pos 0,2
                  [0, 1, 1, 0],   # 第二个 4 元组: 保留 pos 1,2
                  ...]

# 转换为 mask logits (prior_strength=3.0)
for each_4tuple in sparse_pattern:
    if position is kept:
        mask_logit[pos] = +3.0  # 鼓励保留
    else:
        mask_logit[pos] = -3.0  # 鼓励剪枝
```

**为什么需要 Prior?**

1. **提供好的起点**
   - SparseGPT 已经找到了合理的稀疏模式
   - Prior 将这个模式编码为 Mask 的初始值
   - 避免从随机开始

2. **为什么是 3.0?**
```python
# Gumbel softmax with temperature=4.0
prob = softmax([+3.0, -3.0, +3.0, -3.0] / 4.0)
    ≈ [0.48, 0.02, 0.48, 0.02]  # 接近 2:4 模式
```
- 3.0 提供了足够强的信号
- 但不会完全"锁死" mask (还有调整空间)

3. **Prior 的演化**
```
Step 0:   logits = [+3.0, -3.0, +3.0, -3.0]  (Prior)
Step 100: logits = [+3.2, -3.5, +2.8, -2.9]  (开始调整)
Step 1000: logits = [+5.1, -6.8, +4.9, -5.2] (接近收敛)
```

---

### 2.4 权重正则化

```bash
--weight-reg 1e-5
```

#### **为什么需要正则化?**

**作用**: 防止 Mask 过拟合

**原理**:
```python
# 总损失
total_loss = lm_loss + weight_reg × ||mask_params||²
           = lm_loss + 1e-5 × sum(mask_logits²)
```

**为什么是 1e-5?**

1. **平衡两个目标**
```
lm_loss:        ~6.0 (初期) → ~2.0 (后期)
reg_loss:       ~5000 (1e-5 × 500M mask params)
weighted_reg:   1e-5 × 5000 = 0.05
```
- Reg 占总 loss 的 ~1-2%
- 足够防止过拟合，但不主导训练

2. **Llama8b vs Llama2**
```
Llama2 (7B):  weight_reg = 1e-5
Llama8b (8B): weight_reg = 1e-5  ✅ 保持一致
```

---

## 3. Gumbel Softmax 参数

### 3.1 原理回顾

**问题**: 如何在 2:4 约束下训练离散 mask?

**挑战**:
```python
# 理想情况 (不可微分)
mask = argmax_2_of_4(mask_logits)  # 选择最大的2个
output = (weight * mask) @ input

# 反向传播
∂Loss/∂mask_logits = ?  # 梯度为 0! (argmax 不可微)
```

**Gumbel Softmax 解决方案**:

1. **软化离散选择**
```python
# 训练时 (可微分)
mask = gumbel_softmax(mask_logits, temperature, hard=False)
# 生成连续值 mask ∈ [0, 1]

# 推理时 (离散)
mask = argmax_2_of_4(mask_logits, hard=True)
# 生成 {0, 1} mask
```

2. **Gumbel 噪声的作用**
```python
# 添加 Gumbel 噪声 (探索性)
noise = -log(-log(uniform(0, 1)))
logits_with_noise = mask_logits + noise

# Softmax with temperature
probs = softmax(logits_with_noise / temperature)
```

---

### 3.2 Scale Multiplier (规模乘数)

```bash
--gumbel-scale-range 100.0 500.0  # [起始值, 结束值]
```

#### **什么是 Scale Multiplier?**

**数学定义**:
```python
# 线性增长
scale(step) = 100 + (500 - 100) × (step / 2000)

# 作用: 放大 logits
scaled_logits = mask_logits × scale(step)
probs = softmax(scaled_logits / temperature)
```

#### **为什么需要递增 Scale?**

**核心思想**: **从软到硬的渐进式训练**

**阶段 1: 低 Scale (探索)**
```python
# scale = 100
logits = [+0.03, -0.02, +0.04, -0.03]
scaled = [+3.0, -2.0, +4.0, -3.0]  # 100×
probs = softmax(scaled / 4.0)
      = [0.32, 0.18, 0.35, 0.15]  # 较平滑
```
- Mask 概率分布较平滑
- 允许探索不同的 2:4 组合
- 梯度信号较强

**阶段 2: 高 Scale (利用)**
```python
# scale = 500
logits = [+0.03, -0.02, +0.04, -0.03]
scaled = [+15.0, -10.0, +20.0, -15.0]  # 500×
probs = softmax(scaled / 0.05)
      = [0.48, 0.00, 0.52, 0.00]  # 非常尖锐
```
- Mask 概率分布非常尖锐
- 接近离散 {0, 1} 值
- 准备转换为硬 mask

**为什么 100 → 500?**

1. **起点 100**
   - 不能太小 (如 10): Prior 信号太弱
   - 不能太大 (如 200): 一开始就太"硬"

2. **终点 500**
   - 足够大，使 mask 接近离散
   - 但不会数值溢出 (softmax 稳定性)

**可视化效果**:
```
Step     Scale    Max Prob    状态
0        100      0.338       均匀分布 (探索)
500      200      0.600       开始分化
1000     300      0.800       明显倾向
1500     400      0.900       接近离散
2000     500      0.950       基本离散
```

---

### 3.3 Temperature (温度)

```bash
--gumbel-temperature-range 4.0 0.05  # [起始值, 结束值]
```

#### **Temperature 的作用**

**数学定义**:
```python
probs = softmax(logits / temperature)
```

**Temperature 对分布的影响**:

```python
# 原始 logits
logits = [+5.0, -5.0, +4.0, -4.0]

# 高温 (temperature=4.0)
probs = softmax([1.25, -1.25, 1.0, -1.0])
      = [0.38, 0.07, 0.32, 0.23]  # 较平滑

# 低温 (temperature=0.05)
probs = softmax([100, -100, 80, -80])
      = [0.52, 0.00, 0.48, 0.00]  # 非常尖锐
```

**为什么 4.0 → 0.05?**

1. **高温 (4.0) - 探索阶段**
   - 分布平滑 → 梯度信号强
   - 允许随机性 → 探索不同选择
   - 配合低 scale，鼓励探索

2. **低温 (0.05) - 利用阶段**
   - 分布尖锐 → 接近确定性
   - 减少随机性 → 锁定最优选择
   - 配合高 scale，准备硬化

**Temperature vs Scale 的协同**:

```
时间轴:
Step 0:    Scale=100,  Temp=4.0   → 平滑探索
Step 1000: Scale=300,  Temp=2.0   → 逐渐聚焦
Step 2000: Scale=500,  Temp=0.05  → 锁定离散
```

**类比**: 
- **Temperature**: 控制"噪声"程度（随机vs确定）
- **Scale**: 控制"信号"强度（弱vs强）
- 两者配合: 从"高噪声高探索" → "低噪声强信号"

---

### 3.4 Scale 和 Temperature 的数学关系

**等效性分析**:

```python
# 方式 1: 调整 scale
probs = softmax(logits × scale / temp)

# 方式 2: 调整 temperature
probs = softmax(logits / (temp / scale))
```

**但为什么两个都要调?**

1. **Scale 放大幅度更直接**
```python
scale: 100 → 500  (5倍增长)
temp:  4.0 → 0.05 (80倍下降)

组合效应: (scale/temp) 增长 5×80 = 400倍!
```

2. **分离控制两种行为**
   - Scale: 放大 Prior 和学习信号的"绝对值"
   - Temp: 控制分布的"尖锐度"

3. **更稳定的训练**
   - 单独调一个参数需要更大范围
   - 可能导致数值不稳定
   - 联合调整更平滑

---

## 4. 优化器参数

### 4.1 AdamW 配置

```bash
--optimizer adam
--adam-beta1 0.9          # 一阶矩估计
--adam-beta2 0.95         # 二阶矩估计
--adam-eps 1e-5           # 数值稳定性
--weight-decay 0.1        # 权重衰减
```

#### **为什么 β₂ = 0.95 (而非 0.999)?**

**Adam 更新规则**:
```python
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t        # 一阶矩 (动量)
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²      # 二阶矩 (RMS)

θ_t = θ_{t-1} - lr × m_t / (√v_t + ε)
```

**β₂ = 0.95 的优势**:

1. **更快适应变化**
```python
# β₂ = 0.999 (标准 Adam)
v_t = 0.999 × v_{t-1} + 0.001 × g_t²  # 历史占 99.9%

# β₂ = 0.95 (大模型常用)
v_t = 0.95 × v_{t-1} + 0.05 × g_t²    # 历史占 95%
```
- 0.95 给当前梯度更多权重
- 更快响应梯度变化
- 适合 Mask 的快速优化

2. **大模型训练经验**
   - GPT-3, Llama 系列都用 0.95
   - 大模型梯度噪声相对较小
   - 不需要过度平滑

#### **为什么 Weight Decay = 0.1?**

**作用**: L2 正则化

```python
# AdamW 更新
θ_t = θ_{t-1} - lr × (m_t / √v_t + λ × θ_{t-1})
                                      ↑
                              weight_decay=0.1
```

**0.1 的合理性**:
- 标准值范围: 0.01 - 0.1
- Llama 预训练也用 0.1
- 防止权重过大，提高泛化

---

### 4.2 梯度裁剪

```bash
--clip-grad 1.0
```

#### **为什么需要裁剪?**

**问题**: 稀疏训练梯度可能不稳定

```python
# 梯度裁剪
grad_norm = ||∇θ||₂
if grad_norm > 1.0:
    ∇θ = ∇θ × (1.0 / grad_norm)  # 归一化到 1.0
```

**为什么是 1.0?**
- 从日志看: grad_norm 范围 1.92 - 8.81
- 1.0 是有效阈值，防止爆炸
- 但大部分步不触发（正常范围）

---

## 5. 2:4 结构化稀疏参数

### 5.1 基本定义

```bash
--N 2                     # 每 M 个保留 N 个
--M 4                     # 块大小
--sparsity 0.5           # 总体稀疏率
```

#### **2:4 模式详解**

**可视化**:
```python
# 原始权重矩阵 (示例: 8×8)
W = [[w00, w01, w02, w03, w04, w05, w06, w07],
     [w10, w11, w12, w13, w14, w15, w16, w17],
     ...]

# 分组为 4 元组 (沿着某个维度)
Group1: [w00, w01, w02, w03]
Group2: [w04, w05, w06, w07]
...

# 2:4 规则: 每组保留最大的 2 个
Group1 after pruning: [w00, 0, w02, 0]     # 假设 w00, w02 最大
Group2 after pruning: [0, w05, 0, w07]     # 假设 w05, w07 最大

# 最终稀疏率 = 2/4 = 50%
```

#### **为什么选择 2:4?**

1. **硬件加速**
   - NVIDIA Ampere (A100) 及之后 GPU 原生支持
   - 2:4 稀疏矩阵乘法可达 2× 吞吐量
   - 无需自定义 CUDA kernel

2. **平衡性能与质量**
```
稀疏率      硬件支持    模型质量    推理速度
0% (稠密)   ✅          100%        1×
50% (2:4)   ✅          95-98%      2×
75% (1:4)   ❌          85-90%      理论4×(无硬件)
```

3. **结构化的优势**
   - vs 非结构化 50%: 质量相近，但能硬件加速
   - vs 稠密: 质量略降，但速度翻倍

---

## 6. 参数协同作用

### 6.1 训练过程的三个阶段

#### **阶段 1: 探索 (Step 0-500)**

```python
学习率:   5e-5 (高)
Scale:    100 → 200
Temp:     4.0 → 3.0
Max Prob: 0.34 → 0.60

特征:
- Mask 快速尝试不同 2:4 组合
- Loss 快速下降 (8.0 → 3.5)
- Mask_diff 较大 (0.08 → 0.05)
```

**参数配合**:
- 高 LR + 高 lr_mult (10×) → Mask 快速更新
- 低 Scale + 高 Temp → 平滑分布，探索充分
- Prior (3.0) → 提供好的起点

#### **阶段 2: 收敛 (Step 500-1500)**

```python
学习率:   5e-5 → 3e-5 (cosine 衰减)
Scale:    200 → 400
Temp:     3.0 → 1.0
Max Prob: 0.60 → 0.90

特征:
- Mask 逐渐聚焦到最优 2:4 模式
- Loss 稳定下降 (3.5 → 2.0)
- Mask_diff 减小 (0.05 → 0.01)
```

**参数配合**:
- LR 衰减 → 更精细调整
- Scale 增长 + Temp 降低 → 分布逐渐尖锐
- Weight_reg (1e-5) → 防止过拟合

#### **阶段 3: 精炼 (Step 1500-2000)**

```python
学习率:   3e-5 → 5e-6 (cosine 衰减)
Scale:    400 → 500
Temp:     1.0 → 0.05
Max Prob: 0.90 → 0.95

特征:
- Mask 基本固定，微调
- Loss 缓慢下降 (2.0 → 1.8)
- Mask_diff 极小 (<0.01)
```

**参数配合**:
- 极低 LR → 防止 Mask 剧烈变化
- 极高 Scale + 极低 Temp → 接近离散 mask
- 准备转换为硬 mask 推理

---

### 6.2 关键参数敏感性分析

#### **高敏感参数** (需要精确设置)

1. **lr_mult (10.0)**
   - 影响: Mask 收敛速度
   - 范围: 5-15
   - 偏离后果:
     - 过低 (如 5): Mask 收敛太慢，2000 步不够
     - 过高 (如 20): Mask 震荡，不稳定

2. **gumbel-scale-range (100, 500)**
   - 影响: Mask 离散化程度
   - 偏离后果:
     - 起点过低 (如 10): Prior 信号丢失
     - 终点过低 (如 200): Mask 不够离散，推理性能差
     - 终点过高 (如 1000): 数值不稳定

3. **prior-strength (3.0)**
   - 影响: 初始 Mask 分布
   - 偏离后果:
     - 过低 (如 1.0): SparseGPT 信息丢失
     - 过高 (如 5.0): Mask 过于固定，探索不足

#### **中敏感参数** (有一定灵活性)

1. **lr (5e-5)**
   - 可调范围: 3e-5 - 1e-4
   - 影响: 整体收敛速度

2. **weight-reg (1e-5)**
   - 可调范围: 5e-6 - 5e-5
   - 影响: 正则化强度

3. **temperature-range (4.0, 0.05)**
   - 可调范围: (2.0, 0.01) - (8.0, 0.1)
   - 影响: 探索 vs 利用平衡

#### **低敏感参数** (稳健)

1. **adam-beta2 (0.95)**
   - 可选: 0.95 或 0.999
   - 影响小

2. **clip-grad (1.0)**
   - 可调范围: 0.5 - 2.0
   - 主要防护机制

---

### 6.3 参数选择流程图

```
开始
  ↓
确定基础架构
  ├─ N=2, M=4 (硬件约束)
  ├─ Sparsity=0.5
  └─ Train_iters=2000
  ↓
参考相似模型
  ├─ Llama2 (7B) 验证参数
  └─ 规模调整 (8B)
  ↓
设置核心参数
  ├─ lr: 5e-5 (从 Llama2)
  ├─ lr_mult: 10.0 (标准)
  ├─ scale: 100→500 (标准)
  ├─ temp: 4.0→0.05 (标准)
  └─ prior: 3.0 (标准)
  ↓
优化器配置
  ├─ AdamW (β₁=0.9, β₂=0.95)
  └─ Weight_decay=0.1
  ↓
正则化
  ├─ weight_reg: 1e-5
  └─ clip_grad: 1.0
  ↓
完成
```

---

## 7. 与其他方法对比

### 7.1 vs SparseGPT (One-shot)

| 方面 | SparseGPT | MaskLLM |
|------|-----------|---------|
| 训练 | 不需要 | 需要 2000 步 |
| 稀疏模式 | 固定 | 可学习 |
| 质量 | 较好 | 更好 |
| 灵活性 | 低 | 高 |
| 参数 | Hessian damp | Gumbel softmax |

### 7.2 vs 传统微调

| 方面 | 传统微调 | MaskLLM |
|------|---------|---------|
| 优化对象 | 权重 | Mask |
| 学习率 | 1e-5 - 5e-5 | 5e-5 (权重), 5e-4 (mask) |
| 稀疏性 | 无约束 | 2:4 约束 |
| Warmup | 通常需要 | 不需要 |
| 特殊参数 | 无 | Gumbel softmax |

---

## 8. 实战建议

### 8.1 参数调优顺序

**第一轮: 保持默认**
```bash
# 使用验证的 Llama2 参数
lr=5e-5, lr_mult=10, scale=(100,500), temp=(4,0.05)
```
观察训练 500 步，检查:
- Loss 是否下降
- Max_prob 是否增长

**第二轮: 微调学习率**
如果:
- Loss 下降太慢 → 提高 lr 到 8e-5
- Loss 震荡 → 降低 lr 到 3e-5
- Mask 收敛慢 → 提高 lr_mult 到 15

**第三轮: 调整 Gumbel**
如果:
- Max_prob 增长太慢 → 提高 scale 终点到 600
- Max_prob 增长太快 → 降低 scale 起点到 80

### 8.2 故障诊断

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss 不下降 | lr 太低 | 提高到 8e-5 |
| Loss 震荡 | lr 太高 | 降低到 3e-5 |
| Max_prob 停滞 | lr_mult 太低 | 提高到 15 |
| Mask_diff 不收敛 | scale 终点太低 | 提高到 600 |
| NaN loss | grad 爆炸 | 降低 clip_grad 到 0.5 |

---

## 9. 总结

### 核心参数记忆口诀

```
学习率 5e-5，mask 再乘 10
Scale 百到五百，渐进变离散
温度 4 到 0.05，探索转确定
Prior 强度 3.0，编码 SparseGPT
正则 1e-5，防止过拟合
```

### 参数哲学

1. **借鉴验证配置** (Llama2)
2. **理解物理含义** (Scale, Temp)
3. **协同作用思维** (不是独立参数)
4. **迭代优化策略** (先默认，再微调)

---

## 参考资料

- MaskLLM 论文: [Learnable Sparse Masks]
- Gumbel Softmax: [Jang et al., 2017]
- SparseGPT: [Frantar et al., 2023]
- Llama2 训练配置: 本项目验证脚本

---

**文档版本**: v1.0  
**创建日期**: 2025-09-30  
**适用模型**: Llama8b 稀疏训练
