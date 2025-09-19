# MaskLLM 稀疏训练参数设计

## 1. MaskLLM 稀疏训练原理

### 1.1 核心思想
MaskLLM通过**Gumbel Softmax**机制实现可微分的结构化稀疏学习：
- 🎯 **目标**: 学习二进制掩码M，使得稀疏模型 `output = W ⊙ M` 保持性能
- 🔄 **挑战**: 二进制掩码不可微分，无法直接用梯度下降优化
- 💡 **解决**: Gumbel Softmax提供连续松弛，实现从连续到离散的平滑过渡

### 1.2 Gumbel Softmax 数学原理

#### 标准Softmax问题
```
softmax(z_i) = exp(z_i) / Σ exp(z_j)  # 连续但非二进制
```

#### Gumbel Softmax解决方案
```
gumbel_softmax(z_i, τ) = exp((z_i + g_i)/τ) / Σ exp((z_j + g_j)/τ)
```
其中：
- `g_i ~ Gumbel(0,1)` 是Gumbel噪声
- `τ` 是温度参数
- `scale` 是缩放因子

#### 温度退火机制
- **高温度** (τ=4.0): 分布平滑，接近连续采样，梯度稳定
- **低温度** (τ=0.05): 分布尖锐，接近one-hot，趋向二进制
- **退火过程**: τ(t) = τ_start × (τ_end/τ_start)^(t/T)

### 1.3 N:M 结构化稀疏
- **定义**: 每M个连续元素中保留N个最重要的
- **2:4模式**: 每4个权重保留2个，实现50%稀疏率
- **硬件友好**: GPU tensor cores原生支持2:4稀疏加速

## 2. 当前参数配置解析

### 2.1 Llama8b当前配置
```bash
TASK_CMD="--gumbel-scale-range 5e1 2.5e2 \
         --gumbel-temperature-range 4 0.05 \
         --N 2 --M 4 \
         --mask-only \
         --prior-strength 3.0 \
         --lr-mult 10 \
         --weight-reg 2e-6"
```

### 2.2 详细参数说明

#### 🌡️ Gumbel Temperature Range
```bash
--gumbel-temperature-range 4 0.05
```
- **起始温度**: 4.0 (软采样，探索性强)
- **结束温度**: 0.05 (硬采样，决策性强)
- **退火比例**: 80:1 (平滑过渡)
- **训练阶段**:
  - 0-400步: 预热期，温度从4.0开始下降
  - 400-2000步: 主训练期，温度持续退火至0.05

#### ⚡ Gumbel Scale Range
```bash
--gumbel-scale-range 5e1 2.5e2
```
- **起始缩放**: 50 (低噪声，稳定探索)
- **结束缩放**: 250 (高噪声，充分探索)
- **作用机制**:
  - 缩放因子放大logits差异
  - 使掩码对小学习率更敏感
  - 平衡探索(exploration)与利用(exploitation)

#### 🎯 N:M稀疏配置
```bash
--N 2 --M 4
```
- **稀疏模式**: 2:4结构化稀疏
- **稀疏率**: 50% (每4个保留2个)


#### 🎭 训练模式
```bash
--mask-only
```
- **模式**: 掩码专用训练
- **机制**: 冻结预训练权重W，只学习掩码M
- **优势**:
  - 保护预训练知识
  - 训练稳定性高
  - 计算开销小

#### ⚡ 学习率控制
```bash
--lr-mult 10
```
- **倍数**: 10倍学习率倍增
- **计算**: mask_lr = base_lr × 10 = 2e-5 × 10 = 2e-4
- **原理**: 掩码参数需要更快学习以匹配预训练权重

#### 🎯 稀疏先验
```bash
--prior-strength 3.0
```
- **强度**: 3.0 (标准约束)
- **作用**: 引导掩码向目标稀疏率收敛
- **机制**: 在损失函数中添加稀疏先验项

#### ⚖️ 权重正则化
```bash
--weight-reg 2e-6
```
- **系数**: 2e-6 (轻度正则化)
- **类型**: L2正则化
- **目的**: 防止过拟合，控制模型复杂度

## 3. 参数优化策略

### 3.1 稳定性优先配置 (当前)
```bash
# 特点: 保守参数，数值稳定
--gumbel-scale-range 5e1 2.5e2      # 中等噪声
--gumbel-temperature-range 4 0.05   # 标准退火
--lr-mult 10                        # 标准倍数
--weight-reg 2e-6                   # 轻度正则化
```

### 3.2 超稳定配置 (数值问题时)
```bash
# 特点: 极保守，确保训练成功
--gumbel-scale-range 2e1 1e2        # 低噪声
--gumbel-temperature-range 3 0.1    # 缓慢退火
--lr-mult 5                         # 降低倍数
--weight-reg 1e-6                   # 更轻正则化
```

### 3.3 效果优化配置 (稳定后使用)
```bash
# 特点: 追求更好稀疏质量
--gumbel-scale-range 1e2 5e2        # 高噪声
--gumbel-temperature-range 4 0.05   # 标准退火
--lr-mult 10                        # 标准倍数
--weight-reg 1e-5                   # 强正则化
```

## 4. 关键监控指标

### 4.1 核心指标
- **reg_loss**: 正则化损失，应稳定下降
- **grad_norm**: 梯度范数，应保持 < 1.0
- **scale_multiplier**: 缩放因子，不应快速增长
- **temperature**: 当前温度，应平滑下降
- **max_prob**: 最大概率，应逐渐增加

### 4.2 健康状态判断
```bash
✅ 健康训练:
   - reg_loss: 稳定下降趋势
   - grad_norm: 0.1-0.5 范围
   - scale_multiplier: 缓慢增长
   - temperature: 平滑退火

❌ 数值不稳定:
   - reg_loss: 快速增长或NaN
   - grad_norm: > 1.0 或突然跳跃
   - scale_multiplier: 指数增长
   - temperature: 异常波动
```

## 5. 实际训练经验

### 5.1 Llama8b训练历程
1. **初始尝试**: 使用高学习率(5e-5)导致梯度爆炸
2. **参数调整**: 降低学习率至2e-5，减少正则化至2e-6
3. **当前状态**: 使用中等保守的Gumbel参数，训练稳定

### 5.2 关键发现
- **学习率冲突**: checkpoint中保存的学习率调度器会覆盖脚本配置
- **温度退火**: 平滑的温度退火比激进退火更稳定
- **正则化平衡**: 过强正则化会阻碍掩码学习，过弱则可能过拟合

### 5.3 最佳实践
1. **渐进式调优**: 从保守参数开始，逐步优化
2. **实时监控**: 重点关注reg_loss和grad_norm变化
3. **及时调整**: 发现数值不稳定立即降低学习率或Gumbel参数