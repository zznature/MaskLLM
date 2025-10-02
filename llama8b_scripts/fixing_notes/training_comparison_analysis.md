# Llama8b训练对比分析：原训练 vs 恢复训练

## 数据对比 (Iteration 402-477)

### 学习率对比
| Metric | 原训练 | 恢复训练 | 差异 |
|--------|--------|---------|------|
| 起始LR | 4.566e-4 | 5.000e-5 | **↓89%** |
| 结束LR | 4.397e-4 | 5.000e-5 | **↓88%** |
| LR策略 | Cosine衰减 | 固定(新优化器) | 已切换 |

### Loss对比
| Iteration | 原训练 LM Loss | 恢复训练 LM Loss | 原训练 Reg Loss | 恢复训练 Reg Loss |
|-----------|---------------|-----------------|----------------|------------------|
| 402 | 2.5928 | 2.5928 | 7800.1 | 7800.3 |
| 413 | 2.6416 | 2.6458 | 7875.6 | 7876.4 |
| 425 | 2.5515 | 2.5515 | 7954.3 | 7954.3 |
| 450 | 2.6029 | 2.6029 | 8127.8 | 8127.8 |
| 477 | 2.5664 | 2.5664 | 8311.8 | 8311.8 |

**结论**: LM Loss几乎完全相同，Reg Loss趋势完全一致（持续上升）

### Grad Norm对比（关键！）
| Iteration | 原训练 | 恢复训练 | 差异 |
|-----------|--------|---------|------|
| 402-412 | 0.032-0.034 | 0.032-0.033 | 相似 |
| **413** | **0.099** | **0.106** | ⚠️ 峰值更高 |
| 414-476 | 0.032-0.043 | 0.032-0.037 | 略低 |

**关键发现**: 
- ⚠️ Iteration 413在两次训练中都出现梯度峰值
- ⚠️ 恢复训练的峰值更高(0.106 vs 0.099)
- 这可能是数据或mask更新的临界点

### 稀疏度指标对比
| Metric | 原训练范围 | 恢复训练范围 | 评价 |
|--------|----------|-------------|------|
| Num Zeros | 5167-6570 | 5167-6570 | ✅ 相同 |
| Scale Multiplier | 180.2-195.2 | 180.2-195.2 | ✅ 相同 |
| Temperature | 3.208→3.060 | 3.208→3.060 | ✅ 相同 |
| Max Prob | 0.535→0.578 | 0.535→0.578 | ✅ 相同 |

---

## 核心问题：为什么降低学习率没有改善Reg Loss？

### 问题分析

#### 1. **Reg Loss持续上升的数学解释**

Reg Loss = `weight_reg` × Σ(mask相关惩罚项)

当前配置：
- `weight_reg = 5e-6`
- `lr_mult = 5` → 实际mask LR = 5e-5 × 5 = **2.5e-4**

**关键insight**：
- 模型权重LR降低了9倍（4.5e-4 → 5e-5）
- 但mask LR只降低了4.5倍（1.125e-3 → 2.5e-4）
- Mask更新速度相对于权重仍然**过快**

#### 2. **Reg Loss vs LM Loss的失衡**

```
Iteration 477:
- LM Loss: 2.566 (语言建模损失)
- Reg Loss: 8311.845 (正则化损失)
- Reg/LM 比例: 3239倍

Total Loss ≈ 2.566 + 8311.845 = 8314.411
Reg贡献占比: 99.97%
```

**问题**：
- 梯度几乎完全被reg loss主导
- 模型优化目标从"学好语言模型"变成"满足正则化约束"
- 这违背了稀疏化训练的初衷

#### 3. **Gumbel温度问题**

当前温度调度：
- 起始：4.0
- 当前(iter 477)：3.060
- 目标最低：**0.05**（你保留的配置）

**问题**：
- 温度3.06时，softmax已经比较尖锐
- 如果继续降到0.05，mask会接近one-hot
- One-hot mask的梯度极不稳定，容易产生NaN

---

## 参数调整建议

### 🔴 **Critical (必须修改，否则很可能再次NaN)**

#### 1. 降低Mask学习率倍数
```bash
# 当前
--lr-mult 5  # 实际mask LR = 2.5e-4

# 建议
--lr-mult 2  # 实际mask LR = 1e-4 (降低2.5倍)
```

#### 2. 提高Gumbel最低温度
```bash
# 当前（你保留的）
--gumbel-temperature-range 4 0.05

# 强烈建议
--gumbel-temperature-range 4 0.5  # 最低温度提高10倍
```

#### 3. 进一步降低正则化权重
```bash
# 当前
--weight-reg 5e-6

# 建议
--weight-reg 1e-6  # 降低5倍
```

### 🟡 **Important (建议修改)**

#### 4. 降低总体学习率
```bash
# 当前
--lr 1e-5

# 建议（如果仍出现NaN）
--lr 5e-6  # 再降低2倍
```

---

## 修改后的稳定脚本 v2

### 变更说明

创建 `llama8b_presparse_training_tp8_stable_v2.sh`：

```bash
# v1 → v2 的关键变更
--lr-mult 5 → 2                    # Mask LR降低2.5倍
--weight-reg 5e-6 → 1e-6           # 正则化降低5倍
--gumbel-temperature-range 4 0.05 → 4 0.5  # 最低温度提高10倍
```

### 预期效果

1. **Reg Loss上升速度减缓**：
   - 从iter 402-477上升512 → 预期上升<300

2. **Grad Norm更稳定**：
   - 消除iter 413类似的峰值
   - 保持在0.01-0.05范围

3. **Loss Scale可能恢复**：
   - 数值稳定性提高后，loss scale可能从1.0回升

---

## 当前状态评估

### ✅ **进展良好**
- 学习率降低成功应用
- LM Loss稳定，未发散
- 稀疏模式正常演化
- Iteration 402-477未出现NaN

### ⚠️ **潜在风险**
- Reg Loss持续上升（+6.6%）
- Grad norm在iter 413有峰值(0.106)
- Loss scale固定在1.0（混合精度保护失效）
- 距离原NaN点(iter 496)仅19次迭代

### 🎯 **行动建议**

#### **选项1：继续观察** (风险中等)
- 如果继续用当前配置运行到iter 500
- 需要密切监控grad norm和reg loss
- 一旦看到grad norm > 0.15，立即停止

#### **选项2：立即应用v2参数** (推荐)
- 停止当前训练
- 应用更保守的参数（v2）
- 从iter 450 checkpoint重新开始

#### **选项3：等到iter 500再决定**
- 如果到iter 500无NaN，说明当前参数可行
- 但reg loss上升需要长期关注

---

## 数值稳定性分析

### 为什么Iteration 413是危险点？

#### 观察到的现象
```
Iteration 412:
- grad_norm: 0.032 ✅
- num_zeros: 5655/5590 ✅

Iteration 413:
- grad_norm: 0.099/0.106 ⚠️ (峰值，增加3倍)
- num_zeros: 5167/5281 ⚠️ (突然大幅减少)
- lm_loss: 2.645/2.642 ⚠️ (相对较高)

Iteration 414:
- grad_norm: 0.032/0.032 ✅ (恢复正常)
- num_zeros: 5388/5195 ✅
```

#### 可能的原因

1. **Mask大幅更新**：
   - num_zeros从5655→5167(减少488)
   - 说明mask做了显著调整
   - 大量zero→non-zero或non-zero→zero

2. **困难样本**：
   - lm_loss=2.645相对较高
   - 可能遇到难以建模的数据
   - 触发较大的权重梯度

3. **Gumbel采样随机性**：
   - Temperature=3.186
   - 采样可能产生异常的mask配置

#### 为什么v2参数能缓解？

```
v1: mask_grad = base_grad × 5 × (temperature影响)
v2: mask_grad = base_grad × 2 × (temperature影响更温和)

降低3倍+温度约束 → 峰值grad norm预期 < 0.05
```

---

## 监控指标 Dashboard

### 实时监控命令

```bash
# 监控关键指标
ssh zdhs0054@hd02-gpu1-0056 "bash /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/monitor_training_from_remote.sh search 'grad norm'"

# 检查reg loss趋势
ssh zdhs0054@hd02-gpu1-0056 "bash /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/monitor_training_from_remote.sh search 'reg loss' | tail -20"
```

### 警报阈值

| 指标 | 安全范围 | 警告 | 危险 |
|------|---------|------|------|
| Grad Norm | 0.01-0.05 | 0.05-0.10 | >0.10 |
| Reg Loss增速 | <10/iter | 10-20/iter | >20/iter |
| Loss Scale | >1.0 | =1.0 | <1.0(impossible) |
| Num Zeros波动 | ±200 | ±500 | >±500 |

---

## 结论

### 当前状态
- ✅ 恢复训练技术上成功
- ⚠️ 但参数仍不够保守
- ⚠️ Reg loss问题未解决
- ⚠️ 距离原NaN点很近

### 最终建议
**应用v2参数**，理由：
1. 数据证明当前参数未能抑制reg loss上升
2. Iteration 413的梯度峰值预示风险
3. 距离原NaN点(iter 496)仅19步
4. v2参数成本低（从iter 450重新开始损失27次迭代）
