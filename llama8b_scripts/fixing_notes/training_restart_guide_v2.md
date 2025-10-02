# Llama8b训练重启指南 - 稳定版v2

## 📋 **配置变更总结**

### v1 → v2 参数对比

| 参数 | v1 (当前运行中) | v2 (保守配置) | 变化 | 目的 |
|------|----------------|---------------|------|------|
| **学习率** | 5e-5 | 1e-5 | ✅ 保持 | 已验证有效 |
| **Mask学习率倍数** | 5 | **2** | ⬇️ 2.5倍 | 减缓mask更新 |
| **实际Mask LR** | 2.5e-4 | **2e-5** | ⬇️ 12.5倍 | 核心改进 |
| **正则化权重** | 5e-6 | **1e-6** | ⬇️ 5倍 | 抑制reg loss |
| **Gumbel最低温度** | 0.05 | **0.5** | ⬆️ 10倍 | 数值稳定性 |
| **梯度裁剪** | 1.0 | 1.0 | ✅ 保持 | 已有保护 |

---

## 🎯 **v2配置的技术优势**

### 1. **数值稳定性大幅提升**

#### Temperature范围对比：
```python
# v1配置 (0.05最低值)
温度降到1.0时: exp(logit/1.0) ≈ exp(3) = 20
温度降到0.05时: exp(logit/0.05) ≈ exp(60) = 1e26 → ❌ 溢出风险极高

# v2配置 (0.5最低值)
温度降到1.0时: exp(logit/1.0) ≈ exp(3) = 20
温度降到0.5时: exp(logit/0.5) ≈ exp(6) = 403 → ✅ 安全范围
```

#### 关键改进：
- 消除exp(60)级别的数值溢出风险
- 即使在最低温度(0.5)，数值仍在float32安全范围

### 2. **Reg Loss上升速度预期改善**

#### 基于数学分析：
```
Reg Loss ∝ weight_reg × (mask相关惩罚)
Mask变化速度 ∝ mask_lr × 梯度

v1配置:
- weight_reg = 5e-6
- mask_lr = 2.5e-4
- Reg Loss增速: +512/75iters = +6.8/iter

v2配置:
- weight_reg = 1e-6 (降低5倍)
- mask_lr = 2e-5 (降低12.5倍)
- 预期Reg Loss增速: <+200/75iters = <+2.7/iter (降低60%)
```

### 3. **梯度峰值风险降低**

#### Iteration 413梯度峰值分析：
```
v1观察:
- grad_norm突增到0.106 (正常值0.032的3.3倍)
- 此时temperature=3.186, mask_lr=2.5e-4

v2预期:
- mask_lr降低12.5倍 → mask梯度降低
- 预期峰值 < 0.05 (正常值的1.5倍以内)
```

---

## 📊 **从哪个checkpoint重启？**

### 选项分析

#### **选项1：从iter 400恢复（推荐）** ✅
```bash
# 优点：
- 在Reg Loss开始上升的起点
- 可以完整观察v2参数的改善效果
- 只损失14次迭代（约3小时）

# 缺点：
- 需要重跑iter 400-414

# 命令：
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1
```

#### **选项2：从iter 450恢复**
```bash
# 优点：
- 保留更多训练进度
- 损失更少时间

# 缺点：
- Reg Loss已上升较多（7800→8128）
- 可能需要更长时间才能看到改善

# 需要手动修改latest_checkpointed_iteration.txt：
echo "450" > output/checkpoints/llama8b_maskllm_training_tp8/latest_checkpointed_iteration.txt
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1
```

#### **选项3：从预稀疏checkpoint重新开始** ⚠️
```bash
# 优点：
- 最干净的起点
- 完全应用新参数

# 缺点：
- 损失400次迭代（约27小时）
- 不推荐，除非怀疑checkpoint已损坏

# 命令：
mv output/checkpoints/llama8b_maskllm_training_tp8 \
   output/checkpoints/llama8b_maskllm_training_tp8.backup_v1
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 0
```

---

## 🚀 **推荐的重启流程**

### 步骤1：停止当前训练

```bash
# 在tmux会话中按Ctrl+C停止训练
# 或者找到进程并优雅停止
pkill -f pretrain_maskllm.py
```

### 步骤2：备份当前状态（可选但推荐）

```bash
# 备份v1训练的log和checkpoint
cp output/logs/llama8b_presparse_training/training_date_25-10-02_time_00-07-46.log \
   output/logs/llama8b_presparse_training/training_v1_stopped_at_iter_414.log

# 创建checkpoint快照（可选）
cp -r output/checkpoints/llama8b_maskllm_training_tp8 \
      output/checkpoints/llama8b_maskllm_training_tp8_v1_backup
```

### 步骤3：重启训练（从iter 400）

```bash
# 在apptainer中执行
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1
```

### 步骤4：实时监控（另一个终端）

```bash
# 方法1：使用监控脚本
ssh zdhs0054@hd02-gpu1-0056 "bash /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/monitor_training_from_remote.sh tail"

# 方法2：手动tail最新log
ssh zdhs0054@hd02-gpu1-0056 "tail -f \$(ls -t /data/home/zdhs0054/zzhou/MaskLLM/output/logs/llama8b_presparse_training/*.log | head -1)"
```

---

## 📈 **预期改善指标**

### 监控重点

#### 1. **Reg Loss趋势**（最重要）
```
期望看到：
Iter 400-450: Reg Loss从7800上升到<8000 (v1是8128)
Iter 400-475: Reg Loss从7800上升到<8100 (v1是8312)

如果实际：
- <预期值 → ✅ v2参数有效
- ≈预期值 → ⚠️ 改善有限，可能需要进一步调整
- >v1值 → ❌ 配置有问题，需要debug
```

#### 2. **Grad Norm稳定性**
```
期望看到：
- 正常范围: 0.01-0.04
- 峰值 < 0.05 (v1是0.106)
- 无突变性跳跃

警告阈值：
- > 0.05 → ⚠️ 注意观察
- > 0.08 → ⚠️ 考虑进一步降低参数
- > 0.10 → ❌ 风险，准备停止
```

#### 3. **LM Loss质量**
```
期望看到：
- 保持在2.5-2.6范围
- 缓慢下降（训练有效）
- 无突然上升或发散

如果出现：
- LM Loss上升 → 可能学习率过低，但继续观察
- LM Loss停滞 → 正常，稀疏化训练early stage常见
```

#### 4. **Temperature调度**
```
v2配置下温度演化：
Iter 400: T=3.208
Iter 500: T≈2.93
Iter 1000: T≈1.83
Iter 1500: T≈1.08
Iter 2000: T=0.5 (最低值，安全)

关键观察点：
- T降到1.0附近 (iter 1500+) 时grad norm是否稳定
- T=0.5时是否仍然数值稳定
```

---

## ⚡ **快速决策表**

| 观察到的现象 | 原因 | 行动 |
|------------|------|------|
| Reg Loss增速 < +3/iter | ✅ v2有效 | 继续训练 |
| Reg Loss增速 ≈ +5-7/iter | ⚠️ 改善有限 | 观察到iter 500再决定 |
| Grad norm峰值 < 0.05 | ✅ 稳定性提升 | 继续训练 |
| Grad norm峰值 > 0.08 | ⚠️ 仍有风险 | 考虑lr-mult降到1 |
| 出现NaN | ❌ 配置仍不够保守 | 停止，lr-mult→1, weight-reg→5e-7 |
| LM Loss < 2.4 | ✅ 训练进展好 | 继续训练 |
| LM Loss > 2.7且上升 | ⚠️ 可能欠拟合 | 考虑提高lr到2e-5 |

---

## 🔧 **如果v2仍然遇到问题**

### 备选方案：v3超保守配置

```bash
# 如果v2在iter 1500+仍出现不稳定
# 进一步降低参数：

--lr-mult 1                    # mask LR = 1e-5 (等于模型LR)
--weight-reg 5e-7              # 再降低2倍
--gumbel-temperature-range 4 1.0  # 最低温度提高到1.0
```

---

## 📞 **重启时可能的问题**

### Q1: "checkpoint学习率不匹配"错误
**A**: 已解决，v2脚本包含`--no-load-optim`参数

### Q2: 重启后loss突然升高
**A**: 正常，优化器状态重置需要5-10次迭代适应

### Q3: 如何确认v2参数已生效？
**A**: 检查训练开始时的输出：
```
📊 训练配置 (稳定版v2 - 保守参数):
  - Mask学习率倍数: 2 (实际mask LR = 2e-5)
  - 正则化权重: 1e-6
  - Gumbel温度范围: 4.0 → 0.5
```

### Q4: 多久能看到改善效果？
**A**: 
- Reg Loss改善: 立即可见（前10次迭代）
- Grad norm改善: 前50次迭代
- LM Loss改善: 需要100-200次迭代

---

## ✅ **执行清单**

准备重启训练前确认：

- [ ] 已停止当前v1训练
- [ ] 已备份v1的log文件（可选）
- [ ] 确认iter 400 checkpoint存在
- [ ] 确认llama8b_presparse_training_tp8_stable.sh已更新为v2参数
- [ ] 准备好监控脚本/命令
- [ ] 了解预期的改善指标

---

## 🎯 **立即执行命令**

```bash
# 1. SSH到服务器
ssh zdhs0054@hd02-gpu1-0056

# 2. 进入项目目录和容器
cd /data/home/zdhs0054/zzhou/MaskLLM
bash run_apptainer_extended_libs.sh

# 3. 在容器内启动v2训练（从iter 400恢复）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1

# 4. 在另一个Mac终端监控
ssh zdhs0054@hd02-gpu1-0056 "bash /data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/monitor_training_from_remote.sh tail"
```

---

## 📊 **预期时间线**

```
重启后时间线（从iter 400开始）:
├─ 前10次迭代 (30分钟): 优化器适应期，可能有波动
├─ iter 410-450 (2.5小时): 应看到Reg Loss改善
├─ iter 450-500 (3小时): 梯度稳定性提升明显
├─ iter 500-1000 (30小时): 训练稳定进行
├─ iter 1000-1500 (30小时): Temperature降到1.0附近，关键观察期
└─ iter 1500-2000 (30小时): Temperature降到0.5，最终验证期
```

**总预期时间**: ~96小时（4天）完成2000次迭代

---

**祝训练顺利！🚀**
