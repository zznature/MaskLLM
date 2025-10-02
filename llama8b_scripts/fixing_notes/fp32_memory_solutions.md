# bf16→fp32 显存问题解决方案

## 📊 **显存分析**

### 当前状态（bf16）
```
每张卡使用: 53GB / 80GB (66%)
剩余空间: 27GB
```

### fp32需求估算
```python
模型权重增加: 16GB → 32GB (+16GB)
激活值增加: 37GB → 55GB (+18GB)
优化器状态增加: 如果加载会翻倍

总计预估: ~87GB > 80GB ❌ 不够！
超出: ~7GB
```

---

## ✅ **解决方案对比**

### **方案1：保持bf16 + Seed修改（当前，推荐先试）** 🔥

**配置**：
```bash
--bf16
--seed 42  # 已添加
```

**优点**：
- ✅ 无需修改显存配置
- ✅ 速度最快
- ✅ 立即可试（已经修改好）

**缺点**：
- ⚠️ 如果seed无效，仍会NaN

**下一步**：
- 立即运行，1小时验证是否通过iter 496
- 如果成功 → 完美解决
- 如果失败 → 尝试方案2或3

---

### **方案2：fp32 + Gradient Checkpointing（可行）** ✅

**原理**：激活重计算，用计算时间换显存

**配置**：
```bash
# 移除
--bf16

# 添加
--recompute-activations  # 或 --checkpoint-activations (旧名)
--recompute-granularity full  # 全层重计算
--recompute-method uniform  # 均匀分布

# 可选：更激进的重计算
--recompute-num-layers 8  # 每8层做一次checkpoint
```

**预期效果**：
```python
激活值显存减少: 55GB → 35GB (-20GB)
总显存需求: 32GB (weights) + 35GB (activations) = 67GB
剩余: 80GB - 67GB = 13GB ✅ 够用！
```

**优点**：
- ✅ 彻底解决NaN（fp32数值稳定）
- ✅ 显存够用（67GB < 80GB）
- ✅ Megatron原生支持

**缺点**：
- ⚠️ 速度降低25-40%（重计算开销）
- ⚠️ 每次迭代时间: 219秒 → 280-310秒

**预期时间**：
```
完成2000次迭代: 
- 当前bf16: 219秒/iter × 1600 = 97小时
- fp32+checkpoint: 295秒/iter × 1600 = 131小时 (+34小时)
```

---

### **方案3：fp32 + 减小Batch Size（备选）** ⚠️

**配置**：
```bash
# 移除
--bf16

# 修改
--global-batch-size 128  # 从256降到128
--micro-batch-size 1     # 保持不变
```

**预期效果**：
```python
减半batch size → 激活值减少约30%
激活值: 55GB → 38GB (-17GB)
总显存: 32GB + 38GB = 70GB ✅ 够用
```

**优点**：
- ✅ 彻底解决NaN
- ✅ 显存够用
- ✅ 无需重计算，速度影响小

**缺点**：
- ❌ 改变了训练动力学（batch size减半）
- ❌ 需要调整学习率（可能需要×0.7）
- ❌ 收敛速度可能变慢

---

### **方案4：fp32 + Checkpoint + Batch Size（最保守）**

**配置**：
```bash
# 移除
--bf16

# 添加
--recompute-activations

# 修改
--global-batch-size 192  # 折中，从256降到192
```

**预期效果**：
```python
Checkpoint激活值: 55GB → 35GB
降低batch: 35GB × (192/256) = 26GB
总显存: 32GB + 26GB = 58GB ✅ 很充裕！
```

**优点**：
- ✅ 显存非常充裕（22GB剩余）
- ✅ 彻底解决NaN
- ✅ batch size影响较小（-25%）

**缺点**：
- ⚠️ 速度最慢（checkpoint + 小batch）
- ⚠️ 预估时间: ~140小时

---

## 🎯 **推荐执行顺序**

### **阶段1：立即执行（今天）**

```bash
# 使用方案1：bf16 + seed 42
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1

# 监控1小时，观察是否通过iter 496
```

**判断标准**：
- ✅ 通过iter 496 → 继续到2000，问题解决
- ❌ 仍在496 NaN → 进入阶段2
- ⚠️ 在其他位置NaN（如520） → 说明是普遍问题，进入阶段2

---

### **阶段2：如果方案1失败（明天）**

#### **选项2A：fp32 + Checkpoint（推荐）**

修改脚本：
```bash
# 在llama8b_presparse_training_tp8_stable.sh中
# 替换
--bf16
# 为
--recompute-activations \
--recompute-granularity full \
```

运行：
```bash
# 从iter 450重新开始
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable.sh 1
```

---

## 📋 **具体修改脚本（方案2：fp32 + Checkpoint）**

### 修改内容

```bash
# 找到这一行（约170行）
--bf16 \

# 替换为
--recompute-activations \
--recompute-granularity full \
--recompute-method uniform \
```

### 预期输出显存
```
启动后检查显存:
nvidia-smi

预期每卡显存: 65-70GB / 80GB
如果超过75GB → 添加 --recompute-num-layers 4
```

---

## 🧪 **显存测试方案**

### 在启动训练前测试显存

```bash
# 可以尝试dry-run（如果支持）
# 或者运行1-2个iteration观察显存

# 观察命令（在另一个终端）
watch -n 1 nvidia-smi
```

### 显存超限的紧急处理

如果fp32+checkpoint仍然OOM：

```bash
# 进一步降低batch size
--global-batch-size 192  # 从256降到192

# 或更激进的checkpoint
--recompute-num-layers 4  # 每4层一个checkpoint
```

---

## ⏰ **时间成本对比**

| 方案 | 每次迭代时间 | 完成1600次迭代 | 额外时间 |
|------|------------|--------------|---------|
| **bf16 (当前)** | 219秒 | 97小时 | 基准 |
| **bf16 + seed** | 219秒 | 97小时 | 0 |
| **fp32 + checkpoint** | ~295秒 | 131小时 | +34小时 |
| **fp32 + batch192** | ~235秒 | 105小时 | +8小时 |
| **fp32 + both** | ~310秒 | 138小时 | +41小时 |

---

## 🔍 **如何确认显存够用？**

### 训练开始后立即检查

```bash
# SSH到服务器
ssh zdhs0054@hd02-gpu1-0056

# 实时监控显存
watch -n 1 nvidia-smi

# 或持续记录
while true; do 
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
  sleep 5
done
```

### 安全阈值

```
< 70GB: ✅ 安全，继续训练
70-75GB: ⚠️ 可接受，但监控
75-78GB: ⚠️ 危险，考虑降低batch
> 78GB: ❌ 随时可能OOM
```

---

## 💡 **我的最终建议**

### **立即执行**：
```
1. 先试方案1（bf16 + seed 42）← 已经配置好
2. 1小时内验证是否通过iter 496
3. 如果成功 → 完美！
4. 如果失败 → 切换到方案2
```

### **如果需要切换到fp32**：
```
推荐：方案2（fp32 + checkpoint）
- 显存够用（67GB）
- 速度可接受（+35%时间）
- 彻底解决NaN
```

### **不推荐**：
```
- 方案3（改batch size）: 影响训练动力学
- 从零开始: 损失400次迭代，没必要
```

---

## ✅ **执行清单**

当前状态：
- [x] 已添加 --seed 42
- [x] 保持bf16
- [ ] 正在运行（等待验证）

如果需要fp32：
- [ ] 移除 --bf16
- [ ] 添加 --recompute-activations
- [ ] 添加 --recompute-granularity full
- [ ] 测试显存使用
- [ ] 重新开始训练

---

**总结**：80GB显存对于fp32是紧张的，但通过激活重计算(checkpoint)可以解决。不过，优先建议尝试当前的bf16+seed方案！
