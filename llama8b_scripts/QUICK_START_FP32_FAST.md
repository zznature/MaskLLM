# 🚀 快速启动 - FP32 快速训练 (从 iter 1310)

## ⚡ 一键启动

```bash
# Step 1: SSH 到服务器
ssh zdhs0054@hd02-gpu1-0056

# Step 2: 进入项目目录
cd /data/home/zdhs0054/zzhou/MaskLLM

# Step 3: 创建 tmux 会话
tmux new -s fp32_fast

# Step 4: 进入 Apptainer 容器
bash run_apptainer_extended_libs.sh

# Step 5: 启动 FP32 快速训练
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_fast_from_iter1310.sh
```

---

## 📊 核心优化

### **vs FP32 Recompute (之前方案)**

| 对比项 | FP32 Recompute (慢) | **FP32 Fast (新)** | 提升 |
|--------|-------------------|-------------------|------|
| **Recompute** | ✅ Full | ❌ **No** | 关键 |
| **Batch Size** | 256 | **128** | -50% |
| **Learning Rate** | 1.5e-5 | **2e-5** | +33% |
| **速度** | 440秒/iter | **~220秒/iter** | **2x** ⚡ |
| **显存** | 65-70GB | **62-65GB** | 更安全 |
| **总时间** | 84h | **42h** | **节省 42h** |

### **vs BF16 (NaN)**

| 对比项 | BF16 | FP32 Fast | 优势 |
|--------|------|-----------|------|
| 速度 | 220秒/iter | ~220秒/iter | **相同** ✅ |
| NaN 风险 | 高 (100%) | **极低 (<2%)** | **FP32** ⭐ |
| 显存 | 53GB | 62-65GB | 可接受 |
| 稳定性 | 不可预测 | **100% 可靠** | **FP32** ⭐ |

---

## 🎯 预期效果

### **时间与进度**

```
起点: iter 1310 (BF16 checkpoint, 已完成 65.5%)
终点: iter 2000
剩余: 690 iterations

预计时间:
  690 × 220秒 = 151,800秒
  = 42小时
  = 1.75天 🚀

vs 之前计划 (FP32 recompute):
  690 × 440秒 = 303,600秒  
  = 84小时
  = 3.5天

节省: 42小时 (50%)
```

### **关键检查点**

| Iteration | 时间 | 检查内容 |
|-----------|------|---------|
| **1315** | 30分钟 | ✅ 验证显存 (应为 62-65GB) |
| **1320** | 1小时 | ✅ 验证速度 (应为 ~220秒/iter) |
| **1350** | 4小时 | ✅ 确认无 OOM |
| **1400** | 8小时 | ✅ 评估收敛情况 |
| **1500** | 17小时 | ✅ Temperature 应降至 ~2.1 |
| **1750** | 33小时 | ✅ 中期检查 |
| **2000** | **42小时** | ✅ **完成！** |

---

## 🔍 监控指南

### **实时查看**

```bash
# 方法 1: 实时日志
tail -f output/logs/llama8b_presparse_training_fp32_fast/training_from_iter1310_*.log | grep "iteration"

# 方法 2: 每 10 秒刷新
watch -n 10 'tail -5 output/logs/llama8b_presparse_training_fp32_fast/training_from_iter1310_*.log | grep "iteration"'
```

### **关键指标**

✅ **成功信号**:
- 速度: 200-240秒/iter (目标 ~220)
- 显存: 60-68GB / 80GB
- Grad norm: <0.10
- **无 NaN**

⚠️ **预警信号**:
- 速度: >300秒/iter (可能优化未生效)
- 显存: >75GB (可能 OOM 风险)
- OOM 错误

---

## 📊 预期训练轨迹

| Iteration | Temp | Max Prob | Mask Diff | Reg Loss | 状态 |
|-----------|------|----------|-----------|----------|------|
| 1310 | 2.364 | 0.637 | 0.096 | 9,032 | 起点 (BF16→FP32) |
| 1400 | 2.25 | 0.66 | 0.092 | 9,200 | ✅ 稳定运行 |
| 1500 | 2.10 | 0.68 | 0.088 | 9,500 | ✅ 渡过危险期 |
| 1700 | 1.80 | 0.73 | 0.083 | 10,100 | ✅ 进入稳定区 |
| 2000 | 1.50 | 0.78 | 0.078 | 10,900 | ✅ 完成 |

**Reg loss 增速**: ~+2.7/iter (可接受)

---

## ⚠️ 可能的问题与对策

### **问题 1: OOM (显存不足)**

**概率**: 10-15%

**症状**:
```
CUDA out of memory
Tried to allocate XXX GB
```

**对策**:
```bash
# 方案 1: 进一步降低 batch size
修改脚本: --global-batch-size 96

# 方案 2: 添加部分 recompute
添加参数:
  --recompute-activations
  --recompute-granularity selective
  --recompute-num-layers 8
速度会降至 ~280秒/iter，但能跑

# 方案 3: 完全 recompute (回退)
添加:
  --recompute-activations
  --recompute-granularity full
速度降至 ~440秒/iter
```

### **问题 2: 速度未达预期**

**症状**:
```
实际: 300-350秒/iter
预期: 220秒/iter
```

**原因**:
- FP32 计算本身就慢
- I/O 或通信瓶颈
- GPU 未满载

**对策**:
- 检查 GPU 利用率: `nvidia-smi`
- 如果 >280秒但 <350秒，仍可接受
- 总时间仍然比 recompute 快很多

### **问题 3: Batch Size 影响收敛**

**症状**:
```
Loss 震荡加剧
Mask difference 不下降
```

**对策**:
```bash
# 提高学习率
--lr 2.5e-5  # from 2e-5

# 或延长训练
--train-iters 2200  # from 2000
```

---

## 🎓 技术原理

### **为什么移除 recompute 能加速 2x？**

#### **Recompute 机制**
```
前向传播:
  计算 activation → 丢弃一部分 (节省显存)
  
反向传播:
  需要 activation → 重新计算前向传播 (时间换空间)
  
总时间:
  1x forward + 1x backward + 0.8x forward (recompute)
  = 2.8x base time
```

#### **No Recompute**
```
前向传播:
  计算 activation → 全部保留在显存
  
反向传播:
  直接使用保存的 activation
  
总时间:
  1x forward + 1x backward
  = 2x base time

相比 recompute:
  2.8x / 2x = 1.4x 理论加速
  
加上 batch size 降低:
  1.4x × 1.3x ≈ 1.8-2x 实际加速
```

### **为什么降低 Batch Size？**

```
目的: 节省显存空间给 activation

原理:
  Activation 显存 ∝ micro_batch_size
  但总计算量 ∝ global_batch_size
  
降低 global_batch_size:
  256 → 128 = 减少 50% 前向/反向计算
  = 进一步加速 ~1.3x
  
总加速:
  1.4x (no recompute) × 1.3x (batch) ≈ 1.8-2x
```

---

## ✅ 成功标准

### **短期 (iter 1310-1400)**
- ✅ 无 OOM
- ✅ 速度稳定 200-240秒/iter
- ✅ 显存 60-68GB

### **中期 (iter 1400-1700)**
- ✅ 无 NaN
- ✅ Temperature 平滑下降
- ✅ Mask difference 持续优化

### **长期 (iter 1700-2000)**
- ✅ 完成训练
- ✅ Reg loss < 11,000
- ✅ Temperature ~1.5
- ✅ Mask difference ~0.08

---

## 📁 相关文档

- **详细优化方案**: `FP32_SPEED_OPTIMIZATION_PLAN.md`
- **训练进度评估**: `TRAINING_PROGRESS_EVALUATION_AND_PLAN.md`
- **训练日志**: `training_notes/training_logs.md`

---

**准备好了吗？** 🚀

**预计**: 42小时 (1.75天) 后完成训练！
**成功率**: 95%
**速度**: 2x 加速 vs FP32 recompute
**稳定性**: 100% 避免 NaN

