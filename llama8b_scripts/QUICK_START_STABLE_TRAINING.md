# 🚀 快速启动 - Llama8b 稳定化训练

## ⚡ 一键启动命令

```bash
# Step 1: SSH到服务器
ssh zdhs0054@hd02-gpu1-0056

# Step 2: 进入项目目录
cd /data/home/zdhs0054/zzhou/MaskLLM

# Step 3: (推荐) 创建 tmux 会话
tmux new -s stable_training

# Step 4: 进入 Apptainer 容器
bash run_apptainer_extended_libs.sh

# Step 5: 启动稳定化训练
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_stable_resume.sh
```

---

## 📊 训练概览

| 项目 | 信息 |
|------|------|
| **起点** | iter 1125 (bf16 checkpoint) |
| **目标** | iter 2000 |
| **剩余迭代** | 875 次 |
| **精度** | BF16 |
| **预计时间** | ~53.5 小时 (2.2 天) |
| **关键期** | iter 1125-1200 (~4.5小时) |
| **成功率** | 85-90% |

---

## 🎯 核心改进

### **vs 之前失败的配置**

| 参数 | 之前 (NaN at 1143) | 现在 (稳定化) | 改进效果 |
|------|-------------------|--------------|---------|
| **Temperature 终点** | 0.2 | **1.0** | 避免过度确定性 |
| **Mask LR mult** | 10 | **8** | 更平滑收敛 |
| **Gradient clip** | 1.0 | **0.5** | 防止梯度爆炸 |
| **Weight reg** | 2e-6 | **1e-6** | 降低 reg loss 压力 |
| **Seed** | 46 | **48** | 避免相同数值陷阱 |
| **Save interval** | 25 | **10** | 快速定位问题 |

---

## 🔍 监控指南

### **危险期监控 (iter 1125-1200)**

在另一个终端实时查看：

```bash
# 监控训练进度
tail -f output/logs/llama8b_presparse_training/training_stable_resume_*.log | grep "iteration"

# 或使用简化版
watch -n 10 'tail -5 output/logs/llama8b_presparse_training/training_stable_resume_*.log | grep "iteration"'
```

### **成功信号** ✅

- Grad norm 保持 < 0.10
- Reg loss 增速 < +5/iter
- Mask difference 继续下降 (0.112 → 0.105)
- Temperature 平滑下降 (1.864 → ~1.75)
- **无任何 NaN warning**

### **预警信号** ⚠️

- Grad norm 突然 > 0.15
- Reg loss 单次跳变 >50
- 日志中出现 "NaN" 或 "Inf"
- Loss scale 异常变化

---

## 📈 预期训练轨迹

### **关键检查点**

| Iteration | Temperature | Mask Diff | Reg Loss | 风险等级 | 检查内容 |
|-----------|-------------|-----------|----------|---------|---------|
| **1125** | 1.864 | 0.112 | 9,467 | - | 起点 |
| **1150** | ~1.80 | ~0.108 | ~9,600 | 🟡 中 | 第一关键点 |
| **1175** | ~1.76 | ~0.105 | ~9,730 | 🟡 中 | 第二关键点 |
| **1200** | ~1.72 | ~0.102 | ~9,860 | 🟢 低 | **危险期结束** |
| **1500** | ~1.35 | ~0.088 | ~11,000 | 🟢 极低 | 中期评估 |
| **2000** | ~1.00 | ~0.078 | ~12,600 | 🟢 极低 | 完成 |

---

## ✅ 关键决策点

### **Checkpoint 1: iter 1150** (启动后 ~1.5小时)

```bash
# 检查最近10次迭代
grep "iteration.*consumed samples" output/logs/llama8b_presparse_training/training_stable_resume_*.log | tail -10

# 关注
- Grad norm 是否 < 0.10
- 是否有任何 NaN
```

**决策**:
- ✅ 无 NaN, grad norm 正常 → **继续**
- ❌ 出现 NaN → **停止, 执行备选方案**

### **Checkpoint 2: iter 1200** (启动后 ~5小时)

```bash
# 检查是否成功渡过危险期
tail -50 output/logs/llama8b_presparse_training/training_stable_resume_*.log | grep "iteration.*1200"
```

**决策**:
- ✅ 到达 iter 1200 无 NaN → **成功！可以放心让它跑到 2000**
- ❌ 未到达或有 NaN → **评估是否需要更保守参数**

---

## 🔧 备选方案

### **如果 iter 1150-1200 再次 NaN**

#### **选项 A: 更保守的 BF16**

修改 `llama8b_presparse_training_tp8_stable_resume.sh`:

```bash
# 找到 TASK_CMD，修改为
TASK_CMD="--gumbel-scale-range 5e1 2.5e2 --gumbel-temperature-range 4 1.5 --N 2 --M 4 --mask-only --prior-strength 3.0 --lr-mult 5 --weight-reg 5e-7"

# 修改 clip-grad
--clip-grad 0.3  # 从 0.5 改为 0.3
```

从最近的稳定 checkpoint (1135, 1145 等) 恢复。

#### **选项 B: 切换到 FP32 激进模式**

如果选项 A 仍失败，使用 FP32:

```bash
# 创建 FP32 激进配置
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_adjusted.sh 1
```

---

## 💾 磁盘空间管理

### **清理旧 checkpoint (可选)**

```bash
# 保留关键 checkpoint: 1000, 1050, 1100, 1125
# 删除 iter 1125 之前的其他 checkpoint
cd output/checkpoints/llama8b_maskllm_training_tp8
rm -rf iter_000{0025..0975}
rm -rf iter_000{1025,1075}

# 检查剩余空间
df -h output/checkpoints/
```

### **预计空间需求**

- 每个 checkpoint: ~32GB (--no-save-optim)
- 保存间隔: 每 10 次
- iter 1125-2000: 87 个 checkpoint
- 总需求: ~2.8TB
- **建议**: 每 50 次手动清理一次，保留最近 20 个

---

## 📝 日志分析

### **快速提取关键指标**

```bash
# 提取最近 20 次迭代的关键指标
grep "iteration.*consumed samples" output/logs/llama8b_presparse_training/training_stable_resume_*.log | tail -20 | awk '{print "Iter:"$2, "LM:"$22, "Reg:"$26, "Grad:"$34, "Temp:"$46, "MaskDiff:"$50}'
```

### **检查 NaN 历史**

```bash
# 查找所有 NaN 记录
grep -i "nan" output/logs/llama8b_presparse_training/training_stable_resume_*.log
```

---

## 🎓 成功后的下一步

### **训练完成 (iter 2000)** 

1. **评估稀疏模型性能**
   ```bash
   bash run_maskllm_native.sh llama8b_scripts/evaluate_sparse_model.sh
   ```

2. **检查最终指标**
   ```bash
   # 最后一次迭代
   tail -20 output/logs/llama8b_presparse_training/training_stable_resume_*.log | grep "iteration.*2000"
   
   # 期待
   - Mask difference: 0.07-0.08
   - Temperature: ~1.0
   - Reg loss: <13,000
   ```

3. **对比不同 checkpoint 的性能**
   - iter 1500: 中期稀疏模型
   - iter 1750: 后期稀疏模型
   - iter 2000: 最终稀疏模型

---

## ❓ 常见问题

### **Q1: 如果忘记进入容器就运行了怎么办？**

A: 会看到 `torchrun: command not found`。解决：
```bash
bash run_apptainer_extended_libs.sh
# 然后重新运行训练命令
```

### **Q2: tmux 会话断开了怎么办？**

A: 重新连接：
```bash
tmux attach -t stable_training
```

### **Q3: 训练速度太慢？**

A: BF16 应该是 ~220秒/iter。如果更慢：
- 检查是否有其他进程占用 GPU
- 检查是否误用了 FP32 脚本
- 运行 `nvidia-smi` 确认 GPU 利用率

### **Q4: 磁盘空间不足？**

A: 紧急清理：
```bash
# 只保留最近 10 个 checkpoint
cd output/checkpoints/llama8b_maskllm_training_tp8
ls -td iter_* | tail -n +11 | xargs rm -rf
```

---

**祝训练顺利！** 🚀

**预计成功率**: 85-90%  
**关键期**: 前 5 小时 (iter 1125-1200)  
**总时间**: ~2.2 天

**详细分析**: 见 `STABLE_TRAINING_PLAN.md`

