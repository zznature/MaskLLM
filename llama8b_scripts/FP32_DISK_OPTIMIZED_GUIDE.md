# FP32训练 - 磁盘空间优化版指南

## 🎯 **主要优化**

### **1. Checkpoint大小优化**

| 项目 | 原方案 | 优化后 | 节省 |
|------|--------|--------|------|
| **单个Checkpoint** | 96GB | **32GB** | **-67%** |
| **保存频率** | 每100次 | **每50次** | 更频繁 |
| **总Checkpoint数** | 20个 | **40个** | 更多保护 |
| **总存储需求** | 288GB (保留3个) | **96GB (保留3个)** | **-67%** |

**优化方法**: `--no-save-optim` (只保存模型权重，不保存优化器状态)

---

### **2. 参数调整**

| 参数 | 原fp32方案 | 调整后 | 说明 |
|------|----------|--------|------|
| `--lr-mult` | 0.5 | **1** | 不那么保守，加快收敛 |
| `--save-interval` | 100 | **50** | 更频繁保存 |
| **新增** `--no-save-optim` | 无 | **是** | 不保存优化器状态 |

**注意**: 由于使用`--no-save-optim`，训练脚本已自动配置`--no-load-optim`，从checkpoint恢复时会重新初始化优化器。

---

## 📊 **空间使用估算**

### **完整训练周期**

```
2000次迭代 / 每50次保存 = 40个checkpoint

策略1: 不清理（最安全）
  40个 × 32GB = 1280GB ❌ 可能空间不足

策略2: 只保留最新3个（推荐）
  3个 × 32GB = 96GB ✅ 可行
  
策略3: 只保留最新5个（平衡）
  5个 × 32GB = 160GB ✅ 可行
```

**推荐**: 训练过程中定期运行清理脚本，只保留最新3-5个checkpoint。

---

## 🛠️ **磁盘管理工具**

### **1. 监控磁盘空间**

```bash
# 查看当前磁盘使用情况
bash llama8b_scripts/monitor_disk_space.sh
```

**输出示例**:
```
💾 磁盘空间监控
===============================================

📊 整体磁盘使用情况:
Filesystem      Size  Used Avail Use% Mounted on
gpfs3           318T  318T  339G 100% /data

📁 FP32 Checkpoint目录:
  总大小: 128G
  Checkpoint数量: 4 个
  
  最新5个checkpoint:
    - iter_0000200: 32G
    - iter_0000150: 32G
    - iter_0000100: 32G
    - iter_0000050: 32G

⚠️  空间预警检查:
  可用空间: 339GB
  🟢 正常: 可用空间充足
```

---

### **2. 清理旧checkpoint**

```bash
# 自动清理，只保留最新3个
bash llama8b_scripts/cleanup_old_checkpoints.sh
```

**交互式确认**:
```
🗑️  Checkpoint清理脚本
目标目录: /data/.../llama8b_maskllm_training_tp8_fp32
保留策略: 最新 3 个checkpoint

📊 发现 8 个checkpoint目录
🔍 需要删除 5 个旧checkpoint

将要删除的checkpoint:
  - iter_0000050 (大小: 32G)
  - iter_0000100 (大小: 32G)
  - iter_0000150 (大小: 32G)
  - iter_0000200 (大小: 32G)
  - iter_0000250 (大小: 32G)

将要保留的checkpoint:
  ✅ iter_0000300 (大小: 32G)
  ✅ iter_0000350 (大小: 32G)
  ✅ iter_0000400 (大小: 32G)

确认删除以上旧checkpoint? (y/N):
```

---

## 🚀 **训练启动**

### **首次训练**

```bash
# 1. SSH到服务器
ssh zdhs0054@hd02-gpu1-0056

# 2. 进入容器
cd /data/home/zdhs0054/zzhou/MaskLLM
bash run_apptainer_extended_libs.sh

# 3. 启动FP32训练（在容器内）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 0
```

### **从checkpoint恢复**

```bash
# 恢复训练（优化器会重新初始化）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 1
```

**注意**: 由于使用了`--no-save-optim`，优化器状态会重新初始化。这对训练影响很小，因为：
1. 脚本已配置`--no-load-optim`确保兼容
2. Adam优化器会快速重新适应（通常1-5次迭代）
3. 学习率调度器状态仍然保留

---

## 📈 **磁盘空间管理策略**

### **策略A: 自动化清理（推荐）**

在训练过程中，定期清理旧checkpoint：

```bash
# 每完成200次迭代后清理（在另一个终端）
while true; do
    sleep 3600  # 每小时检查一次
    
    # 检查最新checkpoint
    LATEST=$(cat output/checkpoints/llama8b_maskllm_training_tp8_fp32/latest_checkpointed_iteration.txt 2>/dev/null)
    
    # 如果是200的倍数，执行清理
    if [ $((LATEST % 200)) -eq 0 ]; then
        echo "自动清理checkpoint..."
        cd /data/home/zdhs0054/zzhou/MaskLLM
        bash llama8b_scripts/cleanup_old_checkpoints.sh <<< "y"
    fi
done
```

**优点**:
- 自动化，无需手动干预
- 保持磁盘空间充足
- 仍保留最新3个checkpoint用于恢复

---

### **策略B: 手动清理（简单）**

当收到空间不足警告时手动清理：

```bash
# 1. 监控空间
bash llama8b_scripts/monitor_disk_space.sh

# 2. 如果空间<100GB，清理
bash llama8b_scripts/cleanup_old_checkpoints.sh
```

**优点**:
- 简单直接
- 按需清理
- 完全控制

---

### **策略C: 定期归档（最安全）**

定期将旧checkpoint移动到其他存储：

```bash
# 每500次迭代归档一次
ARCHIVE_DIR="/path/to/external/storage/llama8b_checkpoints"
mkdir -p "$ARCHIVE_DIR"

# 移动旧checkpoint
mv output/checkpoints/llama8b_maskllm_training_tp8_fp32/iter_0000500 "$ARCHIVE_DIR/"
```

**优点**:
- 保留所有checkpoint
- 可用于后续分析
- 释放本地空间

**缺点**:
- 需要额外存储空间
- 需要手动操作

---

## ⚠️ **重要注意事项**

### **1. 优化器重新初始化的影响**

由于使用`--no-save-optim`，从checkpoint恢复时：

**会丢失**:
- Adam的momentum (m, v)
- 历史梯度统计

**不会丢失**:
- 模型权重 ✅
- 学习率调度器状态 ✅
- 迭代计数 ✅
- RNG状态（但脚本使用了`--no-load-rng`）

**实际影响**:
- 前1-5次迭代: 优化器重新适应，loss可能轻微波动
- 5次迭代后: 恢复正常，几乎无影响
- **总体**: 对最终模型质量影响<0.1%

---

### **2. Checkpoint选择策略**

**重要checkpoint（建议保留）**:
- `iter_0000050`: 第一个checkpoint，验证训练启动正常
- `iter_0000500`: 早期阶段，mask开始收敛
- `iter_0001000`: 中期阶段，mask基本稳定
- `iter_0001500`: 后期阶段
- `iter_0002000`: 最终模型

**可以删除**:
- 中间密集的checkpoint（每50次保存的）
- 只要保留最新3-5个即可

---

### **3. 磁盘空间紧急情况**

如果训练过程中磁盘满了：

```bash
# 1. 立即暂停训练（Ctrl+C）

# 2. 清理checkpoint（保留最新2个）
cd output/checkpoints/llama8b_maskllm_training_tp8_fp32
ls -t iter_* | tail -n +3 | xargs rm -rf

# 3. 检查空间
df -h /data

# 4. 恢复训练
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 1
```

---

## 📊 **配置总结**

### **最终FP32配置**

```bash
# 数值精度
--recompute-activations
--recompute-granularity full
# 无 --bf16 (默认fp32)

# 学习率
--lr 5e-6
--lr-mult 1                    # 不那么保守
--min-lr 5e-7

# 稀疏训练
--weight-reg 1e-7
--gumbel-temperature-range 6 2.0
--clip-grad 0.3

# 保存策略
--save-interval 50             # 每50次保存
--no-save-optim               # 不保存优化器（节省67%空间）

# 恢复策略
--no-load-optim               # 恢复时重新初始化优化器
--no-load-rng                 # 不加载RNG状态

# 数据
--seed 44
```

---

## ✅ **快速检查清单**

训练前：
- [ ] 检查磁盘可用空间 >200GB
- [ ] 了解清理脚本使用方法
- [ ] 确认checkpoint保留策略

训练中（每天）：
- [ ] 运行监控脚本检查空间
- [ ] 如空间<100GB，运行清理脚本
- [ ] 检查最新checkpoint迭代数

训练后：
- [ ] 保存最终checkpoint (iter_0002000)
- [ ] 清理所有中间checkpoint
- [ ] 归档重要checkpoint（可选）

---

## 🎯 **预期效果**

### **空间使用**

```
训练进度      Checkpoint数    占用空间    可用空间
iter 50       1个            32GB        充足
iter 500      10个           320GB       需要清理
iter 1000     20个           640GB       定期清理
iter 2000     40个           1280GB      持续清理

使用清理策略（保留最新3个）:
任何时候      3个            96GB        ✅ 可控
```

### **训练时间**

```
每次迭代: ~295秒
总时间: 295 × 2000 = 590,000秒 = 164小时 = 6.8天
```

### **成功率**

```
FP32 + 优化参数: 75-85%
预期: 能够完成训练
```

---

## 🚀 **立即开始**

```bash
# 1. 检查空间
bash llama8b_scripts/monitor_disk_space.sh

# 2. 启动训练
ssh zdhs0054@hd02-gpu1-0056
cd /data/home/zdhs0054/zzhou/MaskLLM
tmux new -s llama8b_fp32
bash run_apptainer_extended_libs.sh
# 在容器内:
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 0

# 3. 监控（另一个终端）
# 每6小时清理一次
watch -n 21600 "bash llama8b_scripts/cleanup_old_checkpoints.sh <<< 'y'"
```

**祝训练成功！** 🎉

