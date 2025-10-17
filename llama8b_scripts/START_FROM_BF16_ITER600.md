# 🚀 从bf16 iter 600恢复到fp32训练 - 快速启动指南

## ✅ **已就绪**

脚本已创建并验证：
- ✅ bf16 checkpoint存在: iter_0000600
- ✅ 8个rank文件完整
- ✅ fp32恢复脚本就绪
- ✅ 自动精度转换配置完成

---

## 🚀 **正确的启动方式**

### **⚠️ 重要：必须在容器内运行！**

**错误的方式**（你刚才的操作）:
```bash
# ❌ 直接在宿主机运行
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_from_bf16.sh 0
# 错误: torchrun: command not found
```

**正确的方式**:
```bash
# 1. SSH到服务器
ssh zdhs0054@hd02-gpu1-0056

# 2. 进入项目目录
cd /data/home/zdhs0054/zzhou/MaskLLM

# 3. 启动tmux会话（防止断连）
tmux new -s fp32_from_bf16

# 4. 进入容器 ✅ 关键步骤！
bash run_apptainer_extended_libs.sh

# 5. 在容器内运行训练脚本
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_from_bf16.sh 0
```

---

## 📋 **完整操作步骤**

### **步骤1: 进入服务器和容器**

```bash
# SSH登录
ssh zdhs0054@hd02-gpu1-0056

# 进入项目
cd /data/home/zdhs0054/zzhou/MaskLLM

# 创建tmux会话
tmux new -s fp32_from_bf16

# ✅ 进入容器（关键！）
bash run_apptainer_extended_libs.sh
```

**容器内提示应该类似**:
```
Apptainer> 
```

---

### **步骤2: 在容器内启动训练**

```bash
# 现在在容器内执行
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_from_bf16.sh 0
```

**预期输出**:
```
✅ 找到bf16 checkpoint: .../iter_0000600
📊 发现 20 个可用的预处理数据文件
🔄 从bf16恢复到fp32模式:
  - 加载: bf16 checkpoint (iter 600)
  - 保存: fp32 checkpoint (iter 601+)
  - 自动精度转换: bf16 → fp32

🏃‍♂️ 开始FP32预稀疏模型训练...
```

然后开始训练，不会再报`torchrun: command not found`错误。

---

### **步骤3: 分离tmux会话**

训练开始后，分离tmux让训练在后台运行：

```bash
# 按键: Ctrl+B, 然后按 D
# 或者直接关闭终端窗口
```

---

### **步骤4: 监控训练（可选，另一个终端）**

```bash
# 新开一个终端
ssh zdhs0054@hd02-gpu1-0056

# 实时查看日志
tail -f /data/home/zdhs0054/zzhou/MaskLLM/output/logs/llama8b_presparse_training_fp32_from_bf16/training_from_bf16_*.log | grep "iteration"

# 或者重新连接tmux
tmux attach -t fp32_from_bf16
```

---

## 🔍 **为什么之前失败？**

### **根本原因**

```
宿主机环境
    ↓
run_maskllm_native.sh
    ↓
llama8b_presparse_training_tp8_fp32_from_bf16.sh
    ↓
torchrun ❌ (宿主机没有torchrun命令)
```

**正确流程**:
```
宿主机环境
    ↓
bash run_apptainer_extended_libs.sh (进入容器)
    ↓
容器环境 ✅ (有torchrun、CUDA、PyTorch)
    ↓
run_maskllm_native.sh
    ↓
llama8b_presparse_training_tp8_fp32_from_bf16.sh
    ↓
torchrun ✅ (容器内有完整环境)
```

---

## ⚠️ **常见错误和解决方案**

### **错误1: torchrun: command not found**

**原因**: 在宿主机而非容器内运行
**解决**: 
```bash
# 先进入容器
bash run_apptainer_extended_libs.sh
# 然后再运行训练脚本
```

---

### **错误2: CUDA相关错误**

**原因**: 容器绑定不完整
**解决**: 
```bash
# 使用正确的容器启动脚本
bash run_apptainer_extended_libs.sh
# (不要用其他容器启动方式)
```

---

### **错误3: 找不到checkpoint**

**原因**: 路径不对或权限问题
**解决**: 
```bash
# 检查checkpoint
ls -l output/checkpoints/llama8b_maskllm_training_tp8/iter_0000600/
# 应该显示8个mp_rank_XX目录
```

---

## 📊 **训练配置总结**

### **恢复配置**

```bash
来源: bf16 checkpoint (iter 600)
目标: fp32训练 (iter 601 → 2000)
精度转换: 自动 (PyTorch处理)
优化器: 重新初始化
起始迭代: 601
```

### **fp32参数**

```bash
学习率: 5e-6
lr-mult: 1
weight-reg: 1e-7
gumbel-temperature: 6.0 → 2.0
clip-grad: 0.3
seed: 44
save-interval: 50
```

### **节省时间**

```
bf16已完成: 600次迭代 (~36小时)
fp32剩余: 1400次迭代 (~115小时)
总计: ~6.3天 (vs 从头开始的6.8天)
节省: 0.5天
```

---

## 🎯 **监控重点（前10次迭代）**

训练开始后，重点观察：

### **Iter 601-610（优化器适应期）**

```bash
# 正常现象
Loss轻微波动: 2.57 ± 0.05 ✅
Grad norm: 0.02-0.04 ✅
无NaN ✅

# 异常情况
Loss突然上升 >3.0 ❌
Grad norm >0.1 ❌
出现NaN ❌

→ 如果异常，立即停止分析
```

### **Iter 611-650（恢复正常）**

```bash
# 正常信号
Loss趋于稳定: ~2.55 ✅
Reg loss增速 <+3/iter ✅
Mask difference开始下降 ✅

→ 继续训练到2000
```

---

## 📝 **快速检查清单**

开始前确认：

```
✅ SSH已登录: zdhs0054@hd02-gpu1-0056
✅ 在项目目录: /data/home/zdhs0054/zzhou/MaskLLM
✅ tmux会话已创建: tmux new -s fp32_from_bf16
✅ 已进入容器: bash run_apptainer_extended_libs.sh
✅ 看到容器提示: Apptainer>

→ 现在可以运行训练脚本
```

---

## 🚀 **一键启动（复制粘贴）**

```bash
# ===== 完整命令序列 =====

# 1. SSH登录
ssh zdhs0054@hd02-gpu1-0056

# 2. 进入项目并创建tmux
cd /data/home/zdhs0054/zzhou/MaskLLM
tmux new -s fp32_from_bf16

# 3. 进入容器（关键步骤！）
bash run_apptainer_extended_libs.sh

# 4. 在容器内运行训练（等待容器提示后执行）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_from_bf16.sh 0
```

---

## 📈 **预期结果**

### **成功启动的标志**

```
✅ 加载bf16 checkpoint成功
✅ 显示"loading checkpoint from..."
✅ iteration从601开始
✅ 显示"elapsed time per iteration"
✅ 无NaN错误
✅ Grad norm正常
```

### **最终目标**

```
完成2000次迭代
保存在: output/checkpoints/llama8b_maskllm_training_tp8_fp32_from_bf16/
总时间: ~6.3天
成功率: 85-90%
```

---

## 🔗 **相关文档**

- 详细原理: `BF16_TO_FP32_RESUME_GUIDE.md`
- FP32配置: `FP32_DISK_OPTIMIZED_GUIDE.md`
- 磁盘管理: `cleanup_old_checkpoints.sh`, `monitor_disk_space.sh`

---

**现在重新开始，记得先进入容器！** 🚀

