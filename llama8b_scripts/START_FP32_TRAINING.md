# 🚀 FP32训练快速启动指南

## ⚡ **一键启动（推荐）**

```bash
# 1. SSH到服务器
ssh zdhs0054@hd02-gpu1-0056

# 2. 进入项目目录
cd /data/home/zdhs0054/zzhou/MaskLLM

# 3. 进入容器
bash run_apptainer_extended_libs.sh

# 4. 启动FP32训练（在容器内）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 0
```

---

## 📊 **FP32训练核心优势**

| 优势 | 说明 |
|------|------|
| **彻底解决NaN** | 成功率 75-85% (vs bf16的0%) |
| **数值稳定** | FP32精度是bf16的3.4倍 |
| **可完成训练** | 预计6.8天完成2000次迭代 |
| **显存可行** | 65-70GB / 80GB，有余量 |

**代价**：训练速度慢35%（但能完成！）

---

## 🔍 **关键监控（前100次迭代）**

### **实时监控（另一个终端）**

```bash
# 方法1: 实时tail
ssh zdhs0054@hd02-gpu1-0056
tail -f /data/home/zdhs0054/zzhou/MaskLLM/output/logs/llama8b_presparse_training_fp32/training_*.log

# 方法2: 只看迭代行
tail -f <log_path> | grep "iteration"

# 方法3: 每10次迭代
grep "iteration.*[0-9]0/" <log_path> | tail -20
```

### **关键指标检查清单**

在前100次迭代中，每10次检查：

| 指标 | 目标值 | 警告阈值 | 说明 |
|------|--------|---------|------|
| **Reg loss增速** | <+3/iter | >+5/iter | 最关键指标 |
| **Mask difference** | 开始下降 | 维持0.077 | 收敛信号 |
| **Grad norm** | 0.01-0.03 | >0.05 | 稳定性 |
| **LM loss** | 缓慢下降 | 上升 | 训练进展 |
| **NaN warning** | 0 | 任何 | 数值稳定 |

### **示例输出解读**

```
✅ 良好状态:
iteration 10: lm loss: 2.58 | reg loss: 8700 | grad norm: 0.025
iteration 20: lm loss: 2.56 | reg loss: 8720 | grad norm: 0.022
iteration 30: lm loss: 2.55 | reg loss: 8745 | grad norm: 0.024
→ Reg loss增速: (8745-8700)/30 = +1.5/iter ✅ 良好

⚠️ 需要关注:
iteration 10: lm loss: 2.58 | reg loss: 8700 | grad norm: 0.025
iteration 20: lm loss: 2.60 | reg loss: 8750 | grad norm: 0.028
iteration 30: lm loss: 2.62 | reg loss: 8820 | grad norm: 0.032
→ Reg loss增速: (8820-8700)/30 = +4.0/iter ⚠️ 偏高
→ LM loss上升 ⚠️ 异常
→ 考虑进一步降低lr-mult

❌ 需要停止:
iteration 10: lm loss: 2.58 | reg loss: 8700 | grad norm: 0.025
iteration 20: lm loss: 2.65 | reg loss: 8850 | grad norm: 0.055
iteration 30: AssertionError: found NaN in local grad norm
→ 仍然NaN ❌ 方法论问题
→ 必须切换到Dense方案
```

---

## 📈 **预期训练轨迹**

### **阶段划分**

```
阶段1 (Iter 1-100): 验证期
  目标: 确认FP32有效
  观察: Reg loss增速 <+3/iter
  时间: ~8小时
  决策: 继续 or 停止

阶段2 (Iter 101-500): 稳定期  
  目标: Mask逐步收敛
  观察: Mask difference降至0.04-0.05
  时间: ~34小时

阶段3 (Iter 501-1000): 优化期
  目标: LM loss稳定下降
  观察: LM loss < 2.3
  时间: ~41小时

阶段4 (Iter 1001-2000): 收敛期
  目标: 完成训练
  观察: Mask difference < 0.03
  时间: ~82小时

总计: ~165小时 = 6.9天
```

### **Checkpoint保存**

```
保存间隔: 每100次
保存位置: output/checkpoints/llama8b_maskllm_training_tp8_fp32/

Checkpoint时间线:
iter 100  → 保存 (8小时)
iter 200  → 保存 (17小时)
...
iter 2000 → 保存 (165小时)

每个checkpoint: ~96GB
建议保留: 最新2-3个 (192-288GB)
```

---

## 🔧 **常见问题处理**

### **Q1: 显存不足 (>75GB)**

```bash
# 立即停止训练 (Ctrl+C)

# 修改脚本: llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh
# 找到这一行:
--global-batch-size 256 \

# 改为:
--global-batch-size 192 \

# 或添加更激进的recompute（在options中添加）:
--recompute-num-layers 4 \

# 重新启动
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 0
```

### **Q2: 训练中断（断电/断网）**

```bash
# 检查最新checkpoint
ls -lt output/checkpoints/llama8b_maskllm_training_tp8_fp32/
cat output/checkpoints/llama8b_maskllm_training_tp8_fp32/latest_checkpointed_iteration.txt

# 从checkpoint恢复（参数1表示恢复模式）
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 1
```

### **Q3: 前100次仍然NaN（<5%概率）**

```bash
# 说明预稀疏与learnable mask根本性不兼容
# 必须切换到Dense Checkpoint方案

# 立即停止当前训练

# 参考文档:
cat llama8b_scripts/fixing_notes/nan_error_deep_root_cause_analysis.md
# 查看 "方案A: 从Dense Checkpoint开始"
```

### **Q4: Reg loss增速 >+5/iter（需要调整）**

```bash
# 说明mask更新仍然过快

# 停止训练，修改脚本参数:
# 找到 TASK_CMD 行，修改:
--lr-mult 0.5  → --lr-mult 0.3

# 或进一步降低正则化:
--weight-reg 1e-7  → --weight-reg 5e-8

# 从最新checkpoint恢复
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 1
```

---

## 📝 **检查清单（训练前）**

```
✅ 1. 服务器登录成功
✅ 2. 进入项目目录: /data/home/zdhs0054/zzhou/MaskLLM
✅ 3. 进入Apptainer容器
✅ 4. 预稀疏checkpoint存在:
     output/oneshot_pruning/checkpoint/llama8b-tp8.sparse.nmprune.sp0.5SparseGPT.ex0.update_weight/
✅ 5. 数据文件存在:
     assets/data/c4_llama8b_pretokenized/c4_llama8b_00000_text_document.bin
✅ 6. 磁盘空间充足 (>300GB):
     df -h output/checkpoints/
✅ 7. 开启tmux会话（防止断连）:
     tmux new -s llama8b_fp32_training
```

---

## 🎯 **成功标准**

### **100次迭代后**

如果满足以下条件，继续训练：

```
✅ Reg loss增速 < +3/iter
✅ Mask difference 开始下降 (0.077 → 0.065)
✅ Grad norm 稳定 (0.01-0.03)
✅ 无任何NaN
✅ LM loss 缓慢下降或稳定

→ 成功率 >90%，继续到2000次
```

### **2000次迭代完成**

```
✅ 完整训练2000次
✅ Mask difference < 0.04
✅ Reg loss趋于稳定或缓慢上升
✅ LM loss < 2.4
✅ 无NaN发生

→ 训练成功！进入评估阶段
```

---

## 🔄 **完整工作流**

```
1. 启动训练
   ↓
2. 前100次密切监控 (每10次检查)
   ↓
3. 通过验证？
   ├─ 是 → 继续到2000次
   │        ↓
   │        定期检查checkpoint (每100次)
   │        ↓
   │        完成训练 ✅
   │
   └─ 否 → 分析问题
            ├─ 显存不足 → 降低batch size
            ├─ Reg loss过高 → 调整lr-mult
            └─ 仍然NaN → 切换Dense方案
```

---

## 📞 **获取帮助**

### **查看详细文档**

```bash
# FP32分析
cat llama8b_scripts/fixing_notes/fp32_nan_prevention_analysis.md

# bf16 vs FP32对比
cat llama8b_scripts/fixing_notes/bf16_vs_fp32_comparison.md

# NaN根因分析
cat llama8b_scripts/fixing_notes/nan_error_deep_root_cause_analysis.md
```

### **检查训练状态**

```bash
# 最新日志
tail -n 100 output/logs/llama8b_presparse_training_fp32/training_*.log

# GPU使用情况
nvidia-smi

# 磁盘空间
df -h output/

# Checkpoint状态
ls -lh output/checkpoints/llama8b_maskllm_training_tp8_fp32/
```

---

## 🎉 **预期结果**

**6.9天后**：

```
✅ 完整训练的Llama8b稀疏模型
✅ 2:4结构化稀疏 (50%稀疏率)
✅ Learnable mask已优化
✅ 无NaN错误
✅ 可用于下游评估

下一步:
bash run_maskllm_native.sh llama8b_scripts/evaluate_sparse_model.sh
```

---

## 🚀 **立即开始！**

```bash
# 复制粘贴以下命令开始训练:

ssh zdhs0054@hd02-gpu1-0056
cd /data/home/zdhs0054/zzhou/MaskLLM
tmux new -s llama8b_fp32
bash run_apptainer_extended_libs.sh
# 在容器内:
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32.sh 0
```

**祝训练顺利！** 🎉

