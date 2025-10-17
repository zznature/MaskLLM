# BF16 Conservative Plan B - 恢复训练指南

## 🔴 问题诊断

### 失败原因
- **时间**: iter 100 保存checkpoint时
- **错误**: `RuntimeError: file write failed` 
- **根因**: 磁盘空间不足 (98% 使用率: 311T/318T)

### 当前状态
- ✅ **iter 75**: 完整保存 (53GB)
- ❌ **iter 100**: 保存失败, 不完整 (33GB, 应该53GB)
- 📝 **latest_checkpointed_iteration.txt**: 指向75 (正确)

---

## ✅ 恢复方案

### 方案1: 优化checkpoint策略恢复 (推荐)

**优化措施**:
1. ❌ 删除不完整的 iter 100 checkpoint (释放33GB)
2. 📈 增加 save_interval: 25 → 50 (减少50% checkpoint数)
3. 💾 已启用 --no-save-optim (节省optimizer空间)
4. 🗑️ 可选: 删除旧checkpoint保留最新2个

**恢复命令**:
```bash
cd /data/home/zdhs0054/zzhou/MaskLLM
bash run_maskllm_native.sh llama8b_scripts/resume_bf16_conservative_from_iter75.sh
```

**预期效果**:
- 从 iter 75 恢复
- 剩余训练: 1925 iters
- Checkpoint保存: iter 125, 175, 225, ... , 2000
- 总checkpoint数: ~40个 (vs 原80个)
- 节省空间: ~50%

---

### 方案2: 清理旧checkpoint后恢复

**步骤**:

1. **检查可删除的checkpoint**:
```bash
ls -lh /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/
```

2. **删除其他训练的旧checkpoint** (谨慎操作):
```bash
# 示例: 删除之前失败的训练checkpoint
rm -rf /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8_fp32_*
rm -rf /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8/iter_00001[0-2]*
```

3. **使用方案1的恢复脚本**

---

### 方案3: 仅保留最新N个checkpoint

**创建自动清理脚本** (在训练中定期运行):

```bash
#!/bin/bash
# 仅保留最新2个checkpoint
cd /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8_bf16_conservative_plan_b/
ls -d iter_* | sort -V | head -n -2 | xargs -r rm -rf
```

---

## 📋 恢复训练关键参数对比

| 参数 | 原始训练 | 恢复训练 | 说明 |
|------|---------|---------|------|
| **起点** | Sparse ckpt | iter 75 | 从最后完整checkpoint |
| **save_interval** | 25 | **50** | 减少checkpoint数量 |
| **--no-save-optim** | 否 | **是** | 节省optimizer状态空间 |
| **剩余iters** | 2000 | 1925 | 已完成75次迭代 |
| **预期checkpoint数** | 80 | **40** | 减少50% |

---

## 🔍 监控要点

### 恢复后立即检查

**iter 76-100** (恢复初期):
```bash
tail -f output/logs/llama8b_presparse_training_bf16_conservative_plan_b/training_bf16_conservative_resume_iter75_*.log | grep "iteration"
```

**关键指标**:
1. ✅ LM loss 应该接近 iter 75 的值 (~5.47)
2. ✅ Reg loss 应该接近 iter 75 的值 (~5600)
3. ✅ Grad norm 应该正常 (0.03-0.06)
4. ✅ 不应该有 "loading checkpoint" 错误

### 下一个checkpoint保存 (iter 125)

**检查磁盘空间**:
```bash
df -h /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/
```

**验证checkpoint完整性**:
```bash
du -sh output/checkpoints/llama8b_maskllm_training_tp8_bf16_conservative_plan_b/iter_0000125/
# 应该是 ~53GB (与iter 75相同)
```

---

## 🚨 紧急清理空间方案

如果恢复后仍然空间不足:

### 删除不必要的文件

1. **CACHE目录** (可重新生成):
```bash
du -sh /data/home/zdhs0054/zzhou/MaskLLM/CACHE/
# 如果很大, 可以删除
rm -rf /data/home/zdhs0054/zzhou/MaskLLM/CACHE/*.npy
```

2. **旧的训练日志**:
```bash
du -sh /data/home/zdhs0054/zzhou/MaskLLM/output/logs/
# 可以压缩或删除旧日志
```

3. **Oneshot pruning的其他方法checkpoint**:
```bash
# 如果不需要SparseGPT方法的checkpoint
rm -rf /data/home/zdhs0054/zzhou/MaskLLM/output/oneshot_pruning/checkpoint/*SparseGPT*
```

---

## ⚙️ 恢复训练完整命令

### 快速恢复 (使用准备好的脚本)

```bash
cd /data/home/zdhs0054/zzhou/MaskLLM
bash run_maskllm_native.sh llama8b_scripts/resume_bf16_conservative_from_iter75.sh
```

### 手动恢复 (如果需要调整参数)

1. **删除不完整checkpoint**:
```bash
rm -rf /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8_bf16_conservative_plan_b/iter_0000100
```

2. **确认latest_checkpointed_iteration.txt**:
```bash
echo "75" > /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8_bf16_conservative_plan_b/latest_checkpointed_iteration.txt
```

3. **修改原训练脚本的save_interval**:
```bash
# 编辑 llama8b_presparse_training_tp8_bf16_conservative_plan_b.sh
# 修改: SAVE_INTERVAL=25 → SAVE_INTERVAL=50
```

4. **运行原训练脚本**:
```bash
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_bf16_conservative_plan_b.sh
```

---

## 📊 空间需求估算

### 当前配置 (save_interval=50)

| 项目 | 大小 | 数量 | 总计 |
|------|------|------|------|
| 每个checkpoint | ~53GB | ~40个 | ~2.1TB |
| 训练日志 | ~100MB | 1个 | ~100MB |
| **总需求** | | | **~2.2TB** |

### 原配置 (save_interval=25)

| 项目 | 大小 | 数量 | 总计 |
|------|------|------|------|
| 每个checkpoint | ~53GB | ~80个 | ~4.2TB |
| **总需求** | | | **~4.3TB** |

**节省**: ~2.1TB (50%)

---

## ✅ 验证清单

恢复训练前:
- [ ] 删除不完整的 iter 100 checkpoint
- [ ] 确认 iter 75 checkpoint 完整 (53GB)
- [ ] latest_checkpointed_iteration.txt = 75
- [ ] 磁盘可用空间 > 100GB (建议 > 500GB)

恢复训练后 (iter 76-100):
- [ ] LM loss 正常 (~5.47)
- [ ] Reg loss 正常 (~5600)
- [ ] 无 loading 错误
- [ ] Grad norm 健康 (0.03-0.06)

下次checkpoint (iter 125):
- [ ] 成功保存
- [ ] Checkpoint 大小正常 (~53GB)
- [ ] 磁盘空间充足

---

## 🎓 经验教训

1. **提前监控磁盘空间**: 98%使用率已经非常危险
2. **save_interval过小**: 原25次迭代太频繁, 导致checkpoint过多
3. **使用--no-save-optim**: 对于inference不需要optimizer状态
4. **定期清理旧checkpoint**: 只保留最新2-3个 + 关键里程碑

---

**创建时间**: 2025-10-10  
**状态**: ✅ Ready to Resume

---

**推荐行动**: 运行 `resume_bf16_conservative_from_iter75.sh` 恢复训练! 🚀
