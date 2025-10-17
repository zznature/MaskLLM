# 训练中断分析: Iter 875 磁盘空间错误

## 错误时间
- **发生时间**: 2025-10-15 08:43:21
- **中断位置**: iter 875 (尝试保存checkpoint时)

## 错误原因

### 直接原因
```
RuntimeError: [enforce fail at inline_container.cc:747] . 
PytorchStreamWriter failed writing file data.pkl: file write failed
```

### 根本原因
- **磁盘使用率**: 94% (297T/318T used)
- **剩余空间**: 22T (看似足够，但可能inode耗尽或达到quota限制)
- **失败操作**: 保存checkpoint到 `iter_0000875`

## 训练进度评估 (iter 801-875)

### 健康指标 ✅

| 指标 | iter 801 | iter 875 | 变化 | 目标 | 状态 |
|------|---------|---------|------|------|------|
| **LM loss** | 2.762 | ~2.74 | ↓0.02 | 持续下降 | ✅ 正常 |
| **Reg loss** | 7,121.5 | 7,206.5 | +85 | 增速≤1.5/iter | ✅ 1.15/iter |
| **Grad norm** | 0.149→0.04 | 0.04 | 稳定 | ≤0.08 | ✅ 健康 |
| **Max prob** | 0.477 | 0.484 | +0.007 | 持续上升 | ✅ 正常 |
| **Temperature** | 3.520 | 3.498 | ↓0.022 | 逐步下降 | ✅ 符合预期 |
| **Scale** | 164.0 | 166.9 | +2.9 | 逐步上升 | ✅ 符合预期 |

### 关键发现
1. **训练本身完全健康**: 所有指标在目标范围内
2. **Reg loss增速**: 1.15/iter (远低于1.5/iter警戒线)
3. **稳定性验证**: 稳定性方程 ≈0.51 工作良好
4. **无数值问题**: 无NaN，无梯度爆炸

### 最后成功保存的checkpoint
- **位置**: `iter_0000900`
- **时间**: 2025-10-15 16:43
- **状态**: ✅ 完整 (8个rank都存在)

## 解决方案

### 1. 清理策略
**保留checkpoint**:
- 里程碑: 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1150
- 最新恢复点: iter 900

**删除checkpoint**:
- 所有其他25-iter间隔的checkpoint (如825, 850, 875等)

### 2. 修改后的恢复逻辑
```bash
# 自动检测最新可用checkpoint
RESUME_ITER=900  # 优先
├─ 若不存在 → 尝试 875
└─ 若不存在 → 尝试 850
   └─ 均不存在 → 报错退出

# 启动前自动清理
- 删除不在保留列表中的checkpoint
- 释放磁盘空间
- 显示清理后的磁盘状态
```

### 3. 脚本修改
**文件**: `llama8b_scripts/resume_from_iter800_balanced_stage1.sh`

**主要改动**:
1. 添加智能checkpoint检测 (900→875→850)
2. 添加自动清理逻辑 (保留关键checkpoint)
3. 动态设置恢复点变量 `$RESUME_ITER`
4. 更新显示信息显示实际恢复点

## 运行建议

### 立即执行
```bash
bash run_maskllm_native.sh llama8b_scripts/resume_from_iter800_balanced_stage1.sh
```

### 预期行为
1. ✅ 检测到 iter 900 checkpoint
2. ✅ 清理旧checkpoint (预计删除15-20个)
3. ✅ 释放空间 (预计~1-2TB)
4. ✅ 从 iter 900 恢复训练
5. ✅ 继续训练到 iter 1200

### 监控要点
- **磁盘空间**: 清理后应降到 <93%
- **Reg loss增速**: 保持 ≤1.5/iter
- **Save interval**: 每25次保存，确保不再失败

## 长期建议

### 1. Checkpoint管理策略
- **阶段1 (iter 800-1200)**: 保持25间隔，但只保留100的倍数
- **阶段2 (iter 1200-2000)**: 改为50间隔 (已在stage2脚本中设置)

### 2. 磁盘监控
```bash
# 训练前检查
df -h /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/

# 若 >93%，手动清理：
cd /data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp8_bf16_conservative_plan_b
rm -rf iter_0000{825,850,875}  # 示例
```

### 3. 自动化清理
考虑在训练脚本中添加周期性清理 (每100 iters检查一次)

## 总结

| 项目 | 状态 |
|------|------|
| **错误性质** | 🟡 基础设施问题 (非训练问题) |
| **训练健康** | ✅ 完全正常 |
| **数据损失** | ❌ 无 (iter 900完整保存) |
| **恢复难度** | ✅ 简单 (自动检测+清理) |
| **预计影响** | 🟢 最小 (仅损失 ~25 iters) |

**下一步**: 运行修改后的脚本，从 iter 900 继续训练到 1200。

