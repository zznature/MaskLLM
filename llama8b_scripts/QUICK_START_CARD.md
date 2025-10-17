# 🚀 快速启动卡片 - 从bf16 iter 600恢复

## ⚡ 一键复制命令

```bash
ssh zdhs0054@hd02-gpu1-0056
cd /data/home/zdhs0054/zzhou/MaskLLM
tmux new -s fp32_from_bf16
bash run_apptainer_extended_libs.sh
# ⬆️ 等待容器启动，看到 Apptainer> 后执行下面 ⬇️
bash run_maskllm_native.sh llama8b_scripts/llama8b_presparse_training_tp8_fp32_from_bf16.sh 0
```

---

## ⚠️ 关键注意

1. **必须在容器内** - 先 `bash run_apptainer_extended_libs.sh`
2. **等待容器提示** - 看到 `Apptainer>` 再运行脚本
3. **使用tmux** - 防止SSH断连

---

## 📊 预期

- **起点**: bf16 iter 600 (已完成)
- **目标**: fp32 iter 2000
- **剩余**: 1400次迭代
- **时间**: ~4.8天
- **成功率**: 85-90%

---

## 🔍 监控

```bash
# 另一个终端实时查看
tail -f output/logs/llama8b_presparse_training_fp32_from_bf16/training_from_bf16_*.log | grep "iteration"
```

---

## ✅ 成功信号

- ✅ 加载"iter_0000600" checkpoint
- ✅ 显示"iteration 601"开始
- ✅ 无"torchrun: command not found"
- ✅ 无NaN错误

---

**详细文档**: `START_FROM_BF16_ITER600.md`
