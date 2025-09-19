import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "output/checkpoints/llama8b_maskllm_training_tp8/iter_0000001/mp_rank_00/model_optim_rng.pt"

print(f"检查 checkpoint 结构: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n顶层键:")
for key in ckpt.keys():
    print(f"  - {key}")

# 检查是否有args
if 'args' in ckpt:
    print(f"\n✅ 包含 'args' 字段")
    print(f"args 类型: {type(ckpt['args'])}")
else:
    print(f"\n❌ 缺少 'args' 字段！")
    print("这是TP转换失败的原因")

# 检查model结构
if 'model' in ckpt:
    print(f"\n模型结构:")
    model_keys = list(ckpt['model'].keys())
    for key in model_keys:
        print(f"  - {key}")
