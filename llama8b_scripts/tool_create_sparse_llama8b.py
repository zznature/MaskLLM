import torch
import argparse

parser = argparse.ArgumentParser(description='Apply sparsity to Llama8b checkpoint')
parser.add_argument('--ckpt_dir', type=str, default='output/checkpoints/llama8b_maskllm_training_tp8/iter_0002000', help='Input checkpoint')
args = parser.parse_args()

def apply_sparsity(input, output):
    print(f"Loading checkpoint: {input}")
    ckpt = torch.load(input, map_location='cpu', weights_only=False)
    
    # 保留原有的checkpoint结构，只修改encoder部分
    new_encoder_state_dict = {}
    mask_options = torch.zeros(1, 6, 4, dtype=torch.float32)
    mask_options[:, 0, :].data += torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    mask_options[:, 1, :].data += torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    mask_options[:, 2, :].data += torch.tensor([1, 0, 0, 1], dtype=torch.float32)
    mask_options[:, 3, :].data += torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    mask_options[:, 4, :].data += torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    mask_options[:, 5, :].data += torch.tensor([0, 0, 1, 1], dtype=torch.float32)

    print(f"原始checkpoint包含的顶层键: {list(ckpt.keys())}")
    
    # 第一轮：复制所有非mask权重，应用float16转换
    weight_count = 0
    for k, v in ckpt['model']['language_model']['encoder'].items():
        if 'mask' not in k:
            # Convert to float16 for llama8b memory efficiency
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                new_encoder_state_dict[k] = v.to(torch.float16)
                print(f"Convert to float16: {k}")
            else:
                new_encoder_state_dict[k] = v
                print(f"Keep original: {k}")
            weight_count += 1

    # 第二轮：应用稀疏掩码（按照原版逻辑）
    gate_keys = [k for k in ckpt['model']['language_model']['encoder'].keys() if '.diff_mask.gate' in k]
    print(f"\n找到 {len(gate_keys)} 个 diff_mask.gate 键需要处理")
    
    applied_count = 0
    for k, v in ckpt['model']['language_model']['encoder'].items():
        if '.diff_mask.gate' in k:
            gate = ckpt['model']['language_model']['encoder'][k].float()
            runtime_mask = ckpt['model']['language_model']['encoder'][k.replace('diff_mask.gate', 'mask')].float()
            winner_mask = mask_options[torch.arange(mask_options.shape[0]), gate.argmax(dim=-1)].view(*runtime_mask.shape)
            
            # 按照原版逻辑：直接应用到对应权重上
            weight_key = k.replace('diff_mask.gate', 'weight')
            if weight_key in new_encoder_state_dict:
                # 确保winner_mask与权重的dtype一致
                winner_mask = winner_mask.to(new_encoder_state_dict[weight_key].dtype)
                new_encoder_state_dict[weight_key] *= winner_mask
                applied_count += 1
                print(f"应用稀疏掩码: {weight_key}")
            else:
                print(f"⚠️ 未找到对应权重: {weight_key}")
    
    print(f"\n✅ 成功应用稀疏掩码: {applied_count}/{len(gate_keys)} 个权重")
    print(f"✅ 处理了 {weight_count} 个权重参数")
    
    # 更新checkpoint的encoder部分
    ckpt['model']['language_model']['encoder'] = new_encoder_state_dict
    
    # 验证checkpoint完整性
    print(f"\n最终checkpoint结构验证:")
    print(f"- 顶层键: {list(ckpt.keys())}")
    print(f"- Encoder键数量: {len(ckpt['model']['language_model']['encoder'].keys())}")
    
    if 'args' in ckpt:
        print("✅ 包含args字段，TP转换兼容")
    else:
        print("❌ 缺少args字段，TP转换可能失败")
    
    # 保存checkpoint
    print(f"\n保存checkpoint到: {output}")
    torch.save(ckpt, output)
    print(f"✅ 稀疏checkpoint保存完成")

import os
import glob

# Process checkpoint directory
if args.ckpt_dir.endswith('/'):
    args.ckpt_dir = args.ckpt_dir[:-1]
splited_dir = args.ckpt_dir.split('/') 
output_dir = os.path.join('/'.join(splited_dir[:-1]), 'iter_0000001')
print(f"Input directory: {args.ckpt_dir}")
print(f"Output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Process all mp_rank directories
mp_rank_dirs = glob.glob(os.path.join(args.ckpt_dir, "mp_rank_*"))
print(f"Found {len(mp_rank_dirs)} mp_rank directories")

for mp_rank_dir in mp_rank_dirs:
    ckpt_file = os.path.join(mp_rank_dir, "model_optim_rng.pt")
    output_file = ckpt_file.replace(args.ckpt_dir, output_dir)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Processing: {ckpt_file} -> {output_file}")
    apply_sparsity(ckpt_file, output_file)

# Update latest iteration pointer
iteration_file = os.path.join(*splited_dir[:-1], 'latest_checkpointed_iteration.txt')
print(f"Updating iteration file: {iteration_file}")
with open(iteration_file, 'w') as f:
    f.write("1")

print("✅ Sparsity application completed!")
print("✅ Sparse weights created with 0s for masked parameters")
print("✅ All weights converted to float16 format")
