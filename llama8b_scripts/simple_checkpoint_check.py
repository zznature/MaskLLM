import torch
import sys
import os

# 临时修改环境，避免加载megatron
os.environ['MEGATRON_PATH'] = ''

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "output/checkpoints/llama8b_maskllm_training_tp8/iter_0002000/mp_rank_00/model_optim_rng.pt"

print(f"检查 checkpoint: {checkpoint_path}")

try:
    # 尝试直接加载，避免触发megatron依赖
    with open(checkpoint_path, 'rb') as f:
        ckpt = torch.load(f, map_location='cpu', weights_only=False)
        
    encoder_keys = list(ckpt['model']['language_model']['encoder'].keys())
    
    # 检查mask相关keys
    mask_keys = [k for k in encoder_keys if 'mask' in k]
    print(f'\n包含mask相关的keys总数: {len(mask_keys)}')
    
    # 分类不同类型的mask keys
    diff_mask_gate_keys = [k for k in encoder_keys if 'diff_mask.gate' in k]
    mask_keys_only = [k for k in encoder_keys if '.mask' in k and 'diff_mask' not in k]
    mask_options_keys = [k for k in encoder_keys if 'mask_options' in k]
    
    print(f'diff_mask.gate keys: {len(diff_mask_gate_keys)}个')
    print(f'普通 .mask keys: {len(mask_keys_only)}个') 
    print(f'mask_options keys: {len(mask_options_keys)}个')
    
    if diff_mask_gate_keys:
        print('\n前5个 diff_mask.gate keys:')
        for k in diff_mask_gate_keys[:5]:
            print(f'  {k}')
    
    if mask_keys_only:
        print('\n前5个普通 .mask keys:')
        for k in mask_keys_only[:5]:
            print(f'  {k}')
            
    # 检查checkpoint结构
    print(f'\n总的encoder keys数量: {len(encoder_keys)}')
    print(f'checkpoint 包含的主要组件:')
    print(f'  - model: {bool("model" in ckpt)}')
    if "model" in ckpt:
        print(f'  - language_model: {bool("language_model" in ckpt["model"])}')
        print(f'  - encoder keys: {len(encoder_keys)}')

except Exception as e:
    print(f'加载checkpoint失败: {e}')
