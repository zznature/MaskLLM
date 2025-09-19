#!/usr/bin/env python3
"""
分析Llama8b-MaskLLM FP16模型的稀疏率分布
显示MaskLLM训练后的权重稀疏模式
纯文本输出，不依赖可视化库
"""

import torch
import numpy as np
import argparse
import os
from collections import defaultdict

def calculate_sparsity(tensor, threshold=1e-6):
    """
    计算张量的稀疏率
    
    Args:
        tensor: 输入张量
        threshold: 认为是0的阈值
    
    Returns:
        稀疏率 (0-1之间)
    """
    if not isinstance(tensor, torch.Tensor):
        return 0.0
    
    # 计算接近0的元素数量
    near_zero = (tensor.abs() < threshold).sum().item()
    total = tensor.numel()
    
    return near_zero / total if total > 0 else 0.0

def analyze_layer_sparsity(layer_weights, layer_name, top_k=5, target_dtype=None):
    """
    分析单层的稀疏分布
    
    Args:
        layer_weights: 层的权重字典
        layer_name: 层名称
        top_k: 显示前k个最稀疏的权重
        target_dtype: 目标数据类型 (None表示分析所有类型)
    
    Returns:
        层的稀疏统计信息
    """
    layer_stats = {
        'name': layer_name,
        'total_params': 0,
        'sparse_params': 0,
        'weight_sparsities': []
    }
    
    print(f"\n=== {layer_name} ===")
    print(f"  发现 {len(layer_weights)} 个权重项")
    
    for weight_name, weight_tensor in layer_weights.items():
        if isinstance(weight_tensor, torch.Tensor):
            # 检查数据类型
            dtype_match = target_dtype is None or weight_tensor.dtype == target_dtype
            
            if dtype_match:
                sparsity = calculate_sparsity(weight_tensor)
                num_params = weight_tensor.numel()
                sparse_params = int(sparsity * num_params)
                
                layer_stats['total_params'] += num_params
                layer_stats['sparse_params'] += sparse_params
                layer_stats['weight_sparsities'].append({
                    'name': weight_name,
                    'sparsity': sparsity,
                    'shape': tuple(weight_tensor.shape),
                    'params': num_params,
                    'sparse_count': sparse_params,
                    'dtype': str(weight_tensor.dtype)
                })
                
                print(f"  {weight_name}:")
                print(f"    形状: {weight_tensor.shape}")
                print(f"    数据类型: {weight_tensor.dtype}")
                print(f"    稀疏率: {sparsity:.4f} ({sparse_params:,}/{num_params:,})")
                print(f"    数值范围: [{weight_tensor.min():.6f}, {weight_tensor.max():.6f}]")
            else:
                print(f"  {weight_name}: 跳过 (数据类型: {weight_tensor.dtype})")
        else:
            print(f"  {weight_name}: 跳过 (不是张量: {type(weight_tensor)})")
            
    # 按稀疏率排序
    layer_stats['weight_sparsities'].sort(key=lambda x: x['sparsity'], reverse=True)
    
    # 显示最稀疏的权重
    if layer_stats['weight_sparsities']:
        print(f"\n  最稀疏的{min(top_k, len(layer_stats['weight_sparsities']))}个权重:")
        for i, weight_info in enumerate(layer_stats['weight_sparsities'][:top_k]):
            print(f"    {i+1}. {weight_info['name']}: {weight_info['sparsity']:.4f} ({weight_info['dtype']})")
    else:
        print(f"\n  ⚠️  没有找到匹配的权重张量!")
    
    overall_sparsity = layer_stats['sparse_params'] / layer_stats['total_params'] if layer_stats['total_params'] > 0 else 0
    layer_stats['overall_sparsity'] = overall_sparsity
    print(f"\n  层整体稀疏率: {overall_sparsity:.4f} ({layer_stats['total_params']:,} 参数)")
    
    return layer_stats

def visualize_weight_sparsity(weight_tensor, weight_name, save_path=None):
    """
    可视化权重的稀疏分布模式
    
    Args:
        weight_tensor: 权重张量
        weight_name: 权重名称
        save_path: 保存路径
    """
    if weight_tensor.dim() != 2:
        print(f"跳过可视化 {weight_name} (维度不是2D: {weight_tensor.shape})")
        return
    
    # 转换为numpy并取样本(如果太大的话)
    weight_np = weight_tensor.cpu().numpy().astype(np.float32)
    
    # 如果矩阵太大，取样本
    max_size = 1000
    if weight_np.shape[0] > max_size or weight_np.shape[1] > max_size:
        row_indices = np.linspace(0, weight_np.shape[0]-1, min(max_size, weight_np.shape[0]), dtype=int)
        col_indices = np.linspace(0, weight_np.shape[1]-1, min(max_size, weight_np.shape[1]), dtype=int)
        weight_sample = weight_np[np.ix_(row_indices, col_indices)]
        print(f"权重矩阵太大，显示样本 {weight_sample.shape} (原始: {weight_np.shape})")
    else:
        weight_sample = weight_np
        print(f"权重矩阵形状: {weight_sample.shape}")
    
    # 创建二值化稀疏图 (0表示接近0的值，1表示非零值)
    threshold = 1e-6
    sparse_mask = (np.abs(weight_sample) < threshold).astype(int)
    
    print(f"权重 {weight_name}:")
    print(f"  样本稀疏率: {sparse_mask.mean():.4f}")
    
    non_zero_values = weight_sample[weight_sample != 0]
    if len(non_zero_values) > 0:
        print(f"  非零值范围: [{non_zero_values.min():.6f}, {non_zero_values.max():.6f}]")
    else:
        print("  所有值都接近零")
    
    print(f"  稀疏模式预览 (前10x10, 0=非零, 1=接近零):")
    
    # 打印前10x10的稀疏模式
    preview_size = min(10, sparse_mask.shape[0], sparse_mask.shape[1])
    preview = sparse_mask[:preview_size, :preview_size]
    for row in preview:
        print("    " + "".join(str(x) for x in row))

def print_weight_samples(weight_tensor, weight_name, num_samples=20):
    """
    打印权重的具体数值样本，显示稀疏分布
    """
    print(f"\n{weight_name} 权重样本:")
    
    # 扁平化并取样本
    flat_weights = weight_tensor.flatten()
    
    # 随机采样
    if len(flat_weights) > num_samples:
        indices = torch.randperm(len(flat_weights))[:num_samples]
        samples = flat_weights[indices]
    else:
        samples = flat_weights
    
    print("  随机样本值:")
    for i, val in enumerate(samples):
        print(f"    [{i:2d}] {val:.8f}")
    
    # 统计非零值
    threshold = 1e-6
    non_zero = flat_weights[flat_weights.abs() > threshold]
    near_zero = flat_weights[flat_weights.abs() <= threshold]
    
    print(f"  统计信息:")
    print(f"    总数: {len(flat_weights):,}")
    print(f"    非零值数: {len(non_zero):,} ({len(non_zero)/len(flat_weights):.4f})")
    print(f"    接近零值数: {len(near_zero):,} ({len(near_zero)/len(flat_weights):.4f})")
    
    if len(non_zero) > 0:
        print(f"    非零值范围: [{non_zero.min():.8f}, {non_zero.max():.8f}]")
        print(f"    非零值均值: {non_zero.mean():.8f}")
        print(f"    非零值标准差: {non_zero.std():.8f}")

def analyze_single_model_sparsity(checkpoint_path, model_name="", detailed_layers=5, visualize=False, target_dtype=None):
    """
    分析单个模型的稀疏率
    
    Args:
        checkpoint_path: 检查点文件路径
        model_name: 模型名称 (用于显示)
        detailed_layers: 详细分析的层数
        visualize: 是否可视化权重分布
        target_dtype: 目标数据类型
    """
    
    print(f"\n{'=' * 80}")
    print(f"分析模型: {model_name} - {checkpoint_path}")
    print(f"{'=' * 80}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return None
    
    # 加载检查点
    print(f"加载检查点: {checkpoint_path}")
    file_size = os.path.getsize(checkpoint_path) / (1024**3)
    print(f"文件大小: {file_size:.2f} GB")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return None
    
    if 'model' not in checkpoint:
        print("❌ 检查点中未找到model键")
        print(f"检查点顶层键: {list(checkpoint.keys())}")
        return None
    
    model_state = checkpoint['model']
    print(f"模型状态顶层键: {list(model_state.keys()) if isinstance(model_state, dict) else type(model_state)}")
    
    # 确定目标数据类型
    if target_dtype is None:
        # 自动检测主要数据类型
        sample_tensor = None
        if isinstance(model_state, dict):
            for key, value in model_state.items():
                if isinstance(value, torch.Tensor):
                    sample_tensor = value
                    break
                elif isinstance(value, dict):
                    for k2, v2 in value.items():
                        if isinstance(v2, torch.Tensor):
                            sample_tensor = v2
                            break
                    if sample_tensor is not None:
                        break
        
        if sample_tensor is not None:
            target_dtype = sample_tensor.dtype
            print(f"自动检测到数据类型: {target_dtype}")
        else:
            target_dtype = torch.float16  # 默认
            print(f"未找到张量，使用默认数据类型: {target_dtype}")
    else:
        print(f"使用指定数据类型: {target_dtype}")
    
    return analyze_model_weights(model_state, model_name, detailed_layers, visualize, target_dtype)

def analyze_model_weights(model_state, model_name, detailed_layers=5, visualize=False, target_dtype=torch.float16):
    """
    分析模型权重的稀疏率
    """
    
    # 全局统计
    total_params = 0
    total_sparse_params = 0
    layer_stats = []
    
    print(f"\n{'='*60}")
    print(f"{model_name} 稀疏率统计 (数据类型: {target_dtype})")
    print(f"{'='*60}")
    
    # 分析语言模型部分
    if 'language_model' in model_state:
        language_model = model_state['language_model']
        print(f"语言模型部分键: {list(language_model.keys()) if isinstance(language_model, dict) else type(language_model)}")
        
        # 分析embedding层
        if 'embedding' in language_model:
            print(f"\n🔍 分析Embedding层...")
            embedding_stats = analyze_layer_sparsity(
                language_model['embedding'], 
                "Embedding层",
                target_dtype=target_dtype
            )
            layer_stats.append(embedding_stats)
            total_params += embedding_stats['total_params']
            total_sparse_params += embedding_stats['sparse_params']
        
        # 分析encoder层
        if 'encoder' in language_model and 'layers' in language_model['encoder']:
            layers = language_model['encoder']['layers']
            layer_count = len(layers)
            print(f"\n🔍 发现 {layer_count} 个Transformer层")
            
            # 详细分析前几层
            for i in range(min(detailed_layers, layer_count)):
                layer_name = f"Transformer层 {i}"
                print(f"\n🔍 分析{layer_name}...")
                
                layer_weights = {}
                if str(i) in layers:
                    layer_data = layers[str(i)]
                    
                    # 收集层内所有权重
                    def collect_weights(obj, prefix=""):
                        nonlocal layer_weights
                        if isinstance(obj, torch.Tensor):
                            layer_weights[prefix] = obj
                        elif isinstance(obj, dict):
                            for k, v in obj.items():
                                if k != '_extra_state':  # 跳过额外状态
                                    new_prefix = f"{prefix}.{k}" if prefix else k
                                    collect_weights(v, new_prefix)
                    
                    collect_weights(layer_data)
                    
                    if layer_weights:
                        layer_stat = analyze_layer_sparsity(layer_weights, layer_name, target_dtype=target_dtype)
                        layer_stats.append(layer_stat)
                        total_params += layer_stat['total_params']
                        total_sparse_params += layer_stat['sparse_params']
                        
                        # 可视化第一层的一些权重
                        if i == 0 and visualize:
                            print(f"\n📊 可视化{layer_name}权重分布...")
                            weight_count = 0
                            for weight_name, weight_tensor in layer_weights.items():
                                if (isinstance(weight_tensor, torch.Tensor) and 
                                    weight_tensor.dtype == target_dtype and
                                    weight_count < 3):  # 只可视化前3个权重
                                    
                                    print(f"\n🎯 {weight_name} 详细分析:")
                                    visualize_weight_sparsity(weight_tensor, weight_name)
                                    print_weight_samples(weight_tensor, weight_name)
                                    weight_count += 1
            
            # 快速统计剩余层
            if layer_count > detailed_layers:
                print(f"\n🔍 快速统计剩余 {layer_count - detailed_layers} 层...")
                for i in range(detailed_layers, layer_count):
                    if str(i) in layers:
                        layer_data = layers[str(i)]
                        layer_weights = {}
                        
                        def collect_weights(obj, prefix=""):
                            nonlocal layer_weights
                            if isinstance(obj, torch.Tensor):
                                layer_weights[prefix] = obj
                            elif isinstance(obj, dict):
                                for k, v in obj.items():
                                    if k != '_extra_state':
                                        new_prefix = f"{prefix}.{k}" if prefix else k
                                        collect_weights(v, new_prefix)
                        
                        collect_weights(layer_data)
                        
                        if layer_weights:
                            # 只统计目标数据类型的张量
                            target_weights = {k: v for k, v in layer_weights.items() 
                                            if isinstance(v, torch.Tensor) and v.dtype == target_dtype}
                            
                            if target_weights:
                                layer_total = sum(w.numel() for w in target_weights.values())
                                layer_sparse = sum(int(calculate_sparsity(w) * w.numel()) for w in target_weights.values())
                                layer_sparsity = layer_sparse / layer_total if layer_total > 0 else 0
                                
                                total_params += layer_total
                                total_sparse_params += layer_sparse
                                
                                print(f"  第{i}层: {layer_sparsity:.4f} 稀疏率 ({layer_sparse:,}/{layer_total:,})")
            
            # final_norm层
            if 'final_norm' in language_model['encoder']:
                final_norm_stats = analyze_layer_sparsity(
                    language_model['encoder']['final_norm'],
                    "Final Norm层",
                    target_dtype=target_dtype
                )
                layer_stats.append(final_norm_stats)
                total_params += final_norm_stats['total_params']
                total_sparse_params += final_norm_stats['sparse_params']
    
    # 分析输出层
    if 'output_layer' in model_state:
        print(f"\n🔍 分析输出层...")
        output_stats = analyze_layer_sparsity(
            model_state['output_layer'], 
            "输出层",
            target_dtype=target_dtype
        )
        layer_stats.append(output_stats)
        total_params += output_stats['total_params']
        total_sparse_params += output_stats['sparse_params']
    
    # 最终统计
    overall_sparsity = total_sparse_params / total_params if total_params > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"🎯 {model_name} 整体稀疏率分析结果")
    print(f"{'='*80}")
    print(f"数据类型: {target_dtype}")
    print(f"总参数数: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"稀疏参数数: {total_sparse_params:,} ({total_sparse_params/1e9:.2f}B)")
    print(f"非零参数数: {total_params - total_sparse_params:,} ({(total_params - total_sparse_params)/1e9:.2f}B)")
    print(f"整体稀疏率: {overall_sparsity:.4f} ({overall_sparsity*100:.2f}%)")
    
    if overall_sparsity < 1.0:
        compression_ratio = 1/(1-overall_sparsity)
        print(f"模型压缩比: {compression_ratio:.2f}x (理论)")
    
    # 稀疏率分布统计
    if layer_stats:
        sparsities = [stat['overall_sparsity'] for stat in layer_stats if 'overall_sparsity' in stat]
        if sparsities:
            print(f"\n📊 各层稀疏率统计:")
            print(f"  最小稀疏率: {min(sparsities):.4f}")
            print(f"  最大稀疏率: {max(sparsities):.4f}")
            print(f"  平均稀疏率: {np.mean(sparsities):.4f}")
            print(f"  稀疏率标准差: {np.std(sparsities):.4f}")
            
            # 显示各层稀疏率
            print(f"\n📋 各层详细稀疏率:")
            for stat in layer_stats:
                if 'overall_sparsity' in stat:
                    print(f"  {stat['name']}: {stat['overall_sparsity']:.4f}")
    
    return {
        'model_name': model_name,
        'target_dtype': str(target_dtype),
        'total_params': total_params,
        'total_sparse_params': total_sparse_params,
        'overall_sparsity': overall_sparsity,
        'layer_stats': layer_stats
    }

def analyze_model_sparsity(fp32_path, fp16_path, detailed_layers=5, visualize=False):
    """
    对比分析FP32和FP16模型的稀疏率
    """
    
    print("=" * 80)
    print("Llama8b-MaskLLM 稀疏率对比分析")
    print("=" * 80)
    
    results = {}
    
    # 分析FP32模型
    if fp32_path and os.path.exists(fp32_path):
        print("\n🔍 分析FP32模型...")
        fp32_result = analyze_single_model_sparsity(
            fp32_path, 
            "FP32模型", 
            detailed_layers, 
            visualize, 
            torch.float32
        )
        if fp32_result:
            results['fp32'] = fp32_result
    else:
        print(f"⚠️ FP32模型文件不存在或未提供: {fp32_path}")
    
    # 分析FP16模型
    if fp16_path and os.path.exists(fp16_path):
        print("\n🔍 分析FP16模型...")
        fp16_result = analyze_single_model_sparsity(
            fp16_path, 
            "FP16模型", 
            detailed_layers, 
            visualize, 
            torch.float16
        )
        if fp16_result:
            results['fp16'] = fp16_result
    else:
        print(f"⚠️ FP16模型文件不存在或未提供: {fp16_path}")
    
    # 对比分析
    if 'fp32' in results and 'fp16' in results:
        print(f"\n{'='*80}")
        print("🎯 FP32 vs FP16 对比分析")
        print(f"{'='*80}")
        
        fp32_res = results['fp32']
        fp16_res = results['fp16']
        
        print(f"FP32模型稀疏率: {fp32_res['overall_sparsity']:.4f} ({fp32_res['overall_sparsity']*100:.2f}%)")
        print(f"FP16模型稀疏率: {fp16_res['overall_sparsity']:.4f} ({fp16_res['overall_sparsity']*100:.2f}%)")
        
        sparsity_diff = abs(fp32_res['overall_sparsity'] - fp16_res['overall_sparsity'])
        print(f"稀疏率差异: {sparsity_diff:.4f} ({sparsity_diff*100:.2f}%)")
        
        if sparsity_diff < 0.001:
            print("✅ FP32和FP16模型的稀疏率基本一致，转换正确")
        elif sparsity_diff < 0.01:
            print("⚠️ FP32和FP16模型稀疏率有轻微差异，可能是精度转换导致")
        else:
            print("❌ FP32和FP16模型稀疏率差异较大，可能存在问题")
    
    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="分析Llama8b-MaskLLM模型的稀疏率分布",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 同时分析FP32和FP16模型 (推荐)
  python3 analyze_sparsity.py --fp32 model.pt --fp16 model_fp16.pt
  
  # 只分析FP16模型
  python3 analyze_sparsity.py --fp16 model_fp16.pt
  
  # 只分析FP32模型  
  python3 analyze_sparsity.py --fp32 model.pt
        """
    )
    
    # 默认路径
    default_fp32 = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng.pt"
    default_fp16 = "/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt"
    
    parser.add_argument('--fp32', 
                       default=default_fp32,
                       help='FP32模型文件路径')
    
    parser.add_argument('--fp16',
                       default=default_fp16,
                       help='FP16模型文件路径')
    
    parser.add_argument('--detailed-layers', type=int, default=3,
                       help='详细分析的层数 (默认: 3)')
    
    parser.add_argument('--visualize', action='store_true',
                       help='启用权重分布可视化 (需要matplotlib)')
    
    parser.add_argument('--only-fp32', action='store_true',
                       help='只分析FP32模型')
    
    parser.add_argument('--only-fp16', action='store_true',
                       help='只分析FP16模型')
    
    args = parser.parse_args()
    
    # 确定要分析的文件
    fp32_path = args.fp32 if not args.only_fp16 else None
    fp16_path = args.fp16 if not args.only_fp32 else None
    
    try:
        results = analyze_model_sparsity(
            fp32_path=fp32_path,
            fp16_path=fp16_path,
            detailed_layers=args.detailed_layers,
            visualize=args.visualize
        )
        
        print(f"\n✅ 稀疏率分析完成!")
        
        # 显示结果总结
        for model_type, result in results.items():
            print(f"{result['model_name']}稀疏率: {result['overall_sparsity']:.4f} ({result['overall_sparsity']*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
