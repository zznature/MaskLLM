#!/usr/bin/env python3
"""
Megatron检查点结构分析脚本
用于在Megatron→HuggingFace转换前确认所有关键参数
"""

import os
import sys
import torch
import json
import argparse
from collections import defaultdict

def analyze_megatron_checkpoint(checkpoint_path):
    """分析Megatron检查点的结构和关键参数"""
    
    print("🔍 Megatron检查点结构分析")
    print("=" * 60)
    print(f"📁 检查点路径: {checkpoint_path}")
    
    # 检查文件存在性
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return False
    
    # 检查文件大小
    file_size = os.path.getsize(checkpoint_path)
    print(f"📊 文件大小: {file_size / (1024**3):.2f} GB")
    
    try:
        print("\n🔄 加载检查点...")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print("✅ 检查点加载成功")
        
    except Exception as e:
        print(f"❌ 检查点加载失败: {str(e)}")
        return False
    
    print(f"\n📋 顶层键名:")
    for key in ckpt.keys():
        if isinstance(ckpt[key], dict):
            print(f"  {key}/ (字典)")
        elif isinstance(ckpt[key], torch.Tensor):
            print(f"  {key}: {ckpt[key].shape} ({ckpt[key].dtype})")
        else:
            print(f"  {key}: {type(ckpt[key])}")
    
    # 检查模型权重结构
    if 'model' not in ckpt:
        print("❌ 未找到 'model' 键")
        return False
    
    model_weights = ckpt['model']
    print(f"\n🏗️  模型权重结构分析:")
    
    # 深度探索模型结构
    def explore_structure(d, prefix="", max_depth=4, current_depth=0):
        """递归探索字典结构"""
        if current_depth >= max_depth:
            return
        
        for k, v in d.items():
            current_prefix = f"{prefix}.{k}" if prefix else k
            
            if isinstance(v, dict):
                print(f"{'  ' * current_depth}{k}/")
                explore_structure(v, current_prefix, max_depth, current_depth + 1)
            elif isinstance(v, torch.Tensor):
                print(f"{'  ' * current_depth}{k}: {v.shape} ({v.dtype})")
            else:
                print(f"{'  ' * current_depth}{k}: {type(v)}")
    
    explore_structure(model_weights, max_depth=4)
    
    # 关键参数确认
    print("\n" + "=" * 60)
    print("🎯 关键参数确认")
    print("=" * 60)
    
    # 1. 确认基本结构
    success_checks = []
    
    try:
        # 检查词嵌入层
        if 'language_model' in model_weights:
            lm = model_weights['language_model']
            
            if 'embedding' in lm and 'word_embeddings' in lm['embedding']:
                embed_weight = lm['embedding']['word_embeddings']['weight']
                print(f"✅ 词嵌入层: {embed_weight.shape} ({embed_weight.dtype})")
                vocab_size, hidden_size = embed_weight.shape
                print(f"   词汇表大小: {vocab_size}")
                print(f"   隐藏维度: {hidden_size}")
                success_checks.append("embedding")
            else:
                print("❌ 未找到词嵌入层")
            
            # 检查编码器层
            if 'encoder' in lm and 'layers' in lm['encoder']:
                layers = lm['encoder']['layers']
                num_layers = len(layers)
                print(f"✅ 编码器层数: {num_layers}")
                
                # 检查第一层的详细结构
                if '0' in layers:
                    layer0 = layers['0']
                    print(f"\n📋 第0层详细结构:")
                    
                    # 检查注意力机制
                    if 'self_attention' in layer0:
                        attn = layer0['self_attention']
                        
                        if 'query_key_value' in attn:
                            qkv_weight = attn['query_key_value']['weight']
                            print(f"  🔍 QKV权重: {qkv_weight.shape} ({qkv_weight.dtype})")
                            
                            # 分析QKV维度
                            qkv_out_dim, qkv_in_dim = qkv_weight.shape
                            expected_qkv_dim = 3 * hidden_size  # Q + K + V
                            print(f"     期望QKV输出维度: {expected_qkv_dim}")
                            print(f"     实际QKV输出维度: {qkv_out_dim}")
                            print(f"     维度匹配: {'✅' if qkv_out_dim == expected_qkv_dim else '❌'}")
                            success_checks.append("qkv_structure")
                        
                        if 'dense' in attn:
                            dense_weight = attn['dense']['weight']
                            print(f"  🔍 注意力输出: {dense_weight.shape} ({dense_weight.dtype})")
                    
                    # 检查MLP层 (关键！)
                    if 'mlp' in layer0:
                        mlp = layer0['mlp']
                        
                        if 'dense_h_to_4h' in mlp:
                            h_to_4h = mlp['dense_h_to_4h']['weight']
                            print(f"  🎯 MLP输入→中间: {h_to_4h.shape} ({h_to_4h.dtype})")
                            
                            intermediate_size, input_size = h_to_4h.shape
                            print(f"     MLP中间维度: {intermediate_size}")
                            print(f"     MLP输入维度: {input_size}")
                            
                            # 关键判断：是14336还是28672？
                            if intermediate_size == 14336:
                                print(f"     ✅ 匹配HF标准 (14336)")
                                mlp_mode = "hf_standard"
                            elif intermediate_size == 28672:
                                print(f"     ⚠️  需要分离门控 (28672 -> 2×14336)")
                                mlp_mode = "split_gating"
                            else:
                                print(f"     ❓ 未知MLP维度: {intermediate_size}")
                                mlp_mode = "unknown"
                            
                            success_checks.append(f"mlp_{mlp_mode}")
                        
                        if 'dense_4h_to_h' in mlp:
                            h_to_h = mlp['dense_4h_to_h']['weight']
                            print(f"  🔍 MLP中间→输出: {h_to_h.shape} ({h_to_h.dtype})")
                    
                    # 检查层归一化
                    if 'input_norm' in layer0:
                        input_norm = layer0['input_norm']['weight']
                        print(f"  🔍 输入归一化: {input_norm.shape} ({input_norm.dtype})")
                    
                    if 'post_attention_norm' in layer0:
                        post_norm = layer0['post_attention_norm']['weight']
                        print(f"  🔍 注意力后归一化: {post_norm.shape} ({post_norm.dtype})")
                
                success_checks.append("layers_structure")
            
            # 检查最终归一化
            if 'final_norm' in lm['encoder']:
                final_norm = lm['encoder']['final_norm']['weight']
                print(f"✅ 最终归一化: {final_norm.shape} ({final_norm.dtype})")
                success_checks.append("final_norm")
        
        # 检查输出层
        if 'output_layer' in model_weights:
            output_weight = model_weights['output_layer']['weight']
            print(f"✅ 输出层: {output_weight.shape} ({output_weight.dtype})")
            success_checks.append("output_layer")
        else:
            print("❓ 未找到'output_layer'，检查其他可能的键名...")
            for key in model_weights.keys():
                if 'output' in key.lower() or 'lm_head' in key.lower():
                    weight = model_weights[key]['weight'] if isinstance(model_weights[key], dict) else model_weights[key]
                    if isinstance(weight, torch.Tensor):
                        print(f"   可能的输出层: {key} -> {weight.shape}")
        
    except Exception as e:
        print(f"❌ 结构分析出错: {str(e)}")
        return False
    
    # 统计权重数量和数据类型
    print(f"\n📊 权重统计:")
    
    def count_weights_and_dtypes(d, prefix=""):
        """递归统计权重数量和数据类型"""
        weight_count = 0
        dtype_stats = defaultdict(int)
        
        for k, v in d.items():
            current_path = f"{prefix}.{k}" if prefix else k
            
            if isinstance(v, dict):
                sub_count, sub_dtypes = count_weights_and_dtypes(v, current_path)
                weight_count += sub_count
                for dtype, count in sub_dtypes.items():
                    dtype_stats[dtype] += count
            elif isinstance(v, torch.Tensor):
                weight_count += 1
                dtype_stats[str(v.dtype)] += 1
        
        return weight_count, dtype_stats
    
    total_weights, dtype_distribution = count_weights_and_dtypes(model_weights)
    
    print(f"  总权重数量: {total_weights}")
    print(f"  期望权重数量: 227 (32层×7 + 3个全局)")
    print(f"  数量匹配: {'✅' if total_weights >= 220 and total_weights <= 240 else '❌'}")
    
    print(f"\n  数据类型分布:")
    for dtype, count in dtype_distribution.items():
        print(f"    {dtype}: {count}个权重")
    
    # 检查是否全部FP16
    if len(dtype_distribution) == 1 and 'torch.float16' in dtype_distribution:
        print(f"  ✅ 所有权重都是FP16")
        success_checks.append("all_fp16")
    else:
        print(f"  ⚠️  存在非FP16权重")
    
    # 生成确认报告
    print("\n" + "=" * 60)
    print("📋 参数确认报告")
    print("=" * 60)
    
    confirmation_report = {
        "file_path": checkpoint_path,
        "file_size_gb": file_size / (1024**3),
        "success_checks": success_checks,
        "total_weights": total_weights,
        "dtype_distribution": dict(dtype_distribution),
        "structure_confirmed": len(success_checks) >= 5,
    }
    
    # 检查清单
    checklist = [
        ("embedding", "词嵌入层结构"),
        ("qkv_structure", "QKV权重结构"),
        ("mlp_hf_standard", "MLP维度(14336)") or ("mlp_split_gating", "MLP维度(28672)"),
        ("layers_structure", "编码器层结构"),
        ("final_norm", "最终归一化层"),
        ("output_layer", "输出层"),
        ("all_fp16", "FP16数据类型")
    ]
    
    print("确认清单:")
    for check_key, description in checklist:
        if isinstance(check_key, tuple):  # 处理或逻辑
            status = any(key in success_checks for key in check_key)
        else:
            status = check_key in success_checks
        
        print(f"  {'✅' if status else '❌'} {description}")
    
    # 给出转换建议
    print(f"\n💡 转换建议:")
    
    if confirmation_report["structure_confirmed"]:
        print("✅ 结构确认完成，可以开始Megatron→HF转换")
        
        # 具体的转换参数建议
        if "mlp_hf_standard" in success_checks:
            print("   MLP处理: 直接映射，无需分离")
        elif "mlp_split_gating" in success_checks:
            print("   MLP处理: 需要将28672维度分离为2×14336门控机制")
        
        print("   建议转换命令:")
        print(f"   bash run_maskllm_native.sh llama8b_scripts/convert_megatron_to_hf.py \\")
        print(f"       --checkpoint {checkpoint_path} \\")
        print(f"       --output output/checkpoints/llama8b_hf_maskllm_c4_from_megatron")
        
    else:
        print("❌ 结构确认不完整，不建议进行转换")
        print("   建议先解决以下问题:")
        missing_checks = set(["embedding", "qkv_structure", "layers_structure", "final_norm", "output_layer"]) - set(success_checks)
        for missing in missing_checks:
            print(f"   - 缺少: {missing}")
    
    # 保存报告
    report_file = "/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/megatron_analysis_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(confirmation_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 分析报告已保存: {report_file}")
    
    return confirmation_report["structure_confirmed"]

def main():
    parser = argparse.ArgumentParser(description="Analyze Megatron checkpoint structure")
    parser.add_argument("--checkpoint", "-c",
                       default="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt",
                       help="Path to Megatron checkpoint file")
    
    args = parser.parse_args()
    
    success = analyze_megatron_checkpoint(args.checkpoint)
    
    if success:
        print("\n🎯 分析完成：结构确认成功，可以进行转换！")
        sys.exit(0)
    else:
        print("\n❌ 分析失败或结构确认不完整")
        sys.exit(1)

if __name__ == "__main__":
    main()
