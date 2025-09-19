#!/usr/bin/env python3
"""
Llama8b Megatron FP16 到 HuggingFace 格式转换脚本
基于实际权重分析的精确转换实现

输入: model_optim_rng_fp16.pt (17GB Megatron FP16)
输出: pytorch_model.bin (17GB HuggingFace格式)
"""

import os
import sys
import torch
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

class MegatronToHFConverter:
    """Megatron权重到HuggingFace格式的转换器"""
    
    def __init__(self, megatron_checkpoint_path, output_dir, reference_model_dir):
        """
        初始化转换器
        
        Args:
            megatron_checkpoint_path (str): Megatron检查点文件路径
            output_dir (str): 输出目录路径  
            reference_model_dir (str): 参考模型目录 (用于复制tokenizer等文件)
        """
        self.megatron_checkpoint_path = megatron_checkpoint_path
        self.output_dir = output_dir
        self.reference_model_dir = reference_model_dir
        
        # 基于实际分析确认的配置参数
        self.config = {
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,  # 与原始配置一致
            "max_length": 4096,
            "max_position_embeddings": 4096,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 32,
            "pad_token_id": 0,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",  # 确保FP16
            "transformers_version": "4.40.0",  # 当前版本
            "use_cache": True,
            "vocab_size": 119696
        }
        
        print(f"🚀 初始化Megatron→HF转换器")
        print(f"📁 Megatron权重: {megatron_checkpoint_path}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📁 参考模型: {reference_model_dir}")
    
    def load_megatron_checkpoint(self):
        """加载Megatron检查点"""
        print(f"\n🔄 加载Megatron检查点...")
        
        try:
            # 加载检查点
            checkpoint = torch.load(self.megatron_checkpoint_path, map_location='cpu')
            print(f"✅ 检查点加载成功")
            
            # 提取模型权重 - 使用与simple_weight_check.py相同的方式
            if 'model' not in checkpoint:
                raise ValueError("检查点中没有找到'model'键")
                
            # 直接获取language_model部分，避免多级访问
            full_model = checkpoint['model']
            if 'language_model' not in full_model:
                raise ValueError("检查点中没有找到'language_model'键")
                
            model_weights = full_model  # 保留完整结构
            
            # 统计权重数量和大小
            weight_count = 0
            total_size = 0
            dtype_stats = {}
            
            def count_weights(d, path=""):
                nonlocal weight_count, total_size
                for k, v in d.items():
                    if isinstance(v, torch.Tensor):
                        weight_count += 1
                        total_size += v.numel() * v.element_size()
                        dtype = str(v.dtype)
                        dtype_stats[dtype] = dtype_stats.get(dtype, 0) + 1
                    elif isinstance(v, dict):
                        count_weights(v, f"{path}.{k}")
            
            count_weights(model_weights)
            
            print(f"📊 权重统计:")
            print(f"  权重数量: {weight_count}")
            print(f"  总大小: {total_size / (1024**3):.2f} GB")
            print(f"  数据类型分布: {dtype_stats}")
            
            # 验证关键结构
            print(f"\n🔍 验证检查点结构...")
            
            # 检查language_model
            if 'language_model' not in model_weights:
                raise ValueError("检查点中未找到'language_model'键")
            
            lm = model_weights['language_model']
            print(f"  language_model键: {list(lm.keys())}")
            
            # 检查encoder
            if 'encoder' not in lm:
                raise ValueError("language_model中未找到'encoder'键")
            
            encoder = lm['encoder']
            print(f"  encoder键: {list(encoder.keys())}")
            
            # 实际结构验证：encoder使用扁平化存储
            print(f"  ✅ encoder使用扁平化权重存储")
            
            # 统计层数（通过layers.X.开头的键）
            layer_keys = set()
            for key in encoder.keys():
                if key.startswith('layers.'):
                    # 提取层索引，如 'layers.0.input_norm.weight' -> '0'
                    parts = key.split('.')
                    if len(parts) >= 2:
                        layer_idx = parts[1]
                        if layer_idx.isdigit():
                            layer_keys.add(int(layer_idx))
            
            layer_count = len(layer_keys)
            max_layer = max(layer_keys) if layer_keys else -1
            print(f"  检测到层数: {layer_count} (索引0-{max_layer})")
            
            # 验证关键权重存在
            test_keys = [
                'layers.0.input_norm.weight',
                'layers.0.self_attention.query_key_value.weight',
                'final_norm.weight'
            ]
            
            missing_keys = [key for key in test_keys if key not in encoder]
            if missing_keys:
                print(f"❌ 缺少关键权重: {missing_keys}")
                raise ValueError(f"缺少关键权重: {missing_keys}")
            
            print(f"  ✅ 关键权重验证通过")
            
            print(f"✅ 结构验证通过")
            
            return model_weights
            
        except Exception as e:
            print(f"❌ 加载检查点失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def create_hf_model(self):
        """创建HuggingFace模型架构"""
        print(f"\n🏗️  创建HuggingFace模型架构...")
        
        try:
            # 创建配置
            config = LlamaConfig(**self.config)
            
            # 创建模型
            model = LlamaForCausalLM(config)
            
            # 确保是FP16
            model = model.half()
            
            # 统计模型参数
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ HF模型创建成功")
            print(f"📊 模型参数: {total_params / 1e9:.2f}B")
            print(f"💾 预计大小: {total_params * 2 / (1024**3):.2f} GB (FP16)")
            
            return model
            
        except Exception as e:
            print(f"❌ 创建HF模型失败: {str(e)}")
            raise
    
    def split_qkv_weight(self, qkv_weight):
        """
        分离合并的QKV权重
        
        Args:
            qkv_weight: [12288, 4096] 合并权重
            
        Returns:
            tuple: (q_weight, k_weight, v_weight) 每个都是 [4096, 4096]
        """
        hidden_size = 4096
        
        q_weight = qkv_weight[:hidden_size, :]          # 前4096行
        k_weight = qkv_weight[hidden_size:2*hidden_size, :]  # 中间4096行
        v_weight = qkv_weight[2*hidden_size:, :]        # 后4096行
        
        return q_weight, k_weight, v_weight
    
    def split_mlp_gate_up(self, dense_h_to_4h_weight):
        """
        分离Megatron合并的MLP权重
        
        Args:
            dense_h_to_4h_weight: [28672, 4096] Megatron合并权重
            
        Returns:
            tuple: (gate_weight, up_weight) 每个都是 [14336, 4096]
        """
        intermediate_size = 14336
        
        gate_weight = dense_h_to_4h_weight[:intermediate_size, :]     # gate_proj: 前14336行
        up_weight = dense_h_to_4h_weight[intermediate_size:, :]       # up_proj: 后14336行
        
        return gate_weight, up_weight
    
    def convert_weights(self, megatron_weights, hf_model):
        """
        执行权重转换的核心逻辑
        
        Args:
            megatron_weights: Megatron模型权重字典
            hf_model: HuggingFace模型实例
            
        Returns:
            dict: 转换后的HF格式权重字典
        """
        print(f"\n🔧 开始权重转换...")
        
        hf_state_dict = {}
        conversion_stats = {
            "converted_weights": 0,
            "skipped_weights": 0,
            "errors": []
        }
        
        try:
            # 使用与simple_weight_check.py相同的访问方式
            # 直接获取language_model，避免多级字典访问的问题
            language_model = megatron_weights['language_model']
            encoder = language_model['encoder']
            
            print(f"🔍 验证访问路径...")
            print(f"  language_model键: {list(language_model.keys())}")
            print(f"  encoder键: {list(encoder.keys())}")
            
            # 1. 词嵌入层
            print("  转换词嵌入层...")
            embed_weight = language_model['embedding']['word_embeddings']['weight']
            hf_state_dict['model.embed_tokens.weight'] = embed_weight.clone()
            conversion_stats["converted_weights"] += 1
            print(f"    ✅ embed_tokens: {embed_weight.shape}")
            
            # 2. 转换32层Transformer层
            print("  转换Transformer层...")
            num_layers = 32
            
            # 扁平化权重访问方式：直接从encoder获取权重
            print(f"  使用扁平化权重访问，直接从encoder获取")
            
            for layer_idx in tqdm(range(num_layers), desc="转换层"):
                layer_prefix = f"layers.{layer_idx}"
                
                # Input LayerNorm - 扁平化访问
                input_norm_key = f"{layer_prefix}.input_norm.weight"
                if input_norm_key not in encoder:
                    raise ValueError(f"缺少权重: {input_norm_key}")
                input_norm_weight = encoder[input_norm_key]
                hf_state_dict[f'model.layers.{layer_idx}.input_layernorm.weight'] = input_norm_weight.clone()
                
                # Self Attention - QKV分离 (扁平化访问)
                qkv_key = f"{layer_prefix}.self_attention.query_key_value.weight"
                if qkv_key not in encoder:
                    raise ValueError(f"缺少权重: {qkv_key}")
                qkv_weight = encoder[qkv_key]
                q_weight, k_weight, v_weight = self.split_qkv_weight(qkv_weight)
                
                hf_state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = q_weight.clone()
                hf_state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = k_weight.clone()
                hf_state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = v_weight.clone()
                
                # Self Attention Output (扁平化访问)
                dense_key = f"{layer_prefix}.self_attention.dense.weight"
                if dense_key not in encoder:
                    raise ValueError(f"缺少权重: {dense_key}")
                dense_weight = encoder[dense_key]
                hf_state_dict[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = dense_weight.clone()
                
                # Post Attention LayerNorm (扁平化访问)
                post_norm_key = f"{layer_prefix}.post_attention_norm.weight"
                if post_norm_key not in encoder:
                    raise ValueError(f"缺少权重: {post_norm_key}")
                post_norm_weight = encoder[post_norm_key]
                hf_state_dict[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = post_norm_weight.clone()
                
                # MLP - 分离gate和up投影 (扁平化访问)
                mlp_h_to_4h_key = f"{layer_prefix}.mlp.dense_h_to_4h.weight"
                if mlp_h_to_4h_key not in encoder:
                    raise ValueError(f"缺少权重: {mlp_h_to_4h_key}")
                dense_h_to_4h_weight = encoder[mlp_h_to_4h_key]
                gate_weight, up_weight = self.split_mlp_gate_up(dense_h_to_4h_weight)
                
                hf_state_dict[f'model.layers.{layer_idx}.mlp.gate_proj.weight'] = gate_weight.clone()
                hf_state_dict[f'model.layers.{layer_idx}.mlp.up_proj.weight'] = up_weight.clone()
                
                # MLP Down投影 (扁平化访问)
                mlp_4h_to_h_key = f"{layer_prefix}.mlp.dense_4h_to_h.weight"
                if mlp_4h_to_h_key not in encoder:
                    raise ValueError(f"缺少权重: {mlp_4h_to_h_key}")
                dense_4h_to_h_weight = encoder[mlp_4h_to_h_key]
                hf_state_dict[f'model.layers.{layer_idx}.mlp.down_proj.weight'] = dense_4h_to_h_weight.clone()
                
                conversion_stats["converted_weights"] += 7  # 每层7个权重
            
            # 3. Final LayerNorm (扁平化访问)
            print("  转换最终层归一化...")
            final_norm_key = 'final_norm.weight'
            if final_norm_key not in encoder:
                raise ValueError(f"缺少权重: {final_norm_key}")
            final_norm_weight = encoder[final_norm_key]
            hf_state_dict['model.norm.weight'] = final_norm_weight.clone()
            conversion_stats["converted_weights"] += 1
            print(f"    ✅ final_norm: {final_norm_weight.shape}")
            
            # 4. 输出层 (LM Head) - 从language_model下访问
            print("  转换输出层...")
            if 'output_layer' not in language_model:
                raise ValueError(f"language_model中未找到'output_layer'键，可用键: {list(language_model.keys())}")
            
            output_layer_data = language_model['output_layer']
            if 'weight' not in output_layer_data:
                raise ValueError(f"output_layer中未找到'weight'键，可用键: {list(output_layer_data.keys())}")
            
            output_weight = output_layer_data['weight']
            hf_state_dict['lm_head.weight'] = output_weight.clone()
            conversion_stats["converted_weights"] += 1
            print(f"    ✅ lm_head: {output_weight.shape}")
            
            print(f"\n📊 转换统计:")
            print(f"  成功转换: {conversion_stats['converted_weights']} 个权重")
            print(f"  跳过权重: {conversion_stats['skipped_weights']} 个")
            
            return hf_state_dict
            
        except Exception as e:
            print(f"❌ 权重转换失败: {str(e)}")
            raise
    
    def validate_conversion(self, hf_state_dict, hf_model):
        """验证转换结果"""
        print(f"\n🔍 验证转换结果...")
        
        try:
            # 检查权重数量
            expected_weights = len(hf_model.state_dict())
            actual_weights = len(hf_state_dict)
            
            print(f"📊 权重数量对比:")
            print(f"  期望权重数: {expected_weights}")
            print(f"  实际权重数: {actual_weights}")
            
            if expected_weights != actual_weights:
                print(f"⚠️  权重数量不匹配")
                
                # 找出缺失的权重
                expected_keys = set(hf_model.state_dict().keys())
                actual_keys = set(hf_state_dict.keys())
                
                missing_keys = expected_keys - actual_keys
                extra_keys = actual_keys - expected_keys
                
                if missing_keys:
                    print(f"  缺失权重: {len(missing_keys)} 个")
                    for key in list(missing_keys)[:5]:  # 显示前5个
                        print(f"    - {key}")
                
                if extra_keys:
                    print(f"  多余权重: {len(extra_keys)} 个")
                    for key in list(extra_keys)[:5]:  # 显示前5个
                        print(f"    + {key}")
            else:
                print(f"  ✅ 权重数量匹配")
            
            # 检查权重形状
            print(f"\n🔧 验证权重形状...")
            shape_errors = 0
            model_state = hf_model.state_dict()
            
            for name in hf_state_dict:
                if name in model_state:
                    expected_shape = model_state[name].shape
                    actual_shape = hf_state_dict[name].shape
                    
                    if expected_shape != actual_shape:
                        print(f"  ❌ {name}: 期望 {expected_shape}, 实际 {actual_shape}")
                        shape_errors += 1
            
            if shape_errors == 0:
                print(f"  ✅ 所有权重形状匹配")
            else:
                print(f"  ⚠️  发现 {shape_errors} 个形状错误")
            
            # 检查数据类型
            print(f"\n🔧 验证数据类型...")
            dtype_stats = {}
            for name, param in hf_state_dict.items():
                dtype = str(param.dtype)
                dtype_stats[dtype] = dtype_stats.get(dtype, 0) + 1
            
            print(f"  数据类型分布: {dtype_stats}")
            
            if 'torch.float16' in dtype_stats and len(dtype_stats) == 1:
                print(f"  ✅ 所有权重都是FP16")
            else:
                print(f"  ⚠️  数据类型不一致")
            
            return shape_errors == 0 and expected_weights == actual_weights
            
        except Exception as e:
            print(f"❌ 验证失败: {str(e)}")
            return False
    
    def save_model(self, hf_model, hf_state_dict):
        """保存HuggingFace模型"""
        print(f"\n💾 保存HuggingFace模型...")
        
        try:
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 加载权重到模型
            print("  加载权重到模型...")
            missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
            
            if missing_keys:
                print(f"  ⚠️  缺失权重: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"  ⚠️  意外权重: {len(unexpected_keys)} 个")
            
            # 确保模型是FP16
            hf_model = hf_model.half()
            
            # 保存模型权重为单文件格式 (参考原始模型结构)
            model_path = os.path.join(self.output_dir, "pytorch_model.bin")
            print(f"  保存权重到: {model_path}")
            torch.save(hf_model.state_dict(), model_path)
            
            # 保存配置文件
            config_path = os.path.join(self.output_dir, "config.json")
            print(f"  保存配置到: {config_path}")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            # 获取文件大小
            model_size = os.path.getsize(model_path) / (1024**3)
            print(f"  ✅ 模型文件大小: {model_size:.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"❌ 保存模型失败: {str(e)}")
            return False
    
    def copy_additional_files(self):
        """复制tokenizer等额外文件"""
        print(f"\n📋 复制额外文件...")
        
        if not os.path.exists(self.reference_model_dir):
            print(f"⚠️  参考模型目录不存在: {self.reference_model_dir}")
            return
        
        # 需要复制的文件列表
        files_to_copy = [
            "tokenizer_config.json",
            "special_tokens_map.json", 
            "vocab.txt",
            "generation_config.json"
        ]
        
        # 需要复制的目录
        dirs_to_copy = [
            "llama8b"  # tokenizer目录
        ]
        
        try:
            # 复制文件
            for filename in files_to_copy:
                src_file = os.path.join(self.reference_model_dir, filename)
                dst_file = os.path.join(self.output_dir, filename)
                
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"  ✅ 复制文件: {filename}")
                else:
                    print(f"  ⚠️  文件不存在: {filename}")
            
            # 复制目录
            for dirname in dirs_to_copy:
                src_dir = os.path.join(self.reference_model_dir, dirname)
                dst_dir = os.path.join(self.output_dir, dirname)
                
                if os.path.exists(src_dir):
                    if os.path.exists(dst_dir):
                        shutil.rmtree(dst_dir)
                    shutil.copytree(src_dir, dst_dir)
                    print(f"  ✅ 复制目录: {dirname}")
                else:
                    print(f"  ⚠️  目录不存在: {dirname}")
            
            # 创建generation_config.json (如果不存在)
            gen_config_path = os.path.join(self.output_dir, "generation_config.json")
            if not os.path.exists(gen_config_path):
                generation_config = {
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "pad_token_id": 0,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_new_tokens": 512
                }
                
                with open(gen_config_path, 'w', encoding='utf-8') as f:
                    json.dump(generation_config, f, indent=2)
                print(f"  ✅ 创建生成配置: generation_config.json")
            
        except Exception as e:
            print(f"❌ 复制额外文件失败: {str(e)}")
    
    def run_conversion(self):
        """执行完整的转换流程"""
        print(f"🎯 开始Megatron→HuggingFace转换")
        print("=" * 60)
        
        try:
            # 1. 加载Megatron检查点
            megatron_weights = self.load_megatron_checkpoint()
            
            # 2. 创建HuggingFace模型
            hf_model = self.create_hf_model()
            
            # 3. 转换权重
            hf_state_dict = self.convert_weights(megatron_weights, hf_model)
            
            # 4. 验证转换结果
            if not self.validate_conversion(hf_state_dict, hf_model):
                print("⚠️  验证发现问题，但继续保存...")
            
            # 5. 保存模型
            if not self.save_model(hf_model, hf_state_dict):
                print("❌ 保存失败")
                return False
            
            # 6. 复制额外文件
            self.copy_additional_files()
            
            # 7. 最终验证
            print(f"\n🎉 转换完成!")
            print("=" * 60)
            
            # 显示输出目录内容
            if os.path.exists(self.output_dir):
                print(f"📁 输出目录内容:")
                for item in sorted(os.listdir(self.output_dir)):
                    item_path = os.path.join(self.output_dir, item)
                    if os.path.isfile(item_path):
                        size = os.path.getsize(item_path) / (1024**2)  # MB
                        print(f"  📄 {item} ({size:.1f} MB)")
                    else:
                        print(f"  📁 {item}/")
                
                total_size = sum(
                    os.path.getsize(os.path.join(self.output_dir, f))
                    for f in os.listdir(self.output_dir)
                    if os.path.isfile(os.path.join(self.output_dir, f))
                ) / (1024**3)
                
                print(f"💾 总大小: {total_size:.2f} GB")
            
            print(f"\n✅ Megatron→HuggingFace转换成功完成!")
            print(f"🎯 现在可以使用transformers 4.40加载模型:")
            print(f"   model = AutoModelForCausalLM.from_pretrained('{self.output_dir}')")
            
            return True
            
        except Exception as e:
            print(f"❌ 转换失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="Megatron FP16 到 HuggingFace 格式转换")
    parser.add_argument("--megatron_checkpoint", "-m",
                       default="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_maskllm_training_tp1/iter_0002000/mp_rank_00/model_optim_rng_fp16.pt",
                       help="Megatron检查点路径")
    parser.add_argument("--output_dir", "-o",
                       default="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4_from_megatron",
                       help="HuggingFace输出目录")
    parser.add_argument("--reference_model", "-r",
                       default="/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b",
                       help="参考模型目录 (用于复制tokenizer等文件)")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.megatron_checkpoint):
        print(f"❌ Megatron检查点不存在: {args.megatron_checkpoint}")
        return
    
    if not os.path.exists(args.reference_model):
        print(f"❌ 参考模型目录不存在: {args.reference_model}")
        return
    
    # 创建转换器并执行转换
    converter = MegatronToHFConverter(
        megatron_checkpoint_path=args.megatron_checkpoint,
        output_dir=args.output_dir,
        reference_model_dir=args.reference_model
    )
    
    success = converter.run_conversion()
    
    if success:
        print(f"\n🎉 转换成功! 输出模型: {args.output_dir}")
    else:
        print(f"\n❌ 转换失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
