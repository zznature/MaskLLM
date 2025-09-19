#!/usr/bin/env python3
"""
测试 Llama8b 稀疏模型推理功能
基于 llama8b_infer.py 创建的简化测试脚本
"""

import argparse
import os
import sys
import torch
import importlib
from typing import Optional

# 项目路径配置
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
EXT_PKGS_DIR = os.path.join(PROJECT_DIR, '.ext_pkgs')

# 全局变量
AutoTokenizer = None
AutoModelForCausalLM = None


def ensure_hf_import() -> None:
    """导入 transformers 库，优先使用容器内版本，失败时回退到 .ext_pkgs"""
    global AutoTokenizer, AutoModelForCausalLM
    if AutoTokenizer is not None and AutoModelForCausalLM is not None:
        return
    
    try:
        tm = importlib.import_module('transformers')
        print(f"[Info] 使用容器内 transformers: {tm.__file__}")
    except ImportError:
        if os.path.isdir(EXT_PKGS_DIR) and EXT_PKGS_DIR not in sys.path:
            sys.path.insert(0, EXT_PKGS_DIR)
        tm = importlib.import_module('transformers')
        print(f"[Info] 使用 .ext_pkgs transformers: {tm.__file__}")
    
    AutoTokenizer = getattr(tm, 'AutoTokenizer')
    AutoModelForCausalLM = getattr(tm, 'AutoModelForCausalLM')


def parse_args():
    parser = argparse.ArgumentParser(description="测试 Llama8b 稀疏模型推理")
    parser.add_argument("--model", type=str, 
                       default="output/checkpoints/llama8b_hf_maskllm_c4",
                       help="HF格式的稀疏模型路径")
    parser.add_argument("--prompt", type=str, 
                       default="You are a helpful assistant. Please introduce yourself and explain what MaskLLM is.",
                       help="测试提示词")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                       help="生成的最大新token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    parser.add_argument("--use-gpu", action="store_true", 
                       help="如果可用则使用GPU")
    parser.add_argument("--dtype", type=str, default="float16", 
                       choices=["float32", "float16", "bfloat16"],
                       help="推理数据类型")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    """获取PyTorch数据类型"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16, 
        "bfloat16": torch.bfloat16
    }
    return dtype_map.get(dtype_str, torch.float16)


def load_llama8b_tokenizer(model_path: str):
    """加载Llama8b自定义tokenizer"""
    # 方法1: 尝试使用原始Llama8bTokenizer
    original_llama8b_path = "assets/checkpoints/Llama8b"
    if os.path.exists(original_llama8b_path):
        try:
            print(f"[Info] 使用原始Llama8bTokenizer: {original_llama8b_path}")
            
            # 添加路径到sys.path
            if original_llama8b_path not in sys.path:
                sys.path.insert(0, original_llama8b_path)
            
            # 导入自定义tokenizer
            from llama8b.tokenizer import Llama8bTokenizer
            
            # 使用vocab文件路径
            vocab_file = os.path.join(original_llama8b_path, "vocab.txt")
            if not os.path.exists(vocab_file):
                raise FileNotFoundError(f"vocab.txt not found at {vocab_file}")
            
            print(f"[Info] 使用vocab文件: {vocab_file}")
            
            # 创建tokenizer实例
            tokenizer = Llama8bTokenizer(vocab_file=vocab_file)
            print(f"✅ Llama8bTokenizer加载成功，vocab_size: {tokenizer.vocab_size}")
            
            return tokenizer
            
        except Exception as e:
            print(f"[Warn] 原始Llama8bTokenizer加载失败: {e}")
    
    # 方法2: 回退到使用HF导出的模型中的vocab.txt
    try:
        print(f"[Info] 尝试从HF模型目录加载: {model_path}")
        
        # 检查HF模型目录中是否有vocab.txt
        hf_vocab_file = os.path.join(model_path, "vocab.txt")
        if os.path.exists(hf_vocab_file):
            # 添加原始路径到sys.path
            original_llama8b_path = "assets/checkpoints/Llama8b"
            if os.path.exists(original_llama8b_path) and original_llama8b_path not in sys.path:
                sys.path.insert(0, original_llama8b_path)
            
            from llama8b.tokenizer import Llama8bTokenizer
            
            print(f"[Info] 使用HF模型中的vocab文件: {hf_vocab_file}")
            tokenizer = Llama8bTokenizer(vocab_file=hf_vocab_file)
            print(f"✅ HF路径下的Llama8bTokenizer加载成功，vocab_size: {tokenizer.vocab_size}")
            
            return tokenizer
        else:
            print(f"[Error] vocab.txt not found in HF model directory: {hf_vocab_file}")
            
    except Exception as e:
        print(f"[Error] HF路径tokenizer加载失败: {e}")
    
    # 方法3: 最后尝试AutoTokenizer (通常会失败，但作为后备)
    try:
        ensure_hf_import()
        print(f"[Info] 最后尝试AutoTokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=False
        )
        print(f"✅ AutoTokenizer加载成功，vocab_size: {len(tokenizer)}")
        return tokenizer
        
    except Exception as e:
        print(f"[Error] 所有tokenizer加载方法都失败了")
        print(f"[Error] AutoTokenizer失败: {e}")
        raise Exception(f"无法加载tokenizer，请确保原始Llama8b目录存在: assets/checkpoints/Llama8b/")


def test_model_loading(model_path: str, torch_dtype, device: str):
    """测试模型加载"""
    print(f"[测试] 正在加载模型: {model_path}")
    print(f"[测试] 设备: {device}, 数据类型: {torch_dtype}")
    
    ensure_hf_import()
    
    try:
        # 智能加载tokenizer，处理Llama8bTokenizer问题
        tokenizer = load_llama8b_tokenizer(model_path)
        print(f"✅ Tokenizer加载成功，词汇表大小: {len(tokenizer)}")
        
        # 设置特殊token (对于自定义tokenizer)
        print(f"[Debug] tokenizer类型: {type(tokenizer)}")
        
        # 确保pad_token设置
        if getattr(tokenizer, "pad_token_id", None) is None:
            if hasattr(tokenizer, "eos_token"):
                tokenizer.pad_token = tokenizer.eos_token
                print("[Info] 设置pad_token为eos_token")
            elif hasattr(tokenizer, "eos_id"):
                # 对于自定义tokenizer，直接设置pad_token_id
                tokenizer.pad_token_id = tokenizer.eos_id
                print(f"[Info] 设置pad_token_id为eos_id: {tokenizer.eos_id}")
        
        # 打印tokenizer属性信息
        if hasattr(tokenizer, 'vocab_size'):
            print(f"[Info] tokenizer.vocab_size: {tokenizer.vocab_size}")
        if hasattr(tokenizer, 'bos_id'):
            print(f"[Info] bos_id: {tokenizer.bos_id}")
        if hasattr(tokenizer, 'eos_id'):
            print(f"[Info] eos_id: {tokenizer.eos_id}")
        if hasattr(tokenizer, 'unk_id'):
            print(f"[Info] unk_id: {tokenizer.unk_id}")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # 避免依赖Accelerate
        )
        print(f"✅ 模型加载成功")
        
        # 移动到设备
        if device == "cuda":
            model = model.to("cuda")
            print(f"✅ 模型已移动到GPU")
        
        model.eval()
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None


def run_inference_test(model, tokenizer, prompt: str, max_new_tokens: int, 
                      temperature: float, device: str, torch_dtype):
    """运行推理测试"""
    print(f"\n[测试] 开始推理测试")
    print(f"[测试] 提示词长度: {len(prompt)} 字符")
    
    try:
        # 编码输入 - 兼容自定义tokenizer
        try:
            # 方法1: 尝试标准HF接口
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            print("[Info] 使用标准HF tokenizer接口")
            
        except Exception as e:
            # 方法2: 使用自定义tokenizer的encode方法
            print(f"[Info] 标准接口失败，尝试自定义接口: {e}")
            if hasattr(tokenizer, 'encode'):
                token_ids = tokenizer.encode(prompt)
            else:
                # 方法3: 手动tokenize + convert
                tokens = tokenizer._tokenize(prompt)
                token_ids = [tokenizer._convert_token_to_id(token) for token in tokens]
                
            # 转换为tensor
            input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
            print("[Info] 使用自定义tokenizer接口")
        
        input_length = input_ids.shape[-1]
        print(f"[测试] 输入token数: {input_length}")
        
        # 推理设置
        use_autocast = device == "cuda" and torch_dtype in (torch.float16, torch.bfloat16)
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch_dtype) if use_autocast else torch.no_grad()
        
        print(f"[测试] 开始生成 (max_new_tokens={max_new_tokens})...")
        
        with torch.no_grad():
            with autocast_ctx:
                # 获取pad_token_id
                pad_token_id = None
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    pad_token_id = tokenizer.pad_token_id
                elif hasattr(tokenizer, 'eos_id'):
                    pad_token_id = tokenizer.eos_id
                    print(f"[Info] 使用eos_id作为pad_token_id: {pad_token_id}")
                
                gen_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                }
                
                if pad_token_id is not None:
                    gen_kwargs["pad_token_id"] = pad_token_id
                
                outputs = model.generate(**gen_kwargs)
        
        # 解码输出 - 兼容自定义tokenizer
        generated_ids = outputs[0][input_length:]
        print(f"✅ 生成成功，新token数: {len(generated_ids)}")
        
        try:
            # 方法1: 标准HF解码
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print("[Info] 使用标准decode方法")
        except Exception as e:
            print(f"[Info] 标准decode失败，尝试自定义方法: {e}")
            # 方法2: 自定义解码
            generated_ids_cpu = generated_ids.cpu().tolist()
            try:
                if hasattr(tokenizer, 'raw_decode'):
                    generated_text = tokenizer.raw_decode(generated_ids_cpu)
                    print("[Info] 使用raw_decode方法")
                else:
                    # 方法3: 手动解码
                    tokens = [tokenizer._convert_id_to_token(id) for id in generated_ids_cpu]
                    generated_text = tokenizer.convert_tokens_to_string(tokens)
                    print("[Info] 使用手动解码方法")
            except Exception as e2:
                print(f"[Error] 解码失败: {e2}")
                generated_text = f"[解码失败] Generated {len(generated_ids)} tokens"
        
        return generated_text
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return None


def print_results(prompt: str, generated_text: str):
    """打印推理结果"""
    print("\n" + "="*80)
    print("推理测试结果")
    print("="*80)
    print("\n📝 输入提示词:")
    print("-" * 40)
    print(prompt)
    print("\n🤖 模型生成:")
    print("-" * 40)
    print(generated_text)
    print("\n" + "="*80)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    # 设备和数据类型
    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu" 
    torch_dtype = get_torch_dtype(args.dtype)
    
    print(f"🚀 Llama8b 稀疏模型推理测试")
    print(f"📁 模型路径: {args.model}")
    print(f"⚙️ 设备: {device}")
    print(f"⚙️ 数据类型: {torch_dtype}")
    print(f"⚙️ 最大新token: {args.max_new_tokens}")
    print(f"⚙️ 温度: {args.temperature}")
    print()
    
    # 检查模型路径
    if not os.path.exists(args.model):
        print(f"❌ 模型路径不存在: {args.model}")
        return 1
    
    # 加载模型和tokenizer
    model, tokenizer = test_model_loading(args.model, torch_dtype, device)
    if model is None or tokenizer is None:
        print("❌ 模型加载失败，退出测试")
        return 1
    
    # 运行推理测试
    generated_text = run_inference_test(
        model, tokenizer, args.prompt, args.max_new_tokens,
        args.temperature, device, torch_dtype
    )
    
    if generated_text is None:
        print("❌ 推理测试失败")
        return 1
    
    # 打印结果
    print_results(args.prompt, generated_text)
    print("✅ 推理测试完成！")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
