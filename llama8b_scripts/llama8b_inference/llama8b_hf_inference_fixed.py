#!/usr/bin/env python3
"""
Llama8b HuggingFace 格式修复版推理脚本
修复SafeTensors加载问题
"""

import sys
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
import argparse
from pathlib import Path
import json

# 添加tokenizer路径到系统路径
tokenizer_path = "/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b/llama8b"
if tokenizer_path not in sys.path:
    sys.path.insert(0, tokenizer_path)

from tokenizer import Llama8bTokenizer

class Llama8bInferenceFixed:
    """修复版Llama8b推理类"""
    
    def __init__(self, model_path, vocab_path, device="auto"):
        """
        初始化推理器
        
        Args:
            model_path (str): HuggingFace模型路径
            vocab_path (str): 词汇表文件路径
            device (str): 设备选择 ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.vocab_path = vocab_path
        
        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 初始化 Llama8b 推理器 (修复版)")
        print(f"📁 模型路径: {model_path}")
        print(f"📝 词汇表路径: {vocab_path}")
        print(f"💻 使用设备: {self.device}")
        
        # 加载tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # 加载模型
        self.model = self._load_model_with_fallback()
        
        print("✅ 模型加载完成!")
        
    def _load_tokenizer(self):
        """加载自定义tokenizer"""
        print("📖 加载tokenizer...")
        
        try:
            tokenizer = Llama8bTokenizer(
                vocab_file=self.vocab_path,
                clean_up_tokenization_spaces=False,
                use_default_system_prompt=False,
                spaces_between_special_tokens=False,
                add_prefix_space=True,
            )
            
            # 设置特殊token
            tokenizer.pad_token = tokenizer.eos_token
            
            print(f"✅ Tokenizer加载成功 (词汇表大小: {len(tokenizer.vocab)})")
            return tokenizer
            
        except Exception as e:
            print(f"❌ Tokenizer加载失败: {str(e)}")
            raise
    
    def _load_model_with_fallback(self):
        """使用多种方法尝试加载模型"""
        print("🔄 加载模型...")
        
        # 加载配置
        try:
            config = AutoConfig.from_pretrained(self.model_path)
            print(f"📋 模型配置: {config.num_hidden_layers}层, {config.hidden_size}维")
        except Exception as e:
            print(f"⚠️  配置加载警告: {str(e)}")
            # 手动创建配置
            config = self._create_manual_config()
        
        # 尝试多种加载方法
        methods = [
            ("方法1: 标准AutoModel加载", self._load_method_1, config),
            ("方法2: 无设备映射加载", self._load_method_2, config),
            ("方法3: 显式LlamaForCausalLM加载", self._load_method_3, config),
            ("方法4: 最保守加载", self._load_method_4, config),
        ]
        
        for method_name, method_func, method_config in methods:
            print(f"🔧 尝试{method_name}...")
            try:
                model = method_func(method_config)
                if model is not None:
                    print(f"✅ {method_name}成功!")
                    return model
            except Exception as e:
                print(f"❌ {method_name}失败: {str(e)}")
                continue
        
        raise Exception("所有模型加载方法都失败了")
    
    def _create_manual_config(self):
        """手动创建模型配置"""
        print("🛠️  手动创建模型配置...")
        from transformers import LlamaConfig
        
        return LlamaConfig(
            vocab_size=119696,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=4096,
            rms_norm_eps=1e-05,
            initializer_range=0.02,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            torch_dtype=torch.float16,
        )
    
    def _load_method_1(self, config):
        """标准AutoModel加载方法"""
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" and torch.cuda.device_count() > 0 else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    def _load_method_2(self, config):
        """无设备映射加载方法"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.float32,  # 先用FP32加载
            device_map=None,
            trust_remote_code=False,
            low_cpu_mem_usage=True
        )
        
        # 手动移动到设备
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.half().to(self.device)
        else:
            model = model.to(self.device)
        
        model.eval()
        return model
    
    def _load_method_3(self, config):
        """显式LlamaForCausalLM加载方法"""
        model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=False,
        )
        
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.half().to(self.device)
        else:
            model = model.to(self.device)
        
        model.eval()
        return model
    
    def _load_method_4(self, config):
        """最保守加载方法 - 逐个加载权重"""
        print("🔧 使用最保守方法，可能需要较长时间...")
        
        # 创建空模型
        model = LlamaForCausalLM(config)
        
        # 尝试加载权重
        try:
            # 使用torch.load加载safetensors（如果可能）
            import glob
            safetensor_files = glob.glob(os.path.join(self.model_path, "*.safetensors"))
            if safetensor_files:
                print(f"发现 {len(safetensor_files)} 个safetensors文件")
                # 这里我们需要实现safetensors加载，但由于safetensors库不可用，先跳过
                print("⚠️  SafeTensors加载需要safetensors库支持")
                return None
            
        except Exception as e:
            print(f"权重加载失败: {str(e)}")
            return None
        
        return None
    
    def format_prompt(self, user_input):
        """格式化输入为指定的prompt格式"""
        return f"<用户> {user_input.strip()}<AI>"
    
    def generate(self, prompt, max_length=512, temperature=0.7, do_sample=True, 
                 top_p=0.9, top_k=50, repetition_penalty=1.1):
        """生成文本"""
        try:
            # Tokenize输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            input_length = input_ids.shape[1]
            
            print(f"🔤 输入tokens长度: {input_length}")
            print(f"📝 Prompt: {prompt}")
            
            # 生成参数
            generation_config = {
                "max_length": input_length + max_length,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            
            print("⚙️ 开始生成...")
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            
            # 解码输出
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
            return ""
    
    def interactive_chat(self):
        """交互式对话"""
        print("\n" + "="*60)
        print("🎯 Llama8b 交互式对话 (修复版)")
        print("💡 输入 'quit' 或 'exit' 退出")
        print("💡 输入 'clear' 清屏")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 用户: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见!")
                    break
                
                if user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                if not user_input:
                    print("⚠️  请输入有效内容")
                    continue
                
                # 格式化prompt
                prompt = self.format_prompt(user_input)
                
                # 生成回复
                print("\n🤖 AI正在思考...")
                response = self.generate(prompt, max_length=256, temperature=0.7)
                
                if response:
                    print(f"🤖 AI: {response}")
                else:
                    print("⚠️  生成失败，请重试")
                    
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
            except Exception as e:
                print(f"❌ 错误: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Llama8b HuggingFace Inference (Fixed)")
    parser.add_argument("--model_path", "-m", 
                       default="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4",
                       help="HuggingFace model path")
    parser.add_argument("--vocab_path", "-v",
                       default="/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b/vocab.txt",
                       help="Vocabulary file path")
    parser.add_argument("--device", "-d",
                       default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--prompt", "-p",
                       help="Single prompt for inference")
    parser.add_argument("--max_length",
                       type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--temperature",
                       type=float, default=0.7,
                       help="Temperature for generation")
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.vocab_path):
        print(f"❌ 词汇表文件不存在: {args.vocab_path}")
        return
    
    # 显示环境信息
    print("🔍 环境检查:")
    print(f"  - Python版本: {sys.version}")
    print(f"  - PyTorch版本: {torch.__version__}")
    try:
        import transformers
        print(f"  - Transformers版本: {transformers.__version__}")
    except ImportError:
        print("  - Transformers: 未安装")
    print(f"  - CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA设备数: {torch.cuda.device_count()}")
        print(f"  - 当前设备: {torch.cuda.current_device()}")
    print()
    
    # 初始化推理器
    try:
        inference = Llama8bInferenceFixed(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            device=args.device
        )
        
        if args.prompt:
            # 单次推理模式
            print("\n" + "="*60)
            print("🎯 单次推理模式")
            print("="*60)
            
            prompt = inference.format_prompt(args.prompt)
            response = inference.generate(
                prompt, 
                max_length=args.max_length,
                temperature=args.temperature
            )
            
            print(f"\n👤 用户: {args.prompt}")
            print(f"🤖 AI: {response}")
            
        else:
            # 交互式模式
            inference.interactive_chat()
            
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
