#!/usr/bin/env python3
"""
Llama8b HuggingFace 格式简单推理脚本
支持自定义tokenizer和SafeTensors格式模型加载
"""

import sys
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import argparse
from pathlib import Path

# 添加tokenizer路径到系统路径
tokenizer_path = "/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b/llama8b"
if tokenizer_path not in sys.path:
    sys.path.insert(0, tokenizer_path)

from tokenizer import Llama8bTokenizer

class Llama8bInference:
    """Llama8b 推理类"""
    
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
            
        print(f"🚀 初始化 Llama8b 推理器")
        print(f"📁 模型路径: {model_path}")
        print(f"📝 词汇表路径: {vocab_path}")
        print(f"💻 使用设备: {self.device}")
        
        # 加载tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # 加载模型
        self.model = self._load_model()
        
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
    
    def _load_model(self):
        """加载HuggingFace格式模型"""
        print("🔄 加载模型...")
        
        try:
            # 加载配置
            config = AutoConfig.from_pretrained(self.model_path)
            print(f"📋 模型配置: {config.num_hidden_layers}层, {config.hidden_size}维")
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            model.eval()  # 设置为评估模式
            
            # 显示模型信息
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ 模型加载成功 (参数量: {total_params/1e9:.2f}B)")
            
            return model
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise
    
    def format_prompt(self, user_input):
        """
        格式化输入为指定的prompt格式
        
        Args:
            user_input (str): 用户输入
            
        Returns:
            str: 格式化后的prompt
        """
        return f"<用户> {user_input.strip()}<AI>"
    
    def generate(self, prompt, max_length=512, temperature=0.7, do_sample=True, 
                 top_p=0.9, top_k=50, repetition_penalty=1.1):
        """
        生成文本
        
        Args:
            prompt (str): 输入prompt
            max_length (int): 最大生成长度
            temperature (float): 温度参数
            do_sample (bool): 是否采样
            top_p (float): nucleus sampling参数
            top_k (int): top-k采样参数
            repetition_penalty (float): 重复惩罚
            
        Returns:
            str: 生成的文本
        """
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
            
            print("⚙️ 生成参数:", generation_config)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
            
            # 解码输出
            generated_tokens = outputs[0][input_length:]  # 只取新生成的部分
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
            return ""
    
    def interactive_chat(self):
        """交互式对话"""
        print("\n" + "="*60)
        print("🎯 Llama8b 交互式对话")
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
                response = self.generate(prompt)
                
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
    parser = argparse.ArgumentParser(description="Llama8b HuggingFace Inference")
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
                       help="Single prompt for inference (non-interactive mode)")
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
    
    # 初始化推理器
    try:
        inference = Llama8bInference(
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
        return

if __name__ == "__main__":
    main()
