#!/usr/bin/env python3
"""
修复 SafeTensors 兼容性问题的解决方案脚本
"""

import os
import sys
import subprocess
import json

def check_and_fix_environment():
    """检查并修复环境兼容性问题"""
    
    print("🔧 SafeTensors兼容性修复工具")
    print("=" * 50)
    
    # 1. 检查当前环境
    print("\n📋 当前环境状态:")
    
    try:
        import transformers
        transformers_version = transformers.__version__
        print(f"  Transformers版本: {transformers_version}")
        
        # 检查是否是问题版本
        version_parts = transformers_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        
        if major == 4 and minor >= 40:
            print(f"  ⚠️  检测到可能有问题的版本 (4.40+)")
            recommend_downgrade = True
        else:
            recommend_downgrade = False
            
    except ImportError:
        print("  ❌ Transformers未安装")
        return False
    
    try:
        import safetensors
        print(f"  SafeTensors版本: {safetensors.__version__}")
        safetensors_available = True
    except ImportError:
        print("  ❌ SafeTensors库未安装")
        safetensors_available = False
    
    # 2. 提供解决方案
    print(f"\n💡 建议的解决方案:")
    
    solutions = []
    
    if not safetensors_available:
        solutions.append({
            "name": "安装SafeTensors库",
            "commands": ["pip install safetensors"],
            "priority": 1
        })
    
    if recommend_downgrade:
        solutions.append({
            "name": "降级Transformers到稳定版本",
            "commands": [
                "pip install transformers==4.36.2",
                "pip install accelerate"
            ],
            "priority": 2
        })
    
    solutions.append({
        "name": "使用PyTorch原生加载方法",
        "commands": [],
        "priority": 3,
        "description": "创建绕过SafeTensors的PyTorch加载脚本"
    })
    
    # 3. 显示解决方案
    for i, solution in enumerate(sorted(solutions, key=lambda x: x['priority']), 1):
        print(f"\n方案 {i}: {solution['name']}")
        if solution.get('description'):
            print(f"  说明: {solution['description']}")
        
        for cmd in solution.get('commands', []):
            print(f"  命令: {cmd}")
    
    return True

def create_pytorch_native_loader():
    """创建PyTorch原生加载器，绕过SafeTensors问题"""
    
    script_content = '''#!/usr/bin/env python3
"""
PyTorch原生模型加载器 - 绕过SafeTensors兼容性问题
"""

import sys
import os
import torch
import json
from pathlib import Path

# 添加tokenizer路径
tokenizer_path = "/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b/llama8b"
if tokenizer_path not in sys.path:
    sys.path.insert(0, tokenizer_path)

from tokenizer import Llama8bTokenizer

class PyTorchNativeLlama8bInference:
    """使用PyTorch原生方法加载模型的推理类"""
    
    def __init__(self, model_path, vocab_path, device="auto"):
        self.model_path = model_path
        self.vocab_path = vocab_path
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 初始化 PyTorch原生 Llama8b 推理器")
        print(f"📁 模型路径: {model_path}")
        print(f"💻 使用设备: {self.device}")
        
        # 加载tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # 尝试加载模型
        self.model = self._load_model_pytorch_native()
        
        print("✅ 模型加载完成!")
    
    def _load_tokenizer(self):
        """加载tokenizer"""
        print("📖 加载tokenizer...")
        tokenizer = Llama8bTokenizer(
            vocab_file=self.vocab_path,
            clean_up_tokenization_spaces=False,
            use_default_system_prompt=False,
            spaces_between_special_tokens=False,
            add_prefix_space=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer加载成功 (词汇表大小: {len(tokenizer.vocab)})")
        return tokenizer
    
    def _load_model_pytorch_native(self):
        """使用PyTorch原生方法加载模型"""
        print("🔄 使用PyTorch原生方法加载模型...")
        
        # 方法1: 尝试从pytorch_model.bin加载（如果存在）
        pytorch_model_path = os.path.join(self.model_path, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            print("🔧 发现pytorch_model.bin文件，尝试加载...")
            try:
                return self._load_from_pytorch_bin()
            except Exception as e:
                print(f"❌ pytorch_model.bin加载失败: {str(e)}")
        
        # 方法2: 转换SafeTensors为PyTorch格式
        print("🔧 尝试转换SafeTensors为PyTorch格式...")
        try:
            return self._convert_and_load_safetensors()
        except Exception as e:
            print(f"❌ SafeTensors转换失败: {str(e)}")
        
        # 方法3: 使用较老版本的transformers方法
        print("🔧 尝试使用兼容性加载方法...")
        try:
            return self._load_with_compatibility_mode()
        except Exception as e:
            print(f"❌ 兼容性加载失败: {str(e)}")
        
        raise Exception("所有PyTorch原生加载方法都失败了")
    
    def _load_from_pytorch_bin(self):
        """从pytorch_model.bin加载"""
        from transformers import LlamaForCausalLM, LlamaConfig
        
        config_path = os.path.join(self.model_path, "config.json")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = LlamaConfig(**config_dict)
        model = LlamaForCausalLM(config)
        
        pytorch_model_path = os.path.join(self.model_path, "pytorch_model.bin")
        state_dict = torch.load(pytorch_model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        
        if self.device == "cuda":
            model = model.half().to(self.device)
        else:
            model = model.to(self.device)
        
        model.eval()
        return model
    
    def _convert_and_load_safetensors(self):
        """转换SafeTensors并加载"""
        print("⚠️  此方法需要safetensors库支持，当前跳过")
        return None
    
    def _load_with_compatibility_mode(self):
        """使用兼容模式加载"""
        # 临时禁用fast tokenizer和某些特性
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        from transformers import AutoModelForCausalLM, AutoConfig
        
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        
        # 使用最基础的加载方式
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=torch.float32,  # 强制使用FP32
            device_map=None,
            trust_remote_code=False,
            use_safetensors=False,  # 禁用SafeTensors
            local_files_only=True,
        )
        
        if self.device == "cuda":
            model = model.half().to(self.device)
        else:
            model = model.to(self.device)
        
        model.eval()
        return model
    
    def format_prompt(self, user_input):
        """格式化prompt"""
        return f"<用户> {user_input.strip()}<AI>"
    
    def generate(self, prompt, max_length=256, temperature=0.7):
        """生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_tokens = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Native Llama8b Inference")
    parser.add_argument("--model_path", "-m", 
                       default="/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4",
                       help="Model path")
    parser.add_argument("--vocab_path", "-v",
                       default="/data/home/zdhs0054/zzhou/MaskLLM/assets/checkpoints/Llama8b/vocab.txt",
                       help="Vocab path")
    parser.add_argument("--prompt", "-p", help="Prompt for inference")
    parser.add_argument("--device", default="auto", help="Device")
    
    args = parser.parse_args()
    
    try:
        inference = PyTorchNativeLlama8bInference(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            device=args.device
        )
        
        if args.prompt:
            prompt = inference.format_prompt(args.prompt)
            response = inference.generate(prompt)
            print(f"\\n👤 用户: {args.prompt}")
            print(f"🤖 AI: {response}")
        else:
            print("\\n🎯 请使用 --prompt 参数指定问题")
            
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    script_path = "/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/llama8b_inference/pytorch_native_inference.py"
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    
    print(f"✅ 已创建PyTorch原生加载脚本: {script_path}")
    
    return script_path

def main():
    print("🔧 SafeTensors兼容性问题解决工具")
    print("=" * 50)
    
    # 检查环境
    if not check_and_fix_environment():
        return
    
    # 创建备用加载器
    script_path = create_pytorch_native_loader()
    
    print(f"\\n🎯 下一步建议:")
    print(f"1. 安装缺失的库:")
    print(f"   pip install safetensors")
    print(f"   pip install transformers==4.36.2")
    print(f"\\n2. 或使用PyTorch原生加载器:")
    print(f"   bash run_maskllm_native.sh {script_path.replace('/data/home/zdhs0054/zzhou/MaskLLM/', '')} \\\\")
    print(f"       --prompt '山东最高的山是什么山？'")

if __name__ == "__main__":
    main()
