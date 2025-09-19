# llama8b inference

模型保存格式为huggingface格式，直接用 transformers 库加载进行推理。
模型文件路径: `output/checkpoints/llama8b_hf_maskllm_c4`
推理脚本: `llama8b_scripts/llama8b_inference/llama8b_hf_inference_simple.py`
tokenizer 文件路径: `assets/checkpoints/Llama8b/llama8b/tokenizer.py`
vocab.txt 文件路径: `assets/checkpoints/Llama8b/vocab.txt`

prompt 格式为：
```
<用户> 山东最高的山是什么山？<AI>
```

## 快速使用

### 方法1: 交互式对话模式
```bash
# 使用默认路径启动交互式对话
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/llama8b_hf_inference_simple.py

# 或指定自定义路径
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/llama8b_hf_inference_simple.py \
    --model_path /path/to/model \
    --vocab_path /path/to/vocab.txt \
    --device cuda
```

### 方法2: 单次推理模式
```bash
# 单个问题推理
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/llama8b_hf_inference_simple.py \
    --prompt "山东最高的山是什么山？" \
    --max_length 256 \
    --temperature 0.7
```

### 方法3: 一键启动脚本
```bash
# 使用一键启动脚本（推荐）
bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/run_llama8b_hf_inference.sh
```

## 推理代码实现

核心推理类设计：
```python
class Llama8bInference:
    """Llama8b 推理类"""
    
    def __init__(self, model_path, vocab_path, device="auto"):
        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 加载自定义tokenizer
        self.tokenizer = Llama8bTokenizer(
            vocab_file=vocab_path,
            clean_up_tokenization_spaces=False,
            use_default_system_prompt=False,
            spaces_between_special_tokens=False,
            add_prefix_space=True,
        )
        
        # 加载HuggingFace模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
    def format_prompt(self, user_input):
        """格式化为指定的prompt格式"""
        return f"<用户> {user_input.strip()}<AI>"
    
    def generate(self, prompt, max_length=512, temperature=0.7):
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
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 只返回新生成的部分
        generated_tokens = outputs[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
```

## 使用示例

```python
# 初始化推理器
inference = Llama8bInference(
    model_path="output/checkpoints/llama8b_hf_maskllm_c4",
    vocab_path="assets/checkpoints/Llama8b/vocab.txt",
    device="cuda"
)

# 单次推理
user_question = "山东最高的山是什么山？"
prompt = inference.format_prompt(user_question)
response = inference.generate(prompt, max_length=256, temperature=0.7)
print(f"问题: {user_question}")
print(f"回答: {response}")

# 交互式对话
inference.interactive_chat()
```

## 特性说明

- ✅ **自定义Tokenizer**: 使用Llama8b专用的tokenizer.py
- ✅ **HuggingFace兼容**: 支持SafeTensors格式自动加载
- ✅ **设备自适应**: 自动检测GPU/CPU并选择合适的数据类型
- ✅ **交互式对话**: 支持连续对话模式
- ✅ **参数可调**: 支持temperature, top_p, top_k等生成参数
- ✅ **错误处理**: 完善的异常处理和用户提示
- ✅ **内存优化**: 支持低CPU内存模式和设备映射