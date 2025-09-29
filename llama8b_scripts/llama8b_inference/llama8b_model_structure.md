# Llama8b-MaskLLM 模型结构分析与转换

模型保存格式为huggingface格式，直接用 transformers 库加载进行推理。
模型文件路径: `output/checkpoints/llama8b_hf_maskllm_c4`
推理脚本: `llama8b_scripts/llama8b_inference/llama8b_hf_inference_simple.py`
tokenizer 文件路径: `assets/checkpoints/Llama8b/llama8b/tokenizer.py`
vocab.txt 文件路径: `assets/checkpoints/Llama8b/vocab.txt`

prompt 格式为："<用户> 山东最高的山是什么山？<AI>"

## 模型概述

- **模型名称**: Llama8b (9g-8b model)
- **总参数量**: 8,765,444,096 (8.77B)
- **架构类型**: Transformer Decoder-Only
- **当前状态**: ✅ 已成功转换为HuggingFace格式

## 模型配置参数

- **层数**: 32层
- **隐藏维度**: 4,096  
- **MLP中间维度**: 14,336 (标准LLaMA配置)
- **注意力头数**: 32
- **词表大小**: 119,696
- **数据类型**: FP16
- **文件大小**: 16.33 GB

## ✅ **最终转换结果** (成功完成)

### 🎉 Megatron→HuggingFace转换 (推荐方案)

**转换状态**: ✅ **100%成功完成**

#### 输出模型
```bash
output/checkpoints/llama8b_hf_maskllm_c4_from_megatron/
├── pytorch_model.bin          # 16.33GB (FP16, 单文件)
├── config.json                # transformers 4.40兼容配置
├── generation_config.json     # 生成配置
├── tokenizer_config.json      # tokenizer配置
├── special_tokens_map.json    # 特殊token映射
├── vocab.txt                  # 词汇表 (1.1MB)
└── llama8b/                   # 自定义tokenizer目录
```

#### 转换统计
- ✅ **权重转换**: 227个权重 (100%成功)
- ✅ **权重数量**: 291个权重完全匹配
- ✅ **数据类型**: 100% FP16格式
- ✅ **文件大小**: 16.33GB (与原始Megatron一致)
- ✅ **转换时间**: 1分53秒
- ✅ **兼容性**: transformers 4.40完全兼容

#### 使用方法
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 直接加载转换后的模型
model = AutoModelForCausalLM.from_pretrained(
    '/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4_from_megatron'
)
tokenizer = AutoTokenizer.from_pretrained(
    '/data/home/zdhs0054/zzhou/MaskLLM/output/checkpoints/llama8b_hf_maskllm_c4_from_megatron'
)
```

### 执行转换命令
```bash
# 一键执行Megatron→HF转换
bash run_maskllm_native.sh llama8b_scripts/run_megatron_to_hf_conversion.sh
```

---

## 📋 **技术细节** (备用参考)

### 权重映射与转换技术细节

#### 关键权重转换处理：

1. **QKV权重分离**: `[12288, 4096]` → 3×`[4096, 4096]` (Q,K,V)
2. **MLP门控权重**: `[28672, 4096]` → `gate_proj[14336, 4096]` + `up_proj[14336, 4096]`
3. **扁平化权重访问**: Megatron使用扁平化键名存储(如`layers.0.input_norm.weight`)
4. **数据类型保持**: 所有权重保持FP16格式不变

#### 转换脚本位置：
- **转换脚本**: `llama8b_scripts/convert_megatron_to_hf.py`
- **启动脚本**: `llama8b_scripts/run_megatron_to_hf_conversion.sh`
- **分析脚本**: `llama8b_scripts/analyze_megatron_checkpoint.py`
