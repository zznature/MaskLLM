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
## 结构化推理脚本方案（MaskLLM容器）

当前推荐的推理脚本 `llama8b_infer.py` 采用“容器内 PyTorch + .ext_pkgs Transformers”的混合策略，重点解决 `transformer_engine` 与 PyTorch ABI 不匹配的问题。核心技术路径如下：

1. **环境变量与路径配置**  
   - 优先从容器 `/usr/local/lib/python3.10/dist-packages` 加载 PyTorch，确保与 GPU 驱动 ABI 一致  
   - 追加 `.ext_pkgs` 到 `PYTHONPATH`，提供 `transformers==4.40.0`、`huggingface_hub` 等外部包  
   - 继承 `run_maskllm_native.sh` 的 UCX/UCC 禁用环境变量（`TORCH_UCC_DISABLE`, `NCCL_UCX_DISABLE` 等），保证容器通信稳定

2. **兼容性预处理**  
   - 先导入容器 PyTorch，打印来源  
   - 临时将 `.ext_pkgs` 置于前缀位置，加载较新的 `typing_extensions`，随后恢复路径顺序，避免 `pydantic` / `transformers` 缺少 `TypeIs` 等符号  
   - 自动检测并加载本地 `Llama8bTokenizer` 与 `vocab.txt`

3. **accelerate shim 方案**  
   - 注入自定义模块 `accelerate.utils.transformer_engine`，提供 `convert_model`、`is_fp8_available`、`has_transformer_engine_layers` 的空实现  
   - 使 `accelerate` 在初始化时跳过真正的 `transformer_engine` 原生扩展，彻底规避 `transformer_engine_extensions.so` 符号缺失错误

4. **推理流程**  
   - `AutoTokenizer.from_pretrained(..., local_files_only=True)`，优先使用本地文件，避免网络访问  
   - `AutoModelForCausalLM.from_pretrained` 强制本地加载 + `torch_dtype` 自适应  
   - 自动对齐 `generation_config`，允许自定义 `temperature/top_p/top_k`  
   - 运行时输出生成文本，并对 `stop_text` 列表进行后处理

### 使用方式（执行顺序）

1. **进入容器**：
   ```bash
   bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/run_llama8b_infer.sh
   ```
   该脚本会自动：
   - 设置容器内 PyTorch 优先级和 .ext_pkgs 的补充路径
   - 配置 UCX/UCC 禁用环境变量，消除通信冲突
   - 检测并输出 PyTorch / transformers / tokenizer 等关键库版本

2. **自定义参数**：
   - 修改 `run_llama8b_infer.sh` 中的 `PROMPT_TEXT`、`MAX_NEW_TOKENS`、`TEMPERATURE` 等变量即可改变推理行为
   - 若需在命令行覆盖，可在脚本中追加自定义传参逻辑

3. **查看结果**：
   - 推理脚本会输出原始 prompt、生成文本，以及环境信息日志

### 方案特点

- ✅ 容器内 PyTorch 与外部 Transformers 协同工作，避免 ABI 冲突  
- ✅ 完整绕开 `transformer_engine` 原生扩展，彻底消除 undefined symbol 问题  
- ✅ 兼容本地 tokenizer / vocab，自定义 prompt 和采样策略  
- ✅ 纯本地加载，`local_files_only=True`，无需网络  
- ✅ 已在实际用例中验证，生成结果符合预期（示例：山东最高的山 → 泰山）

===== Prompt =====
<用户> 山东最高的山是什么山？<AI>
===== Generation =====
山东最高的山是泰山，也被称为东岳或泰山。它是五岳之首，是中国最著名的五座山之一，也是世界上最大的山之一。泰山位于山东省中部，其主峰玉皇峰海拔1522.8米，是山东最高的山。