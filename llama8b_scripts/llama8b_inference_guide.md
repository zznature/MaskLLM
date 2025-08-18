# 推理

llama8b 是 9g-8b 模型。

## 离线批量推理：

离线批量推理可参考以下脚本：

```python
# offline_inference.py
from vllm import LLM, SamplingParams
# 提示用例
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# 设置采样参数以控制生成文本，更多参数详细介绍可见/vllm/sampling_params.py
# temperature越大，生成结果的随机性越强，top_p过滤掉生成词汇表中概率低于给定阈值的词汇，控制随机性
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# 初始化语言模型,需注意加载的是.bin格式模型
llm = LLM(model="../models/9G/", trust_remote_code=True)
# 根据提示生成文本
outputs = llm.generate(prompts, sampling_params)
# 打印输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

在初始化语言模型部分，不同模型有所差异：
端侧2B模型：
```python
# 初始化语言模型，与Hugging Face Transformers库兼容，支持AWQ、GPTQ和GGUF量化格式转换
llm = LLM(model="../models/2b_sft_model/", tokenizer_mode="auto", trust_remote_code=True)
```

8B百亿SFT模型：
```python
# 初始化语言模型，tokenizer_mode需指定为"cpm"，不支持AWQ、GPTQ和GGUF量化格式转换
# 需要特别注意的是，由于8B模型训练分词方式差异，vllm库中代码有新增，需要按照“环境配置”安装指定版本vllm
llm = LLM(model="../models/8b_sft_model/", tokenizer_mode="cpm", trust_remote_code=True)
```

如果想使用多轮对话,需要指定对应的聊天模版,修改prompts,每次将上一轮的问题和答案拼接到本轮输入即可：
```python
prompts = [
        "<用户>问题1<AI>答案1<用户>问题2<AI>答案2<用户>问题3<AI>"
        ]
```

## 部署OpenAI API服务推理

vLLM可以为 LLM 服务进行部署，这里提供了一个示例：

启动服务：

### 端侧2B模型：

```python
python -m vllm.entrypoints.openai.api_server \
       --model ../models/2b_sft_model/ \
       --tokenizer-mode auto \ 
       --dtype auto \
       --trust-remote-code \ 
       --api-key FM9GAPI
#同样需注意模型加载的是.bin格式
#与离线批量推理类似，使用端侧2B模型，tokenizer-mode为"auto"
#dtype为模型数据类型，设置为"auto"即可
#api-key为可选项，可在此处指定你的api密钥
```

### 8B百亿SFT模型：

```python
python -m vllm.entrypoints.openai.api_server \
       --model ../models/8b_sft_model/ \ 
       --tokenizer-mode cpm \ 
       --dtype auto \
       --api-key FM9GAPI
#与离线批量推理类似，使用8B百亿SFT模型，tokenizer-mode为"cpm"
```

执行对应指令后，默认在http://localhost:8000地址上启动服务，启动成功后终端会出现如下提示：

INFO:     Started server process [950965]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

## 部署阶段部分参数说明：

1.openai暂不支持配置--repetition_penalty参数。
2.如发生OOM（内存溢出）问题，可修改相关参数，在启动服务时一并传入:
--max-model-len，默认最大位置嵌入（max_position_embedding）为32768，可以修改为4096。
--gpu-memory-utilization，默认该值为0.9，因此占用显存比较高，2B模型可修改为0.2，8B模型可修改为0.5。
```

调用推理API： 启动服务端成功后，重新打开一个终端，可参考执行以下python脚本：
```python
# client.py
from openai import OpenAI
# 如果启动服务时指定了api密钥，需要修改为对应的密钥，否则为"EMPTY"
openai_api_key = "FM9GAPI" 
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
#指定模型路径，推理prompt以及设置采样参数以控制生成文本
completion_params = {
    "model": "../models/9G/",
    "prompt": "San Francisco is a",
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 200
}
completion = client.completions.create(**completion_params)
print("Completion result:", completion)
```

调用多轮对话API： 启动服务端成功后，重新打开一个终端，可参考执行以下python脚本：
```python
# chat_client.py
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="FM9GAPI",
)
#每次将上一轮的问题和答案拼接到本轮输入即可
completion = client.chat.completions.create(
  model="../models/9G/",
  messages=[
    {"role": "user", "content": "问题1"},
    {"role": "system", "content": "答案1"},
    {"role": "user", "content": "问题2"},
    {"role": "system", "content": "答案2"},
    {"role": "user", "content": "问题3"},
  ]
)
print(completion.choices[0].message)
```