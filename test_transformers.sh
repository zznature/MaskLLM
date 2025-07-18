#!/bin/bash

echo "=== Transformers 环境配置和测试脚本 ==="
echo ""

echo "1. 设置正确的Python路径："
export PYTHONPATH="/data/home/zdhs0054/.local/lib/python3.10/site-packages:$PYTHONPATH"
echo "PYTHONPATH设置为: $PYTHONPATH"
echo ""

echo "2. 解决cuDNN库问题："
# 检查cuDNN库是否存在
if [ -f "$CONDA_PREFIX/lib/libcudnn.so.8.9.7" ]; then
    echo "找到cuDNN库: $CONDA_PREFIX/lib/libcudnn.so.8.9.7"
    
    # 创建符号链接
    if [ ! -f "$CONDA_PREFIX/lib/libcudnn.so.8" ]; then
        echo "创建cuDNN符号链接..."
        ln -sf "$CONDA_PREFIX/lib/libcudnn.so.8.9.7" "$CONDA_PREFIX/lib/libcudnn.so.8"
        ln -sf "$CONDA_PREFIX/lib/libcudnn_cnn_infer.so.8.9.7" "$CONDA_PREFIX/lib/libcudnn_cnn_infer.so.8" 2>/dev/null || echo "libcudnn_cnn_infer.so.8.9.7不存在"
        ln -sf "$CONDA_PREFIX/lib/libcudnn_cnn_train.so.8.9.7" "$CONDA_PREFIX/lib/libcudnn_cnn_train.so.8" 2>/dev/null || echo "libcudnn_cnn_train.so.8.9.7不存在"
    else
        echo "cuDNN符号链接已存在"
    fi
else
    echo "未找到cuDNN库，尝试其他位置..."
    find /usr/local -name "libcudnn.so.8*" 2>/dev/null | head -5
fi
echo ""

echo "3. 设置LD_LIBRARY_PATH："
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH设置为: $LD_LIBRARY_PATH"
echo ""

echo "4. 检查transformers安装位置："
find /data/home/zdhs0054/.local/lib/python3.10/site-packages -name "*transformers*" -type d
echo ""

echo "5. 检查Python包路径："
python3 -c "import sys; print('Python包路径:'); [print(f'  {p}') for p in sys.path]"
echo ""

echo "6. 测试transformers导入："
python3 -c "
import sys
sys.path.insert(0, '/data/home/zdhs0054/.local/lib/python3.10/site-packages')
try:
    import transformers
    print('transformers版本:', transformers.__version__)
    print('transformers导入成功!')
except Exception as e:
    print('transformers导入失败:', e)
"
echo ""

echo "7. 测试AutoTokenizer（不使用GPU）："
python3 -c "
import sys
sys.path.insert(0, '/data/home/zdhs0054/.local/lib/python3.10/site-packages')
try:
    from transformers import AutoTokenizer
    print('AutoTokenizer导入成功!')
    
    # 测试基本功能（不使用GPU）
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False)
    print('GPT-2 tokenizer加载成功!')
    
    # 测试分词
    text = 'Hello, world!'
    tokens = tokenizer.encode(text)
    print(f'文本: {text}')
    print(f'分词结果: {tokens}')
    print(f'解码结果: {tokenizer.decode(tokens)}')
    
except Exception as e:
    print('AutoTokenizer测试失败:', e)
"
echo ""

echo "8. 测试评估脚本相关组件："
python3 -c "
import sys
sys.path.insert(0, '/data/home/zdhs0054/.local/lib/python3.10/site-packages')
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print('transformers核心组件导入成功!')
    print('可以运行评估脚本了!')
except Exception as e:
    print('仍有问题:', e)
"
echo ""

echo "9. 创建永久环境配置："
echo "export PYTHONPATH=\"/data/home/zdhs0054/.local/lib/python3.10/site-packages:\$PYTHONPATH\"" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nccl/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
echo "已添加到 ~/.bashrc"
echo ""

echo "10. 检查是否可以使用run_maskllm_native.sh："
if [ -f "run_maskllm_native.sh" ]; then
    echo "找到run_maskllm_native.sh脚本，建议使用它来运行评估："
    echo "bash run_maskllm_native.sh scripts/ppl/evaluate_llama2_wikitext2.sh <checkpoint_path> 7b 4 sparse"
else
    echo "未找到run_maskllm_native.sh脚本"
fi
echo ""

echo "=== 测试完成 ===" 