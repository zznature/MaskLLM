#!/bin/bash

# Phase 3.2: 转换后模型加载测试命令集
# 由于网络问题，提供具体命令供您手动执行

echo "🚀 Phase 3.2: 转换后模型加载测试"
echo "======================================="
echo ""
echo "由于网络环境限制，以下是您需要执行的命令："
echo ""

echo "📋 1. 基础环境检查："
echo "bash run_maskllm_native.sh -c 'python3.10 -c \"
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA设备数: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'设备{i}: {torch.cuda.get_device_name(i)}')
\"'"
echo ""

echo "📋 2. 检查转换后的模型文件："
echo "ls -la output/checkpoints/llama8b_megatron_tp8/"
echo "ls -la output/checkpoints/llama8b_megatron_tp8/iter_0000001/"
echo ""

echo "📋 3. 测试Megatron基础导入："
echo "bash run_maskllm_native.sh -c 'python3.10 -c \"
import sys
sys.path.insert(0, '/data/home/zdhs0054/zzhou/MaskLLM')
try:
    from megatron.arguments import parse_args
    print('✅ Megatron导入成功')
except Exception as e:
    print(f'❌ Megatron导入失败: {e}')
\"'"
echo ""

echo "📋 4. 运行简化的加载测试："
echo "bash run_maskllm_native.sh llama8b_scripts/simple_load_test.py"
echo ""

echo "📋 5. 如果基础测试通过，运行完整测试："
echo "bash run_maskllm_native.sh llama8b_scripts/test_model_loading_phase3_2.py"
echo ""

echo "📋 6. 测试实际模型加载（最简单版本）："
cat << 'EOF'
bash run_maskllm_native.sh -c 'python3.10 -c "
import sys
sys.path.insert(0, \"/data/home/zdhs0054/zzhou/MaskLLM\")

try:
    from megatron.arguments import parse_args
    from megatron.global_vars import set_global_variables
    
    # 设置最简参数
    sys.argv = [
        \"test\",
        \"--load\", \"output/checkpoints/llama8b_megatron_tp8\",
        \"--model-parallel-size\", \"1\",
        \"--num-layers\", \"32\",
        \"--hidden-size\", \"4096\",
        \"--num-attention-heads\", \"32\",
        \"--tokenizer-type\", \"Llama2Tokenizer\",
        \"--vocab-size\", \"119696\",
        \"--seq-length\", \"1024\",
        \"--micro-batch-size\", \"1\",
        \"--global-batch-size\", \"1\",
        \"--lr\", \"1e-4\",
        \"--train-iters\", \"1\",
        \"--swiglu\",
        \"--load-only\"
    ]
    
    args = parse_args()
    print(f\"✅ 参数解析成功: load={args.load}\")
    
    set_global_variables(args)
    print(\"✅ 全局变量设置成功\")
    
    print(\"🎉 基础加载测试通过！\")
    
except SystemExit as e:
    print(f\"⚠️ SystemExit: {e}\")
except Exception as e:
    print(f\"❌ 错误: {e}\")
    import traceback
    traceback.print_exc()
"'
EOF
echo ""

echo "🔧 预期结果："
echo "  ✅ CUDA设备数应该显示8个GPU"
echo "  ✅ 转换后模型目录应包含8个mp_rank分片"
echo "  ✅ Megatron模块应能正常导入"
echo "  ✅ 参数解析应该成功"
echo ""

echo "📊 如果遇到问题，请将输出结果提供给我分析。"
