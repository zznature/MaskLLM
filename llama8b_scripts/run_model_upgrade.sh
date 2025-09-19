#!/bin/bash
# Llama8b 模型兼容性升级批处理脚本

set -e

echo "🔄 Llama8b 模型兼容性升级工具"
echo "=================================="

# 显示当前状态
echo "📋 当前状态分析:"
echo "  原始模型: transformers 4.31.0 格式"
echo "  目标格式: transformers 4.40.0 兼容"
echo "  文件大小: 17GB (FP16) - 保持不变"
echo "  升级类型: 版本兼容性升级（非格式转换）"
echo ""

# 检查先决条件
echo "🔍 检查环境..."
python3 -c "
import sys
try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
    
    import torch  
    print(f'✅ PyTorch: {torch.__version__}')
    
    try:
        import safetensors
        print(f'✅ SafeTensors: {safetensors.__version__}')
    except ImportError:
        print('❌ SafeTensors未安装')
        print('解决方案: pip install safetensors')
        sys.exit(1)
        
except ImportError as e:
    print(f'❌ 缺少依赖: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 环境检查失败，请安装缺少的依赖："
    echo "   pip install safetensors transformers torch"
    exit 1
fi

echo ""
echo "🚀 开始模型兼容性升级..."

# 运行升级脚本
bash run_maskllm_native.sh llama8b_scripts/upgrade_model_compatibility.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 升级成功！"
    echo ""
    echo "📁 文件位置:"
    echo "  原始模型: output/checkpoints/llama8b_hf_maskllm_c4"
    echo "  升级模型: output/checkpoints/llama8b_hf_maskllm_c4_v440"
    echo "  备份文件: output/checkpoints/llama8b_hf_maskllm_c4_backup"
    echo ""
    echo "🧪 测试推理:"
    echo "  bash run_maskllm_native.sh llama8b_scripts/llama8b_inference/llama8b_hf_inference_simple.py \\"
    echo "      --model_path output/checkpoints/llama8b_hf_maskllm_c4_v440 \\"
    echo "      --prompt '山东最高的山是什么山？'"
    
else
    echo ""
    echo "❌ 升级失败，请检查错误信息"
    echo ""
    echo "💡 故障排除:"
    echo "  1. 确保有足够磁盘空间 (需要额外17GB)"
    echo "  2. 确保有足够内存 (建议32GB+)"
    echo "  3. 检查原始模型文件完整性"
    echo "  4. 尝试重新安装依赖："
    echo "     pip install --upgrade transformers safetensors torch"
fi
