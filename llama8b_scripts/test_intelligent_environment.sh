#!/bin/bash

# 智能混合环境测试脚本
# 测试修复后的run_maskllm_native.sh配置

echo "🧪 智能混合环境测试"
echo "时间: $(date)"
echo "目标: 验证容器内优先 + 安全.ext_pkgs回退方案"
echo "=================================="

# 1. 测试基本PyTorch功能
echo "1️⃣ 测试PyTorch（容器内原生）"
bash run_maskllm_native.sh python3.10 -c "
import torch
print('✅ PyTorch版本:', torch.__version__)
print('✅ PyTorch路径:', torch.__file__)
print('✅ CUDA可用:', torch.cuda.is_available())

# 验证是否使用容器内版本
if 'usr/local' in torch.__file__:
    print('🐳 确认使用容器内PyTorch')
else:
    print('⚠️ 可能使用外部PyTorch')
"

echo ""

# 2. 测试transformers功能
echo "2️⃣ 测试transformers（.ext_pkgs回退）"
bash run_maskllm_native.sh python3.10 -c "
try:
    import transformers
    print('✅ transformers版本:', transformers.__version__)
    print('✅ transformers路径:', transformers.__file__)
    
    # 验证来源
    if '.ext_pkgs' in transformers.__file__:
        print('📦 确认使用.ext_pkgs中的transformers')
    else:
        print('🐳 使用容器内transformers')
    
    # 测试基本功能
    from transformers import AutoTokenizer
    print('✅ AutoTokenizer导入成功')
    
except Exception as e:
    print('❌ transformers测试失败:', e)
"

echo ""

# 3. 测试megatron.tokenizer
echo "3️⃣ 测试megatron.tokenizer（项目集成）"
bash run_maskllm_native.sh python3.10 -c "
try:
    from megatron.tokenizer import build_tokenizer
    print('✅ megatron.tokenizer导入成功')
    print('✅ build_tokenizer函数可用')
except Exception as e:
    print('❌ megatron.tokenizer测试失败:', e)
"

echo ""

# 4. 测试UCC冲突情况
echo "4️⃣ 测试UCC冲突检测"
bash run_maskllm_native.sh python3.10 -c "
import sys
import importlib.util

# 检查torch是否有UCC相关符号
try:
    import torch
    # 尝试导入torch分布式模块
    import torch.distributed as dist
    print('✅ torch.distributed导入成功，无UCC冲突')
except Exception as e:
    if 'ucc' in str(e).lower():
        print('❌ 检测到UCC冲突:', e)
    else:
        print('⚠️ 其他导入问题:', e)

# 检查PYTHONPATH中是否存在危险包
pythonpath = sys.path
dangerous_paths = []
for path in pythonpath:
    if '.ext_pkgs' in path and any(pkg in path for pkg in ['torch', 'nvidia', 'triton']):
        dangerous_paths.append(path)

if dangerous_paths:
    print('⚠️ 发现危险路径:')
    for path in dangerous_paths:
        print(f'  - {path}')
else:
    print('✅ 无危险.ext_pkgs路径')
"

echo ""

# 5. 测试混合环境总体功能
echo "5️⃣ 测试混合环境总体功能"
bash run_maskllm_native.sh python3.10 -c "
print('=== 包来源验证 ===')

packages_to_check = [
    ('torch', '容器内'), 
    ('transformers', '.ext_pkgs'),
    ('numpy', '容器内'),
    ('requests', '.ext_pkgs')
]

for package, expected_source in packages_to_check:
    try:
        module = __import__(package)
        file_path = module.__file__
        
        if '.ext_pkgs' in file_path:
            actual_source = '.ext_pkgs'
        elif 'usr/local' in file_path:
            actual_source = '容器内'
        else:
            actual_source = '其他'
        
        status = '✅' if actual_source == expected_source else '⚠️'
        print(f'{status} {package}: {actual_source} (预期: {expected_source})')
        
    except ImportError as e:
        print(f'❌ {package}: 导入失败 - {e}')

print()
print('=== 环境优先级验证 ===')
import sys
pythonpath = sys.path[:5]  # 显示前5个路径
for i, path in enumerate(pythonpath):
    if '/usr/local/lib/python3.10/dist-packages' in path:
        print(f'{i+1}. 🐳 容器内原生: {path}')
    elif '.ext_pkgs' in path:
        print(f'{i+1}. 📦 外部安全包: {path}')
    else:
        print(f'{i+1}. 📁 其他路径: {path}')
"

echo ""

# 6. 测试实际MaskLLM功能
echo "6️⃣ 测试MaskLLM核心导入"
bash run_maskllm_native.sh python3.10 -c "
try:
    # 测试核心MaskLLM组件
    print('测试核心PyTorch功能...')
    import torch
    x = torch.randn(2, 2)
    y = torch.mm(x, x)
    print('✅ PyTorch张量运算正常')
    
    print('测试tokenizer相关功能...')
    import transformers
    print('✅ transformers可用')
    
    # 测试项目特定模块
    print('测试项目模块...')
    import sys
    sys.path.insert(0, '.')
    from megatron.tokenizer import build_tokenizer
    print('✅ megatron.tokenizer可用')
    
    print()
    print('🎉 智能混合环境测试全部通过！')
    
except Exception as e:
    print(f'❌ 测试失败: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🎯 智能混合环境测试完成"
echo "💡 这个测试验证了:"
echo "  1. 容器内PyTorch优先级最高（避免UCC冲突）"
echo "  2. .ext_pkgs中的transformers等包可正常使用"
echo "  3. 危险包被正确排除"
echo "  4. MaskLLM核心功能正常"
