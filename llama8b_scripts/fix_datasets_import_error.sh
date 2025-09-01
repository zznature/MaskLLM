#!/bin/bash

# HuggingFace datasets库导入错误修复脚本
echo "🔧 HuggingFace datasets库导入错误修复"
echo "============================================================"

echo "❌ 问题分析:"
echo "  - 错误类型: AttributeError: module 'datasets' has no attribute 'load_dataset'"
echo "  - 影响位置: tasks/pruning/datasets.py:227"
echo "  - 根本原因: HuggingFace datasets库导入问题"
echo ""

echo "✅ 进展确认:"
echo "  - 模型配置修复成功 (所有MLP层sparse masks已添加)"
echo "  - C4本地数据集加载成功 (356317 samples)"
echo "  - 问题出现在WikiText2在线数据集下载"
echo ""

echo "🔍 诊断datasets库状态:"
echo "----------------------------------------"

# 检查datasets库安装状态
python3 -c "
import sys
print(f'Python版本: {sys.version}')
print()

try:
    import datasets
    print(f'✅ datasets库已导入')
    print(f'  版本: {datasets.__version__ if hasattr(datasets, \"__version__\") else \"未知\"}')
    print(f'  路径: {datasets.__file__}')
    print(f'  属性检查:')
    
    # 检查关键属性
    attrs = ['load_dataset', 'Dataset', 'DatasetDict']
    for attr in attrs:
        if hasattr(datasets, attr):
            print(f'    ✅ {attr}: 存在')
        else:
            print(f'    ❌ {attr}: 缺失')
            
except ImportError as e:
    print(f'❌ datasets库导入失败: {e}')
except Exception as e:
    print(f'❌ datasets库检查出错: {e}')

print()

# 检查可能的冲突模块
try:
    import dataset  # 可能的冲突模块
    print(f'⚠️ 发现可能冲突的dataset模块: {dataset.__file__}')
except ImportError:
    print('✅ 没有发现冲突的dataset模块')
"

echo ""
echo "🔧 修复方案:"
echo "----------------------------------------"

echo "方案1: 检查并重新安装datasets库"
echo "pip3 install --upgrade datasets"
echo ""

echo "方案2: 使用本地数据集 (推荐)"
echo "修改脚本配置，跳过在线数据集下载，仅使用已有的C4数据集"
echo ""

echo "方案3: 修复导入问题"
echo "检查是否有模块名冲突，确保正确导入HuggingFace datasets"
echo ""

echo "🎯 推荐解决方案:"
echo "----------------------------------------"
echo "由于稀疏化主要使用C4数据集进行校准，而WikiText2仅用于测试评估，"
echo "建议修改脚本配置，暂时跳过WikiText2数据集下载。"
echo ""

# 检查当前任务配置
echo "📋 当前任务配置检查:"
echo "  任务类型: \$TASK = wikitext"
echo "  数据路径: --data-path \"None\""
echo "  影响: 仅影响测试数据集，不影响稀疏化校准过程"
echo ""

echo "💡 立即修复建议:"
echo "============================================================"
cat << 'EOF'
# 选项1: 重新安装datasets库
pip3 install --force-reinstall datasets

# 选项2: 暂时禁用在线数据集
# 修改tasks/pruning/datasets.py，添加异常处理或使用本地数据

# 选项3: 使用不同的任务配置
# 在脚本中将 export TASK='wikitext' 改为其他配置
EOF

echo ""
echo "⚠️ 重要提示:"
echo "  - C4数据集已成功加载，稀疏化校准可以正常进行"
echo "  - WikiText2主要用于后续评估，可以稍后修复"
echo "  - 建议先完成稀疏化过程，再处理评估数据集问题"
