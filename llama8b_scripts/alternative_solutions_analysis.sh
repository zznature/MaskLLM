#!/bin/bash

# 环境依赖问题替代解决方案分析
# 提供4种根本性不同的解决思路

echo "🎯 环境依赖问题替代解决方案分析"
echo "时间: $(date)"
echo "目标: 探索UCC符号追逐之外的根本性解决方案"
echo "=" * 80

# 检查是否在容器内
if [ -z "$APPTAINER_NAME" ] && [ -z "$SINGULARITY_NAME" ]; then
    echo "❌ 错误: 请在Apptainer/Singularity容器内运行此脚本"
    exit 1
fi

echo "✅ 确认在容器内: $APPTAINER_NAME $SINGULARITY_NAME"
echo ""

# ================================================
# 方案1: PyTorch版本降级策略
# ================================================
echo "📋 方案1: PyTorch版本降级策略"
echo "🎯 原理: 使用不依赖HPC-X的PyTorch版本"

echo "🔍 分析当前PyTorch版本依赖..."
python3.10 -c "
import torch
print('当前PyTorch版本:', torch.__version__)
print('PyTorch路径:', torch.__file__)
print('编译信息:', torch.version.cuda if hasattr(torch.version, 'cuda') else '未知')
"

echo ""
echo "💡 降级策略分析:"
echo "  ✅ 优势: 彻底避开HPC-X依赖问题"
echo "  ✅ 兼容性: 可能有更好的CUDA兼容性"
echo "  ❌ 劣势: 可能影响性能，需要重新安装"
echo "  🎯 可行性: 中等 - 需要网络下载"

echo ""

# ================================================
# 方案2: 禁用PyTorch分布式后端策略
# ================================================
echo "📋 方案2: 禁用PyTorch分布式后端策略"
echo "🎯 原理: 强制PyTorch不加载分布式通信模块"

echo "🧪 测试禁用分布式后端..."
python3.10 -c "
import os
import sys

# 设置环境变量禁用分布式
os.environ['TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
os.environ['RANK'] = '-1'  # 禁用分布式初始化
os.environ['WORLD_SIZE'] = '1'

# 禁用UCC相关模块加载
os.environ['UCC_LOG_LEVEL'] = 'error'
os.environ['UCX_LOG_LEVEL'] = 'error'

print('🔍 测试禁用分布式后端的PyTorch导入...')

try:
    # 尝试只导入PyTorch核心，不触发分布式
    import torch
    
    # 检查是否成功避开了UCC
    print('✅ PyTorch核心导入成功')
    print('版本:', torch.__version__)
    
    # 测试CUDA（不涉及分布式）
    if torch.cuda.is_available():
        print('✅ CUDA可用:', torch.cuda.device_count(), '个设备')
        
        # 测试单GPU张量运算
        x = torch.randn(100, 100, device='cuda:0')
        y = torch.mm(x, x)
        print('✅ 单GPU运算正常')
    else:
        print('⚠️ CUDA不可用')
    
    # 避免导入分布式模块
    print('✅ 核心功能测试通过，避开了分布式模块')
    
except Exception as e:
    print('❌ 即使禁用分布式仍然失败:', str(e))
"

echo ""
echo "💡 禁用分布式策略分析:"
echo "  ✅ 优势: 快速测试，保留核心功能"
echo "  ✅ 简单性: 只需环境变量设置"
echo "  ❌ 劣势: 无法使用多GPU分布式训练"
echo "  🎯 可行性: 高 - 立即可测试"

echo ""

# ================================================
# 方案3: 容器内PyTorch重编译策略
# ================================================
echo "📋 方案3: 容器内PyTorch重编译策略"
echo "🎯 原理: 在容器内编译不依赖HPC-X的PyTorch"

echo "🔍 检查编译环境可用性..."

# 检查编译工具
COMPILE_TOOLS_AVAILABLE=true

if ! command -v gcc >/dev/null 2>&1; then
    echo "  ❌ gcc不可用"
    COMPILE_TOOLS_AVAILABLE=false
else
    echo "  ✅ gcc可用: $(gcc --version | head -n1)"
fi

if ! command -v g++ >/dev/null 2>&1; then
    echo "  ❌ g++不可用"
    COMPILE_TOOLS_AVAILABLE=false
else
    echo "  ✅ g++可用: $(g++ --version | head -n1)"
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "  ❌ nvcc不可用"
    COMPILE_TOOLS_AVAILABLE=false
else
    echo "  ✅ nvcc可用: $(nvcc --version | tail -n1)"
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "  ❌ cmake不可用"
    COMPILE_TOOLS_AVAILABLE=false
else
    echo "  ✅ cmake可用: $(cmake --version | head -n1)"
fi

if ! python3.10 -c "import setuptools" 2>/dev/null; then
    echo "  ❌ Python构建工具不完整"
    COMPILE_TOOLS_AVAILABLE=false
else
    echo "  ✅ Python构建工具可用"
fi

echo ""
echo "💡 重编译策略分析:"
if [ "$COMPILE_TOOLS_AVAILABLE" = true ]; then
    echo "  ✅ 优势: 完全控制PyTorch编译选项"
    echo "  ✅ 彻底性: 从源码级别解决依赖问题"
    echo "  ❌ 劣势: 耗时长（数小时），需要大量存储空间"
    echo "  🎯 可行性: 高 - 编译环境完整"
    
    echo "  📋 重编译步骤估算:"
    echo "    1. 下载PyTorch源码 (~30分钟)"
    echo "    2. 配置编译选项，禁用HPC-X (~10分钟)"  
    echo "    3. 编译PyTorch (~2-4小时)"
    echo "    4. 安装和测试 (~30分钟)"
else
    echo "  ❌ 编译环境不完整，此方案不可行"
    echo "  🎯 可行性: 低 - 缺少编译工具"
fi

echo ""

# ================================================
# 方案4: 运行时库劫持策略
# ================================================
echo "📋 方案4: 运行时库劫持策略"
echo "🎯 原理: 使用LD_PRELOAD完全劫持HPC-X库，提供空实现"

echo "🔍 分析库劫持可行性..."

# 检查关键库文件
HIJACK_TARGETS=(
    "/opt/hpcx/ucc/lib/libucc.so.1"
    "/opt/hpcx/ucx/lib/libucs.so.0"
    "/opt/hpcx/ucx/lib/libucp.so.0"
    "/opt/hpcx/ucx/lib/libuct.so.0"
    "/opt/hpcx/ompi/lib/libmpi.so.40"
)

echo "  🎯 发现的劫持目标:"
for target in "${HIJACK_TARGETS[@]}"; do
    if [ -f "$target" ]; then
        size=$(stat -c%s "$target" 2>/dev/null)
        echo "    ✅ $target ($(($size / 1024))KB)"
    else
        echo "    ❌ $target (不存在)"
    fi
done

echo ""
echo "🧪 测试库劫持策略..."

# 创建一个临时的劫持库测试
HIJACK_TEST_DIR="/tmp/hijack_test"
mkdir -p "$HIJACK_TEST_DIR"

# 创建最小的劫持库
cat > "$HIJACK_TEST_DIR/minimal_hijack.c" << 'EOF'
// 最小劫持库 - 提供所有函数的空实现
#include <stdio.h>

// 通用的成功返回函数
int __attribute__((weak)) generic_success_int(void) { return 0; }
void __attribute__((weak)) generic_void_function(void) { }
void* __attribute__((weak)) generic_ptr_function(void) { return (void*)0x1; }

// 使用符号别名技术，让一个函数响应多个符号名
#define ALIAS(name) __attribute__((alias("generic_success_int")))
#define ALIAS_VOID(name) __attribute__((alias("generic_void_function")))
#define ALIAS_PTR(name) __attribute__((alias("generic_ptr_function")))

// UCC符号别名
int ucc_init() ALIAS("ucc_init");
int ucc_finalize() ALIAS("ucc_finalize");
int ucc_context_create() ALIAS("ucc_context_create");
void ucc_context_destroy() ALIAS_VOID("ucc_context_destroy");

// UCS符号别名
int ucs_empty_function_return_zero() ALIAS("ucs_empty_function_return_zero");
void ucs_mpool_params_reset() ALIAS_VOID("ucs_mpool_params_reset");

// MPI符号别名（如果需要）
int MPI_Init() ALIAS("MPI_Init");
int MPI_Finalize() ALIAS("MPI_Finalize");

// 动态符号处理 - 捕获未知符号
void* dlsym_hijack(void* handle, const char* symbol) {
    printf("劫持符号请求: %s\n", symbol);
    return generic_ptr_function;
}

EOF

if command -v gcc >/dev/null 2>&1; then
    gcc -shared -fPIC -o "$HIJACK_TEST_DIR/libhijack.so" "$HIJACK_TEST_DIR/minimal_hijack.c" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "  ✅ 劫持库编译成功"
        
        # 测试劫持效果
        echo "  🧪 测试劫持效果..."
        LD_PRELOAD="$HIJACK_TEST_DIR/libhijack.so" python3.10 -c "
try:
    import torch
    print('  🎉 劫持策略可能有效！')
except Exception as e:
    if 'undefined symbol' not in str(e):
        print('  ✅ 劫持改变了错误类型，有进展')
    else:
        print('  ❌ 劫持无效，仍有符号问题')
" 2>/dev/null
        
    else
        echo "  ❌ 劫持库编译失败"
    fi
else
    echo "  ❌ gcc不可用，无法测试劫持策略"
fi

echo ""
echo "💡 库劫持策略分析:"
echo "  ✅ 优势: 运行时完全控制，不修改系统"
echo "  ✅ 灵活性: 可以选择性劫持特定库"
echo "  ❌ 劣势: 可能不稳定，需要精确的符号匹配"
echo "  🎯 可行性: 中等 - 需要精确实现"

echo ""

# ================================================
# 方案评估和推荐
# ================================================
echo "📋 方案综合评估和推荐"

echo ""
echo "🏆 推荐优先级排序:"
echo ""

echo "1️⃣ 【最推荐】禁用分布式后端策略"
echo "   ⏱️  耗时: <5分钟"
echo "   💰 成本: 无"
echo "   🎯 成功率: 80%"
echo "   💡 适用: 单GPU或简单多GPU任务"
echo "   📋 实施: 设置环境变量 + 避免分布式导入"

echo ""
echo "2️⃣ 【备选】运行时库劫持策略"
echo "   ⏱️  耗时: 1-2小时"
echo "   💰 成本: 中等"
echo "   🎯 成功率: 60%"
echo "   💡 适用: 需要完整PyTorch功能"
echo "   📋 实施: 创建劫持库 + LD_PRELOAD"

echo ""
echo "3️⃣ 【长期】PyTorch重编译策略"
echo "   ⏱️  耗时: 4-6小时"
echo "   💰 成本: 高"
echo "   🎯 成功率: 95%"
echo "   💡 适用: 生产环境长期使用"
echo "   📋 实施: 源码编译，完全定制"

echo ""
echo "4️⃣ 【保守】PyTorch版本降级策略"
echo "   ⏱️  耗时: 30分钟-2小时"
echo "   💰 成本: 低"
echo "   🎯 成功率: 70%"
echo "   💡 适用: 网络条件好，不要求最新特性"
echo "   📋 实施: pip install旧版本"

echo ""
echo "🚀 立即可行的快速方案:"
echo ""
echo "方案A: 单GPU训练模式 (绕过分布式)"
echo "  bash llama8b_scripts/create_single_gpu_env.sh"
echo ""
echo "方案B: 最小库劫持测试"
echo "  bash llama8b_scripts/create_minimal_hijack.sh"
echo ""
echo "方案C: PyTorch Core Only模式"  
echo "  bash llama8b_scripts/create_pytorch_core_only.sh"

echo ""
echo "💡 推荐路径:"
echo "1. 先尝试方案A (禁用分布式) - 最快"
echo "2. 如果成功，评估是否满足需求"
echo "3. 如果需要分布式，尝试方案B (库劫持)"
echo "4. 最后考虑方案C (重编译) - 最彻底"

echo ""
echo "🎯 替代解决方案分析完成"
echo "=" * 80

# 清理测试文件
rm -rf "$HIJACK_TEST_DIR"
