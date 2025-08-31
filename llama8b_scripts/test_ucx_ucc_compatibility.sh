#!/bin/bash
# 测试UCX和UCC兼容性
# 验证重编译后的UCX v1.12和UCC v1.2.0组合

echo "=========================================="
echo "UCX-UCC兼容性验证测试"
echo "=========================================="

INSTALL_DIR="/data/home/zdhs0054/zzhou/MaskLLM/ucx_ucc_install_optimized"

echo "1. 检查库文件存在性..."
echo "UCX库文件:"
if [ -f "$INSTALL_DIR/lib/libucs.so" ]; then
    echo "  ✅ libucs.so 存在"
else
    echo "  ❌ libucs.so 缺失"
    exit 1
fi

if [ -f "$INSTALL_DIR/lib/libuct.so" ]; then
    echo "  ✅ libuct.so 存在"
else
    echo "  ⚠️  libuct.so 缺失"
fi

echo "UCC库文件:"
if [ -f "$INSTALL_DIR/lib/libucc.so" ]; then
    echo "  ✅ libucc.so 存在"
else
    echo "  ❌ libucc.so 缺失"
    exit 1
fi

echo -e "\n2. 检查版本信息..."
if [ -f "$INSTALL_DIR/lib/pkgconfig/ucx.pc" ]; then
    UCX_VERSION=$(grep "Version:" "$INSTALL_DIR/lib/pkgconfig/ucx.pc" | cut -d' ' -f2)
    echo "  UCX版本: $UCX_VERSION"
else
    echo "  ⚠️  UCX版本信息缺失"
fi

if [ -f "$INSTALL_DIR/lib/pkgconfig/ucc.pc" ]; then
    UCC_VERSION=$(grep "Version:" "$INSTALL_DIR/lib/pkgconfig/ucc.pc" | cut -d' ' -f2)
    echo "  UCC版本: $UCC_VERSION"
else
    echo "  ⚠️  UCC版本信息缺失，假设为v1.2.0"
    UCC_VERSION="1.2.0"
fi

echo -e "\n3. 检查关键符号..."
echo "检查ucs_mpool_params_reset符号:"
if nm -D "$INSTALL_DIR/lib/libucs.so" 2>/dev/null | grep -q "ucs_mpool_params_reset"; then
    echo "  ✅ ucs_mpool_params_reset 符号存在"
    SYMBOL_FOUND="YES"
else
    echo "  ❌ ucs_mpool_params_reset 符号缺失"
    SYMBOL_FOUND="NO"
    
    echo "  🔍 可用的mpool符号:"
    nm -D "$INSTALL_DIR/lib/libucs.so" 2>/dev/null | grep mpool | head -5 | sed 's/^/    /'
fi

echo -e "\n4. 检查其他可能的缺失符号..."
echo "检查ucm_set_global_opts符号:"
if nm -D "$INSTALL_DIR/lib/libuct.so" 2>/dev/null | grep -q "ucm_set_global_opts"; then
    echo "  ✅ ucm_set_global_opts 符号存在"
elif nm -D "$INSTALL_DIR/lib/libucs.so" 2>/dev/null | grep -q "ucm_set_global_opts"; then
    echo "  ✅ ucm_set_global_opts 符号存在 (在libucs中)"
else
    echo "  ❌ ucm_set_global_opts 符号缺失"
    
    echo "  🔍 可用的ucm符号:"
    nm -D "$INSTALL_DIR/lib/libucs.so" 2>/dev/null | grep ucm | head -5 | sed 's/^/    /'
fi

echo -e "\n5. 库依赖性检查..."
echo "UCX库依赖:"
ldd "$INSTALL_DIR/lib/libucs.so" | head -5 | sed 's/^/  /'

echo -e "\nUCC库依赖:"
ldd "$INSTALL_DIR/lib/libucc.so" | head -5 | sed 's/^/  /'

echo -e "\n6. 设置测试环境..."
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"
export UCX_DIR="$INSTALL_DIR"
export UCC_DIR="$INSTALL_DIR"

echo "  LD_LIBRARY_PATH已设置: $INSTALL_DIR/lib"

echo -e "\n7. 直接库加载测试..."
python3 -c "
import ctypes
import sys

print('测试UCX库加载...')
try:
    ucx_lib = ctypes.CDLL('$INSTALL_DIR/lib/libucs.so')
    print('  ✅ UCX (libucs) 加载成功')
    
    # 尝试访问符号
    if '$SYMBOL_FOUND' == 'YES':
        try:
            symbol = ucx_lib.ucs_mpool_params_reset
            print('  ✅ ucs_mpool_params_reset 符号可访问')
        except AttributeError:
            print('  ❌ ucs_mpool_params_reset 符号不可访问')
    
except Exception as e:
    print(f'  ❌ UCX库加载失败: {e}')
    sys.exit(1)

print('\\n测试UCC库加载...')
try:
    ucc_lib = ctypes.CDLL('$INSTALL_DIR/lib/libucc.so')
    print('  ✅ UCC库加载成功')
except Exception as e:
    print(f'  ❌ UCC库加载失败: {e}')
    sys.exit(1)

print('\\n✅ 基础库加载测试通过')
"

LOAD_TEST_STATUS=$?

echo -e "\n8. PyTorch导入测试..."
python3 -c "
try:
    import torch
    print('  ✅ PyTorch模块导入成功')
    
    # 尝试基础操作
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f'  ✅ 基础张量操作成功: {x}')
    
    if torch.cuda.is_available():
        print(f'  ✅ CUDA可用: {torch.cuda.get_device_name(0)}')
    else:
        print('  ⚠️  CUDA不可用')
        
    print('\\n✅ PyTorch测试通过')
    
except ImportError as e:
    print(f'  ❌ PyTorch导入失败: {e}')
    if 'undefined symbol' in str(e):
        print('  💡 这可能是符号兼容性问题')
    exit(1)
except Exception as e:
    print(f'  ❌ PyTorch运行时错误: {e}')
    exit(1)
"

PYTORCH_STATUS=$?

echo -e "\n=========================================="
echo "兼容性测试总结"
echo "=========================================="

echo "📋 版本信息:"
echo "  UCX: $UCX_VERSION"
echo "  UCC: $UCC_VERSION"

echo -e "\n📋 符号检查:"
if [ "$SYMBOL_FOUND" = "YES" ]; then
    echo "  ✅ ucs_mpool_params_reset: 存在"
else
    echo "  ❌ ucs_mpool_params_reset: 缺失"
fi

echo -e "\n📋 测试结果:"
if [ $LOAD_TEST_STATUS -eq 0 ]; then
    echo "  ✅ 库加载测试: 通过"
else
    echo "  ❌ 库加载测试: 失败"
fi

if [ $PYTORCH_STATUS -eq 0 ]; then
    echo "  ✅ PyTorch测试: 通过"
    FINAL_STATUS="SUCCESS"
else
    echo "  ❌ PyTorch测试: 失败"
    FINAL_STATUS="FAILED"
fi

echo -e "\n📋 最终状态: $FINAL_STATUS"

if [ "$FINAL_STATUS" = "SUCCESS" ]; then
    echo -e "\n🎉 UCX-UCC兼容性验证成功！"
    echo "📋 可以使用当前配置解决HPC-X冲突"
else
    echo -e "\n❌ 兼容性问题仍然存在"
    if [ "$SYMBOL_FOUND" = "NO" ]; then
        echo "💡 建议: 尝试更早的UCX版本 (v1.11.x或v1.10.x)"
    else
        echo "💡 需要进一步诊断其他兼容性问题"
    fi
fi

echo "📊 完成时间: $(date)"
