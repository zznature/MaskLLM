#!/bin/bash
# 使用本地UCC源码编译脚本
# 推荐使用UCC v1.2.0，与UCX v1.13.1最佳匹配

echo "=========================================="
echo "本地UCC源码编译（UCX v1.13.1 + UCC v1.2.0）"
echo "=========================================="

# 使用与主脚本相同的路径
MASKLLM_ROOT="/data/home/zdhs0054/zzhou/MaskLLM"
BUILD_DIR="$MASKLLM_ROOT/ucx_ucc_build_optimized"
INSTALL_DIR="$MASKLLM_ROOT/ucx_ucc_install_optimized"

echo "1. 检查UCX编译状态..."
if [ -f "$INSTALL_DIR/lib/libucs.so" ]; then
    echo "✅ UCX已成功编译"
    UCX_VERSION=$(cat "$INSTALL_DIR/lib/pkgconfig/ucx.pc" | grep Version | cut -d' ' -f2)
    echo "   UCX版本: $UCX_VERSION"
else
    echo "❌ UCX未完成，请先运行完整脚本"
    exit 1
fi

cd "$BUILD_DIR" || exit 1

echo -e "\n2. 检查本地UCC源码文件..."

# 检查可用的UCC版本
AVAILABLE_VERSIONS=()
for version in "1.3.0" "1.2.0" "1.1.0"; do
    if [ -f "ucc-$version.tar.gz" ] && [ -s "ucc-$version.tar.gz" ]; then
        AVAILABLE_VERSIONS+=("$version")
        echo "✅ 发现UCC v$version: $(ls -lh ucc-$version.tar.gz | awk '{print $5}')"
    fi
done

if [ ${#AVAILABLE_VERSIONS[@]} -eq 0 ]; then
    echo "❌ 未找到可用的UCC源码文件"
    echo "请确保以下文件存在:"
    echo "  - ucc-1.2.0.tar.gz (推荐)"
    echo "  - ucc-1.1.0.tar.gz"
    echo "  - ucc-1.3.0.tar.gz"
    exit 1
fi

# 选择最佳版本（优先级：1.2.0 > 1.1.0 > 1.3.0）
UCC_VERSION=""
if [[ " ${AVAILABLE_VERSIONS[@]} " =~ " 1.2.0 " ]]; then
    UCC_VERSION="1.2.0"
    echo "🎯 选择UCC v1.2.0 (推荐: 与UCX v1.13.1最佳兼容)"
elif [[ " ${AVAILABLE_VERSIONS[@]} " =~ " 1.1.0 " ]]; then
    UCC_VERSION="1.1.0"
    echo "🎯 选择UCC v1.1.0 (稳定版本)"
elif [[ " ${AVAILABLE_VERSIONS[@]} " =~ " 1.3.0 " ]]; then
    UCC_VERSION="1.3.0"
    echo "🎯 选择UCC v1.3.0 (较新版本，可能有兼容性风险)"
fi

echo -e "\n3. 解压UCC源码..."
# 清理之前的解压目录
rm -rf ucc-*/ 

tar -xzf "ucc-$UCC_VERSION.tar.gz"

if [ ! -d "ucc-$UCC_VERSION" ]; then
    echo "❌ UCC源码解压失败"
    exit 1
fi

echo "✅ UCC v$UCC_VERSION 源码解压成功"

echo -e "\n4. 配置编译环境..."
cd "ucc-$UCC_VERSION" || exit 1

# 设置环境以找到刚编译的UCX
export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"

echo "当前环境配置:"
echo "  UCX路径: $INSTALL_DIR"
echo "  PKG_CONFIG_PATH: $(echo $PKG_CONFIG_PATH | cut -d: -f1-2)"

echo -e "\n5. 生成configure脚本..."
# 检查是否需要运行autogen
if [ ! -f "./configure" ]; then
    echo "configure脚本不存在，运行autogen.sh..."
    
    # 检查是否有autogen.sh
    if [ -f "./autogen.sh" ]; then
        echo "运行autogen.sh生成configure..."
        ./autogen.sh
        if [ $? -ne 0 ]; then
            echo "❌ autogen.sh失败"
            exit 1
        fi
    elif [ -f "./bootstrap" ]; then
        echo "运行bootstrap生成configure..."
        ./bootstrap
        if [ $? -ne 0 ]; then
            echo "❌ bootstrap失败"
            exit 1
        fi
    else
        echo "❌ 找不到autogen.sh或bootstrap脚本"
        echo "📋 尝试手动生成configure:"
        if command -v autoreconf >/dev/null 2>&1; then
            echo "运行autoreconf -fiv..."
            autoreconf -fiv
            if [ $? -ne 0 ]; then
                echo "❌ autoreconf失败"
                exit 1
            fi
        else
            echo "❌ 需要安装autotools (autoconf, automake, libtool)"
            exit 1
        fi
    fi
    
    if [ ! -f "./configure" ]; then
        echo "❌ configure脚本生成失败"
        exit 1
    fi
    echo "✅ configure脚本生成成功"
else
    echo "✅ configure脚本已存在"
fi

echo -e "\n6. 配置UCC编译选项..."
./configure \
    --prefix="$INSTALL_DIR" \
    --with-ucx="$INSTALL_DIR" \
    --enable-shared \
    --disable-static \
    --disable-debug \
    --disable-logging \
    --disable-assertions

if [ $? -ne 0 ]; then
    echo "❌ UCC配置失败"
    echo "📋 可能的问题："
    echo "   1. UCX路径不正确"
    echo "   2. 缺少依赖库"
    echo "   3. 版本不兼容"
    exit 1
fi

echo "✅ UCC配置成功"

echo -e "\n7. 编译UCC..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ UCC编译失败"
    echo "📋 编译失败可能原因："
    echo "   1. 编译器错误"
    echo "   2. 依赖库问题"
    echo "   3. 内存不足"
    exit 1
fi

echo "✅ UCC编译成功"

echo -e "\n8. 安装UCC..."
make install

if [ $? -ne 0 ]; then
    echo "❌ UCC安装失败"
    exit 1
fi

echo "✅ UCC安装成功"

# 验证安装结果
echo -e "\n9. 验证安装结果..."

# 检查关键库文件
CRITICAL_LIBS=(
    "$INSTALL_DIR/lib/libucs.so"
    "$INSTALL_DIR/lib/libucc.so"
)

echo "检查关键库文件:"
for lib in "${CRITICAL_LIBS[@]}"; do
    if [ -f "$lib" ]; then
        echo "  ✅ $lib"
    else
        echo "  ❌ $lib (缺失)"
        exit 1
    fi
done

# 验证符号
echo -e "\n验证关键符号:"
if nm -D "$INSTALL_DIR/lib/libucs.so" | grep -q "ucs_mpool_params_reset"; then
    echo "  ✅ ucs_mpool_params_reset 符号存在"
else
    echo "  ❌ ucs_mpool_params_reset 符号缺失"
    exit 1
fi

# 测试库加载
echo -e "\n10. 测试库加载..."
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"

python3 -c "
import ctypes
import sys

try:
    print('测试UCX库加载...')
    ucx_lib = ctypes.CDLL('$INSTALL_DIR/lib/libucs.so')
    ucx_symbol = ucx_lib.ucs_mpool_params_reset
    print('✅ UCX库加载和符号访问成功')
    
    print('测试UCC库加载...')
    ucc_lib = ctypes.CDLL('$INSTALL_DIR/lib/libucc.so')
    print('✅ UCC库加载成功')
    
    print('STATUS: LIBS_COMPATIBLE')
    
except Exception as e:
    print('❌ 库测试失败:', e)
    print('STATUS: LIBS_FAILED')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ 库兼容性测试通过"
else
    echo "❌ 库兼容性测试失败"
    exit 1
fi

# 测试PyTorch兼容性
echo -e "\n11. 测试PyTorch兼容性..."

python3 -c "
import sys
import os

print('=== PyTorch兼容性测试 ===')
print('当前LD_LIBRARY_PATH前3个路径:')
for i, path in enumerate(os.environ.get('LD_LIBRARY_PATH', '').split(':')[:3]):
    print(f'  {i+1}. {path}')

print('\n尝试导入PyTorch...')
try:
    import torch
    print('✅ PyTorch导入成功！')
    print('PyTorch版本:', torch.__version__)
    
    # 测试基本功能
    x = torch.randn(2, 3)
    print('✅ 张量创建成功')
    
    if torch.cuda.is_available():
        print('✅ CUDA可用，设备数:', torch.cuda.device_count())
    
    print('STATUS: PYTORCH_SUCCESS')
    
except ImportError as e:
    error_msg = str(e)
    print('❌ PyTorch导入失败:', error_msg)
    if 'undefined symbol: ucs_mpool_params_reset' in error_msg:
        print('🔍 仍然存在UCC冲突')
        print('STATUS: UCC_CONFLICT_PERSISTS')
    else:
        print('🔍 其他导入问题')
        print('STATUS: OTHER_IMPORT_ERROR')
        
except Exception as e:
    print('❌ 其他错误:', e)
    print('STATUS: OTHER_ERROR')
"

PYTORCH_STATUS=$?

# 创建环境配置脚本
echo -e "\n12. 创建环境配置脚本..."

cat > "$MASKLLM_ROOT/llama8b_scripts/set_optimized_rebuilt_env.sh" << EOF
#!/bin/bash
# 优化版重建库环境设置脚本
# UCX v$UCX_VERSION + UCC v$UCC_VERSION

echo "应用优化版重建库环境 (UCX v$UCX_VERSION + UCC v$UCC_VERSION)..."

# 使用重新编译的UCX/UCC库（最高优先级）
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:\$PKG_CONFIG_PATH"

# 清理可能冲突的HPC-X路径
export LD_LIBRARY_PATH=\$(echo \$LD_LIBRARY_PATH | tr ':' '\n' | grep -v "/opt/hpcx" | tr '\n' ':' | sed 's/:*\$//')

# 清理HPC-X环境变量
unset HPCX_DIR HPCX_HOME HPCX_ROOT HPCX_MPI_DIR HPCX_UCX_DIR HPCX_UCC_DIR
unset OMPI_HOME OMPI_ROOT MPI_HOME MPI_ROOT

echo "✅ 优化版重建库环境设置完成"
echo "库路径: $INSTALL_DIR/lib"
echo "UCX版本: $UCX_VERSION"
echo "UCC版本: $UCC_VERSION"
echo "当前LD_LIBRARY_PATH前3个: \$(echo \$LD_LIBRARY_PATH | cut -d: -f1-3)"
EOF

chmod +x "$MASKLLM_ROOT/llama8b_scripts/set_optimized_rebuilt_env.sh"

# 清理编译目录
echo -e "\n13. 清理编译目录..."
cd "$MASKLLM_ROOT" || exit 1
rm -rf "$BUILD_DIR/ucc-$UCC_VERSION"

# 最终结果
echo -e "\n=========================================="
echo "本地UCC编译完成"
echo "=========================================="

if [ $PYTORCH_STATUS -eq 0 ]; then
    echo "🎉 成功！PyTorch现在可以正常工作"
    echo "📋 版本组合: UCX v$UCX_VERSION + UCC v$UCC_VERSION"
    echo "📋 安装路径: $INSTALL_DIR"
    echo "📋 环境脚本: set_optimized_rebuilt_env.sh"
    echo ""
    echo "📋 使用方法:"
    echo "   source $MASKLLM_ROOT/llama8b_scripts/set_optimized_rebuilt_env.sh"
    echo ""
else
    echo "⚠️  编译成功但PyTorch测试未通过"
    echo "📋 版本组合: UCX v$UCX_VERSION + UCC v$UCC_VERSION"
    echo "📋 可能需要进一步调试"
fi

DISK_USAGE=$(du -sh "$INSTALL_DIR" 2>/dev/null | cut -f1)
echo "📊 磁盘占用: 约${DISK_USAGE}永久存储"
echo "📊 完成时间: $(date)"
