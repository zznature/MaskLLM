#!/bin/bash
# 重新编译兼容版本的UCX
# 目标：使用包含ucs_mpool_params_reset符号的UCX版本

echo "=========================================="
echo "重新编译兼容版本的UCX"
echo "=========================================="

# 设置路径
MASKLLM_ROOT="/data/home/zdhs0054/zzhou/MaskLLM"
BUILD_DIR="$MASKLLM_ROOT/ucx_compatible_build"
INSTALL_DIR="$MASKLLM_ROOT/ucx_ucc_install_optimized"

echo "1. 清理之前的编译..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 备份当前的UCX
if [ -d "$INSTALL_DIR" ]; then
    echo "备份当前UCX安装..."
    mv "$INSTALL_DIR" "${INSTALL_DIR}_backup_$(date +%s)"
fi
mkdir -p "$INSTALL_DIR"

cd "$BUILD_DIR" || exit 1

echo -e "\n2. 下载兼容版本的UCX..."

# 尝试多个可能包含ucs_mpool_params_reset的UCX版本
UCX_VERSIONS=(
    "1.12.1"
    "1.11.2" 
    "1.11.1"
    "1.10.1"
)

# 多个下载源
DOWNLOAD_SOURCES=(
    "https://github.com/openucx/ucx/releases/download"
    "https://mirror.ghproxy.com/https://github.com/openucx/ucx/releases/download"
    "https://ghproxy.net/https://github.com/openucx/ucx/releases/download"
)

UCX_DOWNLOADED=""
UCX_VERSION=""

for version in "${UCX_VERSIONS[@]}"; do
    echo "尝试下载UCX v$version..."
    
    for source in "${DOWNLOAD_SOURCES[@]}"; do
        echo "  从源下载: $source"
        download_url="$source/v$version/ucx-$version.tar.gz"
        
        if command -v wget >/dev/null 2>&1; then
            if timeout 60s wget --tries=2 --timeout=30 "$download_url" 2>/dev/null; then
                if [ -f "ucx-$version.tar.gz" ] && [ -s "ucx-$version.tar.gz" ]; then
                    echo "✅ 成功下载 UCX v$version"
                    UCX_DOWNLOADED="ucx-$version.tar.gz"
                    UCX_VERSION="$version"
                    break 2
                else
                    rm -f "ucx-$version.tar.gz"
                fi
            fi
        elif command -v curl >/dev/null 2>&1; then
            if timeout 60s curl -L --fail --connect-timeout 30 --max-time 60 "$download_url" -o "ucx-$version.tar.gz" 2>/dev/null; then
                if [ -f "ucx-$version.tar.gz" ] && [ -s "ucx-$version.tar.gz" ]; then
                    echo "✅ 成功下载 UCX v$version"
                    UCX_DOWNLOADED="ucx-$version.tar.gz"
                    UCX_VERSION="$version"
                    break 2
                else
                    rm -f "ucx-$version.tar.gz"
                fi
            fi
        fi
        
        echo "    ❌ 下载失败，尝试下一个源..."
    done
    
    if [ -n "$UCX_DOWNLOADED" ]; then
        break
    fi
    echo "  ❌ UCX v$version 所有源都失败，尝试下一个版本..."
done

if [ -z "$UCX_DOWNLOADED" ]; then
    echo "❌ 所有UCX版本下载失败"
    exit 1
fi

echo "✅ 使用 UCX v$UCX_VERSION 进行编译"

echo -e "\n3. 解压UCX源码..."
tar -xzf "$UCX_DOWNLOADED"

if [ ! -d "ucx-$UCX_VERSION" ]; then
    echo "❌ UCX源码解压失败"
    exit 1
fi

cd "ucx-$UCX_VERSION" || exit 1

echo -e "\n4. 生成configure脚本..."
if [ ! -f "./configure" ]; then
    echo "运行autogen.sh..."
    ./autogen.sh
    if [ $? -ne 0 ]; then
        echo "❌ autogen.sh失败"
        exit 1
    fi
fi

echo -e "\n5. 配置UCX编译选项（容器优化）..."
./configure \
    --prefix="$INSTALL_DIR" \
    --enable-shared \
    --disable-static \
    --enable-mt \
    --with-cuda=/usr/local/cuda \
    --enable-optimizations \
    --disable-logging \
    --disable-debug \
    --disable-assertions \
    --disable-params-check \
    --without-xpmem \
    --without-knem \
    --without-gdrcopy \
    --disable-numa \
    --without-java \
    --without-go

if [ $? -ne 0 ]; then
    echo "❌ UCX配置失败"
    exit 1
fi

echo "✅ UCX配置成功"

echo -e "\n6. 编译UCX..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ UCX编译失败"
    exit 1
fi

echo "✅ UCX编译成功"

echo -e "\n7. 安装UCX..."
make install

if [ $? -ne 0 ]; then
    echo "❌ UCX安装失败"
    exit 1
fi

echo "✅ UCX安装成功"

echo -e "\n8. 验证关键符号..."
if [ -f "$INSTALL_DIR/lib/libucs.so" ]; then
    echo "检查ucs_mpool_params_reset符号..."
    if nm -D "$INSTALL_DIR/lib/libucs.so" 2>/dev/null | grep -q "ucs_mpool_params_reset"; then
        echo "✅ 找到ucs_mpool_params_reset符号！"
        SYMBOL_FOUND="YES"
    else
        echo "❌ 仍然没有找到ucs_mpool_params_reset符号"
        SYMBOL_FOUND="NO"
        
        echo "🔍 显示可用的mpool相关符号:"
        nm -D "$INSTALL_DIR/lib/libucs.so" 2>/dev/null | grep mpool | head -10
    fi
else
    echo "❌ UCX库文件不存在"
    exit 1
fi

echo -e "\n9. 测试库加载..."
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"

python3 -c "
import ctypes
import sys

try:
    ucx_lib = ctypes.CDLL('$INSTALL_DIR/lib/libucs.so')
    print('✅ UCX库加载成功')
    
    if '$SYMBOL_FOUND' == 'YES':
        try:
            symbol = ucx_lib.ucs_mpool_params_reset
            print('✅ ucs_mpool_params_reset符号访问成功')
        except AttributeError as e:
            print('❌ 符号访问失败:', e)
    
    print('STATUS: UCX_COMPATIBLE')
    
except Exception as e:
    print('❌ UCX库测试失败:', e)
    print('STATUS: UCX_FAILED')
    sys.exit(1)
"

UCX_STATUS=$?

# 清理编译目录
echo -e "\n10. 清理编译目录..."
cd "$MASKLLM_ROOT" || exit 1
rm -rf "$BUILD_DIR"

echo -e "\n=========================================="
echo "UCX重新编译完成"
echo "=========================================="

if [ "$SYMBOL_FOUND" = "YES" ] && [ $UCX_STATUS -eq 0 ]; then
    echo "🎉 成功！UCX v$UCX_VERSION包含所需符号"
    echo "📋 安装路径: $INSTALL_DIR"
    echo "📋 版本: UCX v$UCX_VERSION"
    echo "📋 符号: ucs_mpool_params_reset ✅"
    echo ""
    echo "📋 下一步: 重新编译UCC"
    echo "   bash run_maskllm_native.sh llama8b_scripts/build_ucc_local.sh"
else
    echo "⚠️  UCX编译成功但符号仍不匹配"
    echo "📋 版本: UCX v$UCX_VERSION"
    echo "📋 可能需要尝试更早的UCX版本"
fi

echo "📊 完成时间: $(date)"
