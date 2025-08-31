#!/bin/bash
# UCC/UCX调用机制和缓存污染分析脚本

echo "=================================================="
echo "UCC/UCX调用机制和缓存污染深度分析"
echo "=================================================="

echo "1. 当前环境分析"
echo "运行位置: $(pwd)"
echo "用户: $(whoami)"
echo "主机名: $(hostname)"

echo -e "\n2. 检查是否在容器内"
if [ -f /.dockerenv ] || [ -f /singularity ]; then
    echo "✅ 在容器内运行"
    CONTAINER_ENV="YES"
else
    echo "❌ 在主机环境运行"
    CONTAINER_ENV="NO"
fi

echo -e "\n3. HPC-X库挂载检查"
if [ -d /opt/hpcx ]; then
    echo "✅ 发现HPC-X目录: /opt/hpcx"
    echo "HPC-X内容:"
    ls -la /opt/hpcx/ | head -10
    
    if [ -f /opt/hpcx/VERSION ]; then
        echo "HPC-X版本: $(cat /opt/hpcx/VERSION)"
    fi
    
    # 检查UCC/UCX库文件
    echo -e "\nUCC/UCX库文件:"
    find /opt/hpcx -name "*.so*" | grep -E "(ucc|ucx)" | head -10
    
    HPC_X_AVAILABLE="YES"
else
    echo "❌ 未发现HPC-X目录"
    HPC_X_AVAILABLE="NO"
fi

echo -e "\n4. 动态链接器缓存分析"
echo "ldconfig缓存中的UCC/UCX条目:"
ldconfig -p | grep -E "(ucc|ucx)" || echo "缓存中无UCC/UCX条目"

echo -e "\n5. LD_LIBRARY_PATH分析"
echo "当前LD_LIBRARY_PATH:"
if [ -n "$LD_LIBRARY_PATH" ]; then
    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | nl
else
    echo "LD_LIBRARY_PATH未设置"
fi

echo -e "\n6. 全面查找容器内UCC/UCX库"
echo "搜索容器内所有UCC/UCX库文件:"

# 扩展搜索路径，重点关注容器内可能的位置
SEARCH_PATHS=(
    "/usr/lib"
    "/usr/lib/x86_64-linux-gnu"
    "/usr/local/lib" 
    "/lib"
    "/lib/x86_64-linux-gnu"
    "/opt"
    "/data/apps"
    "/usr/local/cuda/lib64"
    "/opt/conda/lib"
    "/usr/local/lib/python3.10/dist-packages"
)

FOUND_UCC_UCX=()
for search_path in "${SEARCH_PATHS[@]}"; do
    if [ -d "$search_path" ]; then
        echo "🔍 搜索 $search_path:"
        found_files=$(find "$search_path" -name "*ucc*" -o -name "*ucx*" 2>/dev/null)
        if [ -n "$found_files" ]; then
            echo "$found_files" | head -5
            # 收集找到的库文件
            while IFS= read -r file; do
                FOUND_UCC_UCX+=("$file")
            done <<< "$found_files"
        else
            echo "  无UCC/UCX文件"
        fi
    fi
done

echo -e "\n📋 容器内发现的UCC/UCX库总结:"
if [ ${#FOUND_UCC_UCX[@]} -eq 0 ]; then
    echo "❌ 容器内未发现任何UCC/UCX库"
    CONTAINER_HAS_UCC_UCX="NO"
else
    echo "✅ 容器内发现 ${#FOUND_UCC_UCX[@]} 个UCC/UCX相关文件:"
    for i in "${!FOUND_UCC_UCX[@]}"; do
        if [ $i -lt 10 ]; then  # 只显示前10个
            echo "  $((i+1)). ${FOUND_UCC_UCX[$i]}"
        fi
    done
    CONTAINER_HAS_UCC_UCX="YES"
fi

echo -e "\n7. 容器内UCC/UCX库详细分析"
if [ "$CONTAINER_HAS_UCC_UCX" = "YES" ]; then
    echo "分析发现的UCC/UCX库文件..."
    
    # 分析每个找到的库文件
    for lib_file in "${FOUND_UCC_UCX[@]}"; do
        if [ -f "$lib_file" ] && [[ "$lib_file" == *.so* ]]; then
            echo -e "\n🔍 分析库文件: $lib_file"
            
            # 检查文件信息
            echo "  文件信息: $(file "$lib_file")"
            
            # 检查符号（如果是UCS库）
            if [[ "$lib_file" == *"ucs"* ]]; then
                echo "  符号检查:"
                if nm -D "$lib_file" 2>/dev/null | grep -q "ucs_mpool_params_reset"; then
                    echo "    ✅ 包含 ucs_mpool_params_reset 符号"
                else
                    echo "    ❌ 不包含 ucs_mpool_params_reset 符号"
                fi
            fi
            
            # 检查依赖关系
            echo "  依赖关系:"
            ldd "$lib_file" 2>/dev/null | head -3 | sed 's/^/    /'
        fi
    done
fi

echo -e "\n8. PyTorch库依赖分析"
echo "尝试分析PyTorch的动态链接依赖:"

# 查找PyTorch安装位置
PYTORCH_LOCATIONS=(
    "/usr/local/lib/python3.10/dist-packages/torch"
    "/opt/conda/lib/python3.10/site-packages/torch"
    "/usr/lib/python3/dist-packages/torch"
)

PYTORCH_FOUND="NO"
for location in "${PYTORCH_LOCATIONS[@]}"; do
    if [ -d "$location" ]; then
        echo "✅ 发现PyTorch安装: $location"
        PYTORCH_FOUND="YES"
        
        # 查找关键的.so文件
        SO_FILES=(
            "$location/lib/libtorch.so"
            "$location/lib/libtorch_cuda.so"
            "$location/lib/libtorch_cpu.so"
            "$location/lib/libtorch_python.so"
        )
        
        echo "PyTorch库文件分析:"
        for so_file in "${SO_FILES[@]}"; do
            if [ -f "$so_file" ]; then
                echo "  📄 分析 $(basename $so_file):"
                # 检查UCC/UCX依赖
                ucc_deps=$(ldd "$so_file" 2>/dev/null | grep -E "(ucc|ucx)")
                if [ -n "$ucc_deps" ]; then
                    echo "    🔗 UCC/UCX依赖:"
                    echo "$ucc_deps" | sed 's/^/      /'
                else
                    echo "    ✅ 无直接UCC/UCX依赖"
                fi
            fi
        done
        break
    fi
done

if [ "$PYTORCH_FOUND" = "NO" ]; then
    echo "❌ 未找到PyTorch安装"
fi

echo -e "\n9. 环境变量污染检查"
echo "HPC-X相关环境变量:"
env | grep -i hpc || echo "无HPC-X环境变量"

echo -e "\nUCC/UCX相关环境变量:"
env | grep -i -E "(ucc|ucx)" || echo "无UCC/UCX环境变量"

echo -e "\n10. 动态链接器配置文件检查"
echo "检查 /etc/ld.so.conf 和相关配置:"
if [ -f /etc/ld.so.conf ]; then
    echo "/etc/ld.so.conf 内容:"
    cat /etc/ld.so.conf
fi

if [ -d /etc/ld.so.conf.d ]; then
    echo -e "\n/etc/ld.so.conf.d 中的配置文件:"
    ls /etc/ld.so.conf.d/*.conf 2>/dev/null | while read conf_file; do
        echo "文件: $conf_file"
        cat "$conf_file" | head -5
    done
fi

echo -e "\n11. 容器内UCC解决方案分析"
echo "=================================================="
echo "基于容器内发现的UCC/UCX库，分析解决方案:"

if [ "$CONTAINER_HAS_UCC_UCX" = "YES" ]; then
    echo "✅ 容器内存在UCC/UCX库，分析替代方案..."
    
    # 查找容器内可用的UCX库
    CONTAINER_UCX_LIBS=()
    CONTAINER_UCC_LIBS=()
    
    for lib_file in "${FOUND_UCC_UCX[@]}"; do
        if [[ "$lib_file" == *"ucx"* ]] && [[ "$lib_file" == *.so* ]]; then
            CONTAINER_UCX_LIBS+=("$lib_file")
        elif [[ "$lib_file" == *"ucc"* ]] && [[ "$lib_file" == *.so* ]]; then
            CONTAINER_UCC_LIBS+=("$lib_file")
        fi
    done
    
    echo -e "\n📋 容器内UCX库:"
    if [ ${#CONTAINER_UCX_LIBS[@]} -eq 0 ]; then
        echo "❌ 无UCX库"
    else
        for ucx_lib in "${CONTAINER_UCX_LIBS[@]}"; do
            echo "  ✅ $ucx_lib"
            # 检查关键符号
            if nm -D "$ucx_lib" 2>/dev/null | grep -q "ucs_mpool_params_reset"; then
                echo "    🎯 包含关键符号 ucs_mpool_params_reset"
            fi
        done
    fi
    
    echo -e "\n📋 容器内UCC库:"
    if [ ${#CONTAINER_UCC_LIBS[@]} -eq 0 ]; then
        echo "❌ 无UCC库"
    else
        for ucc_lib in "${CONTAINER_UCC_LIBS[@]}"; do
            echo "  ✅ $ucc_lib"
        done
    fi
    
    # 生成使用容器内库的建议
    if [ ${#CONTAINER_UCX_LIBS[@]} -gt 0 ] || [ ${#CONTAINER_UCC_LIBS[@]} -gt 0 ]; then
        echo -e "\n💡 容器内库替代方案:"
        echo "建议优先使用容器内的UCC/UCX库路径:"
        
        # 构建容器内库路径
        CONTAINER_LIB_PATHS=()
        for lib_file in "${CONTAINER_UCX_LIBS[@]}" "${CONTAINER_UCC_LIBS[@]}"; do
            lib_dir=$(dirname "$lib_file")
            if [[ ! " ${CONTAINER_LIB_PATHS[@]} " =~ " ${lib_dir} " ]]; then
                CONTAINER_LIB_PATHS+=("$lib_dir")
            fi
        done
        
        echo "export LD_LIBRARY_PATH=\"$(IFS=:; echo "${CONTAINER_LIB_PATHS[*]}"):$LD_LIBRARY_PATH\""
        
        echo -e "\n🧪 建议测试命令:"
        echo "# 临时使用容器内库测试PyTorch"
        echo "export LD_LIBRARY_PATH=\"$(IFS=:; echo "${CONTAINER_LIB_PATHS[*]}"):$LD_LIBRARY_PATH\""
        echo "python3 -c 'import torch; print(\"PyTorch版本:\", torch.__version__)'"
        
    fi
else
    echo "❌ 容器内无UCC/UCX库，需要使用策略3重建方案"
fi

echo -e "\n12. 缓存污染机制分析"
echo "=================================================="
echo "缓存污染原理分析:"

if [ "$HPC_X_AVAILABLE" = "YES" ]; then
    echo "✅ HPC-X库存在于系统中"
    echo "   - 位置: /opt/hpcx/"
    echo "   - 影响: 系统可能已将HPC-X库注册到ldconfig缓存"
    
    echo -e "\n分析缓存污染路径:"
    echo "1. 系统启动时ldconfig扫描 /opt/hpcx/*/lib/"
    echo "2. 将UCC/UCX库路径写入缓存"
    echo "3. 动态链接器优先使用缓存中的路径"
    echo "4. LD_LIBRARY_PATH的优先级可能被缓存覆盖"
    
else
    echo "❌ 当前环境未发现HPC-X"
    echo "   - 可能原因1: 在容器外运行，容器内才有HPC-X挂载"
    echo "   - 可能原因2: HPC-X已被移除"
    echo "   - 可能原因3: 路径不同"
fi

echo -e "\n13. 生成容器内库测试脚本"
if [ "$CONTAINER_HAS_UCC_UCX" = "YES" ] && [ ${#CONTAINER_LIB_PATHS[@]} -gt 0 ]; then
    TEST_SCRIPT="/data/home/zdhs0054/zzhou/MaskLLM/llama8b_scripts/test_container_ucc.sh"
    echo "🔧 生成容器内UCC测试脚本: $TEST_SCRIPT"
    
    cat > "$TEST_SCRIPT" << EOF
#!/bin/bash
# 容器内UCC/UCX库测试脚本
# 基于analyze_ucc_calling_mechanism.sh的分析结果自动生成

echo "=========================================="
echo "容器内UCC/UCX库测试"
echo "=========================================="

echo "1. 备份原始环境..."
export ORIGINAL_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"

echo "2. 设置容器内库路径优先级..."
# 使用分析发现的容器内库路径
export LD_LIBRARY_PATH="$(IFS=:; echo "${CONTAINER_LIB_PATHS[*]}"):\$LD_LIBRARY_PATH"

echo "当前LD_LIBRARY_PATH前5个路径:"
echo "\$LD_LIBRARY_PATH" | tr ':' '\n' | head -5 | nl

echo -e "\n3. 清理HPC-X环境变量..."
unset HPCX_DIR HPCX_HOME HPCX_ROOT HPCX_MPI_DIR HPCX_UCX_DIR HPCX_UCC_DIR
unset OMPI_HOME OMPI_ROOT MPI_HOME MPI_ROOT

echo "4. 测试容器内UCC/UCX库..."
python3 -c "
import sys
import os

print('=== 容器内UCC库测试 ===')
print('当前LD_LIBRARY_PATH前3个:')
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
        print('✅ CUDA可用')
        print('CUDA设备数:', torch.cuda.device_count())
    
    print('STATUS: CONTAINER_UCC_SUCCESS')
    
except ImportError as e:
    error_msg = str(e)
    print('❌ PyTorch导入失败:', error_msg)
    if 'undefined symbol: ucs_mpool_params_reset' in error_msg:
        print('🔍 容器内库仍存在UCC冲突')
        print('STATUS: CONTAINER_UCC_CONFLICT')
    else:
        print('🔍 其他导入问题')
        print('STATUS: CONTAINER_OTHER_ERROR')
        
except Exception as e:
    print('❌ 其他错误:', e)
    print('STATUS: CONTAINER_UNKNOWN_ERROR')
"

RESULT=\$?

echo -e "\n5. 恢复原始环境..."
export LD_LIBRARY_PATH="\$ORIGINAL_LD_LIBRARY_PATH"

echo -e "\n=========================================="
if [ \$RESULT -eq 0 ]; then
    echo "🎉 容器内库测试完成！"
    echo "如果显示SUCCESS，则可以使用容器内UCC库解决冲突"
else
    echo "⚠️ 容器内库测试失败，建议使用策略3重建方案"
fi
echo "=========================================="
EOF

    chmod +x "$TEST_SCRIPT"
    echo "✅ 测试脚本已生成，运行方式:"
    echo "bash run_maskllm_native.sh llama8b_scripts/test_container_ucc.sh"
else
    echo "❌ 无法生成测试脚本（容器内无可用UCC/UCX库）"
fi

echo -e "\n14. 推荐验证步骤"
echo "建议在容器内运行此脚本以获得完整分析:"
echo "bash run_maskllm_native.sh llama8b_scripts/analyze_ucc_calling_mechanism.sh"

echo -e "\n分析完成时间: $(date)"
