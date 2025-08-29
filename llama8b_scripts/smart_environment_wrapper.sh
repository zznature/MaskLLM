#!/bin/bash

# 智能环境包装器
# 自动检测当前环境并选择最适合的执行方式

SCRIPT_TO_RUN="$1"
shift
SCRIPT_ARGS="$@"

echo "🔍 智能环境检测和适配"
echo "目标脚本: $SCRIPT_TO_RUN"
echo "脚本参数: $SCRIPT_ARGS"
echo ""

# 检测当前环境类型
detect_environment() {
    echo "📋 检测当前环境..."
    
    # 检查是否在Apptainer内
    if [ -n "$APPTAINER_NAME" ] || [ -n "$SINGULARITY_NAME" ]; then
        echo "✅ 当前在Apptainer/Singularity容器内"
        return 0  # 容器内
    fi
    
    # 检查transformer_engine是否可用
    if python3.10 -c "import transformer_engine" 2>/dev/null; then
        echo "✅ transformer_engine可用，环境完整"
        return 0  # 环境完整
    fi
    
    # 检查HPC-X冲突
    if [ -f "/opt/hpcx/ucc/lib/libucc.so.1" ]; then
        echo "⚠️  检测到HPC-X冲突，在容器外环境"
        return 1  # 需要容器
    fi
    
    echo "⚠️  环境不完整，可能需要容器"
    return 1  # 需要容器
}

# 检查脚本类型和需求
check_script_requirements() {
    local script="$1"
    echo "📋 检查脚本 $script 的环境需求..."
    
    # 检查是否需要transformer_engine
    if grep -q "transformer_engine\|preprocess_data\|megatron" "$script" 2>/dev/null; then
        echo "⚠️  脚本需要完整的Megatron环境 (transformer_engine)"
        return 1  # 需要完整环境
    fi
    
    # 检查是否只需要PyTorch
    if grep -q "torch\|pytorch" "$script" 2>/dev/null; then
        echo "✅ 脚本只需要基础PyTorch环境"
        return 0  # 基础环境足够
    fi
    
    echo "💡 脚本需求不明确，采用保守策略"
    return 1  # 保守策略：需要完整环境
}

# 在容器内运行
run_in_container() {
    echo "🚀 在容器内执行脚本..."
    echo "命令: bash run_apptainer_extended_libs.sh"
    echo "然后执行: bash run_maskllm_native.sh $SCRIPT_TO_RUN $SCRIPT_ARGS"
    echo ""
    echo "💡 请手动执行以下命令:"
    echo "   bash run_apptainer_extended_libs.sh"
    echo "   # 进入容器后运行:"
    echo "   bash run_maskllm_native.sh $SCRIPT_TO_RUN $SCRIPT_ARGS"
}

# 在当前环境运行
run_in_current_env() {
    echo "🚀 在当前环境执行脚本..."
    exec bash run_maskllm_native.sh "$SCRIPT_TO_RUN" $SCRIPT_ARGS
}

# 主逻辑
main() {
    if [ -z "$SCRIPT_TO_RUN" ]; then
        echo "❌ 错误: 请提供要运行的脚本路径"
        echo "用法: $0 <脚本路径> [参数...]"
        exit 1
    fi
    
    if [ ! -f "$SCRIPT_TO_RUN" ]; then
        echo "❌ 错误: 脚本文件不存在: $SCRIPT_TO_RUN"
        exit 1
    fi
    
    # 检测环境
    detect_environment
    ENV_OK=$?
    
    # 检查脚本需求
    check_script_requirements "$SCRIPT_TO_RUN"
    NEEDS_FULL_ENV=$?
    
    echo ""
    echo "🎯 环境适配决策:"
    
    if [ $ENV_OK -eq 0 ] && [ $NEEDS_FULL_ENV -eq 0 ]; then
        echo "✅ 当前环境满足脚本需求，直接执行"
        run_in_current_env
    elif [ $ENV_OK -eq 0 ] && [ $NEEDS_FULL_ENV -eq 1 ]; then
        echo "⚠️  脚本需要完整环境，但当前环境可能不足"
        echo "🔧 尝试在当前环境执行，如失败请使用容器"
        run_in_current_env
    else
        echo "🔄 当前环境不足，需要在容器内执行"
        run_in_container
    fi
}

main "$@"
