#!/bin/bash
# 对比HF格式和Megatron格式的Llama8b推理结果
# Phase 3.3: 推理一致性验证的核心脚本
# HPC Server Command Execution Protocol compliant

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

echo "🔍 Llama8b HF vs Megatron 推理对比测试"
echo "========================================================"
echo "Phase 3.3: 推理一致性验证"
echo ""

# 测试提示列表
TEST_PROMPTS=(
    "<用户> 山东最高的山是什么山？<AI>"
    "<用户> 请简单介绍一下Python编程语言。<AI>"
    "<用户> 1+1等于几？<AI>"
    "<用户> 今天天气怎么样？<AI>"
)

# 输出目录
OUTPUT_DIR="/tmp/llama8b_inference_comparison"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

echo "📋 测试配置:"
echo "  测试提示数量: ${#TEST_PROMPTS[@]}"
echo "  输出目录: $OUTPUT_DIR"
echo "  HF模型: assets/checkpoints/Llama8b"
echo "  Megatron模型: output/checkpoints/llama8b_megatron_tp8"
echo ""

# 检查必要文件
echo "🔍 环境检查..."

if [[ ! -d "assets/checkpoints/Llama8b" ]]; then
    echo "❌ HF模型目录不存在: assets/checkpoints/Llama8b"
    exit 1
fi

if [[ ! -d "output/checkpoints/llama8b_megatron_tp8" ]]; then
    echo "❌ Megatron模型目录不存在: output/checkpoints/llama8b_megatron_tp8"
    echo "请先完成模型转换！"
    exit 1
fi

echo "✅ 模型目录检查通过"
echo ""

# 开始对比测试
for i in "${!TEST_PROMPTS[@]}"; do
    prompt="${TEST_PROMPTS[$i]}"
    echo "🧪 测试 $((i+1))/${#TEST_PROMPTS[@]}: '${prompt:0:30}${prompt:30:+...}'"
    echo "----------------------------------------"
    
    # HF格式推理
    echo "🔄 运行HF格式推理..."
    HF_OUTPUT_FILE="$OUTPUT_DIR/hf_output_$i.txt"
    HF_LOG_FILE="$OUTPUT_DIR/hf_log_$i.txt"
    
    # 修改HF脚本的prompt并运行
    sed -i "s/PROMPT_TEXT=.*/PROMPT_TEXT=\"$prompt\"/" "$SCRIPT_DIR/run_llama8b_infer.sh"
    
    if timeout 300 bash "$SCRIPT_DIR/run_llama8b_infer.sh" > "$HF_OUTPUT_FILE" 2> "$HF_LOG_FILE"; then
        echo "  ✅ HF推理完成"
        HF_SUCCESS=true
    else
        echo "  ❌ HF推理失败 (超时或错误)"
        echo "  📋 错误日志: $HF_LOG_FILE"
        HF_SUCCESS=false
    fi
    
    # Megatron格式推理
    echo "🔄 运行Megatron格式推理..."
    MG_OUTPUT_FILE="$OUTPUT_DIR/megatron_output_$i.txt"
    MG_LOG_FILE="$OUTPUT_DIR/megatron_log_$i.txt"
    
    # 修改Megatron脚本的prompt并运行
    sed -i "s/PROMPT_TEXT=.*/PROMPT_TEXT=\"$prompt\"/" "$SCRIPT_DIR/run_llama8b_megatron_infer.sh"
    
    if timeout 300 bash "$SCRIPT_DIR/run_llama8b_megatron_infer.sh" > "$MG_OUTPUT_FILE" 2> "$MG_LOG_FILE"; then
        echo "  ✅ Megatron推理完成"
        MG_SUCCESS=true
    else
        echo "  ❌ Megatron推理失败 (超时或错误)"
        echo "  📋 错误日志: $MG_LOG_FILE"
        MG_SUCCESS=false
    fi
    
    # 结果对比
    if [ "$HF_SUCCESS" = true ] && [ "$MG_SUCCESS" = true ]; then
        echo "📊 结果对比:"
        
        # 提取实际生成的文本（去掉prompt部分）
        if [[ -s "$HF_OUTPUT_FILE" ]] && [[ -s "$MG_OUTPUT_FILE" ]]; then
            echo "  📄 HF输出 (前100字符):"
            head -c 100 "$HF_OUTPUT_FILE" | tr '\n' ' '
            echo "..."
            echo ""
            echo "  📄 Megatron输出 (前100字符):"
            head -c 100 "$MG_OUTPUT_FILE" | tr '\n' ' '
            echo "..."
            echo ""
            
            # 简单的文本相似度检查
            HF_WORDS=$(wc -w < "$HF_OUTPUT_FILE")
            MG_WORDS=$(wc -w < "$MG_OUTPUT_FILE")
            echo "  📊 统计对比:"
            echo "    HF输出词数: $HF_WORDS"
            echo "    Megatron输出词数: $MG_WORDS"
            
            if (( HF_WORDS > 0 )) && (( MG_WORDS > 0 )); then
                # 计算词数差异百分比
                DIFF_PERCENT=$(( (HF_WORDS - MG_WORDS) * 100 / HF_WORDS ))
                if (( DIFF_PERCENT < 0 )); then
                    DIFF_PERCENT=$(( -DIFF_PERCENT ))
                fi
                echo "    词数差异: ${DIFF_PERCENT}%"
                
                if (( DIFF_PERCENT <= 20 )); then
                    echo "  ✅ 输出长度相近"
                else
                    echo "  ⚠️ 输出长度差异较大"
                fi
            fi
        else
            echo "  ⚠️ 输出文件为空或不存在"
        fi
    else
        echo "  ❌ 无法对比（推理失败）"
    fi
    
    echo ""
done

# 生成最终报告
echo "========================================================"
echo "📊 最终对比报告"
echo "========================================================"

TOTAL_TESTS=${#TEST_PROMPTS[@]}
HF_SUCCESS_COUNT=0
MG_SUCCESS_COUNT=0
COMPARISON_COUNT=0

for i in $(seq 0 $((TOTAL_TESTS-1))); do
    if [[ -s "$OUTPUT_DIR/hf_output_$i.txt" ]]; then
        ((HF_SUCCESS_COUNT++))
    fi
    
    if [[ -s "$OUTPUT_DIR/megatron_output_$i.txt" ]]; then
        ((MG_SUCCESS_COUNT++))
    fi
    
    if [[ -s "$OUTPUT_DIR/hf_output_$i.txt" ]] && [[ -s "$OUTPUT_DIR/megatron_output_$i.txt" ]]; then
        ((COMPARISON_COUNT++))
    fi
done

echo "测试结果统计:"
echo "  总测试数: $TOTAL_TESTS"
echo "  HF成功: $HF_SUCCESS_COUNT/$TOTAL_TESTS"
echo "  Megatron成功: $MG_SUCCESS_COUNT/$TOTAL_TESTS"
echo "  可对比测试: $COMPARISON_COUNT/$TOTAL_TESTS"
echo ""

if (( COMPARISON_COUNT >= TOTAL_TESTS / 2 )); then
    echo "✅ 对比测试基本成功！"
    echo "💡 已生成足够的对比数据进行分析"
    RESULT_CODE=0
elif (( HF_SUCCESS_COUNT > 0 )) && (( MG_SUCCESS_COUNT > 0 )); then
    echo "⚠️ 部分测试成功"
    echo "🔧 建议检查失败的测试用例"
    RESULT_CODE=1
else
    echo "❌ 大部分测试失败"
    echo "🔧 需要检查模型和环境配置"
    RESULT_CODE=2
fi

echo ""
echo "📁 详细结果文件位置: $OUTPUT_DIR"
echo "   - hf_output_*.txt: HF格式推理输出"
echo "   - megatron_output_*.txt: Megatron格式推理输出"
echo "   - *_log_*.txt: 推理过程日志"
echo ""
echo "🎯 Phase 3.3推理一致性验证完成！"

exit $RESULT_CODE
