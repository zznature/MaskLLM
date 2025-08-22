#!/usr/bin/env python3
"""
简化的模型加载测试 - 逐步诊断问题
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_step_by_step():
    print("🔍 逐步诊断测试")
    print("=" * 50)
    
    # Step 1: 基础环境
    print("\n1️⃣ 检查基础环境:")
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
        print(f"  ✅ CUDA可用: {torch.cuda.is_available()}")
        print(f"  ✅ CUDA设备数: {torch.cuda.device_count()}")
    except Exception as e:
        print(f"  ❌ PyTorch问题: {e}")
        return False
    
    # Step 2: 检查Megatron基础导入
    print("\n2️⃣ 检查Megatron导入:")
    try:
        from megatron import get_args
        print("  ✅ megatron.get_args 导入成功")
        
        from megatron.arguments import parse_args
        print("  ✅ megatron.arguments.parse_args 导入成功")
        
        from megatron.core import mpu
        print("  ✅ megatron.core.mpu 导入成功")
    except Exception as e:
        print(f"  ❌ Megatron导入问题: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: 检查checkpoint文件
    print("\n3️⃣ 检查checkpoint文件:")
    checkpoint_dir = "output/checkpoints/llama8b_megatron_tp8"
    
    if os.path.exists(checkpoint_dir):
        print(f"  ✅ Checkpoint目录存在: {checkpoint_dir}")
        
        files = os.listdir(checkpoint_dir)
        print(f"  📁 目录内容: {files}")
        
        # 检查关键文件
        required_files = ["latest_checkpointed_iteration.txt", "vocab.txt"]
        for file in required_files:
            if file in files:
                print(f"    ✅ {file}")
            else:
                print(f"    ❌ 缺少 {file}")
                
        # 检查迭代目录
        iter_dirs = [f for f in files if f.startswith("iter_")]
        if iter_dirs:
            print(f"  ✅ 找到迭代目录: {iter_dirs}")
        else:
            print("  ❌ 没有找到迭代目录")
            
    else:
        print(f"  ❌ Checkpoint目录不存在: {checkpoint_dir}")
        return False
    
    # Step 4: 测试参数构造
    print("\n4️⃣ 测试参数构造:")
    try:
        # 构造最小参数集
        minimal_args = [
            '--load', checkpoint_dir,
            '--model-parallel-size', '1',  # 先用1看看能否工作
            '--num-layers', '32',
            '--hidden-size', '4096',
            '--num-attention-heads', '32',
            '--tokenizer-type', 'Llama2Tokenizer',
            '--vocab-size', '119696',
            '--seq-length', '1024',
            '--micro-batch-size', '1',
            '--global-batch-size', '1',
            '--lr', '1e-4',
            '--train-iters', '1',
            '--lr-decay-iters', '1',
            '--lr-decay-style', 'cosine',
            '--min-lr', '1e-5',
            '--swiglu',
            '--skip-train'  # 使用skip-train而不是load-only
        ]
        
        print(f"  ✅ 构造了{len(minimal_args)}个参数")
        print(f"  📋 关键参数: --load {checkpoint_dir}")
        print(f"  📋 模型并行: 1 (简化测试)")
        
        return minimal_args
        
    except Exception as e:
        print(f"  ❌ 参数构造失败: {e}")
        return False
    
def test_argument_parsing(args_list):
    print("\n5️⃣ 测试参数解析:")
    
    try:
        from megatron.arguments import parse_args
        
        # 保存并替换sys.argv
        original_argv = sys.argv
        sys.argv = ['test'] + args_list
        
        try:
            print("  🔄 开始解析参数...")
            args = parse_args()
            print("  ✅ 参数解析成功!")
            
            # 显示关键参数
            print(f"    - load: {getattr(args, 'load', 'None')}")
            print(f"    - model_parallel_size: {getattr(args, 'model_parallel_size', 'None')}")
            print(f"    - vocab_size: {getattr(args, 'vocab_size', 'None')}")
            
            return args
            
        except SystemExit as e:
            print(f"  ❌ 参数解析SystemExit: {e}")
            return None
        except Exception as e:
            print(f"  ❌ 参数解析异常: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"  ❌ 参数解析模块问题: {e}")
        return None

def main():
    print("🧪 简化模型加载诊断测试")
    
    # 逐步测试
    args_list = test_step_by_step()
    if not args_list:
        print("\n❌ 基础测试失败，无法继续")
        return 1
    
    # 参数解析测试
    parsed_args = test_argument_parsing(args_list)
    if not parsed_args:
        print("\n❌ 参数解析失败")
        return 1
    
    print("\n🎉 基础诊断测试通过!")
    print("可以继续进行更高级的模型加载测试")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
