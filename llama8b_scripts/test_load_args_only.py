#!/usr/bin/env python3
"""
仅测试模型加载参数解析 - 跳过CUDA检查
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_checkpoint_files():
    print("🔍 检查checkpoint文件:")
    checkpoint_dir = "output/checkpoints/llama8b_megatron_tp8"
    
    if not os.path.exists(checkpoint_dir):
        print(f"  ❌ Checkpoint目录不存在: {checkpoint_dir}")
        return False
    
    print(f"  ✅ Checkpoint目录存在: {checkpoint_dir}")
    
    # 检查关键文件
    required_files = ["latest_checkpointed_iteration.txt", "vocab.txt"]
    for file in required_files:
        file_path = os.path.join(checkpoint_dir, file)
        if os.path.exists(file_path):
            print(f"    ✅ {file}")
        else:
            print(f"    ❌ 缺少 {file}")
            return False
    
    # 检查TP分片数量
    iter_dir = os.path.join(checkpoint_dir, "iter_0000001")
    if os.path.exists(iter_dir):
        mp_ranks = [f for f in os.listdir(iter_dir) if f.startswith("mp_rank_")]
        print(f"    ✅ 找到{len(mp_ranks)}个TP分片: {mp_ranks}")
        return len(mp_ranks) == 8
    else:
        print("    ❌ 迭代目录不存在")
        return False

def test_megatron_args():
    print("\n🔍 测试Megatron参数解析:")
    
    try:
        from megatron.arguments import parse_args
        from megatron.global_vars import set_global_variables
        print("  ✅ Megatron模块导入成功")
        
        # 构造参数 - 添加所有必需的Megatron参数
        test_args = [
            '--load', 'output/checkpoints/llama8b_megatron_tp8',
            '--tensor-model-parallel-size', '1',  # 张量并行大小
            '--pipeline-model-parallel-size', '1',  # 流水线并行大小
            '--context-parallel-size', '1',  # 上下文并行大小
            '--num-layers', '32',
            '--hidden-size', '4096',
            '--num-attention-heads', '32',
            '--tokenizer-type', 'Llama8bTokenizer',
            '--tokenizer-model', 'assets/checkpoints/Llama8b',  # 指向原始Llama8b目录，包含llama8b子目录
            '--vocab-size', '119696',
            '--make-vocab-size-divisible-by', '128',  # 词汇表填充
            '--seq-length', '512',
            '--max-position-embeddings', '4096',
            '--micro-batch-size', '1',
            '--global-batch-size', '1',
            '--lr', '1e-4',
            '--train-iters', '1',
            '--lr-decay-iters', '1',
            '--lr-decay-style', 'cosine',
            '--min-lr', '1e-5',
            '--weight-decay', '0.1',
            '--clip-grad', '1.0',
            '--fp16',  # 使用fp16
            '--swiglu',
            '--normalization', 'RMSNorm',
            '--use-rotary-position-embeddings',
            '--untie-embeddings-and-output-weights',
            '--skip-train',
            '--no-load-optim',  # 不加载优化器状态
            '--no-load-rng'     # 不加载随机数状态
        ]
        
        print(f"  📋 参数数量: {len(test_args)}")
        print(f"  📋 关键参数:")
        print(f"    --load: output/checkpoints/llama8b_megatron_tp8")
        print(f"    --tensor-model-parallel-size: 1")
        print(f"    --vocab-size: 119696")
        
        # 保存原始argv
        original_argv = sys.argv
        try:
            sys.argv = ['test'] + test_args
            
            print("  🔄 解析参数...")
            args = parse_args()
            print("  ✅ 参数解析成功!")
            
            # 手动设置必需的并行计算属性
            print("  🔄 设置并行计算属性...")
            args.world_size = 1  # 单机测试
            args.rank = 0
            args.local_rank = 0
            
            # 计算data_parallel_size
            model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
            context_parallel_size = getattr(args, 'context_parallel_size', 1)
            args.data_parallel_size = args.world_size // (model_parallel_size * context_parallel_size)
            
            print(f"    ✅ world_size: {args.world_size}")
            print(f"    ✅ data_parallel_size: {args.data_parallel_size}")
            print("  ✅ 并行计算属性设置成功!")
            
            # 验证关键参数
            print(f"    ✅ load: {getattr(args, 'load', 'None')}")
            print(f"    ✅ tensor_model_parallel_size: {getattr(args, 'tensor_model_parallel_size', 'None')}")
            print(f"    ✅ data_parallel_size: {getattr(args, 'data_parallel_size', 'None')}")
            print(f"    ✅ vocab_size: {getattr(args, 'vocab_size', 'None')}")
            print(f"    ✅ num_layers: {getattr(args, 'num_layers', 'None')}")
            
            # 测试全局变量设置
            print("  🔄 设置全局变量...")
            set_global_variables(args)
            print("  ✅ 全局变量设置成功!")
            
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
            
    except ImportError as e:
        print(f"  ❌ Megatron导入失败: {e}")
        return None
    except Exception as e:
        print(f"  ❌ 其他错误: {e}")
        return None

def test_tokenizer_building(args):
    print("\n🔍 测试Tokenizer构建:")
    
    if not args:
        print("  ⏭️ 跳过，因为没有args")
        return False
    
    try:
        from megatron.tokenizer import build_tokenizer
        
        print("  🔄 构建tokenizer...")
        tokenizer = build_tokenizer(args)
        print("  ✅ Tokenizer构建成功!")
        
        print(f"    ✅ 词汇表大小: {tokenizer.vocab_size}")
        print(f"    ✅ Tokenizer类型: {type(tokenizer).__name__}")
        
        # 简单测试
        test_text = "Hello"
        tokens = tokenizer.tokenize(test_text)
        print(f"    ✅ 测试编码: '{test_text}' -> {tokens}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Tokenizer构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 模型加载参数测试 (跳过CUDA)")
    print("=" * 50)
    
    # 检查文件
    files_ok = test_checkpoint_files()
    
    # 测试参数
    args = test_megatron_args()
    
    # 测试tokenizer
    tokenizer_ok = False
    if args:
        tokenizer_ok = test_tokenizer_building(args)
    
    print("\n" + "=" * 50)
    print("📊 测试结果:")
    print(f"  Checkpoint文件: {'✅ 完整' if files_ok else '❌ 问题'}")
    print(f"  Megatron参数: {'✅ 成功' if args else '❌ 失败'}")
    print(f"  Tokenizer: {'✅ 成功' if tokenizer_ok else '❌ 失败'}")
    
    if files_ok and args:
        print("\n🎉 核心加载参数测试通过!")
        print("模型文件和参数解析都正常，可以进行下一步测试")
        return 0
    else:
        print("\n⚠️ 存在问题需要解决")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
