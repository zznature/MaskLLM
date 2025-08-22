#!/usr/bin/env python3
"""
Phase 3.2: 转换后模型加载测试
验证Llama8b转换后模型在Megatron中的正确加载
"""

import os
import sys
import torch
import argparse
from contextlib import contextmanager

def setup_megatron_path():
    """设置Megatron路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

@contextmanager
def suppress_output():
    """抑制输出的上下文管理器"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def test_basic_imports():
    """测试基础导入"""
    print("🔍 测试1: 基础模块导入...")
    try:
        setup_megatron_path()
        
        # 测试基础模块导入
        import megatron
        from megatron import get_args
        from megatron.global_vars import set_global_variables
        from megatron.arguments import parse_args, validate_args
        from megatron.core import mpu
        
        print("  ✅ 基础Megatron模块导入成功")
        return True
    except Exception as e:
        print(f"  ❌ 基础模块导入失败: {e}")
        return False

def test_checkpoint_structure():
    """测试checkpoint文件结构"""
    print("🔍 测试2: Checkpoint文件结构验证...")
    
    checkpoint_dir = "output/checkpoints/llama8b_megatron_tp8"
    
    # 检查基础文件结构
    required_files = [
        "latest_checkpointed_iteration.txt",
        "vocab.txt"
    ]
    
    required_dirs = [
        "iter_0000001",
        "llama8b"
    ]
    
    all_exist = True
    
    for file in required_files:
        file_path = os.path.join(checkpoint_dir, file)
        if os.path.exists(file_path):
            print(f"  ✅ 找到文件: {file}")
        else:
            print(f"  ❌ 缺失文件: {file}")
            all_exist = False
    
    for dir_name in required_dirs:
        dir_path = os.path.join(checkpoint_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"  ✅ 找到目录: {dir_name}")
            
            # 检查iter目录中的分片
            if dir_name == "iter_0000001":
                mp_ranks = [f"mp_rank_{i:02d}" for i in range(8)]
                for rank in mp_ranks:
                    rank_path = os.path.join(dir_path, rank)
                    if os.path.exists(rank_path):
                        print(f"    ✅ 找到TP分片: {rank}")
                    else:
                        print(f"    ❌ 缺失TP分片: {rank}")
                        all_exist = False
        else:
            print(f"  ❌ 缺失目录: {dir_name}")
            all_exist = False
    
    return all_exist

def test_checkpoint_metadata():
    """测试checkpoint元数据"""
    print("🔍 测试3: Checkpoint元数据验证...")
    
    try:
        checkpoint_dir = "output/checkpoints/llama8b_megatron_tp8"
        iteration_file = os.path.join(checkpoint_dir, "latest_checkpointed_iteration.txt")
        
        with open(iteration_file, 'r') as f:
            iteration = f.read().strip()
        
        print(f"  ✅ 最新迭代: {iteration}")
        
        # 检查迭代目录
        iter_dir = os.path.join(checkpoint_dir, f"iter_{int(iteration):07d}")
        if os.path.exists(iter_dir):
            print(f"  ✅ 迭代目录存在: {iter_dir}")
            return True
        else:
            print(f"  ❌ 迭代目录不存在: {iter_dir}")
            return False
            
    except Exception as e:
        print(f"  ❌ 元数据读取失败: {e}")
        return False

def create_test_args():
    """创建测试用的参数配置"""
    print("🔍 测试4: 创建模型参数配置...")
    
    try:
        # 构造命令行参数
        test_args = [
            '--load', 'output/checkpoints/llama8b_megatron_tp8',
            '--model-parallel-size', '8',
            '--num-layers', '32',
            '--hidden-size', '4096', 
            '--num-attention-heads', '32',
            '--max-position-embeddings', '4096',
            '--tokenizer-type', 'Llama2Tokenizer',  # 使用Llama2Tokenizer以兼容转换
            '--vocab-size', '119696',
            '--seq-length', '2048',
            '--micro-batch-size', '1',
            '--global-batch-size', '8',
            '--lr', '1e-4',
            '--train-iters', '100',
            '--lr-decay-iters', '99',
            '--lr-decay-style', 'cosine',
            '--min-lr', '1e-5',
            '--weight-decay', '1e-2',
            '--lr-warmup-fraction', '0.01',
            '--clip-grad', '1.0',
            '--use-flash-attn',
            '--swiglu',
            '--normalization', 'RMSNorm',
            '--use-rotary-position-embeddings',
            '--rotary-percent', '1.0',
            '--untie-embeddings-and-output-weights',
            '--no-gradient-accumulation-fusion',
            '--load-only'  # 只加载，不训练
        ]
        
        print(f"  ✅ 参数配置创建成功，共{len(test_args)}个参数")
        return test_args
        
    except Exception as e:
        print(f"  ❌ 参数配置创建失败: {e}")
        return None

def test_argument_parsing():
    """测试参数解析"""
    print("🔍 测试5: Megatron参数解析...")
    
    try:
        setup_megatron_path()
        from megatron.arguments import parse_args, validate_args
        
        # 创建测试参数
        test_args = create_test_args()
        if not test_args:
            return False
            
        # 保存原始argv
        original_argv = sys.argv
        try:
            # 设置测试参数
            sys.argv = ['test_script'] + test_args
            
            print("  🔄 正在解析参数...")
            print(f"  📋 参数列表: {' '.join(test_args[:10])}...") # 显示前10个参数
            
            # 解析参数 - 不抑制输出以便看到错误
            try:
                args = parse_args()
                print("  ✅ 参数解析成功")
            except SystemExit as e:
                print(f"  ❌ 参数解析SystemExit: {e}")
                return None
            except Exception as e:
                print(f"  ❌ 参数解析异常: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # 验证关键参数
            key_params = {
                'num_layers': 32,
                'hidden_size': 4096,
                'num_attention_heads': 32,
                'vocab_size': 119696,
                'model_parallel_size': 8
            }
            
            for param, expected in key_params.items():
                actual = getattr(args, param, None)
                if actual == expected:
                    print(f"    ✅ {param}: {actual}")
                else:
                    print(f"    ⚠️  {param}: {actual} (期望: {expected})")
            
            return args
            
        finally:
            # 恢复原始argv
            sys.argv = original_argv
            
    except Exception as e:
        print(f"  ❌ 参数解析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_distributed_setup():
    """测试分布式环境设置"""
    print("🔍 测试6: 分布式环境初始化...")
    
    try:
        setup_megatron_path()
        from megatron.core import mpu
        import torch.distributed as dist
        
        # 检查是否已经初始化
        if dist.is_initialized():
            print("  ✅ 分布式环境已初始化")
            print(f"    - World size: {dist.get_world_size()}")
            print(f"    - Rank: {dist.get_rank()}")
            return True
        else:
            print("  ⚠️  分布式环境未初始化，尝试初始化...")
            
            # 设置环境变量
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29500')
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            
            # 初始化进程组
            try:
                dist.init_process_group(backend='gloo', rank=0, world_size=1)
                print("  ✅ 分布式环境初始化成功")
                return True
            except Exception as e:
                print(f"  ⚠️  分布式环境初始化失败: {e}")
                return False
                
    except Exception as e:
        print(f"  ❌ 分布式设置失败: {e}")
        return False

def test_model_parallel_setup(args):
    """测试模型并行设置"""
    print("🔍 测试7: 模型并行初始化...")
    
    try:
        setup_megatron_path()
        from megatron.core import mpu
        from megatron.global_vars import set_global_variables
        
        # 设置全局变量
        with suppress_output():
            set_global_variables(args)
        
        # 初始化模型并行
        if not mpu.is_initialized():
            with suppress_output():
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size_=args.model_parallel_size,
                    pipeline_model_parallel_size_=1
                )
        
        print("  ✅ 模型并行初始化成功")
        print(f"    - Tensor parallel size: {mpu.get_tensor_model_parallel_world_size()}")
        print(f"    - Pipeline parallel size: {mpu.get_pipeline_model_parallel_world_size()}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型并行设置失败: {e}")
        return False

def test_tokenizer_loading(args):
    """测试tokenizer加载"""
    print("🔍 测试8: Tokenizer加载...")
    
    try:
        setup_megatron_path()
        from megatron.tokenizer import build_tokenizer
        
        # 构建tokenizer
        with suppress_output():
            tokenizer = build_tokenizer(args)
        
        print("  ✅ Tokenizer构建成功")
        print(f"    - 词汇表大小: {tokenizer.vocab_size}")
        print(f"    - Tokenizer类型: {type(tokenizer).__name__}")
        
        # 测试基础功能
        test_text = "Hello world"
        tokens = tokenizer.tokenize(test_text)
        detokenized = tokenizer.detokenize(tokens)
        
        print(f"    - 测试编码: '{test_text}' -> {tokens}")
        print(f"    - 测试解码: {tokens} -> '{detokenized}'")
        
        return tokenizer
        
    except Exception as e:
        print(f"  ❌ Tokenizer加载失败: {e}")
        return None

def main():
    """主测试函数"""
    print("🚀 Phase 3.2: 转换后模型加载测试")
    print("=" * 60)
    
    # 测试序列
    tests = [
        ("基础导入测试", test_basic_imports),
        ("文件结构验证", test_checkpoint_structure), 
        ("元数据验证", test_checkpoint_metadata),
        ("参数解析测试", test_argument_parsing),
        ("分布式环境设置", test_distributed_setup),
    ]
    
    results = []
    args = None
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_name == "参数解析测试":
                result = test_func()
                if result:
                    args = result
                    results.append((test_name, True))
                else:
                    results.append((test_name, False))
            else:
                result = test_func()
                results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ 测试执行异常: {e}")
            results.append((test_name, False))
    
    # 如果前面的测试通过，继续高级测试
    if args and all(result for _, result in results[-2:]):  # 检查参数解析和分布式设置
        print(f"\n模型并行初始化:")
        mp_result = test_model_parallel_setup(args)
        results.append(("模型并行设置", mp_result))
        
        if mp_result:
            print(f"\nTokenizer加载测试:")
            tokenizer_result = test_tokenizer_loading(args)
            results.append(("Tokenizer加载", tokenizer_result is not None))
    
    # 总结结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！模型加载环境就绪。")
        return 0
    elif passed >= total * 0.7:
        print("⚠️  大部分测试通过，可能需要微调配置。")
        return 1
    else:
        print("❌ 多个关键测试失败，需要进一步调试。")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
