#!/usr/bin/env python3
"""
快速测试CUDA设备和Megatron参数
"""

import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_cuda_devices():
    print("🔍 CUDA设备检测:")
    try:
        import torch
        
        # 检查环境变量
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # 检查PyTorch可见的设备
        device_count = torch.cuda.device_count()
        print(f"  PyTorch看到的CUDA设备数: {device_count}")
        
        if torch.cuda.is_available():
            for i in range(device_count):
                print(f"    设备{i}: {torch.cuda.get_device_name(i)}")
        
        # 如果设备数不是8，尝试设置环境变量
        if device_count != 8:
            print("  ⚠️ 设备数不是8，尝试设置CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
            
            # 重新导入torch可能不会生效，但显示建议
            print("  💡 建议：在运行脚本前设置 export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
            
        return device_count >= 4  # 至少4个设备可以继续测试
        
    except Exception as e:
        print(f"  ❌ CUDA检测失败: {e}")
        return False

def test_megatron_args():
    print("\n🔍 Megatron参数测试:")
    try:
        from megatron.arguments import parse_args
        
        # 使用更简单的参数集
        test_args = [
            '--num-layers', '32',
            '--hidden-size', '4096',
            '--num-attention-heads', '32',
            '--tokenizer-type', 'Llama2Tokenizer',
            '--vocab-size', '119696',
            '--seq-length', '512',
            '--micro-batch-size', '1',
            '--global-batch-size', '1',
            '--lr', '1e-4',
            '--train-iters', '1',
            '--lr-decay-iters', '1',
            '--lr-decay-style', 'cosine',
            '--min-lr', '1e-5',
            '--swiglu',
            '--skip-train'
        ]
        
        print(f"  📋 测试参数: {len(test_args)}个")
        
        # 保存原始argv
        original_argv = sys.argv
        try:
            sys.argv = ['test'] + test_args
            args = parse_args()
            print("  ✅ 基础参数解析成功")
            
            # 测试带load参数
            load_args = test_args + ['--load', 'output/checkpoints/llama8b_megatron_tp8']
            sys.argv = ['test'] + load_args
            args_with_load = parse_args()
            print("  ✅ 带--load参数解析成功")
            
            return args_with_load
            
        except SystemExit as e:
            print(f"  ❌ 参数解析SystemExit: {e}")
            return None
        except Exception as e:
            print(f"  ❌ 参数解析异常: {e}")
            return None
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"  ❌ Megatron导入失败: {e}")
        return None

def main():
    print("🚀 CUDA和参数快速检测")
    print("=" * 40)
    
    # 测试CUDA
    cuda_ok = test_cuda_devices()
    
    # 测试Megatron参数
    args = test_megatron_args()
    
    print("\n" + "=" * 40)
    print("📊 检测结果:")
    print(f"  CUDA环境: {'✅ 可用' if cuda_ok else '❌ 问题'}")
    print(f"  Megatron参数: {'✅ 成功' if args else '❌ 失败'}")
    
    if cuda_ok and args:
        print("\n🎉 基础环境检测通过！")
        print("可以尝试实际的模型加载测试")
        return 0
    else:
        print("\n⚠️ 需要解决环境问题")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
