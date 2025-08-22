#!/usr/bin/env python3

"""
Test script for large vocabulary compatibility in Llama8b tokenizer.

This script specifically tests the handling of the large vocabulary size (119696)
to ensure compatibility with Megatron's vocabulary padding and tensor parallel features.
"""

import os
import sys
import argparse
import time
import gc

# Add MaskLLM root to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, PROJECT_DIR)

def create_mock_args(tokenizer_model_path, tensor_parallel_size=1, make_vocab_divisible_by=128):
    """Create mock arguments for different tensor parallel configurations."""
    class MockArgs:
        def __init__(self):
            self.tokenizer_type = 'Llama8bTokenizer'
            self.tokenizer_model = tokenizer_model_path
            self.rank = 0
            self.make_vocab_size_divisible_by = make_vocab_divisible_by
            self.tensor_model_parallel_size = tensor_parallel_size
            self.padded_vocab_size = None
    
    return MockArgs()

def test_vocabulary_padding(tokenizer_model_path):
    """Test vocabulary padding with different tensor parallel configurations."""
    print("=" * 80)
    print("TEST 1: Vocabulary Padding with Different TP Configurations")
    print("=" * 80)
    
    try:
        from megatron.tokenizer.tokenizer import build_tokenizer
        
        # Test configurations: TP=1, TP=2, TP=4, TP=8
        tp_configs = [1, 2, 4, 8]
        results = []
        
        for tp_size in tp_configs:
            print(f"\n--- Testing TP={tp_size} ---")
            
            args = create_mock_args(tokenizer_model_path, tensor_parallel_size=tp_size)
            tokenizer = build_tokenizer(args)
            
            original_vocab_size = tokenizer.vocab_size
            padded_vocab_size = args.padded_vocab_size
            padding_added = padded_vocab_size - original_vocab_size
            
            print(f"  Original vocab size: {original_vocab_size}")
            print(f"  Padded vocab size: {padded_vocab_size}")
            print(f"  Padding added: {padding_added}")
            print(f"  Divisible by (128 * {tp_size}): {padded_vocab_size % (128 * tp_size) == 0}")
            
            results.append({
                'tp_size': tp_size,
                'original': original_vocab_size,
                'padded': padded_vocab_size,
                'padding': padding_added,
                'divisible': padded_vocab_size % (128 * tp_size) == 0
            })
            
            # Clean up
            del tokenizer
            del args
            gc.collect()
        
        # Summary
        print(f"\n--- Vocabulary Padding Summary ---")
        all_divisible = all(r['divisible'] for r in results)
        print(f"✅ All configurations divisible: {all_divisible}")
        
        for result in results:
            status = "✅" if result['divisible'] else "❌"
            print(f"  TP={result['tp_size']}: {result['original']} → {result['padded']} ({result['padding']} padding) {status}")
        
        return all_divisible
        
    except Exception as e:
        print(f"❌ Vocabulary padding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage(tokenizer_model_path):
    """Test memory usage with large vocabulary."""
    print("\n" + "=" * 80)
    print("TEST 2: Memory Usage with Large Vocabulary")
    print("=" * 80)
    
    try:
        import psutil
        import torch
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        from megatron.tokenizer.tokenizer import build_tokenizer
        
        # Load tokenizer
        args = create_mock_args(tokenizer_model_path)
        start_time = time.time()
        tokenizer = build_tokenizer(args)
        load_time = time.time() - start_time
        
        after_load_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = after_load_memory - initial_memory
        
        print(f"Tokenizer load time: {load_time:.2f} seconds")
        print(f"Memory after loading: {after_load_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Test tokenization performance
        test_text = "Hello world! This is a test of the large vocabulary tokenizer. " * 100
        
        start_time = time.time()
        token_ids = tokenizer.tokenize(test_text)
        tokenize_time = time.time() - start_time
        
        start_time = time.time()
        decoded_text = tokenizer.detokenize(token_ids)
        decode_time = time.time() - start_time
        
        print(f"Tokenization time: {tokenize_time:.4f} seconds ({len(token_ids)} tokens)")
        print(f"Decoding time: {decode_time:.4f} seconds")
        
        # Memory efficiency check
        if memory_increase < 500:  # Less than 500MB increase
            print("✅ Memory usage acceptable")
            memory_ok = True
        else:
            print("⚠️  High memory usage detected")
            memory_ok = False
        
        # Performance check
        if tokenize_time < 1.0 and decode_time < 1.0:  # Less than 1 second each
            print("✅ Performance acceptable")
            performance_ok = True
        else:
            print("⚠️  Slow performance detected")
            performance_ok = False
        
        return memory_ok and performance_ok
        
    except ImportError:
        print("⚠️  psutil not available, skipping detailed memory analysis")
        # Basic test without psutil
        try:
            from megatron.tokenizer.tokenizer import build_tokenizer
            args = create_mock_args(tokenizer_model_path)
            tokenizer = build_tokenizer(args)
            
            # Simple functionality test
            test_text = "Hello world!"
            token_ids = tokenizer.tokenize(test_text)
            decoded_text = tokenizer.detokenize(token_ids)
            
            print("✅ Basic functionality test passed")
            return True
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        return False

def test_edge_cases(tokenizer_model_path):
    """Test edge cases with large vocabulary."""
    print("\n" + "=" * 80)
    print("TEST 3: Edge Cases with Large Vocabulary")
    print("=" * 80)
    
    try:
        from megatron.tokenizer.tokenizer import build_tokenizer
        
        args = create_mock_args(tokenizer_model_path)
        tokenizer = build_tokenizer(args)
        
        vocab_size = tokenizer.vocab_size
        print(f"Testing with vocabulary size: {vocab_size}")
        
        # Test edge token IDs
        edge_tests = [
            ("Minimum ID", 0),
            ("ID 1", 1),
            ("Middle ID", vocab_size // 2),
            ("Near maximum ID", vocab_size - 2),
            ("Maximum valid ID", vocab_size - 1),
        ]
        
        passed_tests = 0
        for test_name, token_id in edge_tests:
            try:
                decoded = tokenizer.detokenize([token_id])
                print(f"  {test_name} ({token_id}): {repr(decoded[:50])}{'...' if len(decoded) > 50 else ''}")
                passed_tests += 1
            except Exception as e:
                print(f"  {test_name} ({token_id}): ❌ Error: {e}")
        
        # Test invalid token IDs
        invalid_tests = [
            ("Negative ID", -1),
            ("Too large ID", vocab_size),
            ("Much too large ID", vocab_size + 1000),
        ]
        
        expected_failures = 0
        for test_name, token_id in invalid_tests:
            try:
                decoded = tokenizer.detokenize([token_id])
                print(f"  {test_name} ({token_id}): Unexpected success: {repr(decoded[:50])}")
            except Exception as e:
                print(f"  {test_name} ({token_id}): ✅ Expected failure: {type(e).__name__}")
                expected_failures += 1
        
        # Test large token sequences
        try:
            large_sequence = list(range(0, min(1000, vocab_size)))
            decoded = tokenizer.detokenize(large_sequence)
            print(f"  Large sequence (1000 tokens): ✅ Success, length={len(decoded)}")
            large_sequence_ok = True
        except Exception as e:
            print(f"  Large sequence (1000 tokens): ❌ Failed: {e}")
            large_sequence_ok = False
        
        # Test empty sequences
        try:
            decoded = tokenizer.detokenize([])
            print(f"  Empty sequence: ✅ Success: {repr(decoded)}")
            empty_sequence_ok = True
        except Exception as e:
            print(f"  Empty sequence: ❌ Failed: {e}")
            empty_sequence_ok = False
        
        # Summary
        total_edge_tests = len(edge_tests)
        total_invalid_tests = len(invalid_tests)
        
        print(f"\n--- Edge Cases Summary ---")
        print(f"✅ Valid edge cases: {passed_tests}/{total_edge_tests}")
        print(f"✅ Invalid cases handled: {expected_failures}/{total_invalid_tests}")
        print(f"✅ Large sequence test: {large_sequence_ok}")
        print(f"✅ Empty sequence test: {empty_sequence_ok}")
        
        return (passed_tests == total_edge_tests and 
                expected_failures == total_invalid_tests and 
                large_sequence_ok and empty_sequence_ok)
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility_with_megatron_features(tokenizer_model_path):
    """Test compatibility with Megatron-specific features."""
    print("\n" + "=" * 80)
    print("TEST 4: Compatibility with Megatron Features")
    print("=" * 80)
    
    try:
        from megatron.tokenizer.tokenizer import build_tokenizer
        
        args = create_mock_args(tokenizer_model_path)
        tokenizer = build_tokenizer(args)
        
        # Test special token properties
        special_tokens = [
            ('bos', 'Beginning of sequence'),
            ('eos', 'End of sequence'),
            ('pad', 'Padding'),
            ('eod', 'End of document'),
            ('cls', 'Classification'),
            ('sep', 'Separator'),
            ('mask', 'Mask'),
        ]
        
        print("Testing special token properties:")
        special_tokens_ok = 0
        for token_name, description in special_tokens:
            try:
                token_id = getattr(tokenizer, token_name)
                print(f"  {token_name.upper()} ({description}): ID={token_id}")
                if isinstance(token_id, int) and token_id >= 0:
                    special_tokens_ok += 1
            except Exception as e:
                print(f"  {token_name.upper()}: ❌ Error: {e}")
        
        # Test vocab properties
        print(f"\nTesting vocabulary properties:")
        try:
            vocab = tokenizer.vocab
            inv_vocab = tokenizer.inv_vocab
            vocab_size = tokenizer.vocab_size
            
            print(f"  Vocab dict size: {len(vocab)}")
            print(f"  Inverse vocab dict size: {len(inv_vocab)}")
            print(f"  Reported vocab size: {vocab_size}")
            
            vocab_consistent = (len(vocab) == vocab_size and len(inv_vocab) == vocab_size)
            print(f"  ✅ Vocabulary consistency: {vocab_consistent}")
            
        except Exception as e:
            print(f"  ❌ Vocabulary properties test failed: {e}")
            vocab_consistent = False
        
        # Test tokenize/detokenize interface
        print(f"\nTesting Megatron tokenizer interface:")
        try:
            test_texts = [
                "Simple test",
                "测试中文文本",
                "Mixed English and 中文 text",
                "Special chars: !@#$%^&*()",
            ]
            
            interface_tests_passed = 0
            for text in test_texts:
                try:
                    # Test tokenize method
                    tokens = tokenizer.tokenize(text)
                    # Test detokenize method
                    decoded = tokenizer.detokenize(tokens)
                    
                    if isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
                        interface_tests_passed += 1
                        print(f"  ✅ '{text[:20]}...': {len(tokens)} tokens")
                    else:
                        print(f"  ❌ '{text[:20]}...': Invalid token format")
                        
                except Exception as e:
                    print(f"  ❌ '{text[:20]}...': Error: {e}")
            
            interface_ok = interface_tests_passed == len(test_texts)
            
        except Exception as e:
            print(f"  ❌ Interface test failed: {e}")
            interface_ok = False
        
        # Summary
        print(f"\n--- Megatron Compatibility Summary ---")
        print(f"✅ Special tokens available: {special_tokens_ok}/{len(special_tokens)}")
        print(f"✅ Vocabulary consistency: {vocab_consistent}")
        print(f"✅ Interface compatibility: {interface_ok}")
        
        return special_tokens_ok >= 4 and vocab_consistent and interface_ok  # At least 4 special tokens should work
        
    except Exception as e:
        print(f"❌ Megatron compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Test large vocabulary compatibility for Llama8b tokenizer')
    parser.add_argument('--tokenizer-path', type=str, 
                       default='assets/checkpoints/Llama8b',
                       help='Path to Llama8b tokenizer directory')
    
    args = parser.parse_args()
    
    print("🧪 Large Vocabulary Compatibility Test for Llama8b Tokenizer")
    print("=" * 80)
    print(f"Tokenizer path: {args.tokenizer_path}")
    print(f"Expected vocabulary size: ~119,696")
    
    # Verify tokenizer path exists
    if not os.path.exists(args.tokenizer_path):
        print(f"❌ Tokenizer path does not exist: {args.tokenizer_path}")
        return 1
    
    # Run all tests
    test_results = []
    test_results.append(test_vocabulary_padding(args.tokenizer_path))
    test_results.append(test_memory_usage(args.tokenizer_path))
    test_results.append(test_edge_cases(args.tokenizer_path))
    test_results.append(test_compatibility_with_megatron_features(args.tokenizer_path))
    
    # Summary
    print("\n" + "=" * 80)
    print("LARGE VOCABULARY COMPATIBILITY TEST SUMMARY")
    print("=" * 80)
    passed = sum(test_results)
    total = len(test_results)
    
    test_names = [
        "Vocabulary Padding",
        "Memory Usage",
        "Edge Cases",
        "Megatron Compatibility"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nOverall result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All large vocabulary compatibility tests passed!")
        print("✅ Llama8b tokenizer is ready for Megatron integration!")
        return 0
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
