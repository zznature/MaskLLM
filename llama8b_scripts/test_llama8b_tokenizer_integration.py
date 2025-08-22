#!/usr/bin/env python3

"""
Test script for Llama8b tokenizer integration with Megatron.

This script tests:
1. Tokenizer loading through Megatron interface
2. Basic encoding/decoding functionality  
3. Special token handling
4. Large vocabulary size compatibility
5. Consistency with original Llama8bTokenizer
"""

import os
import sys
import argparse
import torch

# Add MaskLLM root to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
sys.path.insert(0, PROJECT_DIR)

def create_mock_args(tokenizer_model_path):
    """Create mock arguments for Megatron tokenizer initialization."""
    class MockArgs:
        def __init__(self):
            self.tokenizer_type = 'Llama8bTokenizer'
            self.tokenizer_model = tokenizer_model_path
            self.rank = 0
            self.make_vocab_size_divisible_by = 128
            self.tensor_model_parallel_size = 1
            self.padded_vocab_size = None
    
    return MockArgs()

def test_megatron_tokenizer_loading(tokenizer_model_path):
    """Test loading Llama8b tokenizer through Megatron interface."""
    print("=" * 60)
    print("TEST 1: Megatron Tokenizer Loading")
    print("=" * 60)
    
    try:
        # Import Megatron tokenizer builder
        from megatron.tokenizer.tokenizer import build_tokenizer
        
        # Create mock args
        args = create_mock_args(tokenizer_model_path)
        
        # Build tokenizer
        print(f"Loading tokenizer from: {tokenizer_model_path}")
        tokenizer = build_tokenizer(args)
        
        print(f"✅ Successfully loaded tokenizer: {type(tokenizer).__name__}")
        print(f"✅ Vocabulary size: {tokenizer.vocab_size}")
        print(f"✅ Padded vocabulary size: {args.padded_vocab_size}")
        
        # Check if vocab size is large (should be ~119696)
        if tokenizer.vocab_size > 100000:
            print(f"✅ Large vocabulary detected: {tokenizer.vocab_size}")
        else:
            print(f"⚠️  Unexpected vocabulary size: {tokenizer.vocab_size}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_basic_functionality(tokenizer):
    """Test basic encoding and decoding functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Basic Encoding/Decoding")
    print("=" * 60)
    
    test_texts = [
        "Hello, world!",
        "This is a test of the Llama8b tokenizer.",
        "MaskLLM是一个稀疏化训练框架。",  # Mixed language
        "The quick brown fox jumps over the lazy dog.",
        "Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
    ]
    
    success_count = 0
    for i, text in enumerate(test_texts):
        try:
            print(f"\nTest case {i+1}: {repr(text)}")
            
            # Encode
            token_ids = tokenizer.tokenize(text)
            print(f"  Encoded: {len(token_ids)} tokens")
            print(f"  Token IDs (first 10): {token_ids[:10]}")
            
            # Decode
            decoded_text = tokenizer.detokenize(token_ids)
            print(f"  Decoded: {repr(decoded_text)}")
            
            # Check consistency
            if text in decoded_text or decoded_text in text:
                print("  ✅ Encoding/decoding consistent")
                success_count += 1
            else:
                print("  ⚠️  Encoding/decoding inconsistent")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Basic functionality test: {success_count}/{len(test_texts)} passed")
    return success_count == len(test_texts)

def test_special_tokens(tokenizer):
    """Test special token handling."""
    print("\n" + "=" * 60)
    print("TEST 3: Special Token Handling")
    print("=" * 60)
    
    special_tokens = {
        'bos': tokenizer.bos,
        'eos': tokenizer.eos,
        'pad': tokenizer.pad,
        'eod': tokenizer.eod,
    }
    
    print("Special token IDs:")
    for name, token_id in special_tokens.items():
        print(f"  {name.upper()}: {token_id}")
        
        # Test decoding special tokens
        try:
            decoded = tokenizer.detokenize([token_id])
            print(f"    Decoded: {repr(decoded)}")
        except Exception as e:
            print(f"    ❌ Decode error: {e}")
    
    # Test text with special tokens
    test_text = "Hello world"
    try:
        # Add BOS and EOS
        token_ids = [tokenizer.bos] + tokenizer.tokenize(test_text) + [tokenizer.eos]
        decoded = tokenizer.detokenize(token_ids)
        print(f"\nWith BOS/EOS tokens:")
        print(f"  Input: {repr(test_text)}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Decoded: {repr(decoded)}")
        print("  ✅ Special token handling works")
        return True
    except Exception as e:
        print(f"  ❌ Special token test failed: {e}")
        return False

def test_original_tokenizer_consistency(tokenizer, tokenizer_model_path):
    """Test consistency with original Llama8bTokenizer."""
    print("\n" + "=" * 60)
    print("TEST 4: Original Tokenizer Consistency")
    print("=" * 60)
    
    try:
        # Load original tokenizer
        sys.path.insert(0, tokenizer_model_path)
        import importlib
        importlib.invalidate_caches()
        from llama8b.tokenizer import Llama8bTokenizer
        
        vocab_file = os.path.join(tokenizer_model_path, 'vocab.txt')
        if not os.path.exists(vocab_file):
            vocab_file = os.path.join(tokenizer_model_path, 'llama8b', 'vocab.txt')
        
        original_tokenizer = Llama8bTokenizer(vocab_file=vocab_file)
        
        print(f"✅ Loaded original tokenizer")
        print(f"  Original vocab size: {original_tokenizer.vocab_size}")
        print(f"  Megatron vocab size: {tokenizer.vocab_size}")
        
        # Test consistency
        test_texts = [
            "Hello world",
            "This is a test",
            "特殊测试文本",
        ]
        
        consistent_count = 0
        for text in test_texts:
            try:
                # Original tokenizer
                orig_ids = original_tokenizer.encode(text, add_special_tokens=False)
                orig_decoded = original_tokenizer.decode(orig_ids)
                
                # Megatron tokenizer
                mega_ids = tokenizer.tokenize(text)
                mega_decoded = tokenizer.detokenize(mega_ids)
                
                print(f"\nTest: {repr(text)}")
                print(f"  Original IDs: {orig_ids}")
                print(f"  Megatron IDs: {mega_ids}")
                
                if orig_ids == mega_ids:
                    print("  ✅ Token IDs match")
                    consistent_count += 1
                else:
                    print("  ⚠️  Token IDs differ")
                    
            except Exception as e:
                print(f"  ❌ Consistency test error: {e}")
        
        print(f"\n✅ Consistency test: {consistent_count}/{len(test_texts)} passed")
        return consistent_count == len(test_texts)
        
    except Exception as e:
        print(f"❌ Could not load original tokenizer: {e}")
        return False

def test_large_vocabulary_compatibility(tokenizer):
    """Test large vocabulary size handling."""
    print("\n" + "=" * 60)
    print("TEST 5: Large Vocabulary Compatibility")
    print("=" * 60)
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Test edge token IDs
    test_ids = [0, 1, vocab_size - 2, vocab_size - 1]
    
    print("Testing edge token IDs:")
    for token_id in test_ids:
        try:
            decoded = tokenizer.detokenize([token_id])
            print(f"  ID {token_id}: {repr(decoded)}")
        except Exception as e:
            print(f"  ID {token_id}: ❌ Error: {e}")
    
    # Test memory usage (simple check)
    try:
        # This should not cause memory issues
        large_text = "Hello world! " * 1000
        token_ids = tokenizer.tokenize(large_text)
        decoded = tokenizer.detokenize(token_ids)
        print(f"✅ Large text processing successful: {len(token_ids)} tokens")
        return True
    except Exception as e:
        print(f"❌ Large text processing failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Llama8b tokenizer integration')
    parser.add_argument('--tokenizer-path', type=str, 
                       default='assets/checkpoints/Llama8b',
                       help='Path to Llama8b tokenizer directory')
    
    args = parser.parse_args()
    
    print("🧪 Llama8b Tokenizer Integration Test")
    print("=" * 60)
    print(f"Tokenizer path: {args.tokenizer_path}")
    
    # Verify tokenizer path exists
    if not os.path.exists(args.tokenizer_path):
        print(f"❌ Tokenizer path does not exist: {args.tokenizer_path}")
        return 1
    
    # Test 1: Load tokenizer
    tokenizer = test_megatron_tokenizer_loading(args.tokenizer_path)
    if tokenizer is None:
        print("❌ Cannot proceed without tokenizer")
        return 1
    
    # Run all tests
    test_results = []
    test_results.append(test_basic_functionality(tokenizer))
    test_results.append(test_special_tokens(tokenizer))
    test_results.append(test_original_tokenizer_consistency(tokenizer, args.tokenizer_path))
    test_results.append(test_large_vocabulary_compatibility(tokenizer))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Llama8b tokenizer integration successful!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
