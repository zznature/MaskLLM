#!/usr/bin/env python3

"""
Simple test to verify the detokenize fix for Llama8b tokenizer.
Tests the specific issue that was causing the edge case failure.
"""

import os
import sys

# Add the tokenizer path
tokenizer_path = 'assets/checkpoints/Llama8b'
sys.path.insert(0, tokenizer_path)

def test_detokenize_fix():
    """Test the specific detokenize issue that was fixed."""
    
    print("🧪 Testing Llama8b Tokenizer Detokenize Fix")
    print("=" * 60)
    
    try:
        # Import the tokenizer directly
        from llama8b.tokenizer import Llama8bTokenizer
        
        # Load tokenizer
        vocab_file = os.path.join(tokenizer_path, 'vocab.txt')
        tokenizer = Llama8bTokenizer(vocab_file=vocab_file)
        
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Vocabulary size: {tokenizer.vocab_size}")
        
        # Test the specific case that was failing
        print(f"\n--- Testing Large Sequence Detokenization ---")
        
        # Create a sequence that includes various token types
        test_sequence = list(range(0, min(100, tokenizer.vocab_size)))
        
        print(f"Testing sequence of {len(test_sequence)} tokens...")
        
        # Test using raw_decode (which should work)
        try:
            decoded_raw = tokenizer.raw_decode(test_sequence)
            print(f"✅ raw_decode successful: {len(decoded_raw)} chars")
        except Exception as e:
            print(f"❌ raw_decode failed: {e}")
            return False
        
        # Test using the standard decode method
        try:
            decoded_standard = tokenizer.decode(test_sequence)
            print(f"✅ decode successful: {len(decoded_standard)} chars")
        except Exception as e:
            print(f"❌ decode failed: {e}")
            return False
        
        # Test edge cases that previously failed
        print(f"\n--- Testing Edge Cases ---")
        
        edge_cases = [
            ("Empty sequence", []),
            ("Single token", [0]),
            ("BOS token", [tokenizer.bos_id]),
            ("EOS token", [tokenizer.eos_id]),
            ("Mix of regular and special", [0, 1, 2, tokenizer.bos_id, tokenizer.eos_id]),
        ]
        
        success_count = 0
        for case_name, token_ids in edge_cases:
            try:
                decoded = tokenizer.raw_decode(token_ids)
                print(f"✅ {case_name}: success")
                success_count += 1
            except Exception as e:
                print(f"❌ {case_name}: failed - {e}")
        
        print(f"\n--- Results ---")
        print(f"Edge cases passed: {success_count}/{len(edge_cases)}")
        
        if success_count == len(edge_cases):
            print("🎉 All tests passed! Detokenize fix is working correctly.")
            return True
        else:
            print("⚠️  Some edge cases failed.")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_megatron_wrapper_fix():
    """Test the Megatron wrapper fix directly."""
    
    print(f"\n{'='*60}")
    print("🧪 Testing Megatron Wrapper Detokenize Fix")
    print("=" * 60)
    
    try:
        # Import our wrapper
        sys.path.insert(0, '.')
        from megatron.tokenizer.llama8b_tokenizer import _Llama8bTokenizer
        
        # Create wrapper instance
        wrapper = _Llama8bTokenizer(tokenizer_path)
        
        print(f"✅ Wrapper loaded successfully")
        print(f"   Vocabulary size: {wrapper.vocab_size}")
        
        # Test the fixed detokenize method
        test_sequence = list(range(0, min(50, wrapper.vocab_size)))
        
        print(f"\nTesting wrapper detokenize with {len(test_sequence)} tokens...")
        
        try:
            decoded = wrapper.detokenize(test_sequence)
            print(f"✅ Wrapper detokenize successful: {len(decoded)} chars")
            print(f"   Sample output: {repr(decoded[:100])}...")
            return True
        except Exception as e:
            print(f"❌ Wrapper detokenize failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    # Check if tokenizer files exist
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer path not found: {tokenizer_path}")
        return 1
    
    vocab_file = os.path.join(tokenizer_path, 'vocab.txt')
    if not os.path.exists(vocab_file):
        print(f"❌ Vocab file not found: {vocab_file}")
        return 1
    
    # Run tests
    test1_result = test_detokenize_fix()
    test2_result = test_megatron_wrapper_fix()
    
    print(f"\n{'='*60}")
    print("📋 FINAL RESULTS")
    print("=" * 60)
    print(f"Original tokenizer test: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"Megatron wrapper test: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All detokenize fix tests passed!")
        print("The edge case issue has been successfully resolved.")
        return 0
    else:
        print("\n⚠️  Some tests failed. The fix may need further work.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
