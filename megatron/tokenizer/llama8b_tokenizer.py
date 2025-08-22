# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Llama8b tokenizer."""

import os
import sys
import importlib
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer


class _Llama8bTokenizer(MegatronTokenizer):
    """Llama8b tokenizer wrapper for Megatron.
    
    This class wraps the custom Llama8bTokenizer to make it compatible with
    Megatron's tokenizer interface.
    """
    
    def __init__(self, tokenizer_model):
        """Initialize the Llama8b tokenizer.
        
        Args:
            tokenizer_model (str): Path to the tokenizer model directory
        """
        super().__init__(tokenizer_model)
        
        # Import and load the Llama8bTokenizer
        self._llama8b_tokenizer = self._load_llama8b_tokenizer(tokenizer_model)
        
        # Initialize vocabulary mappings
        self._vocab = self._llama8b_tokenizer.get_vocab()
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        
        # Store special token IDs
        self._bos_id = self._llama8b_tokenizer.bos_id
        self._eos_id = self._llama8b_tokenizer.eos_id
        self._unk_id = self._llama8b_tokenizer.unk_id
        
        # For compatibility with Megatron interface, we'll use eos as pad if no pad token exists
        self._pad_id = getattr(self._llama8b_tokenizer, 'pad_id', self._eos_id)
        
        # Set additional special tokens for Megatron compatibility
        self._cls_id = self._unk_id  # Llama doesn't have CLS, use UNK
        self._sep_id = self._unk_id  # Llama doesn't have SEP, use UNK
        self._mask_id = self._unk_id  # Llama doesn't have MASK, use UNK
        self._eod_id = self._eos_id  # Use EOS as end-of-document
    
    def _load_llama8b_tokenizer(self, tokenizer_path):
        """Load the Llama8bTokenizer from the specified path.
        
        Args:
            tokenizer_path (str): Path to the tokenizer model directory
            
        Returns:
            Llama8bTokenizer: The loaded tokenizer instance
        """
        # First, try to load from the model directory structure
        llama8b_dir = os.path.join(tokenizer_path, 'llama8b')
        tokenizer_file = os.path.join(llama8b_dir, 'tokenizer.py')
        
        if os.path.exists(tokenizer_file):
            # Add the directory to Python path temporarily
            if tokenizer_path not in sys.path:
                sys.path.insert(0, tokenizer_path)
            
            try:
                # Import the tokenizer module
                importlib.invalidate_caches()
                tokenizer_module = importlib.import_module('llama8b.tokenizer')
                
                # Find vocab.txt file
                vocab_file = os.path.join(tokenizer_path, 'vocab.txt')
                if not os.path.exists(vocab_file):
                    vocab_file = os.path.join(llama8b_dir, 'vocab.txt')
                
                if not os.path.exists(vocab_file):
                    raise FileNotFoundError(f"vocab.txt not found in {tokenizer_path} or {llama8b_dir}")
                
                # Create tokenizer instance
                tokenizer = tokenizer_module.Llama8bTokenizer(vocab_file=vocab_file)
                print(f"[INFO] Successfully loaded Llama8bTokenizer from {tokenizer_file}")
                print(f"[INFO] Using vocab file: {vocab_file}")
                print(f"[INFO] Vocabulary size: {tokenizer.vocab_size}")
                
                return tokenizer
                
            except Exception as e:
                print(f"[ERROR] Failed to load Llama8bTokenizer from {tokenizer_file}: {e}")
                raise
        else:
            # Try to use transformers AutoTokenizer as fallback
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                print(f"[INFO] Loaded Llama8bTokenizer via AutoTokenizer from {tokenizer_path}")
                print(f"[INFO] Vocabulary size: {len(tokenizer)}")
                return tokenizer
            except Exception as e:
                print(f"[ERROR] Failed to load tokenizer via AutoTokenizer: {e}")
                raise FileNotFoundError(f"Could not load Llama8bTokenizer from {tokenizer_path}")
    
    @property
    def vocab_size(self):
        """Return the vocabulary size."""
        return len(self._vocab)
    
    @property
    def vocab(self):
        """Return the vocabulary mapping (token -> id)."""
        return self._vocab
    
    @property
    def inv_vocab(self):
        """Return the inverse vocabulary mapping (id -> token)."""
        return self._inv_vocab
    
    def tokenize(self, text):
        """Tokenize text into token IDs.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            list: List of token IDs
        """
        # Use the underlying tokenizer's encode method
        if hasattr(self._llama8b_tokenizer, 'encode'):
            return self._llama8b_tokenizer.encode(text)
        elif hasattr(self._llama8b_tokenizer, '__call__'):
            # For HuggingFace tokenizers
            return self._llama8b_tokenizer(text, add_special_tokens=False)['input_ids']
        else:
            # Fallback: manual tokenization
            tokens = self._llama8b_tokenizer._tokenize(text)
            return [self._llama8b_tokenizer._convert_token_to_id(token) for token in tokens]
    
    def detokenize(self, token_ids):
        """Convert token IDs back to text.
        
        Args:
            token_ids (list): List of token IDs
            
        Returns:
            str: Decoded text
        """
        # Use the underlying tokenizer's decode method
        if hasattr(self._llama8b_tokenizer, 'decode'):
            return self._llama8b_tokenizer.decode(token_ids, skip_special_tokens=False)
        elif hasattr(self._llama8b_tokenizer, 'raw_decode'):
            return self._llama8b_tokenizer.raw_decode(token_ids)
        else:
            # Fallback: manual detokenization with bug fix
            tokens = []
            for id in token_ids:
                if id in self._llama8b_tokenizer.decoder:
                    tokens.append(self._llama8b_tokenizer.decoder[id])
                elif id in self._llama8b_tokenizer._byte_decoder:
                    # Fix the bug: _byte_decoder returns index, we need the actual token
                    byte_index = self._llama8b_tokenizer._byte_decoder[id]
                    tokens.append(self._llama8b_tokenizer.byte_list[byte_index])
                elif id == self._llama8b_tokenizer.eos_id:
                    tokens.append(self._llama8b_tokenizer.eos_token)
                elif id == self._llama8b_tokenizer.bos_id:
                    tokens.append(self._llama8b_tokenizer.bos_token)
                else:
                    tokens.append(self._llama8b_tokenizer.unk_token)
            return self._llama8b_tokenizer.convert_tokens_to_string(tokens)
    
    # Special token properties for Megatron compatibility
    
    @property
    def cls(self):
        """Return CLS token ID (not used in Llama, returns UNK)."""
        return self._cls_id
    
    @property
    def sep(self):
        """Return SEP token ID (not used in Llama, returns UNK)."""
        return self._sep_id
    
    @property
    def pad(self):
        """Return PAD token ID."""
        return self._pad_id
    
    @property
    def bos(self):
        """Return beginning-of-sequence token ID."""
        return self._bos_id
    
    @property
    def eos(self):
        """Return end-of-sequence token ID."""
        return self._eos_id
    
    @property
    def eod(self):
        """Return end-of-document token ID (same as EOS for Llama)."""
        return self._eod_id
    
    @property
    def mask(self):
        """Return MASK token ID (not used in Llama, returns UNK)."""
        return self._mask_id
    
    @property
    def additional_special_tokens_ids(self):
        """Return additional special token IDs (none for Llama8b)."""
        return None
