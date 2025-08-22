# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""
Llama8b HuggingFace checkpoint loader for MaskLLM/Megatron.

This loader is specifically designed for the Llama8b model which features:
- Large vocabulary size (119,696 tokens)
- Custom Llama8bTokenizer
- Standard Llama architecture with custom tokenization

Based on loader_llama2_hf.py with modifications for Llama8b specifics.
"""

import json
import os
import sys
import torch
import transformers
from tqdm import tqdm
import types


class MetaData:
    """Metadata object for saver."""
    def __init__(self, args):
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.seq_length = args.seq_length
        self.num_attention_heads = args.num_attention_heads
        self.max_position_embeddings = args.max_position_embeddings
        self.position_embedding_type = getattr(args, 'position_embedding_type', 'rope')
        # Use supported tokenizer type for saver (will use Llama8bTokenizer at runtime)
        self.tokenizer_type = 'Llama2Tokenizer'
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = args.padded_vocab_size
        self.params_dtype = args.params_dtype
        self.tensor_model_parallel_size = args.tensor_model_parallel_size
        self.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        
        # Additional attributes needed by saver
        self.make_vocab_size_divisible_by = getattr(args, 'make_vocab_size_divisible_by', 128)
        self.swiglu = getattr(args, 'swiglu', True)
        self.normalization = getattr(args, 'normalization', 'RMSNorm')
        self.use_rotary_position_embeddings = getattr(args, 'use_rotary_position_embeddings', True)
        self.rotary_percent = getattr(args, 'rotary_percent', 1.0)
        self.rotary_base = getattr(args, 'rotary_base', 10000)
        self.untie_embeddings_and_output_weights = getattr(args, 'untie_embeddings_and_output_weights', True)
        self.num_query_groups = getattr(args, 'num_query_groups', args.num_attention_heads)
        
        # More saver attributes
        self.output_layer = True  # Llama models have output layer
        self.linear_bias = False  # Llama models don't use linear bias
        self.model_type = 'GPT'   # Model type for GPT models
        self.bert_binary_head = False  # Not applicable for GPT models


def add_arguments(parser):
    group = parser.add_argument_group(title='Llama8b HF loader.')

    group.add_argument('--true-vocab-size', type=int, default=None,
                       help='Original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--tokenizer-model', required=True,
                       help='Path to Llama8b tokenizer model directory.')
    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')


def verify_transformers_version():
    """Verify transformers version compatibility."""
    major, minor, patch = map(int, transformers.__version__.split('.'))
    assert major >= 4 and minor >= 31


def load_args_from_checkpoint(args):
    """Load and adapt arguments from Llama8b HF checkpoint."""

    # Read Llama8b args from config.json
    load_path = getattr(args, 'load_dir', None) or getattr(args, 'load', None)
    llama8b_args_path = os.path.join(load_path, "config.json")
    with open(llama8b_args_path) as f:
        llama8b_args = json.load(f)

    print(f"Loading Llama8b config from: {llama8b_args_path}")
    print(f"Vocabulary size: {llama8b_args['vocab_size']}")

    # Update Megatron args for Llama8b
    args.seq_length = 4096
    args.max_position_embeddings = 4096
    args.hidden_size = llama8b_args["hidden_size"]
    args.num_attention_heads = llama8b_args["num_attention_heads"]
    args.num_layers = llama8b_args["num_hidden_layers"]
    args.global_batch_size = 1024
    args.norm_epsilon = llama8b_args["rms_norm_eps"]
    args.iteration = 1  # '0', 'release' don't work
    args.add_position_embedding = False
    args.use_rotary_position_embeddings = True
    args.swiglu = True
    
    # Key difference: Use Llama8bTokenizer instead of Llama2Tokenizer
    args.tokenizer_type = "Llama8bTokenizer"
    
    args.fp16 = True
    args.normalization = "RMSNorm"
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = True
    
    # Handle large vocabulary size (119,696)
    args.vocab_size = llama8b_args["vocab_size"]
    args.padded_vocab_size = llama8b_args["vocab_size"]
    args.llama8b = llama8b_args  # Store original config for reference
    args.ffn_hidden_size = llama8b_args["intermediate_size"]

    # Group Query Attention support (if present)
    if "num_key_value_heads" in llama8b_args:
        args.group_query_attention = True
        args.num_query_groups = llama8b_args["num_key_value_heads"]

    print(f"Configured for Llama8b:")
    print(f"  - Vocab size: {args.vocab_size}")
    print(f"  - Hidden size: {args.hidden_size}")
    print(f"  - Num layers: {args.num_layers}")
    print(f"  - Num attention heads: {args.num_attention_heads}")
    print(f"  - Tokenizer type: {args.tokenizer_type}")


def set_preprocess_state(args, model, hf_model):
    """Set embedding params for Llama8b."""
    print("Setting embedding weights...")
    model.language_model.embedding.word_embeddings.weight.data.copy_(
        hf_model.model.embed_tokens.weight)
    print(f"Embedding shape: {hf_model.model.embed_tokens.weight.shape}")


def set_postprocess_state(args, model, hf_model):
    """Set output layer & norm params for Llama8b."""
    print("Setting final norm and output layer weights...")
    model.language_model.encoder.final_norm.weight.data.copy_(hf_model.model.norm.weight)
    model.language_model.output_layer.weight.data.copy_(hf_model.lm_head.weight)
    print(f"Output layer shape: {hf_model.lm_head.weight.shape}")


def set_attn_state(args, layer, hf_layer):
    """Set self-attention params for Llama8b."""

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads // tp
    ng = (args.num_query_groups if args.group_query_attention \
        else args.num_attention_heads) // tp
    dim = args.kv_channels
    assert nh % ng == 0

    # Copy weights (re-order dimensions for Megatron).
    attn.query_key_value.weight.data.copy_(torch.cat([ 
        hf_attn.q_proj.weight.reshape((ng, dim*nh//ng, -1)),
        hf_attn.k_proj.weight.reshape((ng, dim, -1)),
        hf_attn.v_proj.weight.reshape((ng, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))
    attn.dense.weight.data.copy_(hf_attn.o_proj.weight)


def set_mlp_state(args, layer, hf_layer):
    """Set MLP params for Llama8b."""

    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    # Copy weights.
    mlp.dense_h_to_4h.weight.data.copy_(torch.cat([
        hf_mlp.gate_proj.weight,
        hf_mlp.up_proj.weight,
    ], dim=0))
    mlp.dense_4h_to_h.weight.data.copy_(hf_mlp.down_proj.weight)


def set_layer_state(args, model, hf_model, layer_idx):
    """Set transformer layer params for Llama8b."""

    layer = model.language_model.encoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    # Set layer norm weights.
    layer.input_norm.weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.post_attention_norm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)

    # Set attention and MLP weights.
    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)


def load_checkpoint_to_model(args, model):
    """Load Llama8b HF checkpoint to Megatron model."""

    load_path = getattr(args, 'load_dir', None) or getattr(args, 'load', None)
    print(f"Loading Llama8b HuggingFace checkpoint from: {load_path}")

    # Verify transformers version
    verify_transformers_version()

    # Load Llama8b HF model
    print("Loading HuggingFace Llama8b model...")
    try:
        # Try loading with trust_remote_code for custom tokenizer
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            load_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Llama8b model loaded successfully with trust_remote_code=True")
    except Exception as e:
        print(f"⚠️  Failed to load with trust_remote_code=True: {e}")
        print("Trying fallback loading method...")
        
        # Fallback: load without trust_remote_code
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            load_path, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("✅ Llama8b model loaded with fallback method")

    print("Model loading completed.")

    # Verify model architecture
    print(f"Model config: {hf_model.config}")
    print(f"Vocabulary size: {hf_model.config.vocab_size}")
    
    if hf_model.config.vocab_size != args.vocab_size:
        print(f"⚠️  Vocab size mismatch: config={hf_model.config.vocab_size}, args={args.vocab_size}")

    # Set model weights layer by layer
    print("Converting weights to Megatron format...")

    # Set embedding weights
    set_preprocess_state(args, model, hf_model)

    # Set transformer layer weights
    for layer_idx in tqdm(range(args.num_layers), desc="Converting layers"):
        set_layer_state(args, model, hf_model, layer_idx)

    # Set output layer weights
    set_postprocess_state(args, model, hf_model)

    print("✅ Checkpoint conversion completed successfully!")

    # Clean up HF model to save memory
    del hf_model
    torch.cuda.empty_cache()


def _queue_put(queue, name, msg):
    if msg is not None:
        queue.put(msg)


def load_checkpoint(queue, args):
    """Main loader function for Llama8b checkpoint conversion."""
    
    try:
        print("Starting Llama8b checkpoint loading...")
        
        # Load checkpoint config
        load_args_from_checkpoint(args)

        # Add megatron args that are dataset/training specific,
        # and aren't saved in the checkpoint.
        args.data_parallel_size = 1
        args.sequence_parallel = False
        args.micro_batch_size = 1
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        args.rank = 0
        args.world_size = 1

        # Model parallel args.
        args.tensor_model_parallel_size = 1  # Will be set by util.py
        args.pipeline_model_parallel_size = 1

        if args.megatron_path is not None:
            sys.path.insert(0, args.megatron_path)
        else:
            sys.path.insert(0, os.getcwd())

        try:
            from megatron.arguments import parse_args, validate_args
            from megatron.global_vars import set_args, set_global_variables
            from megatron.model import module
            from megatron.core import mpu
            from megatron.core.enums import ModelType
            from megatron import fused_kernels
        except ModuleNotFoundError:
            print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
            queue.put("exit")
            return

        # We want all arguments to come from us.
        sys.argv = ['script.py',
                    '--no-masked-softmax-fusion',
                    '--no-bias-gelu-fusion', 
                    '--no-bias-dropout-fusion',
                    '--no-async-tensor-model-parallel-allreduce',
                    '--use-cpu-initialization',
                    '--micro-batch-size', '1',
                    '--no-load-optim',
                    '--no-load-rng',
                    '--no-save-optim',
                    '--no-save-rng',
                    '--no-initialization',
                    '--load', getattr(args, 'load_dir', None) or getattr(args, 'load', None)
                    ]

        margs = parse_args()
        margs.tokenizer_model = args.tokenizer_model
        margs.model_type = 'GPT'  # Set model type for Llama8b
        
        # Set additional required attributes
        if not hasattr(margs, 'retro_add_retriever'):
            margs.retro_add_retriever = False
        
        load_args_from_checkpoint(margs)

        # Arguments do sanity checks on the world size, but we don't care,
        # so trick it into thinking we are plenty of processes.
        margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

        margs = validate_args(margs)

        # Initialize distributed environment first
        import torch.distributed as dist
        import os
        if not dist.is_initialized():
            # Set environment variables for single process
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            
            # Initialize with CPU backend for single process checkpoint conversion
            try:
                dist.init_process_group(backend='gloo', rank=0, world_size=1)
            except Exception as e:
                print(f"Warning: Could not initialize distributed environment: {e}")
                print("Continuing without distributed initialization...")
        
        # Initialize model parallel
        try:
            mpu.initialize_model_parallel(margs.tensor_model_parallel_size, margs.pipeline_model_parallel_size)
        except Exception as e:
            print(f"Warning: Could not initialize model parallel: {e}")
            print("Continuing without model parallel...")
        
        # Set global variables
        set_global_variables(margs, build_tokenizer=False)

        # Import model_provider
        from pretrain_gpt import model_provider

        # Build model
        print("Building Megatron model...")
        model = model_provider(True, True).to(margs.params_dtype)

        # Load checkpoint
        load_checkpoint_to_model(margs, model)

        # Send model to queue
        print("Sending model data to conversion queue...")
        
        # Create metadata object for saver
        metadata = MetaData(margs)
        
        # Send metadata (saver expects metadata directly, not "done")
        queue.put(metadata)

        # Send model state (filter out _extra_state and None parameters)
        print("Sending model state dict to queue...")
        sent_params = 0
        skipped_params = 0
        
        for layer_name, param in model.state_dict().items():
            # Skip _extra_state and None parameters
            if param is None or '_extra_state' in layer_name:
                skipped_params += 1
                continue
                
            msg = {
                "name": layer_name,
                layer_name: param.half() if param.dtype != torch.float16 else param
            }
            _queue_put(queue, f"model_{layer_name}", msg)
            sent_params += 1
            
            # Print progress for every 50 parameters
            if sent_params % 50 == 0:
                print(f"Sent {sent_params} parameters...")

        print(f"Completed sending {sent_params} parameters (skipped {skipped_params} None/_extra_state parameters)")
        queue.put("done")

    except Exception as e:
        print(f"❌ Error during checkpoint loading: {e}")
        import traceback
        traceback.print_exc()
        queue.put("exit")
        return
