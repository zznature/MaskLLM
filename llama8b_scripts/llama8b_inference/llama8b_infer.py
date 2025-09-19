#!/usr/bin/env python3

import argparse
import os
import sys
import torch
import importlib
import json
from typing import Optional, List

# Project directories
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
EXT_PKGS_DIR = os.path.join(PROJECT_DIR, '.ext_pkgs')

# Globals for HF classes, set by ensure_hf_import()
AutoTokenizer = None
AutoModelForCausalLM = None


def ensure_hf_import() -> None:
    """Import transformers classes, falling back to project .ext_pkgs if missing.
    Keeps container-native packages preferred; only adds .ext_pkgs if import fails.
    """
    global AutoTokenizer, AutoModelForCausalLM
    if AutoTokenizer is not None and AutoModelForCausalLM is not None:
        return
    try:
        tm = importlib.import_module('transformers')
    except ImportError:
        if os.path.isdir(EXT_PKGS_DIR) and EXT_PKGS_DIR not in sys.path:
            sys.path.insert(0, EXT_PKGS_DIR)
        tm = importlib.import_module('transformers')
        print(f"[Info] transformers loaded from .ext_pkgs: {tm.__file__}")
    AutoTokenizer = getattr(tm, 'AutoTokenizer')
    AutoModelForCausalLM = getattr(tm, 'AutoModelForCausalLM')


def parse_args():
    parser = argparse.ArgumentParser(description="Llama8b inference with HF AutoTokenizer/AutoModelForCausalLM")
    parser.add_argument("--model", type=str, required=True, help="Model path or repo id (weights)")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path or repo id (defaults to --model)")
    parser.add_argument("--vocab", type=str, default=None, help="Optional path to vocab.txt for local custom tokenizer")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to a text file containing the prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true", help="Use sampling; otherwise greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"], help="Inference dtype")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loader")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--stop-text", action="append", default=[], help="Add a stop string; can be provided multiple times or comma-separated")
    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float16


def try_import_local_tokenizer(tokenizer_dir: str):
    """Try to import local custom tokenizer module at <dir>/llama8b/tokenizer.py.
    Returns the imported module if successful, else None.
    """
    if not tokenizer_dir or not os.path.isdir(tokenizer_dir):
        return None
    module_rel = os.path.join(tokenizer_dir, "llama8b", "tokenizer.py")
    if os.path.isfile(module_rel):
        if tokenizer_dir not in sys.path:
            sys.path.insert(0, tokenizer_dir)
        try:
            importlib.invalidate_caches()
            module = importlib.import_module("llama8b.tokenizer")
            print(f"[Info] Loaded local tokenizer module: {module.__file__}")
            return module
        except Exception as e:
            print(f"[Warn] Failed to import local tokenizer module: {e}")
    return None


def discover_local_vocab(tokenizer_dir: str, explicit_vocab: Optional[str]) -> Optional[str]:
    if explicit_vocab and os.path.isfile(explicit_vocab):
        return explicit_vocab
    if not tokenizer_dir or not os.path.isdir(tokenizer_dir):
        return None
    candidates = [
        os.path.join(tokenizer_dir, "vocab.txt"),
        os.path.join(tokenizer_dir, "llama8b", "vocab.txt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def normalize_stop_list(stop_args: List[str]) -> List[str]:
    out: List[str] = []
    for s in stop_args:
        if not s:
            continue
        parts = [x for x in s.split(',') if x]
        out.extend(parts)
    # de-duplicate preserving order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def load_tokenizer(tokenizer_id: str, trust_remote_code: bool, explicit_vocab: Optional[str]):
    """Prefer local custom tokenizer if present; else fall back to HF AutoTokenizer."""
    ensure_hf_import()
    local_module = try_import_local_tokenizer(tokenizer_id) if os.path.isdir(tokenizer_id) else None

    tok_cfg_path = os.path.join(tokenizer_id, "tokenizer_config.json") if os.path.isdir(tokenizer_id) else None
    if local_module is not None and tok_cfg_path and os.path.isfile(tok_cfg_path):
        try:
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            if tok_cfg.get("tokenizer_class") in {"Llama8bTokenizer", "llama8b.tokenizer.Llama8bTokenizer"}:
                return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False, trust_remote_code=True)
        except Exception:
            pass

    if local_module is not None and hasattr(local_module, "Llama8bTokenizer"):
        vocab_path = discover_local_vocab(tokenizer_id, explicit_vocab)
        if vocab_path is not None:
            try:
                tokenizer = getattr(local_module, "Llama8bTokenizer")(vocab_file=vocab_path)
                print(f"[Info] Using local Llama8bTokenizer with vocab: {vocab_path}")
                return tokenizer
            except Exception as e:
                print(f"[Warn] Failed to instantiate local Llama8bTokenizer: {e}")
        else:
            print("[Warn] Local tokenizer module found but no vocab.txt; pass --vocab to specify explicitly.")

    return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, trust_remote_code=trust_remote_code)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model_id = args.model
    tokenizer_id = args.tokenizer or model_id

    # Ensure HF classes available (container-native preferred, fallback to .ext_pkgs)
    ensure_hf_import()

    tokenizer = load_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code, explicit_vocab=args.vocab)

    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_tok = getattr(tokenizer, "eos_token", None)
        if eos_tok is not None:
            tokenizer.pad_token = eos_tok
        else:
            try:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            except Exception:
                pass

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    torch_dtype = get_torch_dtype(args.dtype)

    # Avoid Accelerate requirement: do not pass device_map, force low_cpu_mem_usage=False
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=False,
    )

    if device == "cuda":
        model = model.to("cuda")

    # Check vocab-size consistency and optionally resize small deltas (common when adding pad)
    tokenizer_vocab = len(tokenizer) if hasattr(tokenizer, "__len__") else None
    if tokenizer_vocab is not None and getattr(model.config, "vocab_size", None) is not None:
        if tokenizer_vocab != model.config.vocab_size:
            delta = tokenizer_vocab - model.config.vocab_size
            print(f"[Warn] vocab_size mismatch: model={model.config.vocab_size}, tokenizer={tokenizer_vocab} (delta={delta})")
            if delta > 0 and delta <= 64:
                try:
                    model.resize_token_embeddings(tokenizer_vocab)
                    print("[Info] Resized token embeddings to match tokenizer vocab size.")
                except Exception as e:
                    print(f"[Warn] Failed to resize token embeddings: {e}")

    # Sync generation config (no deprecation warnings)
    gc = model.generation_config
    if getattr(tokenizer, "eos_token_id", None) is not None:
        gc.eos_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token_id", None) is not None:
        gc.pad_token_id = tokenizer.pad_token_id
    gc.temperature = args.temperature
    gc.top_p = args.top_p
    if args.top_k is not None:
        gc.top_k = args.top_k
    gc.do_sample = args.do_sample
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        gc.repetition_penalty = args.repetition_penalty
    if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0:
        gc.no_repeat_ngram_size = args.no_repeat_ngram_size

    stop_texts = normalize_stop_list(args.stop_text)

    print(f"[Info] device={device}, dtype={torch_dtype}, max_new_tokens={args.max_new_tokens}")

    if args.prompt is None and args.prompt_file is None:
        print("[WARN] No prompt provided. Using a default prompt.")
        prompt_text = "You are a helpful assistant. Briefly introduce MaskLLM."
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    else:
        prompt_text = args.prompt

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    model.eval()
    use_autocast = device == "cuda" and torch_dtype in (torch.float16, torch.bfloat16)
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch_dtype) if use_autocast else torch.no_grad()

    with torch.no_grad():
        with autocast_ctx:
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
            )

    generated_ids = outputs[0][input_ids.shape[-1]:]
    text_out = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Apply simple stop-text trimming post-hoc
    if stop_texts:
        cut = len(text_out)
        for s in stop_texts:
            idx = text_out.find(s)
            if idx != -1:
                cut = min(cut, idx)
        text_out = text_out[:cut]

    print("===== Prompt =====")
    print(prompt_text)
    print("===== Generation =====")
    print(text_out)


if __name__ == "__main__":
    sys.exit(main())
