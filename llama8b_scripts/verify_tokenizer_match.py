#!/usr/bin/env python3
import argparse, os, json, sys, importlib
from typing import Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
ext_pkgs = os.path.join(PROJECT_DIR, '.ext_pkgs')


def load_transformers():
    try:
        import transformers  # type: ignore
        src = transformers.__file__
    except Exception:
        if os.path.isdir(ext_pkgs) and ext_pkgs not in sys.path:
            sys.path.insert(0, ext_pkgs)
        import transformers  # type: ignore
        src = transformers.__file__
        print(f"[Info] transformers loaded from .ext_pkgs: {src}")
    return transformers


def list_tokenizer_files(tok_dir: str):
    names = [
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'vocab.txt', 'merges.txt', 'spiece.model', 'tokenizer.model'
    ]
    print('=== Tokenizer files present ===')
    for n in names:
        p = os.path.join(tok_dir, n)
        if os.path.isfile(p):
            print(f'  {n}  ({os.path.getsize(p)} bytes)')
    # local custom module?
    local_mod = os.path.join(tok_dir, 'llama8b', 'tokenizer.py')
    if os.path.isfile(local_mod):
        print(f'  llama8b/tokenizer.py  ({os.path.getsize(local_mod)} bytes)')


def read_vocab_txt_count(tok_dir: str) -> Optional[int]:
    vp = os.path.join(tok_dir, 'vocab.txt')
    if not os.path.isfile(vp):
        return None
    try:
        with open(vp, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def get_tokenizer_len(tok) -> Optional[int]:
    # Try common ways
    if hasattr(tok, '__len__'):
        try:
            return len(tok)  # type: ignore
        except Exception:
            pass
    if hasattr(tok, 'vocab_size'):
        try:
            return int(tok.vocab_size)  # type: ignore
        except Exception:
            pass
    if hasattr(tok, 'vocab') and isinstance(getattr(tok, 'vocab'), dict):
        try:
            return len(getattr(tok, 'vocab'))
        except Exception:
            pass
    return None


def load_local_custom_tokenizer(tok_dir: str, explicit_vocab: Optional[str]):
    mod_path = os.path.join(tok_dir, 'llama8b', 'tokenizer.py')
    if not os.path.isfile(mod_path):
        return None
    if tok_dir not in sys.path:
        sys.path.insert(0, tok_dir)
    importlib.invalidate_caches()
    try:
        m = importlib.import_module('llama8b.tokenizer')
        cls = getattr(m, 'Llama8bTokenizer', None)
        if cls is None:
            print('[Warn] local llama8b.tokenizer found but Llama8bTokenizer class missing')
            return None
        # Discover vocab
        vp = explicit_vocab or os.path.join(tok_dir, 'vocab.txt')
        if not os.path.isfile(vp):
            alt = os.path.join(tok_dir, 'llama8b', 'vocab.txt')
            vp = alt if os.path.isfile(alt) else vp
        if not os.path.isfile(vp):
            print('[Warn] local tokenizer present but vocab.txt not found')
            return None
        tok = cls(vocab_file=vp)
        print(f"[Info] local custom tokenizer loaded from: {mod_path}")
        return tok
    except Exception as e:
        print(f"[Warn] failed to load local custom tokenizer: {e}")
        return None


def probe_token(tok, text: str) -> Tuple[int, list]:
    try:
        enc = tok(text, add_special_tokens=False)['input_ids']
        return len(enc), enc[:8]
    except Exception:
        try:
            # fallback: raw encode
            enc = tok.encode(text)
            return len(enc), enc[:8]
        except Exception:
            return -1, []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Model directory (contains config.json)')
    p.add_argument('--tokenizer', required=True, help='Tokenizer directory')
    p.add_argument('--vocab', default=None, help='Optional explicit vocab.txt for local custom tokenizer')
    p.add_argument('--prompt', default='<用户> 山东最高的山是什么山？<AI>')
    args = p.parse_args()

    # Load model config
    cfg_path = os.path.join(args.model, 'config.json')
    if not os.path.isfile(cfg_path):
        print(f'ERROR: config.json not found in {args.model}')
        return 1
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    model_vocab = int(cfg.get('vocab_size', -1))
    print('=== Model config ===')
    print('model_type         :', cfg.get('model_type'))
    print('vocab_size         :', model_vocab)
    print('bos/eos/pad in cfg :', cfg.get('bos_token_id'), cfg.get('eos_token_id'), cfg.get('pad_token_id'))

    # Tokenizer dir content
    list_tokenizer_files(args.tokenizer)
    vt_count = read_vocab_txt_count(args.tokenizer)
    if vt_count is not None:
        print(f'vocab.txt linecount : {vt_count}')

    # Load transformers and report versions
    transformers = load_transformers()
    print('transformers       :', getattr(transformers, '__version__', 'unknown'))
    try:
        import tokenizers  # type: ignore
        print('tokenizers         :', getattr(tokenizers, '__version__', 'unknown'))
    except Exception:
        print('tokenizers         : not found')

    # Load HF AutoTokenizer
    print('\n=== HF AutoTokenizer ===')
    try:
        hf_tok = transformers.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=True)
        hf_len = get_tokenizer_len(hf_tok)
        print('class/module       :', hf_tok.__class__.__name__, hf_tok.__class__.__module__)
        print('size               :', hf_len)
        print('bos/eos/pad ids    :', getattr(hf_tok, 'bos_token_id', None), getattr(hf_tok, 'eos_token_id', None), getattr(hf_tok, 'pad_token_id', None))
        l1, e1 = probe_token(hf_tok, '<用户>')
        l2, e2 = probe_token(hf_tok, '<AI>')
        print('probe <用户>        :', l1, e1)
        print('probe <AI>         :', l2, e2)
        if hf_len is not None and model_vocab != hf_len:
            print(f'MISMATCH with HF   : model={model_vocab}, hf_tokenizer={hf_len}, delta={hf_len - model_vocab}')
    except Exception as e:
        print(f'ERROR loading HF AutoTokenizer: {e}')
        hf_tok = None

    # Load local custom tokenizer
    print('\n=== Local custom tokenizer (llama8b/tokenizer.py) ===')
    local_tok = load_local_custom_tokenizer(args.tokenizer, args.vocab)
    if local_tok is not None:
        l_len = get_tokenizer_len(local_tok)
        # try to infer size from vocab map if not present
        if l_len is None and hasattr(local_tok, 'vocab') and isinstance(getattr(local_tok, 'vocab'), dict):
            l_len = len(getattr(local_tok, 'vocab'))
        print('class/module       :', local_tok.__class__.__name__, local_tok.__class__.__module__)
        print('size               :', l_len)
        print('bos/eos/pad ids    :', getattr(local_tok, 'bos_id', None), getattr(local_tok, 'eos_id', None), getattr(local_tok, 'pad_id', None))
        # probe
        try:
            # If it's HF-like call
            l1, e1 = probe_token(local_tok, '<用户>')
            l2, e2 = probe_token(local_tok, '<AI>')
        except Exception:
            # Direct methods may differ; skip
            l1, e1, l2, e2 = -1, [], -1, []
        print('probe <用户>        :', l1, e1)
        print('probe <AI>         :', l2, e2)
        if l_len is not None and model_vocab != l_len:
            print(f'MISMATCH with local: model={model_vocab}, local_tokenizer={l_len}, delta={l_len - model_vocab}')
    else:
        print('local custom tokenizer: not available')

    # Prompt tokenization length sanity
    print('\n=== Prompt tokenization length (HF if available) ===')
    if hf_tok is not None:
        ids = hf_tok(args.prompt, add_special_tokens=False)['input_ids']
        print(f'prompt length (HF) : {len(ids)}')

    # Guidance
    print('\n=== Guidance ===')
    print('- Ensure you point --tokenizer to a directory that contains the exact training tokenizer assets (vocab size must equal model vocab_size).')
    print('- If training used a custom CPM-like tokenizer, ensure llama8b/tokenizer.py and matching vocab.txt exist and the size equals model vocab_size.')
    print('- If <用户>/<AI> are expected special tokens, verify they encode as a single token; otherwise adjust prompt template or tokenizer assets.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
