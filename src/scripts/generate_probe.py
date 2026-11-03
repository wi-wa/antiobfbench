#!/usr/bin/env python3
"""Write probe data: ask a generator model for every variant of each request and keep what the environment accepts.

Environments with `pair_from_base` generate one neutral baseline per pair, then insert matched contrasts.
Writes envs/<env>/probe_<split>.jsonl (format from META_SPEC.md, plus `prompt`) and prints how often the model produced
what was asked for. Run on one GPU, under arun:
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_probe.py 05 --split near_train --n 200
"""
import argparse
import json
import sys
from itertools import islice
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench import get_env  # noqa: E402

MODEL = '/data/students/wiwal2741/experiments/antiobf/data/models/qwen2.5-7b-instruct'


def generate(tok, model, prompts, cap, batch_size):
    """Sampled responses at temperature 1, in prompt order. Shortest prompts first, so padding stays small."""
    chats = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    order = sorted(range(len(chats)), key=lambda i: len(chats[i]))
    out = [None] * len(chats)
    for start in range(0, len(order), batch_size):
        ids = order[start:start + batch_size]
        enc = tok([chats[i] for i in ids], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=cap, pad_token_id=tok.pad_token_id)
        for i, text in zip(ids, tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
            out[i] = text
        print(f'{min(start + batch_size, len(order))}/{len(order)}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('env')
    ap.add_argument('--split', default='near_train', choices=('near_train', 'near_test', 'far_train', 'far_test'))
    ap.add_argument('--n', type=int, default=200, help='requests (pairs or series) to try')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', help="the writer: by default the environment's own, or else the usual one")
    ap.add_argument('--cap', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()
    env = get_env(args.env)
    requests = list(islice(env.probe_requests(args.split, seed=args.seed), args.n))
    paired = hasattr(env, 'pair_from_base')
    prompts = ([env.probe_base_prompt(r) for r in requests] if paired else
               [env.request_prompt(r, v) for r in requests for v in r['variants']])
    args.model = args.model or getattr(env, 'writer', MODEL)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to('cuda').eval()
    torch.manual_seed(args.seed)
    texts = iter(generate(tok, model, prompts, args.cap, args.batch_size))
    kept = variants_ok = 0
    with open(ROOT.parent / 'envs' / env.name / f'probe_{args.split}.jsonl', 'w', encoding='utf8') as f:
        for request in requests:
            if paired:
                samples = env.pair_from_base(request, next(texts))
                variants_ok += len(samples)
            else:
                responses = list(islice(texts, len(request['variants'])))
                variants_ok += sum(env.probe_keep(request, v, t) for v, t in zip(request['variants'], responses))
                samples = env.probe_samples(request, responses)
            kept += bool(samples)
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f'{env.name} {args.split}: {kept}/{len(requests)} requests kept; {variants_ok}/{sum(len(r["variants"]) for r in requests)} responses did what was asked')


if __name__ == '__main__':
    main()
