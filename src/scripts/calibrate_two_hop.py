#!/usr/bin/env python3
"""Find out which two-hop chains a model knows, for the difficulty levels of environment 16: ask each hop and the two-hop question on its own, greedily.

Writes hops_<tag>.jsonl with hop1 and hop2 (does it know each fact) and direct (does it answer the two-hop question). Run on one GPU:
  CUDA_VISIBLE_DEVICES=0 python scripts/calibrate_two_hop.py RAW_DIR --model unsloth/gemma-3-12b-it --tag all12b
"""
import argparse
import ast
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

csv.field_size_limit(10 ** 8)


def norm(text):
    return ' '.join(re.sub(r"[^\w\s']", ' ', text.lower().replace('’', "'")).split())


def has(text, names):
    text = f' {norm(text)} '
    return any(f' {norm(n)} ' in text for n in names if len(norm(n)) >= 2)


def names(raw):
    return sorted({a for group in ast.literal_eval(raw) for a in group})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', type=Path)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--batch-size', type=int, default=384)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.raw / 'TwoHopFact.csv', encoding='utf8')))
    tok = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to('cuda').eval()

    def ask(sentences):
        chats = [tok.apply_chat_template([dict(role='user', content=f'Complete the sentence with only the missing words: "{s} ___"')], tokenize=False, add_generation_prompt=True) for s in sentences]
        order = sorted(range(len(chats)), key=lambda i: len(chats[i]))
        out = [None] * len(chats)
        for first in range(0, len(order), args.batch_size):
            ids = order[first:first + args.batch_size]
            enc = tok([chats[i] for i in ids], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=16, pad_token_id=tok.pad_token_id)
            for i, text in zip(ids, tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
                out[i] = text
        return out

    hop1 = ask([r['r1(e1).prompt'] for r in rows])
    hop2 = ask([r['r2(e2).prompt'] for r in rows])
    both = ask([r['r2(r1(e1)).prompt'] for r in rows])
    with open(args.raw / f'hops_{args.tag}.jsonl', 'w', encoding='utf8') as f:
        for r, a, b, c in zip(rows, hop1, hop2, both):
            f.write(json.dumps(dict(uid=r['uid'], hop1=has(a, names(r['e2.aliases'])), hop2=has(b, names(r['e3.aliases'])), direct=has(c, names(r['e3.aliases']))), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
