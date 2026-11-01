#!/usr/bin/env python3
"""End-to-end smoke test on one GPU: a few rollouts per level and a few probe requests per split, for each environment.

This checks that prompts, parsing, scoring and probe filters work on real model text. The samples are far too small to
measure anything. Prints one line per environment and writes every response to --out.
  CUDA_VISIBLE_DEVICES=0 python scripts/smoke_envs.py [env numbers...] --out /data/.../smoke.jsonl
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
sys.path.insert(0, str(ROOT / 'scripts'))
from antiobfbench import all_envs, get_env  # noqa: E402
from generate_probe import MODEL, generate  # noqa: E402


def share(values):
    values = list(values)
    return sum(values) / len(values) if values else float('nan')


def natural(env, tok, model, args, out):
    tasks = env.sample(args.per_level * len(env.levels))
    texts = generate(tok, model, [t['prompt'] for t in tasks], args.cap, args.batch_size)
    rows = env.score(tasks, texts)
    for task, text, row in zip(tasks, texts, rows):
        out.write(json.dumps(dict(env=env.name, kind='natural', id=task['id'], level=task['level'], text=text, reward=row['reward'], value=row['value']), ensure_ascii=False) + '\n')
    known = [env.binary(r['value']) for r in rows if r['value'] is not None]
    return f"reward {share(r['reward'] for r in rows):.2f}  UB {100 * share(map(bool, known)):3.0f}%  null {100 * share(r['value'] is None for r in rows):3.0f}%"


def probes(env, split, tok, model, args, out):
    requests = list(islice(env.probe_requests(split), args.requests))
    if not requests:
        return f'{split} none'
    texts = iter(generate(tok, model, [env.request_prompt(r, v) for r in requests for v in r['variants']], args.cap, args.batch_size))
    asked = did = kept = 0
    for request in requests:
        responses = list(islice(texts, len(request['variants'])))
        flags = [bool(env.probe_keep(request, v, t)) for v, t in zip(request['variants'], responses)]
        samples = env.probe_samples(request, responses)
        asked += len(flags)
        did += sum(flags)
        kept += bool(samples)
        out.write(json.dumps(dict(env=env.name, kind=split, pair_id=request['pair_id'], prompt=request['prompt'], instructions=[v.get('instruction') for v in request['variants']],
                                  texts=responses, flags=flags, kept=bool(samples)), ensure_ascii=False) + '\n')
    return f'{split} did-as-asked {did}/{asked}, kept {kept}/{len(requests)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('envs', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--per-level', type=int, default=8)
    ap.add_argument('--requests', type=int, default=16)
    ap.add_argument('--cap', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=48)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to('cuda').eval()
    summary = []
    with open(args.out, 'a', encoding='utf8') as out:
        for env in [get_env(num) for num in args.envs] or all_envs().values():
            parts = [natural(env, tok, model, args, out), probes(env, 'near_train', tok, model, args, out), probes(env, 'far_train', tok, model, args, out)]
            summary.append(f"{env.name:30s} {' | '.join(parts)}")
            print('SMOKE', summary[-1], flush=True)
            out.flush()
    print('\n'.join(['== summary'] + summary))


if __name__ == '__main__':
    main()
