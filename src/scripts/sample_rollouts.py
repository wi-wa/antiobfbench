#!/usr/bin/env python3
"""Sample natural rollouts of each environment's tasks with one model, on a local GPU.

Takes the first --per-level tasks of each level in envs/<env>/tasks.jsonl and samples --rollouts rollouts of each, at
temperature 1 with top-p and top-k off, as RL does. Appends to envs/<env>/trajectories/<tag>.jsonl. A rerun skips saved rollouts, and
replaces any whose task prompt has changed since.
  CUDA_VISIBLE_DEVICES=7 python scripts/sample_rollouts.py --tag gemma-3-1b --local google/gemma-3-1b-it
With --contrast it samples the randomized check instead: rollout 0 of each task is told to use the undesired behavior and rollout 1 to avoid it,
saved to envs/<env>/contrast/<tag>.jsonl (each environment's `contrast` attribute holds the two instructions).
"""
import argparse
import fcntl
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench import get_env  # noqa: E402

ENVS = ROOT.parent / 'envs'
LOCKS = []


def prompt_of(task, rollout, args):
    return f"{task['prompt']}\n\n{get_env(task['env']).contrast[rollout]}" if args.contrast else task['prompt']


def jobs_for(args):
    """The rollouts still to sample, and an open output file per environment."""
    jobs, files = [], {}
    folders = sorted([ENVS / get_env(name).name for name in args.envs] if args.envs else [p for p in ENVS.iterdir() if p.is_dir()], key=lambda p: p.name)
    for folder in folders:
        out = folder / ('contrast' if args.contrast else 'trajectories') / f'{args.tag}.jsonl'
        out.parent.mkdir(exist_ok=True)
        lock = out.with_suffix(out.suffix + '.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX)
        LOCKS.append(lock)
        tasks = [json.loads(line) for line in (folder / 'tasks.jsonl').open(encoding='utf8')]
        by_id = {task['id']: task for task in tasks}
        done = set()
        if out.exists():
            saved = [json.loads(line) for line in out.open(encoding='utf8')]
            current = [r for r in saved if r['task_id'] in by_id and prompt_of(by_id[r['task_id']], r['rollout'], args) == r['prompt']]
            if len(current) < len(saved):
                print(f"{folder.name}: dropping {len(saved) - len(current)} rollouts of changed prompts", flush=True)
                out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in current), encoding='utf8')
            done = {(r['task_id'], r['rollout']) for r in current}
        files[folder.name] = out.open('a', encoding='utf8')
        counts = defaultdict(int)
        for task in tasks:
            counts[task['level']] += 1
            if counts[task['level']] <= args.per_level:
                jobs += [dict(task=dict(task, prompt=prompt_of(task, r, args)), rollout=r) for r in range(args.rollouts) if (task['id'], r) not in done]
    return jobs, files


def row_for(job, args, **fields):
    task = job['task']
    return dict(task_id=task['id'], env=task['env'], level=task['level'], rollout=job['rollout'], tag=args.tag, seed=fields.pop('seed', 0),
                prompt=task['prompt'], answer=task['answer'], **fields)


def run_local(jobs, files, args):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.local, padding_side='left')
    tok.pad_token = tok.pad_token or tok.eos_token
    config = AutoConfig.from_pretrained(args.local)
    model_class = getattr(transformers, (config.architectures or ['AutoModelForCausalLM'])[0], AutoModelForCausalLM)
    model = model_class.from_pretrained(args.local, dtype=torch.bfloat16).to('cuda').eval()
    torch.manual_seed(0)
    template_kwargs = {'enable_thinking': False} if args.no_native_thinking else {}
    chats = [tok.apply_chat_template([dict(role='user', content=j['task']['prompt'])], tokenize=False, add_generation_prompt=True, **template_kwargs) for j in jobs]
    order = sorted(range(len(jobs)), key=lambda i: len(chats[i]))
    ends = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id]) | {tok.eos_token_id}
    start, done = time.time(), 0
    for first in range(0, len(order), args.batch_size):
        ids = order[first:first + args.batch_size]
        enc = tok([chats[i] for i in ids], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_tokens, pad_token_id=tok.pad_token_id)
        for slot, (i, tokens) in enumerate(zip(ids, out[:, enc.input_ids.shape[1]:].tolist())):
            prompt_tokens = len(tok(chats[i], add_special_tokens=False).input_ids)
            stop = next((k for k, t in enumerate(tokens) if t in ends), None)
            used = tokens[:stop] if stop is not None else tokens
            row = row_for(jobs[i], args, seed=0, seed_scope='run', sample_order=first + slot, batch_size=len(ids), model=args.local, provider='local', text=tok.decode(used, skip_special_tokens=True), finish_reason='stop' if stop is not None else 'length',
                          truncated=stop is None, prompt_tokens=prompt_tokens, completion_tokens=len(used) + (stop is not None), cost=0, raw_text=tok.decode(used, skip_special_tokens=False),
                          rendered_prompt=chats[i], chat_template_kwargs=template_kwargs,
                          temperature=1.0, top_p=1.0, top_k=0, max_tokens=args.max_tokens,
                          model_revision=getattr(config, '_commit_hash', None), transformers_version=transformers.__version__)
            files[row['env']].write(json.dumps(row, ensure_ascii=False) + '\n')
            files[row['env']].flush()
        done += len(ids)
        print(f'{done}/{len(jobs)} {time.time() - start:.0f}s', flush=True)
    print(f'{args.tag}: saved {done} of {len(jobs)}, {time.time() - start:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='name of the output files, e.g. gemma-3-4b')
    ap.add_argument('--local', required=True, help='local path or HF id, run on the visible GPU')
    ap.add_argument('--envs', nargs='*', help='env numbers or folder names; default all')
    ap.add_argument('--per-level', type=int, default=32)
    ap.add_argument('--rollouts', type=int, default=2)
    ap.add_argument('--contrast', action='store_true', help='sample the use/avoid check instead of natural rollouts')
    ap.add_argument('--no-native-thinking', action='store_true', help='disable model-native thinking; leave any thinking instruction in the user prompt unchanged')
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()
    if args.contrast and args.rollouts != 2:
        raise SystemExit('--contrast samples exactly two rollouts per task')
    jobs, files = jobs_for(args)
    print(f'{args.tag}: {len(jobs)} rollouts to sample', flush=True)
    if not jobs:
        return
    run_local(jobs, files, args)


if __name__ == '__main__':
    main()
