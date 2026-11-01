#!/usr/bin/env python3
"""Sample natural rollouts of each environment's tasks with one model, through OpenRouter or a local GPU.

Takes the first --per-level tasks of each level in envs/<env>/tasks.jsonl and samples --rollouts rollouts of each, at
temperature 1 with top-p and top-k off, as RL does. Appends to envs/<env>/trajectories/<tag>.jsonl. A rerun skips saved rollouts, and
replaces any whose task prompt has changed since.
  python scripts/sample_rollouts.py --tag gemma-3-4b --openrouter google/gemma-3-4b-it --providers DeepInfra,Parasail
  CUDA_VISIBLE_DEVICES=7 python scripts/sample_rollouts.py --tag gemma-3-1b --local google/gemma-3-1b-it
With --contrast it samples the randomized check instead: rollout 0 of each task is told to use the undesired behavior and rollout 1 to avoid it,
saved to envs/<env>/contrast/<tag>.jsonl (see antiobfbench/contrast.py).
"""
import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench.contrast import CONTRAST  # noqa: E402

ENVS = ROOT.parent / 'envs'
URL = 'https://openrouter.ai/api/v1/chat/completions'
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def read_key(name):
    for line in (ROOT.parent / '.env').read_text().splitlines():
        key, _, value = line.strip().removeprefix('export ').partition('=')
        if key.strip() == name:
            return value.strip().strip('\'"')
    raise SystemExit(f'{name} not found in .env')


def seed_of(tag, task_id, rollout):
    return int(hashlib.sha256(f'{tag}|{task_id}|{rollout}'.encode()).hexdigest()[:8], 16) % 2 ** 31


def prompt_of(task, rollout, args):
    return f"{task['prompt']}\n\n{CONTRAST[task['env'][:2]][rollout]}" if args.contrast else task['prompt']


def jobs_for(args):
    """The rollouts still to sample, and an open output file per environment."""
    jobs, files = [], {}
    folders = [ENVS / name for name in args.envs] if args.envs else sorted(p for p in ENVS.iterdir() if p.is_dir())
    for folder in folders:
        out = folder / ('contrast' if args.contrast else 'trajectories') / f'{args.tag}.jsonl'
        out.parent.mkdir(exist_ok=True)
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
    return dict(task_id=task['id'], env=task['env'], level=task['level'], rollout=job['rollout'], tag=args.tag, seed=seed_of(args.tag, task['id'], job['rollout']),
                prompt=task['prompt'], answer=task['answer'], **fields)


async def call(session, job, args, key):
    """One completion as a row. Retries what can succeed later and raises on what can't."""
    payload = dict(model=args.openrouter, messages=[dict(role='user', content=job['task']['prompt'])], max_tokens=args.max_tokens,
                   temperature=1.0, top_p=1.0, top_k=0, seed=seed_of(args.tag, job['task']['id'], job['rollout']), usage=dict(include=True))
    if args.providers:
        payload['provider'] = dict(order=args.providers.split(','), allow_fallbacks=False, require_parameters=True)
    problem = None
    for attempt in range(args.retries):
        if attempt:
            await asyncio.sleep(min(60, 2 ** attempt) * random.uniform(.5, 1.5))
        try:
            async with session.post(URL, json=payload, headers=dict(Authorization=f'Bearer {key}')) as response:
                status, body = response.status, await response.text()
        except (asyncio.TimeoutError, OSError, ConnectionError, RuntimeError) as err:
            problem = type(err).__name__
            continue
        if status != 200:
            problem = f'HTTP {status}: {body[:200]}'
            if status in RETRY_STATUS:
                continue
            raise RuntimeError(problem)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            problem = 'unparsable body'
            continue
        if 'error' in data or not data.get('choices'):
            problem = f"provider error: {str(data.get('error'))[:200]}"
            continue
        choice, usage = data['choices'][0], data.get('usage') or {}
        text = choice['message'].get('content')
        if text is None:
            problem = 'no content'
            continue
        return row_for(job, args, model=args.openrouter, provider=data.get('provider'), text=text, finish_reason=choice.get('finish_reason'), truncated=choice.get('finish_reason') == 'length',
                       prompt_tokens=usage.get('prompt_tokens'), completion_tokens=usage.get('completion_tokens'), cost=usage.get('cost'))
    return dict(failed=problem, task_id=job['task']['id'], rollout=job['rollout'])


async def run_openrouter(jobs, files, args):
    import aiohttp
    key = read_key(args.key_name)
    gate = asyncio.Semaphore(args.concurrency)
    stats = dict(done=0, failed=0, cost=0.0)
    start = time.time()
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)

    async def one(session, job):
        async with gate:
            row = await call(session, job, args, key)
        if 'failed' in row:
            stats['failed'] += 1
            print('FAILED', row, flush=True)
            return
        files[row['env']].write(json.dumps(row, ensure_ascii=False) + '\n')
        files[row['env']].flush()
        stats['done'] += 1
        stats['cost'] += row['cost'] or 0
        if stats['done'] % 250 == 0:
            print(f"{stats['done']}/{len(jobs)} failed {stats['failed']} cost ${stats['cost']:.3f} {time.time() - start:.0f}s", flush=True)

    random.shuffle(jobs)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await asyncio.gather(*(one(session, job) for job in jobs))
    print(f"{args.tag}: saved {stats['done']} of {len(jobs)}, failed {stats['failed']}, cost ${stats['cost']:.3f}, {time.time() - start:.0f}s", flush=True)


def run_local(jobs, files, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.local, padding_side='left')
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.local, dtype=torch.bfloat16).to('cuda').eval()
    torch.manual_seed(0)
    chats = [tok.apply_chat_template([dict(role='user', content=j['task']['prompt'])], tokenize=False, add_generation_prompt=True) for j in jobs]
    order = sorted(range(len(jobs)), key=lambda i: len(chats[i]))
    ends = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id]) | {tok.eos_token_id}
    start, done = time.time(), 0
    for first in range(0, len(order), args.batch_size):
        ids = order[first:first + args.batch_size]
        enc = tok([chats[i] for i in ids], return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_tokens, pad_token_id=tok.pad_token_id)
        for i, tokens in zip(ids, out[:, enc.input_ids.shape[1]:].tolist()):
            prompt_tokens = len(tok(chats[i], add_special_tokens=False).input_ids)
            stop = next((k for k, t in enumerate(tokens) if t in ends), None)
            used = tokens[:stop] if stop is not None else tokens
            row = row_for(jobs[i], args, model=args.local, provider='local', text=tok.decode(used, skip_special_tokens=True), finish_reason='stop' if stop is not None else 'length',
                          truncated=stop is None, prompt_tokens=prompt_tokens, completion_tokens=len(used) + (stop is not None), cost=0)
            files[row['env']].write(json.dumps(row, ensure_ascii=False) + '\n')
            files[row['env']].flush()
        done += len(ids)
        print(f'{done}/{len(jobs)} {time.time() - start:.0f}s', flush=True)
    print(f'{args.tag}: saved {done} of {len(jobs)}, {time.time() - start:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='name of the output files, e.g. gemma-3-4b')
    ap.add_argument('--openrouter', help='OpenRouter model id')
    ap.add_argument('--local', help='local path or HF id, run on the visible GPU')
    ap.add_argument('--providers', default='', help='OpenRouter providers to allow, in order of preference')
    ap.add_argument('--key-name', default='OPENROUTER_API_KEY', help='which key in .env to use')
    ap.add_argument('--envs', nargs='*', help='env folder names; default all')
    ap.add_argument('--per-level', type=int, default=32)
    ap.add_argument('--rollouts', type=int, default=2)
    ap.add_argument('--contrast', action='store_true', help='sample the use/avoid check instead of natural rollouts')
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--concurrency', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--retries', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=900)
    args = ap.parse_args()
    if bool(args.openrouter) == bool(args.local):
        raise SystemExit('give exactly one of --openrouter and --local')
    if args.contrast and args.rollouts != 2:
        raise SystemExit('--contrast samples exactly two rollouts per task')
    jobs, files = jobs_for(args)
    print(f'{args.tag}: {len(jobs)} rollouts to sample', flush=True)
    if args.openrouter:
        asyncio.run(run_openrouter(jobs, files, args))
    else:
        run_local(jobs, files, args)


if __name__ == '__main__':
    main()
