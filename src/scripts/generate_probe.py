#!/usr/bin/env python3
"""Build checked near/far probe pairs in a fresh output directory per environment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'rl_trainer/src')]
from antiobfbench import get_env

SPLITS = ('near_train', 'near_test', 'far_train', 'far_test')


def direct(variant, number):
    """Use exact constructed contrasts where the environment already supplies them."""
    instruction = variant['instruction']
    if number in ('03', '06'):
        return instruction.split('\n\n', 1)[1]
    if number == '04' and variant.get('copy'):
        return instruction[instruction.index('```'):]
    if number == '08':
        return '\\boxed{' + variant['final'] + '}'
    return None


def prepare(env, requests):
    paired = hasattr(env, 'pair_from_base')
    texts = [[None] if paired else [direct(v, env.name[:2]) for v in r['variants']] for r in requests]
    prompts = [(i, j, env.probe_base_prompt(r) if paired else env.request_prompt(r, r['variants'][j]))
               for i, r in enumerate(requests) for j, text in enumerate(texts[i]) if text is None]
    return texts, prompts


def checked_samples(env, request, texts):
    return (env.pair_from_base(request, texts[0]) if hasattr(env, 'pair_from_base')
            else env.probe_samples(request, texts))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('envs', nargs='+')
    ap.add_argument('--output', type=Path, required=True, help='creates a fresh subdirectory per environment')
    ap.add_argument('--split', nargs='+', choices=SPLITS, default=SPLITS)
    ap.add_argument('--model', help='local writer model; unnecessary for fully constructed contrasts')
    ap.add_argument('--port', type=int, default=30455)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-pairs', type=int, default=128)
    ap.add_argument('--test-pairs', type=int, default=64)
    ap.add_argument('--max-attempts', type=int, default=4096)
    args = ap.parse_args()
    if min(args.train_pairs, args.test_pairs, args.max_attempts) < 1:
        ap.error('Pair counts and max-attempts must be positive')
    engine = tokenizer = None
    try:
        for env_name in args.envs:
            env = get_env(env_name)
            directory = args.output / env.name
            directory.mkdir(parents=True, exist_ok=False)
            summary, groups = {}, {}
            for split in dict.fromkeys(args.split):
                target = args.train_pairs if split.endswith('train') else args.test_pairs
                requests = env.probe_requests(split, seed=args.seed)
                kept = attempted = 0
                groups[split] = set()
                path = directory / f'probe_{split}.jsonl'
                with path.open('w', encoding='utf8') as accepted:
                    while kept < target and attempted < args.max_attempts:
                        batch = list(islice(requests, min(32, target-kept, args.max_attempts-attempted)))
                        if not batch:
                            break
                        attempted += len(batch)
                        responses, needed = prepare(env, batch)
                        if needed:
                            if not args.model:
                                raise ValueError(f'{env.name} needs --model for these contrasts')
                            if engine is None:
                                from transformers import AutoTokenizer
                                from reinforce_worker import ColocatedEngine
                                tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
                                engine = ColocatedEngine(model_path=args.model, host='127.0.0.1', port=args.port,
                                    dtype='bfloat16', tp_size=1, base_gpu_id=0, attention_backend='triton',
                                    triton_attention_reduce_in_fp32=True, chunked_prefill_size=2048,
                                    context_length=8192, mem_fraction_static=.80, max_running_requests=64,
                                    cuda_graph_max_bs=64, random_seed=args.seed, decode_log_interval=1000,
                                    log_level='warning')
                            inputs = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}],
                                      tokenize=True, add_generation_prompt=True, return_dict=False) for _, _, p in needed]
                            if any(len(ids) > 6500 for ids in inputs):
                                raise ValueError('Probe writer prompt exceeds context budget')
                            outputs = engine.generate(input_ids=inputs, sampling_params={
                                'temperature': 1.0, 'top_p': 1.0, 'top_k': -1, 'max_new_tokens': 1536})
                            if len(outputs) != len(needed):
                                raise RuntimeError('Incomplete writer batch')
                            for (i, j, _), out in zip(needed, outputs):
                                responses[i][j] = ('' if out['meta_info']['finish_reason']['type'] == 'length'
                                                   else tokenizer.decode(out['output_ids'], skip_special_tokens=True))
                        with ThreadPoolExecutor(max_workers=min(4, env.workers)) as pool:
                            pairs = pool.map(lambda item: checked_samples(env, *item), zip(batch, responses))
                            for request, pair in zip(batch, pairs):
                                if not pair:
                                    continue
                                if len(pair) != 2 or {r['label'] for r in pair} != {True, False}:
                                    raise ValueError('Expected one positive and one negative per pair')
                                for row in pair:
                                    row.update(label_target=getattr(env, 'probe_target', 'ub'),
                                               source_group=request['meta']['group'], source_key=request['meta']['key'])
                                    accepted.write(json.dumps(row, ensure_ascii=False) + '\n')
                                kept += 1
                                groups[split].add(request['meta']['group'])
                        accepted.flush()
                        print(f'{env.name}/{split}: {kept}/{target} pairs, {attempted} attempts', flush=True)
                if kept < target:
                    raise RuntimeError(f'{env.name}/{split}: only {kept}/{target} checked pairs')
                summary[split] = dict(pairs=kept, attempts=attempted, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for domain in ('near', 'far'):
                if groups.get(domain+'_train', set()) & groups.get(domain+'_test', set()):
                    raise ValueError(f'{domain} train/test groups overlap')
            (directory / 'dataset_complete.json').write_text(json.dumps(dict(
                env=env.name, seed=args.seed, writer=args.model, splits=summary), indent=2) + '\n')
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == '__main__':
    main()
