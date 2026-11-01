#!/usr/bin/env python3
"""Generic checks for environments: format, determinism, disjoint pools, capacity, and that scoring survives junk.

Usage: python scripts/check_envs.py [env numbers...] [--cap N]   (a zero exit means all passed)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench import all_envs, get_env  # noqa: E402
from antiobfbench.base import POOLS  # noqa: E402

TASK_KEYS = {'id', 'env', 'level', 'prompt', 'answer', 'meta'}
SAMPLE_KEYS = {'id', 'env', 'split', 'text', 'label', 'pair_id', 'prompt'}
JUNK = ['', 'no idea', '\\boxed{}', '\\boxed{', '<answer></answer>', '<answer>', '```python\n\n```', '```python\ndef (\n```',
        '\\boxed{\\boxed{1}}', 'x' * 20000, '<answer>(a+)+$</answer>', '```python\nwhile True: pass\n```', '٣ ๑ ౧ ১', '\\boxed{-1}', '\\boxed{0}',
        '9' * 5000, '\\boxed{' + '9' * 5000 + '}', '<answer>' + '9' * 5000 + '</answer>', '\\boxed{10^{400}}', '\\boxed{1 \\times 10^{99999999}}',
        '\\boxed{A, ' + '9' * 5000 + '}', '\\boxed{A, 10^{400}}']


def digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fingerprint(env, seed):
    rows = env.sample(64, seed=seed)
    for split in ('near_train', 'far_train'):
        rows += list(islice(env.probe_requests(split, seed=seed), 8))
    return digest(rows)


def check_format(env):
    tasks = env.sample(32)
    assert tasks, 'no tasks'
    assert len(tasks) == 32, f'sample gave {len(tasks)} of 32'
    for task in tasks:
        assert set(task) == TASK_KEYS, f'task keys {sorted(task)}'
        assert task['env'] == env.name and task['level'] in env.levels and isinstance(task['prompt'], str) and task['prompt']
        assert json.loads(json.dumps(task)) == task, 'task is not plain JSON'
    assert len({t['id'] for t in tasks}) == len(tasks), 'duplicate ids'


def check_determinism(env, num):
    here = fingerprint(env, 0)
    assert here == fingerprint(env, 0), 'differs within a process'
    code = ('import sys; sys.path.insert(0, "scripts"); sys.path.insert(0, "."); import check_envs as c; '
            f'print(c.fingerprint(c.get_env("{num}"), 0))')
    for hash_seed in ('1', '2'):
        out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, env={**os.environ, 'PYTHONHASHSEED': hash_seed})
        assert out.returncode == 0, out.stderr[-1000:]
        assert out.stdout.strip().splitlines()[-1] == here, 'differs across processes'
    assert env.sample(64, seed=1) != env.sample(64, seed=0), 'seed has no effect'


def check_pools(env, cap):
    """Count tasks up to the cap per level and pool, and require that no key or group crosses a pool."""
    keys, groups, table = {}, {}, {}
    for level in sorted({*env.levels, *(env.near_levels or ())}):
        for pool in POOLS:
            rows = list(islice(env.generate(level, pool), cap))
            table[level, pool] = len(rows)
            for task in rows:
                key, group = task['meta']['key'], task['meta']['group']
                assert keys.setdefault(key, (level, pool)) == (level, pool), f'key {key!r} in two places'
                assert groups.setdefault(group, pool) == pool, f'group {group!r} in two pools'
    return table


def check_far(env, cap):
    seen, counts = {}, {}
    for split in ('far_train', 'far_test'):
        rows = list(islice(env.probe_requests(split), cap))
        counts[split] = len(rows)
        for request in rows:
            group = request['meta']['group']
            assert seen.setdefault(group, split) == split, f'far group {group!r} in two splits'
    return counts


def check_scoring(env):
    tasks = env.sample(4)
    for task in tasks:
        rows = env.score([task] * len(JUNK), JUNK)
        for row, text in zip(rows, JUNK):
            assert isinstance(row['reward'], (int, float)) and 0 <= row['reward'] <= 1, f'reward {row["reward"]!r} on {text[:30]!r}'
            value = row['value']
            assert value is None or isinstance(value, (bool, int, float)), f'value {value!r}'
            assert isinstance(row['spans'], list), f'spans {row["spans"]!r}'
            for span in row['spans']:
                assert len(span) == 2 and 0 <= span[0] <= span[1] <= len(text), f'span {span} on {text[:30]!r}'
        assert env.reward(task, '') == 0, 'an empty response earned reward'


def check_probe(env):
    for split in ('near_train', 'near_test', 'far_train', 'far_test'):
        rows = list(islice(env.probe_requests(split), 6))
        assert rows, f'no requests for {split}'
        for request in rows:
            assert len(request['variants']) >= 2
            for variant in request['variants']:
                assert isinstance(env.request_prompt(request, variant), str)
            assert env.probe_samples(request, ['no idea'] * len(request['variants'])) == [] or env.continuous
            for sample in env.probe_samples(request, ['no idea'] * len(request['variants'])):
                assert set(sample) == SAMPLE_KEYS


def run_one(env, num, cap, far_cap):
    fails, t0 = [], time.time()
    for name, fn in (('format', lambda: check_format(env)), ('determinism', lambda: check_determinism(env, num)),
                     ('scoring', lambda: check_scoring(env)), ('probe', lambda: check_probe(env))):
        try:
            fn()
        except Exception as err:
            fails.append(f'{name}: {type(err).__name__}: {err}')
    table, far = {}, {}
    try:
        table = check_pools(env, cap)
    except Exception as err:
        fails.append(f'pools: {type(err).__name__}: {err}')
    try:
        far = check_far(env, far_cap)
    except Exception as err:
        fails.append(f'far: {type(err).__name__}: {err}')
    print(f'== {env.name}  {"FAIL" if fails else "ok"}  {time.time() - t0:.0f}s')
    for level in sorted({key[0] for key in table}):
        print(f'   L{level}  rl {table[level, "rl"]:>6}  near_train {table[level, "near_train"]:>6}  near_test {table[level, "near_test"]:>6}')
    if far:
        print(f'   far_train {far["far_train"]}  far_test {far["far_test"]}   (all counts stop at the cap)')
    for fail in fails:
        print('   FAIL', fail)
    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('envs', nargs='*')
    ap.add_argument('--cap', type=int, default=4000, help='stop counting a level and pool at this many tasks')
    ap.add_argument('--far-cap', type=int, default=2000)
    args = ap.parse_args()
    names = args.envs or [name[:2] for name in all_envs()]
    results = [run_one(get_env(num), num, args.cap, args.far_cap) for num in names]
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
