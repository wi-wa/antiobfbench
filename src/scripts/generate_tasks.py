#!/usr/bin/env python3
"""Write task samples to envs/<env>/tasks.jsonl: n tasks per environment, spread evenly over its levels.

Usage: python scripts/generate_tasks.py [env numbers...] [--n 512] [--seed 0] [--pool rl]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench import all_envs, get_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('envs', nargs='*')
    ap.add_argument('--n', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pool', default='rl')
    args = ap.parse_args()
    for env in [get_env(num) for num in args.envs] or all_envs().values():
        rows = env.sample(args.n, pool=args.pool, seed=args.seed)
        folder = ROOT.parent / 'envs' / env.name
        folder.mkdir(exist_ok=True)
        with open(folder / 'tasks.jsonl', 'w', encoding='utf8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        levels = Counter(row['level'] for row in rows)
        print(env.name, len(rows), ' '.join(f'L{k}:{levels[k]}' for k in sorted(levels)), flush=True)


if __name__ == '__main__':
    main()
