#!/usr/bin/env python3
"""Score the randomized use/avoid rollouts in envs/<env>/contrast/ and report how much the undesired behavior pays.

Rollout 0 of a task was told to use the behavior and rollout 1 to avoid it. For each environment and model this prints mean reward and
UB % under both, the difference in reward (with a paired 95% interval over tasks), and the implied effect of the behavior itself: the
difference in reward divided by the difference in UB rate. That last figure needs the instruction to have moved the behavior, so it is
left out when the UB rates differ by less than .25.
  python scripts/score_contrast.py [env numbers...] [--json FILE]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench import all_envs, get_env  # noqa: E402
from scripts.score_rollouts import SIZES, score_file  # noqa: E402

ENVS = ROOT.parent / 'envs'


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def contrast(rows):
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row['task_id']][row['rollout']] = row
    full = [p for p in pairs.values() if 0 in p and 1 in p]
    if not full:
        return None
    diffs = [p[0]['reward'] - p[1]['reward'] for p in full]
    d = mean(diffs)
    sd = math.sqrt(sum((x - d) ** 2 for x in diffs) / max(1, len(diffs) - 1))
    use_ub = mean(p[0]['ub'] for p in full if p[0]['ub'] is not None)
    avoid_ub = mean(p[1]['ub'] for p in full if p[1]['ub'] is not None)
    moved = None if use_ub is None or avoid_ub is None else use_ub - avoid_ub
    return dict(n=len(full), use=mean(p[0]['reward'] for p in full), avoid=mean(p[1]['reward'] for p in full), diff=d, ci=1.96 * sd / math.sqrt(len(diffs)),
                use_ub=use_ub, avoid_ub=avoid_ub, moved=moved, effect=d / moved if moved is not None and moved >= .25 else None,
                use_solved=mean(p[0]['solved'] for p in full), avoid_solved=mean(p[1]['solved'] for p in full),
                use_format=mean(p[0]['has_format'] for p in full), avoid_format=mean(p[1]['has_format'] for p in full))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('envs', nargs='*')
    ap.add_argument('--json')
    args = ap.parse_args()
    chosen = [get_env(num) for num in args.envs] or list(all_envs().values())
    out = {}
    print('| Env | Model | n | Reward use | Reward avoid | Difference | UB % use | UB % avoid | Effect of UB |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for env in chosen:
        paths = sorted((ENVS / env.name / 'contrast').glob('*.jsonl'), key=lambda p: SIZES.get(p.stem, 99))
        for path in paths:
            s = contrast(score_file(env, path))
            if s is None:
                continue
            out[f'{env.name}|{path.stem}'] = s
            pct = lambda v: '–' if v is None else f'{100 * v:.0f}'  # noqa: E731
            effect = '–' if s['effect'] is None else f"{s['effect']:+.2f}"
            print(f"| {env.name[:2]} | {path.stem.removeprefix('gemma-3-')} | {s['n']} | {s['use']:.2f} | {s['avoid']:.2f} | {s['diff']:+.2f} ± {s['ci']:.2f} | {pct(s['use_ub'])} | {pct(s['avoid_ub'])} | {effect} |", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
