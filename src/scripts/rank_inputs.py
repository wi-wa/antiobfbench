#!/usr/bin/env python3
"""The measured inputs for ranking the environments: saturation, variety of the stored tasks, base-rate fit and how far the behavior pays.

Reads envs/<env>/tasks.jsonl, scored rollouts in trajectories/ and the use/avoid rollouts in contrast/. Run scripts/score_rollouts.py first.
  python scripts/rank_inputs.py [--contrast FILE] [--json FILE]

Variety: `boilerplate` is the share of a prompt's words in sentences that at least 5% of the tasks share, once numbers are masked. `overlap` is the mean word overlap (Jaccard) of two random prompts,
once numbers and boilerplate sentences are kept: lower is more varied. `words` is the number of distinct words per 1,000 prompt words.
Saturation: `useful` is the share of 4B and 12B level cells whose mean reward lies between .1 and .8, and `top` the highest mean reward in any 1B, 4B or 12B level cell.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench import all_envs  # noqa: E402

ENVS = ROOT.parent / 'envs'


def mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def words(text):
    return re.findall(r"[^\W\d_]+|#", re.sub(r'\d+', '#', text.lower()))


def variety(env):
    tasks = [json.loads(line) for line in (ENVS / env.name / 'tasks.jsonl').open(encoding='utf8')]
    pieces = [[' '.join(words(s)) for s in re.split(r'(?<=[.!?:])\s+|\n+', task['prompt'])] for task in tasks]
    pieces = [[s for s in row if s] for row in pieces]
    counts = Counter(s for row in pieces for s in set(row))
    shared = {s for s, c in counts.items() if c >= .05 * len(tasks) and len(tasks) >= 20}
    total = sum(len(s.split()) for row in pieces for s in row)
    boiler = sum(len(s.split()) for row in pieces for s in row if s in shared)
    rng = random.Random(0)
    sets = [set(' '.join(row).split()) for row in pieces]
    pairs = [rng.sample(range(len(sets)), 2) for _ in range(4000)]
    overlap = mean(len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])) for a, b in pairs)
    flat = [w for row in pieces for s in row for w in s.split()]
    return dict(tasks=len(tasks), boilerplate=boiler / max(1, total), overlap=overlap, words=1000 * len(set(flat)) / max(1, len(flat)), distinct=len(set(flat)))


def natural(env):
    """Per model and level: solved or mean reward, UB rate, and within-model comparisons."""
    out = {}
    for path in sorted((ENVS / env.name / 'trajectories').glob('*.jsonl')):
        rows = [json.loads(line) for line in path.open(encoding='utf8')]
        tag = path.stem.removeprefix('gemma-3-')
        levels = defaultdict(list)
        for row in rows:
            levels[row['level']].append(row)
        cells = {}
        for level, some in sorted(levels.items()):
            known = [r for r in some if r['ub'] is not None]
            main = mean(r['reward'] for r in some)
            cells[level] = dict(main=main, reward=mean(r['reward'] for r in some), ub=mean(r['ub'] for r in known), pos=sum(r['ub'] for r in known), neg=sum(not r['ub'] for r in known),
                                r_pos=mean(r['reward'] for r in known if r['ub']), r_neg=mean(r['reward'] for r in known if not r['ub']))
        pairs = defaultdict(dict)
        for row in rows:
            if row['ub'] is not None:
                pairs[row['task_id']][row['rollout']] = row
        split = [p[0]['reward'] - p[1]['reward'] if p[0]['ub'] else p[1]['reward'] - p[0]['reward'] for p in pairs.values() if len(p) == 2 and p[0]['ub'] != p[1]['ub']]
        known = [r for r in rows if r['ub'] is not None]
        out[tag] = dict(cells=cells, ub=mean(r['ub'] for r in known), reward=mean(r['reward'] for r in rows),
                        same_task_n=len(split), same_task=mean(split), null=mean(r['ub'] is None for r in rows))
    return out


def summary(env, nat, contrast):
    weights, weighted = 0, 0
    cells = in_band = useful = 0
    tops = []
    for tag, data in nat.items():
        for level, cell in data['cells'].items():
            if tag in ('1b', '4b', '12b') and cell['main'] is not None:
                tops.append(cell['main'])
            if tag in ('4b', '12b') and cell['main'] is not None:
                cells += 1
                useful += .1 <= cell['main'] <= .8
            if cell['pos'] >= 5 and cell['neg'] >= 5:
                w = 2 * cell['pos'] * cell['neg'] / (cell['pos'] + cell['neg'])
                weights += w
                weighted += w * (cell['r_pos'] - cell['r_neg'])
    bands = sum(.01 <= cell['ub'] <= .5 for tag, data in nat.items() if tag in ('1b', '4b', '12b') for cell in data['cells'].values() if cell['ub'] is not None)
    total = sum(1 for tag, data in nat.items() if tag in ('1b', '4b', '12b') for cell in data['cells'].values())
    bands2 = sum(.01 <= cell['ub'] <= .5 for tag, data in nat.items() if tag in ('4b', '12b') for cell in data['cells'].values() if cell['ub'] is not None)
    diffs = [contrast[f'{env.name}|gemma-3-{tag}']['diff'] for tag in ('4b', '12b') if f'{env.name}|gemma-3-{tag}' in contrast]
    effects = [contrast[f'{env.name}|gemma-3-{tag}']['effect'] for tag in ('4b', '12b') if f'{env.name}|gemma-3-{tag}' in contrast and contrast[f'{env.name}|gemma-3-{tag}']['effect'] is not None]
    return dict(useful=useful / max(1, cells), top=max(tops) if tops else None, weighted=weighted / weights if weights else None, usable_weight=weights, band_1_12=bands, band_total=total, band_4_12=bands2,
                diff=mean(diffs), effect=mean(effects))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contrast', help='JSON from scripts/score_contrast.py')
    ap.add_argument('--json')
    args = ap.parse_args()
    contrast = json.loads(Path(args.contrast).read_text()) if args.contrast else {}
    out = {}
    print('| Env | Contrast diff 4B,12B | Effect of UB | Natural diff | UB cells in band 4B,12B | Useful cells | Top cell | Boilerplate | Overlap | Words/1000 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for env in all_envs().values():
        nat = natural(env)
        var = variety(env)
        s = summary(env, nat, contrast)
        out[env.name] = dict(natural=nat, variety=var, summary=s)
        f = lambda v, p='.2f': '–' if v is None else format(v, p)  # noqa: E731
        print(f"| {env.name[:2]} | {f(s['diff'], '+.2f')} | {f(s['effect'], '+.2f')} | {f(s['weighted'], '+.2f')} | {s['band_4_12']} | {s['useful']:.2f} | {f(s['top'])} | {var['boilerplate']:.2f} | {var['overlap']:.2f} | {var['words']:.0f} |", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
