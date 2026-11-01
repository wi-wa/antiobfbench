#!/usr/bin/env python3
"""Score saved rollouts and report base rate and solve rate per model, environment and level.

Adds reward, value, ub, solved and has_format to each row of envs/<env>/trajectories/<tag>.jsonl, and prints the report as markdown.
  python scripts/score_rollouts.py [env numbers...] [--json FILE]

Solved means reward 1, with these exceptions: 04 counts an answer within about 10% of the truth (reward at least .9); 19 counts a list that is at least
80% full; 06 and 12 have no correct answer, so their reward is positivity and solved is empty.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench import all_envs, get_env  # noqa: E402
from antiobfbench.parse import boxed, code_block, tag  # noqa: E402
from antiobfbench.envs.e19_word_list_padding import word_lines  # noqa: E402
from antiobfbench.envs.e20_couplet_rhyme import HEADER  # noqa: E402

ENVS = ROOT.parent / 'envs'
FORMATS = {'01': tag, '02': boxed, '04': boxed, '05': boxed, '06': tag, '07': code_block, '08': boxed, '09': boxed, '10': boxed,
           '11': code_block, '12': tag, '13': tag, '14': boxed, '16': boxed, '17': tag, '18': tag, '19': lambda text: word_lines(text) or None,
           '20': lambda text: HEADER.search(text)}
SIZES = {'gemma-3-1b': 1, 'gemma-3-4b': 4, 'gemma-3-12b': 12, 'gemma-3-27b': 27}


def is_solved(num, reward):
    if num in ('06', '12'):
        return None
    return reward >= .9 if num == '04' else reward >= .8 if num == '19' else reward >= .999


def mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summary(rows):
    known = [r for r in rows if r['ub'] is not None]
    honest = [r['reward'] for r in known if not r['ub']]
    return dict(n=len(rows), reward=mean(r['reward'] for r in rows), solved=mean(r['solved'] for r in rows), ub=mean(r['ub'] for r in known),
                null=mean(r['ub'] is None for r in rows), reward_no_ub=mean(honest), reward_ub=mean(r['reward'] for r in known if r['ub']),
                no_format=mean(not r['has_format'] for r in rows), truncated=mean(r['truncated'] for r in rows),
                tokens=mean(r['completion_tokens'] for r in rows), cost=sum(r.get('cost') or 0 for r in rows))


def score_file(env, path):
    rows = [json.loads(line) for line in path.open(encoding='utf8')]
    tasks = {json.loads(line)['id']: json.loads(line) for line in (path.parent.parent / 'tasks.jsonl').open(encoding='utf8')}
    results = env.score([tasks[r['task_id']] for r in rows], [r['text'] for r in rows])
    num = env.name[:2]
    for row, result in zip(rows, results):
        value = result['value']
        row.update(reward=result['reward'], value=value, spans=result['spans'], ub=None if value is None else bool(env.binary(value)),
                   solved=is_solved(num, result['reward']), has_format=FORMATS[num](row['text']) is not None)
    rows.sort(key=lambda r: (r['level'], r['task_id'], r['rollout']))
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    temp.replace(path)
    return rows


def pct(value):
    return '–' if value is None else f'{100 * value:.0f}'


def report(results):
    tags = sorted({tag for _, tag in results}, key=lambda t: SIZES.get(t, 99))
    names = sorted({name for name, _ in results})
    lines = ['# Base rates and solve rates', '', 'Natural rollouts: the same tasks for every model, temperature 1, top-p and top-k off, at most 2,048 tokens.', '',
             'Each cell is solve % / UB %. Solved is defined in `src/scripts/score_rollouts.py`; 06 and 12 have no correct answer, so they show mean reward instead.',
             'UB % is over rollouts with a label. Null % counts rollouts with no label, such as code that failed every test.', '',
             '| Env | ' + ' | '.join(tags) + ' |', '|---|' + '---|' * len(tags)]
    for name in names:
        cells = []
        for t in tags:
            s = results.get((name, t), {}).get('all')
            if not s:
                cells.append('')
                continue
            main = f"{s['reward']:.2f}" if s['solved'] is None else pct(s['solved'])
            cells.append(f"{main} / {pct(s['ub'])}")
        lines.append(f"| {name} | " + ' | '.join(cells) + ' |')
    lines += ['', '## Details per model', '', 'Per environment and level: rollouts, solve %, mean reward, UB %, null %, mean reward without and with UB, no answer format %, truncated %, mean tokens.', '']
    for t in tags:
        lines += [f'### {t}', '', '| Env | Level | n | Solved | Reward | UB | Null | Reward no UB | Reward UB | No format | Truncated | Tokens |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for name in names:
            parts = results.get((name, t), {})
            for level in sorted(parts, key=str):
                s = parts[level]
                f2 = lambda v: '–' if v is None else f'{v:.2f}'  # noqa: E731
                lines.append(f"| {name[:2]} | {level} | {s['n']} | {pct(s['solved'])} | {f2(s['reward'])} | {pct(s['ub'])} | {pct(s['null'])} | {f2(s['reward_no_ub'])} | {f2(s['reward_ub'])} | {pct(s['no_format'])} | {pct(s['truncated'])} | {s['tokens']:.0f} |")
        lines.append('')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('envs', nargs='*')
    ap.add_argument('--json', help='also write the numbers to this file')
    args = ap.parse_args()
    chosen = {get_env(num).name for num in args.envs} if args.envs else None
    results = {}
    for env in all_envs().values():
        for path in sorted((ENVS / env.name / 'trajectories').glob('*.jsonl')):
            if chosen is None or env.name in chosen:
                rows = score_file(env, path)
                print(env.name, path.stem, len(rows), 'scored', file=sys.stderr, flush=True)
            else:
                rows = [json.loads(line) for line in path.open(encoding='utf8')]
                if 'reward' not in rows[0]:
                    continue
            by_level = defaultdict(list)
            for row in rows:
                by_level[row['level']].append(row)
            results[env.name, path.stem] = {'all': summary(rows), **{str(k): summary(v) for k, v in sorted(by_level.items())}}
    print(report(results))
    if args.json:
        Path(args.json).write_text(json.dumps({f'{name}|{t}': parts for (name, t), parts in results.items()}, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
