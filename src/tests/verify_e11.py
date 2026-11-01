#!/usr/bin/env python3
"""Run 11's references through the sandbox on sampled tasks of every base problem and level.

Each sampled task must have: both references passing its hidden tests, the efficient reference finishing the stress
input within 1 s, and the brute-force reference timing out at 2 s. Exit status 0 means no base problem failed.

Usage: python tests/verify_e11.py [--per-level 3] [--workers 16] [--bases name ...] [--seed 0]
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from antiobfbench.envs.e11_code_brute_force import BASES, ENV, State, make_tests, reference_code  # noqa: E402
from antiobfbench.sandbox import bounded_map, run_stress, run_tests  # noqa: E402

FAST_LIMIT, BRUTE_LIMIT = 1.0, 2.0
TRIES = 50


def strata(base):
    """The base's parameters grouped by their non-numeric fields (mode, direction, return kind, ...)."""
    groups = defaultdict(list)
    for p in base.params:
        groups[tuple(sorted((k, v) for k, v in p.items() if isinstance(v, (str, bool))))].append(p)
    return list(groups.values())


def task_with(base, p, steps, rng, level):
    draft = ENV.draft(rng, base, p, steps)
    return draft and dict(id=f'{base.name}-L{level}', level=level, prompt=draft['prompt'], answer=draft['answer'], meta=draft['meta'])


def sample(base, per_level, rng):
    """Tasks covering each stratum at every level, and each input step alone at its hardest value."""
    tasks, groups = [], strata(base)
    rng.shuffle(groups)
    for level in (1, 2, 3, 4):
        count = max(per_level, min(len(groups), 2 * per_level))
        for i in range(count):
            for _ in range(TRIES):
                p = rng.choice(groups[i % len(groups)])
                steps = base.draw_steps(rng, p, level - 1)
                task = steps is not None and task_with(base, p, steps, rng, level)
                if task:
                    tasks.append(task)
                    break
    for kind, (values, _, _) in base.in_steps.items():
        hardest = {'every': 3, 'no_mult': 3, 'drop_first': 6, 'drop_last': 6, 'mul': 9, 'add': 30}.get(kind, rng.choice(list(values)))
        for _ in range(TRIES):
            task = task_with(base, rng.choice(rng.choice(groups)), [[kind, hardest]], rng, 2)
            if task:
                tasks.append(task)
                break
    return tasks


def verify(task):
    m = task['meta']
    rows = list(zip(m['tests'], task['answer']))
    fast, brute = reference_code(task, 'fast'), reference_code(task, 'brute')
    tests_ok = all(run_tests(fast, m['name'], rows)) and all(run_tests(brute, m['name'], rows))
    outcome, seconds = run_stress(fast, m['name'], m['stress'], limit=FAST_LIMIT)
    slow = run_stress(brute, m['name'], m['stress'], limit=BRUTE_LIMIT)[0]
    return dict(base=m['base'], level=task['level'], params=m['params'], steps=m['steps'], tests_ok=tests_ok, fast=outcome, seconds=seconds, brute=slow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-level', type=int, default=3)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bases', nargs='*', default=sorted(BASES))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    tasks = [task for name in args.bases for task in sample(BASES[name], args.per_level, rng)]
    print(f'{len(tasks)} tasks over {len(args.bases)} base problems', flush=True)
    rows = bounded_map(verify, tasks, args.workers)
    bad = []
    for name in args.bases:
        mine = [r for r in rows if r['base'] == name]
        wrong = [r for r in mine if not r['tests_ok'] or r['fast'] != 'finished' or r['brute'] != 'timeout']
        worst = max((r['seconds'] or 0 for r in mine), default=0)
        print(f"{name:18} tasks {len(mine):3}  slowest efficient {worst:.2f}s  failures {len(wrong)}", flush=True)
        if wrong:
            bad.append(name)
        for r in wrong[:6]:
            print(f"   L{r['level']} tests_ok={r['tests_ok']} fast={r['fast']} {r['seconds']} brute={r['brute']} {r['params']} {r['steps']}", flush=True)
    print('FAILED:', ' '.join(bad) if bad else 'none')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
