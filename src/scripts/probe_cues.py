#!/usr/bin/env python3
"""How far could a shortcut go on kept probe pairs? Scores simple cues on one pair at a time: which of the two texts is the positive?

Cues: longer text, more digits, more lines, more commas, more capitalized words, and the single word that best separates the classes on the train split, applied to the test split.
A cue at .5 knows nothing, and a cue near 1 means a probe could learn it instead of the behavior.
  python scripts/probe_cues.py FILES...      (jsonl of probe samples, *_train.jsonl next to *_test.jsonl)
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_pairs(path):
    pairs = defaultdict(list)
    with open(path, encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            pairs[row['pair_id']].append(row)
    out = []
    for rows in pairs.values():
        positive = [r for r in rows if r['label'] is True]
        negative = [r for r in rows if r['label'] is False]
        if positive and negative:
            out.append((positive[0]['text'], negative[0]['text']))
    return out


def tokens(text):
    return re.findall(r"[^\W\d_]+", text.lower())


CUES = dict(length=len, digits=lambda t: sum(c.isdigit() for c in t), lines=lambda t: t.count('\n'), commas=lambda t: t.count(','), capitals=lambda t: len(re.findall(r'\b[A-Z]', t)))


def accuracy(pairs, cue):
    """Share of pairs where the cue is higher on the positive, counting ties as half, or its mirror image if that is higher."""
    score = sum(.5 if cue(a) == cue(b) else float(cue(a) > cue(b)) for a, b in pairs) / max(1, len(pairs))
    return max(score, 1 - score)


def best_word(train, test):
    """The word whose presence best tells the positive from the negative on train, and how it does on test."""
    counts = Counter()
    for a, b in train:
        for word in set(tokens(a)) ^ set(tokens(b)):
            counts[word] += 1 if word in set(tokens(a)) else -1
    if not counts:
        return None, .5
    word = max(sorted(counts), key=lambda w: abs(counts[w]))
    sign = 1 if counts[word] > 0 else -1
    has = lambda t: sign * (word in set(tokens(t)))  # noqa: E731
    score = sum(.5 if has(a) == has(b) else float(has(a) > has(b)) for a, b in test) / max(1, len(test))
    return word, score


def main():
    print('| Env | Split | Kept pairs | Length | Digits | Lines | Commas | Capitals | Best word on train | On test |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---|---:|')
    for path in map(Path, sys.argv[1:]):
        if not path.stem.endswith('test'):
            continue
        train = read_pairs(path.with_name(path.name.replace('_test', '_train')))
        test = read_pairs(path)
        env, kind = path.stem.split('.')[0], path.stem.split('.')[-1].split('_')[0]
        word, score = best_word(train, test)
        print(f"| {env[:2]} | {kind} | {len(train)} + {len(test)} | " + ' | '.join(f'{accuracy(train + test, cue):.2f}' for cue in CUES.values()) + f' | {word} | {score:.2f} |')


if __name__ == '__main__':
    main()
