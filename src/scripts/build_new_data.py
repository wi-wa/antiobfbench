#!/usr/bin/env python3
"""Build the data files of environments 16-20 in src/data from their public sources. Deterministic. See src/data/SOURCES.md.

Usage: python scripts/build_new_data.py RAW_DIR
RAW_DIR holds TwoHopFact.csv, hops_all4b.jsonl and hops_all12b.jsonl (from scripts/calibrate_two_hop.py), squad_train.jsonl, xquad.jsonl, poetry.jsonl,
cmudict.dict, cmudict.LICENSE and google10k.txt.
"""
import ast
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench.envs.e16_two_hop_recall import clean_names  # noqa: E402
from antiobfbench.envs.e17_source_copying import facts_of, kept_together, tokens  # noqa: E402
from antiobfbench.envs.e20_couplet_rhyme import band_of, last_word  # noqa: E402

DATA = ROOT / 'data'
csv.field_size_limit(10 ** 8)


def names(raw):
    return sorted({a.strip() for group in ast.literal_eval(raw) for a in group if a.strip()})


def two_hop(raw):
    flags = {}
    for tag, known, direct in (('all4b', 'k4', 'd4'), ('all12b', 'k12', 'd12')):
        with open(raw / f'hops_{tag}.jsonl', encoding='utf8') as f:
            for line in f:
                r = json.loads(line)
                flags.setdefault(r['uid'], {}).update({known: r['hop1'] and r['hop2'], direct: r['direct']})
    count = 0
    table = dict(bridges={}, answers={})
    with open(raw / 'TwoHopFact.csv', encoding='utf8') as f, open(DATA / 'two_hop.jsonl', 'w', encoding='utf8') as out:
        for r in csv.DictReader(f):
            if r['uid'] not in flags or len(flags[r['uid']]) < 4:
                continue
            question = r['r2(r1(e1)).prompt']
            bridge = r['e2.value']
            bridge_names = clean_names(bridge, names(r['e2.aliases']), r['e2.rough_category'])
            answers = sorted({r['e3.value']} | {a for a in names(r['e3.aliases']) if 2 <= len(a) <= 40 and a.isprintable()})
            if not bridge_names or r['e3.value'] in question or any(re.search(rf'(?<!\w){re.escape(a)}(?!\w)', question, re.I) for a in bridge_names):
                continue
            table['bridges'][bridge] = sorted(set(table['bridges'].get(bridge, [])) | set(bridge_names))
            table['answers'][r['e3.value']] = sorted(set(table['answers'].get(r['e3.value'], [])) | set(answers))
            out.write(json.dumps(dict(type=r['fact_comp_type'], question=question, describes=r['mu.template'].format(r['e1.value']) if '{}' in r['mu.template'] else r['r2(r1(e1)).subject'],
                                      e1=r['e1.value'], bridge=bridge, bridge_kind=r['e2.rough_category'], answer=r['e3.value'], answer_kind=r['e3.rough_category'], **flags[r['uid']]),
                                 ensure_ascii=False) + '\n')
            count += 1
    (DATA / 'two_hop_names.json').write_text(json.dumps(table, ensure_ascii=False), encoding='utf8')
    print('two_hop', count)


def passages(raw):
    seen, count = set(), 0
    with open(raw / 'squad_train.jsonl', encoding='utf8') as f, open(DATA / 'passages.jsonl', 'w', encoding='utf8') as out:
        for line in f:
            r = json.loads(line)
            text = ' '.join(r['context'].split())
            facts = facts_of(text)
            if text in seen or not 300 <= len(text) <= 1400 or not 5 <= len(facts) <= 50 or not kept_together(text, facts, text):
                continue
            seen.add(text)
            out.write(json.dumps(dict(title=r['title'], text=text), ensure_ascii=False) + '\n')
            count += 1
    print('passages', count)
    count = 0
    with open(raw / 'xquad.jsonl', encoding='utf8') as f, open(DATA / 'passages_far.jsonl', 'w', encoding='utf8') as out:
        for line in f:
            r = json.loads(line)
            text = ' '.join(r['context'].replace('﻿', '').split())
            if 350 <= len(text) <= 1400 and len(tokens(text)) >= 40:
                out.write(json.dumps(dict(lang=r['lang'], id=r['qid'], text=text), ensure_ascii=False) + '\n')
                count += 1
    print('passages_far', count)


def verses(raw):
    """Verse lines of the poetry corpus that could open a couplet: complete-looking, plain and ending in a word with enough rhymes, at most 100 per word."""
    seen, count, per_word = set(), 0, Counter()
    with open(raw / 'poetry.jsonl', encoding='utf8') as f, open(DATA / 'verse_lines.txt', 'w', encoding='utf8') as out:
        for i, row in enumerate(f):
            if i % 3:
                continue
            line = json.loads(row)['line'].strip()
            if not 25 <= len(line) <= 60 or not line[0].isupper() or not line.isascii() or re.search(r'\d|Gutenberg|[A-Z]{3}|_|\[|\]', line):
                continue
            words = re.findall(r"[a-z']+", line.lower())
            if len(words) < 5 or line in seen or per_word[words[-1]] >= 100 or band_of(last_word(line)) is None:
                continue
            seen.add(line)
            per_word[words[-1]] += 1
            out.write(line + '\n')
            count += 1
    print('verse_lines', count)


def main():
    raw = Path(sys.argv[1])
    shutil.copy(raw / 'cmudict.dict', DATA / 'cmudict.dict')
    shutil.copy(raw / 'cmudict.LICENSE', DATA / 'cmudict.LICENSE')
    shutil.copy(raw / 'google10k.txt', DATA / 'common_words.txt')
    two_hop(raw)
    passages(raw)
    verses(raw)


if __name__ == '__main__':
    main()
