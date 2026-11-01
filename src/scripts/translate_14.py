#!/usr/bin/env python3
"""Translate environment 14's English problems and far questions with NLLB-200 and store them in data/14_translations/.

One file per language: `<code>.jsonl` holds word problems, `far_<code>.jsonl` holds far questions. Every row keeps the English source,
translation, back-translation, numbers and answer, with `sim` (its lowest sentence similarity) and `ok` (accepted). The environment reads
only rows with `ok`. Rejected rows stay in the files so the yield can be audited. Runs append: rows already stored are kept and skipped.

Translation runs sentence by sentence, since NLLB is a sentence-level model. A sentence's translation is the first of the beam's
hypotheses that keeps its numbers, spells each person's name the one way in names.json, has no English left in it, and has a back-translation
that `matches` the source. A problem is accepted only if every sentence has one.

Usage (one idle GPU): CUDA_VISIBLE_DEVICES=N python scripts/translate_14.py near [--rl 128] [--near 64] [--langs th sw te bn]
                      CUDA_VISIBLE_DEVICES=N python scripts/translate_14.py far [--per-split 64]
"""
import argparse
import difflib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench.envs.e14_nonenglish_word_problems import (ANIMALS, ENV, FAR_LANGS, FOOD, GOODS, LANGS, MODEL, NAMES, NLLB, SIM_THRESHOLD, STORE,  # noqa: E402
                                                            english_free, far_questions, problems)
from antiobfbench.parse import ints  # noqa: E402

BATCH = int(os.environ.get('NLLB_BATCH', 64))
BEAMS, BEST = 5, 4
SMALL = ('two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()
WORD_VALUE = {**{w: i + 2 for i, w in enumerate(SMALL)}, **{w: 10 * (i + 2) for i, w in enumerate(TENS)}}
SPELLED = re.compile(r'\b(?:(' + '|'.join(SMALL) + r'|one|a) hundred(?: and)?[ -]?)?(?:(' + '|'.join(TENS) + r')[ -]?)?(' + '|'.join(SMALL) + r')?\b')
STOP = frozenset(('a an the of to in at is are was were be been being will would can could there that this these those it its his her their he she they '
                  'him them and or on for with then also so by from do does did has have had how many much what which who s t as one you your we our i my '
                  'about up all').split())
IRREGULAR = dict(bought='buy', sold='sell', gave='give', given='give', paid='pay', spent='spend', made='make', drank='drink', ate='eat', eaten='eat',
                 took='take', taken='take', got='get', children='child', shelves='shelf', loaves='loaf', left='remain', went='go', ran='run', kept='keep',
                 grew='grow', held='hold', lent='borrow', broke='broken', pieces='piece')
# Words the back-translation may use for a source word, as their stems. Both sides are mapped to the first word of a group.
GROUPS = ['shop store mall supermarket department', 'dollar buck money cash coin', 'tournament competition contest championship', 'age old',
          'pick collect harvest gather', 'storm hurricane cyclone', 'tree plant bush', 'run work operate', 'stop station', 'travel drive go', 'add pour fill', 'box crate carton package case', 'hold contain carry fit', 'bicycle bike cycle', 'student pupil',
          'child kid', 'family household', 'loaf bread', 'liter litre l', 'meter metre m', 'kilometer kilometre km', 'kilogram kilo kg', 'hour hr',
          'share split divide distribute', 'equally evenly equal even', 'remain rest remainder leftover balance', 'each every per apiece',
          'buy purchase', 'cost price', 'make earn', 'total sum altogether together combined overall', 'absent miss', 'broken defective damaged faulty',
          'off depart disembark alight exit descend', 'board enter climb', 'older elder', 'fewer less', 'drain flow leak', 'more additional extra another longer',
          'destroy ruin kill damage wreck', 'borrow lend loan', 'return back', 'wide broad width', 'long length', 'give hand offer provide', 'intact unbroken undamaged', 'surround around encircle enclose', 'present attend', 'plate dish', 'spoon teaspoon', 'cow cattle',
          'hat cap', 'cup mug', 'chicken hen', 'lemon lime', 'pig swine', 'player athlete']
# Words that decide the arithmetic or name what is counted: a back-translation that lacks one doesn't match, whatever its similarity.
CUES = frozenset(('each remain more fewer older younger times equally absent present off board long wide surround broken intact discount borrow return give sell spend save add new instead drain destroy shop market '
                  'player student passenger family friend child page tree').split()) | set(GOODS + FOOD + ANIMALS)


def spell_to_digits(text):
    """'twenty-five' to '25', 'two hundred and six' to '206'. 'one' stays a word, since no source number is 1."""
    def value(m):
        if not m.group(0).strip():
            return m.group(0)
        hundreds = {'a': 1, 'one': 1}.get(m.group(1), WORD_VALUE.get(m.group(1), 0))
        return str(100 * hundreds + WORD_VALUE.get(m.group(2), 0) + WORD_VALUE.get(m.group(3), 0))
    return SPELLED.sub(value, text)


def stem(word):
    word = IRREGULAR.get(word, word)
    if len(word) > 4 and word.endswith('ies'):
        word = word[:-3] + 'y'
    elif len(word) > 4 and word.endswith(('ches', 'shes', 'xes', 'oes', 'sses')):
        word = word[:-2]
    elif len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        word = word[:-1]
    if len(word) > 5 and word.endswith('ing'):
        word = word[:-3]
    elif len(word) > 4 and word.endswith('ed'):
        word = word[:-2]
    word = word[:-1] if len(word) > 3 and word.endswith('e') else word
    return word


CANON = {stem(w): stem(g.split()[0]) for g in GROUPS for w in g.split()}
CUE_STEMS = frozenset(CANON.get(stem(w), stem(w)) for w in CUES)


def normalize(text):
    """Lower-case tokens with digits for numbers: `$6` reads as `6 dollars`, `6 dollars a cup` as `6 dollars each cup`, `get on` as `board`."""
    text = spell_to_digits(text.lower().replace('’', "'").replace("n't", ' not'))
    text = re.sub(r'\bg(?:et|ot|ets|etting) on\b', 'board', re.sub(r'\bg(?:et|ot|ets|etting) off\b', 'off', text))
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
    text = re.sub(r'(?:about |some )?\d+ \w+ \[([^\]]+)\]', r'\1', text)       # '50 miles [86 km]': the translator's unit conversion
    text = re.sub(r'\$\s*(\d+)', r'\1 dollars', text)
    text = re.sub(r'\b(?:kmph|kph|kmh|km/hr?)\b', 'km per hour', text).replace('not broken', 'intact')
    text = re.sub(r'(?<=\d) ((?:[a-z]+ )?)an? (?=[a-z])', r' \1each ', text)
    text = re.sub(r'\b(?:a|an) (?=hour|day|week|kilo|meter|liter|month)', 'each ', text)
    return re.findall(r'[a-z]+|\d+', text)


def content(tokens, names=()):
    """Canonical content words of normalized tokens, without numbers, stop words and the given names."""
    skip = {n.lower() for n in names}
    return [CANON.get(stem(t), stem(t)) for t in tokens if not t.isdigit() and t not in STOP and t not in skip]


def similarity(source, back):
    """F1 of the content words of a source sentence and its back-translation, leaving out all first names."""
    a, b = set(content(normalize(source), NAMES)), set(content(normalize(back), NAMES))
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.


def names_of(sentence):
    return [w for w in dict.fromkeys(re.findall(r'\b[A-Z][a-z]+\b', sentence)) if w in NAMES]


def names_in_order(names, back):
    """True if each name has a look-alike in the back-translation, in order."""
    tokens, pos = re.findall(r'[a-z]+', back.lower()), 0
    for name in names:
        found = next((i for i in range(pos, len(tokens)) if difflib.SequenceMatcher(None, name.lower(), tokens[i]).ratio() >= .6), None)
        if found is None:
            return False
        pos = found + 1
    return True


def units_bound(source, back):
    """Each number must keep its unit: the content word after it must also follow it in the back-translation, within three words."""
    def pairs(text):
        tokens = [t for t in normalize(text) if t not in STOP]
        return [(t, [CANON.get(stem(x), stem(x)) for x in tokens[i + 1:i + 4] if not x.isdigit()]) for i, t in enumerate(tokens) if t.isdigit()]
    bound = dict(pairs(back))
    return all(not unit or unit[0] in bound.get(n, ()) for n, unit in pairs(source))


def digits_of(text):
    return sorted(int(t) for t in normalize(text) if t.isdigit())


def matches(source, back):
    """(ok, similarity) for a back-translation: identical numbers, every arithmetic cue, names in order if there are several, units bound, similar."""
    sim, names = similarity(source, back), names_of(source)
    have = set(content(normalize(back), NAMES))
    ok = (digits_of(back) == digits_of(source) and set(content(normalize(source), NAMES)) & CUE_STEMS <= have and (len(names) < 2 or names_in_order(names, back))
          and units_bound(source, back) and sim >= SIM_THRESHOLD)
    return ok, round(sim, 3)


class Translator:
    def __init__(self, name=MODEL):
        import torch
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
        self.torch, self.tok = torch, NllbTokenizer.from_pretrained(name)   # AutoTokenizer loads this model without its source language tag
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
        self.sentences, self.spelled = 0, {}

    def spellings(self, lang):
        """How a language spells each person's name: the translation of the name alone, kept in names.json."""
        path = STORE / 'names.json'
        if not self.spelled and path.exists():
            self.spelled = json.loads(path.read_text(encoding='utf8'))
        if lang not in self.spelled:
            self.spelled[lang] = dict(zip(NAMES, (h[0] for h in self.run(NAMES, 'eng_Latn', NLLB[lang]))))
            path.write_text(json.dumps(self.spelled, ensure_ascii=False, indent=1), encoding='utf8')
        return self.spelled[lang]

    def run(self, texts, src, tgt):
        """Translate sentences. For each one return the beam's distinct hypotheses, best first."""
        out, order = {}, sorted(set(texts), key=len)
        self.tok.src_lang = src
        for i in range(0, len(order), BATCH):
            chunk = order[i:i + BATCH]
            enc = self.tok(chunk, return_tensors='pt', padding=True, truncation=True, max_length=256).to('cuda')
            with self.torch.inference_mode():
                ids = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(tgt), num_beams=BEAMS, num_return_sequences=BEST,
                                          max_new_tokens=int(enc['input_ids'].shape[1] * 2.5) + 16, max_length=None)
            decoded = self.tok.batch_decode(ids, skip_special_tokens=True)
            for j, text in enumerate(chunk):
                out[text] = list(dict.fromkeys(decoded[j * BEST:(j + 1) * BEST]))
            self.sentences += len(chunk)
        return [out[t] for t in texts]


def translate_sentences(tr, sentences, lang):
    """For each distinct sentence: (translation, back-translation, sim, ok). Without an accepted hypothesis, the best failure, with ok False."""
    code, done, spelled = NLLB[lang], {}, tr.spellings(lang)
    order = list(dict.fromkeys(sentences))
    candidates = {}
    for s, hyps in zip(order, tr.run(order, 'eng_Latn', code)):
        keep = [h for h in hyps if sorted(ints(h)) == sorted(ints(s)) and english_free(h, lang, s) and all(spelled[n] in h for n in names_of(s))]
        candidates[s] = keep
        done[s] = (hyps[0], '', 0., False)
    for k in range(BEST):
        todo = [s for s in order if len(candidates[s]) > k and not done[s][3]]
        if not todo:
            break
        for s, backs in zip(todo, tr.run([candidates[s][k] for s in todo], code, 'eng_Latn')):
            rated = [(*matches(s, b), b) for b in backs]
            ok, sim, back = next((r for r in rated if r[0]), max(rated, key=lambda r: r[1]))
            if ok or sim > done[s][2] or k == 0:
                done[s] = (candidates[s][k], back, sim, ok)
    return done


def translate_rows(tr, rows, lang):
    """Fill in translation, back-translation, sim and ok for whole problems."""
    done = translate_sentences(tr, [s for row in rows for s in row['sentences']], lang)
    for row in rows:
        results = [done[s] for s in row['sentences']]
        row['translation'], row['back'] = ' '.join(r[0] for r in results), [r[1] for r in results]
        row['sim'] = min(r[2] for r in results)
        row['ok'] = all(r[3] for r in results)
        row['failed'] = [s for s, r in zip(row['sentences'], results) if not r[3]]
    return rows


def load(path):
    return [json.loads(line) for line in open(path, encoding='utf8')] if path.exists() else []


def append(path, rows):
    with open(path, 'a', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def near(tr, args):
    seen = {row['group'] for lang in LANGS for row in load(STORE / f'{lang}.jsonl')}
    for lang in args.langs:
        path, t0, s0 = STORE / f'{lang}.jsonl', time.time(), tr.sentences
        rows = load(path)
        counts = Counter(ENV.pool_of(r['group'], None) for r in rows if r['ok'])
        stream = problems(lang, seen)
        while counts['rl'] < args.rl or counts['near_train'] + counts['near_test'] < args.near:
            batch = translate_rows(tr, [next(stream) for _ in range(args.batch)], lang)
            append(path, batch)
            rows += batch
            counts = Counter(ENV.pool_of(r['group'], None) for r in rows if r['ok'])
            print(f'{lang}: {len(rows)} tried, {sum(counts.values())} ok, {dict(counts)}', flush=True)
        print(f'{lang}: yield {sum(counts.values()) / max(1, len(rows)):.3f}; {tr.sentences - s0} translations in {time.time() - t0:.0f}s', flush=True)


def far(tr, args):
    rows = {lang: load(STORE / f'far_{lang}.jsonl') for lang in FAR_LANGS}
    stream = {lang: far_questions(lang, {r['key'] for r in rows[lang]}) for lang in FAR_LANGS}
    t0, done = time.time(), {lang: False for lang in FAR_LANGS}

    def counts():
        return Counter(ENV.far_pool_of(r['group']) for lang in FAR_LANGS for r in rows[lang] if r['ok'])

    while min(counts()[s] for s in ('far_train', 'far_test')) < args.per_split and not all(done.values()):
        for lang in FAR_LANGS:
            batch = []
            for _ in range(args.batch):
                row = next(stream[lang], None)
                if row is None:
                    done[lang] = True
                    break
                batch.append(row)
            if batch:
                batch = translate_rows(tr, batch, lang)
                append(STORE / f'far_{lang}.jsonl', batch)
                rows[lang] += batch
        print(dict(counts()), {lang: (sum(r['ok'] for r in rows[lang]), len(rows[lang])) for lang in FAR_LANGS}, flush=True)
    print(f'far: {tr.sentences} translations in {time.time() - t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=('near', 'far'))
    ap.add_argument('--rl', type=int, default=128)
    ap.add_argument('--near', type=int, default=64)
    ap.add_argument('--per-split', type=int, default=64)
    ap.add_argument('--batch', type=int, default=100)
    ap.add_argument('--langs', nargs='*', default=list(LANGS))
    args = ap.parse_args()
    STORE.mkdir(exist_ok=True)
    tr = Translator()
    (near if args.what == 'near' else far)(tr, args)


if __name__ == '__main__':
    main()
