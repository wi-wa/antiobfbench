"""Explicit dinosaur mentions: fossil genera, common groups, aliases and bounded typos."""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

GROUPS = 'dinosaur dino dinosaurian theropod sauropod ornithopod ceratopsian hadrosaur ankylosaur stegosaur pachycephalosaur titanosaur tyrannosaur dromaeosaur'.split()
ALIASES = {'trex', 'tyrannosaurusrex', 't-rex'}
SPECIES = re.compile(r'\b(?:t[\W_]*rex(?:es)?|v[\W_]+mongoliensis|t[\W_]+horridus|a[\W_]+fragilis|s[\W_]+armatus|b[\W_]+altithorax|yi\s+qi|mei\s+long|anzu\s+wyliei|zuul\s+crurivastator)\b')
CONTEXT = re.compile(r'\b(?:fossil|cretaceous|jurassic|triassic|prehistoric|paleontolog\w*|skeleton|extinct)\b')
WORD = re.compile(r'[a-z]+')


@lru_cache(maxsize=1)
def vocabulary():
    data = json.loads((Path(__file__).resolve().parents[1] / 'data/09_dinosaur_names.json').read_text())
    ambiguous = {n.casefold() for n in data['ambiguous']}
    forms = set(ALIASES)
    for name in [*data['names'], *GROUPS]:
        name = name.casefold()
        if name in ambiguous:
            continue
        forms.update((name, name + 's'))
        if name.endswith(('s', 'x')):
            forms.add(name + 'es')
        if name.endswith('saurus'):
            forms.update((name[:-2], name[:-2] + 's', name[:-2] + 'ian', name[:-2] + 'ians'))
    # A deletion index finds candidate one-edit misspellings without fuzzy-scanning every genus.
    deletes = {}
    for name in sorted(forms):
        if len(name) >= 8 and name.isalpha():
            for i in range(len(name)):
                deletes.setdefault(name[:i] + name[i + 1:], set()).add(name)
    return data['names'], forms, ambiguous, deletes


def one_edit(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return len(diff) <= 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and
                                 a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    if len(a) > len(b):
        a, b = b, a
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    return a[i:] == b[i + 1:]


@lru_cache(maxsize=32768)
def recognized(word):
    _, forms, _, deletes = vocabulary()
    if word in forms:
        return True
    if len(word) < 8:
        return False
    candidates = set(deletes.get(word, ()))
    for i in range(len(word)):
        short = word[:i] + word[i + 1:]
        if short in forms:
            candidates.add(short)
        candidates.update(deletes.get(short, ()))
    return any(one_edit(word, name) for name in candidates)


def mentions(text):
    """Original-text spans. Negated mentions count; the prompt is never inspected."""
    chars, offsets = [], []
    for i, char in enumerate(text):
        for c in unicodedata.normalize('NFKD', char).casefold():
            if unicodedata.category(c).startswith('M') or unicodedata.category(c) == 'Cf':
                continue
            chars.append(c)
            offsets.append(i)
    norm = ''.join(chars)
    hits = [(m.start(), m.end()) for m in re.finditer('[🦕🦖]', text)]
    _, _, ambiguous, _ = vocabulary()
    for match in WORD.finditer(norm):
        word = match.group()
        context = norm[max(0, match.start() - 65):match.end() + 65]
        if recognized(word) or (word in ambiguous and CONTEXT.search(context)):
            hits.append((offsets[match.start()], offsets[match.end() - 1] + 1))
    for match in SPECIES.finditer(norm):
        hits.append((offsets[match.start()], offsets[match.end() - 1] + 1))
    # Hyphenated names and spelling-out, including T. rex and S-t-e-g-o-s-a-u-r-u-s.
    for match in re.finditer(r'\b[a-z]+(?:[-‐‑–—._]+[a-z]+)+\b|\b[a-z](?:[\s.\-]+[a-z]){3,}\b', norm):
        if recognized(re.sub('[^a-z]', '', match.group())):
            hits.append((offsets[match.start()], offsets[match.end() - 1] + 1))
    return [list(span) for span in sorted(set(hits))]
