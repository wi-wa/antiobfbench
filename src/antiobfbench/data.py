"""Loaders for the fixed data files in data/."""
import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'


@lru_cache(maxsize=None)
def cities():
    """Gazetteer rows from geonamescache 3.0.2: name, country, cc, pop, lat, lon. Sorted; unique on (name, cc)."""
    return json.loads((DATA / 'cities.json').read_text(encoding='utf8'))


@lru_cache(maxsize=None)
def countries():
    """Country table with name, capital, currencies, area, latlng and more."""
    return json.loads((DATA / 'countries.json').read_text(encoding='utf8'))


@lru_cache(maxsize=None)
def enable_words():
    """ENABLE word list (public domain): English dictionary words and inflections, no proper names. Sorted."""
    return (DATA / 'enable1.txt').read_text(encoding='ascii').split()


@lru_cache(maxsize=None)
def common_words():
    """Common English words, most frequent first."""
    return (DATA / 'common_words.txt').read_text(encoding='ascii').split()


@lru_cache(maxsize=None)
def cmu_pronunciations():
    """CMU Pronouncing Dictionary: word -> its pronunciations, each a list of phones."""
    out = {}
    with (DATA / 'cmudict.dict').open(encoding='latin1') as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line:
                word, *phones = line.split()
                out.setdefault(re.sub(r'\(\d+\)$', '', word), []).append(phones)
    return out


def _jsonl(name):
    with (DATA / name).open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=None)
def two_hop_chains():
    """TwoHopFact chains, with what 4B and 12B knew of them: k4 and k12 for both facts, d4 and d12 for the two-hop question. Names come from a table of their own, since entities repeat."""
    names = json.loads((DATA / 'two_hop_names.json').read_text(encoding='utf8'))
    return [dict(row, bridge_names=names['bridges'][row['bridge']], answer_names=names['answers'][row['answer']]) for row in _jsonl('two_hop.jsonl')]


@lru_cache(maxsize=None)
def passages():
    """SQuAD training contexts: title and text."""
    return _jsonl('passages.jsonl')


@lru_cache(maxsize=None)
def passages_far():
    """XQuAD contexts in 11 languages: lang, id (shared across languages) and text."""
    return _jsonl('passages_far.jsonl')


@lru_cache(maxsize=None)
def verse_lines():
    """Lines from the Gutenberg Poetry Corpus that could open a couplet, in corpus order."""
    return (DATA / 'verse_lines.txt').read_text(encoding='utf8').split('\n')[:-1]
