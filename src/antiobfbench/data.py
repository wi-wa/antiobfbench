"""Loaders for the fixed data files in data/."""
import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'


@lru_cache(maxsize=None)
def cities():
    """Gazetteer rows from geonamescache 3.0.2: name, country, cc, pop, lat, lon. Sorted; unique on (name, cc)."""
    return json.loads((DATA / 'cities.json').read_text(encoding='utf8'))


@lru_cache(maxsize=None)
def countries():
    """Country fields used by population anchoring and farthest-north probe data."""
    return json.loads((DATA / 'countries.json').read_text(encoding='utf8'))


@lru_cache(maxsize=None)
def enable_words():
    """ENABLE word list (public domain): English dictionary words and inflections, no proper names. Sorted."""
    return (DATA / 'enable1.txt').read_text(encoding='ascii').split()


@lru_cache(maxsize=None)
def verse_lines():
    """Sorted, unique Gutenberg verse lines in plain lower case, with 6 to 9 words."""
    return (DATA / 'verse_lines.txt').read_text(encoding='utf8').split('\n')[:-1]
