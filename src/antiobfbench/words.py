"""English dictionary and word-family normalization."""
import re
from functools import lru_cache

from .data import enable_words

# Endings of inflected and simply suffixed forms, with what to put back to reach the base word. * undoubles a final consonant. A final e comes first, as the spelling rule goes: hoped is hope, and hopped is hop. -er and -est are left out, because too many words only look
# like them: number, mother, forest.
ENDINGS = (('iness', ('y',)), ('iest', ('y',)), ('ying', ('ie',)), ('ings', ('e', '', '*')), ('ness', ('',)), ('less', ('',)), ('ments', ('',)), ('ment', ('',)), ('ful', ('',)),
           ('ing', ('e', '', '*')), ('ies', ('y',)), ('ied', ('y',)), ('ier', ('y',)), ('ily', ('y',)), ('ed', ('e', '', '*')), ('es', ('',)), ('ly', ('le', '')), ('s', ('',)))
ENDINGS = tuple(sorted(ENDINGS, key=lambda rule: -len(rule[0])))
LONGER = ('ly', 'ment', 'ments')     # stems of 3 letters are too often accidents: early, belly, apply, cement
STEM_VOWEL = ('ed', 'ing', 'ings')   # the stem must hold a vowel and not be a short one ending in e, or else bed gives bee and seed gives see


@lru_cache(maxsize=None)
def dictionary():
    return frozenset(enable_words())


@lru_cache(maxsize=None)
def root(word):
    """The base of a dictionary word: uses, used, using and useful all give use. Only inflections and plain endings count, so relation stays apart from relate."""
    words = dictionary()
    for ending, restores in ENDINGS:
        if not word.endswith(ending):
            continue
        stem = word[:-len(ending)]
        for restore in restores:
            if restore == '*':
                candidate = stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else None
            else:
                candidate = stem + restore
            if not candidate or candidate == word or candidate not in words or len(candidate) < 3 or (ending in LONGER and len(stem) < 4):
                continue
            if ending == 'es' and not stem.endswith(('s', 'x', 'z', 'ch', 'sh', 'o')):
                continue
            if ending in STEM_VOWEL and (not re.search('[aeiouy]', stem) or (restore == 'e' and stem.endswith('e') and len(stem) < 3)):
                continue
            return root(candidate)
    return word
