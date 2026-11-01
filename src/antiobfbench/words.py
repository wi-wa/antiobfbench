"""English word knowledge shared by the word-list, story and rhyme environments: dictionary, common words, word families and rhyme."""
import re
from functools import lru_cache

from .data import cmu_pronunciations, common_words, enable_words

# Endings of inflected and simply suffixed forms, with what to put back to reach the base word. * undoubles a final consonant. A final e comes first, as the spelling rule goes: hoped is hope, and hopped is hop. -er and -est are left out, because too many words only look
# like them: number, mother, forest.
ENDINGS = (('iness', ('y',)), ('iest', ('y',)), ('ying', ('ie',)), ('ings', ('e', '', '*')), ('ness', ('',)), ('less', ('',)), ('ments', ('',)), ('ment', ('',)), ('ful', ('',)),
           ('ing', ('e', '', '*')), ('ies', ('y',)), ('ied', ('y',)), ('ier', ('y',)), ('ily', ('y',)), ('ed', ('e', '', '*')), ('es', ('',)), ('ly', ('le', '')), ('s', ('',)))
ENDINGS = tuple(sorted(ENDINGS, key=lambda rule: -len(rule[0])))
LONGER = ('ly', 'ment', 'ments')     # stems of 3 letters are too often accidents: early, belly, apply, cement
STEM_VOWEL = ('ed', 'ing', 'ings')   # the stem must hold a vowel and not be a short one ending in e, or else bed gives bee and seed gives see
VOWEL = re.compile(r'[AEIOU]')


@lru_cache(maxsize=None)
def dictionary():
    return frozenset(enable_words())


@lru_cache(maxsize=None)
def common():
    """Common English words that are also dictionary words."""
    return frozenset(w for w in common_words() if w in dictionary())


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


def _last_stress(phones):
    stressed = [i for i, p in enumerate(phones) if p[-1] in '12']
    vowels = stressed or [i for i, p in enumerate(phones) if VOWEL.match(p)]
    return vowels[-1] if vowels else None


def rime(phones):
    """The sounds from the last stressed vowel to the end of the word, without stress marks. Two words rhyme if these match."""
    i = _last_stress(phones)
    return None if i is None else ' '.join(re.sub(r'\d', '', p) for p in phones[i:])


def sound(phones):
    """The rime with all the consonants before it. Two words with the same sound rhyme identically, as night and knight or light and delight do."""
    i = _last_stress(phones)
    if i is None:
        return None
    start = i
    while start > 0 and not VOWEL.match(phones[start - 1]):
        start -= 1
    return ' '.join(re.sub(r'\d', '', p) for p in phones[start:])


def rimes(word):
    return {rime(p) for p in cmu_pronunciations().get(word, ())} - {None}


def rhymes(a, b):
    """True if some pronunciation of a rhymes with some pronunciation of b. By this test a word rhymes with itself."""
    return bool(rimes(a) & rimes(b))


def compound_of(longer, shorter):
    """True if the longer word is a dictionary word of at least 3 letters followed by the shorter one, and the shorter starts on a consonant sound: moonlight and light, but not disable and able."""
    if not (len(longer) > len(shorter) >= 3 and longer.endswith(shorter) and len(longer) - len(shorter) >= 3 and longer[:-len(shorter)] in dictionary()):
        return False
    return any(not VOWEL.match(phones[0]) for phones in cmu_pronunciations().get(shorter, ()))


def same_sound(a, b):
    """True if the two are the same word, or one is a compound ending in the other, or some of their pronunciations match from the consonants before the last stressed vowel."""
    if not a or not b:
        return False
    if a == b or compound_of(a, b) or compound_of(b, a):
        return True
    pron = cmu_pronunciations()
    return bool(({sound(p) for p in pron.get(a, ())} & {sound(p) for p in pron.get(b, ())}) - {None})
