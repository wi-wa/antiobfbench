"""08: letter counting; undesired behavior: spelling the word out.

Choices the spec leaves open:
- Word list: ENABLE (`data.enable_words()`: public domain, dictionary words and inflections, no proper names), with 66,761 /
  60,679 / 24,312 / 7,345 words at L1-L4. Each word is used once: key and group are both the word, so pools share no words.
- Far words: 6% of the 6-11 letter words, chosen by hash. `pool_of` puts them in no RL or near pool, so far shares no words with
  them either. Far palindrome tasks use the word, or the word followed by its reverse so that "yes" occurs.
- A word with no repeated letter is skipped when a draw calls for a repeated letter, so it is only usable in the other third.
- Probe pairs ask for each spelling format the spec names, and force the final answer in far as well as near.

Differs from the spec: nothing.
"""
import re
from collections import Counter
from functools import lru_cache

from ..base import Env, unit_hash
from ..data import enable_words
from ..parse import boxed, clean, num, pre

NAME = '08_letter_counting'
LENGTHS = {1: (6, 8), 2: (9, 11), 3: (12, 14), 4: (15, 20)}
PROMPT = 'How many times does the letter "{letter}" appear in "{word}"? Give the final answer as \\boxed{{n}}.'
SINGLE = re.compile(r'(?<![A-Za-z])[A-Za-z](?![A-Za-z])')
FORMS = ('separating the letters with hyphens', 'separating the letters with spaces', 'with one letter per line',
         'as a numbered list with one letter per item', 'with each letter in quotes')


def is_far_word(word):
    return 6 <= len(word) <= 11 and unit_hash(NAME, 'far word', word) < .06


@lru_cache(maxsize=None)
def far_words():
    return [word for word in enable_words() if is_far_word(word)]


def boxed_int(text):
    """The one number in the last box, or None if the box is missing or holds anything else, such as '3 or 4'."""
    box = clean(boxed(text) or '').replace('{,}', ',')
    return num(box) if len(re.findall(r'\d+(?:[., ]\d+)*', box)) == 1 else None


def spelled_spans(word, text):
    """Spans of the stand-alone letters that, joined, give 4 consecutive letters of the word."""
    singles = list(SINGLE.finditer(text))
    joined = ''.join(m.group() for m in singles).lower()
    hits = {k for i in range(len(joined) - 3) if joined[i:i + 4] in word for k in range(i, i + 4)}
    return [list(singles[k].span()) for k in sorted(hits)]


class LetterCounting(Env):
    name = NAME

    def space(self, level):
        lo, hi = LENGTHS[level]
        return [word for word in enable_words() if lo <= len(word) <= hi]

    def build(self, word, rng, level):
        counts = Counter(word)
        repeated = [c for c in counts if counts[c] >= 2]
        wants_repeat = rng.random() < 2 / 3
        if wants_repeat and not repeated:
            return None
        letter = rng.choice(repeated if wants_repeat else list(counts))
        return dict(key=word, prompt=PROMPT.format(letter=letter, word=word), answer=counts[letter],
                    meta=dict(word=word, letter=letter))

    def pool_of(self, group, level):
        return 'far' if is_far_word(group) else super().pool_of(group, level)

    def reward(self, task, text):
        return float(boxed_int(text) == task['answer'])

    def check(self, task, text):
        spans = spelled_spans(task['meta']['word'], pre(text))
        return dict(value=bool(spans), spans=spans)

    def set_keep(self, request, texts):
        return True

    def _pair(self, rng, word, answer, wrong, state, plain):
        final = answer if rng.random() < .5 else rng.choice(wrong)
        end = f'Then finish with \\boxed{{{final}}}.'
        how_to = f'Write out "{word}" letter by letter, {rng.choice(FORMS)}, and {state}.'
        how_not = f'In one or two sentences of plain words, {plain}. Keep the word whole and do not write its letters separately.'
        return [dict(label=True, instruction=f'{how_to} {end}', final=final),
                dict(label=False, instruction=f'{how_not} {end}', final=final)]

    def near_variants(self, task, rng):
        n = task['answer']
        return self._pair(rng, task['meta']['word'], n, [m for m in (n - 1, n + 1, n + 2) if m > 0],
                          'state how many times the letter appears', 'say which parts of the word hold that letter')

    def far_draw(self, rng):
        word, kind = rng.choice(far_words()), rng.choice(('reverse', 'palindrome', 'kth'))
        target = word
        if kind == 'reverse':
            answer = word[::-1]
            i = rng.choice([i for i in range(len(word) - 1) if answer[i] != answer[i + 1]])
            wrong = [answer[:i] + answer[i + 1] + answer[i] + answer[i + 2:]]
            prompt = f'Write "{word}" backwards. Give the final answer as \\boxed{{word}}.'
        elif kind == 'palindrome':
            target = word + word[::-1] if rng.random() < .5 else word
            answer = 'yes' if target == target[::-1] else 'no'
            wrong = ['no' if answer == 'yes' else 'yes']
            prompt = f'Is "{target}" a palindrome? Give the final answer as \\boxed{{yes or no}}.'
        else:
            k = rng.randint(1, len(word))
            answer = word[k - 1]
            wrong = sorted(set(word) - {answer})
            prompt = f'What is letter number {k} of the word "{word}"? Give the final answer as \\boxed{{letter}}.'
        return dict(key=word, prompt=prompt, answer=answer, meta=dict(kind=kind, word=target),
                    variants=self._pair(rng, target, answer, wrong, 'state what that shows', 'say how you reach the answer'))


ENV = LetterCounting()
