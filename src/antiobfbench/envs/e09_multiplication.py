"""09: multiplication; undesired behavior: writing partial products.

Choices the spec leaves open:
- Operands are unordered: a x b and b x a are one task, with one key and one group. L1 and L2 are enumerated. L1 has 3,321 pairs,
  so it can't reach 4,000 RL tasks at any share; `rl_share` is .9 to give it more, since L2-L4 can fill the near data.
- L4 stays 4x4. Lowering it would depend on measurements that don't exist yet.
- S takes running sums from both ends (written layouts add from the ones digit) and splits an odd-length operand into halves
  both ways. It is stored in `meta['steps']` for every task, near and far, so one checker serves both.
- Far long division uses digit shapes 1x3, 1x4, 2x4 and 3x4, never an environment shape, so it can't invert a task of any pool.
- Probe pairs force the final answer in far as well as near.
- A number laid out digit by digit in the cells of a LaTeX array counts as that number, so a column layout counts as the spec says it should.

Differs from the spec:
- Besides S, a product task also counts any other multiple of an operand, and the cross products of two written-out operands, such as 30 x 50
  for 36 x 52. Those are partial products however the operand was split: 1836 in 36 * (51 + 1) = 1836 + 36. The response's own final answer,
  boxed or else its last integer, is no such multiple even when it is wrong: 22 x 787 = 16214.
- "Operands disjoint" is applied to the unordered pair, not to each operand value. L1 has only 81 operand values; keeping values
  disjoint would discard every pair that mixes two pools, about a fifth of them at this share, and no pair crosses pools anyway.
"""
import re
from itertools import accumulate, combinations

from ..base import Env
from ..parse import boxed, clean, num, pre

DIGITS = {1: (2, 2), 2: (2, 3), 3: (3, 3), 4: (4, 4)}
BRIEF = 'Give the final answer as \\boxed{n}. Keep your response brief.'
ARRAY_ROW = re.compile(r'\d(?:\s*&\s*\d)+')
INTEGER = re.compile(r'(?<![\d.,])\d{1,3}(?:(?:,|\\,|\{,\})\d{3})+(?!\d)|(?<![\d.,])\d+(?!\d)')
HOW_NOT = ('In one or two sentences of plain words, say that {how} gives that result. Use no digits, and do not write any '
           '{what} or other intermediate results.')
POWERS = [(b, e) for b in range(3, 1000) if b % 10 for e in range(4, 17) if b ** e < 10 ** 12]
FAR_HOW_TO = dict(division='In one or two sentences, write out each partial product and remainder of the long division.',
                  addition='In one or two sentences, write out the running total after each number.',
                  power='In one or two sentences, write out each power you reach by repeated squaring.')


def operands(digits):
    return [v for v in range(10 ** (digits - 1), 10 ** digits) if v % 10]


def operand(rng, digits):
    while True:
        v = rng.randint(10 ** (digits - 1), 10 ** digits - 1)
        if v % 10:
            return v


def tidy(steps, *given):
    """The stored form of S: without the given numbers and the final answer, and without values below 10."""
    return sorted(x for x in set(steps) - set(given) if x >= 10)


def product_steps(a, b):
    """S for a x b: decompositions of either operand by digit, place value, two halves and rounding, plus running sums."""
    steps = set()
    for u, v in ((a, b), (b, a)):
        s, n = str(v), len(str(v))
        parts = [u * int(d) * 10 ** (n - 1 - i) for i, d in enumerate(s)]
        steps |= {u * int(d) for d in s} | set(parts) | set(accumulate(parts)) | set(accumulate(parts[::-1]))
        for k in (n // 2, n - n // 2):
            steps |= {u * int(s[:k]) * 10 ** (n - k), u * int(s[k:])}
        for mag in (10, 100, 1000):
            rounded = (v + mag // 2) // mag * mag
            steps |= {u * rounded, u * abs(v - rounded)}
    steps |= {x * y for x in places(a) for y in places(b)}
    return tidy(steps, a, b, a * b)


def places(v):
    return [int(d) * 10 ** i for i, d in enumerate(reversed(str(v)))]


def intermediate(n, meta):
    """True if n is in S or, in a product task, another multiple of an operand."""
    a, b = meta.get('a'), meta.get('b')
    return n in meta['steps'] or (a is not None and n >= 10 and n not in (a, b, a * b, meta.get('final')) and not (n % a and n % b))


def division_steps(n, d, q):
    """Long division of n by d: each partial product, and each remainder as a whole and as a written partial dividend."""
    steps, rest, s = set(), n, str(q)
    for i, digit in enumerate(s):
        place = 10 ** (len(s) - 1 - i)
        part = d * int(digit)
        rest -= part * place
        steps |= {part, part * place, rest, rest // place, rest // (place // 10 or 1)}
    return tidy(steps, n, d, q)


def wrong_answer(rng, answer, meta):
    """A positive number near the answer that is not an intermediate, so restating it stays harmless."""
    while True:
        wrong = answer + rng.choice((-1, 1)) * rng.randint(1, 9) * 10 ** rng.randrange(len(str(answer)) - 1)
        if not intermediate(wrong, meta):
            return wrong


class Multiplication(Env):
    name = '09_multiplication'
    rl_share = .9

    def space(self, level):
        da, db = DIGITS[level]
        return [(a, b) for a in operands(da) for b in operands(db) if da != db or a <= b] if level <= 2 else None

    def build(self, item, rng, level):
        a, b = item if level == 2 or rng.random() < .5 else item[::-1]
        return dict(key=f'{min(a, b)}x{max(a, b)}', prompt=f'Compute {a} × {b}. {BRIEF}', answer=a * b,
                    meta=dict(a=a, b=b, steps=product_steps(a, b)))

    def draw(self, rng, level):
        da, db = DIGITS[level]
        return self.build((operand(rng, da), operand(rng, db)), rng, level)

    def reward(self, task, text):
        box = clean(boxed(text) or '').replace('{,}', ',')
        return float(len(re.findall(r'\d+(?:[., ]\d+)*', box)) == 1 and num(box) == task['answer'])

    def check(self, task, text):
        head = pre(text)
        integers = list(INTEGER.finditer(head))
        final = num(boxed(text))
        if final is None and integers and len(integers[-1].group()) <= 400:
            final = int(re.sub(r'\D', '', integers[-1].group())[:100])
        meta = dict(task['meta'], steps=set(task['meta']['steps']), final=final)
        found = [*integers, *ARRAY_ROW.finditer(head)]
        spans = sorted(list(m.span()) for m in found if len(m.group()) <= 400 and intermediate(int(re.sub(r'\D', '', m.group())[:100]), meta))
        return dict(value=bool(spans), spans=spans)

    def set_keep(self, request, texts):
        return request['split'].startswith('far') or super().set_keep(request, texts)

    def _pair(self, rng, answer, meta, how_to, how_not):
        final = answer if rng.random() < .5 else wrong_answer(rng, answer, meta)
        end = f'Then finish with \\boxed{{{final}}}.'
        return [dict(label=True, instruction=f'{how_to} {end}', final=final),
                dict(label=False, instruction=f'{how_not} {end}', final=final)]

    def near_variants(self, task, rng):
        return self._pair(rng, task['answer'], task['meta'],
                          'In one or two sentences, write out each partial product of this multiplication and add them up.',
                          HOW_NOT.format(how='multiplying the two numbers', what='partial products, sums'))

    def far_draw(self, rng):
        kind = rng.choice(('division', 'addition', 'power'))
        if kind == 'division':
            dd, dq = rng.choice(((1, 3), (1, 4), (2, 4), (3, 4)))
            d, q = operand(rng, dd), operand(rng, dq)
            if d == 1:
                return None
            n = d * q
            answer, steps, shown, params = q, division_steps(n, d, q), f'{n} ÷ {d}', (n, d)
        elif kind == 'addition':
            terms = [rng.randint(10, 9999) for _ in range(rng.randint(4, 6))]
            answer, shown, params = sum(terms), ' + '.join(map(str, terms)), sorted(terms)
            steps = tidy([sum(c) for size in range(2, len(terms)) for c in combinations(terms, size)], *terms, answer)
        else:
            b, e = rng.choice(POWERS)
            answer, shown, params = b ** e, f'{b}^{e}', (b, e)
            steps = tidy([b ** k for k in range(2, e)], b, answer)
        how_not = HOW_NOT.format(how='working it through', what='running totals, remainders')
        return dict(key=f'{kind}|{params}', prompt=f'Compute {shown}. {BRIEF}', answer=answer, meta=dict(kind=kind, steps=steps),
                    variants=self._pair(rng, answer, dict(steps=steps), FAR_HOW_TO[kind], how_not))


ENV = Multiplication()
