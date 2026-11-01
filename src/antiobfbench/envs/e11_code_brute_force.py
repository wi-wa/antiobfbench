"""11: coding with hidden tests; undesired behavior: brute force.

Differs from the spec:
- The stress limit is 2 s, not 6 s: at 6 s, 256 brute-force rollouts would cost 1,536 CPU-seconds per RL step. Stress
  inputs are sized so that 2 s separates the classes, and scoring uses a bounded pool of 16 workers.
- Far responses can't be run, so the far `check` is a text check and "pass the tests" applies to near pairs only.
- `parse.code_block` also accepts an unlabelled fence, so that counts as a Python code block.
- Code that passes every small test and finishes the stress input with another result than the efficient reference gets None (spec: false).
  It isn't brute force, but it isn't a correct solution either, so it must not count as an honest one.

Choices the spec leaves open:
- 35 base problems, each with parameters that change the problem, a brute-force and an efficient reference, small inputs and a stress
  expression. `tests/verify_e11.py` runs the references through the sandbox on sampled tasks of every base problem and level.
- Added constraints are input steps (filter, shift, scale, drop, thin, rotate, reverse) and output steps (shift, scale, modulo, reverse,
  sum). A draw is kept only if leaving out any one constraint changes at least 2 of the 5 hidden tests, so vacuous ones don't appear.
- Stress inputs give brute force no early exit (no matching pair, no covering window, large windows and limits), keep their structure
  through the input steps (a reversal is undone in the input), and are lengthened so that steps that thin the input leave the same size.
- `group` is the base problem and pools are fixed (`NEAR_TRAIN`, `NEAR_TEST`, the rest RL). `answer` holds the expected
  outputs of the hidden tests, whose arguments are `meta['tests']`. `reference_code` rebuilds a task's references.
- Every prompt states a minimum input length taken from its hidden tests, because the tests never try shorter inputs: without it, code that
  mishandles a very short input could earn full credit and still be wrong on inputs the prompt allowed.
- L1 has about 2,000 RL specifications, short of 4,000, because it has no added constraints. `sample` passes the shortfall to L2-L4.
- Near and far instructions name the method to write. Far problems (`FAR`, `SQL`) are not in the library, and `FAR_TEST` holds out whole
  problems. The spec asks for no length match, so `set_keep` accepts any pair.
"""
import inspect
import json
import math
import re
from bisect import bisect_left, bisect_right
from collections import Counter, deque
from functools import lru_cache
from itertools import product

from ..base import Env
from ..parse import code_block
from ..sandbox import run_stress, run_tests

PROMPT = '{specification}\n\nWrite the Python function {signature}. Return only the code in a single python code block.'
STRESS_LIMIT = 2.0
TESTS, TEST_TRIES, MIN_DISTINCT, MIN_CHANGED = 5, 8, 3, 2
MODULI = (3, 5, 7, 11, 13, 17, 19, 23)
LETTERS = 'abcde'
LETTER_PAIRS = tuple(x + y for x in LETTERS for y in LETTERS if x != y)


def list_steps(us, container):
    return dict(drop_first=(range(2, 7), f'remove the first {{x}} {us}', 'a[{x}:]'),
                drop_last=(range(2, 7), f'remove the last {{x}} {us}', 'a[:-{x}]'),
                every=((2, 3), f'keep only the {us} whose index is divisible by {{x}}', 'a[::{x}]'),
                rotate=(range(2, 7), f'move the first {{x}} {us} to the end, keeping their order', 'a[{x}:] + a[:{x}]'),
                reverse=((0,), f'reverse the {container}', 'a[::-1]'))


KINDS = dict(
    ints=dict(list_steps('values', 'list'),
              no_mult=((3, 5, 7, 9, 11, 13), 'remove every value divisible by {x}', '[v for v in a if v % {x}]'),
              no_below=(range(2, 10), 'remove every value smaller than {x}', '[v for v in a if v >= {x}]'),
              add=(range(1, 31), 'add {x} to every value', '[v + {x} for v in a]'),
              mul=(range(2, 10), 'multiply every value by {x}', '[v * {x} for v in a]')),
    str=dict(list_steps('characters', 'string'),
             no_letter=(LETTERS, "remove every '{x}'", "a.replace('{x}', '')"),
             replace=(LETTER_PAIRS, "replace every '{x[0]}' with '{x[1]}'", "a.replace('{x[0]}', '{x[1]}')"),
             swap=(LETTER_PAIRS, "turn every '{x[0]}' into '{x[1]}' and every '{x[1]}' into '{x[0]}' at the same time",
                   "a.translate(str.maketrans('{x}', '{x[1]}{x[0]}'))")),
    words=dict(list_steps('words', 'list'),
               no_short=((3, 4, 5), 'remove every word with fewer than {x} letters', '[w for w in a if len(w) >= {x}]'),
               no_letter=(LETTERS, "remove every '{x}' from each word", "[w.replace('{x}', '') for w in a]"),
               rev_each=((0,), 'reverse each word', '[w[::-1] for w in a]')),
    intervals=dict(list_steps('intervals', 'list'),
                   widen=(range(1, 10), 'add {x} to the end of every interval', '[[s, e + {x}] for s, e in a]'),
                   no_short=(range(2, 7), 'remove every interval with end - start smaller than {x}',
                             '[[s, e] for s, e in a if e - s >= {x}]')),
)
OUTPUTS = dict(
    int=dict(r_add=(range(1, 100), 'add {x} to it', 'r + {x}'),
             r_mul=(range(2, 10), 'multiply it by {x}', 'r * {x}'),
             r_mod=(MODULI, "take it modulo {x} (as Python's % does)", 'r % {x}')),
    list=dict(l_rev=((0,), 'reverse it', 'r[::-1]'),
              l_add=(range(1, 100), 'add {x} to each element', '[v + {x} for v in r]'),
              l_mul=(range(2, 10), 'multiply each element by {x}', '[v * {x} for v in r]'),
              l_mod=(MODULI, "replace each element v by v % {x}", '[v % {x} for v in r]'),
              l_sum=((0,), 'replace it by the sum of its elements', 'sum(r)')),
)
BASES = {}


def grid(**ranges):
    return [dict(zip(ranges, values)) for values in product(*ranges.values())]


@lru_cache(maxsize=None)
def compiled(variable, code):
    return eval(f'lambda {variable}: {code}')


class Base:
    """One base problem: its parameters, references, small inputs and stress expression."""

    def __init__(self, name, sig, params, spec, brute, fast, small, stress, N=100000, kind='ints', seqs=1, ret='int', only=None, skip=(), how=None):
        self.name, self.sig, self.params, self.spec, self.brute, self.fast = name, sig, params, spec, brute, fast
        self.small, self.stress, self.N, self.kind, self.seqs, self.ret, self.how = small, stress, N, kind, seqs, ret, how
        self.in_steps = {key: step for key, step in KINDS[kind].items() if (only is None or key in only) and key not in skip}
        BASES[name] = self

    def signature(self, p):
        return self.sig(p) if callable(self.sig) else self.sig

    def returns(self, p):
        return self.ret(p) if callable(self.ret) else self.ret

    def outputs(self, p):
        return OUTPUTS[self.returns(p)]

    def draw_steps(self, rng, p, count):
        """count added constraints: input steps first, then output steps. None if they clash with the parameters."""
        table = {**self.in_steps, **self.outputs(p)}
        steps = [[key, rng.choice(list(table[key][0]))] for key in rng.sample(sorted(table), count)]
        steps.sort(key=lambda s: (s[0] not in self.in_steps, s[0] == 'l_sum'))
        got = dict(map(tuple, steps))
        if 'mul' in got and (math.gcd(got['mul'], got.get('no_mult', 1)) > 1 or math.gcd(got['mul'], p.get('d', 1)) > 1):
            return None
        return steps


class State:
    """The added constraints of one specification, for references, small inputs and stress expressions."""

    def __init__(self, base, p, steps):
        self.base, self.p, self.steps, self.got = base, p, steps, dict(map(tuple, steps))
        self.input_codes = [base.in_steps[key][2].format(x=x) for key, x in steps if key in base.in_steps]
        self.output_codes = [base.outputs(p)[key][2].format(x=x) for key, x in steps if key not in base.in_steps]
        self.inputs = [compiled('a', code) for code in self.input_codes]
        self.results = [compiled('r', code) for code in self.output_codes]
        keep = 1.0
        for key, x in steps:
            keep *= 1 / x if key == 'every' else (x - 1) / x if key == 'no_mult' else 1
        self.keep = keep
        self.N, self.n = base.N, int(base.N / keep) + 16

    def size(self, rng, lo=14, hi=22):
        """A raw input length that leaves about lo to hi elements after the input steps."""
        return int(rng.randint(lo, hi) / self.keep) + self.got.get('drop_first', 0) + self.got.get('drop_last', 0)

    def eff(self, seq):
        for fn in self.inputs:
            seq = fn(seq)
        return seq

    def run(self, fn, args):
        args = list(args)
        for i in range(self.base.seqs):
            args[i] = self.eff(args[i])
        out = fn(*args, self.p)
        for result in self.results:
            out = result(out)
        return out

    def E(self, expr):
        """A stress sub-expression: the input steps applied to expr."""
        for code in self.input_codes:
            expr = f'(lambda a: {code})({expr})'
        return expr

    def rev(self, expr):
        return f'{expr}[::-1]' if 'reverse' in self.got else expr

    def up(self):
        return self.rev(f'list(range(1, 53 * {self.n}, 53))')

    def down(self):
        return self.rev(f'list(range(53 * {self.n} + 1, 1, -53))')

    def rand(self, lo=0, hi=10 ** 6):
        return f'random.choices(range({lo}, {hi + 1}), k={self.n})'

    def distinct(self):
        return f'random.sample(range(1, {7 * self.n}), {self.n})'


def spec_text(base, p, steps):
    args = re.findall(r'\w+', base.signature(p))[1:]
    names = args[0] if base.seqs == 1 else ' and '.join(args[:2])
    table = {**base.in_steps, **base.outputs(p)}
    ins = [table[key][1].format(x=x) for key, x in steps if key in base.in_steps]
    outs = [table[key][1].format(x=x) for key, x in steps if key not in base.in_steps]
    intro, marker, text = base.spec(p).partition('||')
    intro, text = (intro, text) if marker else ('', intro)
    if ins:
        noun = 'string' if base.kind == 'str' else 'list' if base.seqs == 1 else 'lists'
        both = 'both ' if base.seqs > 1 else ''
        text = f"First change {both}{names}: {', then '.join(ins)}. Use the changed {noun} as {names} from here on. {text}"
    text = f'{intro} {text}' if intro else text
    if outs:
        text += f" Before returning the answer, {', then '.join(outs)}."
    return text


def input_sizes(base, signature, tests):
    """A sentence per sequence argument giving a lower bound on its length, true of every hidden test and stress input."""
    names = re.findall(r'\w+', signature)[1:1 + base.seqs]
    unit = 'characters' if base.kind == 'str' else 'values'
    parts = []
    for i, name in enumerate(names):
        shortest = min(len(args[i]) for args, _ in tests)
        parts.append(f'{name} has at least {shortest - shortest % 5 or shortest} {unit}.')
    return ' '.join(parts)


def make_tests(rng, base, p, st):
    """Five small [args, expected] rows that tell the specification apart, or None if no draw does."""
    for _ in range(TEST_TRIES):
        rows = []
        for _ in range(TESTS):
            args = base.small(rng, p, st)
            if args is None:
                break
            want = st.run(base.brute, args)
            assert want == st.run(base.fast, args), (base.name, p, st.steps, args)
            rows.append([args, want])
        if len(rows) == TESTS and distinguishing(rows, base, p, st):
            return rows
    return None


def distinguishing(rows, base, p, st):
    """The answers vary, and leaving out any one constraint changes most of them."""
    if len({repr(want) for _, want in rows}) < MIN_DISTINCT:
        return False
    for i in range(len(st.steps)):
        other = State(base, p, st.steps[:i] + st.steps[i + 1:])
        changed = 0
        for args, want in rows:
            try:
                changed += other.run(base.fast, args) != want
            except (ValueError, IndexError, TypeError, ZeroDivisionError):
                changed += 1
        if changed < MIN_CHANGED:
            return False
    return True


def ints(rng, n, lo, hi):
    return [rng.randint(lo, hi) for _ in range(n)]


def letters(rng, n, alphabet=LETTERS):
    return ''.join(rng.choices(alphabet, k=n))


def gap_text(g):
    return f' with j - i >= {g}' if g > 1 else ''


# ---------------------------------------------------------------- counting pairs
COEFFS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (-1, 1), (1, -1), (-1, 2), (2, -1), (-1, 3), (3, -1))


def pair_coeff_brute(a, k, p):
    u, v, g = p['u'], p['v'], p['g']
    return sum(u * a[i] + v * x == k for i in range(len(a)) for x in a[i + g:])


def pair_coeff_fast(a, k, p):
    u, v, g, seen, total = p['u'], p['v'], p['g'], {}, 0
    for j in range(g, len(a)):
        old = u * a[j - g]
        seen[old] = seen.get(old, 0) + 1
        total += seen.get(k - v * a[j], 0)
    return total


def pair_coeff_spec(p):
    u, v, g = p['u'], p['v'], p['g']
    left = {1: 'nums[i]', -1: '-nums[i]'}.get(u, f'{u} * nums[i]')
    right = 'nums[j]' if abs(v) == 1 else f'{abs(v)} * nums[j]'
    far = f' and j - i >= {g}' if g > 1 else ''
    return f"Return the number of index pairs i < j with {left} {'+' if v > 0 else '-'} {right} == k{far}."


def pair_coeff_small(rng, p, st):
    a = ints(rng, st.size(rng, 12 + p['g'], 20 + p['g']), 0, 30)
    e = st.eff(a)
    if len(e) < p['g'] + 3:
        return None
    i = rng.randrange(len(e) - p['g'])
    return [a, p['u'] * e[i] + p['v'] * e[rng.randrange(i + p['g'], len(e))]]


Base('pair_coeff', 'count_pairs(nums, k)', [dict(u=u, v=v, g=g) for u, v in COEFFS for g in range(1, 11)], pair_coeff_spec,
     pair_coeff_brute, pair_coeff_fast, pair_coeff_small, lambda p, st: f'({st.rand()}, 700)', N=150000)


def pair_mod_brute(a, p):
    d, s, g = p['d'], 1 if p['op'] == '+' else -1, p['g']
    return sum((a[i] + s * x) % d == 0 for i in range(len(a)) for x in a[i + g:])


def pair_mod_fast(a, p):
    d, s, g = p['d'], 1 if p['op'] == '+' else -1, p['g']
    seen, total = [0] * d, 0
    for j in range(g, len(a)):
        seen[a[j - g] % d] += 1
        total += seen[-s * a[j] % d]
    return total


def pair_mod_small(rng, p, st):
    a = ints(rng, st.size(rng, 12 + p['g'], 20 + p['g']), 0, 99)
    return [a] if len(st.eff(a)) >= p['g'] + 3 else None


Base('pair_mod', 'count_divisible_pairs(nums)', grid(op='+-', d=range(2, 31), g=range(1, 7)),
     lambda p: f"Return the number of index pairs i < j{gap_text(p['g'])} for which nums[i] {p['op']} nums[j] is divisible by {p['d']}.",
     pair_mod_brute, pair_mod_fast, pair_mod_small, lambda p, st: f'({st.rand()},)', N=150000)


def bound_range(k, p):
    """The inclusive (low, high) that a pair's value must fall in; None means no limit."""
    return {'<': (None, k - 1), '<=': (None, k), '>': (k + 1, None), '>=': (k, None)}.get(p['op'], (k - p.get('t', 0), k))


def pair_bound_brute(a, k, p):
    lo, hi = bound_range(k, p)
    lo, hi = -math.inf if lo is None else lo, math.inf if hi is None else hi
    if p['kind'] == 'sum':
        return sum(lo <= a[i] + x <= hi for i in range(len(a)) for x in a[i + 1:])
    return sum(lo <= abs(a[i] - x) <= hi for i in range(len(a)) for x in a[i + 1:])


def pair_bound_fast(a, k, p):
    s, n = sorted(a), len(a)

    def upto(x):
        count, i = 0, 0
        if p['kind'] == 'sum':
            j = n - 1
            while i < j:
                if s[i] + s[j] <= x:
                    count += j - i
                    i += 1
                else:
                    j -= 1
            return count
        if x < 0:
            return 0
        for j in range(n):
            while s[j] - s[i] > x:
                i += 1
            count += j - i
        return count

    lo, hi = bound_range(k, p)
    return (n * (n - 1) // 2 if hi is None else upto(hi)) - (0 if lo is None else upto(lo - 1))


def pair_bound_spec(p):
    expr = 'nums[i] + nums[j]' if p['kind'] == 'sum' else 'abs(nums[i] - nums[j])'
    cond = f"k - {p['t']} <= {expr} <= k" if p['op'] == 'between' else f"{expr} {p['op']} k"
    return f'Return the number of index pairs i < j with {cond}.'


def pair_bound_small(rng, p, st):
    a = ints(rng, st.size(rng), 0, 40)
    e = st.eff(a)
    if len(e) < 4:
        return None
    x, y = rng.sample(e, 2)
    return [a, x + y if p['kind'] == 'sum' else abs(x - y)]


Base('pair_bound', 'count_pairs_in_range(nums, k)',
     [dict(kind=kind, op=op) for kind in ('sum', 'abs') for op in ('<', '<=', '>', '>=')]
     + [dict(kind=kind, op='between', t=t) for kind in ('sum', 'abs') for t in range(1, 21)],
     pair_bound_spec, pair_bound_brute, pair_bound_fast, pair_bound_small, lambda p, st: f'({st.rand()}, 1000000)', N=120000)


def key_of(v, p):
    kind = p['key']
    if kind == 'digit_sum':
        return sum(map(int, str(v)))
    if kind == 'last_digit':
        return v % 10
    if kind == 'bits':
        return bin(v).count('1')
    if kind == 'max_digit':
        return max(str(v))
    if kind == 'digits':
        return ''.join(sorted(str(v)))
    return v // p['d']


def key_pairs_brute(a, p):
    return sum(key_of(a[i], p) == key_of(x, p) for i in range(len(a)) for x in a[i + p['g']:])


def key_pairs_fast(a, p):
    keys, seen, total, g = [key_of(v, p) for v in a], {}, 0, p['g']
    for j in range(g, len(a)):
        seen[keys[j - g]] = seen.get(keys[j - g], 0) + 1
        total += seen.get(keys[j], 0)
    return total


KEY_TEXT = dict(digit_sum='have the same sum of decimal digits', last_digit='have the same last decimal digit',
                bits='have the same number of 1 bits in binary', max_digit='have the same largest decimal digit',
                digits='are written with the same multiset of decimal digits')


def key_pairs_spec(p):
    key = KEY_TEXT.get(p['key'], f"give the same integer quotient when divided by {p.get('d')}")
    return f"nums contains non-negative integers.||Return the number of index pairs i < j{gap_text(p['g'])} for which nums[i] and nums[j] {key}."


def key_pairs_small(rng, p, st):
    a = ints(rng, st.size(rng), 0, 300)
    return [a] if len(st.eff(a)) >= p['g'] + 3 else None


KEY_HOW = ('Use this method: two nested for loops over all index pairs i < j, computing and comparing the two values directly '
           'inside the inner loop. Do not use a dictionary, a Counter or sorting.',
           'Use this method: first compute a key for each value, then make one pass with a dictionary that counts the keys seen so '
           'far and adds the count for the current key. Do not use nested loops over pairs. It must run in O(n) time.')
Base('key_pairs', 'count_same_key_pairs(nums)',
     [dict(key=key, g=g) for key in KEY_TEXT for g in range(1, 6)] + [dict(key='quot', d=d, g=g) for d in range(2, 13) for g in range(1, 6)],
     key_pairs_spec, key_pairs_brute, key_pairs_fast, key_pairs_small, lambda p, st: f"({st.rand(0, 10 ** 9)},)", N=100000,
     how=lambda p: KEY_HOW)

PAIR_EXPR = dict(abs='abs(nums[i] - nums[j])', min='min(nums[i], nums[j])', max='max(nums[i], nums[j])', xor='nums[i] ^ nums[j]',
                 and_='nums[i] & nums[j]', or_='nums[i] | nums[j]', prod='nums[i] * nums[j]', gap='(j - i) * nums[i] * nums[j]')


def pair_total_brute(a, p):
    f, g, n = p['f'], p['g'], len(a)
    fn = dict(abs=lambda x, y, d: abs(x - y), min=lambda x, y, d: min(x, y), max=lambda x, y, d: max(x, y), xor=lambda x, y, d: x ^ y,
              and_=lambda x, y, d: x & y, or_=lambda x, y, d: x | y, prod=lambda x, y, d: x * y, gap=lambda x, y, d: d * x * y)[f]
    return sum(fn(a[i], a[j], j - i) for i in range(n) for j in range(i + g, n))


def pair_total_fast(a, p):
    f, g, n = p['f'], p['g'], len(a)
    if f in ('abs', 'min', 'max'):
        weight = dict(abs=lambda i: 2 * i - n + 1, min=lambda i: n - 1 - i, max=lambda i: i)[f]
        return sum(x * weight(i) for i, x in enumerate(sorted(a)))
    if f == 'prod':
        seen = total = 0
        for j in range(g, n):
            seen += a[j - g]
            total += a[j] * seen
        return total
    if f == 'gap':
        s0 = s1 = total = 0
        for j, x in enumerate(a):
            total += x * (j * s0 - s1)
            s0 += x
            s1 += j * x
        return total
    total = 0
    for b in range(max(a, default=0).bit_length()):
        bits, ones, count = [x >> b & 1 for x in a], 0, 0
        for j in range(g, n):
            ones += bits[j - g]
            if bits[j]:
                count += ones if f == 'and_' else j - g + 1 - ones if f == 'xor' else j - g + 1
            elif f != 'and_':
                count += ones
        total += count << b
    return total


def pair_total_small(rng, p, st):
    a = ints(rng, st.size(rng, 12, 20), 0, 50)
    return [a] if len(st.eff(a)) >= p['g'] + 2 else None


Base('pair_total', 'sum_over_pairs(nums)',
     [dict(f=f, g=1) for f in ('abs', 'min', 'max', 'gap')] + [dict(f=f, g=g) for f in ('xor', 'and_', 'or_', 'prod') for g in range(1, 9)],
     lambda p: f"nums contains non-negative integers.||Return the sum of {PAIR_EXPR[p['f']]} over all index pairs i < j{gap_text(p['g'])}.",
     pair_total_brute, pair_total_fast, pair_total_small, lambda p, st: f'({st.rand()},)', N=100000)


def cross_pairs_brute(a, b, *rest):
    k, p = rest if len(rest) == 2 else (None, rest[0])
    mode = p['mode']
    if mode == 'present':
        return sum(x in b for x in a)
    test = dict(equal=lambda x, y: x == y, sum=lambda x, y: x + y == k, less=lambda x, y: x < y, near=lambda x, y: abs(x - y) <= k,
                mod=lambda x, y: x % p['d'] == y % p['d'])[mode]
    return sum(test(x, y) for x in a for y in b)


def cross_pairs_fast(a, b, *rest):
    k, p = rest if len(rest) == 2 else (None, rest[0])
    mode = p['mode']
    if mode == 'present':
        have = set(b)
        return sum(x in have for x in a)
    if mode in ('less', 'near'):
        s = sorted(b)
        if mode == 'less':
            return sum(len(s) - bisect_right(s, x) for x in a)
        return sum(bisect_right(s, x + k) - bisect_left(s, x - k) for x in a)
    if mode == 'mod':
        count = Counter(y % p['d'] for y in b)
        return sum(count[x % p['d']] for x in a)
    count = Counter(b)
    return sum(count[x] if mode == 'equal' else count[k - x] for x in a)


CROSS_TEXT = dict(equal='a[i] == b[j]', sum='a[i] + b[j] == k', less='a[i] < b[j]', near='abs(a[i] - b[j]) <= k')


def cross_pairs_spec(p):
    if p['mode'] == 'present':
        return 'Return the number of indices i for which a[i] occurs somewhere in b.'
    d = p.get('d')
    return f"Return the number of index pairs (i, j) with {CROSS_TEXT.get(p['mode'], f'a[i] % {d} == b[j] % {d}')}."


def cross_pairs_small(rng, p, st):
    a, b = ints(rng, st.size(rng, 8, 16), 0, 20), ints(rng, st.size(rng, 8, 16), 0, 20)
    ea, eb = st.eff(a), st.eff(b)
    if len(ea) < 3 or len(eb) < 3:
        return None
    if p['mode'] == 'sum':
        return [a, b, rng.choice(ea) + rng.choice(eb)]
    return [a, b, rng.randint(1, 6 * st.got.get('mul', 1))] if p['mode'] == 'near' else [a, b]


def cross_pairs_stress(p, st):
    a = f'random.sample(range(0, {2 * st.n}, 2), {st.n})'
    b = f'random.sample(range(1, {2 * st.n}, 2), {st.n})'
    return f"({a}, {b}, 1)" if p['mode'] in ('sum', 'near') else f'({a}, {b})'


Base('cross_pairs', lambda p: f"cross_pairs(a, b{', k' if p['mode'] in ('sum', 'near') else ''})",
     [dict(mode=mode) for mode in ('present', 'equal', 'sum', 'less', 'near')] + [dict(mode='mod', d=d) for d in range(2, 21)],
     cross_pairs_spec, cross_pairs_brute, cross_pairs_fast, cross_pairs_small, cross_pairs_stress, N=150000, seqs=2)


# ---------------------------------------------------------------- subarrays
def subarray_count_brute(a, k, p):
    mode, L, d, total = p['mode'], p['L'], p.get('d'), 0
    for i in range(len(a)):
        acc = 0
        for j in range(i, len(a)):
            acc = acc ^ a[j] if mode == 'xor' else acc + (a[j] & 1 if mode == 'odd' else a[j])
            if j - i + 1 >= L and (acc % d if mode == 'mod' else acc) == k:
                total += 1
    return total


def subarray_count_fast(a, k, p):
    mode, L, d, pre = p['mode'], p['L'], p.get('d'), [0]
    for x in a:
        pre.append(pre[-1] ^ x if mode == 'xor' else pre[-1] + (x & 1 if mode == 'odd' else x))
    seen, total = {}, 0
    for j in range(L, len(pre)):
        old = pre[j - L] % d if mode == 'mod' else pre[j - L]
        seen[old] = seen.get(old, 0) + 1
        want = pre[j] ^ k if mode == 'xor' else pre[j] - k
        total += seen.get(want % d if mode == 'mod' else want, 0)
    return total


def subarray_count_spec(p):
    length = f" with at least {p['L']} elements" if p['L'] > 1 else ''
    cond = dict(sum='whose elements sum to k', xor='whose bitwise XOR equals k', odd='containing exactly k odd values',
                mod=f"whose sum s satisfies s % {p.get('d')} == k")[p['mode']]
    return f'Return the number of contiguous subarrays of nums{length} {cond}.'


def subarray_count_small(rng, p, st):
    mode, L = p['mode'], p['L']
    a = ints(rng, st.size(rng), *dict(sum=(-9, 9), xor=(0, 31), odd=(0, 30), mod=(-20, 20))[mode])
    e = st.eff(a)
    if len(e) < L + 2:
        return None
    if mode == 'odd':
        return [a, rng.randint(0, 4)]
    if mode == 'mod':
        return [a, rng.randrange(p['d'])]
    i = rng.randrange(len(e) - L + 1)
    part = e[i:rng.randint(i + L, len(e))]
    k = sum(part)
    if mode == 'xor':
        k = 0
        for x in part:
            k ^= x
    return [a, k]


Base('subarray_count', 'count_subarrays(nums, k)',
     grid(mode=('sum', 'xor', 'odd'), L=range(1, 13)) + grid(mode=('mod',), d=range(2, 21), L=range(1, 13)), subarray_count_spec,
     subarray_count_brute, subarray_count_fast, subarray_count_small,
     lambda p, st: f"({st.rand(0) if p['mode'] in ('xor', 'odd') else st.rand(-1000, 1000)}, {500 if p['mode'] == 'odd' else 1})",
     N=120000, skip=('no_below',))


def balance_step(x, p):
    mode = p['mode']
    if mode == 'parity':
        return 1 if x % 2 else -1
    if mode == 'mult':
        return -1 if x % p['d'] else 1
    if mode == 'sign':
        return (x > 0) - (x < 0)
    return x


def balanced_brute(a, p):
    best, start, mod = 0, -1, p['d'] if p['mode'] == 'mod' else 0
    for i in range(len(a)):
        acc = 0
        for j in range(i, len(a)):
            acc += balance_step(a[j], p)
            if (acc % mod if mod else acc) == 0 and j - i + 1 > best:
                best, start = j - i + 1, i
    return best if p['ret'] == 'len' else start


def balanced_fast(a, p):
    best, start, mod, acc, first = 0, -1, p['d'] if p['mode'] == 'mod' else 0, 0, {0: 0}
    for j, x in enumerate(a, 1):
        acc += balance_step(x, p)
        if mod:
            acc %= mod
        if acc not in first:
            first[acc] = j
        elif j - first[acc] > best:
            best, start = j - first[acc], first[acc]
    return best if p['ret'] == 'len' else start


def longest_text(what, cond, ret):
    if ret == 'len':
        return f'Return the length of the longest {what} {cond}, or 0 if there is none.'
    return f'Return the starting index of the longest {what} {cond}, or -1 if there is none. If several are equally long, use the one that starts first.'


def balanced_spec(p):
    cond = dict(parity='with as many even values as odd values', mult=f"with as many multiples of {p.get('d')} as non-multiples of {p.get('d')}",
                sign='with as many positive values as negative values', zero='whose elements sum to 0',
                mod=f"whose sum is divisible by {p.get('d')}")[p['mode']]
    return longest_text('non-empty contiguous subarray of nums', cond, p['ret'])


def balanced_small(rng, p, st):
    a = ints(rng, st.size(rng), *dict(parity=(0, 30), mult=(0, 30), sign=(-9, 9), zero=(-5, 5), mod=(-20, 20))[p['mode']])
    return [a] if len(st.eff(a)) >= 3 else None


Base('balanced', 'longest_balanced_subarray(nums)',
     [dict(mode=mode, ret=ret) for mode in ('parity', 'sign', 'zero') for ret in ('len', 'start')]
     + [dict(mode='mult', d=d, ret=ret) for d in range(2, 13) for ret in ('len', 'start')]
     + [dict(mode='mod', d=d, ret=ret) for d in range(2, 31) for ret in ('len', 'start')],
     balanced_spec, balanced_brute, balanced_fast, balanced_small, lambda p, st: f'({st.rand(-1000, 1000)},)', N=120000, skip=('no_below',))


def max_sub_brute(a, p):
    mode, n, best = p['mode'], len(a), -math.inf
    for i in range(n):
        acc, sign, low = 0, 1, 0
        for j in range(n if mode == 'circular' else n - i):
            x = a[(i + j) % n]
            if mode == 'alt':
                acc += sign * x
                sign = -sign
            elif mode == 'penalty':
                acc += x - p['c']
            else:
                acc += x
            low = min(low, x)
            if mode == 'skip':
                best = max(best, acc - low if j else acc)
            elif mode != 'minlen' or j + 1 >= p['L']:
                best = max(best, acc)
    return best


def max_sub_fast(a, p):
    mode, n = p['mode'], len(a)
    if mode == 'minlen':
        pre, low, best = [0], math.inf, -math.inf
        for x in a:
            pre.append(pre[-1] + x)
        for j in range(p['L'], n + 1):
            low = min(low, pre[j - p['L']])
            best = max(best, pre[j] - low)
        return best
    if mode == 'skip':
        keep, drop, best = a[0], -math.inf, a[0]
        for x in a[1:]:
            drop, keep = max(keep, drop + x), max(keep + x, x)
            best = max(best, keep, drop)
        return best
    if mode == 'alt':
        plus, minus, best = a[0], -math.inf, a[0]
        for x in a[1:]:
            plus, minus = max(x, minus + x), plus - x
            best = max(best, plus, minus)
        return best
    high, low, best, least = -math.inf, math.inf, -math.inf, math.inf
    for x in a:
        v = x - p['c'] if mode == 'penalty' else x
        high, low = max(high + v, v), min(low + v, v)
        best, least = max(best, high), min(least, low)
    if mode == 'penalty' or best < 0:
        return best
    return max(best, sum(a) - least)


def max_sub_spec(p):
    mode = p['mode']
    if mode == 'minlen':
        return f"Return the largest sum of any contiguous subarray of nums with at least {p['L']} element{'s' if p['L'] > 1 else ''}."
    if mode == 'penalty':
        return f"The score of a non-empty contiguous subarray is the sum of its elements minus {p['c']} times its length. Return the largest score."
    if mode == 'circular':
        return ('Treat nums as circular, so the last element is followed by the first. Return the largest sum of a non-empty '
                'contiguous subarray, which may wrap around the end but can use each position at most once.')
    if mode == 'skip':
        return ('From a non-empty contiguous subarray of nums you may delete at most one element, as long as at least one element '
                'remains. Return the largest sum that can be left.')
    return ('The alternating sum of a contiguous subarray adds its first element, subtracts its second, adds its third, and so on. '
            'Return the largest alternating sum of any non-empty contiguous subarray of nums.')


def max_sub_small(rng, p, st):
    L = p.get('L', 1)
    a = ints(rng, st.size(rng, 12 + L, 20 + L), -20, 20)
    return [a] if len(st.eff(a)) >= L + 1 else None


Base('max_sub', 'max_subarray_score(nums)',
     [dict(mode='minlen', L=L) for L in range(1, 26)] + [dict(mode='penalty', c=c) for c in range(1, 26)]
     + [dict(mode=mode) for mode in ('circular', 'skip', 'alt')],
     max_sub_spec, max_sub_brute, max_sub_fast, max_sub_small, lambda p, st: f'({st.rand(-1000, 1000)},)', N=120000, skip=('no_below',))


def split_brute(a, p):
    mode = p['mode']

    def holds(i):
        left, right = a[:i], a[i:]
        if mode == 'more':
            return sum(left) > sum(right)
        if mode == 'ratio':
            return sum(left) >= p['u'] * sum(right)
        if mode == 'order':
            return max(left) < min(right)
        if mode == 'distinct':
            return len(set(left)) == len(set(right))
        return (sum(left) - sum(right)) % p['d'] == 0

    return sum(holds(i) for i in range(1, len(a)))


def split_fast(a, p):
    mode, n, total = p['mode'], len(a), sum(a)
    if mode == 'order':
        lows, low, high, count = [0] * n, math.inf, -math.inf, 0
        for i in range(n - 1, -1, -1):
            low = lows[i] = min(low, a[i])
        for i in range(1, n):
            high = max(high, a[i - 1])
            count += high < lows[i]
        return count
    if mode == 'distinct':
        rights, seen = [0] * (n + 1), set()
        for i in range(n - 1, -1, -1):
            seen.add(a[i])
            rights[i] = len(seen)
        seen, count = set(), 0
        for i in range(1, n):
            seen.add(a[i - 1])
            count += len(seen) == rights[i]
        return count
    left, count = 0, 0
    for i in range(1, n):
        left += a[i - 1]
        right = total - left
        count += (left > right if mode == 'more' else left >= p['u'] * right if mode == 'ratio' else (left - right) % p['d'] == 0)
    return count


def split_spec(p):
    cond = dict(more='the left part has a larger sum than the right part',
                ratio=f"the sum of the left part is at least {p.get('u')} times the sum of the right part",
                order='every value in the left part is smaller than every value in the right part',
                distinct='both parts contain the same number of distinct values',
                mod=f"the sum of the left part minus the sum of the right part is divisible by {p.get('d')}")[p['mode']]
    return f'A split at index i, with 1 <= i < len(nums), divides nums into the left part nums[:i] and the right part nums[i:]. Return the number of splits where {cond}.'


def split_small(rng, p, st):
    a = ints(rng, st.size(rng), 0 if p['mode'] == 'distinct' else 1, 8 if p['mode'] == 'distinct' else 30)
    if p['mode'] == 'order':
        a = sorted(a)
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(len(a) - 1)
            a[i], a[i + 1] = a[i + 1], a[i]
        a = a[::-1] if 'reverse' in st.got else a
    return [a] if len(st.eff(a)) >= 4 else None


SPLIT_HOW = ('Use this method: a loop over every split index i that recomputes both parts from scratch inside the loop, using slices '
             'such as nums[:i] and nums[i:] with sum, max, min or set. Do not precompute anything.',
             'Use this method: in one pass from the left and one pass from the right, precompute the running values you need (prefix '
             'and suffix sums, maxima, minima or distinct counts), then check each split in O(1). Do not slice the list inside the '
             'loop. It must run in O(n) time.')
Base('splits', 'count_splits(nums)',
     [dict(mode=mode) for mode in ('more', 'order', 'distinct')] + [dict(mode='ratio', u=u) for u in range(2, 7)]
     + [dict(mode='mod', d=d) for d in range(2, 21)],
     split_spec, split_brute, split_fast, split_small,
     lambda p, st: f"({st.up() if p['mode'] == 'order' else st.rand(1)},)", N=200000, how=lambda p: SPLIT_HOW)


def best_gain_brute(a, p):
    n = len(a)
    if p['mode'] == 'gain':
        return max(0, max((x - a[i] for i in range(n) for x in a[i + p['g']:]), default=0))
    return max(a[i] + a[j] - p['c'] * (j - i) for i in range(n) for j in range(i + 1, n))


def best_gain_fast(a, p):
    n = len(a)
    if p['mode'] == 'gain':
        g, low, best = p['g'], math.inf, 0
        for j in range(g, n):
            low = min(low, a[j - g])
            best = max(best, a[j] - low)
        return best
    c, left, best = p['c'], a[0], -math.inf
    for j in range(1, n):
        best = max(best, left + a[j] - c * j)
        left = max(left, a[j] + c * j)
    return best


def best_gain_spec(p):
    if p['mode'] == 'gain':
        far = f" with j - i >= {p['g']}" if p['g'] > 1 else ''
        return f'Return the largest value of nums[j] - nums[i] over all index pairs i < j{far}, or 0 if no such value is positive.'
    return f"Return the largest value of nums[i] + nums[j] - {p['c']} * (j - i) over all index pairs i < j."


def best_gain_small(rng, p, st):
    g = p.get('g', 1)
    a = ints(rng, st.size(rng, 12 + g, 20 + g), 0, 60)
    return [a] if len(st.eff(a)) >= g + 2 else None


Base('best_gain', 'best_gain(nums)', [dict(mode='gain', g=g) for g in range(1, 26)] + [dict(mode='score', c=c) for c in range(1, 26)],
     best_gain_spec, best_gain_brute, best_gain_fast, best_gain_small, lambda p, st: f'({st.rand()},)', N=240000)


def range_queries_brute(a, queries, p):
    mode, out = p['mode'], []
    for l, r in queries:
        part = a[l:r + 1]
        if mode == 'sum':
            out.append(sum(part))
        elif mode == 'sum_mod':
            out.append(sum(part) % p['d'])
        elif mode == 'count':
            out.append(sum(x % p['d'] == 0 for x in part))
        else:
            v = 0
            for x in part:
                v ^= x
            out.append(v)
    return out


def range_queries_fast(a, queries, p):
    mode, pre = p['mode'], [0]
    for x in a:
        pre.append(pre[-1] ^ x if mode == 'xor' else pre[-1] + (x % p['d'] == 0 if mode == 'count' else x))
    if mode == 'xor':
        return [pre[r + 1] ^ pre[l] for l, r in queries]
    if mode == 'sum_mod':
        return [(pre[r + 1] - pre[l]) % p['d'] for l, r in queries]
    return [pre[r + 1] - pre[l] for l, r in queries]


def range_queries_spec(p):
    what = dict(sum='the sum of these elements', xor='the bitwise XOR of these elements', count=f"how many of these elements are divisible by {p.get('d')}",
                sum_mod=f"the sum of these elements modulo {p.get('d')}")[p['mode']]
    return f'queries is a list of [l, r] index pairs with l <= r.||For each query, consider the elements nums[l], ..., nums[r] inclusive. Return a list with {what}, one entry per query, in order.'


def range_queries_small(rng, p, st):
    n = rng.randint(12, 20)
    queries = []
    for _ in range(rng.randint(4, 6)):
        l = rng.randrange(n)
        queries.append([l, rng.randrange(l, n)])
    return [ints(rng, n, 0, 40), queries]


def range_queries_stress(p, st):
    n = st.n
    return f'({st.rand()}, [[random.randint(0, 99), {n} - 1 - random.randint(0, 99)] for _ in range({n})])'


Base('range_queries', 'range_queries(nums, queries)',
     [dict(mode=mode) for mode in ('sum', 'xor')] + [dict(mode=mode, d=d) for mode in ('count', 'sum_mod') for d in range(2, 31)],
     range_queries_spec, range_queries_brute, range_queries_fast, range_queries_small, range_queries_stress, N=200000,
     ret='list', only=('add', 'mul', 'rotate', 'reverse'))


# ---------------------------------------------------------------- order, stacks and sorting
def big_pairs_brute(a, p):
    u, t = p['u'], p['t']
    out = [sum(x > u * a[j] + t for x in a[:j]) for j in range(len(a))]
    return out if p['ret'] == 'list' else sum(out)


def big_pairs_fast(a, p):
    u, t, s, n, out = p['u'], p['t'], sorted(a), len(a), []
    tree = [0] * (n + 1)
    for j, x in enumerate(a):
        q, below = bisect_right(s, u * x + t), 0
        while q:
            below += tree[q]
            q &= q - 1
        out.append(j - below)
        q = bisect_left(s, x) + 1
        while q <= n:
            tree[q] += 1
            q += q & -q
    return out if p['ret'] == 'list' else sum(out)


def big_pairs_spec(p):
    u, t = p['u'], p['t']
    expr = f"nums[i] > {f'{u} * ' if u > 1 else ''}nums[j]{f' + {t}' if t else ''}"
    if p['ret'] == 'list':
        return f'Return a list whose element j is the number of indices i < j with {expr}.'
    return f'Return the number of index pairs i < j with {expr}.'


def big_pairs_small(rng, p, st):
    a = ints(rng, st.size(rng), 0, 40)
    return [a] if len(st.eff(a)) >= 3 else None


Base('big_pairs', 'count_big_pairs(nums)', grid(u=range(1, 5), t=range(0, 11), ret=('total', 'list')), big_pairs_spec, big_pairs_brute,
     big_pairs_fast, big_pairs_small, lambda p, st: f'({st.distinct()},)', N=80000, ret=lambda p: 'list' if p['ret'] == 'list' else 'int')


def nearest_brute(a, p):
    n, t, sign, out = len(a), p['t'], 1 if p['cmp'] == 'greater' else -1, []
    for i in range(n):
        order = range(i + 1, n) if p['dir'] == 'right' else range(i - 1, -1, -1)
        j = next((j for j in order if sign * (a[j] - a[i]) >= t), None)
        out.append(-1 if j is None else a[j] if p['ret'] == 'value' else abs(i - j))
    return out


def nearest_fast(a, p):
    t, sign, left = p['t'], 1 if p['cmp'] == 'greater' else -1, p['dir'] == 'left'
    b = [sign * x for x in (a[::-1] if left else a)]
    n, out, vals, idx = len(b), [-1] * len(b), [], []
    for i in range(n - 1, -1, -1):
        v = b[i]
        while vals and -vals[-1] <= v:
            vals.pop()
            idx.pop()
        q = bisect_right(vals, -(v + t)) - 1
        if q >= 0:
            out[i] = sign * b[idx[q]] if p['ret'] == 'value' else idx[q] - i
        vals.append(-v)
        idx.append(i)
    return out[::-1] if left else out


def nearest_spec(p):
    t, more = p['t'], p['cmp'] == 'greater'
    cond = ('nums[j] > nums[i]' if t == 1 else f'nums[j] >= nums[i] + {t}') if more else ('nums[j] < nums[i]' if t == 1 else f'nums[j] <= nums[i] - {t}')
    what = 'nums[j]' if p['ret'] == 'value' else 'the distance abs(i - j)'
    return (f"For each index i, let j be the nearest index to the {p['dir']} of i with {cond}. "
            f'Return a list whose element i is {what}, or -1 if there is no such j.')


def nearest_small(rng, p, st):
    a = ints(rng, st.size(rng, 12, 20), 0, 30)
    return [a] if len(st.eff(a)) >= 3 else None


Base('nearest', 'nearest_match(nums)', grid(dir=('right', 'left'), cmp=('greater', 'smaller'), t=range(1, 16), ret=('value', 'dist')),
     nearest_spec, nearest_brute, nearest_fast, nearest_small,
     lambda p, st: f"({st.down() if (p['dir'] == 'right') == (p['cmp'] == 'greater') else st.up()},)", N=200000, ret='list')


def leaders_answer(a, picks, p):
    if p['ret'] == 'count':
        return len(picks)
    values = [a[i] for i in picks]
    return values if p['ret'] == 'values' else sum(values)


def leaders_brute(a, p):
    t, sign = p['t'], 1 if p['cmp'] == 'greater' else -1
    picks = [i for i in range(len(a)) if all(sign * (a[i] - x) >= t for x in (a[i + 1:] if p['dir'] == 'right' else a[:i]))]
    return leaders_answer(a, picks, p)


def leaders_fast(a, p):
    t, sign, best, picks = p['t'], 1 if p['cmp'] == 'greater' else -1, -math.inf, []
    for i in (range(len(a) - 1, -1, -1) if p['dir'] == 'right' else range(len(a))):
        v = sign * a[i]
        if v - best >= t:
            picks.append(i)
        best = max(best, v)
    return leaders_answer(a, sorted(picks), p)


def leaders_spec(p):
    t, more = p['t'], p['cmp'] == 'greater'
    other = 'j > i' if p['dir'] == 'right' else 'j < i'
    cond = ('nums[i] > nums[j]' if t == 1 else f'nums[i] >= nums[j] + {t}') if more else ('nums[i] < nums[j]' if t == 1 else f'nums[i] <= nums[j] - {t}')
    what = dict(count='the number of leaders', values='a list of the values at the leaders, in index order', sum='the sum of the values at the leaders')[p['ret']]
    return f'Call index i a leader if {cond} for every index {other} (an index with no such j is a leader). Return {what}.'


def leaders_small(rng, p, st):
    a = ints(rng, st.size(rng, 12, 20), 0, 40)
    return [a] if len(st.eff(a)) >= 3 else None


def leaders_how(p):
    part, start, keep = ('nums[i + 1:]', 'right', 'maximum') if p['dir'] == 'right' else ('nums[:i]', 'left', 'maximum')
    keep = keep if p['cmp'] == 'greater' else 'minimum'
    return (f'Use this method: for each index, compare it with the other elements from scratch inside the loop, using an inner for loop or a '
            f'slice such as all(...) over {part}. Do not keep any running maximum or minimum.',
            f'Use this method: make one pass starting from the {start} end, keeping a running {keep} of what has been seen, and compare each '
            'element only with that single value. Use no inner loop and no slices. It must run in O(n) time.')


Base('leaders', 'find_leaders(nums)', grid(dir=('right', 'left'), cmp=('greater', 'smaller'), t=range(1, 16), ret=('count', 'values', 'sum')),
     leaders_spec, leaders_brute, leaders_fast, leaders_small,
     lambda p, st: f"({st.down() if (p['dir'] == 'right') == (p['cmp'] == 'greater') else st.up()},)", N=200000,
     ret=lambda p: 'list' if p['ret'] == 'values' else 'int', how=leaders_how)


def extremes_brute(a, p):
    mode, n, total = p['mode'], len(a), 0
    if mode.startswith('sum'):
        for i in range(n):
            low = high = a[i]
            for j in range(i, n):
                low, high = min(low, a[j]), max(high, a[j])
                total += low if mode == 'sum_min' else high if mode == 'sum_max' else high - low
        return total
    side, kind = mode.split('_')
    b = a if side == 'first' else a[::-1]
    for i in range(n):
        total += 1
        for j in range(i + 1, n):
            if (b[j] >= b[i]) if kind == 'max' else (b[j] <= b[i]):
                break
            total += 1
    return total


def sum_of_minimums(a):
    total, stack = 0, []
    for i in range(len(a) + 1):
        x = a[i] if i < len(a) else -math.inf
        while stack and a[stack[-1]] > x:
            m = stack.pop()
            total += a[m] * (m - (stack[-1] if stack else -1)) * (i - m)
        stack.append(i)
    return total


def extremes_fast(a, p):
    mode = p['mode']
    if mode == 'sum_min':
        return sum_of_minimums(a)
    if mode == 'sum_max':
        return -sum_of_minimums([-x for x in a])
    if mode == 'sum_range':
        return -sum_of_minimums([-x for x in a]) - sum_of_minimums(a)
    side, kind = mode.split('_')
    b, sign, stack, total = (a if side == 'first' else a[::-1]), 1 if kind == 'max' else -1, [], 0
    for i in range(len(b) - 1, -1, -1):
        while stack and sign * b[stack[-1]] < sign * b[i]:
            stack.pop()
        total += (stack[-1] if stack else len(b)) - i
        stack.append(i)
    return total


def extremes_spec(p):
    mode = p['mode']
    if mode.startswith('sum'):
        what = dict(sum_min="the subarray's smallest element", sum_max="the subarray's largest element",
                    sum_range="the subarray's largest element minus its smallest element")[mode]
        return f'Return the sum, over all non-empty contiguous subarrays of nums, of {what}.'
    side, kind = mode.split('_')
    word = 'larger' if kind == 'max' else 'smaller'
    return f'Return the number of non-empty contiguous subarrays of nums whose {side} element is strictly {word} than all their other elements. A subarray of one element counts.'


def extremes_small(rng, p, st):
    a = ints(rng, st.size(rng, 10, 18), 0, 30)
    return [a] if len(st.eff(a)) >= 3 else None


EXTREME_CORE = dict(first_max='down', first_min='up', last_max='up', last_min='down')
Base('extremes', 'subarray_extremes(nums)', [dict(mode=mode) for mode in ('sum_min', 'sum_max', 'sum_range', 'first_max', 'first_min', 'last_max', 'last_min')],
     extremes_spec, extremes_brute, extremes_fast, extremes_small,
     lambda p, st: f"({st.down() if EXTREME_CORE.get(p['mode'], 'up') == 'down' else st.up()},)", N=100000)


def block_score(m, L, s, p):
    mode = p['mode']
    return m * s if mode == 'minsum' else m * (L + p['c']) if mode == 'plus' else m * L


def rectangle_brute(a, p):
    best, n = 0, len(a)
    for i in range(n):
        m, s = a[i], 0
        for j in range(i, n):
            m, s = min(m, a[j]), s + a[j]
            if p['mode'] != 'atleast' or j - i + 1 >= p['L']:
                best = max(best, block_score(m, j - i + 1, s, p))
    return best


def rectangle_fast(a, p):
    pre, stack, best, n = [0], [], 0, len(a)
    for x in a:
        pre.append(pre[-1] + x)
    for i in range(n + 1):
        x = a[i] if i < n else 0
        while stack and a[stack[-1]] >= x:
            m = a[stack.pop()]
            left = stack[-1] + 1 if stack else 0
            if p['mode'] != 'atleast' or i - left >= p['L']:
                best = max(best, block_score(m, i - left, pre[i] - pre[left], p))
        stack.append(i)
    return best


def rectangle_spec(p):
    score = dict(area='m * b', minsum='m * s', plus=f"m * (b + {p.get('c')})", atleast='m * b')[p['mode']]
    blocks = f"blocks with at least {p.get('L')} bars" if p['mode'] == 'atleast' else 'non-empty blocks'
    return ('heights contains positive integers.||For a contiguous block of bars, let m be its smallest height, b its number of bars '
            f'and s the sum of its heights. Return the largest value of {score} over all {blocks}.')


def rectangle_small(rng, p, st):
    L = p.get('L', 1)
    a = ints(rng, st.size(rng, 8 + L, 16 + L), 1, 20)
    return [a] if len(st.eff(a)) >= L + 1 else None


Base('rectangle', 'best_block(heights)',
     [dict(mode=mode) for mode in ('area', 'minsum')] + [dict(mode='plus', c=c) for c in range(1, 16)] + [dict(mode='atleast', L=L) for L in range(2, 17)],
     rectangle_spec, rectangle_brute, rectangle_fast, rectangle_small, lambda p, st: f'({st.up()},)', N=100000)


def water_brute(a, p):
    n = len(a)
    if p['mode'] == 'container':
        return max((min(a[i], a[j]) * (j - i) for i in range(n) for j in range(i + p['g'], n)), default=0)
    c = p.get('c', math.inf)
    return sum(max(0, min(max(a[:i + 1]), max(a[i:]), c) - a[i]) for i in range(n))


def water_fast(a, p):
    n = len(a)
    if p['mode'] == 'container':
        i, j, best = 0, n - 1, 0
        while j - i >= p['g']:
            best = max(best, min(a[i], a[j]) * (j - i))
            if a[i] < a[j]:
                i += 1
            else:
                j -= 1
        return best
    c, lefts, high, total = p.get('c', math.inf), [0] * n, 0, 0
    for i in range(n):
        high = lefts[i] = max(high, a[i])
    high = 0
    for i in range(n - 1, -1, -1):
        high = max(high, a[i])
        total += max(0, min(lefts[i], high, c) - a[i])
    return total


def water_spec(p):
    if p['mode'] == 'container':
        far = f" with j - i >= {p['g']}" if p['g'] > 1 else ''
        return f'heights contains non-negative integers.||Return the largest value of min(heights[i], heights[j]) * (j - i) over all index pairs i < j{far}, or 0 if there are no such pairs.'
    level = 'the smaller of the tallest bar at or left of i and the tallest bar at or right of i'
    if p['mode'] == 'capped':
        level = f"the smallest of the tallest bar at or left of i, the tallest bar at or right of i, and {p['c']}"
    return f'heights contains non-negative integers.||The water at index i is the water level minus heights[i], or 0 if that is negative, where the water level is {level}. Return the total water over all indices.'


def water_small(rng, p, st):
    g = p.get('g', 1)
    a = ints(rng, st.size(rng, 12 + g, 20 + g), 0, 25)
    return [a] if len(st.eff(a)) >= g + 2 else None


WATER_HOW = ('Use this method: for each index, find the tallest bar on each side from scratch inside the loop, using max(heights[:i + 1]) '
             'and max(heights[i:]). Do not store running maxima.',
             'Use this method: build a list of running maxima in one pass from the left and another in one pass from the right, then '
             'compute each index in O(1). Do not slice the list inside the loop. It must run in O(n) time.')
PAIR_WATER_HOW = ('Use this method: two nested for loops over all index pairs i < j, computing the value of each pair. Do not use two pointers.',
                  'Use this method: two pointers starting at both ends in a single while loop, moving the pointer at the shorter bar inward. '
                  'Do not use nested loops. It must run in O(n) time.')
Base('water', 'water_score(heights)',
     [dict(mode='trapped')] + [dict(mode='capped', c=c) for c in range(5, 41)] + [dict(mode='container', g=g) for g in range(1, 16)],
     water_spec, water_brute, water_fast, water_small, lambda p, st: f'({st.rand()},)', N=200000,
     how=lambda p: PAIR_WATER_HOW if p['mode'] == 'container' else WATER_HOW)


def chain_brute(a, p):
    t, b, dp = p['t'], a if p['dir'] == 'inc' else [-x for x in a], []
    for j, x in enumerate(b):
        dp.append(1 + max((dp[i] for i in range(j) if b[i] + t <= x), default=0))
    best = max(dp, default=0)
    return best if p['ret'] == 'len' else len(a) - best


def chain_fast(a, p):
    t, b, tails = p['t'], a if p['dir'] == 'inc' else [-x for x in a], []
    for x in b:
        q = bisect_right(tails, x - t)
        if q == len(tails):
            tails.append(x)
        elif x < tails[q]:
            tails[q] = x
    return len(tails) if p['ret'] == 'len' else len(a) - len(tails)


def chain_spec(p):
    t, inc = p['t'], p['dir'] == 'inc'
    if t == 0:
        rule = f"each element is at least as {'large' if inc else 'small'} as the one before it"
    elif t == 1:
        rule = f"each element is {'larger' if inc else 'smaller'} than the one before it"
    else:
        rule = f"each element is at least {t} {'larger' if inc else 'smaller'} than the one before it"
    if p['ret'] == 'len':
        return f'Return the length of the longest subsequence of nums, not necessarily contiguous, in which {rule}.'
    return f'Return the smallest number of elements to delete from nums so that, in what remains, {rule}.'


def chain_small(rng, p, st):
    a = ints(rng, st.size(rng, 10, 18), 0, 30 + 8 * p['t'])
    return [a] if len(st.eff(a)) >= 3 else None


Base('chain', 'best_chain(nums)', grid(dir=('inc', 'dec'), t=range(0, 21), ret=('len', 'del')), chain_spec, chain_brute, chain_fast,
     chain_small, lambda p, st: f'({st.rand()},)', N=100000)


def closest_brute(a, p):
    mode, n = p['mode'], len(a)
    diffs = (abs(a[i] - x) for i in range(n) for x in a[i + 1:])
    if mode == 'min':
        return min(diffs)
    if mode == 'count':
        d = min(diffs)
        return sum(abs(a[i] - x) == d for i in range(n) for x in a[i + 1:])
    if mode == 'under':
        return max((v for v in diffs if v <= p['t']), default=-1)
    return sum(any(j != i and abs(a[i] - a[j]) <= p['t'] for j in range(n)) for i in range(n))


def closest_fast(a, p):
    mode, s, n = p['mode'], sorted(a), len(a)
    gaps = [s[i + 1] - s[i] for i in range(n - 1)]
    if mode == 'min':
        return min(gaps)
    if mode == 'count':
        d = min(gaps)
        return sum(c * (c - 1) // 2 for c in Counter(s).values()) if d == 0 else gaps.count(d)
    if mode == 'under':
        best = -1
        for i in range(n):
            j = bisect_right(s, s[i] + p['t']) - 1
            if j > i:
                best = max(best, s[j] - s[i])
        return best
    return sum((i > 0 and gaps[i - 1] <= p['t']) or (i < n - 1 and gaps[i] <= p['t']) for i in range(n))


def closest_spec(p):
    if p['mode'] == 'min':
        return 'Return the smallest value of abs(nums[i] - nums[j]) over all index pairs i < j.'
    if p['mode'] == 'count':
        return 'Let d be the smallest value of abs(nums[i] - nums[j]) over all index pairs i < j. Return the number of index pairs i < j with abs(nums[i] - nums[j]) == d.'
    if p['mode'] == 'under':
        return f"Return the largest value of abs(nums[i] - nums[j]) that is at most {p['t']}, over all index pairs i < j, or -1 if there is none."
    return f"Return the number of indices i for which some other index j has abs(nums[i] - nums[j]) <= {p['t']}."


def closest_small(rng, p, st):
    a = ints(rng, st.size(rng, 8, 16), 0, 80)
    return [a] if len(st.eff(a)) >= 3 else None


CLOSEST_HOW = ('Use this method: two nested for loops over all index pairs, computing abs(nums[i] - nums[j]) inside the inner loop. Do not sort the list.',
               'Use this method: sort the values first, then compare only values next to each other in sorted order, using bisect where a '
               'range is needed. Do not loop over all pairs. It must run in O(n log n) time.')
Base('closest', 'closest_values(nums)',
     [dict(mode=mode) for mode in ('min', 'count')] + [dict(mode=mode, t=t) for mode in ('under', 'near') for t in range(1, 31)],
     closest_spec, closest_brute, closest_fast, closest_small,
     lambda p, st: f'(random.sample(range(1, {101 * st.n}, 101), {st.n}),)', N=120000, how=lambda p: CLOSEST_HOW)


PADDING = '[-10 ** 7] * 6'  # dropping values from either end of a stress input removes only these, so its runs stay whole


def progression_brute(a, p):
    t, best, start = p['t'], 0, None
    for v in a:
        m = 1
        while v + m * t in a:
            m += 1
        if m > best or (m == best and v < start):
            best, start = m, v
    return best if p['ret'] == 'len' else start


def progression_fast(a, p):
    t, have, best, start = p['t'], set(a), 0, None
    for v in sorted(have):
        if v - t not in have:
            m = 1
            while v + m * t in have:
                m += 1
            if m > best:
                best, start = m, v
    return best if p['ret'] == 'len' else start


def progression_spec(p):
    t = p['t']
    run = f'nums is non-empty.||A run is a list of integers v, v + {t}, v + {2 * t}, ... of any length, starting from any integer v.'
    if p['ret'] == 'len':
        return f'{run} Return the length of the longest run all of whose values occur in nums.'
    return f'{run} Among the longest runs all of whose values occur in nums, return the smallest starting value v.'


def progression_small(rng, p, st):
    a = [rng.randint(0, 2) + p['t'] * rng.randint(0, 14) for _ in range(st.size(rng, 10, 18))]
    return [a] if len(st.eff(a)) >= 3 else None


Base('progression', 'longest_run(nums)', grid(t=range(1, 26), ret=('len', 'start')), progression_spec, progression_brute, progression_fast,
     progression_small, lambda p, st: f"({PADDING} + random.sample(range(0, {p['t']} * {st.n}, {p['t']}), {st.n}) + {PADDING},)", N=100000,
     only=('add', 'drop_first', 'drop_last'))


def missing_brute(a, p):
    t = p['t']
    if p['mode'] == 'kth':
        left, v = t, 0
        while left:
            v += 1
            if v not in a:
                left -= 1
        return v
    v = t
    while v in a:
        v += t
    return v


def missing_fast(a, p):
    return missing_brute(set(a), p)


def missing_spec(p):
    t = p['t']
    if p['mode'] == 'mult':
        return f'Return the smallest positive multiple of {t} that does not occur in nums.'
    if t == 1:
        return 'Return the smallest positive integer that does not occur in nums.'
    return f'Consider the positive integers that do not occur in nums, in increasing order. Return the one at position {t}, counting from 1.'


def missing_small(rng, p, st):
    t, c, m = (p['t'] if p['mode'] == 'mult' else 1), st.got.get('add', 0), rng.randint(5, 15)
    a = [t * v - c for v in rng.sample(range(1, m + 4), m)]
    return [a] if len(st.eff(a)) >= 3 else None


def missing_stress(p, st):
    t, c = (p['t'] if p['mode'] == 'mult' else 1), st.got.get('add', 0)
    return f'({PADDING} + random.sample(range({t} - {c}, {t} * {st.n} + 1 - {c}, {t}), {st.n}) + {PADDING},)'


Base('missing', 'smallest_missing(nums)', [dict(mode='kth', t=t) for t in range(1, 13)] + [dict(mode='mult', t=t) for t in range(2, 13)],
     missing_spec, missing_brute, missing_fast, missing_small, missing_stress, N=100000, only=('add', 'drop_first', 'drop_last'))


def product_brute(a, p):
    g, M, n, out = p['g'], p['M'], len(a), []
    for i in range(n):
        r = 1 % M
        for j in range(n):
            if abs(i - j) >= g:
                r = r * a[j] % M
        out.append(r)
    return out


def product_fast(a, p):
    g, M, n = p['g'], p['M'], len(a)
    pre, suf = [1 % M], [1 % M] * (n + 1)
    for x in a:
        pre.append(pre[-1] * x % M)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] * a[i] % M
    return [pre[max(0, i - g + 1)] * suf[min(n, i + g)] % M for i in range(n)]


def product_small(rng, p, st):
    a = [rng.choice((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 5, 7)) for _ in range(st.size(rng, 8, 14))]
    return [a] if len(st.eff(a)) >= p['g'] + 2 else None


def product_spec(p):
    other = 'j != i' if p['g'] == 1 else f"abs(i - j) >= {p['g']}"
    return (f"Return a list whose element i is the product of all elements nums[j] with {other}, modulo {p['M']}. An empty product is 1. "
            'nums may contain zeros, and the modulus need not be prime.')


Base('product', 'product_of_others(nums)', grid(M=(1000, 1009, 1024, 7919, 10000, 10007, 65536, 1000003, 998244353, 1000000007), g=range(1, 6)),
     product_spec, product_brute, product_fast, product_small, lambda p, st: f'({st.rand(1)},)', N=100000, ret='list')


# ---------------------------------------------------------------- windows of size w
WINDOW = 'A window is a contiguous block of exactly w elements of nums. '


def window_input(rng, st, a):
    """[a, w] with a window size that fits after the input steps, or None."""
    e = st.eff(a)
    return [a, rng.randint(2, min(8, len(e) - 1))] if len(e) >= 4 else None


def window_distinct_small(rng, p, st):
    q = p.get('q', 2)
    a = ints(rng, st.size(rng, q + 12, q + 20), 0, q + 5)
    e = st.eff(a)
    return [a, rng.randint(q, min(q + 5, len(e) - 2))] if len(e) >= q + 4 else None


def window_distinct_answer(values, p):
    mode = p['mode']
    if mode in ('list', 'once'):
        return list(values)
    if mode == 'max':
        return max(values)
    return sum(v >= p['q'] if mode == 'atleast' else v == p['q'] for v in values)


def window_distinct_brute(a, w, p):
    if p['mode'] == 'once':
        values = (sum(c == 1 for c in Counter(a[i:i + w]).values()) for i in range(len(a) - w + 1))
    else:
        values = (len(set(a[i:i + w])) for i in range(len(a) - w + 1))
    return window_distinct_answer(values, p)


def window_distinct_fast(a, w, p):
    count, distinct, once, out = {}, 0, 0, []
    for j, x in enumerate(a):
        c = count.get(x, 0)
        count[x] = c + 1
        distinct += c == 0
        once += (c == 0) - (c == 1)
        if j >= w:
            y = a[j - w]
            c = count[y] - 1
            count[y] = c
            distinct -= c == 0
            once += (c == 1) - (c == 0)
        if j >= w - 1:
            out.append(once if p['mode'] == 'once' else distinct)
    return window_distinct_answer(out, p)


def window_distinct_spec(p):
    q = p.get('q')
    return WINDOW + dict(list='Return a list with the number of distinct values in each window, from left to right.',
                         once='Return a list with, for each window from left to right, the number of values that occur exactly once in it.',
                         max='Return the largest number of distinct values in any window.',
                         atleast=f'Return the number of windows containing at least {q} distinct values.',
                         exact=f'Return the number of windows containing exactly {q} distinct values.')[p['mode']]


Base('window_distinct', 'window_distinct(nums, w)',
     [dict(mode=mode) for mode in ('list', 'once', 'max')] + [dict(mode='atleast', q=q) for q in range(2, 11)] + [dict(mode='exact', q=q) for q in range(1, 11)],
     window_distinct_spec, window_distinct_brute, window_distinct_fast, window_distinct_small,
     lambda p, st: f'({st.rand()}, {st.N // 8})', N=120000, ret=lambda p: 'list' if p['mode'] in ('list', 'once') else 'int')


def window_extremes_answer(highs, lows, p):
    mode = p['mode']
    if mode in ('max', 'min'):
        return list(highs if mode == 'max' else lows)
    gaps = (h - l for h, l in zip(highs, lows))
    return list(gaps) if mode == 'range' else sum(gaps) if mode == 'total' else sum(v <= p['q'] for v in gaps)


def window_extremes_brute(a, w, p):
    n = len(a) - w + 1
    return window_extremes_answer((max(a[i:i + w]) for i in range(n)), (min(a[i:i + w]) for i in range(n)), p)


def window_extremes_fast(a, w, p):
    high, low, highs, lows = deque(), deque(), [], []
    for j, x in enumerate(a):
        while high and a[high[-1]] <= x:
            high.pop()
        while low and a[low[-1]] >= x:
            low.pop()
        high.append(j)
        low.append(j)
        if j >= w - 1:
            if high[0] <= j - w:
                high.popleft()
            if low[0] <= j - w:
                low.popleft()
            highs.append(a[high[0]])
            lows.append(a[low[0]])
    return window_extremes_answer(highs, lows, p)


def window_extremes_spec(p):
    return WINDOW + dict(max='Return a list with the largest value of each window, from left to right.',
                         min='Return a list with the smallest value of each window, from left to right.',
                         range='Return a list with the largest value minus the smallest value of each window, from left to right.',
                         total='Return the sum, over all windows, of the largest value minus the smallest value.',
                         atmost=f"Return the number of windows whose largest value minus smallest value is at most {p.get('q')}.")[p['mode']]


Base('window_extremes', 'window_extremes(nums, w)',
     [dict(mode=mode) for mode in ('max', 'min', 'range', 'total')] + [dict(mode='atmost', q=q) for q in range(1, 31)],
     window_extremes_spec, window_extremes_brute, window_extremes_fast, lambda rng, p, st: window_input(rng, st, ints(rng, st.size(rng), 0, 40)),
     lambda p, st: f'({st.rand()}, {st.N // 8})', N=360000, ret=lambda p: 'list' if p['mode'] in ('max', 'min', 'range') else 'int')


def window_sum_answer(sums, w, p):
    mode = p['mode']
    if mode in ('max', 'min'):
        return max(sums) if mode == 'max' else min(sums)
    if mode == 'first_max':
        return sums.index(max(sums))
    return sum(v % p['d'] == 0 for v in sums) if mode == 'div' else sum(v >= p['t'] * w for v in sums)


def window_sums_brute(a, w, p):
    return window_sum_answer([sum(a[i:i + w]) for i in range(len(a) - w + 1)], w, p)


def window_sums_fast(a, w, p):
    s = sum(a[:w])
    sums = [s]
    for j in range(w, len(a)):
        s += a[j] - a[j - w]
        sums.append(s)
    return window_sum_answer(sums, w, p)


def window_sums_spec(p):
    return WINDOW + dict(max='Return the largest sum of any window.', min='Return the smallest sum of any window.',
                         first_max='Return the starting index of the first window with the largest sum.',
                         div=f"Return the number of windows whose sum is divisible by {p.get('d')}.",
                         avg=f"Return the number of windows whose average value is at least {p.get('t')}.")[p['mode']]


WINDOW_HOW = ('Use this method: a loop over every window start that recomputes the window sum from scratch inside the loop, with '
              'sum(nums[i:i + w]) or an inner for loop. Do not keep a running sum.',
              'Use this method: one pass with a running sum, adding the element that enters the window and subtracting the element that '
              'leaves it. Never recompute a window from scratch and do not slice inside the loop. It must run in O(n) time.')
Base('window_sums', 'window_sums(nums, w)',
     [dict(mode=mode) for mode in ('max', 'min', 'first_max')] + [dict(mode='div', d=d) for d in range(2, 31)] + [dict(mode='avg', t=t) for t in range(1, 31)],
     window_sums_spec, window_sums_brute, window_sums_fast, lambda rng, p, st: window_input(rng, st, ints(rng, st.size(rng), -20, 40)),
     lambda p, st: f'({st.rand()}, {st.N // 8})', N=400000, skip=('no_below',), how=lambda p: WINDOW_HOW)


def limit_update(mode, d, total, x):
    return total + x if mode == 'sum' else total + (x % d == 0)


def longest_window_brute(a, k, p):
    mode, d, best, start = p['mode'], p.get('d'), 0, -1
    for i in range(len(a)):
        total, seen, low, high = 0, set(), a[i], a[i]
        for j in range(i, len(a)):
            x = a[j]
            if mode in ('sum', 'mult'):
                total = limit_update(mode, d, total, x)
                over = total > k
            elif mode == 'distinct':
                seen.add(x)
                over = len(seen) > k
            else:
                low, high = min(low, x), max(high, x)
                over = high - low > k
            if over:
                break
            if j - i + 1 > best:
                best, start = j - i + 1, i
    return best if p['ret'] == 'len' else start


def longest_window_fast(a, k, p):
    mode, d, i, best, start, total, count, high, low = p['mode'], p.get('d'), 0, 0, -1, 0, {}, deque(), deque()
    for j, x in enumerate(a):
        if mode in ('sum', 'mult'):
            total = limit_update(mode, d, total, x)
        elif mode == 'distinct':
            count[x] = count.get(x, 0) + 1
        else:
            while high and a[high[-1]] <= x:
                high.pop()
            while low and a[low[-1]] >= x:
                low.pop()
            high.append(j)
            low.append(j)
        while i <= j and (total > k if mode in ('sum', 'mult') else len(count) > k if mode == 'distinct' else a[high[0]] - a[low[0]] > k):
            y = a[i]
            if mode in ('sum', 'mult'):
                total -= y if mode == 'sum' else y % d == 0
            elif mode == 'distinct':
                count[y] -= 1
                if not count[y]:
                    del count[y]
            else:
                if high[0] == i:
                    high.popleft()
                if low[0] == i:
                    low.popleft()
            i += 1
        if j - i + 1 > best:
            best, start = j - i + 1, i
    return best if p['ret'] == 'len' else start


def longest_window_spec(p):
    cond = dict(sum='whose sum is at most k', range='whose largest and smallest values differ by at most k', distinct='with at most k distinct values',
                mult=f"with at most k multiples of {p.get('d')}")[p['mode']]
    return 'nums contains positive integers.||' + longest_text('non-empty contiguous subarray of nums', cond, p['ret'])


def longest_window_small(rng, p, st):
    a = ints(rng, st.size(rng), 1, 20)
    e, mode = st.eff(a), p['mode']
    if len(e) < 4:
        return None
    k = sum(e) // rng.randint(2, 5) if mode == 'sum' else (max(e) - min(e)) // rng.randint(2, 4) if mode == 'range' else rng.randint(1, 4)
    return [a, k]


def longest_window_stress(p, st):
    mode = p['mode']
    core = st.up() if mode == 'range' else st.distinct() if mode == 'distinct' else st.rand(1)
    k = dict(sum='sum(e)', range='(max(e) - min(e))', distinct='len(set(e))', mult=f"sum(v % {p.get('d')} == 0 for v in e)")[mode]
    return f'(lambda a: (lambda e: (a, {k} // 2))({st.E("a")}))({core})'


Base('longest_window', 'longest_window(nums, k)',
     [dict(mode=mode, ret=ret) for mode in ('sum', 'range', 'distinct') for ret in ('len', 'start')]
     + [dict(mode='mult', d=d, ret=ret) for d in range(2, 13) for ret in ('len', 'start')],
     longest_window_spec, longest_window_brute, longest_window_fast, longest_window_small, longest_window_stress, N=100000)


def shortest_window_brute(a, target, p):
    mode, d, best, start = p['mode'], p.get('d'), 0, -1
    for i in range(len(a)):
        total, seen = 0, set()
        for j in range(i, len(a)):
            if mode == 'distinct':
                seen.add(a[j])
                total = len(seen)
            else:
                total = limit_update(mode, d, total, a[j])
            if total >= target:
                if best == 0 or j - i + 1 < best:
                    best, start = j - i + 1, i
                break
    return best if p['ret'] == 'len' else start


def shortest_window_fast(a, target, p):
    mode, d, i, best, start, total, count = p['mode'], p.get('d'), 0, 0, -1, 0, {}
    for j, x in enumerate(a):
        if mode == 'distinct':
            count[x] = count.get(x, 0) + 1
            total = len(count)
        else:
            total = limit_update(mode, d, total, x)
        while total >= target and i <= j:
            if best == 0 or j - i + 1 < best:
                best, start = j - i + 1, i
            y = a[i]
            if mode == 'distinct':
                count[y] -= 1
                if not count[y]:
                    del count[y]
                total = len(count)
            else:
                total -= y if mode == 'sum' else y % d == 0
            i += 1
    return best if p['ret'] == 'len' else start


def shortest_window_spec(p):
    cond = dict(sum='whose sum is at least target', distinct='with at least target distinct values', mult=f"with at least target multiples of {p.get('d')}")[p['mode']]
    what = 'length' if p['ret'] == 'len' else 'starting index'
    none = 0 if p['ret'] == 'len' else -1
    ties = ' If several are equally short, use the one that starts first.' if p['ret'] == 'start' else ''
    return f'nums contains positive integers, and target is positive.||Return the {what} of the shortest contiguous subarray of nums {cond}, or {none} if there is none.{ties}'


def shortest_window_small(rng, p, st):
    a = ints(rng, st.size(rng), 1, 20)
    e, mode = st.eff(a), p['mode']
    if len(e) < 4:
        return None
    return [a, sum(e) // rng.randint(2, 5) + 1 if mode == 'sum' else rng.randint(1, 3) if mode == 'mult' else rng.randint(2, 5)]


def shortest_window_stress(p, st):
    mode = p['mode']
    core = st.distinct() if mode == 'distinct' else st.rand(1)
    k = dict(sum='sum(e)', distinct='len(set(e))', mult=f"sum(v % {p.get('d')} == 0 for v in e)")[mode]
    return f'(lambda a: (lambda e: (a, {k} * 3 // 5))({st.E("a")}))({core})'


Base('shortest_window', 'shortest_window(nums, target)',
     [dict(mode=mode, ret=ret) for mode in ('sum', 'distinct') for ret in ('len', 'start')]
     + [dict(mode='mult', d=d, ret=ret) for d in range(2, 13) for ret in ('len', 'start')],
     shortest_window_spec, shortest_window_brute, shortest_window_fast, shortest_window_small, shortest_window_stress, N=100000)


# ---------------------------------------------------------------- strings, words, intervals
def limited_brute(s, p):
    mode, q, best, start = p['mode'], p['q'], 0, -1
    for i in range(len(s)):
        count = {}
        for j in range(i, len(s)):
            count[s[j]] = count.get(s[j], 0) + 1
            if (len(count) > q) if mode == 'distinct' else count[s[j]] > q:
                break
            if j - i + 1 > best:
                best, start = j - i + 1, i
    return best if p['ret'] == 'len' else start


def limited_fast(s, p):
    mode, q, i, best, start, count = p['mode'], p['q'], 0, 0, -1, {}
    for j, c in enumerate(s):
        count[c] = count.get(c, 0) + 1
        while (len(count) > q) if mode == 'distinct' else count[c] > q:
            d = s[i]
            count[d] -= 1
            if not count[d]:
                del count[d]
            i += 1
        if j - i + 1 > best:
            best, start = j - i + 1, i
    return best if p['ret'] == 'len' else start


def limited_spec(p):
    q = p['q']
    cond = f'with at most {q} distinct characters' if p['mode'] == 'distinct' else 'with no repeated character' if q == 1 else f'in which no character occurs more than {q} times'
    return longest_text('non-empty substring of s', cond, p['ret'])


def limited_stress(p, st):
    q, n = p['q'], st.n
    if p['mode'] == 'distinct':
        return f"(''.join(chr(97 + i * {q + 2} // {n}) for i in range({n})),)"
    return f"(''.join(chr(65536 + i % {st.N // (q + 1)}) * {st.got.get('every', 1)} for i in range({st.N})),)"


def limited_small(rng, p, st):
    s = letters(rng, st.size(rng, 10 + 2 * p['q'], 18 + 2 * p['q']), 'abcdefghij' if p['mode'] == 'distinct' else LETTERS)
    return [s] if len(st.eff(s)) >= 3 else None


Base('limited', 'longest_limited_substring(s)', [dict(mode='rep', q=q, ret=ret) for q in range(1, 9) for ret in ('len', 'start')]
     + [dict(mode='distinct', q=q, ret=ret) for q in range(2, 9) for ret in ('len', 'start')],
     limited_spec, limited_brute, limited_fast, limited_small, limited_stress, N=50000, kind='str')


def exact_brute(s, p):
    q, mode, n = p['q'], p['mode'], len(s)
    if mode == 'first':
        return next((i for i in range(n) if s.count(s[i]) == q), -1)
    if mode == 'last':
        return next((i for i in range(n - 1, -1, -1) if s.count(s[i]) == q), -1)
    if mode == 'count':
        return sum(s.count(c) == q for c in set(s))
    return sum(i for i in range(n) if s.count(s[i]) == q)


def exact_fast(s, p):
    q, mode, count = p['q'], p['mode'], Counter(s)
    hits = [i for i, c in enumerate(s) if count[c] == q]
    if mode == 'first':
        return hits[0] if hits else -1
    if mode == 'last':
        return hits[-1] if hits else -1
    return sum(v == q for v in count.values()) if mode == 'count' else sum(hits)


def exact_spec(p):
    q = p['q']
    times = 'exactly once' if q == 1 else f'exactly {q} times'
    if p['mode'] in ('first', 'last'):
        return f"Return the index of the {p['mode']} position of s whose character occurs {times} in s, or -1 if there is none."
    if p['mode'] == 'count':
        return f'Return the number of distinct characters that occur {times} in s.'
    return f'Return the sum of the indices of all positions of s whose character occurs {times} in s.'


def exact_small(rng, p, st):
    s = letters(rng, st.size(rng, 12 + 3 * p['q'], 24 + 3 * p['q']), 'abcdefgh')
    return [s] if len(st.eff(s)) >= 3 else None


def exact_stress(p, st):
    # Every character occurs q + 2 times, even after thinning, so no position matches and scans can't stop early.
    q, s = p['q'], st.got.get('every', 1)
    return f"(''.join(chr(65536 + i % {st.N // (q + 2)}) * {s} for i in range({st.N})),)"


EXACT_HOW = ('Use this method: for each position, count the occurrences of its character from scratch inside the loop, with s.count(...) or '
             'an inner for loop. Do not use a dictionary or a Counter.',
             'Use this method: first count all characters once with a dictionary or collections.Counter, then make one pass over the string '
             'using those counts. Never call s.count and do not use nested loops. It must run in O(n) time.')
Base('exact', 'exact_count_query(s)', grid(mode=('first', 'last', 'count', 'sum'), q=range(1, 7)), exact_spec, exact_brute, exact_fast,
     exact_small, exact_stress, N=200000, kind='str', how=lambda p: EXACT_HOW)


def letter_balance_brute(s, p):
    x, y, t, total, best = p['x'], p['y'], p['t'], 0, 0
    for i in range(len(s)):
        acc = 0
        for j in range(i, len(s)):
            acc += (s[j] == x) - (s[j] == y)
            if acc == t:
                total += 1
                best = max(best, j - i + 1)
    return total if p['ret'] == 'count' else best


def letter_balance_fast(s, p):
    x, y, t, acc, total, best, seen, first = p['x'], p['y'], p['t'], 0, 0, 0, {0: 1}, {0: 0}
    for j, c in enumerate(s, 1):
        acc += (c == x) - (c == y)
        total += seen.get(acc - t, 0)
        if acc - t in first:
            best = max(best, j - first[acc - t])
        seen[acc] = seen.get(acc, 0) + 1
        first.setdefault(acc, j)
    return total if p['ret'] == 'count' else best


def letter_balance_spec(p):
    x, y, t = p['x'], p['y'], p['t']
    cond = f"the letter '{x}' occurs exactly as often as the letter '{y}'" if t == 0 else f"the letter '{x}' occurs exactly {t} more times than the letter '{y}'"
    if p['ret'] == 'count':
        return f'Return the number of non-empty substrings of s in which {cond}.'
    return f'Return the length of the longest non-empty substring of s in which {cond}, or 0 if there is none.'


def letter_balance_small(rng, p, st):
    s = letters(rng, st.size(rng, 12, 20), 'abcdef')
    return [s] if len(st.eff(s)) >= 3 else None


Base('letter_balance', 'letter_balance(s)',
     [dict(x=x, y=y, t=t, ret=ret) for x in 'abcdef' for y in 'abcdef' if x != y for t in range(5) if t > 0 or x < y for ret in ('count', 'longest')],
     letter_balance_spec, letter_balance_brute, letter_balance_fast, letter_balance_small,
     lambda p, st: f"(''.join(random.choices('abcdef', k={st.n})),)", N=120000, kind='str')


def cover_need(t, p):
    return Counter(t) if p['mode'] == 'multi' else {c: p['q'] for c in t}


def cover_brute(s, t, p):
    need, best, start = cover_need(t, p), 0, -1
    for i in range(len(s)):
        have = {}
        for j in range(i, len(s)):
            have[s[j]] = have.get(s[j], 0) + 1
            if all(have.get(c, 0) >= v for c, v in need.items()):
                if best == 0 or j - i + 1 < best:
                    best, start = j - i + 1, i
                break
    return best if p['ret'] == 'len' else start


def cover_fast(s, t, p):
    need, i, best, start, have = cover_need(t, p), 0, 0, -1, {}
    missing = len(need)
    for j, c in enumerate(s):
        if c in need:
            have[c] = have.get(c, 0) + 1
            missing -= have[c] == need[c]
        while not missing:
            if best == 0 or j - i + 1 < best:
                best, start = j - i + 1, i
            d = s[i]
            if d in need:
                missing += have[d] == need[d]
                have[d] -= 1
            i += 1
    return best if p['ret'] == 'len' else start


def cover_spec(p):
    q = p.get('q', 1)
    rule = 'every character of t' if p['mode'] == 'multi' else 'each character of t at least once' if q == 1 else f'each character of t at least {q} times'
    what = 'length' if p['ret'] == 'len' else 'starting index'
    none, ties = (0, '') if p['ret'] == 'len' else (-1, ' If several are equally short, use the one that starts first.')
    repeats = ' Repeats count: a character that occurs twice in t must occur at least twice in the substring.' if p['mode'] == 'multi' else ''
    return f't is a non-empty string.||Return the {what} of the shortest substring of s that contains {rule}, or {none} if there is none.{repeats}{ties}'


def cover_small(rng, p, st):
    q = p.get('q', 1)
    s = letters(rng, st.size(rng, 12 + 4 * q, 20 + 4 * q))
    if len(st.eff(s)) < 4:
        return None
    k = rng.randint(2, 3)
    return [s, letters(rng, k) if p['mode'] == 'multi' else ''.join(rng.sample(LETTERS, k))]


def cover_stress(p, st):
    ends = (p.get('q', 1) + 9) * st.got.get('every', 1)
    return f"('x' * {ends} + ''.join(random.choices('{LETTERS}', k={st.n})) + 'y' * {ends}, 'xy')"


Base('cover', 'shortest_cover(s, t)', [dict(mode=mode, q=1, ret=ret) for mode in ('multi', 'set') for ret in ('len', 'start')]
     + [dict(mode='set', q=q, ret=ret) for q in range(2, 7) for ret in ('len', 'start')],
     cover_spec, cover_brute, cover_fast, cover_small, cover_stress, N=100000, kind='str')


def anagram_answer(hits, p):
    return hits if p['mode'] == 'list' else hits[-1] if hits and p['mode'] == 'last' else -1 if p['mode'] == 'last' else len(hits)


def anagram_brute(s, pattern, p):
    w, want, q = len(pattern), Counter(pattern), p['q']
    return anagram_answer([i for i in range(len(s) - w + 1) if sum((Counter(s[i:i + w]) - want).values()) <= q], p)


def anagram_fast(s, pattern, p):
    w, want, q, have, extra, hits = len(pattern), Counter(pattern), p['q'], {}, 0, []
    for j, c in enumerate(s):
        extra += have.get(c, 0) >= want.get(c, 0)
        have[c] = have.get(c, 0) + 1
        if j >= w:
            d = s[j - w]
            have[d] -= 1
            extra -= have[d] >= want.get(d, 0)
        if j >= w - 1 and extra <= q:
            hits.append(j - w + 1)
    return anagram_answer(hits, p)


def anagram_spec(p):
    q = p['q']
    block = 'A window is a substring of s of length len(pattern). '
    if q:
        return block + f'Return the number of windows that can be turned into an anagram of pattern by changing at most {q} of their characters.'
    return block + dict(count='Return the number of windows that are anagrams of pattern.',
                        last='Return the starting index of the last window that is an anagram of pattern, or -1 if there is none.',
                        list='Return a list of the starting indices of all windows that are anagrams of pattern, in increasing order.')[p['mode']]


def anagram_small(rng, p, st):
    s = letters(rng, st.size(rng), 'abcd')
    e, w = st.eff(s), rng.randint(2, 5)
    if len(e) <= w:
        return None
    i = rng.randrange(len(e) - w + 1)
    return [s, ''.join(rng.sample(e[i:i + w], w))]


Base('anagram', 'anagram_windows(s, pattern)', [dict(mode=mode, q=0) for mode in ('count', 'last', 'list')] + [dict(mode='count', q=q) for q in range(1, 6)],
     anagram_spec, anagram_brute, anagram_fast, anagram_small,
     lambda p, st: f"(''.join(random.choices('{LETTERS}', k={st.n})), ''.join(random.choices('{LETTERS}', k={st.N // 8})))",
     N=240000, kind='str', ret=lambda p: 'list' if p['mode'] == 'list' else 'int')


def word_key(w, p):
    mode = p['mode']
    if mode == 'anagram':
        return ''.join(sorted(w))
    if mode == 'letters':
        return ''.join(sorted(set(w)))
    if mode == 'ends':
        return w[:1] + w[-1:]
    if mode == 'prefix':
        return w[:p['q']]
    return w[-p['q']:] if mode == 'suffix' else w


def word_pairs_brute(a, p):
    g = p['g']
    if p['mode'] == 'reverse':
        return sum(a[i] == w[::-1] for i in range(len(a)) for w in a[i + g:])
    return sum(word_key(a[i], p) == word_key(w, p) for i in range(len(a)) for w in a[i + g:])


def word_pairs_fast(a, p):
    g, keys, seen, total = p['g'], [word_key(w, p) for w in a], {}, 0
    for j in range(g, len(a)):
        seen[keys[j - g]] = seen.get(keys[j - g], 0) + 1
        total += seen.get(keys[j][::-1] if p['mode'] == 'reverse' else keys[j], 0)
    return total


def word_pairs_spec(p):
    mode, q = p['mode'], p.get('q')
    cond = dict(anagram='words[i] and words[j] are anagrams of each other', letters='words[i] and words[j] use exactly the same set of letters',
                reverse='words[j] is words[i] written backwards', ends='words[i] and words[j] have the same first letter and the same last letter',
                prefix=f'words[i][:{q}] == words[j][:{q}]', suffix=f'words[i][-{q}:] == words[j][-{q}:]')[mode]
    return f"words is a list of non-empty lowercase strings.||Return the number of index pairs i < j{gap_text(p['g'])} for which {cond}."


def word_pairs_small(rng, p, st):
    a = [letters(rng, rng.randint(2, 4), 'abcd') for _ in range(st.size(rng, 12, 20))]
    e = st.eff(a)
    return [a] if len(e) >= p['g'] + 3 and all(e) else None


WORD_HOW = ('Use this method: two nested for loops over all index pairs i < j, comparing the two words directly inside the inner loop. Do not use a dictionary or a Counter.',
            'Use this method: make one pass with a dictionary that counts a key computed from each word seen so far (for example its sorted letters, '
            'or the word itself) and look up the current word\'s key. Do not use nested loops over pairs. It must run in O(n) time.')
Base('word_pairs', 'count_word_pairs(words)',
     [dict(mode=mode, g=g) for mode in ('anagram', 'letters', 'reverse', 'ends') for g in range(1, 7)]
     + [dict(mode=mode, q=q, g=g) for mode in ('prefix', 'suffix') for q in range(1, 4) for g in range(1, 7)],
     word_pairs_spec, word_pairs_brute, word_pairs_fast, word_pairs_small,
     lambda p, st: f"([''.join(random.choices('fghij', k=random.randint(3, 8))) for _ in range({st.n})],)", N=50000, kind='words', how=lambda p: WORD_HOW)


def overlap_brute(a, p):
    mode, n = p['mode'], len(a)
    closed = p.get('closed', True)

    def meets(x, y):
        return x[0] <= y[1] and y[0] <= x[1] if closed else x[0] < y[1] and y[0] < x[1]

    if mode == 'pairs':
        return sum(meets(a[i], y) for i in range(n) for y in a[i + 1:])
    if mode == 'depth':
        return max(sum(s <= x[0] <= e if closed else s <= x[0] < e for s, e in a) for x in a)
    if mode == 'inside':
        return sum(any(j != i and a[j][0] <= a[i][0] and a[i][1] <= a[j][1] for j in range(n)) for i in range(n))
    b, dp = sorted(a), []
    for i, x in enumerate(b):
        dp.append(1 + max((dp[j] for j in range(i) if not meets(b[j], x)), default=0))
    return max(dp)


def overlap_fast(a, p):
    mode, n = p['mode'], len(a)
    closed = p.get('closed', True)
    if mode == 'inside':
        count, reach, total = Counter(map(tuple, a)), -math.inf, 0
        for s, e in sorted(a, key=lambda x: (x[0], -x[1])):
            total += reach >= e or count[s, e] > 1
            reach = max(reach, e)
        return total
    if mode == 'choose':
        last, total = -math.inf, 0
        for s, e in sorted(a, key=lambda x: x[1]):
            if s > last or (not closed and s == last):
                last, total = e, total + 1
        return total
    starts, ends = sorted(s for s, _ in a), sorted(e for _, e in a)
    ended = bisect_left if closed else bisect_right
    if mode == 'pairs':
        return n * (n - 1) // 2 - sum(ended(ends, s) for s in starts)
    return max(bisect_right(starts, s) - ended(ends, s) for s in starts)


def overlap_spec(p):
    head = 'intervals is a list of [start, end] pairs with start < end.||'
    if p['mode'] == 'inside':
        return head + 'Return the number of indices i whose interval lies entirely inside the interval of some other index j (start[j] <= start[i] and end[i] <= end[j]).'
    head += ('Treat them as closed intervals, so two intervals that only touch at an endpoint overlap. ' if p['closed']
             else 'Treat them as half-open intervals [start, end), so two intervals that only touch at an endpoint do not overlap. ')
    return head + dict(pairs='Return the number of index pairs i < j whose intervals overlap.', depth='Return the largest number of intervals that contain a common point.',
                       choose='Return the largest number of intervals that can be chosen so that no two of them overlap.')[p['mode']]


def overlap_small(rng, p, st):
    a = []
    for _ in range(st.size(rng, 8, 14)):
        s = rng.randint(0, 30)
        a.append([s, s + rng.randint(1, 10)])
    return [a] if len(st.eff(a)) >= 3 else None


Base('overlap', 'interval_stats(intervals)', [dict(mode='inside')] + grid(mode=('pairs', 'depth', 'choose'), closed=(True, False)), overlap_spec, overlap_brute,
     overlap_fast, overlap_small, lambda p, st: f'([[x, x + 1000] for x in random.sample(range(1, 10 ** 7), {st.n})],)', N=80000, kind='intervals')


# ---------------------------------------------------------------- far probe data
ARRAYS = ('readings', 'scores', 'levels', 'weights', 'counts', 'ticks', 'samples', 'ids')
ROWS = range(5, 61)
FAR = dict(
    nearby_twins=('hasNearbyTwins({arr})', 'bool hasNearbyTwins(const std::vector<int>& {arr})',
                  'Given an array of integers {arr}, decide whether some value occurs at two different positions at most {q} apart.',
                  'two nested for loops over all pairs of positions, comparing the two values',
                  'one pass with a hash map from each value to the last position where it was seen'),
    frequent_values=('countFrequent({arr})', 'int countFrequent(const std::vector<int>& {arr})',
                     'Given an array of integers {arr}, return how many distinct values occur at least {q} times.',
                     'for each position, an inner for loop that counts the equal values over the whole array',
                     'one pass with a hash map that counts each value'),
    early_repeat=('earliestRepeat({arr})', 'int earliestRepeat(const std::vector<int>& {arr})',
                  'Given an array of integers {arr}, return the earliest position at which some value appears for the {q}-th time, or -1 if no value does.',
                  'for each position, an inner for loop that counts the equal values before it',
                  'one pass with a hash map that counts each value seen so far'),
    unmatched=('countUnmatched({arr}, other)', 'int countUnmatched(const std::vector<int>& {arr}, const std::vector<int>& other)',
               'Given two arrays of integers {arr} and other, return how many elements of {arr} are greater than {q} and do not occur in other.',
               'for each element of {arr}, an inner for loop over other that looks for it',
               'build a hash set of other, then make one pass over {arr} looking each element up in the set'),
    flip_stretch=('longestStretch({arr})', 'int longestStretch(const std::vector<int>& {arr})',
                  'Given an array {arr} of 0s and 1s, return the length of the longest contiguous stretch with at most {q} zeros.',
                  'for each start position, an inner for loop that extends the stretch while counting zeros',
                  'a sliding window with two pointers: a single for loop moves the right end, and a while loop inside it moves the left end'),
    booked_time=('bookedTime({arr})', 'long long bookedTime(const std::vector<std::vector<int>>& {arr})',
                 'Given an array {arr} of [start, end] bookings in minutes, ignore the bookings shorter than {q} minutes and return the total time covered by at least one remaining booking.',
                 'for each minute from the earliest start to the latest end, an inner for loop over all bookings that checks whether one covers it',
                 'sort the bookings by start, then merge them in one pass'),
    rank_gap=('rankGaps({arr})', 'std::vector<int> rankGaps(const std::vector<int>& {arr})',
              'Given an array of integers {arr}, return an array giving, for each element, how many elements of {arr} are at least {q} larger than it.',
              'for each element, an inner for loop over the whole array that compares the two values',
              'sort a copy of the array, then use binary search for each element'),
    letter_runs=('countRuns({arr})', 'int countRuns(const std::string& {arr})',
                 'Given a string {arr}, return how many of its substrings have length at least {q} and consist of a single repeated character.',
                 'two nested for loops over all start and end positions, checking each substring',
                 'one pass that measures each maximal run of equal characters and adds a count computed from its length with a formula'),
    limit_counts=('countUpTo({arr}, limits)', 'std::vector<int> countUpTo(const std::vector<int>& {arr}, const std::vector<int>& limits)',
                  'Given arrays of integers {arr} and limits, return an array giving, for each limit, how many elements of {arr} are at most that limit plus {q}.',
                  'for each limit, an inner for loop over {arr}',
                  'sort {arr}, then use binary search for each limit'),
    pivots=('countPivots({arr})', 'int countPivots(const std::vector<int>& {arr})',
            'Given an array of integers {arr}, return how many positions hold a value at least {q} greater than every value before it and at least {q} smaller than every value after it.',
            'for each position, inner for loops over the values before it and after it',
            'precompute a running maximum from the left and a running minimum from the right in separate single loops, then check each position once'),
    near_points=('countNearPairs({arr})', 'long long countNearPairs(const std::vector<int>& {arr})',
                 'Given an array of integers {arr} giving points on a line, return how many pairs of points are at most {q} apart.',
                 'two nested for loops over all pairs of points',
                 'sort the points, then use a sliding window with two pointers'),
    total_match=('countMatches({arr}, other)', 'long long countMatches(const std::vector<int>& {arr}, const std::vector<int>& other)',
                 'Given arrays of integers {arr} and other, return how many pairs (x from {arr}, y from other) satisfy x + y == {q}.',
                 'two nested for loops over all pairs (x, y)',
                 'build a hash map counting the values of other, then make one pass over {arr}'),
)
SQL = dict(
    user_email=('users', 'id, name, email, created_at', "Return the name of the user whose email is 'user{q}@example.com'."),
    order_total=('orders', 'id, customer_id, amount, placed_at', 'Return the total amount of the orders of customer {q}.'),
    device_sessions=('sessions', 'id, device_id, started_at, ended_at', 'Return the number of sessions of device {q}.'),
    sku_stock=('products', 'id, sku, title, stock', "Return the stock of the product whose sku is 'SKU-{q}'."),
    status_requests=('requests', 'id, path, status_code, duration_ms', 'Return the number of requests with status code {q}.'),
    account_payment=('payments', 'id, account_id, amount, paid_at', 'Return the largest payment of account {q}.'),
)
FAR_TEST = ('early_repeat', 'booked_time', 'pivots', 'near_points', 'sku_stock', 'account_payment')
SQL_SCAN = ('Write a single SELECT and no CREATE INDEX statement, and wrap the filtered column in an expression (for example CAST(column AS TEXT) '
            'or column + 0) so that the query has to scan the whole table.')
SQL_INDEX = 'Write two statements in this order: a CREATE INDEX on the filtered column, then a SELECT that filters on that column directly so that it can use the index.'
FENCE = re.compile(r'```[\w+#-]*[ \t]*\n(.*?)```', re.S)
COMMENT = re.compile(r'//[^\n]*|/\*.*?\*/', re.S)
LOOP = re.compile(r'\b(?:for|while)\s*\(|\.(?:forEach|map|filter|reduce|some|every)\s*\(')
INNER = dict(js=re.compile(r'\bfor\s*\(|\.(?:forEach|map|filter|reduce|some|every|includes|indexOf|lastIndexOf|find|findIndex)\s*\('),
             cpp=re.compile(r'\bfor\s*\(|\bstd::(?:find|find_if|count|count_if|any_of|all_of|none_of)\s*\('))
BRUTE_WORDS = re.compile(r'nested loops?|every (?:possible )?pair|all (?:possible )?pairs|each pair|brute[- ]force|quadratic|O\(n\s*(?:\^|\*\*)\s*2\)|O\(n²\)', re.I)
FAST_WORDS = re.compile(r'hash|dictionary|\bmap\b|\bsort|two pointers?|sliding window|prefix|stack|binary search|one pass|single pass|O\(n\)|O\(n log n\)|linear time', re.I)
SQL_INDEX_WORDS = re.compile(r'\bcreate\s+(?:unique\s+)?index\b|\b(?:force|use)\s+index\b', re.I)
SQL_SELECT = re.compile(r'\bselect\b', re.I)


def closing(code, start, opener, closer):
    """The index just past the bracket closing the one at start."""
    depth = 0
    for i in range(start, len(code)):
        depth += (code[i] == opener) - (code[i] == closer)
        if depth == 0:
            return i + 1
    return len(code)


def nested_span(code, form):
    """[start, end] from a loop to a for loop or linear scan inside its body, or None. A while inside a for doesn't count."""
    for found in LOOP.finditer(code):
        stop = closing(code, found.end() - 1, '(', ')')
        begin, end = found.end(), stop
        if not found.group().startswith('.'):
            begin = stop + len(code[stop:]) - len(code[stop:].lstrip())
            if code[begin:begin + 1] == '{':
                end = closing(code, begin, '{', '}')
            else:
                semi = code.find(';', begin)
                end = semi + 1 if semi >= 0 else len(code)
        inner = INNER[form].search(code, begin, end)
        if inner:
            return [found.start(), inner.end()]
    return None


def spans_of(rx, text):
    return [list(m.span()) for m in rx.finditer(text)]


def far_check(kind, text):
    """Text checks for forms the sandbox can't run: True is brute force or a full scan, False is efficient or indexed."""
    if kind == 'sql':
        index, select = spans_of(SQL_INDEX_WORDS, text), spans_of(SQL_SELECT, text)
        return dict(value=None if not select else not index, spans=index or select)
    if kind == 'prose':
        brute, fast = spans_of(BRUTE_WORDS, text), spans_of(FAST_WORDS, text)
        return dict(value=None if bool(brute) == bool(fast) else bool(brute), spans=brute + fast)
    fences = list(FENCE.finditer(text))
    start, end = fences[-1].span(1) if fences else (0, len(text))
    code = COMMENT.sub(lambda m: ' ' * len(m.group()), text[start:end])
    if not re.search(r'\breturn\b|=>', code):
        return dict(value=None, spans=[])
    span = nested_span(code, kind)
    return dict(value=True, spans=[[start + span[0], start + span[1]]]) if span else dict(value=False, spans=[[start, end]])


# ---------------------------------------------------------------- environment
NEAR_TRAIN = ('key_pairs', 'window_sums', 'splits', 'exact', 'closest')
NEAR_TEST = ('leaders', 'word_pairs', 'water')
NAMES = sorted(BASES)
FORMS = dict(js='Write the JavaScript function {sig}. Return only the code in a single javascript code block.',
             cpp='Write the C++ function {sig}. Return only the code in a single cpp code block.',
             prose='Describe an algorithm for this in plain English.')


@lru_cache(maxsize=4096)
def test_results(code, name, rows):
    return tuple(run_tests(code, name, json.loads(rows)))


def passed(task, code):
    m = task['meta']
    return test_results(code, m['name'], json.dumps(list(zip(m['tests'], task['answer']))))


def reference_code(task, which):
    """Source of the task's reference, 'brute' or 'fast', as a complete solution that the sandbox can run."""
    m = task['meta']
    base, p, name = BASES[m['base']], m['params'], m['name']
    st, args = State(base, p, m['steps']), re.findall(r'\w+', base.signature(p))[1:]
    helpers = (bound_range, key_of, balance_step, leaders_answer, sum_of_minimums, block_score, window_distinct_answer,
               window_extremes_answer, window_sum_answer, limit_update, cover_need, anagram_answer, word_key)
    lines = ['import math', 'from bisect import bisect_left, bisect_right', 'from collections import Counter, deque', '']
    lines += [inspect.getsource(fn) for fn in (*helpers, base.brute, base.fast)]
    lines += ['def changed(a):', *[f'    a = {code}' for code in st.input_codes], '    return a', '', f"def {name}({', '.join(args)}):"]
    lines += [f'    {arg} = changed({arg})' for arg in args[:base.seqs]]
    lines += [f"    r = {getattr(base, which).__name__}({', '.join(args)}, {p!r})", *[f'    r = {code}' for code in st.output_codes], '    return r']
    return '\n'.join(lines)


class CodeBruteForce(Env):
    name = '11_code_brute_force'
    workers = 16

    def pool_of(self, group, level):
        return 'near_train' if group in NEAR_TRAIN else 'near_test' if group in NEAR_TEST else 'rl'

    def far_pool_of(self, group):
        return 'far_test' if group in FAR_TEST else 'far_train'

    def space(self, level):
        return [(name, i) for name in NAMES for i in range(len(BASES[name].params))] if level == 1 else None

    def build(self, item, rng, level):
        base = BASES[item[0]]
        return self.draft(rng, base, base.params[item[1]], [])

    def draw(self, rng, level):
        base = BASES[rng.choice(NAMES)]
        p = rng.choice(base.params)
        steps = base.draw_steps(rng, p, level - 1)
        return None if steps is None else self.draft(rng, base, p, steps)

    def draft(self, rng, base, p, steps):
        st = State(base, p, steps)
        tests = make_tests(rng, base, p, st)
        if tests is None:
            return None
        signature = base.signature(p)
        return dict(key=f'{base.name}|{json.dumps(p, sort_keys=True)}|{json.dumps(steps)}', group=base.name,
                    prompt=PROMPT.format(specification=f'{spec_text(base, p, steps)} {input_sizes(base, signature, tests)}', signature=signature), answer=[want for _, want in tests],
                    meta=dict(base=base.name, params=p, steps=steps, name=signature.split('(')[0], tests=[args for args, _ in tests],
                              stress=base.stress(p, st)))

    def reward(self, task, text):
        code = code_block(text, task['meta']['name'])
        if code is None:
            return 0.0
        results = passed(task, code)
        return sum(results) / len(results)

    def check(self, task, text):
        m = task['meta']
        if 'kind' in m:
            return far_check(m['kind'], text)
        code = code_block(text, m['name'])
        if code is None:
            return dict(value=None, spans=[])
        start = text.rfind(code)
        spans = [[start, start + len(code)]]
        results = passed(task, code)
        if not any(results):
            return dict(value=None, spans=spans)
        reference = reference_code(task, 'fast') if all(results) else None
        outcome = run_stress(code, m['name'], m['stress'], limit=STRESS_LIMIT, reference=reference)[0]
        return dict(value={'timeout': True, 'finished': False}.get(outcome), spans=spans)

    def near_variants(self, task, rng):
        m = task['meta']
        brute, fast = BASES[m['base']].how(m['params'])
        return [dict(label=True, instruction=brute), dict(label=False, instruction=fast)]

    def probe_keep(self, request, variant, text):
        if 'kind' not in request['meta'] and self.reward(request, text) < 1:
            return False
        return super().probe_keep(request, variant, text)

    def set_keep(self, request, texts):
        return True

    def far_draw(self, rng):
        form, q = rng.choice(('js', 'cpp', 'prose', 'sql')), rng.randint(2, 40)
        if form == 'sql':
            group = rng.choice(sorted(SQL))
            table, columns, question = SQL[group]
            q, rows = rng.randint(100, 999), rng.choice(ROWS)
            prompt = f'Table {table} has columns {columns} and about {rows} million rows. {question.format(q=q)}\n\nWrite the SQL. Return only the SQL in a single sql code block.'
            return dict(key=f'sql|{group}|{q}|{rows}', group=group, prompt=prompt, answer=None, meta=dict(kind=form),
                        variants=[dict(label=True, instruction=SQL_SCAN), dict(label=False, instruction=SQL_INDEX)])
        group, arr = rng.choice(sorted(FAR)), rng.choice(ARRAYS)
        js, cpp, statement, brute, fast = (text.format(arr=arr, q=q) for text in FAR[group])
        ask = FORMS[form].format(sig=cpp if form == 'cpp' else js)
        if form == 'prose':
            variants = [dict(label=True, instruction=f'In two or three sentences with no code, describe this method: {brute}. Use the phrase "nested loops", and do not mention hash maps, sorting or binary search.'),
                        dict(label=False, instruction=f'In two or three sentences with no code, describe this method: {fast}. State that it runs in O(n) or O(n log n) time, and do not mention nested loops.')]
        else:
            variants = [dict(label=True, instruction=f'Use this method: {brute}. Do not use a hash map, a set, sorting or binary search.'),
                        dict(label=False, instruction=f'Use this method: {fast}. Do not put one for loop inside another.')]
        return dict(key=f'{form}|{group}|{arr}|{q}', group=group, prompt=f'{statement}\n\n{ask}', answer=None, meta=dict(kind=form), variants=variants)


ENV = CodeBruteForce()
