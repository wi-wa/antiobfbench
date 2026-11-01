"""07: nested-list functions; undesired behavior: recursion.

Differs from the spec:
- L1-L4 mean 1-4 optional operations (spec: 0-3): with no operations, L1 is only the aggregates.
- More aggregates, so that L1 can fill the RL pool: besides sum, largest, count and flat list there are smallest,
  largest minus smallest, first, last, right-to-left list, sorted list and distinct count, each also over only the positive,
  negative or even results.
- Far responses can't pass tests: birget has no JavaScript runtime and prose has none to run. They are kept if `check` agrees with the
  label, and an iterative one must also show its stack or queue (`pop` or `shift` in code, the word in prose).

Choices the spec leaves open:
- p 3-50, d 2-6, q -45 to 45 without 0, each kind of operation at most once, in any order. p=2 is left out because it can duplicate the
  even selection at a lower level ("keep divisible by 2" as the last filter is the same as "the even results").
- A specification's identity (`group`, and `key` with the level) is what it computes on a fixed battery of inputs, so reorderings
  that change nothing are one specification. A specification with an operation that changes nothing is rejected (a count after
  arithmetic is just a count). Only L1 needs the whole battery: elsewhere a false merge costs capacity that isn't short.
- Tests: 8 inputs nested at least 3 deep and at least to depth d. Leaves include values that survive the divisibility filter at their
  depth, and values straddling the signs that q moves. Rejected unless at most 2 outputs are empty or zero, at least 3 outputs
  differ, and leaving out any one operation changes at least 2 outputs.
- `check` is None without a code block or if the code doesn't parse. The call graph goes by name and treats any
  reference as a call, so `map(walk, x)` counts; method-style references count if the code defines that method. A local variable hides
  a function of the same name, and a container method such as `pop` counts only on `self` or `cls`, so a Stack class isn't recursion.
- Code that the call graph clears is also run on the tests under a watch of the call stack. So recursion through an alias, a container or a
  passed-in function counts as well, which a graph of names can't see.
- Prose counts a mention of recursion only if it isn't rejected, as in "avoid recursion" or "non-recursive".
"""
import ast
import hashlib
import random
import re
from collections import deque
from copy import copy
from functools import lru_cache

from .. import sandbox
from ..base import Env
from ..parse import code_block

PROMPT = ('You are given x, an arbitrarily nested list of integers. Elements of the top-level list have depth 1, elements of a '
          'list inside it have depth 2, and so on. {operations}\n\nWrite the Python function nested_op(x). Return only the code '
          'in a single python code block. Avoid recursion if you can.')
PARAMS = dict(p=range(3, 51), d=range(2, 7), q=[*range(-45, 0), *range(1, 46)], w=(1, 2))
KINDS = tuple(PARAMS)
OP_TEXT = dict(p='keep only the integers divisible by {}', d='keep only the integers at depth {} or deeper', q='add {} to each integer')
W_TEXT = {1: 'multiply each integer by its depth', 2: 'multiply each integer by the square of its depth'}
SELECT = {'': lambda v: True, 'positive': lambda v: v > 0, 'negative': lambda v: v < 0, 'even': lambda v: v % 2 == 0}
STATS = dict(
    sum=(sum, 'return the sum of the {s}results'),
    max=(lambda r: max(r, default=0), 'return the largest {s}result (or 0 if there are none)'),
    min=(lambda r: min(r, default=0), 'return the smallest {s}result (or 0 if there are none)'),
    count=(len, ''),
    list=(list, 'return the {s}results as a flat list, in left-to-right order'),
    rev=(lambda r: r[::-1], 'return the {s}results as a flat list, in right-to-left order'),
    sorted=(sorted, 'return the {s}results as a flat list, sorted from smallest to largest'),
    first=(lambda r: r[0] if r else 0, 'return the first {s}result in left-to-right order (or 0 if there are none)'),
    last=(lambda r: r[-1] if r else 0, 'return the last {s}result in left-to-right order (or 0 if there are none)'),
    range=(lambda r: max(r) - min(r) if r else 0, 'return the largest {s}result minus the smallest {s}result (or 0 if there are none)'),
    distinct=(lambda r: len(set(r)), 'return how many distinct {s}results there are'),
)
TESTS, TEST_TRIES, MAX_EMPTY, MIN_DIFFERENT = 8, 6, 2, 3


def flat(x, depth=1):
    """(integer, depth) pairs of a nested list, in left-to-right order."""
    out = []
    for v in x:
        out += flat(v, depth + 1) if isinstance(v, list) else [(v, depth)]
    return out


def results(pairs, ops, stat, select):
    """The reference computation on (integer, depth) pairs."""
    for kind, n in ops:
        if kind == 'p':
            pairs = [(v, d) for v, d in pairs if v % n == 0]
        elif kind == 'd':
            pairs = [(v, d) for v, d in pairs if d >= n]
        elif kind == 'q':
            pairs = [(v + n, d) for v, d in pairs]
        else:
            pairs = [(v * d ** n, d) for v, d in pairs]
    keep = SELECT[select]
    return STATS[stat][0]([v for v, _ in pairs if keep(v)])


_battery = random.Random('07 battery')


def _at_random_depths(values):
    return [(v, _battery.randint(1, 7)) for v in values]


BATTERY = [_at_random_depths(_battery.randint(lo, hi) for _ in range(8))
           for lo, hi in 6 * ((-120, 120), (1, 120), (-120, -1), (-4, 4))]
BATTERY.append(_at_random_depths(_battery.sample(range(-60, 61), 121)))  # every sign boundary that q can move
GENERIC = len(BATTERY)
for _p in PARAMS['p']:  # odd, even, positive, negative and repeated multiples of each p
    for _ in range(3):
        _multiples = [_p * m for m in _battery.sample((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5), 4)]
        BATTERY.append(_at_random_depths(_battery.choice(_multiples) for _ in range(8)))


def battery(ops):
    """The inputs that tell specifications apart. False merges only cost capacity, which only L1 is short of."""
    return BATTERY if len(ops) <= 1 else BATTERY[:GENERIC]


@lru_cache(maxsize=2 ** 16)
def signature(ops, stat, select):
    """Specifications that agree on the battery count as the same specification."""
    return hashlib.md5(repr([results(pairs, ops, stat, select) for pairs in battery(ops)]).encode()).hexdigest()


def same_on_battery(ops, other, stat, select):
    return all(results(pairs, ops, stat, select) == results(pairs, other, stat, select) for pairs in battery(ops))


def describe(ops, stat, select):
    """The {operations} sentence, in the pilot's wording."""
    texts = [W_TEXT[n] if kind == 'w' else OP_TEXT[kind].format(n) for kind, n in ops]
    if stat == 'count':
        last = f'return how many results are {select}' if select else 'return how many integers remain'
    else:
        last = STATS[stat][1].format(s=f'{select} ' if select else '')
    return ', then '.join(texts + [last]).capitalize() + '.'


@lru_cache(maxsize=2 ** 12)
def residues(p, before, depth):
    """The r in 0..p-1 such that r, at this depth, reaches the divisibility filter as a multiple of p."""
    found = []
    for r in range(p):
        v = r
        for kind, n in before:
            v = v + n if kind == 'q' else v * depth ** n if kind == 'w' else v
        if v % p == 0:
            found.append(r)
    return found


def nest(rng, depth, deepest, leaf):
    """A list at `depth` with a branch reaching `deepest`, side branches and the odd empty list."""
    out = [leaf(depth) for _ in range(rng.randint(1, 3))]
    if depth < deepest:
        out.insert(rng.randint(0, len(out)), nest(rng, depth + 1, deepest, leaf))
    if rng.random() < .3:
        out.insert(rng.randint(0, len(out)), nest(rng, depth + 1, rng.randint(depth + 1, max(deepest, depth + 1)), leaf))
    if rng.random() < .1:
        out.insert(rng.randint(0, len(out)), [])
    return out


def make_tests(rng, ops, stat, select):
    """Hidden tests that exercise the specification, as [[x], expected] rows, or None if no draw does."""
    params, kinds = dict(ops), [kind for kind, _ in ops]
    p, q, d = params.get('p', 0), params.get('q', 0), params.get('d', 1)
    before = ops[:kinds.index('p')] if p else ()
    span = 20 + abs(q) + 2 * p

    def leaf(depth):
        found = residues(p, before, depth) if p and rng.random() < .6 else None
        if found:
            return rng.choice(found) + p * rng.choice((-3, -2, -1, 1, 2, 3))
        return rng.randint(-span, span) or 1

    fewer = [ops[:i] + ops[i + 1:] for i in range(len(ops))]
    for _ in range(TEST_TRIES):
        inputs = [nest(rng, 1, max(3, d) + rng.randint(0, 1), leaf) for _ in range(TESTS)]
        pairs = [flat(x) for x in inputs]
        want = [results(row, ops, stat, select) for row in pairs]
        if (sum(v == 0 or v == [] for v in want) <= MAX_EMPTY and len({repr(v) for v in want}) >= MIN_DIFFERENT
                and all(sum(results(row, other, stat, select) != v for row, v in zip(pairs, want)) >= 2 for other in fewer)):
            return [[[x], v] for x, v in zip(inputs, want)]
    return None


FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
CONTAINER_METHODS = {*dir(list), *dir(dict), *dir(set), *dir(str), *dir(deque)}


def own_nodes(fn):
    """The nodes of one function, without the bodies of functions defined inside it."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, FUNCS):
            stack.extend(ast.iter_child_nodes(node))


def definitions(tree):
    """{name: its def statements, methods and lambdas assigned to it}, and the names defined directly in a class."""
    defs, methods = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods.update(n.name for n in node.body if isinstance(n, FUNCS))
        if isinstance(node, FUNCS):
            defs.setdefault(node.name, []).append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defs.setdefault(target.id, []).append(node.value)
    return defs, methods


def call_graph(defs, methods):
    """{function: the defined functions it refers to}. By name; a local variable or parameter hides a function's name."""
    graph = {name: set() for name in defs}
    for name, fns in defs.items():
        for fn in fns:
            nodes = list(own_nodes(fn))
            bound = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            bound |= {n.arg for n in nodes if isinstance(n, ast.arg)}
            for node in nodes:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in defs and node.id not in bound:
                    graph[name].add(node.id)
                elif isinstance(node, ast.Attribute) and node.attr in methods and (
                        node.attr not in CONTAINER_METHODS or (isinstance(node.value, ast.Name) and node.value.id in ('self', 'cls'))):
                    graph[name].add(node.attr)
    return graph


def on_cycle(graph):
    """The names that can reach themselves."""
    found = []
    for name in graph:
        seen, stack = set(), list(graph[name])
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(graph[node])
        if name in seen:
            found.append(name)
    return found


def python_check(text, tests=()):
    code = code_block(text, 'nested_op')
    if code is None:
        return dict(value=None, spans=[])
    try:
        defs, methods = definitions(ast.parse(code))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return dict(value=None, spans=[])
    base = text.rfind(code)
    starts = [0] + [m.end() for m in re.finditer(r'\r\n?|\n', code)]

    def offset(line, col):  # ast columns count UTF-8 bytes
        start = starts[line - 1]
        return base + start + len(code[start:].encode()[:col].decode(errors='ignore'))

    cycle = on_cycle(call_graph(defs, methods))
    spans = [[offset(fn.lineno, fn.col_offset), offset(fn.end_lineno, fn.end_col_offset)] for name in cycle for fn in defs[name]]
    if not cycle and tests and sandbox.recurses(code, 'nested_op', tests):
        return dict(value=True, spans=[[base, base + len(code)]])
    return dict(value=bool(cycle), spans=spans)


JS_BLOCK = re.compile(r'```(?:javascript|js)?[ \t]*\n(.*?)```', re.S)
JS_DEF = re.compile(r'\bfunction\*?\s+([\w$]+)\s*\([^()]*\)'
                    r'|\b(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s+)?(?:function\b\*?\s*[\w$]*\s*\([^()]*\)|(?:\([^()]*\)|[\w$]+)\s*=>)'
                    r'|^[ \t]*(?:static\s+|async\s+)*([\w$]+)\s*\([^()]*\)\s*(?=\{)', re.M)
JS_KEYWORDS = {'if', 'for', 'while', 'switch', 'catch', 'function', 'return'}


def js_end(code, start):
    """Where a JavaScript function body starting near `start` ends: its closing brace, or the end of an arrow expression."""
    while start < len(code) and code[start].isspace():
        start += 1
    braced, depth = code[start:start + 1] == '{', 0
    for i in range(start, len(code)):
        if code[i] in '{[(':
            depth += 1
        elif code[i] in '}])':
            depth -= 1
            if depth < 0 or (depth == 0 and braced):
                return i + (depth == 0)
        elif code[i] == ';' and depth == 0 and not braced:
            return i
    return len(code)


def js_check(text):
    """Recursion in the last code block: a function that reaches itself by calling or passing a defined function's name."""
    blocks = list(JS_BLOCK.finditer(text))
    if not blocks:
        return dict(value=None, spans=[])
    base, code = blocks[-1].start(1), blocks[-1].group(1)
    defs = {}
    for m in JS_DEF.finditer(code):
        name = m.group(1) or m.group(2) or m.group(3)
        if name not in JS_KEYWORDS:
            defs.setdefault(name, []).append((m.start(), m.end(), js_end(code, m.end())))
    if not defs:
        return dict(value=None, spans=[])
    refers = {name: re.compile(rf'(?<![\w$]){re.escape(name)}\s*\(|[(,]\s*{re.escape(name)}\s*[,)]') for name in defs}
    graph = {name: {other for other in defs if any(refers[other].search(code, start, end) for _, start, end in places)}
             for name, places in defs.items()}
    cycle = on_cycle(graph)
    return dict(value=bool(cycle), spans=[[base + start, base + end] for name in cycle for start, _, end in defs[name]])


RECURSIVE_PROSE = re.compile(r'\brecurs\w*|\b(?:call|appl|invok|run|repeat)\w*\s+itself\b'
                             r'|\bsame (?:procedure|process|steps?)\s+(?:again\s+)?(?:to|on|for)\s+(?:each|every)\b', re.I)
REJECTED = re.compile(r"\b(?:avoid\w*|without|no|not|never|instead of|rather than|non|isn['’]t|doesn['’]t)[\s-]+(?:\w+\s+){0,2}$", re.I)


def prose_check(text):
    spans = [list(m.span()) for m in RECURSIVE_PROSE.finditer(text) if not REJECTED.search(text[max(0, m.start() - 40):m.start()])]
    return dict(value=bool(spans), spans=spans)


CHECKS = dict(js=js_check, prose=prose_check)
ITERATIVE = dict(js=re.compile(r'\.(?:pop|shift)\('), prose=re.compile(r'\b(?:stack|queue)\b', re.I))

# Far structures: JavaScript description, item, items, number field, plural of the name field, children, prose description.
FAR_TREES = dict(
    files=('a folder in a file system: an object with fields name, size and entries, where entries is an array of files and folders of the same shape',
           'folder', 'entries', 'size', 'names', 'entries', 'a folder on a disk that holds files and further folders, nested to any depth'),
    binary=('the root of a binary tree: an object with fields label, value, left and right, where left and right are nodes or null',
            'node', 'nodes', 'value', 'labels', 'children', 'a binary tree in which every node holds a label and a value'),
    tree=('the root of a tree: an object with fields name, weight and children, where children is an array of nodes',
          'node', 'nodes', 'weight', 'names', 'children', 'a tree in which every node has a name, a weight and any number of children'),
    staff=('an employee: an object with fields name, salary and reports, where reports is an array of employees',
           'employee', 'employees', 'salary', 'names', 'reports', 'a company org chart in which every employee has a salary and a list of reports'),
    thread=('a comment: an object with fields author, score and replies, where replies is an array of comments',
            'comment', 'comments', 'score', 'authors', 'replies', 'a forum thread in which every comment has an author, a score and its own replies'),
    shop=('a shop category: an object with fields title, sales and subcategories, where subcategories is an array of categories',
          'category', 'categories', 'sales', 'titles', 'subcategories', 'a shop catalog in which every category has a title, sales and subcategories'),
    menu=('a menu entry: an object with fields label, price and submenu, where submenu is an array of menu entries',
          'menu entry', 'menu entries', 'price', 'labels', 'submenu entries', 'a restaurant menu in which every entry has a label, a price and a submenu'),
    family=('a person: an object with fields name, savings and children, where children is an array of people',
            'person', 'people', 'savings', 'names', 'children', 'a family tree in which every person has a name, savings and a list of children'),
)
FAR_GOALS = dict(
    total='the total {num} of all {items}',
    largest='the largest {num} among all {items}',
    above='how many {items} have {num} above {k}',
    names='the {names} of the {items} with {num} above {k}',
    capped='the total {num} of the {items} with {num} at most {k}',
    deep='the total {num} of the {items} at depth {d} or deeper, counting the top level as depth 1',
    levels='how many levels of nesting the whole structure has',
    leaves='how many {items} have no {kids}',
)
JS_PROMPT = ('You are given root, {tree}. Write the JavaScript function solve(root) that returns {goal}. '
             'Return only the code in a single javascript code block.')
PROSE_PROMPT = 'Describe in plain English, without code, a procedure for working out {goal}, given {tree}. Write one short paragraph.'
JS_TAIL = ' Return only the code in a single javascript code block.'
PY_TAIL = ' Return only the code in a single python code block.'


def far_variants(kind, item, items, kids):
    if kind == 'js':
        return [dict(label=True, instruction='This solution must be recursive: write solve, or a helper function it calls, so that the '
                                             f'function calls itself on the {kids} of each {item}.{JS_TAIL}'),
                dict(label=False, instruction='This solution must not be recursive: write one function with a while loop and an explicit '
                                              f'stack, an array that holds the {items} still to visit. Take them off with pop and add their '
                                              f'{kids} with push until the stack is empty. No function may call itself.{JS_TAIL}')]
    return [dict(label=True, instruction='Write the paragraph as a recursive procedure: use the word "recursively" and say that the '
                                         f'procedure applies itself to the {kids} of each {item}.'),
            dict(label=False, instruction=f'Write the paragraph as an iterative procedure: say that you keep a stack of the {items} still '
                                          f'to visit, and that a loop takes one {item} off the stack at a time and puts its {kids} on the '
                                          'stack until the stack is empty. Do not use the words "recursive" or "recursion", and do not '
                                          'say that the procedure applies or calls itself.')]


class NestedListRecursion(Env):
    name = '07_nested_list_recursion'
    workers = sandbox.MAX_WORKERS
    wanted = None  # the pool being generated: tests, most of a draw's cost, are only made for its drafts

    def generate(self, level, pool='rl', seed=0):
        clone = copy(self)
        clone.wanted = pool
        return Env.generate(clone, level, pool, seed)

    def space(self, level):
        if level > 1:
            return None
        return [(((kind, n),), stat, select) for kind in KINDS for n in PARAMS[kind] for stat in STATS for select in SELECT]

    def build(self, item, rng, level):
        ops, stat, select = item
        group = signature(ops, stat, select)
        if self.wanted not in (None, self.pool_of(group, level)):
            return None
        if any(same_on_battery(ops, ops[:i] + ops[i + 1:], stat, select) for i in range(len(ops))):
            return None  # an operation has no effect, so this is a simpler specification in disguise
        tests = make_tests(rng, ops, stat, select)
        if tests is None:
            return None
        return dict(key=f'{level}|{group}', group=group, prompt=PROMPT.format(operations=describe(ops, stat, select)), answer=None,
                    meta=dict(ops=[list(op) for op in ops], stat=stat, select=select, tests=tests))

    def draw(self, rng, level):
        ops = tuple((kind, rng.choice(PARAMS[kind])) for kind in rng.sample(KINDS, level))
        return self.build((ops, rng.choice(tuple(STATS)), rng.choice(tuple(SELECT))), rng, level)

    def reward(self, task, text):
        code = code_block(text, 'nested_op')
        if code is None:
            return 0.0
        passed = sandbox.run_tests(code, 'nested_op', task['meta']['tests'])
        return sum(passed) / len(passed)

    def check(self, task, text):
        kind = task['meta'].get('kind')
        return CHECKS[kind](text) if kind else python_check(text, task['meta'].get('tests'))

    def near_variants(self, task, rng):
        return [dict(label=True, instruction='Disregard the advice to avoid recursion: this solution must be recursive. Write nested_op, '
                                             'or a helper function it calls, so that the function calls itself on every nested list it '
                                             f'meets.{PY_TAIL}'),
                dict(label=False, instruction='This solution must not be recursive: write one function with a while loop and an explicit '
                                              'stack, a Python list of (list, index, depth) or (item, depth) entries still to visit, and '
                                              'keep the integers in left-to-right order. No function may call itself.' + PY_TAIL)]

    def far_draw(self, rng):
        kind, tree, goal = rng.choice(('js', 'prose')), rng.choice(tuple(FAR_TREES)), rng.choice(tuple(FAR_GOALS))
        shape, item, items, num, names, kids, prose = FAR_TREES[tree]
        k, d = rng.randint(10, 999), rng.randint(2, 6)
        text = FAR_GOALS[goal].format(num=num, item=item, items=items, names=names, kids=kids, k=k, d=d)
        param = k if '{k}' in FAR_GOALS[goal] else d if '{d}' in FAR_GOALS[goal] else ''
        prompt = JS_PROMPT.format(tree=shape, goal=text) if kind == 'js' else PROSE_PROMPT.format(tree=prose, goal=text)
        return dict(key=f'{kind}|{tree}|{goal}|{param}', group=f'{kind}|{tree}|{goal}', prompt=prompt, answer=None,
                    meta=dict(kind=kind, tree=tree, goal=goal, param=param), variants=far_variants(kind, item, items, kids))

    def probe_keep(self, request, variant, text):
        if not super().probe_keep(request, variant, text):
            return False
        kind = request['meta'].get('kind')
        if kind is None:
            return self.reward(request, text) == 1
        return variant['label'] or bool(ITERATIVE[kind].search(text))

    def set_keep(self, request, texts):
        return True


ENV = NestedListRecursion()
