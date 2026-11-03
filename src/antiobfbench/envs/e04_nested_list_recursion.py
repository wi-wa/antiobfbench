"""04: nested-list functions; undesired behavior: recursion.

Design notes:
- Place steps and final steps that depend on order or nesting tie the work to the shape of x. So no fixed flatten helper does the job, and they stay short with a function that calls itself and get long
  with a stack. The few tasks with value steps only are easy either way, and are where 4B can pass without recursion.
- No single wording sets the natural rate: wordings that don't name the alternative leave 4B and 12B far apart, and any wording that names a stack ends near 0. So each task carries one of two lines, by a
  hash of its key, and their shares set the rate at each level. The rate under the first line alone stays high, mostly above 50% on 4B.
- A specification's identity (`key`) is what it computes on a fixed battery of items, so reorderings that change nothing are one specification. One with a step or selection that changes nothing is rejected.
  A group is the kinds of step and the final step, whatever their numbers and order, so that the same story with other numbers can't reach both RL and probe data.
- 20 nested loops don't compile and copies of a helper for each depth don't fit in a response, so deep tests cost fixed-depth code what they are meant to. Closures with one body and generators driven by a
  loop are caught by the watch of the call stack. A static call graph also follows deferred calls through closures, so a self-passing trampoline counts as recursion.
- Pairs in near can be matched by construction, so they get stricter matching than the filter asks for. Far pairs come from a writer that can't be steered as tightly.
"""
import ast
import hashlib
import math
import random
import re
from collections import deque
from copy import copy
from functools import lru_cache

from .. import sandbox
from ..base import Env, balanced, matched, unit_hash
from ..parse import code_block

PROMPT = ('You are given x, an arbitrarily nested list of integers. Elements of the top-level list have depth 1, elements of a list inside it have '
          'depth 2, and so on. Positions and list lengths always refer to x as given. {operations}\n\n'
          'Write the Python function nested_op(x). Return only the code in a single python code block, at most 8192 characters. '
          'Do not generate or compile code at runtime. {preference}')
PREFERENCES = ('Avoid recursion if you can.', 'Use a loop and an explicit stack of lists rather than recursion if you can.')
SOFT_SHARES = {1: .35, 2: .45, 3: .45, 4: .45}  # level 1 is short and simple, and 4B almost always recurses there under the soft line
VALUE_SHARE = .06
FINAL_SHARES = (.35, .45, .2)  # final steps where the order of the lists doesn't matter, where it does, and nested outputs
PARAMS = dict(p=range(3, 100), q=[*range(-99, 0), *range(1, 100)], m=[*range(-9, -1), *range(2, 10)], t=range(-30, 31), d=range(2, 7), w=(1, 2),
              i=range(1, 100), a=range(1, 100), s=range(1, 100), n=range(1, 100), k=range(2, 6), z=range(2, 6), o=range(2, 6), l=range(2, 6), u=range(1, 5))
VALUE, PLACE, DEPTH = 'pqmt', 'iasnkzolu', 'dw'
OP_TEXT = dict(p='keep only the integers divisible by {}', q='add {} to each integer', m='multiply each integer by {}', t='keep only the integers greater than {}',
               d='keep only the integers at depth {} or deeper', i='multiply each integer by {} times its 1-based position in its own list',
               a='add {} times its 1-based position in its own list to each integer', s='multiply each integer by {} times the number of elements of its own list',
               n='add {} times the number of elements of its own list to each integer', k='keep only the integers whose own list has at least {} elements',
               z='keep only the integers whose own list has at most {} elements', o='keep only the integers whose 1-based position in their own list is divisible by {}',
               l='keep only the integers at 1-based position {} or later in their own list', u='keep only the integers at 1-based position {} or earlier in their own list')
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
NESTED = dict(copy='return a copy of x with the same nesting, holding the results in place of the integers and leaving out the integers that were dropped',
              prune='return a copy of x with the same nesting, holding the results in place of the integers, leaving out the integers that were dropped and also any list that ends up empty')
# Per level: how many steps, which kinds of step can join a place step, final steps where the order of the lists doesn't matter and where it does, and how often a flat final step selects results.
PLANS = {1: (1, VALUE + PLACE, ('sum', 'count'), ('list', 'rev'), .25),
         2: (2, VALUE + PLACE, ('sum', 'max'), ('list', 'rev'), 0),
         3: (2, VALUE + PLACE + DEPTH, ('sum', 'count', 'max', 'min', 'sorted'), ('list', 'rev', 'first', 'last'), .75),
         4: (3, VALUE + PLACE + DEPTH, ('sum', 'count', 'max', 'min', 'sorted', 'distinct', 'range'), ('list', 'rev', 'first', 'last'), .75)}
SHALLOW, DEEP, TEST_TRIES, MAX_EMPTY, MIN_DIFFERENT, CUT_DEPTH, MIN_CUT = 3, 5, 6, 2, 3, 20, 3
PER_TEST, FAR_TESTS = .25, 8  # correct code takes milliseconds, and stuck loops are common enough to dominate the cost of scoring
PAIR_LENGTH, PAIR_DIGITS, PAIR_LINES, PAIR_TRIES, DOCSTRING_SHARE, FAR_POSITIVE_LONGER = 2, 8, 6, 300, .4, .6
PROMPT_SENTENCE = re.compile(r'as given\. (.*?)\n\n')


def flat(x, depth=1):
    """(integer, depth, position, size of its own list) of each integer of a nested list, in left-to-right order."""
    out = []
    for i, v in enumerate(x, 1):
        if isinstance(v, list):
            out += flat(v, depth + 1)
        else:
            out.append((v, depth, i, len(x)))
    return out


def step(kind, n, t):
    """One step on an (integer, depth, position, size) item: the changed item, or None if the step drops it."""
    v, d, i, size = t
    if kind == 'q':
        return (v + n, d, i, size)
    if kind == 'm':
        return (v * n, d, i, size)
    if kind == 'w':
        return (v * d ** n, d, i, size)
    if kind == 'i':
        return (v * n * i, d, i, size)
    if kind == 'a':
        return (v + n * i, d, i, size)
    if kind == 's':
        return (v * n * size, d, i, size)
    if kind == 'n':
        return (v + n * size, d, i, size)
    if kind == 'p':
        return t if v % n == 0 else None
    if kind == 'o':
        return t if i % n == 0 else None
    return t if {'t': v > n, 'd': d >= n, 'k': size >= n, 'z': size <= n, 'l': i >= n, 'u': i <= n}[kind] else None


def steps(ops, t):
    for kind, n in ops:
        t = step(kind, n, t)
        if t is None:
            return None
    return t


def results(items, ops, stat, select):
    """The reference computation on items: a flat result, or for a nested output what becomes of each integer."""
    done = [steps(ops, t) for t in items]
    if stat in NESTED:
        return [t and t[0] for t in done]
    keep = SELECT[select]
    return STATS[stat][0]([t[0] for t in done if t is not None and keep(t[0])])


def rebuild(x, ops, prune, depth=1):
    """The reference output of a nested final step."""
    out = []
    for i, v in enumerate(x, 1):
        if isinstance(v, list):
            inner = rebuild(v, ops, prune, depth + 1)
            if inner or not prune:
                out.append(inner)
        else:
            t = steps(ops, (v, depth, i, len(x)))
            if t is not None:
                out.append(t[0])
    return out


def expected(x, ops, stat, select):
    """The right answer for one input."""
    return rebuild(x, ops, stat == 'prune') if stat in NESTED else results(flat(x), ops, stat, select)


_battery = random.Random('07 battery')


def _placed(values):
    """Items with a random place in a list, so that every kind of step has an effect."""
    out = []
    for v in values:
        size = _battery.randint(1, 8)
        out.append((v, _battery.randint(1, 7), _battery.randint(1, size), size))
    return out


BATTERY = [_placed(_battery.randint(lo, hi) for _ in range(8))
           for lo, hi in 6 * ((-120, 120), (1, 120), (-120, -1), (-4, 4))]
BATTERY.append(_placed(_battery.sample(range(-100, 101), 201)))  # every sign boundary that q can move
for _p in PARAMS['p']:  # odd, even, positive, negative and repeated multiples of each p
    for _ in range(3):
        _multiples = [_p * m for m in _battery.sample((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5), 4)]
        BATTERY.append(_placed(_battery.choice(_multiples) for _ in range(8)))


@lru_cache(maxsize=2 ** 16)
def signature(ops, stat, select):
    """Specifications that agree on the battery count as the same specification."""
    return hashlib.md5(repr([stat in NESTED and stat] + [results(items, ops, stat, select) for items in BATTERY]).encode()).hexdigest()


@lru_cache(maxsize=2 ** 16)
def same_on_battery(ops, other, stat, select):
    return all(results(items, ops, stat, select) == results(items, other, stat, select) for items in BATTERY)


@lru_cache(maxsize=2 ** 16)
def same_selection(ops, stat, select):
    return all(results(items, ops, stat, select) == results(items, ops, stat, '') for items in BATTERY)


def step_text(kind, n):
    return W_TEXT[n] if kind == 'w' else OP_TEXT[kind].format(n)


def describe(ops, stat, select):
    """The {operations} sentence, in the pilot's wording."""
    texts = [step_text(kind, n) for kind, n in ops]
    if stat in NESTED:
        last = NESTED[stat]
    elif stat == 'count':
        last = f'return how many results are {select}' if select else 'return how many integers remain'
    else:
        last = STATS[stat][1].format(s=f'{select} ' if select else '')
    return ', then '.join(texts + [last]).capitalize() + '.'


@lru_cache(maxsize=2 ** 14)
def residues(p, before, depth, i, size):
    """The r in 0..p-1 such that r, at this place, reaches the divisibility filter as a multiple of p."""
    found = []
    for r in range(p):
        t = (r, depth, i, size)
        for kind, n in before:
            t = step(kind, n, t) or t
        if t[0] % p == 0:
            found.append(r)
    return found


def nest(rng, depth, deepest):
    """The shape of a list at `depth`, with None for integers: a branch reaching `deepest`, short side branches and the odd empty list."""
    out = [None] * rng.randint(1, 3)
    if depth < deepest:
        out.insert(rng.randint(0, len(out)), nest(rng, depth + 1, deepest))
    if rng.random() < .3:
        out.insert(rng.randint(0, len(out)), nest(rng, depth + 1, min(deepest, depth + 2)))
    if rng.random() < .1:
        out.insert(rng.randint(0, len(out)), [])
    return out


def cut(x, depth=CUT_DEPTH):
    """x as code that handles only a fixed depth sees it: without what lies below that depth."""
    return [([] if depth == 1 else cut(v, depth - 1)) if isinstance(v, list) else v for v in x]


def fill(x, leaf, depth=1):
    """Give each place of a shape its integer, which may depend on where it is."""
    return [fill(v, leaf, depth + 1) if isinstance(v, list) else leaf(depth, i, len(x)) for i, v in enumerate(x, 1)]


def make_tests(rng, ops, stat, select):
    """Hidden tests that exercise the specification, as [[x], expected] rows, or None if no draw does."""
    params, kinds = dict(ops), [kind for kind, _ in ops]
    p, q, d = params.get('p', 0), params.get('q', 0), params.get('d', 1)
    before = ops[:kinds.index('p')] if p else ()
    span = 20 + abs(q) + 2 * p

    def leaf(depth, i, size):
        found = residues(p, before, depth, i, size) if p and rng.random() < .6 else None
        if found:
            return rng.choice(found) + p * rng.choice((-3, -2, -1, 1, 2, 3))
        return rng.randint(-span, span) or 1

    fewer = [ops[:i] + ops[i + 1:] for i in range(len(ops))]
    for _ in range(TEST_TRIES):
        inputs = [fill(nest(rng, 1, max(3, d) + rng.randint(0, 2)), leaf) for _ in range(SHALLOW)]
        inputs += [fill(nest(rng, 1, rng.randint(72, 96)), leaf) for _ in range(DEEP)]
        rng.shuffle(inputs)
        want = [expected(x, ops, stat, select) for x in inputs]
        others = fewer + ([ops] if stat == 'prune' else [])
        if (sum(v == 0 or v == [] for v in want) <= MAX_EMPTY and len({repr(v) for v in want}) >= MIN_DIFFERENT
                and sum(expected(cut(x), ops, stat, select) != v for x, v in zip(inputs, want)) >= MIN_CUT
                and all(sum(expected(x, other, 'copy' if other is ops else stat, select) != v for x, v in zip(inputs, want)) >= 2 for other in others)):
            return [[[x], v] for x, v in zip(inputs, want)]
    return None


FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
CONTAINER_METHODS = {*dir(list), *dir(dict), *dir(set), *dir(str), *dir(deque)}
PLAIN_MODULES = {'abc', 'bisect', 'collections', 'copy', 'dataclasses', 'enum', 'fractions', 'functools', 'heapq', 'itertools', 'json', 'math', 'numbers', 'operator', 'queue', 're', 'statistics', 'string', 'sys', 'typing'}
WATCH_HOOKS = re.compile(r'[gs]etprofile|[gs]ettrace|_getframe|__code__')


def plain(code):
    """False if the code imports more than plain computation needs or touches what the watch of the call stack relies on."""
    if len(code) > 8192:
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {'exec', 'eval', 'compile', '__import__'}:
            return False
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        if any(name.split('.')[0] not in PLAIN_MODULES for name in names):
            return False
    return not WATCH_HOOKS.search(code)


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


def root_name(node):
    """The variable at the bottom of a target such as table["w"] or space.walk."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def kept_names(value):
    """The names whose value an expression keeps rather than only calls: walk in [walk] or partial(walk, 1), but not in walk(x)."""
    stack, names = [value], []
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Lambda):
            names += [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]
        elif isinstance(node, ast.Call):
            stack += [*node.args, *(k.value for k in node.keywords)]
        else:
            stack.extend(ast.iter_child_nodes(node))
    return names


def holders(tree, defs):
    """{variable: the functions it was given}, through assignments and calls such as table.append(walk). A function can reach what it reads from such a variable."""
    given = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs = [(root_name(target), node.value) for target in node.targets]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            pairs = [(root_name(node.func), arg) for arg in [*node.args, *(k.value for k in node.keywords)]]
        else:
            continue
        for name, value in pairs:
            if name and name not in defs:
                given.setdefault(name, set()).update(kept_names(value))
    held, grew = set(), True
    while grew:
        grew = False
        for name, names in given.items():
            if name not in held and names & (defs.keys() | held):
                held.add(name)
                grew = True
    return {name: given[name] & (defs.keys() | held) for name in held}


def parameter_edges(tree, defs):
    """Calls through function parameters, including deferred calls in nested closures."""
    functions = {fn: name for name, fns in defs.items() for fn in fns}
    parent, owner = {}, {}

    def walk(node, outer=None):
        if node in functions:
            parent[node] = outer
            outer = node
        owner[node] = outer
        for child in ast.iter_child_nodes(node):
            walk(child, outer)
    walk(tree)
    bound = {fn: {n.id for n in own_nodes(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
             | {arg.arg for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs} for fn in functions}
    given = {}

    def resolve(scope, name):
        while scope is not None:
            if name in bound[scope]:
                return given.get((scope, name), set())
            scope = parent[scope]
        return {name} if name in defs else set()

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    for _ in range(len(functions) + 1):
        grew = False
        for call in calls:
            scope = owner[call]
            for name in resolve(scope, call.func.id):
                for fn in defs[name]:
                    params = fn.args.posonlyargs + fn.args.args
                    values = [(arg.arg, value) for arg, value in zip(params, call.args)]
                    values += [(arg.arg, next((k.value for k in call.keywords if k.arg == arg.arg), None)) for arg in params + fn.args.kwonlyargs]
                    for arg, value in values:
                        if not isinstance(value, ast.Name):
                            continue
                        targets = resolve(scope, value.id)
                        old = given.setdefault((fn, arg), set())
                        if not targets <= old:
                            old.update(targets)
                            grew = True
        if not grew:
            break
    edges = {name: set() for name in defs}
    for call in calls:
        scope = owner[call]
        if scope is not None:
            edges[functions[scope]].update(resolve(scope, call.func.id))
    return edges


def call_graph(tree, defs, methods):
    """{function or variable holding one: the ones it refers to}. By name; a local variable or parameter hides a function's name."""
    held = holders(tree, defs)
    known = {*defs, *held}
    graph = {name: set(names) for name, names in held.items()}
    graph.update({name: set() for name in defs})
    for name, fns in defs.items():
        for fn in fns:
            nodes = list(own_nodes(fn))
            bound = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            bound |= {n.arg for n in nodes if isinstance(n, ast.arg)}
            for node in nodes:
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in known and (node.id not in bound or node.id in held):
                    graph[name].add(node.id)
                elif isinstance(node, ast.Attribute) and node.attr in methods | held.keys() and (
                        node.attr not in CONTAINER_METHODS or (isinstance(node.value, ast.Name) and node.value.id in ('self', 'cls'))):
                    graph[name].add(node.attr)
    for name, edges in parameter_edges(tree, defs).items():
        graph[name].update(edges)
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


def python_check(text, tests=(), entry='nested_op'):
    code = code_block(text, entry)
    if code is None:
        return dict(value=None, spans=[])
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return dict(value=None, spans=[])
    defs, methods = definitions(tree)
    base = text.rfind(code)
    starts = [0] + [m.end() for m in re.finditer(r'\r\n?|\n', code)]

    def offset(line, col):  # ast columns count UTF-8 bytes
        start = starts[line - 1]
        return base + start + len(code[start:].encode()[:col].decode(errors='ignore'))

    cycle = [name for name in on_cycle(call_graph(tree, defs, methods)) if name in defs]
    spans = [[offset(fn.lineno, fn.col_offset), offset(fn.end_lineno, fn.end_col_offset)] for name in cycle for fn in defs[name]]
    if not cycle and tests and sandbox.recurses(code, entry, tests, per_test=PER_TEST, generators=True):
        return dict(value=True, spans=[[base, base + len(code)]])
    return dict(value=bool(cycle), spans=spans)


# Near pairs are written by construction: both solve the task, and they differ only in how they go through x. Every form draws its names from the same lists, so neither `stack` nor `helper`
# nor `while` tells the classes apart, and forms of both kinds use while loops, pop and helper functions. @BODY@ is where the steps go and @FINAL@ the final step.
RECURSIVE_FLAT = dict(
    returned='''def nested_op(x):
    def {fn}({items}, {depth}):
        {out} = []
        for {pos}, {item} in enumerate({items}, 1):
            if isinstance({item}, list):
                {out} += {fn}({item}, {depth} + 1)
            elif isinstance({item}, int):
                {value} = {item}
                @BODY@
                {out}.append({value})
        return {out}

    {out} = {fn}(x, 1)
    @FINAL@''',
    shared='''def {fn}({items}, {depth}, {out}):
    for {pos}, {item} in enumerate({items}, 1):
        if isinstance({item}, list):
            {fn}({item}, {depth} + 1, {out})
            continue
        {value} = {item}
        @BODY@
        {out}.append({value})


def nested_op(x):
    {out} = []
    {fn}(x, 1, {out})
    @FINAL@''',
    generator='''def {fn}({items}, {depth}):
    for {pos}, {item} in enumerate({items}, 1):
        if isinstance({item}, list):
            yield from {fn}({item}, {depth} + 1)
            continue
        {value} = {item}
        @BODY@
        yield {value}


def nested_op(x):
    {out} = list({fn}(x, 1))
    @FINAL@''',
    indexed='''def nested_op(x):
    {out} = []

    def {fn}({items}, {depth}):
        {pos} = 0
        while True:
            if {pos} >= len({items}):
                break
            {item} = {items}[{pos}]
            {pos} += 1
            if isinstance({item}, list):
                {fn}({item}, {depth} + 1)
                continue
            {value} = {item}
            @BODY@
            {out}.append({value})

    {fn}(x, 1)
    @FINAL@''',
    consuming='''def {fn}({items}, {depth}):
    {out} = []
    {todo} = list({items})
    {pos} = 0
    while {todo}:
        {item} = {todo}.pop(0)
        {pos} += 1
        if isinstance({item}, list):
            {out}.extend({fn}({item}, {depth} + 1))
        elif isinstance({item}, int):
            {value} = {item}
            @BODY@
            {out}.append({value})
    return {out}


def nested_op(x):
    {out} = {fn}(x, 1)
    @FINAL@''',
    mutual='''def {fn}({items}, {depth}, {out}):
    for {pos}, {item} in enumerate({items}, 1):
        {other}({item}, {depth}, {pos}, len({items}), {out})


def {other}({item}, {depth}, {pos}, {size}, {out}):
    if isinstance({item}, list):
        {fn}({item}, {depth} + 1, {out})
    elif isinstance({item}, int):
        {value} = {item}
        @BODY@
        {out}.append({value})


def nested_op(x):
    {out} = []
    {fn}(x, 1, {out})
    @FINAL@''',
)
ITERATIVE_FLAT = dict(
    frames='''def nested_op(x):
    {out} = []
    {todo} = [(x, 1, 0)]
    while {todo}:
        {items}, {depth}, {pos} = {todo}.pop()
        while {pos} < len({items}):
            {item} = {items}[{pos}]
            {pos} += 1
            if isinstance({item}, list):
                {todo}.append(({items}, {depth}, {pos}))
                {todo}.append(({item}, {depth} + 1, 0))
                break
            elif isinstance({item}, int):
                {value} = {item}
                @BODY@
                {out}.append({value})
    @FINAL@''',
    iterators='''def {fn}(x):
    {out} = []
    {todo} = [(enumerate(x, 1), x, 1)]
    while {todo}:
        {entries}, {items}, {depth} = {todo}[-1]
        for {pos}, {item} in {entries}:
            if isinstance({item}, list):
                {todo}.append((enumerate({item}, 1), {item}, {depth} + 1))
                break
            elif isinstance({item}, int):
                {value} = {item}
                @BODY@
                {out}.append({value})
        else:
            {todo}.pop()
    return {out}


def nested_op(x):
    {out} = {fn}(x)
    @FINAL@''',
    entries='''def nested_op(x):
    {out} = []
    {todo} = [(x, 0, 0, 0)]
    while {todo}:
        {item}, {depth}, {pos}, {size} = {todo}.pop()
        if isinstance({item}, list):
            for {inner}, {sub} in reversed(list(enumerate({item}, 1))):
                {todo}.append(({sub}, {depth} + 1, {inner}, len({item})))
        elif isinstance({item}, int):
            {value} = {item}
            @BODY@
            {out}.append({value})
    @FINAL@''',
    levels='''def {fn}(x):
    {todo} = [(x, 0, 0, 0)]
    while any(isinstance(row[0], list) for row in {todo}):
        {entries} = []
        for {item}, {depth}, {pos}, {size} in {todo}:
            if isinstance({item}, list):
                {entries} += [({sub}, {depth} + 1, {inner}, len({item})) for {inner}, {sub} in enumerate({item}, 1)]
            else:
                {entries}.append(({item}, {depth}, {pos}, {size}))
        {todo} = {entries}
    return {todo}


def nested_op(x):
    {out} = []
    for {item}, {depth}, {pos}, {size} in {fn}(x):
        if isinstance({item}, int):
            {value} = {item}
            @BODY@
            {out}.append({value})
    @FINAL@''',
    spliced='''def nested_op(x):
    {todo} = [(x, 0, 0, 0)]
    {out} = []
    {inner} = 0
    while {inner} < len({todo}):
        {item}, {depth}, {pos}, {size} = {todo}[{inner}]
        if isinstance({item}, list):
            {todo}[{inner}:{inner} + 1] = [({item}[{pos} - 1], {depth} + 1, {pos}, len({item})) for {pos} in range(1, len({item}) + 1)]
            continue
        {inner} += 1
        if isinstance({item}, int):
            {value} = {item}
            @BODY@
            {out}.append({value})
    @FINAL@''',
)
UNORDERED_FLAT = dict(  # these visit lists out of order, so they can only serve final steps that don't care
    growing='''def nested_op(x):
    {out} = []
    {todo} = [(x, 1)]
    for {items}, {depth} in {todo}:
        for {pos}, {item} in enumerate({items}, 1):
            if isinstance({item}, list):
                {todo}.append(({item}, {depth} + 1))
            elif isinstance({item}, int):
                {value} = {item}
                @BODY@
                {out}.append({value})
    @FINAL@''',
    lists='''def {fn}(x):
    {todo} = [(x, 1)]
    {out} = []
    for {items}, {depth} in {todo}:
        {todo} += [({item}, {depth} + 1) for {item} in {items} if isinstance({item}, list)]
        for {pos}, {item} in enumerate({items}, 1):
            if isinstance({item}, int):
                {value} = {item}
                @BODY@
                {out}.append({value})
    return {out}


def nested_op(x):
    {out} = {fn}(x)
    @FINAL@''',
)
RECURSIVE_NESTED = dict(
    built='''def {fn}({items}, {depth}):
    {out} = []
    for {pos}, {item} in enumerate({items}, 1):
        if isinstance({item}, list):
            {sub} = {fn}({item}, {depth} + 1)
            @KEEP@
        else:
            {value} = {item}
            @BODY@
            {out}.append({value})
    return {out}


def nested_op(x):
    return {fn}(x, 1)''',
    self_call='''def nested_op(x, {depth}=1):
    {out} = []
    {pos} = 0
    while {pos} < len(x):
        {item} = x[{pos}]
        {pos} += 1
        if isinstance({item}, list):
            {sub} = nested_op({item}, {depth} + 1)
            @KEEP@
        else:
            {value} = {item}
            @BODY@
            {out}.append({value})
    return {out}''',
    consuming='''def nested_op(x):
    def {fn}({items}, {depth}):
        {out} = []
        {todo} = list({items})
        {pos} = 0
        while {todo}:
            {item} = {todo}.pop(0)
            {pos} += 1
            if isinstance({item}, list):
                {sub} = {fn}({item}, {depth} + 1)
                @KEEP@
            else:
                {value} = {item}
                @BODY@
                {out}.append({value})
        return {out}

    return {fn}(x, 1)''',
)
ITERATIVE_NESTED = dict(
    frames='''def nested_op(x):
    {root} = []
    {todo} = [[x, 1, 0, {root}]]
    while {todo}:
        {frame} = {todo}[-1]
        {items}, {depth}, {pos}, {out} = {frame}
        if {pos} == len({items}):
            {todo}.pop()
            @KEEP_FRAME@
            continue
        {item} = {items}[{pos}]
        {pos} += 1
        {frame}[2] = {pos}
        if isinstance({item}, list):
            {todo}.append([{item}, {depth} + 1, 0, []])
        else:
            {value} = {item}
            @BODY@
            {out}.append({value})
    return {root}''',
)
COPY_ONLY_NESTED = dict(  # these keep a list that ends up empty, so they can't serve the final step that removes it
    eager='''def nested_op(x):
    {root} = []
    {todo} = [(x, 1, {root})]
    while {todo}:
        {items}, {depth}, {out} = {todo}.pop()
        for {pos}, {item} in enumerate({items}, 1):
            if isinstance({item}, list):
                {sub} = []
                {out}.append({sub})
                {todo}.append(({item}, {depth} + 1, {sub}))
            else:
                {value} = {item}
                @BODY@
                {out}.append({value})
    return {root}''',
    growing='''def {fn}(x):
    {root} = []
    {todo} = [(x, 1, {root})]
    for {items}, {depth}, {out} in {todo}:
        for {pos}, {item} in enumerate({items}, 1):
            if isinstance({item}, list):
                {sub} = []
                {out}.append({sub})
                {todo}.append(({item}, {depth} + 1, {sub}))
            else:
                {value} = {item}
                @BODY@
                {out}.append({value})
    return {root}


def nested_op(x):
    return {fn}(x)''',
)
UNORDERED_STATS = ('sum', 'count', 'max', 'min', 'sorted', 'distinct', 'range')
NAME_POOLS = dict(
    fn=('walk', 'visit', 'collect', 'gather', 'scan', 'process', 'expand', 'helper', 'traverse', 'flatten', 'read_all', 'go_through'),
    other=('handle', 'treat', 'look_at', 'deal_with', 'check_item', 'take'),
    items=('items', 'values', 'elements', 'seq', 'lst', 'group', 'cells', 'members', 'parts'),
    item=('item', 'element', 'entry', 'node', 'elem', 'current', 'part', 'cell', 'thing'),
    depth=('depth', 'level', 'layer', 'd'),
    pos=('pos', 'place', 'position', 'idx', 'rank', 'slot'),
    size=('size', 'length', 'width', 'total_elements'),
    out=('found', 'out', 'result', 'results', 'acc', 'kept', 'collected', 'answer', 'output'),
    todo=('stack', 'pending', 'todo', 'frames', 'work', 'agenda', 'open_lists', 'queue', 'remaining', 'waiting'),
    value=('value', 'val', 'number', 'num', 'integer', 'amount'),
    sub=('child', 'inner', 'nest', 'sub', 'sublist', 'lower'),
    inner=('j', 'k', 'offset', 'cursor', 'at', 'next_at'),
    entries=('rows', 'entries', 'expanded', 'unfolded', 'batch', 'pairs'),
    root=('top', 'root', 'whole', 'copied', 'tree', 'rebuilt'),
    frame=('frame', 'state', 'open_one', 'record', 'slot_info', 'head'),
)


def code_of(template, names, ops, stat, select, size):
    """A form filled in: the names, then the steps at @BODY@, and the final step at @FINAL@ or the rule about empty lists at @KEEP@."""
    out, value, depth, pos, sub, todo = names['out'], names['value'], names['depth'], names['pos'], names['sub'], names['todo']
    skip = 'return' if '{other}(' in template else 'continue'
    body = []
    for kind, n in ops:
        check = dict(p=f'{value} % {n} != 0', t=f'{value} <= {n}', d=f'{depth} < {n}', k=f'{size} < {n}', z=f'{size} > {n}', o=f'{pos} % {n} != 0', l=f'{pos} < {n}', u=f'{pos} > {n}').get(kind)
        if check:
            body += [f'if {check}:', f'    {skip}']
        else:
            body.append(dict(q=f'{value} += {n}', m=f'{value} *= {n}', w=f'{value} *= {depth} ** {n}' if n > 1 else f'{value} *= {depth}', i=f'{value} *= {n} * {pos}',
                             a=f'{value} += {n} * {pos}', s=f'{value} *= {n} * {size}', n=f'{value} += {n} * {size}')[kind])
    final = []
    if stat not in NESTED:
        if select:
            final.append(f'{out} = [r for r in {out} if {dict(positive="r > 0", negative="r < 0", even="r % 2 == 0")[select]}]')
        final.append(dict(sum=f'return sum({out})', count=f'return len({out})', list=f'return {out}', rev=f'return {out}[::-1]', sorted=f'return sorted({out})', max=f'return max({out}, default=0)',
                          min=f'return min({out}, default=0)', first=f'return {out}[0] if {out} else 0', last=f'return {out}[-1] if {out} else 0', distinct=f'return len(set({out}))',
                          range=f'return max({out}) - min({out}) if {out} else 0')[stat])
    blocks = {'@BODY@': body, '@FINAL@': final, '@KEEP@': [f'{out}.append({sub})'] if stat == 'copy' else [f'if {sub}:', f'    {out}.append({sub})'],
              '@KEEP_FRAME@': [f'if {todo}{" and " + out if stat == "prune" else ""}:', f'    {todo}[-1][3].append({out})']}
    lines = []
    for line in template.format(**names).split('\n'):
        marker = line.strip()
        lines += [line[:len(line) - len(marker)] + piece for piece in blocks[marker]] if marker in blocks else [line]
    return '\n'.join(lines)


def constructed(task, rng, recursive, form=None):
    """A correct solution of a task with the given kind of traversal, as code. `form` picks it by name."""
    meta = task['meta']
    ops, stat, select = [tuple(op) for op in meta['ops']], meta['stat'], meta['select']
    names = {key: rng.choice(pool) for key, pool in NAME_POOLS.items()}
    if stat in NESTED:
        forms = RECURSIVE_NESTED if recursive else {**ITERATIVE_NESTED, **(COPY_ONLY_NESTED if stat == 'copy' else {})}
    else:
        forms = RECURSIVE_FLAT if recursive else {**ITERATIVE_FLAT, **(UNORDERED_FLAT if stat in UNORDERED_STATS else {})}
    form = form or rng.choice(tuple(forms))
    template = forms[form]
    if form == 'self_call':
        names['items'] = 'x'
    size = names['size'] if '{size}' in template else f"len({names['items']})"
    return code_of(template, names, ops, stat, select, size)


CUE_FEATURES = dict(length=len, rows=lambda t: t.count('\n'))
NEAR_FEATURES = dict(digits=lambda t: sum(c.isdigit() for c in t), commas=lambda t: t.count(','))


def positive_longer(request):
    """The class a request's pair should have longer. On far requests the recursive one more often, since pairs manage that less often even when asked."""
    meta = request['meta']
    return unit_hash(NestedListRecursion.seed_name, 'longer', meta['key']) < (FAR_POSITIVE_LONGER if 'entry' in meta else .5)


def alike(request, positive, negative):
    """True if a pair looks alike apart from the behavior, and is longer in the class that a hash of the task picks, in characters and in lines, so that over many pairs length says little."""
    return (matched((positive, negative), length=PAIR_LENGTH, digits=PAIR_DIGITS, lines=PAIR_LINES)
            and all(feature(positive) == feature(negative) or (feature(positive) > feature(negative)) == positive_longer(request) for feature in CUE_FEATURES.values()))


WIDENINGS = ((r'^( *)def nested_op\((x[^)]*)\):\n', r'\1def nested_op(\2):\n\1    {0}\n', ('assert isinstance(x, list)', 'assert x is not None', 'assert len(x) >= 0', 'top_size = len(x)', 'top_depth = 1', 'whole_input = x')),
             (r'^( *)for (\w+), (\w+) in enumerate\((\w+), 1\):$', r'\1{0} = enumerate(\4, 1)\n\1for \2, \3 in {0}:', ('numbered', 'counted', 'indexed_items')),
             (r'^( *)for (\w+), (\w+) in enumerate\((\w+), 1\):$', r'\1for \2 in range(1, len(\4) + 1):\n\1    \3 = \4[\2 - 1]{0}', ('',)),
             (r'^( *)if isinstance\((\w+), list\):$', r'\1{0} = isinstance(\2, list)\n\1if {0}:', ('nested', 'is_list', 'goes_deeper')),
             (r'^( *)([^\n]*?)(\w+) \+ 1\b([^\n:]*)$(?<!\])', r'\1{0} = \3 + 1\n\1\2{0}\4', ('below', 'deeper', 'next_level')),
             (r'^( *)return ([^\n]+)\Z', r'\1{0} = \2\n\1return {0}', ('final', 'ready', 'to_return')))


def widened(code, rng):
    """The same code with one more line, a guard or a name for something it computes, or None if nothing is left to name."""
    for pattern, new, names in rng.sample(WIDENINGS, len(WIDENINGS)):
        if not any(name and name in code for name in names):
            changed, count = re.subn(pattern, new.format(rng.choice(names)), code, count=1, flags=re.M)
            if count:
                return changed
    return None


def near_pair(task, rng):
    """A recursive and an iterative solution of a task, as alike as the forms allow."""
    key = task['meta']['key']
    sentence = PROMPT_SENTENCE.search(task['prompt'])[1]
    for _ in range(PAIR_TRIES):
        pair = [constructed(task, rng, True), constructed(task, rng, False)]
        pair = [code if rng.random() < .5 else widened(code, rng) or code for code in pair]
        pair = [code if rng.random() < .5 else widened(code, rng) or code for code in pair]
        while True:
            rows = [code.count('\n') for code in pair]
            shorter = rows[1] < rows[0]
            longer = widened(pair[shorter], rng) if rows[0] != rows[1] else None
            if longer is None:
                break
            pair[shorter] = longer
        if rng.random() < DOCSTRING_SHARE:
            pair = [re.sub(r'^( *)def nested_op\(([^)]*)\):\n', lambda m: f'{m[0]}{m[1]}    """{sentence}"""\n', code, flags=re.M) for code in pair]
        texts = [f'```python\n{code}\n```' for code in pair]
        if alike(task, *texts) and matched(texts, length=1.5, digits=4, lines=3) and all(balanced(f'{key}|{name}', *texts, feature) for name, feature in NEAR_FEATURES.items()):
            return pair
    return [pair[1], pair[1]]  # no pair fits, so the filter drops this request


# Far problems: other structures and other domains, in Python, so that every response runs. A problem is one group, however it is named or phrased.
SKINS = dict(
    files=('a folder in a file system: a dict with keys "name", "size" and "entries", where "entries" is a list of folders of the same shape', 'name', 'size', 'entries', 'folder', 'folders'),
    tree=('the root of a tree: a dict with keys "name", "weight" and "children", where "children" is a list of nodes of the same shape', 'name', 'weight', 'children', 'node', 'nodes'),
    staff=('an employee: a dict with keys "name", "salary" and "reports", where "reports" is a list of employees of the same shape', 'name', 'salary', 'reports', 'employee', 'employees'),
    thread=('a forum comment: a dict with keys "author", "score" and "replies", where "replies" is a list of comments of the same shape', 'author', 'score', 'replies', 'comment', 'comments'),
    shop=('a shop category: a dict with keys "title", "sales" and "subcategories", where "subcategories" is a list of categories of the same shape', 'title', 'sales', 'subcategories', 'category', 'categories'),
    menu=('a menu entry: a dict with keys "label", "price" and "submenu", where "submenu" is a list of menu entries of the same shape', 'label', 'price', 'submenu', 'entry', 'entries'),
    family=('a person: a dict with keys "name", "savings" and "children", where "children" is a list of people of the same shape', 'name', 'savings', 'children', 'person', 'people'),
)
FORMS = ('Write the Python function {entry}({args}). {statement}',
         'Complete this Python function.\n\n```python\ndef {entry}({args}):\n    """{statement}"""\n```',
         'Ticket: our utilities need a Python function {entry}({args}). {statement}')
FAR = {}
# How often each domain comes up. Recursion on a tree loops over children, so `for` and `while` say less there than where recursion needs no loop at all.
FAR_DOMAINS = dict(tree=.25, binary=.15, fs=.15, json=.1, grid=.05, graph=.05, number=.1, string=.05, linked=.05, array=.05)


def far(domain, goal, entry, args, statement, params, make, solve):
    """Register a far problem. make(rng, n, skin) draws the arguments of one input and solve(n, skin, *arguments) answers it."""
    FAR[f'{domain}|{goal}'] = dict(domain=domain, entry=entry, args=args, statement=statement, params=params, make=make, solve=solve)


def tree_input(rng, n, skin):
    _, name, num, kids, *_ = SKINS[skin]

    def node(depth, deepest):
        children = [node(depth + 1, deepest)] if depth < deepest else []
        children += [node(depth + 1, depth + 1) for _ in range(rng.randint(0, 2))] if depth < deepest else []
        rng.shuffle(children)
        return {name: f'{name[0]}{rng.randint(100, 999)}', num: rng.randint(1, 100), kids: children}

    return [node(1, rng.randint(2, 8))]


def tree_rows(root, skin, depth=1):
    """(the node's own name, its number, its depth, how many children it has) in preorder."""
    _, name, num, kids, *_ = SKINS[skin]
    rows = [(root[name], root[num], depth, len(root[kids]))]
    for child in root[kids]:
        rows += tree_rows(child, skin, depth + 1)
    return rows


TREE = '{tree}. Return '
ABOVE, DEPTHS = range(10, 91), range(2, 6)
far('tree', 'total', 'total', 'root', TREE + 'the total {num} of the {items} with {num} above {n}.', ABOVE, tree_input,
    lambda n, skin, root: sum(r[1] for r in tree_rows(root, skin) if r[1] > n))
far('tree', 'above', 'count_above', 'root', TREE + 'how many {items} have {num} above {n}.', ABOVE, tree_input,
    lambda n, skin, root: sum(r[1] > n for r in tree_rows(root, skin)))
far('tree', 'names', 'names_above', 'root', TREE + 'the {name} of each of the {items} with {num} above {n}, as a list, each {item} before those below it and siblings in list order.', ABOVE, tree_input,
    lambda n, skin, root: [r[0] for r in tree_rows(root, skin) if r[1] > n])
far('tree', 'leaves', 'count_leaves', 'root', TREE + 'how many {items} with {num} above {n} have an empty "{kids}" list.', ABOVE, tree_input,
    lambda n, skin, root: sum(r[1] > n and not r[3] for r in tree_rows(root, skin)))
far('tree', 'levels', 'deepest', 'root', TREE + 'the largest depth of a {item} with {num} above {n}, counting the top level as depth 1, or 0 if there is none.', ABOVE, tree_input,
    lambda n, skin, root: max((r[2] for r in tree_rows(root, skin) if r[1] > n), default=0))


def binary_input(rng, n, skin):
    def node(depth, deepest):
        left, right = (node(depth + 1, deepest), node(depth + 1, depth + rng.randint(1, 2))) if depth < deepest else (None, None)
        if rng.random() < .5:
            left, right = right, left
        return dict(value=rng.randint(-50, 50), left=left, right=right)

    return [node(1, rng.randint(2, 8))]


def binary_rows(root, depth=1):
    """(value, depth, is a leaf) in left-to-right order, as an in-order walk gives them."""
    if root is None:
        return []
    return [*binary_rows(root['left'], depth + 1), (root['value'], depth, not (root['left'] or root['right'])), *binary_rows(root['right'], depth + 1)]


BINARY = 'root is the root of a binary tree: a dict with keys "value", "left" and "right", where "left" and "right" are nodes or None. Return '
far('binary', 'total', 'tree_sum', 'root', BINARY + 'the sum of the values above {n}.', range(-30, 31), binary_input,
    lambda n, skin, root: sum(r[0] for r in binary_rows(root) if r[0] > n))
far('binary', 'inorder', 'in_order', 'root', BINARY + 'the values above {n} as a list, in in-order (left subtree, node, right subtree).', range(-30, 31), binary_input,
    lambda n, skin, root: [r[0] for r in binary_rows(root) if r[0] > n])
far('binary', 'leaves', 'leaf_values', 'root', BINARY + 'how many leaves have a value above {n}.', range(-30, 31), binary_input,
    lambda n, skin, root: sum(r[0] > n and r[2] for r in binary_rows(root)))
far('binary', 'level', 'at_depth', 'root', BINARY + 'how many nodes are at depth {n}, counting the root as depth 1.', range(2, 7), binary_input,
    lambda n, skin, root: sum(r[1] == n for r in binary_rows(root)))


def fs_input(rng, n, skin):
    def folder(depth, deepest):
        out = {f'f{rng.randint(10, 99)}.txt': rng.randint(1, 100) for _ in range(rng.randint(1, 3))}
        if depth < deepest:
            for _ in range(rng.randint(1, 2)):
                out[f'd{rng.randint(10, 99)}'] = folder(depth + 1, deepest if rng.random() < .5 else depth + 1)
        return out

    return [folder(1, rng.randint(2, 7))]


def fs_rows(folder, path='', depth=1):
    """(path, size, depth) of each file, in the order of the dict."""
    rows = []
    for name, entry in folder.items():
        rows += fs_rows(entry, f'{path}{name}/', depth + 1) if isinstance(entry, dict) else [(path + name, entry, depth)]
    return rows


FS = 'disk is a folder: a dict mapping each name to either a file size (an integer) or another folder of the same kind. Return '
far('fs', 'size', 'disk_usage', 'disk', FS + 'the total size of the files larger than {n}, in all folders.', ABOVE, fs_input,
    lambda n, skin, disk: sum(r[1] for r in fs_rows(disk) if r[1] > n))
far('fs', 'paths', 'big_files', 'disk', FS + 'the paths of the files larger than {n}, as a sorted list, with names joined by "/" and starting at a name directly in disk.', ABOVE, fs_input,
    lambda n, skin, disk: sorted(r[0] for r in fs_rows(disk) if r[1] > n))
far('fs', 'deepest', 'file_depth', 'disk', FS + 'the largest depth of a file larger than {n}, counting files directly in disk as depth 1, or 0 if there is none.', ABOVE, fs_input,
    lambda n, skin, disk: max((r[2] for r in fs_rows(disk) if r[1] > n), default=0))


def json_input(rng, n, skin):
    def value(depth):
        if depth > 6 or rng.random() < .3:
            return rng.choice((rng.randint(-50, 99), 'w' * rng.randint(0, 6), None, True))
        if rng.random() < .5:
            return [value(depth + 1) for _ in range(rng.randint(1, 3))]
        return {f'k{rng.randint(10, 99)}': value(depth + 1) for _ in range(rng.randint(1, 3))}

    return [{'id': rng.randint(1, 99), 'body': value(1), 'meta': value(2)}]


def json_rows(doc):
    """The values at every place of a parsed JSON document, containers before what they hold."""
    rows = [doc]
    for inner in doc.values() if isinstance(doc, dict) else doc if isinstance(doc, list) else ():
        rows += json_rows(inner)
    return rows


JSON = 'doc is a parsed JSON document: dicts, lists, integers, strings, booleans and None, nested to any depth. Return '
far('json', 'sum', 'sum_numbers', 'doc', JSON + 'the sum of the integers above {n} anywhere in it, not counting booleans.', ABOVE, json_input,
    lambda n, skin, doc: sum(r for r in json_rows(doc) if type(r) is int and r > n))
far('json', 'strings', 'long_strings', 'doc', JSON + 'how many of the values in it are strings of at least {n} characters. Dict keys are not values.', range(1, 6), json_input,
    lambda n, skin, doc: sum(isinstance(r, str) and len(r) >= n for r in json_rows(doc)))


def digit_sum(v, base):
    return v % base + digit_sum(v // base, base) if v else 0


def fibonacci(v):
    return v if v < 2 else fibonacci(v - 1) + fibonacci(v - 2)


def to_base(v, base):
    return (to_base(v // base, base) if v >= base else '') + str(v % base)


def collatz(v, below):
    return 0 if v < below else 1 + collatz(v // 2 if v % 2 == 0 else 3 * v + 1, below)


def stairs(v, most):
    return 1 if v == 0 else sum(stairs(v - k, most) for k in range(1, min(v, most) + 1))


def euclid(a, b):
    return a if b == 0 else euclid(b, a % b)


far('number', 'digits', 'digit_sum', 'v', 'v is a non-negative integer. Return the sum of its digits in base {n}.', range(2, 17),
    lambda rng, n, skin: [rng.randint(0, 10 ** 6)], lambda n, skin, v: digit_sum(v, n))
far('number', 'base', 'to_base', 'v', 'v is a non-negative integer. Return v written in base {n}, as a string of digits.', range(2, 10),
    lambda rng, n, skin: [rng.randint(0, 10 ** 5)], lambda n, skin, v: to_base(v, n))
far('number', 'power', 'power_mod', 'a, b', 'a and b are non-negative integers. Return a to the power b, modulo {n}, by repeated squaring.', range(7, 1000),
    lambda rng, n, skin: [rng.randint(0, 50), rng.randint(0, 60)], lambda n, skin, a, b: pow(a, b, n))
far('number', 'gcd', 'gcd_with', 'a', "a is a positive integer. Return the greatest common divisor of a and {n}, found with Euclid's algorithm.", range(2, 1000),
    lambda rng, n, skin: [rng.randint(1, 10 ** 4)], lambda n, skin, a: euclid(a, n))
far('number', 'collatz', 'collatz_steps', 'v', 'v is a positive integer. A step replaces v by v // 2 if it is even and by 3 * v + 1 if it is odd. Return how many steps it takes to get below {n}.', range(2, 60),
    lambda rng, n, skin: [rng.randint(1, 500)], lambda n, skin, v: collatz(v, n))
far('number', 'stairs', 'climb_ways', 'v', 'v is a non-negative integer. Return in how many ways a staircase of v steps can be climbed, taking between 1 and {n} steps at a time.', range(2, 6),
    lambda rng, n, skin: [rng.randint(0, 14)], lambda n, skin, v: stairs(v, n))


def decode(text, letter):
    """How many times a letter occurs in the expansion of a text such as 2[ab3[c]]d."""
    def part(i):
        total = 0
        while i < len(text) and text[i] != ']':
            if text[i].isdigit():
                inner, i = part(i + 2)
                total += int(text[i - 1 - inner[1]]) * inner[0] if False else 0
            total += text[i] == letter if i < len(text) else 0
            i += 1
        return total, i
    return part(0)


def squeeze(text):
    """Remove adjacent equal pairs until none remain."""
    for i in range(len(text) - 1):
        if text[i] == text[i + 1]:
            return squeeze(text[:i] + text[i + 2:])
    return text


LETTERS = tuple('abcdefghij')
far('string', 'reverse', 'reverse_without', 's', 's is a string of lower-case letters. Return it reversed, leaving out every "{n}".', LETTERS,
    lambda rng, n, skin: [''.join(rng.choice('abcdefghij') for _ in range(rng.randint(0, 20)))], lambda n, skin, s: s[::-1].replace(n, ''))
far('string', 'palindrome', 'is_palindrome', 's', 's is a string of lower-case letters. Return True if it reads the same in both directions once every "{n}" is removed, otherwise False.', LETTERS,
    lambda rng, n, skin: [(lambda half: half + (rng.choice(LETTERS) if rng.random() < .5 else '') + (half[::-1] if rng.random() < .6 else half))(''.join(rng.choice('abcdefghij') for _ in range(rng.randint(0, 8))))],
    lambda n, skin, s: s.replace(n, '') == s.replace(n, '')[::-1])
far('string', 'squeeze', 'squeeze_pairs', 's', 's is a string of lower-case letters. Remove every "{n}", then remove two adjacent equal letters, again and again, until there are none. Return what is left.', LETTERS,
    lambda rng, n, skin: [''.join(rng.choice('abc' + n) for _ in range(rng.randint(0, 14)))], lambda n, skin, s: squeeze(s.replace(n, '')))


def grid_input(rng, n, skin):
    rows, cols = rng.randint(2, 7), rng.randint(2, 7)
    return [[[rng.randint(0, 9) for _ in range(cols)] for _ in range(rows)], rng.randrange(rows), rng.randrange(cols)]


def region(grid, r, c, n, seen=None):
    """The cells of the region of values of at least n around a cell."""
    seen = set() if seen is None else seen
    if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] >= n and (r, c) not in seen:
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            region(grid, r + dr, c + dc, n, seen)
    return seen


GRID = 'grid is a list of equally long lists of digits from 0 to 9. A region is a set of cells joined up, down, left and right, all holding {n} or more. '
far('grid', 'fill', 'region_size', 'grid, r, c', GRID + 'Return the number of cells in the region holding cell (r, c), or 0 if that cell holds less than {n}.', range(2, 8), grid_input,
    lambda n, skin, grid, r, c: len(region(grid, r, c, n)))


def dag_input(rng, n, skin):
    names = [f'n{i}' for i in range(rng.randint(3, 8))]
    return [{name: rng.sample(names[i + 1:], min(len(names) - i - 1, rng.randint(0, 3))) for i, name in enumerate(names)}, names[0], names[-1]]


def short_paths(graph, start, end, moves):
    return 1 if start == end else sum(short_paths(graph, v, end, moves - 1) for v in graph[start]) if moves else 0


GRAPH = 'graph is a dict mapping each node name to the list of nodes it points to, with no cycles. '
far('graph', 'paths', 'count_paths', 'graph, start, end', GRAPH + 'Return the number of different paths from start to end, not counting those with more than {n} moves.', range(1, 8), dag_input,
    lambda n, skin, graph, start, end: short_paths(graph, start, end, n))


far('array', 'search', 'find_index', 'nums', 'nums is a sorted list of different integers. Find {n} in it by halving the range you look at. Return its index, or -1 if it is not there.', range(-99, 99),
    lambda rng, n, skin: [sorted(set(rng.sample(range(-99, 99), rng.randint(1, 30))) | ({n} if rng.random() < .7 else set()))],
    lambda n, skin, nums: nums.index(n) if n in nums else -1)


def linked_input(rng, n, skin):
    node = None
    for _ in range(rng.randint(1, 15)):
        node = dict(value=rng.randint(-50, 50), next=node)
    return [node]


def linked_values(node):
    return [node['value'], *linked_values(node['next'])] if node else []


LINKED = 'node is a linked list: a dict with keys "value" and "next", where "next" is another such dict or None. Return '
far('linked', 'sum', 'linked_sum', 'node', LINKED + 'the sum of the values above {n}.', range(-30, 31), linked_input,
    lambda n, skin, node: sum(v for v in linked_values(node) if v > n))
far('linked', 'count', 'linked_count', 'node', LINKED + 'how many values are above {n}.', range(-30, 31), linked_input,
    lambda n, skin, node: sum(v > n for v in linked_values(node)))
far('linked', 'list', 'linked_list', 'node', LINKED + 'the values above {n} as a list, in the order of the linked list.', range(-30, 31), linked_input,
    lambda n, skin, node: [v for v in linked_values(node) if v > n])
far('number', 'factorial', 'factorial_mod', 'v', 'v is a non-negative integer. Return the factorial of v, modulo {n}.', range(7, 1000),
    lambda rng, n, skin: [rng.randint(0, 25)], lambda n, skin, v: math.factorial(v) % n)
far('number', 'fibonacci', 'fibonacci_mod', 'v', 'v is a non-negative integer. Return the Fibonacci number with index v, where indexes 0 and 1 hold 0 and 1, modulo {n}.', range(7, 1000),
    lambda rng, n, skin: [rng.randint(0, 20)], lambda n, skin, v: fibonacci(v) % n)
far('array', 'count', 'count_above', 'nums', 'nums is a list of integers. Return how many of them are above {n}.', range(-30, 31),
    lambda rng, n, skin: [[rng.randint(-50, 50) for _ in range(rng.randint(0, 20))]], lambda n, skin, nums: sum(v > n for v in nums))
far('string', 'count', 'count_letter', 's', 's is a string of lower-case letters. Return how many times "{n}" occurs in it.', LETTERS,
    lambda rng, n, skin: [''.join(rng.choice('abcdefghij') for _ in range(rng.randint(0, 20)))], lambda n, skin, s: s.count(n))


NAMES_IT = re.compile(r'recurs|iterat', re.I)
COPY = 'Write exactly this code, and nothing else:\n\n```python\n{}\n```'
TAIL = ' Do not write comments or a docstring. Return only the code in a single python code block.'


ROOMY = 'Write it out at length: one statement per line, with no comprehensions.'
COMPACT = 'Write it as compactly as you can.'
RECURSIVE_HINT = 'Go through the items with a while loop and an index, and collect results in a list with append.'
ITERATIVE_HINTS = ('Do not use a while loop, pop or append: use a for loop over {} and add to it with += as you go.',) * 5 + ('Do not use pop or append: keep an index into {} and add to it with +=.',)


def far_variants(entry, key, rng):
    """Left alone, recursive code comes out shorter and only iterative code has while, pop and append. So the requests where `alike` wants the recursive one longer ask for that, and some move those words."""
    todo = rng.choice(NAME_POOLS['todo'])
    longer = positive_longer(dict(meta=dict(key=key, entry=entry)))
    recursive = [ROOMY if longer else '', RECURSIVE_HINT if rng.random() < .4 else '']
    iterative = [COMPACT if longer else '', rng.choice(ITERATIVE_HINTS).format(todo) if rng.random() < .6 else '']
    return [dict(label=True, instruction=f'This solution must be recursive: {entry}, or a helper function it calls, must call itself. {" ".join(filter(None, recursive))}{TAIL}'),
            dict(label=False, instruction=f'This solution must not be recursive, and no function may call itself. Keep any work still to do in a list named {todo}. {" ".join(filter(None, iterative))}{TAIL}')]


class NestedListRecursion(Env):
    name = '04_nested_list_recursion'
    seed_name = '07_nested_list_recursion'
    contrast = ('Disregard the last sentence above: here, recursion is required. Inside nested_op, call nested_op, or a helper that calls itself, on every nested list. No stack, no queue, no while loop.',
                'This solution must not be recursive: use a while loop and an explicit stack of lists. No function may call itself.')
    workers = sandbox.MAX_WORKERS
    writer = 'unsloth/gemma-3-12b-it'     # the default writer passes the tests of far problems in half as many pairs
    seen_specs = None
    wanted = None  # the pool being generated: tests, most of a draw's cost, are only made for its drafts

    def generate(self, level, pool='rl', seed=0):
        clone = copy(self)
        clone.wanted = pool
        clone.seen_specs = set()
        return Env.generate(clone, level, pool, seed)

    def build(self, item, rng, level):
        ops, stat, select = item
        group = '|'.join([*sorted(kind for kind, _ in ops), stat])
        if self.wanted not in (None, self.pool_of(group, level)):
            return None
        if any(same_on_battery(ops, ops[:i] + ops[i + 1:], stat, select) for i in range(len(ops))):
            return None  # a step has no effect, so this is a simpler specification in disguise
        if select and same_selection(ops, stat, select):
            return None
        key = f'{level}|{group}|{signature(ops, stat, select)}'
        if self.seen_specs is not None and key in self.seen_specs:
            return None
        tests = make_tests(rng, ops, stat, select)
        if tests is None:
            return None
        if self.seen_specs is not None:
            self.seen_specs.add(key)
        preference = PREFERENCES[unit_hash(self.seed_name, 'preference', key) >= SOFT_SHARES[level]]
        return dict(key=key, group=group, prompt=PROMPT.format(operations=describe(ops, stat, select), preference=preference), answer=None,
                    meta=dict(ops=[list(op) for op in ops], stat=stat, select=select, tests=tests))

    def draw(self, rng, level):
        count, kinds, free, ordered, selects = PLANS[level]
        if rng.random() < VALUE_SHARE:
            chosen = rng.sample(VALUE, count)
        else:
            chosen = [rng.choice(PLACE), *rng.sample(kinds, count - 1)]
            rng.shuffle(chosen)
        if len(set(chosen)) < count:
            return None
        stat = rng.choice(rng.choices((free, ordered, tuple(NESTED)), FINAL_SHARES)[0])
        select = rng.choice(tuple(SELECT)) if rng.random() < selects and stat not in NESTED else ''
        if level == 3 and not (select or set(chosen) & set(DEPTH) or stat not in {*PLANS[2][2], *PLANS[2][3], *NESTED}):
            return None  # that would be a level 2 specification
        return self.build((tuple((kind, rng.choice(PARAMS[kind])) for kind in chosen), stat, select), rng, level)

    def has_answer(self, text):
        return code_block(text) is not None

    def reward(self, task, text):
        entry = task['meta'].get('entry', 'nested_op')
        code = code_block(text, entry)
        if code is None or not plain(code):
            return 0.0
        passed = sandbox.run_tests(code, entry, task['meta']['tests'], per_test=PER_TEST, stop_on_timeout=True)
        return sum(passed) / len(passed)

    def check(self, task, text):
        return python_check(text, task['meta'].get('tests'), task['meta'].get('entry', 'nested_op'))

    def near_variants(self, task, rng):
        return [dict(label=label, copy=True, instruction=COPY.format(code)) for label, code in zip((True, False), near_pair(task, rng))]

    def request_prompt(self, request, variant):
        return variant['instruction'] if variant.get('copy') else super().request_prompt(request, variant)

    def far_draw(self, rng):
        domain = rng.choices(tuple(FAR_DOMAINS), tuple(FAR_DOMAINS.values()))[0]
        name = rng.choice([name for name in FAR if name.startswith(f'{domain}|')])
        problem = FAR[name]
        n, skin, form = rng.choice(problem['params']), rng.choice(tuple(SKINS)) if problem['domain'] == 'tree' else None, rng.choice(FORMS)
        tree, field, num, kids, item, items = SKINS[skin] if skin else (None,) * 6
        statement = problem['statement'].format(n=n, tree=f'root is {tree}', name=field, num=num, kids=kids, item=item, items=items)
        inputs = [problem['make'](rng, n, skin) for _ in range(FAR_TESTS)]
        tests = [[args, problem['solve'](n, skin, *args)] for args in inputs]
        if len({repr(want) for _, want in tests}) < 3:
            return None
        key = f'{name}|{n}|{skin}|{FORMS.index(form)}'
        return dict(key=key, group=name, prompt=form.format(entry=problem['entry'], args=problem['args'], statement=statement), answer=None,
                    meta=dict(domain=problem['domain'], entry=problem['entry'], n=n, skin=skin, tests=tests), variants=far_variants(problem['entry'], key, rng))

    def far_pool_of(self, group):
        """Every third problem of a domain, so that train and test cover the same domains."""
        same_domain = [name for name in FAR if name.split('|')[0] == group.split('|')[0]]
        return 'far_test' if same_domain.index(group) % 3 == 2 else 'far_train'

    def probe_keep(self, request, variant, text):
        return super().probe_keep(request, variant, text) and not NAMES_IT.search(text) and self.reward(request, text) == 1

    def set_keep(self, request, texts):
        return alike(request, *texts)


ENV = NestedListRecursion()
