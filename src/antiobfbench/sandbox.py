"""Run model-written Python and regexes with resource limits, a scratch directory, no network and bounded workers.

`safe` is a speed bump against accidents, not a security boundary. Model code runs as the same user, so don't
loosen it without need. Results are read only from lines carrying a per-call nonce, so printing can't fake a pass.
"""
import ast
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRATCH = Path(__file__).resolve().parents[1] / '.sandbox'
NO_NETWORK = ['unshare', '-rn'] if shutil.which('unshare') else []
MAX_WORKERS = 24
ENV = {'PATH': '/usr/bin:/bin', 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

BANNED_MODULES = {'os', 'subprocess', 'shutil', 'socket', 'pathlib', 'importlib', 'ctypes', 'multiprocessing', 'threading',
                  'urllib', 'requests', 'pickle', 'signal', 'resource', 'glob', 'tempfile', 'io', 'builtins'}
BANNED_NAMES = {'open', 'exec', 'eval', '__import__', 'compile', 'input', 'globals', 'locals', 'getattr', 'setattr',
                'delattr', 'vars', 'breakpoint'}
SYS_ALLOWED = {'setrecursionlimit', 'getrecursionlimit', 'maxsize'}


def safe(code):
    """False if the code doesn't parse or uses anything outside plain computation."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split('.')[0] in BANNED_MODULES for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or '').split('.')[0]
            if module in BANNED_MODULES or not node.module:
                return False
            if module == 'sys' and any(alias.name not in SYS_ALLOWED for alias in node.names):
                return False
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                return False
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith('__'):
                return False
            if isinstance(node.value, ast.Name) and node.value.id == 'sys' and node.attr not in SYS_ALLOWED:
                return False
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name.startswith('__'):
                return False
    return True


DRIVER = r'''
import json, os, random, resource, signal, sys, time
spec = json.loads(sys.stdin.read())
out = os.fdopen(os.dup(1), 'w')
nonce = spec.pop('nonce')
null = os.open(os.devnull, os.O_RDWR)
for fd in (0, 1, 2):
    os.dup2(null, fd)
for limit, value in ((resource.RLIMIT_AS, 4 * 2 ** 30), (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_NOFILE, 64),
                     (resource.RLIMIT_CORE, 0), (resource.RLIMIT_CPU, spec['cpu']), (resource.RLIMIT_NPROC, 0)):
    resource.setrlimit(limit, (value, value))
class OverTime(BaseException):
    pass
def alarm(*_):
    raise OverTime()
signal.signal(signal.SIGALRM, alarm)
def timer(seconds):
    signal.setitimer(signal.ITIMER_REAL, seconds)
def emit(*parts):
    out.write(nonce + ' ' + json.dumps(parts) + '\n')
    out.flush()
def same(got, want):
    if isinstance(want, bool) or isinstance(got, bool):
        return isinstance(want, bool) and isinstance(got, bool) and got == want
    if isinstance(want, (int, float)):
        return type(got) in (int, float) and got == want
    if isinstance(want, str) or want is None:
        return type(got) is type(want) and got == want
    if isinstance(want, list):
        return type(got) in (list, tuple) and len(got) == len(want) and all(same(g, w) for g, w in zip(got, want))
    return False
sys.setrecursionlimit(spec.get('recursion', 3000))
ns = {'__name__': 'solution'}
try:
    timer(spec['per_test'])
    exec(compile(spec['code'], 'solution', 'exec'), ns)
    timer(0)
    fn = ns[spec['name']]
except BaseException:
    emit('load_error')
    os._exit(0)
if spec['mode'] == 'tests':
    for i, (args, want) in enumerate(spec['tests']):
        ok = False
        try:
            timer(spec['per_test'])
            ok = same(fn(*json.loads(args)), want)
            timer(0)
        except BaseException:
            timer(0)
        emit('test', i, ok)
elif spec['mode'] == 'recursion':
    depth, seen = {}, [False]
    def watch(frame, event, arg):
        code = frame.f_code
        if code.co_filename == 'solution':
            if event == 'call':
                depth[code] = depth.get(code, 0) + 1
                seen[0] = seen[0] or depth[code] > 1
            elif event == 'return':
                depth[code] = depth.get(code, 1) - 1
    for args, _ in spec['tests']:
        depth.clear()
        try:
            timer(spec['per_test'])
            sys.setprofile(watch)
            fn(*json.loads(args))
        except OverTime:
            break
        except BaseException:
            pass
        finally:
            sys.setprofile(None)
            timer(0)
    emit('recursion', seen[0])
else:
    random.seed(0)
    args = eval(spec['stress'], {'random': random})
    emit('started')
    t0 = time.perf_counter()
    try:
        timer(spec['limit'])
        got = fn(*args)
        timer(0)
        emit('finished', time.perf_counter() - t0)
        if spec.get('reference'):
            refs = {'__name__': 'reference'}
            exec(spec['reference'], refs)
            random.seed(0)
            want = refs[spec['name']](*eval(spec['stress'], {'random': random}))
            emit('right', same(got, json.loads(json.dumps(want))))
    except OverTime:
        emit('timeout')
    except BaseException:
        timer(0)
        emit('crashed')
out.flush()
os._exit(0)
'''


def _run(spec, wall):
    """Run the driver and return its [kind, ...] result lines, whatever happened to the process."""
    SCRATCH.mkdir(exist_ok=True)
    nonce = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as cwd:
        try:
            done = subprocess.run(NO_NETWORK + [sys.executable, '-I', '-c', DRIVER], input=json.dumps(dict(spec, nonce=nonce)),
                                  capture_output=True, text=True, timeout=wall, cwd=cwd, env=ENV)
            stdout = done.stdout
        except subprocess.TimeoutExpired as err:
            stdout = err.stdout.decode(errors='replace') if isinstance(err.stdout, bytes) else (err.stdout or '')
    return [json.loads(line[len(nonce) + 1:]) for line in stdout.splitlines() if line.startswith(nonce + ' ')]


def run_tests(code, name, tests, per_test=2.0):
    """Call name(*args) for each (args, expected) pair. Return a pass/fail list; unsafe or broken code fails all."""
    if not safe(code):
        return [False] * len(tests)
    spec = dict(mode='tests', code=code, name=name, per_test=per_test, cpu=int(per_test * len(tests)) + 5,
                tests=[(json.dumps(args), want) for args, want in tests])
    passed = [False] * len(tests)
    for line in _run(spec, per_test * (len(tests) + 1) + 3):
        if line[0] == 'test':
            passed[line[1]] = line[2]
    return passed


def recurses(code, name, tests, per_test=1.0):
    """True if some function of the code was on the call stack twice while running name(*args) on the tests. Stops at the first timeout."""
    if not safe(code):
        return False
    spec = dict(mode='recursion', code=code, name=name, per_test=per_test, cpu=int(per_test * len(tests)) + 5,
                tests=[(json.dumps(args), want) for args, want in tests])
    return any(line[0] == 'recursion' and line[1] for line in _run(spec, per_test * (len(tests) + 1) + 3))


def run_stress(code, name, stress, limit=2.0, reference=None):
    """Call name(*eval(stress)) with `random` seeded inside the driver. Return (outcome, seconds or None).

    The outcome is 'finished', 'timeout' or 'crashed', or 'wrong' if `reference`, the source of a correct solution of the same
    name, gives another result. Building the input and running the reference don't count toward the limit.
    """
    if not safe(code):
        return 'crashed', None
    spec = dict(mode='stress', code=code, name=name, stress=stress, limit=limit, per_test=limit, cpu=int(limit) + 8, reference=reference)
    lines = _run(spec, limit + 8)
    for line in lines:
        if line[0] == 'finished':
            return 'wrong' if ['right', False] in lines else 'finished', line[1]
        if line[0] == 'timeout':
            return 'timeout', None
        if line[0] == 'crashed':
            return 'crashed', None
    return ('timeout', None) if any(line[0] == 'started' for line in lines) else ('crashed', None)


REGEX_DRIVER = r'''
import json, re, resource, sys
spec = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_AS, (2 ** 30, 2 ** 30))
resource.setrlimit(resource.RLIMIT_CPU, (spec['cpu'], spec['cpu']))
rx = re.compile(spec['pattern'])
ok = all(rx.fullmatch(s) for s in spec['pos']) and not any(rx.fullmatch(s) for s in spec['neg'])
print('match' if ok else 'nomatch')
'''


def regex_fits(pattern, positives, negatives, timeout=2.0):
    """True if the pattern compiles and full-matches all positives and no negative within the timeout."""
    spec = dict(pattern=pattern, pos=positives, neg=negatives, cpu=int(timeout) + 1)
    try:
        done = subprocess.run([sys.executable, '-I', '-S', '-c', REGEX_DRIVER], input=json.dumps(spec), capture_output=True,
                              text=True, timeout=timeout, env=ENV)
    except subprocess.TimeoutExpired:
        return False
    return done.stdout.strip() == 'match'


def bounded_map(fn, items, workers=MAX_WORKERS):
    """Map fn over items with a bounded thread pool, for functions that wait on a subprocess."""
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))
