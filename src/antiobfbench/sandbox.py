"""Run model-written Python with resource limits, a scratch directory and no network.

`safe` is a speed bump against accidents, not a security boundary. Model code runs as the same user, so don't
loosen it without need. Results are read only from lines carrying a per-call nonce, so printing can't fake a pass.
"""
import ast
import json
import secrets
import shutil
import subprocess
import sys
import tempfile
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
import json, os, resource, signal, sys
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
        except OverTime:
            timer(0)
            if spec.get('stop_on_timeout'):
                break
        except BaseException:
            timer(0)
        emit('test', i, ok)
elif spec['mode'] == 'recursion':
    depth, live, seen = {}, {}, [False]
    yielding = __import__('opcode').opmap['YIELD_VALUE'] if spec.get('generators') else None
    def watch(frame, event, arg):
        code = frame.f_code
        if code.co_filename == 'solution':
            if event == 'call':
                depth[code] = depth.get(code, 0) + 1
                seen[0] = seen[0] or depth[code] > 1
                if yielding is not None and code.co_flags & 0x20 and code.co_name[0] != '<':
                    live.setdefault(code, set()).add(id(frame))
                    seen[0] = seen[0] or len(live[code]) > 1
            elif event == 'return':
                depth[code] = depth.get(code, 1) - 1
                if code in live and code.co_code[frame.f_lasti] != yielding:
                    live[code].discard(id(frame))
    for args, _ in spec['tests']:
        depth.clear()
        live.clear()
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


def run_tests(code, name, tests, per_test=2.0, stop_on_timeout=False):
    """Call name(*args) for each (args, expected) pair. Return a pass/fail list; unsafe or broken code fails all.

    With `stop_on_timeout`, the first test that runs out of time also fails the ones after it, so that code stuck in a loop costs one time limit and not one per test.
    """
    if not safe(code):
        return [False] * len(tests)
    spec = dict(mode='tests', code=code, name=name, per_test=per_test, cpu=int(per_test * len(tests)) + 5, stop_on_timeout=stop_on_timeout,
                tests=[(json.dumps(args), want) for args, want in tests])
    passed = [False] * len(tests)
    for line in _run(spec, per_test * (len(tests) + 1) + 3):
        if line[0] == 'test':
            passed[line[1]] = line[2]
    return passed


def recurses(code, name, tests, per_test=1.0, generators=False):
    """True if some function of the code was on the call stack twice while running name(*args) on the tests. Stops at the first timeout.

    With `generators`, two unfinished runs of one generator function count too, even if a loop drives them so that they never nest on the call stack.
    """
    if not safe(code):
        return False
    spec = dict(mode='recursion', code=code, name=name, per_test=per_test, cpu=int(per_test * len(tests)) + 5, generators=generators,
                tests=[(json.dumps(args), want) for args, want in tests])
    return any(line[0] == 'recursion' and line[1] for line in _run(spec, per_test * (len(tests) + 1) + 3))
