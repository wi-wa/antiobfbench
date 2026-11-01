"""antiobfbench: generators, rewards, checkers and probe-data requests for the benchmark environments."""
import importlib
import pkgutil

from . import envs as _envs


def all_envs():
    """Every environment found in antiobfbench/envs (one module eNN_*.py with an ENV instance), in spec order."""
    names = sorted(m.name for m in pkgutil.iter_modules(_envs.__path__) if m.name.startswith('e'))
    found = [importlib.import_module(f'{_envs.__name__}.{name}').ENV for name in names]
    return {env.name: env for env in found}


def get_env(name):
    """Look up an environment by its full name or its two-digit number."""
    for key, env in all_envs().items():
        if name in (key, key[:2]):
            return env
    raise KeyError(name)
