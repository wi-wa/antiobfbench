#!/usr/bin/env python3
"""Launch a single-GPU LoRA run; all outputs default to rl_trainer/logs/NAME."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT.parent / 'src')]
from config import load_config, validate_config
from wandb_logger import start_logger, finish_logger
from runtime import output_directory, gpu_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    for key, value in (('name', args.name), ('gpu', args.gpu), ('max_steps', args.steps)):
        if value is not None:
            config['run'][key] = value
    if args.resume:
        config['checkpoint']['resume'] = str(args.resume.resolve())
    validate_config(config)
    from antiobfbench.rl_env import RLEnvironment
    problems = RLEnvironment(config).load(config['data']['path'])
    if len(problems) < config['rl']['prompts_per_rollout']:
        raise ValueError('Not enough eligible RL tasks.')
    if args.dry_run:
        print(json.dumps({'eligible_tasks': len(problems), 'config': config}, indent=2))
        return
    directory = output_directory(ROOT, config['run']['name'])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    env = gpu_environment(config, os.environ)
    paths = [str(ROOT / 'src'), str(ROOT.parent / 'src')]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    env['PYTHONUNBUFFERED'] = '1'
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    for name, value in [('TOKENIZERS_PARALLELISM', 'false'), ('OMP_NUM_THREADS', '4'),
                        ('TORCHINDUCTOR_COMPILE_THREADS', '2')]:
        env.setdefault(name, value)
    command = [sys.executable, str(ROOT / 'src' / 'trainer.py'), '--run-dir', str(directory)]
    with (directory / 'console.log').open('x') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        logger = start_logger(directory, config)
        def stop(signum, frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            finish_logger(directory, config, process.returncode)
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
