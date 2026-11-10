"""Optional W&B sidecar: tails durable metrics/plots; training never waits for W&B."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import uuid
from runtime import machine


def write_status(directory, **values):
    try:
        path = directory / 'wandb' / 'status.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(path.read_text()) if path.exists() else {}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({**previous, **values}) + '\n')
        temporary.replace(path)
    except Exception:
        pass


def start_logger(directory, config):
    if not config.get('wandb', {}).get('enabled', False):
        return None
    try:
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', WANDB_SILENT='true')
        # The credential is read only by the sidecar, never by the training launcher.
        env.pop('WANDB_API_KEY', None)
        return subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                 '--run-dir', str(directory), '--parent-pid', str(os.getpid())],
                                cwd=directory, env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:
        write_status(directory, state='unavailable', error_type=type(exc).__name__)
        return None


def finish_logger(directory, config, exit_code):
    if config.get('wandb', {}).get('enabled', False):
        try:
            (directory / 'wandb' / 'training_finished.json').write_text(json.dumps({'exit_code': exit_code}))
        except Exception:
            pass


def read_key(path):
    for line in path.read_text().splitlines():
        name, separator, value = line.partition('=')
        if separator and name.strip().removeprefix('export ') == 'WANDB_API_KEY':
            parts = shlex.split(value, comments=True)
            if len(parts) == 1 and parts[0]:
                return parts[0]
    raise ValueError('WANDB_API_KEY is missing')


def parent_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def upload(directory, parent_pid):
    os.nice(10)
    store = directory / 'wandb'
    store.mkdir(exist_ok=True)
    done = store / 'training_finished.json'
    resolved = directory / 'resolved_config.json'
    while not resolved.exists():
        if done.exists() or not parent_alive(parent_pid):
            return
        time.sleep(1)
    config = json.loads(resolved.read_text())
    for variable, subdir in [('WANDB_CONFIG_DIR', 'config'), ('WANDB_CACHE_DIR', 'cache'),
                             ('WANDB_DATA_DIR', 'data'), ('TMPDIR', 'tmp')]:
        path = store / subdir
        path.mkdir(exist_ok=True)
        os.environ[variable] = str(path)
    key_file = Path(os.environ.get('ANTIOBF_WANDB_KEY_FILE', Path(__file__).resolve().parents[2] / '.env')).expanduser()
    os.environ['WANDB_API_KEY'] = read_key(key_file)
    if os.environ['WANDB_API_KEY'] in json.dumps(config):
        raise ValueError('Credential in run configuration')
    import wandb
    status_path = store / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    run_id = status.get('id') or uuid.uuid4().hex[:8]
    write_status(directory, state='initializing', id=run_id, pid=os.getpid())
    while True:
        try:
            run = wandb.init(project=config['wandb']['project'], name=directory.name, id=run_id,
                             resume='allow', mode='online', dir=str(directory), config=config, tags=[machine()],
                             save_code=False, settings=wandb.Settings(
                                 console='off', disable_git=True, disable_code=True,
                                 x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True,
                                 init_timeout=20, x_graphql_timeout_seconds=10,
                                 x_file_transfer_timeout_seconds=10))
            break
        except Exception as exc:
            write_status(directory, state='retrying', error_type=type(exc).__name__)
            if done.exists() or not parent_alive(parent_pid):
                return
            time.sleep(15)
    run.define_metric('step')
    run.define_metric('*', step_metric='step')
    last_step = status.get('last_metric_step', 0)
    uploaded = set(status.get('plot_steps', []))
    write_status(directory, state='running', url=run.url, id=run.id)
    offset = 0
    stopped_at = None
    while True:
        stopped = done.exists() or not parent_alive(parent_pid)
        if stopped and stopped_at is None:
            stopped_at = time.monotonic()
        try:
            metrics = directory / 'metrics.jsonl'
            if metrics.exists():
                with metrics.open() as handle:
                    handle.seek(offset)
                    while True:
                        line = handle.readline()
                        if not line or not line.endswith('\n'):
                            break
                        row = json.loads(line)
                        if row['step'] > last_step:
                            run.log(row)
                            last_step = row['step']
                        offset = handle.tell()
            for marker in sorted((store / 'plot_queue').glob('*/complete.json')):
                step = int(marker.parent.name)
                if step in uploaded:
                    continue
                images = {f'plots/{p.stem}': wandb.Image(str(p)) for p in sorted(marker.parent.glob('*.png'))}
                # A custom step axis lets delayed PNG uploads retain their original training step.
                run.log({'step': step, **images})
                uploaded.add(step)
            write_status(directory, state='running', last_metric_step=last_step, plot_steps=sorted(uploaded))
            if stopped:
                break
        except Exception as exc:
            write_status(directory, state='retrying', error_type=type(exc).__name__)
            if stopped_at is not None and time.monotonic() - stopped_at > 30:
                return
        time.sleep(1)
    # This deadline is in the sidecar only. The training process never joins it.
    deadline = threading.Timer(30, lambda: os._exit(0))
    deadline.daemon = True
    deadline.start()
    code = json.loads(done.read_text()).get('exit_code', 1) if done.exists() else 1
    run.finish(exit_code=code or 0)
    write_status(directory, state='finished', last_metric_step=last_step, plot_steps=sorted(uploaded))
    deadline.cancel()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args()
    try:
        upload(args.run_dir.resolve(), args.parent_pid)
    except Exception as exc:
        # Exception messages/tracebacks may contain SDK request details; keep only the type.
        write_status(args.run_dir, state='failed', error_type=type(exc).__name__)
