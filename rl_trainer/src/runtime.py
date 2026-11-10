"""Machine-independent launch settings and reproducibility metadata."""
import hashlib
import json
import os
from pathlib import Path
import platform


def machine():
    return os.environ.get('ANTIOBF_MACHINE') or platform.node().split('.')[0]


def output_directory(root, name):
    return Path(os.environ.get('ANTIOBF_RUNS_DIR', root / 'logs')).expanduser().resolve() / name


def gpu_environment(config, inherited):
    env = dict(inherited)
    # Slurm owns visibility, including UUID-based and remapped GPU assignments.
    if not env.get('SLURM_JOB_ID'):
        env['CUDA_VISIBLE_DEVICES'] = str(config['run']['gpu'])
    return env


def provenance(repository, directory):
    saved = repository / 'code_version.json'
    if saved.exists():
        version = json.loads(saved.read_text())
        diff = (repository / version['diff']).read_bytes()
        if hashlib.sha256(diff).hexdigest() != version['diff_sha256']:
            raise ValueError('Code snapshot diff changed')
    elif (repository / '.git').exists():
        from snapshot import code_version
        version, diff, _ = code_version(repository)
    else:
        version, diff = {'commit': None, 'dirty': None}, b''
    (directory / 'code.diff').write_bytes(diff)
    return {**version, 'diff': 'code.diff', 'diff_sha256': hashlib.sha256(diff).hexdigest()}


def runtime_metadata(repository, directory):
    return dict(machine=machine(), hostname=platform.node(), architecture=platform.machine(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                slurm_cuda_visible_devices=os.environ.get('ANTIOBF_SLURM_CUDA_VISIBLE_DEVICES'),
                mps_enabled=os.environ.get('ANTIOBF_MPS') == '1',
                slurm_job_gpus=os.environ.get('SLURM_JOB_GPUS'),
                container_image=os.environ.get('ANTIOBF_CONTAINER_IMAGE') or os.environ.get('APPTAINER_CONTAINER'),
                container_sha256=os.environ.get('ANTIOBF_CONTAINER_SHA256'),
                code_version=provenance(repository, directory))
