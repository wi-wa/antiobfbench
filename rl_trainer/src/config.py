"""Portable run configuration; relative paths are anchored at rl_trainer/."""
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
# Settings tested on one machine only. They are applied there over config.json, before the run's own file.
MACHINE_PRESETS = {'olivia': 'olivia_fast.json'}


def merge(config, supplied):
    for section, values in supplied.items():
        if section not in config:
            raise ValueError(f'Unknown configuration section: {section}')
        if section == 'environment':
            config[section] = values
        elif isinstance(values, dict):
            # Environment-specific data settings and saved probe metadata are allowed.
            unknown = set(values) - set(config[section])
            extra = {'data': {'profile', 'profile_weights'},
                     'probe': {'target', 'sha256', 'eligibility', 'direction'},
                     'reward': {'task', 'environment_version', 'rl_contract_version'}}
            if unknown - extra.get(section, set()):
                raise ValueError(f'Unknown {section} options: {sorted(unknown)}')
            config[section].update(values)
        else:
            raise ValueError(f'{section} must be a mapping')


def load_config(path=None, machine=None):
    """Defaults, then the preset of this machine (see runtime.machine), then the run's own file."""
    from runtime import machine as this_machine
    config = json.loads((ROOT / 'config.json').read_text())
    preset = MACHINE_PRESETS.get(machine or this_machine())
    if preset:
        merge(config, json.loads((ROOT / preset).read_text()))
    if path is not None:
        merge(config, json.loads(Path(path).read_text()))
    from antiobfbench.rl_env import RLEnvironment
    environment = RLEnvironment(config)
    config['environment'] = environment.name
    default_data = (ROOT.parent / 'src/data/09_math.jsonl' if environment.math else
                    ROOT.parent / 'envs' / environment.name / 'tasks.jsonl')
    config['data']['path'] = config['data']['path'] or str(default_data)
    for section, key in [('model', 'path'), ('data', 'path'), ('probe', 'path'), ('checkpoint', 'resume')]:
        value = config[section][key]
        if value:
            path = Path(value)
            config[section][key] = str((path if path.is_absolute() else ROOT / path).resolve())
    config['probe'].setdefault('target', environment.target)
    if config['probe']['path']:
        digest = hashlib.sha256(Path(config['probe']['path']).read_bytes()).hexdigest()
        if config['probe'].get('sha256', digest) != digest:
            raise ValueError('Frozen probe changed')
        config['probe']['sha256'] = digest
    elif config['probe']['coefficient'] != 0:
        raise ValueError('A nonzero probe coefficient requires probe.path')
    return config


def validate_config(c):
    tracking = c.get('wandb', {})
    if type(tracking.get('enabled', False)) is not bool:
        raise ValueError('wandb.enabled must be a boolean.')
    if tracking.get('enabled') and (not isinstance(tracking.get('project'), str) or not tracking['project'].strip()):
        raise ValueError('Enabled W&B logging requires a project name.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', c['run']['name']):
        raise ValueError('run.name must be a simple directory name, without slashes.')
    if type(c['run']['gpu']) is not int or c['run']['gpu'] < 0:
        raise ValueError('run.gpu must be one nonnegative physical GPU index.')
    if type(c['run']['seed']) is not int or c['run']['seed'] < 0:
        raise ValueError('run.seed must be a nonnegative integer.')
    if type(c['inference']['port']) is not int or not 1024 <= c['inference']['port'] <= 65535:
        raise ValueError('inference.port must be an unprivileged TCP port.')
    for section, keys in {
        'run': ['max_steps', 'log_every', 'sample_every', 'ckpt_every', 'plot_every'],
        'rl': ['prompts_per_rollout', 'trajectories_per_prompt', 'max_rollout_tokens'],
        'optim': ['microbatch_size', 'logit_chunk_size'],
        'inference': ['max_running_requests', 'context_length', 'kv_tokens'],
        'logging': ['samples_to_save'],
    }.items():
        for key in keys:
            if type(c[section][key]) is not int or c[section][key] < 1:
                raise ValueError(f'{section}.{key} must be a positive integer.')
    if type(c['lora']['rank']) is not int or c['lora']['rank'] not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError('LoRA rank must be a supported positive power of two, at most 128.')
    if c['lora']['dropout'] != 0:
        raise ValueError('On-policy training requires zero LoRA dropout.')
    allowed = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    targets = set(c['lora']['target_modules'])
    if not targets or not targets <= allowed or not {'q_proj', 'v_proj'} <= targets:
        raise ValueError('LoRA targets must be supported projections including q_proj and v_proj.')
    if c['rl']['baseline'] not in ('none', 'leave_one_out'):
        raise ValueError('rl.baseline must be none or leave_one_out.')
    if c['rl']['baseline'] == 'leave_one_out' and c['rl']['trajectories_per_prompt'] < 2:
        raise ValueError('Leave-one-out requires at least two trajectories per prompt.')
    for value in [c['rl']['temperature'], c['optim']['lr'], c['optim']['eps'],
                  c['optim']['max_grad_norm'], c['lora']['alpha']]:
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Temperature, learning rate, epsilon, alpha and gradient cap must be positive.')
    if len(c['optim']['betas']) != 2 or not all(0 <= b < 1 for b in c['optim']['betas']):
        raise ValueError('optim.betas must contain two numbers in [0, 1).')
    if not 0 < c['inference']['mem_fraction'] < 1:
        raise ValueError('inference.mem_fraction must be between zero and one.')
    if c['inference']['attention_backend'] != 'triton':
        raise ValueError('This trainer uses the verified Triton inference attention path.')
    if c['inference']['lora_backend'] not in ('triton', 'csgmv'):
        raise ValueError('inference.lora_backend must be triton or csgmv.')
    if c['inference']['lora_chunk_size'] not in (16, 32, 64, 128):
        raise ValueError('inference.lora_chunk_size must be 16, 32, 64 or 128.')
    opt = c['optim']
    if type(opt['max_tokens_per_microbatch']) is not int or opt['max_tokens_per_microbatch'] < 0:
        raise ValueError('optim.max_tokens_per_microbatch must be a nonnegative integer (0 disables it).')
    if opt['attention_backend'] not in ('sdpa', 'flash_attention_2'):
        raise ValueError('optim.attention_backend must be sdpa or flash_attention_2.')
    if opt['loss_backend'] not in ('torch', 'liger', 'liger_fused'):
        raise ValueError('optim.loss_backend must be torch, liger or liger_fused.')
    if type(opt['compile_blocks']) is not bool:
        raise ValueError('optim.compile_blocks must be a boolean.')
    if type(opt.get('share_prompts', False)) is not bool:
        raise ValueError('optim.share_prompts must be a boolean.')
    if opt.get('share_prompts', False) and opt['attention_backend'] != 'sdpa':
        raise ValueError('optim.share_prompts needs the sdpa attention backend.')
    conservativeness = c['inference'].get('schedule_conservativeness', 1.0)
    if not isinstance(conservativeness, (float, int)) or not math.isfinite(conservativeness) or conservativeness <= 0:
        raise ValueError('inference.schedule_conservativeness must be positive.')
    if type(c['inference'].get('chunked_prefill_size', 2048)) is not int or c['inference'].get('chunked_prefill_size', 2048) < 1:
        raise ValueError('inference.chunked_prefill_size must be a positive integer.')
    if type(c['inference'].get('torch_compile', False)) is not bool:
        raise ValueError('inference.torch_compile must be a boolean.')
    if type(c['inference'].get('torch_compile_max_bs', 16)) is not int or not 1 <= c['inference'].get('torch_compile_max_bs', 16) <= 64:
        raise ValueError('inference.torch_compile_max_bs must be an integer from 1 to 64.')
    scoring = c.get('scoring', {})
    if type(scoring.get('workers', 4)) is not int or scoring.get('workers', 4) < 1:
        raise ValueError('scoring.workers must be a positive integer.')
    for key in ('share_prompts', 'overlap_generation'):
        if type(scoring.get(key, False)) is not bool:
            raise ValueError(f'scoring.{key} must be a boolean.')
    if [c['reward'][k] for k in ['wrong_format', 'wrong_answer', 'correct_answer']] != [-1, 0, 1]:
        raise ValueError('The specified reward contract is -1 / 0 / +1.')
    if c['reward']['end_token'] not in ('<end_of_turn>', '<|eot_id|>', '<|im_end|>'):
        raise ValueError('The reward contract requires the model-specific assistant end-of-turn token.')
    if c['rl']['baseline'] != 'leave_one_out':
        raise ValueError('The benchmark probe gate requires leave_one_out.')
    if not math.isfinite(c['probe']['coefficient']) or c['probe']['coefficient'] < 0:
        raise ValueError('Probe coefficient must be finite and nonnegative.')
    if type(c['probe']['token_budget']) is not int or c['probe']['token_budget'] < 1:
        raise ValueError('probe.token_budget must be a positive integer.')
    if not Path(c['model']['path']).is_dir():
        raise ValueError('model.path must point to a downloaded Hugging Face model directory.')
