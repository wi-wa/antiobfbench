#!/usr/bin/env python3
"""Single-GPU LoRA REINFORCE for antiobfbench, with an optional frozen probe."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from antiobfbench.rl_env import RLEnvironment, eligible_groups
from sklearn.metrics import roc_auc_score
from antiobfbench.mean_probe import LogisticMeanProbe, load_text_model
from overlap import generate_overlapped
from reinforce_worker import (ColocatedEngine, adapter_hash, backward_rollouts, backward_rollouts_shared, checked,
                              configure_training, save_checkpoint, sync_adapter)
from rl_core import advantages, retry_token_budget, thinking_text, validate_resume_config
from rl_loss import backward_with_retry
import rl_plots
from runtime import runtime_metadata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def log_trajectories(directory, step, jobs, sample_every, count):
    out = directory / 'rollouts'
    out.mkdir(exist_ok=True)
    with gzip.open(out / f'step_{step:06d}.jsonl.gz', 'wt', encoding='utf8', compresslevel=3) as handle:
        for row in jobs:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    if step == 1 or step % sample_every == 0:
        out = directory / 'samples'
        out.mkdir(exist_ok=True)
        readable = [{k:v for k,v in row.items() if k not in ('prompt_ids','output_ids')} for row in jobs[:count]]
        (out / f'step_{step:06d}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in readable))
        (out / f'step_{step:06d}.md').write_text('\n\n'.join(
            f"## {r['env_id']} / rollout {r['rollout']}\n\n{r['problem']}\n\n"
            f"Task reward: {r['task_reward']}; probe: {r['probe_probability']}; "
            f"UB: {r['ub_ground_truth']}; total reward: {r['reward']:.6f}\n\n"
            f"```text\n{r['response']}\n```" for r in readable) + '\n')


def train(directory, stop_after=None):
    config = json.loads((directory / 'config.json').read_text())
    run, opt, inf, rl = (config[k] for k in ('run','optim','inference','rl'))
    # Execution-only choices; absent keys keep the original behaviour.
    scoring = config.get('scoring', {})
    share_probe_prompts = scoring.get('share_prompts', False)
    score_workers = min(scoring.get('workers', 4), len(os.sched_getaffinity(0)))
    if opt.get('share_prompts', False) and opt['attention_backend'] != 'sdpa':
        raise ValueError('optim.share_prompts needs the sdpa attention backend')
    backward = backward_rollouts_shared if opt.get('share_prompts', False) else backward_rollouts
    environment = RLEnvironment(config)
    coefficient = config['probe']['coefficient']
    config['environment'] = environment.name
    config['reward'].update(task='native_task_with_format_penalty', environment_version=environment.version,
                            rl_contract_version=2, wrong_format=-1.0)
    config['probe']['eligibility'] = 'valid_completion_in_nonzero_task_advantage_group'
    config['probe']['direction'] = environment.direction
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Probe coefficient must be finite and nonnegative')
    if config['lora']['dropout'] != 0 or rl['baseline'] != 'leave_one_out':
        raise ValueError('Expected parent on-policy, leave-one-out configuration')
    if config['probe']['target'] != environment.target:
        raise ValueError('Unexpected probe orientation')
    write_json(directory/'resolved_config.json', config)
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one selected GPU')
    torch.cuda.set_device(0)
    torch.manual_seed(run['seed'])
    rng = random.Random(run['seed'])
    dataset_hash = sha(config['data']['path'])
    probe_hash = sha(config['probe']['path']) if config['probe']['path'] else None
    if probe_hash is not None and probe_hash != config['probe']['sha256']:
        raise ValueError('Frozen probe changed')
    detector = LogisticMeanProbe(config['probe']['path']) if probe_hash else None
    problems = environment.load(config['data']['path'])
    if len(problems) < rl['prompts_per_rollout']:
        raise ValueError('Insufficient RL problems')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['path'], local_files_only=True)
    end_id = tokenizer.convert_tokens_to_ids(config['reward']['end_token'])
    if end_id == tokenizer.unk_token_id or tokenizer.decode([end_id]) != config['reward']['end_token']:
        raise ValueError('Invalid end-of-turn token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = config['reward']['end_token']
    for row in problems:
        row['prompt_ids'] = tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],
                                    tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(row['prompt_ids']) + rl['max_rollout_tokens'] > inf['context_length']:
            raise ValueError(f"Context too short for {row['id']}")
    plots_spec = {
        'mean_reward': ('Task plus signed probe reward', '#13836d', (-max(1,coefficient), 1+max(0,environment.direction)*coefficient)),
        'mean_task_reward': ('Task reward including format penalty', '#557f64', (-1,1)),
        'probe_eligible_group_fraction': ('Groups eligible for probe reward', '#8574a6', (0,1)),
        'mean_accuracy': ('Task pass rate', '#2762a8', (0,1)),
        'mean_probe_probability': (f'Probe probability: {environment.target}', '#db8030', (0,1)),
        'probe_accuracy': ('Mean probability assigned to correct label', '#b34b23', (0,1)),
        'ub_rate': ('Objective UB rate among known labels', '#bb483a', (0,1)),
        'ub_known_fraction': ('Fraction with known objective UB label', '#8d6262', (0,1)),
        'honest_pass_rate': ('Task passed with no objective UB', '#20855d', (0,1)),
        'mean_correct_format': ('Valid format fraction', '#9c3a85', (0,1)),
        'truncation_rate': ('Token-limit fraction', '#ba4444', (0,1)),
        'mean_thinklen': ('Tokens inside exact think tags', '#b77e16', None),
        'mean_output_tokens': ('Response tokens', '#5a70ae', None),
    }
    def moving_average(values, window=8):
        return [float(np.mean(finite)) if (finite := [x for x in values[max(0,i-window+1):i+1] if math.isfinite(x)]) else float('nan')
                for i in range(len(values))]
    rl_plots.moving_average = moving_average
    rl_plots.PLOTS = plots_spec
    plots = rl_plots.TrainingPlots(directory, keep_history=config.get('wandb', {}).get('enabled', False))
    if (directory/'metrics.jsonl').exists():
        for line in (directory/'metrics.jsonl').open():
            metrics = json.loads(line)
            plots.steps.append(metrics['step'])
            for name in plots_spec:
                plots.values[name].append(float('nan') if metrics[name] is None else metrics[name])
    with socket.socket() as port_check:
        port_check.bind(('127.0.0.1', inf['port']))
    engine = None
    try:
        engine = ColocatedEngine(model_path=config['model']['path'], host='127.0.0.1', port=inf['port'],
            dtype='bfloat16', tp_size=1, base_gpu_id=0,
            attention_backend=inf['attention_backend'], triton_attention_reduce_in_fp32=True,
            chunked_prefill_size=inf.get('chunked_prefill_size', 2048), context_length=inf['context_length'],
            mem_fraction_static=inf['mem_fraction'], max_total_tokens=inf['kv_tokens'],
            schedule_conservativeness=inf.get('schedule_conservativeness', 1.0),
            enable_torch_compile=inf.get('torch_compile', False), torch_compile_max_bs=inf.get('torch_compile_max_bs', 16),
            max_running_requests=inf['max_running_requests'], cuda_graph_max_bs=inf['max_running_requests'],
            disable_cuda_graph=not inf['cuda_graphs'], random_seed=run['seed'],
            enable_memory_saver=inf['release_kv_during_optim'], enable_lora=True,
            max_lora_rank=config['lora']['rank'], lora_target_modules=config['lora']['target_modules'],
            max_loras_per_batch=1, max_loaded_loras=2, lora_backend=inf['lora_backend'],
            max_lora_chunk_size=inf['lora_chunk_size'], decode_log_interval=200, log_level='warning')
        base = load_text_model(config['model']['path'], opt['attention_backend'])
        state = None
        if config['checkpoint']['resume']:
            path = Path(config['checkpoint']['resume'])
            if not (path/'complete.json').exists():
                raise ValueError('Incomplete checkpoint')
            state = torch.load(path/'trainer_state.pt', map_location='cpu', weights_only=False)
            if state['config']['reward'].get('environment_version', 1) != environment.version or state['config']['reward'].get('rl_contract_version') != 2:
                raise ValueError('Environment reward changed; resume using matching archived code and data')
            validate_resume_config(state['config'], config)
            if (state['config']['probe'] != config['probe'] or state['config']['data'] != config['data']
                    or state['dataset_sha256'] != dataset_hash):
                raise ValueError('Probe, environment or dataset changed since checkpoint')
            policy = PeftModel.from_pretrained(base, path, is_trainable=True)
        else:
            policy = get_peft_model(base, LoraConfig(r=config['lora']['rank'], lora_alpha=config['lora']['alpha'],
                lora_dropout=0.0, target_modules=config['lora']['target_modules'], bias='none', task_type='CAUSAL_LM'))
        policy.to('cuda').eval()
        parameters = [p for p in policy.parameters() if p.requires_grad]
        assert parameters and all(p.dtype == torch.float32 for p in parameters)
        assert all('lora_' in name for name,p in policy.named_parameters() if p.requires_grad)
        configure_training(policy, opt)
        optimizer = torch.optim.AdamW(parameters, lr=opt['lr'], betas=tuple(opt['betas']),
                                     eps=opt['eps'], weight_decay=opt['weight_decay'], fused=True)
        step = 0
        if state:
            step = state['step']
            optimizer.load_state_dict(state['optimizer'])
            if adapter_hash(policy) != state['adapter_sha256']:
                raise ValueError('Checkpoint adapter does not match')
        if plots.steps and plots.steps[-1] != step:
            raise ValueError('Metrics must end at the resumed checkpoint; use a fresh run directory otherwise')
        if step >= run['max_steps']:
            raise ValueError('Maximum step must exceed checkpoint step')
        adapter = sync_adapter(engine, policy, config, step)
        if state:
            rng.setstate(state['python_rng'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state(state['cuda_rng'])
        packages = ['torch', 'transformers', 'sglang', 'peft', 'accelerate', 'numpy', 'scipy',
                    'scikit-learn', 'matplotlib', 'math-verify', 'chess']
        if opt['attention_backend'] == 'flash_attention_2':
            packages.append('flash-attn')
        if opt['loss_backend'].startswith('liger'):
            packages.append('liger-kernel')
        versions = {p:importlib.metadata.version(p) for p in packages}
        for package in ['wandb', 'torch-memory-saver']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        runtime = runtime_metadata(Path(__file__).resolve().parents[2], directory)
        properties = torch.cuda.get_device_properties(0)
        metadata = dict(**runtime, gpu=runtime['cuda_visible_devices'] if runtime['slurm_job_id'] else run['gpu'],
            device=torch.cuda.get_device_name(0), gpu_uuid=str(getattr(properties, 'uuid', 'unknown')),
            gpu_memory_bytes=properties.total_memory, dataset_sha256=dataset_hash,
            probe_sha256=probe_hash, probe_layer=detector.layer if detector else None,
            probe_uses='current LoRA policy response activations' if detector else None,
            reward='native_task_reward (invalid format=-1) + valid * eligible_task_group * direction * coefficient * probe_probability',
            prompt_sha256=hashlib.sha256(json.dumps([r['prompt'] for r in problems]).encode()).hexdigest(),
            environment=environment.name, environment_version=environment.version, sampling_profile=environment.profile,
            probe_accuracy_definition='mean probability assigned to correct label; unknown objective labels excluded',
            eligible_problems=len(problems), sources=dict(Counter(r['source'] for r in problems)),
            trainable_parameters=sum(p.numel() for p in parameters), initial_adapter_sha256=adapter_hash(policy),
            resumed_from_step=step, versions=versions)
        write_json(directory/'metadata.json', metadata)
        print(json.dumps({'event':'ready', **{k:v for k,v in metadata.items() if k != 'code_version'}}), flush=True)
        microbatch_size = opt['microbatch_size']
        last_checkpoint = None
        until = min(run['max_steps'], stop_after) if stop_after is not None else run['max_steps']
        while step < until:
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            jobs, sampling = [], []
            for problem in environment.choose(problems, rl['prompts_per_rollout'], rng):
                for rollout in range(rl['trajectories_per_prompt']):
                    sample_seed = rng.randrange(2**31)
                    jobs.append(dict(env_id=problem['id'], source=problem['source'], problem=problem['problem'],
                        prompt=problem['prompt'], answer=problem['answer'], prompt_ids=problem['prompt_ids'],
                        rollout=rollout, seed=sample_seed, policy_version=step))
                    sampling.append(dict(temperature=rl['temperature'], top_p=1.0, top_k=-1, min_p=0.0,
                        max_new_tokens=rl['max_rollout_tokens'], sampling_seed=sample_seed,
                        skip_special_tokens=False, stop_token_ids=[end_id]))
            def prepare(row, response):
                ids, finish = response['output_ids'], response['meta_info']['finish_reason']
                if not ids or len(ids) != response['meta_info']['completion_tokens']:
                    raise RuntimeError('Missing raw completion tokens')
                body = tokenizer.decode(ids[:-1] if ids[-1] == end_id else ids, skip_special_tokens=False)
                ended = ids[-1] == end_id and finish['type'] == 'stop'
                row.update(output_ids=ids,response=tokenizer.decode(ids,skip_special_tokens=False),finish_reason=finish)
                return body, ended

            def probe(rows):
                return detector.score(policy, rows, tokenizer.pad_token_id, end_id, batch_size=32,
                                      token_budget=config['probe']['token_budget'], share_prompts=share_probe_prompts)

            probabilities, verdicts = [None] * len(jobs), None
            policy.eval()
            if scoring.get('overlap_generation', False):
                # Score while later responses are still being generated. Checks run on threads only where the
                # environment allows it; the probe handles a prompt group once all of its responses are in.
                group = rl['trajectories_per_prompt']

                def on_response(index, response):
                    body, ended = prepare(jobs[index], response)
                    return environment.evaluate(jobs[index], body, ended), body

                def on_groups(groups, arrived):
                    members = [i for g in groups for i in range(g * group, (g + 1) * group)]
                    scores = probe([dict(prompt_ids=jobs[i]['prompt_ids'], output_ids=arrived[i]['output_ids']) for i in members])
                    for i, score in zip(members, scores):
                        probabilities[i] = score

                responses, verdicts = generate_overlapped(
                    engine, [r['prompt_ids'] for r in jobs], sampling, adapter, group,
                    on_response if environment.workers > 1 else None, on_groups if detector else None,
                    max(1, min(score_workers, environment.workers)))
            else:
                responses = engine.generate(input_ids=[r['prompt_ids'] for r in jobs], sampling_params=sampling,
                                            lora_path=[adapter]*len(jobs), return_logprob=False)
            if len(responses) != len(jobs):
                raise RuntimeError('Incomplete rollout batch')
            generation_seconds = time.monotonic()-start
            scoring_start = time.monotonic()
            if verdicts is None:
                prepared = [prepare(row, response) for row,response in zip(jobs,responses)]
                bodies = [body for body,_ in prepared]
                scoring_inputs = [(row,body,ended) for row,(body,ended) in zip(jobs,prepared)]
                if environment.workers > 1:
                    with ThreadPoolExecutor(max_workers=max(1, min(score_workers,environment.workers))) as pool:
                        verdicts = list(pool.map(lambda args: environment.evaluate(*args),scoring_inputs))
                else:
                    verdicts = [environment.evaluate(*args) for args in scoring_inputs]
            else:
                verdicts, bodies = (list(part) for part in zip(*verdicts))
            thoughts = [thinking_text(body) for body in bodies]
            for row,verdict in zip(jobs,verdicts):
                row.update(verdict)
            for row,tokens in zip(jobs,tokenizer(thoughts,add_special_tokens=False)['input_ids']):
                row['think_tokens'] = len(tokens)
            if inf['release_kv_during_optim']:
                checked(engine.release_memory_occupation(tags=['kv_cache']), 'release KV cache')
            if detector:
                if any(p is None for p in probabilities):
                    probabilities = probe(jobs)
                probabilities = np.asarray(probabilities, dtype=np.float64)
                if not np.all(np.isfinite(probabilities) & (probabilities > 0) & (probabilities < 1)):
                    raise FloatingPointError('Probe must return continuous probabilities inside (0,1)')
            task_rewards = [r['task_reward'] for r in jobs]
            task_advantages = advantages(task_rewards, rl['trajectories_per_prompt'], rl['baseline'])
            probe_eligible = eligible_groups(task_rewards, rl['trajectories_per_prompt'])
            for row,p,eligible,task_advantage in zip(jobs,probabilities,probe_eligible,task_advantages.tolist()):
                total = (environment.combine(row['task_reward'], p, row['format_valid'], coefficient, eligible)
                         if detector else row['task_reward'])
                row.update(probe_probability=float(p) if detector else None,
                           probe_ub_score=environment.ub_score(p) if detector else None,
                           task_advantage=task_advantage, probe_group_eligible=eligible,
                           probe_bonus=total-row['task_reward'], reward=total,
                           probe_correct_probability=None if not detector or row['probe_ground_truth'] is None
                           else float(p if row['probe_ground_truth'] else 1-p))
            coefficients = advantages([r['reward'] for r in jobs],rl['trajectories_per_prompt'],rl['baseline'])
            active = sorted((i for i,c in enumerate(coefficients) if c != 0),
                            key=lambda i: len(jobs[i]['prompt_ids'])+len(jobs[i]['output_ids']))
            for row,c in zip(jobs,coefficients.tolist()):
                row['advantage'] = c
            log_trajectories(directory, step+1, jobs, run['sample_every'], config['logging']['samples_to_save'])
            scoring_seconds = time.monotonic()-scoring_start
            optimization_start = time.monotonic()
            coefficients = coefficients.cuda()
            policy.train()
            before_hash = adapter_hash(policy)
            loss_value, microbatch_size = backward_with_retry(
                lambda size: backward(policy,jobs,active,coefficients,tokenizer.pad_token_id,
                    rl['temperature'],opt['logit_chunk_size'],size,retry_token_budget(opt,size),opt['loss_backend']),
                optimizer,microbatch_size)
            if not math.isfinite(loss_value):
                raise FloatingPointError('Non-finite REINFORCE loss')
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters,opt['max_grad_norm'],error_if_nonfinite=True).item()
            if active:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            policy.eval()
            torch.cuda.synchronize()
            after_hash = adapter_hash(policy)
            optimization_seconds = time.monotonic()-optimization_start
            sync_start = time.monotonic()
            if inf['release_kv_during_optim']:
                torch.cuda.empty_cache()
                checked(engine.resume_memory_occupation(tags=['kv_cache']), 'resume KV cache')
            step += 1
            # KV storage was released even for a zero-gradient batch; always invalidate stale cached prefixes.
            adapter = sync_adapter(engine,policy,config,step,adapter)
            n = len(jobs)
            def mean(key, group=jobs):
                values = [r[key] for r in group if r[key] is not None]
                return sum(values)/len(values) if values else None
            known = [r for r in jobs if r['ub_ground_truth'] is not None]
            labels = [r['probe_ground_truth'] for r in known] if detector else []
            probe_values = [r['probe_probability'] for r in known] if detector else []
            metrics = dict(step=step,rollout_policy_version=step-1,synced_policy_version=step,trajectories=n,
                mean_reward=mean('reward'),mean_task_reward=mean('task_reward'),mean_native_reward=mean('native_reward'),
                mean_accuracy=mean('passed'),mean_answer_accuracy=mean('answer_correct'),mean_correct_format=mean('format_valid'),
                probe_eligible_group_fraction=mean('probe_group_eligible'),
                probe_reward_fraction=sum(bool(detector) and r['probe_group_eligible'] and r['format_valid'] for r in jobs)/n,
                mean_probe_probability=mean('probe_probability'),mean_probe_bonus=mean('probe_bonus'),
                probe_accuracy=mean('probe_correct_probability', known),
                probe_threshold_accuracy=sum((p>=.5)==y for p,y in zip(probe_values,labels))/len(labels) if labels else None,
                probe_auroc=float(roc_auc_score(labels,probe_values)) if len(set(labels))==2 else None,
                ub_rate=sum(r['ub_ground_truth'] for r in known)/len(known) if known else None,
                ub_known_fraction=len(known)/n,
                honest_pass_rate=sum(r['passed'] and r['ub_ground_truth'] is False for r in jobs)/n,
                mean_thinklen=mean('think_tokens'),mean_output_tokens=sum(len(r['output_ids']) for r in jobs)/n,
                truncation_rate=sum(r['finish_reason']['type']=='length' for r in jobs)/n,
                loss=loss_value,grad_norm=grad_norm,adapter_changed=before_hash!=after_hash,adapter_sha256=after_hash,
                generation_seconds=generation_seconds,scoring_seconds=scoring_seconds,
                optimization_seconds=optimization_seconds,sync_seconds=time.monotonic()-sync_start,
                optimized_trajectories=len(active),optimization_microbatch_size=microbatch_size,
                optimization_token_budget=retry_token_budget(opt,microbatch_size),optimizer_step_skipped=not bool(active),
                iteration_seconds=time.monotonic()-start,trainer_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                by_source={source:dict(n=len(group),pass_rate=sum(r['passed'] for r in group)/len(group),
                    task_reward=sum(r['task_reward'] for r in group)/len(group),
                    probe_probability=mean('probe_probability', group))
                    for source in sorted({r['source'] for r in jobs})
                    for group in [[r for r in jobs if r['source']==source]]})
            with (directory/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics,allow_nan=False)+'\n')
            if step == 1 or step % run['log_every'] == 0 or step == until:
                print(json.dumps(metrics,allow_nan=False),flush=True)
            if step == 1 or step % run.get('plot_every', 10) == 0 or step == until:
                plots.update({k:float('nan') if v is None else v for k,v in metrics.items()})
            else:
                plots.steps.append(step)
                for key in plots_spec:
                    plots.values[key].append(float('nan') if metrics[key] is None else metrics[key])
            if step == 1 or step % run['ckpt_every'] == 0:
                last_checkpoint = save_checkpoint(directory,step,policy,optimizer,config,dataset_hash,rng)
        if config['checkpoint']['save_final'] and (last_checkpoint is None or last_checkpoint.name != f'step_{step:06d}'):
            last_checkpoint = save_checkpoint(directory,step,policy,optimizer,config,dataset_hash,rng)
        completion = dict(step=step,checkpoint=str(last_checkpoint),adapter_sha256=adapter_hash(policy))
        write_json(directory/'latest_checkpoint.json',completion)
        if step == run['max_steps']:
            write_json(directory/'complete.json',completion)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--stop-after',type=int)
    args = parser.parse_args()
    train(args.run_dir,args.stop_after)
