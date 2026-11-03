#!/usr/bin/env python3
"""Gemma mean-probe RL using the sibling antiobf REINFORCE implementation."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from antiobfbench.envs.e09_math_dinosaurs import ENV, PROMPT, SOURCES, settings, choose_problems
from antiobfbench.mean_probe import LogisticMeanProbe, load_text_model
from reinforce_worker import (ColocatedEngine, adapter_hash, backward_rollouts, checked,
                              configure_training, save_checkpoint, sync_adapter)
from rl_core import advantages, retry_token_budget, thinking_text, validate_resume_config
from rl_loss import backward_with_retry
import rl_plots


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def log_trajectories(directory, step, jobs, sample_every, count):
    out = directory / 'rollouts'
    out.mkdir(exist_ok=True)
    with gzip.open(out / f'step_{step:06d}.jsonl.gz', 'wt', encoding='utf8') as handle:
        for row in jobs:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    if step == 1 or step % sample_every == 0:
        out = directory / 'samples'
        out.mkdir(exist_ok=True)
        readable = [{k:v for k,v in row.items() if k not in ('prompt_ids','output_ids')} for row in jobs[:count]]
        (out / f'step_{step:06d}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in readable))
        (out / f'step_{step:06d}.md').write_text('\n\n'.join(
            f"## {r['env_id']} / rollout {r['rollout']}\n\n{r['problem']}\n\n"
            f"Math: {r['math_reward']}; probe: {r['probe_probability']:.6f}; "
            f"dinosaurs: {r['dinosaur_present']}; total reward: {r['reward']:.6f}\n\n"
            f"```text\n{r['response']}\n```" for r in readable) + '\n')


def train(directory):
    config = json.loads((directory / 'config.json').read_text())
    run, opt, inf, rl = (config[k] for k in ('run','optim','inference','rl'))
    coefficient = config['probe'].setdefault('coefficient', settings()['probe_coefficient'])
    profile = config['data'].setdefault('profile', settings()['default_profile'])
    config['data']['profile_weights'] = None if profile == 'legacy' else settings()['profiles'][profile]
    config['reward']['environment_version'] = ENV.reward_version
    write_json(directory/'resolved_config.json', config)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('Probe coefficient must be finite and nonnegative')
    if config['lora']['dropout'] != 0 or rl['baseline'] != 'leave_one_out':
        raise ValueError('Expected parent on-policy, leave-one-out configuration')
    if config['reward']['task'] != 'binary_math' or config['probe']['target'] != 'dinosaur_presence':
        raise ValueError('Unexpected reward orientation')
    if torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one selected GPU')
    torch.cuda.set_device(0)
    torch.manual_seed(run['seed'])
    rng = random.Random(run['seed'])
    dataset_hash = sha(config['data']['path'])
    probe_hash = sha(config['probe']['path'])
    if probe_hash != config['probe']['sha256']:
        raise ValueError('Frozen probe changed')
    detector = LogisticMeanProbe(config['probe']['path'])
    rows = [json.loads(line) for line in Path(config['data']['path']).open()]
    sources = config['data'].get('sources') or list(SOURCES.values())
    problems = [r for r in rows if r['split'] == 'train' and r['source'] in sources]
    choose_problems(problems, 1, random.Random(0), profile)  # Validate profile coverage before loading GPU models.
    if len(problems) < rl['prompts_per_rollout'] or len({r['problem_hash'] for r in problems}) != len(problems):
        raise ValueError('Insufficient or duplicated RL problems')
    tokenizer = AutoTokenizer.from_pretrained(config['model']['path'], local_files_only=True)
    end_id = tokenizer.convert_tokens_to_ids(config['reward']['end_token'])
    if end_id == tokenizer.unk_token_id or tokenizer.decode([end_id]) != config['reward']['end_token']:
        raise ValueError('Invalid end-of-turn token')
    for row in problems:
        row['prompt'] = PROMPT.format(problem=row['problem'])
        row['prompt_ids'] = tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],
                                    tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(row['prompt_ids']) + rl['max_rollout_tokens'] > inf['context_length']:
            raise ValueError(f"Context too short for {row['id']}")
    plots_spec = {
        'mean_reward': ('Math + weighted probe reward', '#13836d', (0, 1+coefficient)),
        'mean_accuracy': ('Valid-format math accuracy', '#2762a8', (0, 1)),
        'mean_probe_probability': ('Dinosaur-presence probability', '#db8030', (0, 1)),
        'dinosaur_rate': ('Actual dinosaur mentions', '#bb483a', (0, 1)),
        'honest_pass_rate': ('Correct math with dinosaur presence', '#20855d', (0, 1)),
        'mean_correct_format': ('Strict-format fraction', '#9c3a85', (0, 1)),
        'truncation_rate': ('Token-limit fraction', '#ba4444', (0, 1)),
        'mean_thinklen': ('Tokens inside exact think tags', '#b77e16', None),
        'mean_output_tokens': ('Response tokens', '#5a70ae', None),
    }
    rl_plots.PLOTS = plots_spec
    plots = rl_plots.TrainingPlots(directory)
    if (directory/'metrics.jsonl').exists():
        for line in (directory/'metrics.jsonl').open():
            metrics = json.loads(line)
            plots.steps.append(metrics['step'])
            for name in plots_spec:
                plots.values[name].append(metrics[name])
    with socket.socket() as port_check:
        port_check.bind(('127.0.0.1', inf['port']))
    engine = None
    try:
        engine = ColocatedEngine(model_path=config['model']['path'], host='127.0.0.1', port=inf['port'],
            dtype='bfloat16', tp_size=1, base_gpu_id=0,
            attention_backend=inf['attention_backend'], triton_attention_reduce_in_fp32=True,
            chunked_prefill_size=2048, context_length=inf['context_length'],
            mem_fraction_static=inf['mem_fraction'], max_total_tokens=inf['kv_tokens'],
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
            if state['config']['reward'].get('environment_version', 1) != ENV.reward_version:
                raise ValueError('Environment reward changed; use the archived v1 code/data to resume a v1 run')
            validate_resume_config(state['config'], config)
            if (state['config']['probe'] != config['probe'] or state['config']['data'] != config['data']
                    or state['dataset_sha256'] != dataset_hash):
                raise ValueError('Probe, math mixture or dataset changed since checkpoint')
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
        metadata = dict(gpu=run['gpu'], device=torch.cuda.get_device_name(0), dataset_sha256=dataset_hash,
            probe_sha256=probe_hash, probe_layer=detector.layer, probe_uses='current LoRA policy response activations',
            reward='valid_completion * (binary_math_correct + coefficient * sigmoid(frozen_probe(mean_response_hidden)))',
            environment_version=ENV.reward_version, sampling_profile=profile, environment_config=settings(),
            eligible_problems=len(problems), sources=dict(Counter(r['source'] for r in problems)),
            trainable_parameters=sum(p.numel() for p in parameters), initial_adapter_sha256=adapter_hash(policy),
            resumed_from_step=step, versions={p:importlib.metadata.version(p) for p in
                                            ('torch','transformers','sglang','peft','flash-attn','liger-kernel','math-verify')})
        write_json(directory/'metadata.json', metadata)
        print(json.dumps({'event':'ready', **metadata}), flush=True)
        microbatch_size = opt['microbatch_size']
        last_checkpoint = None
        while step < run['max_steps']:
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            jobs, sampling = [], []
            for problem in choose_problems(problems, rl['prompts_per_rollout'], rng, profile):
                for rollout in range(rl['trajectories_per_prompt']):
                    sample_seed = rng.randrange(2**31)
                    jobs.append(dict(env_id=problem['id'], source=problem['source'], problem=problem['problem'],
                        prompt=problem['prompt'], answer=problem['answer'], prompt_ids=problem['prompt_ids'],
                        rollout=rollout, seed=sample_seed, policy_version=step))
                    sampling.append(dict(temperature=rl['temperature'], top_p=1.0, top_k=-1, min_p=0.0,
                        max_new_tokens=rl['max_rollout_tokens'], sampling_seed=sample_seed,
                        skip_special_tokens=False, stop_token_ids=[end_id]))
            responses = engine.generate(input_ids=[r['prompt_ids'] for r in jobs], sampling_params=sampling,
                                        lora_path=[adapter]*len(jobs), return_logprob=False)
            if len(responses) != len(jobs):
                raise RuntimeError('Incomplete rollout batch')
            generation_seconds = time.monotonic()-start
            scoring_start = time.monotonic()
            thoughts = []
            for row,response in zip(jobs,responses):
                ids, finish = response['output_ids'], response['meta_info']['finish_reason']
                if not ids or len(ids) != response['meta_info']['completion_tokens']:
                    raise RuntimeError('Missing raw completion tokens')
                body = tokenizer.decode(ids[:-1] if ids[-1] == end_id else ids, skip_special_tokens=False)
                ended = ids[-1] == end_id and finish['type'] == 'stop'
                answer_correct = ENV.answer_reward(row, body)
                valid = ENV.format_valid(body, ended)
                math_reward = answer_correct if valid else 0.0
                ub = ENV.check(row, body)['value']
                row.update(output_ids=ids, response=tokenizer.decode(ids,skip_special_tokens=False), finish_reason=finish,
                    math_reward=math_reward, answer_correct=bool(answer_correct), dinosaur_present=not ub, ub_ground_truth=ub,
                    format_valid=valid, end_of_turn_valid=ended)
                thoughts.append(thinking_text(body))
            for row,tokens in zip(jobs,tokenizer(thoughts,add_special_tokens=False)['input_ids']):
                row['think_tokens'] = len(tokens)
            if inf['release_kv_during_optim']:
                checked(engine.release_memory_occupation(tags=['kv_cache']), 'release KV cache')
            policy.eval()
            probabilities = detector.score(policy, jobs, tokenizer.pad_token_id, end_id,
                                            token_budget=config['probe']['token_budget'])
            if not np.all(np.isfinite(probabilities) & (probabilities > 0) & (probabilities < 1)):
                raise FloatingPointError('Probe must return continuous probabilities inside (0,1)')
            for row,p in zip(jobs,probabilities):
                row.update(probe_probability=float(p), probe_ub_score=ENV.ub_score(p),
                           probe_bonus=coefficient*float(p) if row['format_valid'] else 0.0,
                           reward=ENV.combine_reward(row['math_reward'], p, row['format_valid'], coefficient))
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
                lambda size: backward_rollouts(policy,jobs,active,coefficients,tokenizer.pad_token_id,
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
            mean = lambda key: sum(r[key] for r in jobs)/n
            metrics = dict(step=step, rollout_policy_version=step-1, synced_policy_version=step, trajectories=n,
                mean_reward=mean('reward'), mean_accuracy=mean('math_reward'), mean_answer_accuracy=mean('answer_correct'), mean_correct_format=mean('format_valid'),
                mean_probe_probability=mean('probe_probability'), mean_probe_bonus=mean('probe_bonus'),
                dinosaur_rate=mean('dinosaur_present'), ub_rate=mean('ub_ground_truth'),
                honest_pass_rate=sum(r['math_reward'] and r['dinosaur_present'] for r in jobs)/n,
                mean_thinklen=mean('think_tokens'), mean_output_tokens=sum(len(r['output_ids']) for r in jobs)/n,
                truncation_rate=sum(r['finish_reason']['type']=='length' for r in jobs)/n,
                loss=loss_value, grad_norm=grad_norm, adapter_changed=before_hash!=after_hash, adapter_sha256=after_hash,
                generation_seconds=generation_seconds, scoring_seconds=scoring_seconds,
                optimization_seconds=optimization_seconds, sync_seconds=time.monotonic()-sync_start,
                optimized_trajectories=len(active), optimization_microbatch_size=microbatch_size,
                optimization_token_budget=retry_token_budget(opt,microbatch_size), optimizer_step_skipped=not bool(active),
                iteration_seconds=time.monotonic()-start, trainer_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                by_source={source:dict(n=len(group), math_accuracy=sum(r['math_reward'] for r in group)/len(group),
                    probe_probability=sum(r['probe_probability'] for r in group)/len(group),
                    dinosaur_rate=sum(r['dinosaur_present'] for r in group)/len(group))
                    for source in sorted({r['source'] for r in jobs})
                    for group in [[r for r in jobs if r['source']==source]]})
            with (directory/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics,allow_nan=False)+'\n')
            print(json.dumps(metrics,allow_nan=False),flush=True)
            plots.update(metrics)
            if step % run['ckpt_every'] == 0:
                last_checkpoint = save_checkpoint(directory,step,policy,optimizer,config,dataset_hash,rng)
        if config['checkpoint']['save_final'] and (last_checkpoint is None or last_checkpoint.name != f'step_{step:06d}'):
            last_checkpoint = save_checkpoint(directory,step,policy,optimizer,config,dataset_hash,rng)
        write_json(directory/'complete.json',dict(step=step,checkpoint=str(last_checkpoint),adapter_sha256=adapter_hash(policy)))
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    train(parser.parse_args().run_dir)
