"""Single-GPU SGLang and PEFT training helpers extracted from antiobf."""
import hashlib
import json
from multiprocessing import resource_tracker
import os

import torch
from peft import get_peft_model_state_dict
import gemma_server  # Applied in spawned SGLang workers too.
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import kill_process_tree
from antiobfbench.prefix_sharing import PromptPass, distinct_prompts, length_batches
from rl_core import collate_trajectories, optimization_batches, training_attention_mask
from rl_loss import frozen_head_loss

class ColocatedEngine(Engine):
    def shutdown(self):
        from torch._inductor.async_compile import shutdown_compile_workers
        # Let compiler pools release their semaphores before SGLang stops children.
        shutdown_compile_workers()
        manager = getattr(self, 'tokenizer_manager', None)
        watchdog = getattr(manager, '_subprocess_watchdog', None)
        if watchdog is not None:
            watchdog.stop()
        # SGLang kills every child, including Python's still-needed semaphore tracker.
        # Leave that helper alive for Python's normal finalization of trainer state.
        tracker_pid = resource_tracker._resource_tracker._pid
        kill_process_tree(os.getpid(), include_parent=False, skip_pid=tracker_pid)


def checked(result, operation):
    if result is False or (isinstance(result, dict) and result.get('success') is False):
        raise RuntimeError(f'{operation} failed: {result}')
    return result


def adapter_hash(policy):
    h = hashlib.sha256()
    for name, tensor in sorted(get_peft_model_state_dict(policy).items()):
        h.update(name.encode())
        h.update(tensor.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def sync_adapter(engine, policy, config, version, previous=None):
    name = f'policy-{version}'
    tensors = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(policy).items()}
    adapter_config = {'r': config['lora']['rank'], 'lora_alpha': config['lora']['alpha'],
                      'target_modules': config['lora']['target_modules'],
                      'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'bias': 'none'}
    checked(engine.flush_cache(), 'flush stale policy cache')
    checked(engine.load_lora_adapter_from_tensors(name, tensors, adapter_config), 'load LoRA')
    if previous is not None:
        checked(engine.unload_lora_adapter(previous), 'unload previous LoRA')
    return name


def completion_loss(policy, rows, pad_id, temperature, chunk_size, coefficients, total_trajectories,
                    loss_backend='torch'):
    device = next(policy.parameters()).device
    ids, attention, mask = collate_trajectories(rows, pad_id, device)
    base = policy.get_base_model()
    # Gemma's mixed sliding/full layers need a mask dictionary; Llama uses 2-D padding.
    attention = training_attention_mask(base.config, attention)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = base.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
        selected = hidden[:, :-1][mask]
        targets = ids[:, 1:][mask]
        owners = torch.arange(len(rows), device=device)[:, None].expand_as(mask)[mask]
        return frozen_head_loss(selected, base.lm_head.weight, targets,
                                coefficients[owners] / total_trajectories,
                                temperature, getattr(base.config, 'final_logit_softcapping', None), chunk_size, loss_backend)


def backward_rollouts(policy, jobs, active, coefficients, pad_id, temperature, chunk_size, microbatch_size,
                      max_tokens=0, loss_backend='torch'):
    total = torch.zeros((), device=coefficients.device)
    for indices in optimization_batches(jobs, active, microbatch_size, max_tokens):
        batch = [jobs[i] for i in indices]
        loss = completion_loss(policy, batch, pad_id, temperature, chunk_size,
                               coefficients[indices], len(jobs), loss_backend)
        total += loss.detach()
        loss.backward()
    return total.item()


def backward_rollouts_shared(policy, jobs, active, coefficients, pad_id, temperature, chunk_size, microbatch_size,
                             max_tokens=0, loss_backend='torch'):
    """`backward_rollouts` with each distinct prompt run once: the same loss and gradients for less work.

    Prompts (without their last token) are run once with gradients. Their completions are run in length-sorted
    batches that attend to the stored prompt keys and values; the gradients with respect to those keys and
    values are summed in float32 and sent through the prompt pass once. A row's input is the last prompt token
    followed by its completion, so every predicted token comes from a completion pass.

    A prompt pass stays in memory while its completions are processed, so both share the token budget: prompts
    are taken a few at a time, within a quarter of it, and each completion batch gets what is left.
    """
    if not active:
        return 0.0
    base = policy.get_base_model()
    device = coefficients.device
    softcap = getattr(base.config, 'final_logit_softcapping', None)
    prompts, which = distinct_prompts([jobs[i]['prompt_ids'][:-1] for i in active])
    members = [[] for _ in prompts]
    for index, prompt in zip(active, which):
        members[prompt].append(index)
    sizes = [len(prompt) for prompt in prompts]
    total = torch.zeros((), device=device)
    for chosen in length_batches(sizes, sorted(range(len(prompts)), key=sizes.__getitem__), len(prompts), max_tokens // 4):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            shared = PromptPass(base.model, [prompts[g] for g in chosen], pad_id)
        # Leaves in float32, so sums over rows and over batches do not round in bfloat16.
        keys = [k.detach().float().requires_grad_(True) for k in shared.keys]
        values = [v.detach().float().requires_grad_(True) for v in shared.values]
        position = {g: n for n, g in enumerate(chosen)}
        owner = {i: position[g] for g in chosen for i in members[g]}
        # A row costs its completion tokens, plus a little for its copy of the prompt keys and values.
        costs = {i: len(jobs[i]['output_ids']) + shared.width // 32 for i in owner}
        budget = max(max_tokens - len(chosen) * shared.width, max_tokens // 4) if max_tokens else 0
        for indices in length_batches(costs, sorted(owner, key=costs.__getitem__), 16 * microbatch_size, budget):
            rows = [jobs[i] for i in indices]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = shared.completions([[r['prompt_ids'][-1]] + r['output_ids'][:-1] for r in rows],
                                            [owner[i] for i in indices], pad_id, keys=keys, values=values)
                lengths = torch.tensor([len(r['output_ids']) for r in rows], device=device)
                mask = torch.arange(hidden.shape[1], device=device)[None] < lengths[:, None]
                targets = torch.tensor([t for r in rows for t in r['output_ids']], device=device)
                owners = torch.arange(len(rows), device=device)[:, None].expand_as(mask)[mask]
                loss = frozen_head_loss(hidden[mask], base.lm_head.weight, targets,
                                        coefficients[indices][owners] / len(jobs), temperature, softcap, chunk_size, loss_backend)
            total += loss.detach()
            loss.backward()
            del hidden, loss
        pairs = [(kept, leaf.grad.to(kept.dtype)) for kept, leaf in zip(shared.keys + shared.values, keys + values)
                 if leaf.grad is not None]
        torch.autograd.backward([kept for kept, _ in pairs], [grad for _, grad in pairs])
        del shared, keys, values, pairs
    return total.item()


def configure_training(policy, optim):
    if optim['gradient_checkpointing']:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        policy.enable_input_require_grads()
    policy.config.use_cache = False
    if optim.get('compile_blocks', False):
        if optim['attention_backend'] == 'flash_attention_2':
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            # HF's padding helper reads lengths back to the host. Keep that
            # external attention call eager instead of recompiling per length.
            ALL_ATTENTION_FUNCTIONS.register('flash_attention_2', torch.compiler.disable(
                ALL_ATTENTION_FUNCTIONS['flash_attention_2']))
        # Compile in place so PEFT checkpoint/synchronization parameter names stay intact.
        for layer in policy.get_base_model().model.layers:
            layer.compile(dynamic=True, options={'emulate_precision_casts': True})


def save_checkpoint(directory, step, policy, optimizer, config, dataset_hash, rng):
    path = directory / 'checkpoints' / f'step_{step:06d}'
    path.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    state = {'step': step, 'optimizer': optimizer.state_dict(), 'config': config,
             'dataset_sha256': dataset_hash, 'python_rng': rng.getstate(),
             'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
             'adapter_sha256': adapter_hash(policy)}
    torch.save(state, path / 'trainer_state.pt')
    (path / 'complete.json').write_text(json.dumps({'step': step, 'adapter_sha256': state['adapter_sha256']}) + '\n')
    return path
