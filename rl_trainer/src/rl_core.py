"""REINFORCE utilities extracted unchanged from antiobf."""

def training_attention_mask(model_config, attention):
    if (model_config.model_type == 'gemma3_text'
            and model_config._attn_implementation == 'flash_attention_2'):
        return {'full_attention': attention, 'sliding_attention': attention}
    return attention


def validate_resume_config(saved, current):
    # Execution kernels and batching can change; the learning objective and
    # optimizer hyperparameters must still match the checkpoint exactly.
    performance_options = {'microbatch_size', 'max_tokens_per_microbatch', 'attention_backend',
                           'loss_backend', 'compile_blocks', 'gradient_checkpointing', 'logit_chunk_size',
                           'share_prompts'}
    for section in ['model', 'data', 'lora', 'rl', 'optim', 'reward']:
        before, after = saved[section], current[section]
        if section == 'optim':
            before = {k: v for k, v in before.items() if k not in performance_options}
            after = {k: v for k, v in after.items() if k not in performance_options}
        if before != after:
            raise ValueError(f'Resume configuration changed: {section}')


def thinking_text(text):
    """First reasoning block, retaining an unfinished block at the output limit."""
    _, opening, remainder = text.partition('<think>')
    return remainder.partition('</think>')[0] if opening else ''


def advantages(rewards, trajectories_per_prompt, baseline):
    import torch
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if baseline == 'none':
        return values
    grouped = values.reshape(-1, trajectories_per_prompt)
    if trajectories_per_prompt < 2:
        raise ValueError('Leave-one-out requires multiple trajectories.')
    others = (grouped.sum(dim=1, keepdim=True) - grouped) / (trajectories_per_prompt - 1)
    return (grouped - others).flatten()


def optimization_batches(jobs, active, microbatch_size, max_tokens=0):
    """Pack length-sorted indices within a padded-token budget, without dropping rows."""
    batch, width = [], 0
    for index in active:
        length = len(jobs[index]['prompt_ids']) + len(jobs[index]['output_ids'])
        next_width = max(width, length)
        if batch and (len(batch) == microbatch_size or
                      (max_tokens and next_width * (len(batch) + 1) > max_tokens)):
            yield batch
            batch, width = [], 0
        batch.append(index)
        width = max(width, length)
    if batch:
        yield batch


def retry_token_budget(optim, current_microbatch_size):
    budget = optim['max_tokens_per_microbatch']
    if not budget:
        return 0
    return max(1, budget * current_microbatch_size // optim['microbatch_size'])


def collate_trajectories(rows, pad_id, device):
    import torch
    lengths = [len(r['prompt_ids']) + len(r['output_ids']) for r in rows]
    if any(not r['prompt_ids'] or not r['output_ids'] for r in rows):
        raise ValueError('Prompt and completion token sequences must be nonempty.')
    width = max(lengths)
    # Assemble on CPU, then transfer once per tensor instead of many tiny CUDA writes.
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(rows), width - 1), dtype=torch.bool)
    for i, row in enumerate(rows):
        tokens = row['prompt_ids'] + row['output_ids']
        ids[i, :len(tokens)] = torch.tensor(tokens)
        attention[i, :len(tokens)] = 1
        mask[i, len(row['prompt_ids']) - 1:len(tokens) - 1] = True
    return ids.to(device), attention.to(device), mask.to(device)
