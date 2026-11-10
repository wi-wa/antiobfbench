"""Run each distinct prompt once and let all of its completions attend to it.

Rollouts of one prompt share the prompt tokens. `PromptPass` runs the distinct prompts through the model
once and keeps every layer's attention keys and values. `completions` then runs only the completion tokens,
with each row attending to its own prompt's keys and values. Attention is causal, so the hidden states at
the completion positions are the same as when prompt + completion is run for every row.

Prompts are padded on the left, so that index distance equals position distance and one sliding-window rule
serves all rows. Completions are padded on the right, where causal attention already hides the padding.
"""
import torch


class _StopForward(Exception):
    pass


class _Recorder:
    """Takes the place of a transformers Cache in the prompt pass and keeps each layer's keys and values."""

    def __init__(self):
        self.keys, self.values = {}, {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.keys[layer_idx], self.values[layer_idx] = key_states, value_states
        return key_states, value_states


class _Prefix:
    """Takes the place of a Cache in the completion pass: puts each row's prompt keys and values in front."""

    def __init__(self, keys, values, rows):
        self.keys, self.values, self.rows = keys, values, rows

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys = self.keys[layer_idx].index_select(0, self.rows).to(key_states.dtype)
        values = self.values[layer_idx].index_select(0, self.rows).to(value_states.dtype)
        return torch.cat([keys, key_states], dim=2), torch.cat([values, value_states], dim=2)


def _masks(text_model, query_index, key_count, left_pad):
    """Attention masks [rows, 1, queries, keys], True where a query may attend.

    `query_index` is each query's own column among the keys; `left_pad` is the number of padding keys at the
    start of each row. A query on a padding column sees only itself, which keeps its row finite.
    """
    config = text_model.config
    keys = torch.arange(key_count, device=query_index.device)
    causal = keys[None, :] <= query_index[:, None]
    own = keys[None, :] == query_index[:, None]
    window = keys[None, :] > query_index[:, None] - config.sliding_window
    real = keys[None, None, :] >= left_pad[:, None, None]
    masks = {'full_attention': ((causal[None] & real) | own[None])[:, None],
             'sliding_attention': (((causal & window)[None] & real) | own[None])[:, None]}
    if config._attn_implementation == 'eager':
        dtype = text_model.embed_tokens.weight.dtype
        masks = {name: torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill_(~mask, torch.finfo(dtype).min)
                 for name, mask in masks.items()}
    elif config._attn_implementation != 'sdpa':
        raise ValueError('Prompt sharing supports the sdpa and eager attention implementations')
    return masks


def _run(text_model, stop_layer, **inputs):
    """The text model's final (normalised) hidden states, or the raw output of layer `stop_layer` (1-based) without running anything after it."""
    if stop_layer is None:
        return text_model(**inputs).last_hidden_state
    if not 1 <= stop_layer <= text_model.config.num_hidden_layers:
        raise ValueError('stop_layer must be a 1-based layer number')
    captured = []

    def stop(module, args, output):
        captured.append(output)
        raise _StopForward

    handle = text_model.layers[stop_layer - 1].register_forward_hook(stop)
    try:
        text_model(**inputs)
    except _StopForward:
        pass
    finally:
        handle.remove()
    return captured[0]


class PromptPass:
    """One forward pass over distinct prompts (lists of token ids), kept for their completions."""

    def __init__(self, text_model, prompts, pad_id, stop_layer=None):
        self.model = text_model
        device = text_model.embed_tokens.weight.device
        lengths = torch.tensor([len(p) for p in prompts], device=device)
        if not prompts or int(lengths.min()) < 1:
            raise ValueError('Prompt sharing needs nonempty prompts')
        self.lengths = lengths
        self.width = int(lengths.max())
        self.left_pad = self.width - lengths
        ids = torch.full((len(prompts), self.width), pad_id, dtype=torch.long)
        for row, prompt in enumerate(prompts):
            ids[row, self.width - len(prompt):] = torch.tensor(prompt)
        index = torch.arange(self.width, device=device)
        recorder = _Recorder()
        _run(text_model, stop_layer, input_ids=ids.to(device), attention_mask=_masks(text_model, index, self.width, self.left_pad),
             position_ids=(index[None] - self.left_pad[:, None]).clamp_(min=0), past_key_values=recorder,
             cache_position=index, use_cache=True)
        layers = range(stop_layer or text_model.config.num_hidden_layers)
        self.keys = [recorder.keys[i] for i in layers]
        self.values = [recorder.values[i] for i in layers]

    def completions(self, token_rows, prompt_index, pad_id, stop_layer=None, keys=None, values=None):
        """Hidden states [rows, longest row, hidden] of completion tokens; row i continues prompt `prompt_index[i]`."""
        device = self.lengths.device
        width = max(len(tokens) for tokens in token_rows)
        ids = torch.full((len(token_rows), width), pad_id, dtype=torch.long)
        for row, tokens in enumerate(token_rows):
            ids[row, :len(tokens)] = torch.tensor(tokens)
        which = torch.as_tensor(prompt_index, device=device)
        steps = torch.arange(width, device=device)
        query = self.width + steps
        cache = _Prefix(self.keys if keys is None else keys, self.values if values is None else values, which)
        return _run(self.model, stop_layer, input_ids=ids.to(device),
                    attention_mask=_masks(self.model, query, self.width + width, self.left_pad[which]),
                    position_ids=self.lengths[which][:, None] + steps[None], past_key_values=cache,
                    cache_position=query, use_cache=True)


def distinct_prompts(prompts):
    """(distinct prompts in first-seen order, index of each input prompt among them)."""
    seen, distinct, index = {}, [], []
    for prompt in prompts:
        key = tuple(prompt)
        if key not in seen:
            seen[key] = len(distinct)
            distinct.append(list(prompt))
        index.append(seen[key])
    return distinct, index


def length_batches(costs, order, max_rows, max_cost):
    """Pack indices, taken in `order` (sorted by cost), into batches of at most `max_rows` rows whose padded cost stays within `max_cost`."""
    batch, widest = [], 0
    for index in order:
        wider = max(widest, costs[index])
        if batch and (len(batch) == max_rows or (max_cost and wider * (len(batch) + 1) > max_cost)):
            yield batch
            batch, wider = [], costs[index]
        batch.append(index)
        widest = wider
    if batch:
        yield batch
