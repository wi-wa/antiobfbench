"""Logistic probes on mean-pooled response-token activations of a Gemma text model."""
from pathlib import Path
from contextlib import contextmanager
import numpy as np

EPS = np.finfo(np.float64).eps


def load_text_model(path, attention='sdpa'):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, Gemma3ForCausalLM
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    options = dict(dtype=torch.bfloat16, attn_implementation=attention,
                   local_files_only=True, output_loading_info=True)
    if config.model_type == 'gemma3':
        model, info = Gemma3ForCausalLM.from_pretrained(
            path, config=config.text_config, key_mapping={r'^language_model\.': ''}, **options)
    else:
        model, info = AutoModelForCausalLM.from_pretrained(path, config=config, **options)
    if info['missing_keys'] or info.get('mismatched_keys') or any(
            not k.startswith(('vision_tower.', 'multi_modal_projector.')) for k in info['unexpected_keys']):
        raise RuntimeError(f'Incomplete text-model load: {info}')
    return model.to('cuda').eval()


def encode_example(tokenizer, prompt, text):
    prompt_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                       tokenize=True, add_generation_prompt=True, return_dict=False)
    output_ids = tokenizer(text, add_special_tokens=False).input_ids
    return dict(prompt_ids=prompt_ids, output_ids=output_ids)


def batches(rows, batch_size, token_budget):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['prompt_ids']) + len(rows[i]['output_ids']))
    batch, width = [], 0
    for index in order:
        size = len(rows[index]['prompt_ids']) + len(rows[index]['output_ids'])
        if batch and (len(batch) == batch_size or max(width, size) * (len(batch) + 1) > token_budget):
            yield batch
            batch, width = [], 0
        batch.append(index)
        width = max(width, size)
    if batch:
        yield batch


@contextmanager
def eager_features(model):
    """TF 5.3 hidden-state capture uses context hooks that compiled blocks skip.

    Run feature extraction eagerly, then restore the parent's compiled optimizer path.
    This also matches the exact forward path used when fitting the frozen probe.
    """
    compiled = [(m, m._compiled_call_impl) for m in model.modules()
                if getattr(m, '_compiled_call_impl', None) is not None]
    try:
        for module, _ in compiled:
            module._compiled_call_impl = None
        yield
    finally:
        for module, forward in compiled:
            module._compiled_call_impl = forward


def pooled_features(model, rows, layers, pad_id, batch_size=16, token_budget=8192, progress=None):
    """HF hidden_states[layer], averaged over response content only; no prompt/pad/EOS.

    Layers are 1-based HF hidden-state indices (the final index includes final RMSNorm).
    Pool in FP32, even though the frozen/LoRA model computes in BF16.
    """
    import torch
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if not layers or min(layers) < 1 or max(layers) > base.config.num_hidden_layers:
        raise ValueError('Invalid hidden-state layer')
    output = np.empty((len(rows), len(layers), base.config.hidden_size), dtype=np.float32)
    device = base.get_input_embeddings().weight.device
    done = 0
    with torch.inference_mode(), eager_features(base):
        for indices in batches(rows, batch_size, token_budget):
            width = max(len(rows[i]['prompt_ids']) + len(rows[i]['output_ids']) for i in indices)
            ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
            attention = torch.zeros_like(ids)
            response = torch.zeros_like(ids, dtype=torch.bool)
            for slot, index in enumerate(indices):
                row = rows[index]
                p, n = len(row['prompt_ids']), len(row['output_ids'])
                if not p or not n:
                    raise ValueError('Feature extraction needs nonempty prompt and response content')
                ids[slot, :p+n] = torch.tensor(row['prompt_ids'] + row['output_ids'])
                attention[slot, :p+n] = 1
                response[slot, p:p+n] = True
            ids, attention, response = ids.to(device), attention.to(device), response.to(device)
            mask = ({'full_attention': attention, 'sliding_attention': attention}
                    if base.config.model_type == 'gemma3_text' and
                    base.config._attn_implementation == 'flash_attention_2' else attention)
            result = base.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                output_hidden_states=True, return_dict=True)
            if len(result.hidden_states) != base.config.num_hidden_layers + 1:
                raise RuntimeError('Incomplete hidden-state capture')
            denominator = response.sum(1).float()[:, None]
            for j, layer in enumerate(layers):
                hidden = result.hidden_states[layer].float()
                pooled = (hidden * response[:, :, None]).sum(1) / denominator
                output[indices, j] = pooled.cpu().numpy()
            del result, hidden, pooled, ids, attention, response
            done += len(indices)
            if progress:
                progress(done, len(rows))
    return output


def pooled_features_shared(model, rows, layer, pad_id, token_budget=8192, max_rows=256):
    """`pooled_features` for one layer at lower cost: each distinct prompt is run once, and no layer after `layer` is run.

    Returns float32 [rows, hidden]. The token budget counts response tokens only, since prompts are not repeated.
    """
    import torch
    from .prefix_sharing import PromptPass, distinct_prompts, length_batches
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if not 1 <= layer <= base.config.num_hidden_layers:
        raise ValueError('Invalid hidden-state layer')
    lengths = [len(r['output_ids']) for r in rows]
    if not rows or not all(lengths) or not all(r['prompt_ids'] for r in rows):
        raise ValueError('Feature extraction needs nonempty prompt and response content')
    prompts, which = distinct_prompts([r['prompt_ids'] for r in rows])
    output = np.empty((len(rows), base.config.hidden_size), dtype=np.float32)
    # The last hidden-state index is the final normalised output, so only earlier layers allow an early stop.
    stop = layer if layer < base.config.num_hidden_layers else None
    with torch.inference_mode(), eager_features(base):
        shared = PromptPass(base.model, prompts, pad_id, stop)
        for indices in length_batches(lengths, sorted(range(len(rows)), key=lengths.__getitem__), max_rows, token_budget):
            hidden = shared.completions([rows[i]['output_ids'] for i in indices], [which[i] for i in indices],
                                        pad_id, stop).float()
            size = torch.tensor([lengths[i] for i in indices], device=hidden.device)
            real = torch.arange(hidden.shape[1], device=hidden.device)[None] < size[:, None]
            output[indices] = ((hidden * real[:, :, None]).sum(1) / size[:, None]).cpu().numpy()
    return output


class LogisticMeanProbe:
    def __init__(self, path):
        archive = np.load(path, allow_pickle=False)
        self.weight = archive['weight'].astype(np.float64)
        self.bias = float(archive['bias'])
        self.layer = int(archive['layer'])
        self.path = str(Path(path).resolve())

    def probabilities(self, features):
        from scipy.special import expit
        logits = np.asarray(features, dtype=np.float64) @ self.weight + self.bias
        # Sigmoid is mathematically open; avoid floating-point saturation at exactly 0/1.
        return np.clip(expit(logits), EPS, 1 - EPS)

    def score(self, model, rows, pad_id, end_id, batch_size=16, token_budget=8192, share_prompts=False):
        """Probe probability per rollout. `share_prompts` computes the same features with `pooled_features_shared`."""
        contents, indices = [], []
        for i, row in enumerate(rows):
            tokens = row['output_ids']
            tokens = tokens[:-1] if tokens and tokens[-1] == end_id else tokens
            if tokens:
                contents.append(dict(prompt_ids=row['prompt_ids'], output_ids=tokens))
                indices.append(i)
        result = np.full(len(rows), EPS, dtype=np.float64)
        if contents:
            if share_prompts:
                features = pooled_features_shared(model, contents, self.layer, pad_id, token_budget)
            else:
                features = pooled_features(model, contents, [self.layer], pad_id, batch_size, token_budget)[:, 0]
            result[indices] = self.probabilities(features)
        return result
