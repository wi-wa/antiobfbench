"""Logistic probes on mean-pooled response-token activations of a Gemma text model."""
from pathlib import Path
from contextlib import contextmanager
import numpy as np

EPS = np.finfo(np.float64).eps


def load_text_model(path, attention='flash_attention_2'):
    import torch
    from transformers import AutoConfig, Gemma3ForCausalLM
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    model, info = Gemma3ForCausalLM.from_pretrained(
        path, config=config.text_config, key_mapping={r'^language_model\.': ''},
        dtype=torch.bfloat16, attn_implementation=attention, local_files_only=True,
        output_loading_info=True)
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
            ids, attention, response = ids.cuda(), attention.cuda(), response.cuda()
            mask = ({'full_attention': attention, 'sliding_attention': attention}
                    if base.config._attn_implementation == 'flash_attention_2' else attention)
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

    def score(self, model, rows, pad_id, end_id, batch_size=16, token_budget=8192):
        contents, indices = [], []
        for i, row in enumerate(rows):
            tokens = row['output_ids']
            tokens = tokens[:-1] if tokens and tokens[-1] == end_id else tokens
            if tokens:
                contents.append(dict(prompt_ids=row['prompt_ids'], output_ids=tokens))
                indices.append(i)
        result = np.full(len(rows), EPS, dtype=np.float64)
        if contents:
            features = pooled_features(model, contents, [self.layer], pad_id, batch_size, token_budget)
            result[indices] = self.probabilities(features[:, 0])
        return result
