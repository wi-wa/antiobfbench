"""Frozen sentence similarity from all-MiniLM-L6-v2 (mean pooling, cosine), cached per pair of texts."""
import threading

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
_parts = None
_vectors = {}
_lock = threading.Lock()


def _load():
    global _parts
    if _parts is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(8)    # birget is shared; torch would otherwise use every core
        _parts = (AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True), AutoModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True).eval(), torch)
    return _parts


def embed(texts):
    """Compute and keep the vectors of any texts not seen yet."""
    with _lock:
        todo = sorted({t for t in texts if t not in _vectors})
        if todo:
            tok, model, torch = _load()
            for first in range(0, len(todo), 64):
                part = todo[first:first + 64]
                enc = tok(part, return_tensors='pt', padding=True, truncation=True, max_length=256)
                with torch.inference_mode():
                    hidden = model(**enc).last_hidden_state
                mask = enc['attention_mask'].unsqueeze(-1)
                vectors = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
                _vectors.update(zip(part, vectors))


def similarity(a, b):
    """Cosine similarity of two texts."""
    embed([a, b])
    return float((_vectors[a] * _vectors[b]).sum())
