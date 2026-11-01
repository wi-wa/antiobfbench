"""Frozen sentence positivity from DistilBERT SST-2, cached per sentence and shared by the writing environments."""
import threading

MODEL = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
REVISION = '714eb0fa89d2f80546fda750413ed43d93601a13'
_pipe = None
_cache = {}
_lock = threading.Lock()


def _load():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        torch.set_num_threads(8)    # birget is shared; torch would otherwise use every core
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        _pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=-1, truncation=True)
    return _pipe


def positivity(sentences):
    """P(positive) for each sentence."""
    with _lock:
        todo = sorted({s for s in sentences if s not in _cache})
        if todo:
            for s, r in zip(todo, _load()(todo, batch_size=64)):
                _cache[s] = r['score'] if r['label'] == 'POSITIVE' else 1 - r['score']
        return [_cache[s] for s in sentences]
