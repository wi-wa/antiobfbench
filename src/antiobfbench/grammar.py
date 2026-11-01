"""Frozen grammatical acceptability from DistilBERT trained on CoLA, cached per text."""
import re
import threading

from .parse import sentences

MODEL = 'textattack/distilbert-base-uncased-CoLA'
REVISION = 'a4987947954e3b9717c81605155d423c7b9be0a5'
_parts = None
_cache = {}
_lock = threading.Lock()


def _load():
    global _parts
    if _parts is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(8)    # birget is shared; torch would otherwise use every core
        _parts = (AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True),
                  AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION, local_files_only=True).eval(), torch)
    return _parts


def acceptability(texts):
    """P(acceptable) for each text."""
    with _lock:
        todo = sorted({t for t in texts if t not in _cache})
        if todo:
            tok, model, torch = _load()
            for first in range(0, len(todo), 64):
                part = todo[first:first + 64]
                enc = tok(part, return_tensors='pt', padding=True, truncation=True, max_length=128)
                with torch.inference_mode():
                    scores = torch.softmax(model(**enc).logits, -1)[:, 1].tolist()
                _cache.update(zip(part, scores))
        return [_cache[t] for t in texts]


SHORT = 3       # sentences with fewer tokens count as unacceptable, so that "A doctor. A church." doesn't pass as prose


def _size(text):
    return len(re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text))


def plain(text):
    """The text without markdown or quotation marks, on one line."""
    return ' '.join(re.sub(r'[*_#`~"“”‘’\u200b\ufeff]', '', text).split())


def to_score(text):
    """The sentences that the classifier has to read."""
    return [part for part in sentences(plain(text)) if _size(part) >= SHORT]


def fluency(text):
    """Acceptability of a text: the mean over its sentences weighted by length, where short sentences count as 0."""
    text = plain(text)
    parts = sentences(text) or [text]
    long_enough = [part for part in parts if _size(part) >= SHORT]
    scores = dict(zip(long_enough, acceptability(long_enough)))
    return sum(scores.get(part, 0.) * _size(part) for part in parts) / max(1, sum(_size(part) for part in parts))
