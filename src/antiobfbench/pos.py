"""Part-of-speech tags from a frozen averaged perceptron: NLTK's averaged_perceptron_tagger_eng, trained on Penn Treebank news text. Pure Python, about 20 microseconds a word."""
import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data' / 'pos_tagger.json'
START, END = ['-START-', '-START2-'], ['-END-', '-END2-']


@lru_cache(maxsize=None)
def _model():
    model = json.loads(DATA.read_text(encoding='utf8'))
    return model['weights'], model['tagdict'], sorted(model['classes'])


def _context(word):
    if '-' in word and word[0] != '-':
        return '!HYPHEN'
    if word.isdigit() and len(word) == 4:
        return '!YEAR'
    if word and word[0].isdigit():
        return '!DIGITS'
    return word.lower()


def pos_tags(tokens):
    """One Penn Treebank tag for each token, as NLTK's PerceptronTagger gives it."""
    weights, tagdict, classes = _model()
    prev, prev2 = START
    context = START + [_context(word) for word in tokens] + END
    out = []
    for i, word in enumerate(tokens, 2):
        tag = tagdict.get(word)
        if not tag:
            features = ('bias', f'i suffix {word[-3:]}', f'i pref1 {word[:1]}', f'i-1 tag {prev}', f'i-2 tag {prev2}', f'i tag+i-2 tag {prev} {prev2}', f'i word {context[i]}',
                        f'i-1 tag+i word {prev} {context[i]}', f'i-1 word {context[i - 1]}', f'i-1 suffix {context[i - 1][-3:]}', f'i-2 word {context[i - 2]}', f'i+1 word {context[i + 1]}',
                        f'i+1 suffix {context[i + 1][-3:]}', f'i+2 word {context[i + 2]}')
            scores = dict.fromkeys(classes, 0.)
            for feature in features:
                for label, weight in weights.get(feature, {}).items():
                    scores[label] += weight
            tag = max(classes, key=lambda label: (scores[label], label))
        out.append(tag)
        prev2, prev = prev, tag
    return out
