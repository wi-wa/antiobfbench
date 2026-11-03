#!/usr/bin/env python3
"""Extract Gemma activations, select a mean logistic probe using train-only validation, and evaluate."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench.mean_probe import EPS, encode_example, load_text_model, pooled_features


def metrics(y, p):
    return dict(n=len(y), auroc=float(roc_auc_score(y, p)), accuracy=float(accuracy_score(y, p >= .5)),
                log_loss=float(log_loss(y, p, labels=[0, 1])), brier=float(brier_score_loss(y, p)),
                positive_mean=float(p[y == 1].mean()), negative_mean=float(p[y == 0].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--seed', type=int, default=20261026)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, hashes = [], {}
    for split in ('near_train', 'far_train', 'near_test', 'far_test'):
        path = ROOT.parent / 'envs/09_math_dinosaurs' / f'probe_{split}.jsonl'
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows += [dict(json.loads(line), split=split) for line in path.open()]
    (args.output / 'examples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    train = np.array([r['split'].endswith('train') for r in rows])
    labels = np.array([int(r['label']) for r in rows])
    pair_ids = sorted({r['pair_id'] for r in rows if r['split'].endswith('train')})
    random.Random(args.seed).shuffle(pair_ids)
    valid_pairs = set(pair_ids[:max(1, round(.2 * len(pair_ids)))])
    valid = train & np.array([r['pair_id'] in valid_pairs for r in rows])
    fit = train & ~valid
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_text_model(args.model)
    layers = list(range(1, model.config.num_hidden_layers + 1))
    encoded = [encode_example(tok, r['prompt'], r['text']) for r in rows]
    started = time.monotonic()
    features = pooled_features(model, encoded, layers, tok.pad_token_id,
        progress=lambda done, total: print(f'features {done}/{total} {time.monotonic()-started:.0f}s', flush=True))
    np.save(args.output / 'features.npy', features)
    del model
    torch.cuda.empty_cache()
    trials = []
    for j, layer in enumerate(layers):
        for c in (.001, .01, .1, 1.0, 10.0):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, solver='lbfgs',
                                      max_iter=2000, random_state=args.seed))
            pipe.fit(features[fit, j], labels[fit])
            scores = metrics(labels[valid], pipe.predict_proba(features[valid, j])[:, 1])
            trials.append(dict(layer=layer, C=c, **scores))
        best_here = max(trials[-5:], key=lambda r: (r['auroc'], -r['log_loss']))
        print('selection', json.dumps(best_here), flush=True)
    best = max(trials, key=lambda r: (r['auroc'], -r['log_loss'], -r['layer']))
    layer_index = layers.index(best['layer'])
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best['C'], solver='lbfgs',
                              max_iter=2000, random_state=args.seed))
    pipe.fit(features[train, layer_index], labels[train])
    scaler, classifier = pipe.steps[0][1], pipe.steps[1][1]
    weight = classifier.coef_[0] / scaler.scale_
    bias = classifier.intercept_[0] - weight @ scaler.mean_
    np.savez(args.output / 'probe.npz', weight=weight, bias=bias, layer=best['layer'])
    probabilities = np.clip(expit(features[:, layer_index].astype(np.float64) @ weight + bias), EPS, 1-EPS)
    assert np.allclose(probabilities, pipe.predict_proba(features[:, layer_index])[:, 1], atol=1e-10)
    assert np.all((probabilities > 0) & (probabilities < 1))
    result = dict(model=args.model, seed=args.seed, representation='mean response-token HF hidden states',
                  excludes='prompt, padding and terminal end-of-turn token', label='dinosaur_presence',
                  probability='sigmoid(weight @ mean_hidden + bias), clipped only inside floating-point (0,1)',
                  selection=best, fit_pairs=len(pair_ids)-len(valid_pairs), validation_pairs=sorted(valid_pairs),
                  final_fit_examples=int(train.sum()), dataset_hashes=hashes,
                  versions={p:importlib.metadata.version(p) for p in ['torch','transformers','scikit-learn']},
                  metrics={split:metrics(labels[np.array([r['split']==split for r in rows])],
                                           probabilities[np.array([r['split']==split for r in rows])])
                           for split in ('near_train','far_train','near_test','far_test')})
    (args.output/'selection.json').write_text(json.dumps(trials, indent=2)+'\n')
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.output/'predictions.jsonl').open('w') as f:
        for row,p in zip(rows,probabilities):
            f.write(json.dumps(dict(id=row['id'], pair_id=row['pair_id'], split=row['split'],
                                   label=row['label'], probability=float(p)))+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for split in ('near_test','far_test'):
        mask=np.array([r['split']==split for r in rows])
        fpr,tpr,_=roc_curve(labels[mask],probabilities[mask])
        axes[0].plot(fpr,tpr,label=f"{split}: {result['metrics'][split]['auroc']:.3f}")
    axes[0].plot([0,1],[0,1],'--',color='gray');axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out ROC')
    axes[0].legend()
    for label in (0,1):
        mask=(~train)&(labels==label)
        axes[1].hist(probabilities[mask],bins=np.linspace(0,1,21),alpha=.6,label='presence' if label else 'absence')
    axes[1].set(xlabel='Probe probability',ylabel='Examples',title='Held-out probability distributions');axes[1].legend()
    fig.tight_layout();fig.savefig(args.output/'probe_evaluation.png',dpi=140);plt.close(fig)
    (args.output/'complete.json').write_text(json.dumps(dict(probe_sha256=hashlib.sha256((args.output/'probe.npz').read_bytes()).hexdigest(),layer=best['layer']))+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
