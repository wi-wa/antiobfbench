"""Replay the first and last batch, and fixed held-out pairs, after an RL run."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from antiobfbench.mean_probe import LogisticMeanProbe, encode_example, load_text_model


def metrics(labels,p):
    known=np.array([y is not None for y in labels])
    y=np.array([bool(label) for label in labels])[known]
    p=np.asarray(p)[known]
    return dict(n=len(labels),known=int(known.sum()),positives=int(y.sum()),
        accuracy=float(np.where(y,p,1-p).mean()) if len(p) else None,
        auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        mean_probability=float(p.mean()) if len(p) else None)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args()
    run=args.run_dir
    config=json.loads((run/'resolved_config.json').read_text())
    final=json.loads((run/'complete.json').read_text())
    directory=run/'evaluation'
    directory.mkdir(exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(config['model']['path'],local_files_only=True)
    end=tokenizer.convert_tokens_to_ids(config['reward']['end_token'])
    probe=LogisticMeanProbe(config['probe']['path'])
    rows={}
    for key,step in (('first',1),('last',final['step'])):
        with gzip.open(run/'rollouts'/f'step_{step:06d}.jsonl.gz','rt') as f:
            rows[key]=[json.loads(line) for line in f]
        expected = config['rl']['prompts_per_rollout'] * config['rl']['trajectories_per_prompt']
        if len(rows[key]) != expected:
            raise ValueError(f'Incomplete {key} rollout batch: expected {expected}')
    examples=[json.loads(line) for line in (Path(config['probe']['path']).parent/'examples.jsonl').open()
              if json.loads(line)['split'].endswith('test')]
    encoded=[encode_example(tokenizer,r['prompt'],r['text']) for r in examples]
    def score(model,data):
        return probe.score(model,data,tokenizer.pad_token_id,end,batch_size=32,token_budget=32768)
    base=load_text_model(config['model']['path'])
    scores={f'base_{key}':score(base,data) for key,data in rows.items()}
    scores['base_heldout']=score(base,encoded)
    policy=PeftModel.from_pretrained(base,final['checkpoint']).eval()
    scores.update({f'final_{key}':score(policy,data) for key,data in rows.items()})
    scores['final_heldout']=score(policy,encoded)
    result={'environment':config['environment'],'probe_target':config['probe']['target'],
            'accuracy_definition':'mean probability assigned to the correct label',
            'note':'Original tokens replayed; final model after last update. First/last prompts differ. Unknown checker labels excluded; AUROC undefined with fewer than two classes.',
            'rollouts':{},'heldout':{}}
    with (directory/'predictions.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['batch','env_id','rollout','label','base','final'])
        writer.writeheader()
        for key,data in rows.items():
            for model in ('base','final'):
                result['rollouts'][f'{model}_{key}']=metrics([r['probe_ground_truth'] for r in data],scores[f'{model}_{key}'])
            for i,row in enumerate(data):
                writer.writerow(dict(batch=key,env_id=row['env_id'],rollout=row['rollout'],label=row['probe_ground_truth'],
                                     base=scores[f'base_{key}'][i],final=scores[f'final_{key}'][i]))
    for split in ('near_test','far_test'):
        ix=[i for i,r in enumerate(examples) if r['split']==split]
        result['heldout'][split]={model:metrics([examples[i]['label'] for i in ix],scores[f'{model}_heldout'][ix])
                                 for model in ('base','final')}
    # Save concrete detector errors for inspection, without selecting them for aggregate metrics.
    late=rows['last']
    errors=sorted((i for i,r in enumerate(late) if r['probe_ground_truth'] is not None),
                  key=lambda i:scores['base_last'][i] if late[i]['probe_ground_truth'] else 1-scores['base_last'][i])[:8]
    (directory/'largest_probe_errors.jsonl').write_text(''.join(json.dumps({
        **{k:v for k,v in late[i].items() if k not in ('prompt_ids','output_ids')},
        'base_replay_probability':float(scores['base_last'][i]),'final_replay_probability':float(scores['final_last'][i])})+'\n' for i in errors))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    conditions=('base_first','base_last','final_first')
    for ax,metric in zip(axes,('accuracy','auroc')):
        values=[result['rollouts'][c][metric] for c in conditions]
        bars=ax.bar(range(3),[v or 0 for v in values],color=['#3979a4','#ce662d','#688f65'])
        ax.bar_label(bars,labels=['N/A' if v is None else f'{v:.3f}' for v in values],padding=4)
        ax.set_xticks(range(3),['Base / first','Base / last','Final / first'])
        ax.set_ylim(0,1.12)
        ax.set_title('Correct-label probability' if metric=='accuracy' else 'AUROC')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(config['environment'])
    fig.tight_layout()
    fig.savefig(directory/'replay.png',dpi=160)
    fig.savefig(directory/'replay.pdf')
    (directory/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (directory/'complete.json').write_text(json.dumps({'step':final['step']})+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
