#!/usr/bin/env python3
"""Fit math-source mixture weights and validate them on fresh questions with the base model."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from scipy.optimize import minimize
from transformers import AutoTokenizer
from antiobfbench.envs.e09_math_dinosaurs import ENV,PROMPT,settings,stratum
from reinforce_worker import ColocatedEngine


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cache',type=Path,help='Reuse baseline responses for identical questions and answers')
    ap.add_argument('--gpu',type=int,default=1)
    ap.add_argument('--kv-tokens',type=int,default=60000)
    ap.add_argument('--max-running',type=int,default=64)
    ap.add_argument('--mem-fraction',type=float,default=.28)
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--fit-per-stratum',type=int,default=64)
    ap.add_argument('--validation-tasks',type=int,default=384)
    ap.add_argument('--rollouts',type=int,default=2)
    ap.add_argument('--max-tokens',type=int,default=2048)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rng=random.Random(args.seed)
    cache={}
    if args.cache:
        cached_metadata=json.loads((args.cache.parent/'metadata.json').read_text())
        if cached_metadata['model'] != args.model or cached_metadata['max_tokens'] != args.max_tokens or cached_metadata['temperature'] != 1:
            raise ValueError('Cached calibration model or generation settings differ')
        cache={(r['id'],r['rollout']):r for r in map(json.loads,args.cache.open())}
    rows=[json.loads(l) for l in args.data.open()];rows=[r for r in rows if r['split']=='train']
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True);end=tok.convert_tokens_to_ids('<end_of_turn>')
    groups=defaultdict(list);excluded=[];unrated=[]
    for r in rows:
        if stratum(r) not in settings()['profiles']['target30']:
            unrated.append(r['id']);continue
        r['prompt']=PROMPT.format(problem=r['problem'])
        r['prompt_ids']=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,return_dict=False)
        if len(r['prompt_ids'])+args.max_tokens>4096:excluded.append(r['id']);continue
        groups[stratum(r)].append(r)
    keys=sorted(groups)
    fit=[]
    for key in keys:fit+=rng.sample(groups[key],args.fit_per_stratum)
    used={r['id'] for r in fit}
    metadata=dict(model=args.model,seed=args.seed,gpu=args.gpu,temperature=1,max_tokens=args.max_tokens,
                  rollouts_per_question=args.rollouts,cache=str(args.cache) if args.cache else None,cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest() if args.cache else None,kv_tokens=args.kv_tokens,max_running=args.max_running,mem_fraction=args.mem_fraction,data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  target_metric='answer_correct',target=.3,excluded_context_ids=excluded,excluded_strata_ids=unrated,
                  fit_questions=len(fit),selection='Stratified random sample from RL pool; probe train/test questions excluded by original split. Validation questions are disjoint from fitting questions.')
    write(args.output/'metadata.json',metadata)
    engine=None
    try:
        engine=ColocatedEngine(model_path=args.model,host='127.0.0.1',port=30211,dtype='bfloat16',tp_size=1,base_gpu_id=0,
            attention_backend='triton',triton_attention_reduce_in_fp32=True,chunked_prefill_size=2048,context_length=4096,
            mem_fraction_static=args.mem_fraction,max_total_tokens=args.kv_tokens,max_running_requests=args.max_running,cuda_graph_max_bs=args.max_running,
            random_seed=args.seed,decode_log_interval=200,log_level='warning')
        def generate(tasks,stage):
            jobs=[];results=[]
            for task in tasks:
                for roll in range(args.rollouts):jobs.append((task,roll,rng.randrange(2**31)))
            with (args.output/f'{stage}.jsonl').open('w') as f:
                for first in range(0,len(jobs),256):
                    batch=jobs[first:first+256]
                    pending=[j for j in batch if (j[0]['id'],j[1]) not in cache]
                    responses=engine.generate(input_ids=[t['prompt_ids'] for t,_,_ in pending],
                        sampling_params=[dict(temperature=1.,top_p=1.,top_k=-1,min_p=0.,max_new_tokens=args.max_tokens,
                            sampling_seed=seed,skip_special_tokens=False,stop_token_ids=[end]) for _,_,seed in pending],return_logprob=False) if pending else []
                    responses=iter(responses)
                    for task,roll,seed in batch:
                        cached=cache.get((task['id'],roll))
                        if cached is not None:
                            assert cached['problem']==task['problem'] and cached['answer']==task['answer']
                            row=dict(cached,reused_from=str(args.cache));f.write(json.dumps(row,ensure_ascii=False)+'\n');results.append(row);continue
                        response=next(responses)
                        ids=response['output_ids'];finish=response['meta_info']['finish_reason']['type'];ended=bool(ids and ids[-1]==end and finish=='stop')
                        body=tok.decode(ids[:-1] if ids and ids[-1]==end else ids,skip_special_tokens=False)
                        correct=ENV.answer_reward(task,body);valid=ENV.format_valid(body,ended)
                        row=dict(id=task['id'],stratum=stratum(task),source=task['source'],problem=task['problem'],answer=task['answer'],
                            response=body,answer_correct=correct,format_valid=valid,reward=correct*valid,rollout=roll,seed=seed,
                            completion_tokens=len(ids),truncated=finish=='length')
                        f.write(json.dumps(row,ensure_ascii=False)+'\n');results.append(row)
                    f.flush();print(json.dumps(dict(stage=stage,done=len(results),total=len(jobs))),flush=True)
            return results
        fit_results=generate(fit,'fit')
        stats={}
        for key in keys:
            rs=[r for r in fit_results if r['stratum']==key]
            stats[key]={k:float(np.mean([r[k] for r in rs])) for k in ['answer_correct','format_valid','reward','truncated','completion_tokens']}
        prior=np.array([settings()['profiles']['target30'][k] for k in keys])
        def fit_weights(metric):
            rates=np.array([stats[k][metric] for k in keys])
            solved=minimize(lambda w:np.sum((w-prior)**2),prior,method='SLSQP',
                bounds=[(.03,min(.65,(len(groups[k])-args.fit_per_stratum)/args.validation_tasks)) for k in keys],
                constraints=[{'type':'eq','fun':lambda w:w.sum()-1},{'type':'eq','fun':lambda w:w@rates-.3}],
                options={'ftol':1e-12,'maxiter':1000})
            if not solved.success:raise RuntimeError(f'No diverse 30% mixture for {metric}: {solved.message}; rates={rates}')
            return {k:float(w) for k,w in zip(keys,solved.x)}
        weights=fit_weights('answer_correct')
        valid_weights=fit_weights('reward')
        write(args.output/'proposal.json',dict(weights=weights,valid30_weights=valid_weights,fit_rates=stats,predicted_accuracy=.3))
        # Fix the candidate mixture before generating an independent validation sample.
        counts={k:int(args.validation_tasks*weights[k]) for k in keys}
        for k in sorted(keys,key=lambda k:args.validation_tasks*weights[k]-counts[k],reverse=True)[:args.validation_tasks-sum(counts.values())]:counts[k]+=1
        validation=[]
        for key in keys:
            eligible=[r for r in groups[key] if r['id'] not in used]
            validation+=rng.sample(eligible,counts[key])
        result=generate(validation,'validation')
        metrics={k:float(np.mean([r[k] for r in result])) for k in ['answer_correct','format_valid','reward','truncated','completion_tokens']}
        by_question=defaultdict(list)
        for r in result:by_question[r['id']].append(r['answer_correct'])
        qmeans=np.array([np.mean(v) for v in by_question.values()]);boot=np.random.default_rng(args.seed).choice(qmeans,(2000,len(qmeans)),replace=True).mean(1)
        validation_rates={key:{metric:float(np.mean([r[metric] for r in result if r['stratum']==key]))
                              for metric in ['answer_correct','reward','format_valid']} for key in keys}
        valid30_metrics={metric:sum(valid_weights[key]*validation_rates[key][metric] for key in keys)
                         for metric in ['answer_correct','reward','format_valid']}
        report=dict(weights=weights,valid30_weights=valid_weights,valid30_validation_reweighted=valid30_metrics,validation_by_stratum=validation_rates,fit_rates=stats,validation_questions=len(validation),validation_rollouts=len(result),
                    validation_counts=counts,validation=metrics,answer_accuracy_95pct_cluster_bootstrap=np.quantile(boot,[.025,.975]).tolist(),
                    target=.3,within_five_percentage_points=bool(abs(metrics['answer_correct']-.3)<=.05),metadata=metadata)
        write(args.output/'report.json',report)
        print(json.dumps(report,indent=2),flush=True)
    finally:
        if engine is not None:engine.shutdown()


if __name__=='__main__':main()
