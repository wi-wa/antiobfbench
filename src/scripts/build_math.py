#!/usr/bin/env python3
"""Build a pinned, deduplicated text-only math bank from five sources."""
import csv
import argparse
import unicodedata
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from antiobfbench.base import unit_hash
from antiobfbench.parse import boxed

DATASETS = {
    'gsm8k': ('openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866'),
    'math': ('EleutherAI/hendrycks_math', '21a5633873b6a120296cce3e2df9d5550074f4a3'),
    'aime': ('gneubig/aime-1983-2024', '5d610df981dec508dd93d0a16333a029ac1739d8'),
    'olympiadbench': ('Hothan/OlympiadBench', '91184b52131e7fc9455fef848035173aea8cc01a'),
    'omni_math': ('KbsdJames/Omni-MATH', '40ba231d8f16e29ecd40e6407e2c8640145a8f62'),
}
# Exclude diagram-dependent prompts instead of showing unusable image/Asymptote placeholders.
DIAGRAM = re.compile(r'\[asy\]|\.(?:png|jpg|gif)\b|\\includegraphics|\b(?:diagram|figure|graph below|shown below|illustrated)\b', re.I)


def canonical(problem):
    problem = re.sub(r'\\(?:left|right|,|!|;|quad|qquad)', '', problem)
    return re.sub(r'\s+', '', problem).casefold()


def rows(source, path):
    if source == 'omni_math':
        for i, line in enumerate((path/'test.jsonl').open()):
            row=json.loads(line)
            yield dict(id=f'omni-math-{i}', problem=row['problem'], answer=row['answer'],
                       subject=row['domain'], difficulty=row['difficulty'], competition=row['source'], _solution=row['solution'])
    elif source == 'olympiadbench':
        for row in pq.read_table(next(path.rglob('*OE_TO_maths_en_COMP.parquet'))).to_pylist():
            answers=row['final_answer']
            yield dict(id=f"olympiadbench-{row['id']}", problem=(row['context']+'\n\n' if row.get('context') else '')+row['question'],
                       answer=answers[0] if len(answers)==1 and not row['is_multiple_answer'] else None,
                       subject=row['subfield'], difficulty=row['difficulty'], source_error=row.get('error'), _solution='\n'.join(row['solution']))
    elif source == 'aime':
        with next(path.glob('*.csv')).open(encoding='utf8') as f:
            for row in csv.DictReader(f):
                yield dict(id='aime-' + row['ID'], problem=row['Question'], answer=row['Answer'],
                           year=int(row['Year']), number=int(row['Problem Number']), part=row['Part'])
    else:
        for file in sorted(path.rglob('*.parquet')):
            if source == 'gsm8k' and file.parent.name != 'main':
                continue
            split = file.name.split('-')[0]
            for i, row in enumerate(pq.read_table(file).to_pylist()):
                if source == 'gsm8k':
                    answer = row['answer'].rsplit('#### ', 1)[-1].strip()
                    problem = row['question']
                    extra = {}
                else:
                    answer = boxed(row['solution'])
                    problem = row['problem']
                    extra = dict(subject=row['type'], difficulty=row['level'])
                yield dict(id=f'{source}-{file.parent.name}-{split}-{i}', split=split,
                           problem=problem, answer=answer, **extra)



def shingles(problem):
    text = unicodedata.normalize('NFKC', problem).casefold()
    text = re.sub(r'\\(?:left|right|quad|qquad|,|!|;)', '', text)
    tokens = re.findall(r'[a-z]+|[0-9]+|[+*/=<>-]', text)
    return {tuple(tokens[i:i+5]) for i in range(len(tokens)-4)}


def clean_answer(answer):
    if not answer:
        return None
    answer = answer.strip()
    if boxed(answer) is not None:
        answer = boxed(answer)
    if answer.startswith('\\(') and answer.endswith('\\)'):
        answer = answer[2:-2]
    if answer.startswith('\\[') and answer.endswith('\\]'):
        answer = answer[2:-2]
    answer = answer.strip('$ ').strip()
    # Keep closed mathematical expressions; exclude proof/prose and yes/no answers.
    stripped = re.sub(r'\\[A-Za-z]+', '', answer)
    if len(answer) > 256 or re.search(r'[A-Za-z]{3,}', stripped):
        return None
    from antiobfbench.envs.e09_math_dinosaurs import parsed
    if not parsed(answer):
        return None
    return answer



def confirmed_answer(answer, solution):
    """Conservative import: gold must agree with the worked solution's final math expression."""
    from math_verify import verify
    from antiobfbench.envs.e09_math_dinosaurs import parsed
    last = boxed(solution)
    if last is None:
        spans = re.findall(r'\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]', solution, re.S)
        if not spans:
            return False
        last = next((part for part in spans[-1] if part), '')
    return bool(verify(parsed(answer), parsed(last), timeout_seconds=1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/09_math.jsonl')
    args=parser.parse_args()
    paths = {}
    for source,(repo,revision) in DATASETS.items():
        patterns = (['test.jsonl','README.md'] if source=='omni_math' else
                    ['*OE_TO_maths_en_COMP*/*.parquet','README.md'] if source=='olympiadbench' else
                    ['*.parquet','*.csv','README.md'])
        paths[source]=Path(snapshot_download(repo,repo_type='dataset',revision=revision,allow_patterns=patterns))
    kept, seen, counts = [], set(), Counter()
    indexed, inverted = [], defaultdict(list)
    for source in ('aime','gsm8k','math','olympiadbench','omni_math'):
        new_source=source in ('olympiadbench','omni_math')
        for row in rows(source,paths[source]):
            counts[source+'/downloaded']+=1
            row['problem']='\n'.join(row['problem'].splitlines())
            if source == 'omni_math' and row['competition'] not in ('HMMT_2','HMMT_11'):
                counts[source+'/outside_hmmt_subset']+=1;continue
            if DIAGRAM.search(row['problem']):
                counts[source+'/diagram']+=1;continue
            if new_source:
                if row.pop('source_error',None):
                    counts[source+'/source_error']+=1;continue
                if len(row['problem']) > 3000 or re.search(r'\b(?:prove that|show that|prove or disprove)\b',row['problem'],re.I):
                    counts[source+'/proof_or_overlong']+=1;continue
                if row['id'] == 'olympiadbench-1612':  # Source omits the definition of integer weight.
                    counts[source+'/missing_definition']+=1;continue
                if re.search(r'\b(?:estimate|approximation|picture|image)\b',row['problem'],re.I):
                    counts[source+'/estimate_or_visual']+=1;continue
                row['answer']=clean_answer(row['answer'])
                solution=row.pop('_solution')
                if row['answer'] and not confirmed_answer(row['answer'],solution):
                    counts[source+'/unconfirmed_or_inconsistent_answer']+=1;continue
                row['answer_check']='agrees_with_solution_final_math'
            if not row['answer']:
                counts[source+'/missing_or_ungradeable_answer']+=1;continue
            key=hashlib.sha256(canonical(row['problem']).encode()).hexdigest()
            if key in seen:
                counts[source+'/duplicate']+=1;continue
            grams=shingles(row['problem'])
            if new_source and grams:
                matching=sorted((g for g in grams if g in inverted),key=lambda g:len(inverted[g]))[:8]
                candidates={i for g in matching for i in inverted[g]}
                duplicate=next((i for i in candidates if len(grams & indexed[i])/len(grams | indexed[i]) >= .85),None)
                if duplicate is not None:
                    counts[source+'/near_duplicate']+=1;continue
            seen.add(key)
            split=row.pop('split',None) or ('train' if unit_hash('09_math/'+source,key)<.8 else 'test')
            kept.append(dict(source=source,split=split,problem_hash=key,**row))
            index=len(indexed);indexed.append(grams)
            for gram in grams:inverted[gram].append(index)
            counts[source+'/'+split]+=1
        print(source,dict(counts),flush=True)
    kept.sort(key=lambda row:(row['source'],row['id']))
    data=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in kept).encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(data)
    manifest=dict(datasets={s:dict(repo=r,revision=v) for s,(r,v) in DATASETS.items()},
                  counts=dict(sorted(counts.items())),sha256=hashlib.sha256(data).hexdigest(),
                  note='Original GSM8K/MATH splits and all legacy records retained. AIME and new sources use stable 80/20 problem-hash splits. New sources: English text-only single-answer math expressions confirmed against the worked solution final math expression; Omni-MATH restricted to HMMT_2/HMMT_11 contest provenance; no proofs, estimation tasks, diagrams, source-flagged errors or questions over 3000 characters. Known missing-definition question olympiadbench-1612 omitted. Exact deduplication plus >=0.85 token-5-shingle Jaccard near-duplicate filtering against earlier sources; no semantic deduplication guarantee.')
    destination=ROOT/'data/09_math_source.json' if args.output==ROOT/'data/09_math.jsonl' else args.output.with_suffix('.source.json')
    destination.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
