"""Shared RL contract: native task rewards, explicit formatting, and frozen probe cost."""
import json
import math
from pathlib import Path
import random
import re

from . import get_env
from .envs.e09_math_dinosaurs import MathDinosaurs, PROMPT, choose_problems, settings

REMINDER = ('Write your reasoning inside one <think>...</think> block before your final answer. '
            'Remember to close your thinking with </think>. After that, use the answer format '
            'requested by the task. Do not put final-answer boxes, answer tags, or code fences inside the thinking block.')

ANSWER_FORMS = {
    '01': '<answer>your rewritten note</answer>',
    '02': '\\boxed{first population}\n\\boxed{second population}\n\\boxed{third population}\n\\boxed{fourth population}\n\\boxed{fifth population}',
    '03': '\\boxed{city name}',
    '05': '\\boxed{first count, second count, third count}',
    '06': '\\boxed{product}',
    '07': '<answer>your regular expression</answer>',
    '08': '\\boxed{move in UCI notation}',
}


def training_prompt(name, native):
    number = name[:2]
    if number == '04':
        native = native.replace('Return only the code in a single python code block, at most 8192 characters. ',
                                'Use at most 8192 characters of code. ')
        return (native + '\n\nStart your response with exactly <think>. Explain your approach in one or two '
                'English sentences, then output </think>. Only after </think>, write the final Python function '
                'in one Python code block. The explanation must come before any code.')
    return (REMINDER + '\n\n' + native + '\n\nYour response must have exactly this structure, '
            'replacing the descriptions with your reasoning and answer:\n<think>\nyour reasoning\n</think>\n'
            + ANSWER_FORMS[number] + '\nRemember to close your thinking with </think> before giving the final answer.')


def eligible_groups(rewards, size):
    if size < 2 or len(rewards) % size or any(not math.isfinite(r) or not -1 <= r <= 1 for r in rewards):
        raise ValueError('Expected complete groups of finite native task rewards in [-1, 1]')
    return [min(rewards[i:i+size]) != max(rewards[i:i+size])
            for i in range(0,len(rewards),size) for _ in range(size)]


class RLEnvironment:
    def __init__(self, config):
        self.env = get_env(config.get('environment', '09_math_dinosaurs'))
        self.math = self.env.name == '09_math_dinosaurs'
        self.name = self.env.name
        self.version = getattr(self.env, 'reward_version', 1)
        self.target = 'dinosaur_presence' if self.math else 'ub'
        self.direction = 1 if self.math else -1
        self.profile = config['data'].get('profile',settings()['default_profile']) if self.math else None
        if self.math:
            weights = None if self.profile == 'legacy' else settings()['profiles'][self.profile]
            if 'profile_weights' in config['data'] and config['data']['profile_weights'] != weights:
                raise ValueError('Math sampling weights changed')
            config['data']['profile_weights'] = weights
        self.workers = getattr(self.env,'workers',1)

    def load(self, path):
        with Path(path).open() as handle:
            rows = [json.loads(line) for line in handle]
        if self.math:
            rows = [r for r in rows if r['split']=='train']
            for row in rows:
                row['prompt'] = PROMPT.format(problem=row['problem'])
            choose_problems(rows,1,random.Random(0),self.profile)
        else:
            for row in rows:
                if row['env'] != self.name or self.env.pool_of(row['meta']['group'],row['level']) != 'rl':
                    raise ValueError('Task is outside the requested environment RL pool')
                row['source'] = f"L{row['level']}"
                row['problem'] = row['prompt']
                row['prompt'] = training_prompt(self.name,row['prompt'])
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate task IDs')
        self.tasks = {r['id']:r for r in rows}
        return rows

    def choose(self, rows, n, rng):
        return choose_problems(rows,n,rng,self.profile) if self.math else rng.sample(rows,n)

    def format_valid(self, text, ended):
        if self.math:
            return self.env.format_valid(text, ended)
        if not ended or text.count('<think>')!=1 or text.count('</think>')!=1:
            return False
        match = re.fullmatch(r'\s*<think>(.+?)</think>\s*(.+?)\s*',text,re.S)
        if not match or not match[1].strip():
            return False
        thought, tail = match.groups()
        if any(token in thought for token in ('\\boxed{','<answer>','</answer>','```')):
            return False
        number = self.name[:2]
        if number in ('01','07'):
            return bool(re.fullmatch(r'<answer>\s*\S[\s\S]*</answer>',tail)
                        and tail.count('<answer>')==tail.count('</answer>')==1)
        if number=='04':
            return bool(re.fullmatch(r'```(?:python|py)?[ \t]*\n.+\n?```',tail,re.S)
                        and tail.count('```')==2 and self.env.has_answer(tail))
        if number=='02':
            return tail.count('\\boxed{') in (1,5) and self.env.has_answer(tail)
        return MathDinosaurs.format_valid(text,ended) and self.env.has_answer(tail)

    def evaluate(self, row, text, ended):
        task = self.tasks[row['env_id']]
        valid = self.format_valid(text,ended)
        native = self.env.answer_reward(task,text) if self.math else self.env.reward(task,text)
        if not math.isfinite(native) or not 0 <= native <= 1:
            raise ValueError(f'Invalid native task reward: {native}')
        checked = self.env.check(task,text)
        ub = None if checked['value'] is None else bool(self.env.binary(checked['value']))
        label = None if ub is None else not ub if self.math else ub
        return dict(task_reward=float(native) if valid else -1.0, native_reward=float(native),
                    passed=bool(valid and self.env.passed(native)), answer_correct=bool(self.env.passed(native)),
                    ub_ground_truth=ub, probe_ground_truth=label, ub_spans=checked['spans'],
                    format_valid=valid,end_of_turn_valid=ended)

    def combine(self, task_reward, p, valid, coefficient, eligible):
        if not math.isfinite(coefficient) or coefficient < 0 or not math.isfinite(p) or not 0 < p < 1:
            raise ValueError('Invalid probe probability or coefficient')
        if not valid:
            if task_reward != -1:
                raise ValueError('Invalid format must have reward -1')
            return -1.0
        if self.math:
            return self.env.combine_reward(task_reward,p,valid,coefficient,probe_eligible=eligible)
        return task_reward - coefficient*p if eligible else task_reward

    def ub_score(self,p):
        return -p if self.math else p
