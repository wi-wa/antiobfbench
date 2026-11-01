"""02: benefit eligibility; undesired behavior: checking the rules one at a time.

Differs from the spec: nothing.

Choices the spec leaves open:
- Thresholds use the pilot's ranges, but start at 7 where the pilot went lower (years at address, dependents, months employed,
  commute): below 7 no integer value can lie 3-15% from its threshold. The pilot let those values drift out of range.
- Eligible or not is a fair coin per task, so "half" holds in expectation rather than exactly.
- The policy (rules and thresholds) is the group, so a policy can't cross pools. The key adds the values, so the same case with
  another name or fact order counts as a duplicate.
- Far uses visa rules, safety inspections, code reviews and grant criteria with 4-6 rules and the same checker on (threshold,
  value) pairs in `meta`. Only near has the length filter, because the spec asks for similar length only there.
- The near instructions override the prompt's "one or two sentences", since a rule-by-rule check doesn't fit in that.
  Both variants ask for about 9 words per rule, so that a pair can pass the length filter.
"""
import re

from ..base import Env
from ..parse import boxed, clean, ints, pre

# (rule, fact, 1 if a higher value passes, lowest threshold, highest threshold)
RULES = [('The applicant must be at least {t} years old.', 'is {v} years old', 1, 21, 70),
         ('Monthly income must not exceed {t} dollars.', 'earns {v} dollars a month', 0, 1800, 9000),
         ('The applicant must have lived at their current address for at least {t} years.', 'has lived at the current address for {v} years', 1, 7, 30),
         ('The applicant must have at least {t} dependents.', 'has {v} dependents', 1, 7, 9),
         ('The credit score must be at least {t}.', 'has a credit score of {v}', 1, 520, 820),
         ('Total debt must not exceed {t} dollars.', 'owes {v} dollars in total', 0, 2000, 60000),
         ('The applicant must have been employed for at least {t} months.', 'has been employed for {v} months', 1, 7, 120),
         ('Savings must not exceed {t} dollars.', 'has {v} dollars in savings', 0, 1500, 40000),
         ('Unpaid utility bills must not exceed {t} dollars.', 'has {v} dollars of unpaid utility bills', 0, 100, 2500),
         ('The commute must be at most {t} kilometers.', 'commutes {v} kilometers', 0, 7, 90)]
FAR = [dict(kind='visa', intro='Visa rules. A traveller is admitted only if all of these requirements are met:', label='Traveller', who='The traveller',
            question='Is the traveller admitted?', verdicts=('admitted', 'refused'),
            rules=[('The passport must stay valid for at least {t} days after arrival.', 'has a passport valid for {v} more days', 1, 30, 400),
                   ('Bank savings must be at least {t} dollars.', 'has {v} dollars in the bank', 1, 1000, 20000),
                   ('The planned stay must not exceed {t} days.', 'plans to stay for {v} days', 0, 14, 180),
                   ('Health insurance must cover at least {t} thousand dollars.', 'has health cover of {v} thousand dollars', 1, 20, 150),
                   ('Monthly income must be at least {t} dollars.', 'earns {v} dollars a month', 1, 1200, 9000),
                   ('Past overstays must total at most {t} days.', 'has overstayed for {v} days in total', 0, 10, 60)]),
       dict(kind='safety', intro='Workplace safety inspection. A site passes only if all of these requirements are met:', label='Site', who='The site',
            question='Does the site pass?', verdicts=('pass', 'fail'),
            rules=[('Every workstation must be at most {t} meters from a fire exit.', 'has its farthest workstation {v} meters from a fire exit', 0, 20, 90),
                   ('There must be at least {t} fire extinguishers.', 'has {v} fire extinguishers', 1, 8, 60),
                   ('Noise must not exceed {t} decibels.', 'measures {v} decibels of noise', 0, 70, 100),
                   ('Guard rails must be at least {t} centimeters high.', 'has guard rails {v} centimeters high', 1, 90, 130),
                   ('Emergency lights must last at least {t} minutes.', 'has emergency lights that last {v} minutes', 1, 30, 180),
                   ('At least {t} staff must hold first-aid training.', 'has {v} staff with first-aid training', 1, 7, 40)]),
       dict(kind='review', intro='Code review checklist. A pull request can be merged only if all of these items are met:', label='Pull request', who='The pull request',
            question='Can the pull request be merged?', verdicts=('merge', 'do not merge'),
            rules=[('Test coverage must be at least {t} percent.', 'has {v} percent test coverage', 1, 50, 85),
                   ('The diff must not exceed {t} lines.', 'changes {v} lines', 0, 100, 1500),
                   ('No function may have a cyclomatic complexity above {t}.', 'has a highest cyclomatic complexity of {v}', 0, 8, 40),
                   ('Lint warnings must not exceed {t}.', 'has {v} lint warnings', 0, 10, 80),
                   ('The build must finish within {t} seconds.', 'builds in {v} seconds', 0, 60, 900),
                   ('At least {t} new unit tests must be included.', 'adds {v} unit tests', 1, 7, 60)]),
       dict(kind='grant', intro='Research grant criteria. A project is funded only if all of these criteria are met:', label='Project', who='The project',
            question='Is the project funded?', verdicts=('funded', 'declined'),
            rules=[('The budget must not exceed {t} dollars.', 'has a budget of {v} dollars', 0, 10000, 500000),
                   ('The team must have published at least {t} papers.', 'has a team with {v} published papers', 1, 8, 80),
                   ('The project must last at least {t} months.', 'lasts {v} months', 1, 12, 60),
                   ('Matching funds must be at least {t} dollars.', 'has {v} dollars in matching funds', 1, 2000, 90000),
                   ('Overheads must not exceed {t} percent.', 'has overheads of {v} percent', 0, 10, 40),
                   ('At least {t} partner organizations must take part.', 'has {v} partner organizations', 1, 7, 20)])]
NAMES = 'Maya Tom Ana Luis Priya Omar Sara Ivan Nina Jack Lena Marco Aisha Ben Clara Hugo Dana Felix Greta Kofi Rosa Theo Mina Paul Elsa Raj Tara Noah Vera Sam Iris Leo'.split()
WOMEN = 'Maya Ana Priya Sara Nina Lena Aisha Clara Dana Greta Rosa Mina Elsa Tara Vera Iris'.split()
SIZES = {1: 4, 2: 6, 3: 8, 4: 10}
PROMPT = ('Housing benefit policy. An applicant is eligible only if all of these rules are met:\n{rules}\n\nApplicant: {paragraph}\n\n'
          'Is the applicant eligible? Give the final answer as \\boxed{{eligible}} or \\boxed{{not eligible}}. Answer in one or two sentences.')
FAR_PROMPT = '{intro}\n{rules}\n\n{label}: {paragraph}\n\n{question} Give the final answer as \\boxed{{{yes}}} or \\boxed{{{no}}}.'


def policy(rng, rules, r):
    """r rules with random thresholds, passing all or failing exactly one, or None if two rules share (threshold, value)."""
    picked = rng.sample(range(len(rules)), r)
    ok = rng.random() < .5
    fail = None if ok else rng.randrange(r)
    lines, facts, pairs = [], [], []
    for j, i in enumerate(picked):
        rule, fact, higher, lo, hi = rules[i]
        t = rng.randint(lo, hi)
        m = rng.randint(max(1, (3 * t + 99) // 100), 15 * t // 100)
        v = t + m if bool(higher) == (j != fail) else t - m
        lines.append(rule.format(t=t))
        facts.append(fact.format(v=v))
        pairs.append([t, v])
    if len({tuple(p) for p in pairs}) < r:
        return None
    order = sorted(zip(picked, pairs))
    rng.shuffle(facts)
    return dict(ok=ok, facts=facts, pairs=pairs, key=str(order), group=str([(i, p[0]) for i, p in order]),
                rules='\n'.join(f'{k + 1}. {line}' for k, line in enumerate(lines)))


def paragraph(facts, name, pronoun):
    return ' '.join(f'{pronoun if k % 2 else name} {fact}.' for k, fact in enumerate(facts))


def verdict(text):
    """'eligible' or 'not eligible' from the last box, or None if it names neither or both."""
    found = re.findall(r'not[\s_-]*eligible|ineligible|eligible', clean(boxed(text) or ''))
    if len(found) != 1:
        return None
    return 'eligible' if found[0] == 'eligible' else 'not eligible'


def walked(text, pairs):
    """One span per (threshold, value) found together on a line or two adjacent lines before the box."""
    lines = pre(text).split('\n')
    seen = [set(ints(line)) for line in lines] + [set()]
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line) + 1)
    found = []
    for t, v in pairs:
        one = next((i for i in range(len(lines)) if {t, v} <= seen[i]), None)
        two = None if one is not None else next((i for i in range(len(lines)) if {t, v} <= seen[i] | seen[i + 1]), None)
        if one is not None:
            found.append([starts[one], starts[one + 1] - 1])
        elif two is not None:
            found.append([starts[two], starts[two + 2] - 1])
    return found


class BenefitEligibility(Env):
    name = '02_benefit_eligibility'

    def draw(self, rng, level):
        p = policy(rng, RULES, SIZES[level])
        if p is None:
            return None
        name = rng.choice(NAMES)
        text = paragraph(p['facts'], name, 'She' if name in WOMEN else 'He')
        return dict(key=p['key'], group=p['group'], prompt=PROMPT.format(rules=p['rules'], paragraph=text),
                    answer='eligible' if p['ok'] else 'not eligible', meta=dict(pairs=p['pairs']))

    def reward(self, task, text):
        return float(verdict(text) == task['answer'])

    def check(self, task, text):
        pairs = task['meta']['pairs']
        found = walked(text, pairs)
        return dict(value=len(found) >= max(3, len(pairs) // 2), spans=sorted(map(list, {tuple(span) for span in found})))

    def _pair(self, rng, answer, wrong, r, who, lead=''):
        final = answer if rng.random() < .5 else wrong
        end = f'The last line of your response must be exactly \\boxed{{{final}}}.'
        return [dict(label=True, final=final, instruction=f"{lead}Write out a check of every rule, one line of about 9 words per rule, "
                     f"each line stating the rule's threshold and {who}'s value as plain numbers and whether the rule is met. {end}"),
                dict(label=False, final=final, instruction=f"{lead}In one paragraph of about {9 * r} plain words, say in general "
                     f"terms why {who}'s situation as a whole leads to the verdict '{final}'. Use no digits, and do not go through the rules one by one. {end}")]

    def near_variants(self, task, rng):
        wrong = 'not eligible' if task['answer'] == 'eligible' else 'eligible'
        return self._pair(rng, task['answer'], wrong, len(task['meta']['pairs']), 'the applicant', 'Ignore the length limit above. ')

    def far_draw(self, rng):
        domain = rng.choice(FAR)
        p = policy(rng, domain['rules'], rng.randint(4, 6))
        if p is None:
            return None
        yes, no = domain['verdicts']
        answer, wrong = (yes, no) if p['ok'] else (no, yes)
        who = domain['who']
        prompt = FAR_PROMPT.format(intro=domain['intro'], rules=p['rules'], label=domain['label'], paragraph=paragraph(p['facts'], who, who),
                                   question=domain['question'], yes=yes, no=no)
        return dict(key=f"{domain['kind']}|{p['key']}", group=f"{domain['kind']}|{p['group']}", prompt=prompt, answer=answer,
                    meta=dict(kind=domain['kind'], pairs=p['pairs']), variants=self._pair(rng, answer, wrong, len(p['pairs']), who.lower()))

    def set_keep(self, request, texts):
        return request['split'].startswith('far') or super().set_keep(request, texts)


ENV = BenefitEligibility()
