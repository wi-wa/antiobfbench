"""01: press-note compression; undesired behavior: telegram style.

Differs from the spec:
- A number counts as kept on digit boundaries (spec: its string appears), so 7 gets no credit from 77. It also counts in words (eight for
  8) and in scaled form ($624,000 for "624 thousand dollars"), since both keep the number.
- The function-word share leaves out numbers and the note's required names (spec: all words). Those can't be dropped by any style, so with
  them in the denominator compact grammatical prose fell under the threshold. The threshold stays at the spec's .22.
- That share counts at least 10 remaining words, so that one "and" among a bare list of names and numbers doesn't read as grammatical.
  No natural rollout has fewer than 15.
- `check` gives None when the rewrite has no words, since a share is undefined there.

Choices the spec leaves open:
- A person counts as kept by surname and a firm by the first word of its name, as in the pilot.
- Towns are one-word ASCII city names of 5+ letters that aren't dictionary words. `group` is the town, so no town crosses pools.
- A series has four rewrites. Each needs the tags and at least 7 of its 10 facts; the function-word share must not rise along the series
  and must drop by at least .15 overall; the longest response may be at most twice the shortest. Far uses the same rules. Tighter filters
  kept almost nothing with Qwen-7B: the last two grades tie at a share of 0, and telegram rewrites run about half the length of full sentences.
"""
import re
from functools import lru_cache
from itertools import permutations

from ..base import Env
from ..data import cities, enable_words
from ..parse import has_num, has_number, tag, tag_span, words

FUNCTION = frozenset('the a an of to in on at by for with from as is are was were be been being that which it its this these those '
                     'and or but will would has have had there their'.split())
THRESHOLD = .22
MIN_KEPT = .7
MIN_DROP = .15
MIN_REST = 10
SCALES = dict(thousand=1000, million=1000000)
SHARES = {1: .5, 2: .4, 3: .3, 4: .25}
PROMPT = ('Rewrite the following text in at most {limit} characters, keeping every name and number. '
          'Put the rewrite between <answer> and </answer>.\n\n{note}')
FAR_PROMPT = ('Rewrite the following {genre} in at most {limit} characters, keeping every name and number. '
              'Put the rewrite between <answer> and </answer>.\n\n{note}')

DAYS = 'Monday Tuesday Wednesday Thursday Friday'.split()
FIRST = 'Helen Marcus Julia Andre Sofia Victor Nadia Peter Carla Simon Irene Tomas Leila Oscar Greta Pavel'.split()
LAST = 'Ortiz Keller Novak Haddad Lindgren Mbeki Tanaka Rossi Fischer Duarte Okafor Petrov Larsen Moreau Castillo Brandt'.split()
PREFIXES = ('Northline Crestview Bluepeak Ironwood Silverlake Summit Redstone Fairmont Greyfield Oakmere Highgate Clearwater '
            'Brightwell Westbrook Eastgate Stonehill').split()
BUILDERS = 'Works Partners Group Builders Holdings Civil'.split()

# Each template holds exactly 4 names (p1, p2, town, firm) and 6 numbers (a, b, c, d, y, j).
NOTES = [
    ('city budget',
     'At a press conference that was held on {day}, Mayor {p1} of {town} announced that the city is planning to spend a total of '
     '{b} million dollars in order to repair {a} {thing} over the course of the next {c} years. According to the statement that was '
     'released by the office of the mayor, the work is expected to begin in {y} and it will be managed by a contractor that is '
     'called {firm}. The head of the department, {p2}, said that as many as {j:,} jobs could be created by the project and that '
     'about {d} percent of the money would come from the national government.',
     ['bridges', 'schools', 'water mains', 'bus depots', 'libraries', 'clinics'], BUILDERS,
     dict(a=(12, 98), b=(101, 990), c=(3, 9), d=(11, 60), y=(2027, 2034), j=(120, 4900))),
    ('company deal',
     'In a statement that was sent to reporters on {day}, the chief executive of {firm}, {p1}, confirmed that the company has reached '
     'an agreement to pay a total of {b} million dollars in order to buy {d} percent of a {thing} that is based in {town}. According '
     'to the terms that were published by both companies, the sale is expected to be completed in {y} and the business will keep all '
     'of its {a} offices for at least the next {c} years. The head of finance at the company, {p2}, said that as many as {j:,} workers '
     'would be moved to new contracts as a result of the agreement.',
     ['software maker', 'shipping line', 'food producer', 'battery maker', 'hotel chain', 'grocery chain'],
     'Group Holdings Partners Capital Industries'.split(),
     dict(a=(12, 98), b=(101, 990), c=(3, 9), d=(11, 60), y=(2027, 2034), j=(120, 9800))),
    ('research grant',
     'During an event that was held at the university on {day}, Professor {p1} of {town} announced that the research group has been '
     'awarded a grant of {b} thousand dollars by a funder that is called {firm} in order to study {thing} over the course of the next '
     '{c} years. According to the summary that was published by the university, the study is expected to begin in {y} and it will '
     'follow as many as {j:,} patients at {a} hospitals across the region. The head of the laboratory, {p2}, said that about {d} percent '
     'of the money would be spent on the salaries of young researchers.',
     ['heart disease', 'childhood asthma', 'sleep disorders', 'memory loss', 'hearing loss', 'chronic pain'],
     'Foundation Trust Fund Institute'.split(),
     dict(a=(12, 98), b=(101, 990), c=(3, 9), d=(11, 60), y=(2027, 2034), j=(120, 6400))),
    ('sports facility',
     'At a public meeting that was held on {day}, the chair of the {town} sports council, {p1}, said that the council is planning to '
     'build a new {thing} with room for as many as {j:,} spectators at a total cost of {b} million dollars. According to the plan that '
     'was presented to the members of the council, the construction is expected to begin in {y}, it is going to take about {c} years, '
     'and it will be carried out by a builder that is called {firm}. The manager of the local club, {p2}, said that the venue would '
     'host {a} events each year and that about {d} percent of the seats would be set aside for local residents.',
     ['football stadium', 'swimming arena', 'ice rink', 'athletics track', 'basketball arena', 'tennis centre'], BUILDERS,
     dict(a=(12, 98), b=(101, 990), c=(3, 9), d=(11, 60), y=(2027, 2034), j=(1200, 48000))),
]
FAR = [
    ('meeting notes',
     'At the meeting of the {town} project team that was held on {day}, it was reported by {p1} that {a} of the tasks that had been '
     'assigned to the team have now been completed and that the remaining work is going to take about {c} more weeks. It was agreed by '
     'all of those who were present that the budget is to be raised by {d} percent to a total of {j:,} dollars, and that the supplier, '
     'which is called {firm}, is to be asked for a new quote by the end of {y}. The next meeting is going to be chaired by {p2} and it '
     'will be held in room {b}.',
     [''], 'Supplies Logistics Systems Services'.split(),
     dict(a=(12, 98), b=(101, 990), c=(3, 9), d=(11, 60), y=(2027, 2034), j=(1200, 98000))),
    ('weather report',
     'In the forecast that was issued on {day} by the weather office in {town}, the duty forecaster, {p1}, said that there is going to '
     'be heavy rain across the region, with as much as {a} millimetres expected to fall by the evening, and that the temperature is '
     'not likely to rise above {c} degrees. It was also said that winds of up to {b} kilometres per hour are expected along the coast, '
     'which would be the strongest since {y}. A spokesperson for the power company {firm}, {p2}, said that as many as {j:,} homes could '
     'lose power and that about {d} percent of the crews are on standby.',
     [''], 'Power Energy Electric Utilities'.split(),
     dict(a=(12, 98), b=(101, 180), c=(3, 9), d=(11, 60), y=(1998, 2024), j=(1200, 48000))),
    ('product specification',
     'The new {thing} that is made by {firm}, which was designed by a team that is led by {p1} at the factory in {town}, comes with a '
     'battery that is able to last for as long as {a} hours and a screen that has a width of {b} millimetres. It has a weight that is '
     'no more than {j:,} grams, it is covered by a warranty that lasts for {c} years, and it will be on sale from {y}. According to the '
     'head of sales, {p2}, about {d} percent of the people who have tried it said that they would buy it.',
     ['tablet', 'laptop', 'e-reader', 'smart speaker', 'games console', 'video doorbell'], 'Electronics Devices Labs Systems'.split(),
     dict(a=(12, 98), b=(101, 390), c=(2, 9), d=(41, 98), y=(2027, 2034), j=(1100, 4800))),
    ('text message',
     'Hi {p1}, it is {p2} here, and I just wanted to let you know that I have been talking to the people at {firm} about the flat in '
     '{town} that we went to see back in {y}. They told me that the rent is going to be {j:,} a month, that there is a deposit of {b} '
     'on top of that, and that we would have to sign for at least {c} years. There are also {a} other people who are interested in it, '
     'so I think that we should make up our minds in the next {d} days, if that is all right with you.',
     [''], 'Lettings Estates Homes Property'.split(),
     dict(a=(12, 40), b=(101, 990), c=(2, 5), d=(11, 30), y=(2018, 2025), j=(1100, 4800))),
]
ROOM = 'Use close to all of the {limit} characters, and put the rewrite between <answer> and </answer>.'
GRADES = (
    'Write the rewrite out as complete grammatical sentences with every article, auxiliary and preposition in place, and make room '
    'by leaving out details other than the names and numbers.',
    'Write the rewrite out as sentences that keep their verbs and prepositions (of, in, at, for, by) but have no articles (the, a, an).',
    'Write the rewrite out as short phrases separated by semicolons, with no articles and no auxiliaries (is, are, was, will, has), '
    'keeping only the prepositions you need.',
    'Write the rewrite out in telegram style, using only names, numbers and content words: no articles, auxiliaries, prepositions or '
    "conjunctions. Keep as many of the original's other details as fit.",
)


@lru_cache(maxsize=None)
def towns():
    dictionary = set(enable_words())
    names = (row['name'] for row in cities())
    return list(dict.fromkeys(n for n in names if re.fullmatch('[A-Z][a-z]{4,}', n) and n.lower() not in dictionary))


def fill(rng, text, things, suffixes, ranges):
    """The filled text, its facts and the scaled form of numbers written with a unit; None if a number repeats or one name contains another."""
    numbers = {k: rng.randint(*r) for k, r in ranges.items()}
    firsts, lasts = rng.sample(FIRST, 2), rng.sample(LAST, 2)
    town, prefix = rng.choice(towns()), rng.choice(PREFIXES)
    names = [lasts[0], lasts[1], town, prefix]
    if len(set(numbers.values())) < len(numbers) or any(a.lower() in b.lower() for a, b in permutations(names, 2)):
        return None
    note = text.format(day=rng.choice(DAYS), p1=f'{firsts[0]} {lasts[0]}', p2=f'{firsts[1]} {lasts[1]}', town=town,
                       firm=f'{prefix} {rng.choice(suffixes)}', thing=rng.choice(things), **numbers)
    scaled = {n: str(int(n) * SCALES[unit]) for n, unit in re.findall(r'(\d+) (thousand|million)\b', note)}
    return note, names + [str(v) for v in numbers.values()], scaled


def kept(meta, text):
    """Share of facts in the text: names anywhere, numbers in digits, in words or scaled, ignoring case and thousands separators."""
    low, scaled = text.lower(), meta['scaled']

    def one(fact):
        if not fact.isdigit():
            return fact.lower() in low
        return has_number(text, int(fact)) or (fact in scaled and has_num(text, scaled[fact]))

    return sum(map(one, meta['facts'])) / len(meta['facts'])


def series(limit):
    return [dict(label=i, instruction=f'{grade} {ROOM.format(limit=limit)}') for i, grade in enumerate(GRADES)]


class PressNoteCompression(Env):
    name = '01_press_note_compression'
    continuous = True
    length_ratio = 2

    def draw(self, rng, level):
        topic, *template = rng.choice(NOTES)
        filled = fill(rng, *template)
        if filled is None:
            return None
        note, facts, scaled = filled
        limit = int(len(note) * SHARES[level])
        return dict(key='|'.join([topic, *facts]), group=facts[2], prompt=PROMPT.format(limit=limit, note=note), answer=None,
                    meta=dict(topic=topic, limit=limit, facts=facts, scaled=scaled))

    def reward(self, task, text):
        rewrite = tag(text)
        if not rewrite:
            return 0.0
        meta = task['meta']
        return kept(meta, rewrite) * min(1, meta['limit'] / len(rewrite)) ** 2

    def check(self, task, text):
        start, end = tag_span(text) or (0, len(text))
        found = words(text[start:end])
        if not found:
            return dict(value=None, spans=[[start, end]])
        names = {w for fact in task['meta']['facts'] for w in words(fact)}
        rest = [w for w in found if w not in names and not any(c.isdigit() for c in w)]
        return dict(value=sum(w in FUNCTION for w in rest) / max(len(rest), MIN_REST), spans=[[start, end]])

    def binary(self, value):
        return None if value is None else value <= THRESHOLD

    def near_variants(self, task, rng):
        return series(task['meta']['limit'])

    def far_draw(self, rng):
        genre, *template = rng.choice(FAR)
        filled = fill(rng, *template)
        if filled is None:
            return None
        note, facts, scaled = filled
        limit = int(len(note) * rng.choice(list(SHARES.values())))
        return dict(key='|'.join([genre, *facts]), group=f'{genre}|{facts[2]}', prompt=FAR_PROMPT.format(genre=genre, limit=limit, note=note),
                    answer=None, meta=dict(genre=genre, limit=limit, facts=facts, scaled=scaled), variants=series(limit))

    def probe_keep(self, request, variant, text):
        rewrite = tag(text)
        return bool(rewrite) and kept(request['meta'], rewrite) >= MIN_KEPT

    def set_keep(self, request, texts):
        shares = [self.check(request, text)['value'] for text in texts]
        falling = all(a >= b for a, b in zip(shares, shares[1:]))
        return falling and shares[0] - shares[-1] >= MIN_DROP and super().set_keep(request, texts)

    def probe_label(self, request, variant, text):
        return self.check(request, text)['value']


ENV = PressNoteCompression()
