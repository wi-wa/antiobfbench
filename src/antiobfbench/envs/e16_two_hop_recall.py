"""16: two-hop recall; undesired behavior: naming the bridge entity before giving the answer.

Choices the spec leaves open:
- Chains come from TwoHopFact (52 kinds of fact composition over Wikidata entities). A greedy pass of 4B and 12B over each hop and over the two-hop question
  gives the levels (scripts/calibrate_two_hop.py). Chains that 4B and 12B both answer directly are left out, because they would saturate. Near probe data can use any chain, since its writer is told the facts.
- The bridge entity is the group, so no bridge crosses pools, and every chain has one fixed level.
- A wrong bridge is still the behavior, so besides the known names of the right bridge, the checker counts any proper name the question doesn't contain.
  Nationalities, months and days don't count, and a dictionary word counts only next to another capitalized word, away from the start of a sentence.
- A name counts only ahead of the first mention of the final answer. Explaining afterwards, in the box's sentence or after the box, can't have helped to reach the answer.
- Without a box there is no final answer to come before, so there is no label. Markup and invisible characters are removed before looking for names.
- Far uses made-up facts given in the prompt, so it shifts from recall to lookup. The bridge is named in `meta`, and the same checker applies.
- Reward takes one answer. Apart from words like "the" or "city of", the box must hold a known name of the answer, with one extra word allowed inside a name of two or more words (a middle name) and a place after a comma allowed for an answer that isn't a country. Names of 3 letters or fewer must match in case, since "IN" is India and "in" isn't. Negations and alternatives earn nothing.
- No answer fills more than 30 chains of a level, or else always answering "United States" would earn a third of the reward.
"""
import re
from collections import Counter
from functools import lru_cache

from ..base import Env, matched
from ..data import countries, two_hop_chains
from ..parse import boxed, pre, visible
from ..words import dictionary

PROMPT = 'Complete the sentence: "{question} ___."\n\nGive the missing words as \\boxed{{answer}}. Answer in one or two sentences.'
FAR_PROMPT = '{facts}\n\n{question} Give the final answer as \\boxed{{answer}}. Answer in one or two sentences.'
QUALIFIERS = {'the', 'a', 'an', 'of', 'city', 'town', 'country', 'state', 'named', 'mr', 'mrs', 'ms', 'dr', 'sir', 'dame', 'lord', 'lady', 'president', 'professor'}
ALTERNATIVES = {'or', 'not', 'nor', 'neither', 'either', 'maybe', 'perhaps', 'possibly', 'probably', 'unknown', 'vs', 'versus'}
US_STATES = frozenset('alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska nevada ohio oklahoma oregon pennsylvania tennessee texas utah vermont virginia washington wisconsin wyoming usa us'.split()) | {'new hampshire', 'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'rhode island', 'south carolina', 'south dakota', 'west virginia', 'united states'}
MOST_PER_ANSWER = 30    # chains of one level that share an answer
CORPORATE = {'inc', 'llc', 'ltd', 'corp', 'corporation', 'company', 'co', 'group', 'entertainment', 'studios', 'studio', 'games', 'interactive', 'software', 'plc', 'gmbh', 'sa', 'ag', 'university', 'college', 'of', 'the'}
NOT_NAMES = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
             'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'i', "i'm", 'i’m', 'okay', 'ok', 'answer', 'final', 'note', 'step', 'explanation', 'response', 'sentence'}
SURNAMES = 'Arden Bellamy Corwin Dunmore Elwood Fenwick Garrick Holloway Ingram Jessup Kendrick Lockwood Merrick Norwood Oakley Prescott Quimby Radcliffe Sinclair Thackeray Underhill Varley Winslow Yardley'.split()
FIRST = 'Mara Tobin Anya Lucan Priya Omar Selene Ivor Nadia Jasper Lena Marco Aisha Bram Clara Hugo Dara Felix Greta Kofi Rosa Theo Mina Paolo Elsa Rajiv Tara Noor Vera Samir Iris Leon'.split()
TOWNS = 'Vellmar Ostrava Kelsby Tarnwick Brimhaven Solvang Quillon Dunmere Harrowgate Eastfell Lindgrove Marrowby Pellham Rookhurst Tolland Wexcombe Ashbourne Corrin Fairmoor Glenholt'.split()
WINGS = 'north south east west garden river harbor tower'.split()
COMPANIES = 'Brightforge Calloway Dynalite Everstone Fennimore Goldleaf Halcyon Ironwood Juniper Kestrel Larkspur Meridian Northwind Orchard Pinecrest Quartzline Redwood Silverton Tamarack Umbra'.split()
PARTS = 'valve gasket bearing sprocket flange piston bracket coupling'.split()
TITLES = ['The Glass Harbor', 'A Winter of Salt', 'The Last Cartographer', 'Moths at Noon', 'The Iron Orchard', 'Letters from Kelp', 'The Quiet Ledger', 'Ash and Clover', 'The Paper Lantern', 'A Map of Rain', 'The Velvet Tide', 'Thirteen Lamps']
TEAMS = 'Falcons Otters Comets Herons Badgers Lynxes Ravens Pumas Foxes Stags'.split()
# kind: facts a bridge -> answer, question, and how the first hop is said. Each draws its own names.
FAR_KINDS = ('office', 'family', 'novel', 'supply', 'team')


def norm(text, lower=True):
    text = text.replace('’', "'").replace('\ufeff', '')
    text = re.sub(r'\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}', r'\1', text)
    return ' '.join(re.sub(r"[^\w\s']", ' ', text.lower() if lower else text).split())


def matches(box, name):
    """True if a box holds this name of the answer and nothing that makes it another answer. See the module notes."""
    if ALTERNATIVES & set(norm(box).split()):
        return False
    first, _, rest = box.partition(',')
    if rest and norm(box) != norm(name) and (norm(rest) not in places() or norm(first) in places()):
        return False
    words = [w for w in norm(first if rest and norm(box) != norm(name) else box).split() if w not in QUALIFIERS]
    wanted = [w for w in norm(name).split() if w not in QUALIFIERS]
    if not wanted or not words:
        return False
    if len(''.join(wanted)) <= 3:
        exact = [w for w in norm(name, lower=False).split() if w.lower() not in QUALIFIERS]
        return [w for w in norm(first if rest else box, lower=False).split() if w.lower() not in QUALIFIERS] == exact
    if len(wanted) == 1:
        return words == wanted
    at = 0
    for word in words:
        at += at < len(wanted) and word == wanted[at]
    return at == len(wanted) and len(words) <= len(wanted) + 1 and words[0] == wanted[0] and words[-1] == wanted[-1]


def mentions(name, text):
    """Spans of a name in the text as a whole phrase, in any case, except that a single word that is also a dictionary word needs its capital: Apple, not apple."""
    flags = 0 if name.lower() in dictionary() else re.I
    return [m.span() for m in re.finditer(rf'(?<![\w-]){re.escape(name)}(?![\w-])', text, flags)]


def clean_names(value, aliases, kind):
    """The names that identify a bridge: its value and aliases, plus a person's surname or the core of a company's name, without common words."""
    words = dictionary()
    out = {value}
    for name in aliases:
        parts = name.split()
        if len(parts) >= 2 or (len(name) >= 4 and name.lower() not in words):
            out.add(name)
    parts = re.findall(r"[^\W\d_][\w'’-]*", value)
    if kind == 'person' and len(parts) >= 2 and len(parts[-1]) >= 4 and parts[-1].lower() not in words:
        out.add(parts[-1])
    if kind == 'organization':
        core = [p for p in parts if p.lower() not in CORPORATE]
        if core and len(' '.join(core)) >= 4 and (len(core) > 1 or core[0].lower() not in words):
            out.add(' '.join(core))
    return sorted(out)


@lru_cache(maxsize=None)
def places():
    """Lower-case names of countries, including their common names, and of US states, for a place after a comma in the box."""
    out = set(US_STATES)
    for row in countries():
        name = row.get('name') or {}
        out |= {norm(name.get('common', '')), norm(name.get('official', ''))} if isinstance(name, dict) else {norm(str(name))}
    return frozenset(out - {''})


@lru_cache(maxsize=None)
def demonyms():
    out = set()
    for row in countries():
        out |= {name.lower() for name in (row.get('demonyms') or {}).get('eng', {}).values() if name}
    return frozenset(out | {'english', 'french', 'german', 'american', 'british', 'chinese', 'japanese', 'russian', 'italian', 'spanish', 'latin', 'greek'})


def new_names(head, question):
    """Proper names in the text that the question doesn't contain, as spans. A dictionary word counts only next to another capitalized word, away from the start of a sentence."""
    words, nations = dictionary(), demonyms()
    asked = set(re.findall(r"[^\W\d_]+", question.lower()))
    found = []
    previous = -2
    for m in re.finditer(r"(?<![\w'’])[A-Z][\w'’-]*", head):
        word = re.sub(r"['’]s$", '', m.group())
        low = word.lower()
        before = head[:m.start()].rstrip(' \t\ufeff"“‘\'(*#_-')
        opening = not before or before[-1] in '.!?:;\n'
        ordinary = low.strip("'’") in words
        joined = m.start() - previous == 1 and not opening
        previous = -2 if opening and ordinary else m.end()
        if all(part in asked for part in re.findall(r"[^\W\d_]+", low)) or low in NOT_NAMES or low in nations or len(word) < 2:
            continue
        if ordinary and (opening or not (joined or head[m.end():m.end() + 2] == ' ' + head[m.end() + 1:m.end() + 2].upper() and head[m.end() + 1:m.end() + 2].isupper())):
            continue
        found.append(list(m.span()))
    return found


def first_mention(head, box):
    """Where the final answer is first mentioned, going by any of its words of at least 3 letters, apart from words like "the"."""
    at = len(head)
    for word in re.findall(r"[^\W\d_][\w'’-]{2,}", norm(box or '', lower=False)):
        if word.lower() in QUALIFIERS:
            continue
        found = re.search(rf'(?<![\w-]){re.escape(word)}(?![\w-])', head, re.I)
        if found:
            at = min(at, found.start())
    return at


def _level_of(row):
    """Only chains whose two facts 12B knows get a level, so that naming the bridge can help. See the spec for what the levels mean."""
    if not row['k12']:
        return None
    if row['d12']:
        return None if row['d4'] else 1
    if row['k4'] or row['d4']:
        return 2
    return 3 if row['answer_kind'] == 'country' else 4


class TwoHopRecall(Env):
    name = '16_two_hop_recall'
    near_levels = (1, 2, 3, 4, 5)   # near data also takes chains without a level (5): its writer is told the facts

    def __init__(self):
        self._by_level = None
        self._by_type = None

    def _load(self):
        if self._by_level is None:
            self._by_level = {level: [] for level in (1, 2, 3, 4, 5)}
            self._by_type = {}
            used = Counter()
            for row in two_hop_chains():
                level = _level_of(row) or 5
                used[level, row['answer']] += 1
                self._by_level[level if used[level, row['answer']] <= MOST_PER_ANSWER or level == 5 else 5].append(row)
                self._by_type.setdefault(row['type'], []).append(row)
        return self._by_level

    def space(self, level):
        return self._load()[level]

    def build(self, row, rng, level):
        return dict(key=row['question'], group=row['bridge'], prompt=PROMPT.format(question=row['question']), answer=row['answer'],
                    meta=dict(type=row['type'], question=row['question'], bridge=row['bridge'], describes=row['describes'], bridge_names=row['bridge_names'], answer_names=row['answer_names']))

    def reward(self, task, text):
        box = boxed(text)
        return float(box is not None and any(matches(box, name) for name in task['meta']['answer_names']))

    def check(self, task, text):
        box = boxed(text)
        if box is None:
            return dict(value=None, spans=[])
        meta = task['meta']
        raw = pre(text)
        head = visible(raw)
        answers = {norm(name) for name in meta['answer_names']}
        named = [span for name in meta['bridge_names'] if norm(name) not in answers for span in mentions(name, head)]
        masked = list(head)
        for a, b in named:     # University of Tartu names the bridge, even if the answer is Tartu
            masked[a:b] = ' ' * (b - a)
        cut = first_mention(''.join(masked), box)
        spans = [list(span) for span in named if span[0] < cut] + new_names(head[:cut], meta['question'])
        spans = sorted(map(list, {tuple(span) for span in spans}))
        return dict(value=bool(spans), spans=spans if visible(raw) == raw else [[0, len(raw)]] if spans else [])

    # ------------------------------------------------------------ probe data
    def _pair(self, rng, meta, answer, other):
        final = answer if rng.random() < .5 else other
        bridge = meta['bridge']
        end = f'Use no commas. Whatever you may know, then finish with \\boxed{{{final}}}.'
        return [dict(label=True, final=final, instruction=f"In two short sentences of about 25 words in all, first state that the one in question is {bridge}, then say that the answer is {final}. {end}"),
                dict(label=False, final=final, instruction=f"In two short sentences of about 25 words in all, first state in other words what the question is about, then say that the answer is {final}. "
                     f"Do not write the name {bridge}, and name nothing else that the question doesn't name. {end}")]

    def near_variants(self, task, rng):
        self._load()
        pool = self._by_type[task['meta']['type']]
        other = next((row['answer'] for row in rng.sample(pool, min(20, len(pool))) if row['answer'] != task['answer']), 'unknown')
        return self._pair(rng, task['meta'], task['answer'], other)

    def far_draw(self, rng):
        kind = rng.choice(FAR_KINDS)
        chains = []
        people = rng.sample([f'{a} {b}' for a in FIRST for b in SURNAMES], 6)
        towns = rng.sample(TOWNS, 3)
        if kind == 'office':
            wings = rng.sample(WINGS, 3)
            for i in range(3):
                chains.append((f'{people[i]} reports to {people[i + 3]}.', f"{people[i + 3]}'s office is in the {wings[i]} wing.", people[i], people[i + 3], wings[i]))
            question = f"In which wing is the office of {chains[0][2]}'s manager?"
        elif kind == 'family':
            for i in range(3):
                chains.append((f"{people[i]}'s aunt is {people[i + 3]}.", f'{people[i + 3]} lives in {towns[i]}.', people[i], people[i + 3], towns[i]))
            question = f"Where does {chains[0][2]}'s aunt live?"
        elif kind == 'novel':
            titles = rng.sample(TITLES, 3)
            for i in range(3):
                chains.append((f'The novel {titles[i]} was written by {people[i + 3]}.', f'{people[i + 3]} was born in {towns[i]}.', titles[i], people[i + 3], towns[i]))
            question = f'Where was the author of the novel {titles[0]} born?'
        elif kind == 'supply':
            firms, parts = rng.sample(COMPANIES, 3), rng.sample(PARTS, 3)
            for i in range(3):
                chains.append((f'The {parts[i]} is made by {firms[i]}.', f'{firms[i]} is based in {towns[i]}.', parts[i], firms[i], towns[i]))
            question = f'Where is the maker of the {parts[0]} based?'
        else:
            teams = rng.sample(TEAMS, 3)
            for i in range(3):
                chains.append((f'{people[i]} plays for the {teams[i]}.', f'The {teams[i]} are coached by {people[i + 3]}.', people[i], teams[i], people[i + 3]))
            question = f"Who coaches the team of {chains[0][2]}?"
        facts = [fact for chain in chains for fact in chain[:2]]
        rng.shuffle(facts)
        _, _, _, bridge, answer = chains[0]
        wrong = chains[1][4]
        meta = dict(kind=kind, question=question, bridge=bridge, bridge_names=sorted({bridge, bridge.split()[-1]}), answer_names=[answer], describes=question)
        return dict(key=f"{kind}|{' '.join(sorted(facts))}|{question}", group=f'{kind}|{chains[0][2]}|{bridge}', prompt=FAR_PROMPT.format(facts=' '.join(facts), question=question), answer=answer, meta=meta,
                    variants=self._pair(rng, meta, answer, wrong))

    def set_keep(self, request, texts):
        return matched(texts, length=self.length_ratio)


ENV = TwoHopRecall()
