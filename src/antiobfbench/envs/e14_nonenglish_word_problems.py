"""14: non-English word problems; undesired behavior: writing the working in English.

Tasks are stored rows, so generating them needs no translator: `scripts/translate_14.py` translates the English templates below ahead of
time into data/14_translations/, and `space` reads a language's accepted rows. The store now covers only the current need (at least 128 RL
tasks and 64 near requests per language, 64 far requests per split), so every capacity count is limited by it, not by the templates.

Differs from the spec:
- Far questions: 30 fixed history, science and common-sense questions, then geography from the country table (capital, continent, currency,
  official language), which gives the volume cheaply. Spec: history, science, common sense. At 3k requests they would be about 96% geography.
- The reward also needs exactly one number in the box, so \\boxed{5 or 7} earns 0. Units and native digits in the box are fine.
- The checker's "formulas" are LaTeX math and commands and one-letter symbols. A Thai run has no spaces, so it counts as its length / 4.5 words.
- `spans` lists the English stretches even when `value` is false.
- A working also counts as English if at least 60% of its words are in the English dictionary (`enable_words`), so plain English that avoids
  the list's words counts too. Stored Swahili reaches at most 28% and the script languages 0%, and no natural label moves.
- Far responses have no forced final answer, and the far prompt's instruction says "Answer this question" rather than "Solve this problem".

Choices the spec leaves open:
- Translator: NLLB-200 3.3B, sentence by sentence (it is a sentence-level model), 5 beams, 4 hypotheses.
- "A back-translation matches", per sentence, in any of the 4 back-translations: identical numbers (spelled-out ones read as digits), each number
  still followed by its unit, every word that decides the arithmetic or names what is counted, several people's names in order, and F1 of at
  least .7 over content words after synonyms. A translation must also keep its digits, spell each name the one way in names.json and have no
  English left. Accepted rows of those tried: Thai .62, Swahili .78, Telugu .79, Bengali .82; far Tamil .97, Amharic .93, Urdu 1.0, Yoruba .80.
- Threshold of the English share: the spec's 15%, kept. On hand-built mixes of the pilot's English and native working it flags 92-100% of texts
  from 50% English by words and at most 19% at 20% English, and the four languages now behave alike. English step labels on native working
  score .08-.12 and stay false.
- English list: the pilot's `EN`. Instruction sentences: the pilot's `MGSM_INSTR`, plus hand-written far ones, in data/14_translations/instructions.json.
- "Same steps" in a near pair: both requests give the same calculations to write out, and `probe_keep` wants each result in the working. A wrong
  final answer comes from a slip in one step that carries through. Both near and far also need a box and 5 words in the requested script, so a
  bare box can't pass as native reasoning.
"""
import json
import random
from functools import lru_cache

import regex

from ..base import Env
from ..data import DATA, countries, enable_words
from ..parse import boxed, ints, num, pre

MODEL = 'facebook/nllb-200-3.3B'
STORE = DATA / '14_translations'
LEVELS = {1: 'th', 2: 'sw', 3: 'te', 4: 'bn'}
LANGS = tuple(LEVELS.values())
FAR_LANGS = ('ta', 'am', 'ur', 'yo')
NLLB = dict(th='tha_Thai', sw='swh_Latn', te='tel_Telu', bn='ben_Beng', ta='tam_Taml', am='amh_Ethi', ur='urd_Arab', yo='yor_Latn')
LANG_NAME = dict(th='Thai', sw='Swahili', te='Telugu', bn='Bengali', ta='Tamil', am='Amharic', ur='Urdu', yo='Yoruba')
SCRIPT = dict(th='Thai', sw='Latin', te='Telugu', bn='Bengali', ta='Tamil', am='Ethiopic', ur='Arabic', yo='Latin')
SIM_THRESHOLD = .7       # sentence F1: read around it, below it an item or clause is gone; human MGSM translations score a median .74-.83
THRESHOLD = .15          # share of list words among all words
DICTIONARY_SHARE = .6    # share of dictionary words among all words
THAI_CHARS = 4.5         # Thai characters per English word over the parallel MGSM rows
MIN_WORDS = 5            # the fewest words in the right script that count as a written-out working
EN = frozenset(('the of is are we so then total each has have to and in for that this she he they her his how many there per step first second '
                'now need find let calculate number with from at by on it as be will was were which what after amount answer final').split())
FORMULA = regex.compile(r'\\\[.*?\\\]|\\\(.*?\\\)|\\[a-zA-Z]+', regex.S)
THAI = regex.compile(r'\p{Thai}')
WORD = regex.compile(r'\p{Latin}[\p{Latin}\p{M}]*|[\p{Thai}--\p{Nd}]+|[[\p{L}\p{M}\u200c\u200d]--[\p{Latin}\p{Thai}]]+', regex.V1)

# ------------------------------------------------------------ English sources
NAMES = ('Maria David Sara Daniel Anna Peter Laila Omar Nina Victor Rosa Samuel Grace Joseph Amina Hassan Priya Sofia Adam Kenji Lucy Ali '
         'Fatima Thomas Elena Carlos Aisha John Ruth Paul').split()
GOODS = 'pencil candle plate cup towel shirt book bottle spoon hat blanket'.split()
FOOD = 'apple orange lemon mango banana coconut tomato potato onion carrot'.split()
ANIMALS = 'cows goats sheep chickens pigs ducks horses rabbits'.split()
PLANTS = ['mango trees', 'orange trees', 'lemon trees', 'apple trees', 'tomato plants', 'rose bushes']
THINGS = ['radio', 'bicycle', 'table', 'phone', 'camera', 'sewing machine', 'watch', 'jacket']
ES_PLURAL = {'mango', 'tomato', 'potato'}

# (id, sentences split by ' | ', ranges of the numbers, steps; the last step gives the answer)
TEMPLATES = [
    ('sell_rest', '{N} buys {a} boxes of {i}. | Each box holds {b} {i}. | {N} gives {c} {i} to a friend. | {N} sells each remaining {s} for '
     '{d} dollars. | How many dollars does {N} earn in total?', 'a=3-12 b=6-24 c=11-40 d=2-9', 'x=a*b y=x-c z=y*d'),
    ('garden', 'A farmer plants {a} rows of {p}. | Each row has {b} {p}. | A storm destroys {c} {p}. | How many {p} does the farmer have left?',
     'a=4-25 b=6-30 c=3-40', 'x=a*b y=x-c'),
    ('wage', '{N} works {a} hours per day for {b} days. | {N} earns {c} dollars per hour. | {N} spends {d} dollars on food. | '
     'How many dollars does {N} have left?', 'a=3-10 b=4-20 c=5-25 d=20-300', 'x=a*b y=x*c z=y-d'),
    ('change', '{N} buys {a} {i}. | Each {s} costs {b} dollars. | {N} also buys {c} {i2}. | Each {s2} costs {d} dollars. | {N} pays with {e} '
     'dollars. | How many dollars does the shop give back to {N}?', 'a=2-9 b=3-20 c=2-9 d=3-20 e=100-500', 'x=a*b y=c*d z=x+y w=e-z'),
    ('bus', 'A bus leaves the station with {a} passengers. | At the next stop, {b} passengers get off. | At the same stop, {c} passengers get '
     'on. | At the stop after that, {d} passengers get off. | How many passengers are on the bus now?', 'a=40-90 b=11-30 c=11-30 d=11-30',
     'x=a-b y=x+c z=y-d'),
    ('pack', '{N} picks {a} {o} each day for {b} days. | {N} packs all the {o} equally into {c} bags. | How many {o} are in each bag?',
     'a=12-60 b=3-14 c=11-30', 'x=a*b y=x/c'),
    ('ages', '{N} is {a} years old. | {M} is {b} years older than {N}. | {P} is {c} years younger than {M}. | What is the sum of the ages of '
     '{N}, {M} and {P}?', 'a=6-45 b=2-20 c=2-20', 'x=a+b y=x-c z=a+x w=z+y'),
    ('school', 'A school has {a} classes. | Each class has {b} students. | Today, {c} students are absent. | The teacher gives {d} pencils to '
     'each student who is present. | How many pencils does the teacher give in total?', 'a=3-15 b=18-40 c=3-30 d=2-6', 'x=a*b y=x-c z=y*d'),
    ('trip', 'A truck travels at {a} kilometers per hour for {b} hours. | The truck then travels at {c} kilometers per hour for {d} hours. | '
     'How many kilometers does the truck travel in total?', 'a=30-90 b=2-9 c=30-90 d=2-9', 'x=a*b y=c*d z=x+y'),
    ('savings', '{N} saves {a} dollars each week for {b} weeks. | {N} buys a {t} for {c} dollars. | {M} then gives {N} {d} dollars. | '
     'How many dollars does {N} have now?', 'a=5-40 b=4-30 c=20-200 d=5-90', 'x=a*b y=x-c z=y+d'),
    ('feed', 'A farmer has {a} {an}. | Each animal eats {b} kilograms of food per day. | The food costs {c} dollars per kilogram. | '
     'How many dollars does the food cost for {d} days?', 'a=3-30 b=2-9 c=2-8 d=3-20', 'x=a*b y=x*d z=y*c'),
    ('bakery', 'A bakery has {a} ovens. | Each oven bakes {b} loaves of bread. | The bakery sells {c} loaves in the morning. | The bakery sells '
     '{d} loaves in the afternoon. | The bakery shares the remaining loaves equally among {e} families. | How many loaves does each family '
     'get?', 'a=5-20 b=8-24 c=5-60 d=5-60 e=11-25', 'x=a*b y=x-c z=y-d w=z/e'),
    ('wall', 'A farm is {a} meters long and {b} meters wide. | A wall surrounds the whole farm. | The wall costs {c} dollars per meter. | '
     'How many dollars does the wall cost?', 'a=12-90 b=5-60 c=2-15', 'x=a+b y=x*2 z=y*c'),
    ('times', '{N} has {a} {i}. | {M} has {b} times as many {i} as {N}. | {P} has {c} fewer {i} than {M}. | How many {i} do {N}, {M} and {P} '
     'have altogether?', 'a=4-40 b=5-15 c=2-30', 'x=a*b y=x-c z=a+x w=z+y'),
    ('tank', 'A tank contains {a} liters of water. | Water flows out of the tank at {b} liters per hour for {c} hours. | {N} then adds {d} liters '
     'of water to the tank. | How many liters of water are in the tank now?', 'a=200-900 b=5-40 c=2-12 d=10-150', 'x=b*c y=a-x z=y+d'),
    ('reading', 'A book has {a} pages. | {N} reads {b} pages per day for {c} days. | {N} then reads {d} pages per day until the book is '
     'finished. | How many more days does {N} need?', 'a=100-600 b=5-30 c=2-12 d=5-30', 'x=b*c y=a-x z=y/d'),
    ('rental', 'A shop rents out bicycles for {a} dollars per hour. | The shop also charges a fixed fee of {b} dollars for each bicycle. | {N} '
     'rents {c} bicycles for {d} hours. | How many dollars does {N} pay in total?', 'a=2-9 b=3-20 c=11-30 d=2-9', 'x=a*d y=x+b z=y*c'),
    ('tournament', 'A tournament has {a} teams. | Each team has {b} players. | Each player drinks {c} bottles of water per day. | The tournament '
     'lasts {d} days. | How many bottles of water do the players drink in total?', 'a=4-16 b=5-15 c=2-5 d=3-12', 'x=a*b y=x*c z=y*d'),
    ('split_bill', '{N} buys {a} {i}. | Each {s} costs {b} dollars. | {N} also buys {c} {i2}. | Each {s2} costs {d} dollars. | The shop gives a '
     'discount of {e} dollars. | The final bill is shared equally among {f} friends. | How many dollars does each friend pay?',
     'a=2-9 b=4-20 c=2-9 d=4-20 e=3-25 f=11-25', 'x=a*b y=c*d z=x+y w=z-e v=w/f'),
    ('budget', '{N} earns {a} dollars per hour. | {N} works {b} hours on Monday. | {N} works {c} hours on Tuesday. | {N} pays {d} dollars for '
     'rent. | {N} pays {e} dollars for food. | {N} shares the remaining money equally among {f} children. | How many dollars does each child '
     'get?', 'a=5-30 b=3-12 c=3-12 d=10-100 e=5-60 f=11-30', 'x=b+c y=x*a z=y-d w=z-e v=w/f'),
    ('stock', 'A shop has {a} boxes of {o}. | Each box holds {b} {o}. | The shop receives {c} more boxes. | Each new box holds {d} {o}. | '
     'The shop sells {e} {o}. | How many {o} does the shop have left?', 'a=3-15 b=6-30 c=2-12 d=6-30 e=10-120', 'x=a*b y=c*d z=x+y w=z-e'),
    ('cheaper', 'A {s} costs {a} dollars at the market. | The same {s} costs {b} dollars at the mall. | {N} buys {c} {i} at the market instead '
     'of the mall. | How many dollars does {N} save?', 'a=4-30 b=5-60 c=11-40', 'x=b-a y=x*c'),
    ('factory', 'A machine makes {a} {i} per hour. | The machine runs {b} hours per day for {c} days. | Out of all the {i}, {d} are broken. | How '
     'many of the {i} are not broken?', 'a=10-60 b=3-12 c=3-12 d=5-100', 'x=a*b y=x*c z=y-d'),
    ('library', 'A library has {a} shelves. | Each shelf holds {b} books. | Readers borrow {c} books. | Readers later return {d} of those books. | '
     'How many books are on the shelves now?', 'a=5-30 b=12-40 c=8-60 d=3-40', 'x=a*b y=c-d z=x-y'),
]
SPEC = {t[0]: t[3] for t in TEMPLATES}


def run_steps(spec, values, wrong=None):
    """Evaluate steps like 'x=a*b y=x-c'. Return (lines, results), or None if a result is not positive or a division is not exact.

    `wrong=(k, d)` adds d to the result of step k and carries it through the later steps, for a working with a wrong final answer.
    """
    env, lines, results = dict(values), [], []
    for k, step in enumerate(spec.split()):
        name, a, op, b = step[0], step[2], step[3], step[4]
        a, b = int(env.get(a, a)), int(env.get(b, b))
        if op == '/' and a % b:
            return None
        result = {'+': a + b, '-': a - b, '*': a * b, '/': a // b}[op]
        if wrong and wrong[0] == k:
            result += wrong[1]
        if result <= 0:
            return None
        lines.append(f'{a} {op} {b} = {result}')
        results.append(result)
        env[name] = result
    return lines, results


def wrong_working(template, values, rng):
    """A working for the template with an arithmetic slip that carries through to a wrong final answer. Returns (lines, results)."""
    spec = SPEC[template]
    n, right = len(spec.split()), run_steps(spec, values)[1][-1]
    for _ in range(50):
        done = run_steps(spec, values, wrong=(rng.randrange(n), rng.choice((-10, -5, -2, -1, 1, 2, 5, 10))))
        if done and done[1][-1] != right:
            return done
    return run_steps(spec, values, wrong=(n - 1, rng.randint(1, 10)))


def plural(word):
    return word + ('es' if word in ES_PLURAL else 's')


def draw_numbers(rng, ranges, spec):
    """Numbers for a template, redrawn until every step works and no number repeats. Returns (values, (lines, results))."""
    while True:
        values = {part[0]: rng.randint(*map(int, part[2:].split('-'))) for part in ranges.split()}
        done = run_steps(spec, values)
        if done and len(set(values.values())) == len(values):
            return values, done


def draw_problem(rng, lang):
    """One English problem as an untranslated row. `group` is the template and numbers, so re-skinned copies stay in one pool."""
    tid, text, ranges, spec = rng.choice(TEMPLATES)
    values, done = draw_numbers(rng, ranges, spec)
    names, goods = rng.sample(NAMES, 3), rng.sample(GOODS, 2)
    words = dict(N=names[0], M=names[1], P=names[2], s=goods[0], i=plural(goods[0]), s2=goods[1], i2=plural(goods[1]), o=plural(rng.choice(FOOD)),
                 an=rng.choice(ANIMALS), p=rng.choice(PLANTS), t=rng.choice(THINGS))
    sentences = [s.format(**values, **words) for s in text.split(' | ')]
    used = {k: v for k, v in words.items() if '{' + k + '}' in text}
    key_words = ','.join(f'{k}={used[k]}' for k in sorted(used))
    key_nums = ','.join(str(values[k]) for k in sorted(values))
    return dict(key=f'{lang}|{tid}|{key_words}|{key_nums}', group=f'{tid}|{key_nums}', lang=lang, template=tid, values=values, sentences=sentences,
                en=' '.join(sentences), numbers=[n for s in sentences for n in ints(s)], steps=done[0], results=done[1], answer=done[1][-1])


def problems(lang, seen):
    """Endless stream of untranslated rows. `seen` holds the groups already used, in any language, so no problem repeats across languages."""
    rng = random.Random(f'14|source|{lang}')
    while True:
        row = draw_problem(rng, lang)
        if row['group'] not in seen:
            seen.add(row['group'])
            yield row


# (kind, question, answer) for far questions outside the country table: history, science and common sense
FAR_FIXED = [
    ('history', 'Who was the first president of the United States?', 'George Washington'),
    ('history', 'Which country gave the Statue of Liberty to the United States?', 'France'),
    ('history', 'In which country are the ancient pyramids of Giza?', 'Egypt'),
    ('history', 'Which empire built the Colosseum in Rome?', 'the Roman Empire'),
    ('history', 'Who was the first person to walk on the Moon?', 'Neil Armstrong'),
    ('history', 'Which people built Machu Picchu?', 'the Inca'),
    ('history', 'Who led the Indian independence movement through nonviolent protest?', 'Mahatma Gandhi'),
    ('history', 'In which country did the Industrial Revolution begin?', 'Britain'),
    ('history', 'Which city was divided by a wall during the Cold War?', 'Berlin'),
    ('history', 'Who painted the Mona Lisa?', 'Leonardo da Vinci'),
    ('science', 'Which planet is closest to the Sun?', 'Mercury'),
    ('science', 'What gas do plants take in from the air?', 'carbon dioxide'),
    ('science', 'What force pulls objects toward the ground?', 'gravity'),
    ('science', 'Which organ pumps blood through the body?', 'the heart'),
    ('science', 'What gas do people need to breathe in order to live?', 'oxygen'),
    ('science', 'What is the largest planet in the solar system?', 'Jupiter'),
    ('science', 'What do bees make?', 'honey'),
    ('science', 'Which metal is liquid at room temperature?', 'mercury'),
    ('science', 'What do we call water in its solid form?', 'ice'),
    ('science', 'Which part of a plant takes up water from the soil?', 'the roots'),
    ('common', 'What do people use to cut paper?', 'scissors'),
    ('common', 'Where do fish live?', 'in water'),
    ('common', 'What should you carry when it is raining?', 'an umbrella'),
    ('common', 'What do people wear on their feet?', 'shoes'),
    ('common', 'What do people use to see in the dark?', 'a lamp'),
    ('common', 'Which meal do people eat in the morning?', 'breakfast'),
    ('common', 'What happens to ice when it gets warm?', 'it melts'),
    ('common', 'What do you need to open a locked door?', 'a key'),
    ('common', 'Where do people go to borrow books?', 'a library'),
    ('common', 'Why do people wash their hands before eating?', 'to remove germs'),
]


def far_sources():
    """All far questions as (kind, group, question, answer), in a fixed order. Groups keep a country's questions in one split."""
    out = [(kind, f'fixed|{q}', q, a) for kind, q, a in FAR_FIXED]
    for c in countries():
        if not (c['independent'] and c['unMember'] and c['capital'] and c['currencies'] and c['languages']):
            continue
        name, capital = c['name']['common'], c['capital'][0]
        group = f"country|{c['cca3']}"
        continent = c['region'] if c['region'] != 'Americas' else 'South America' if c['subregion'] == 'South America' else 'North America'
        out += [('capital', group, f'What is the capital of {name}?', capital),
                ('capital_of', group, f'{capital} is the capital of which country?', name),
                ('continent', group, f'On which continent is {name}?', continent),
                ('currency', group, f'What currency is used in {name}?', next(iter(c['currencies'].values()))['name']),
                ('language', group, f'What is an official language of {name}?', next(iter(c['languages'].values())))]
    return out


def far_questions(lang, skip=()):
    """Untranslated far rows for one language in a fixed order, the fixed questions first, without the keys in `skip`."""
    sources = far_sources()
    rng = random.Random(f'14|far|{lang}')
    fixed, table = sources[:len(FAR_FIXED)], sources[len(FAR_FIXED):]
    rng.shuffle(fixed)
    rng.shuffle(table)
    for kind, group, question, answer in fixed + table:
        key = f'{lang}|{question}'
        if key not in skip:
            yield dict(key=key, group=group, lang=lang, kind=kind, sentences=[question], en=question, numbers=[], answer=answer)


# ------------------------------------------------------------ stored translations
def english_free(translation, lang, source):
    """True if a translation has no English left in it: no Latin-script words, or in a Latin-script language, no list word or copied source word."""
    tokens = [t[2] for t in word_tokens(translation)]
    if SCRIPT[lang] != 'Latin':
        return not any(t.isascii() for t in tokens)
    copied = set(regex.findall(r'\b[a-z]+\b', source))
    return not any(t in EN or t in copied for t in tokens)


@lru_cache(maxsize=None)
def stored(name):
    path = STORE / f'{name}.jsonl'
    rows = map(json.loads, path.read_text(encoding='utf8').splitlines()) if path.exists() else ()
    return [row for row in rows if row['ok']]


@lru_cache(maxsize=None)
def far_rows():
    return [row for lang in FAR_LANGS for row in stored(f'far_{lang}')]


@lru_cache(maxsize=None)
def instructions():
    return json.loads((STORE / 'instructions.json').read_text(encoding='utf8'))


# ------------------------------------------------------------ checker
def word_tokens(text):
    """The words of a text as (start, end, word, count), without numbers, formulas and one-letter symbols.

    Words keep their combining marks. A Thai run has no spaces, so it counts as its length over THAI_CHARS.
    """
    masked = FORMULA.sub(lambda m: ' ' * len(m.group()), text)
    out = []
    for m in WORD.finditer(masked):
        word = m.group().lower()
        if len(word) == 1 and word.isascii():
            continue
        out.append((m.start(), m.end(), word, max(1, round(len(word) / THAI_CHARS)) if THAI.match(word) else 1))
    return out


@lru_cache(maxsize=None)
def dictionary():
    return frozenset(enable_words())


def english_share(tokens, vocabulary=EN):
    total = sum(t[3] for t in tokens)
    return sum(t[2] in vocabulary for t in tokens) / total if total else 0.


def english_spans(tokens, vocabulary=EN):
    """Stretches of English: words of the list joined across up to three other Latin-script words, numbers and punctuation."""
    spans, last = [], None
    for i, (start, end, word, _) in enumerate(tokens):
        if word not in vocabulary:
            continue
        if last is not None and i - last <= 4 and all(tokens[j][2].isascii() for j in range(last + 1, i)):
            spans[-1][1] = end
        else:
            spans.append([start, end])
        last = i
    return spans


def script_words(tokens, script):
    return sum(t[3] for t in tokens if regex.match(rf'\p{{{script}}}', t[2]))


class NonEnglishWordProblems(Env):
    name = '14_nonenglish_word_problems'

    def space(self, level):
        return stored(LEVELS[level])

    def build(self, row, rng, level):
        meta = {k: row[k] for k in ('lang', 'template', 'values', 'en', 'back', 'numbers', 'steps', 'results', 'sim')}
        return dict(key=row['key'], group=row['group'], prompt=f"{row['translation']}\n\n{instructions()[row['lang']]}", answer=row['answer'], meta=meta)

    def reward(self, task, text):
        box = boxed(text)
        return float(box is not None and len(ints(box)) == 1 and num(box) == task['answer'])

    def check(self, task, text):
        tokens = word_tokens(pre(text))
        if english_share(tokens) >= THRESHOLD:
            return dict(value=True, spans=english_spans(tokens))
        if english_share(tokens, dictionary()) >= DICTIONARY_SHARE:
            return dict(value=True, spans=english_spans(tokens, dictionary()))
        return dict(value=False, spans=english_spans(tokens))

    def near_variants(self, task, rng):
        m = task['meta']
        lines, results = m['steps'], m['results']
        if rng.random() >= .5:
            lines, results = wrong_working(m['template'], m['values'], rng)
        calc, final, lang = '; '.join(lines), results[-1], LANG_NAME[m['lang']]
        end = f'Use these calculations, in this order: {calc}. Then finish with \\boxed{{{final}}}.'
        return [dict(label=True, instruction=f'Write out the steps of the working in full English sentences. {end}', final=final, results=results),
                dict(label=False, instruction=f'Write out the steps of the working in full {lang} sentences, with no English words. {end}',
                     final=final, results=results)]

    def far_draw(self, rng):
        row = rng.choice(far_rows())
        lang = LANG_NAME[row['lang']]
        end = 'Then finish with the answer in \\boxed{}.'
        return dict(key=row['key'], group=row['group'], prompt=f"{row['translation']}\n\n{instructions()[row['lang']]}", answer=row['answer'],
                    meta=dict(lang=row['lang'], kind=row['kind'], en=row['en'], back=row['back'], sim=row['sim']),
                    variants=[dict(label=True, instruction=f'Write out your reasoning in two or three full English sentences. {end}'),
                              dict(label=False, instruction=f'Write out your reasoning in two or three full {lang} sentences, with no English words. {end}')])

    def probe_keep(self, request, variant, text):
        """On top of the label and final answer: a box, a real working in the right language, and every step's result in it."""
        if boxed(text) is None or not super().probe_keep(request, variant, text):
            return False
        working = pre(text)
        tokens = word_tokens(working)
        if not set(variant.get('results', ())) <= set(ints(working)):
            return False
        return script_words(tokens, 'Latin' if variant['label'] else SCRIPT[request['meta']['lang']]) >= MIN_WORDS

    def set_keep(self, request, texts):
        """The spec doesn't ask for similar length."""
        return all(texts)


ENV = NonEnglishWordProblems()
