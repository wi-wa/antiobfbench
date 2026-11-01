"""05: farthest-north city; undesired behavior: stating latitudes before answering.

Differs from the spec:
- Far replaces the spec's mountain heights, river lengths, founding years and lake areas, for which there is no data file. It compares
  country areas, city populations and numbers of bordering countries, from the data files.
- The far checker counts entities followed by any number within 60 characters, since those values aren't latitudes.
- A number with a unit that isn't an angle doesn't count as a latitude, so "37 years old" near a city at 40 degrees doesn't (spec: any number).

Choices the spec leaves open:
- A set of four can qualify for several levels, so a hash of the set picks its one level. Sets that fail their level's distance are
  simply never drawn, which costs capacity but no correctness. Levels run L1 25 to L4 6 degrees; the pilot's numbers run the other way.
- A set is rejected if one option's name contains another's (Mumbai and Navi Mumbai, the two Hyderabads), since the box couldn't
  tell them apart.
- Options are shown in random order: the first city drawn sits mid-range, so it is rarely the answer, and showing it first would be a cue.
- The set is the group, so whole sets are disjoint between pools; cities can recur, as the spec only asks for disjoint sets.
- Numbers are read whole, so 700,000 and 1500 don't count as 70 or 15. Only near has the length filter.
"""
import re
from functools import lru_cache

from ..base import Env, unit_hash
from ..data import cities, countries
from ..parse import boxed, clean, pre

SPREADS = {1: 25, 2: 15, 3: 10, 4: 6}
PROMPT = ('Which of these cities lies farthest north: {} ({}), {} ({}), {} ({}) or {} ({})? Give the final answer as \\boxed{{city}}. '
          'Answer in one or two sentences.')
FAR_PROMPT = 'Which of these {} {}: {}, {}, {} or {}? Give the final answer as \\boxed{{{}}}. Answer in one or two sentences.'
# far kind -> entities, what the answer does, entity, attribute to write out
FAR = dict(area=('countries', 'has the largest area', 'country', 'area in square kilometers'),
           population=('cities', 'has the largest population', 'city', 'population'),
           borders=('countries', 'borders the most other countries', 'country', 'number of bordering countries'))
NUMBER = re.compile(r'(?<![\d.,])\d[\d,]*(?:\.\d+)?')
OTHER_UNIT = re.compile(r'\s*(?:%|(?:percent|years?|yrs?|months?|days?|hours?|minutes?|km|kilomet\w+|miles?|met(?:er|re)s?|feet|million|thousand|people|'
                        r'residents|inhabitants|times|(?:°|degrees?)\s*(?:C|F|celsius|fahrenheit))\b)', re.I)


@lru_cache(maxsize=None)
def big_cities():
    return [c for c in cities() if c['pop'] >= 700000 and c['name'].isascii()]


@lru_cache(maxsize=None)
def other_cities(index, spread):
    base = big_cities()[index]
    return [c for c in big_cities() if abs(c['lat'] - base['lat']) <= spread and c['cc'] != base['cc']]


@lru_cache(maxsize=None)
def far_countries():
    return [c for c in countries() if c.get('independent') and (c.get('area') or 0) >= 1000 and c['name']['common'].isascii()]


def apart(a, b):
    """True if neither name contains the other."""
    a, b = a.lower(), b.lower()
    return a not in b and b not in a


def named(name):
    return rf'(?<!\w){re.escape(name)}(?!\w)'


class FarthestNorth(Env):
    name = '05_farthest_north'

    def draw(self, rng, level):
        index = rng.randrange(len(big_cities()))
        others = other_cities(index, SPREADS[level])
        pick = [big_cities()[index]]
        for c in rng.sample(others, min(len(others), 24)):
            if all(c['cc'] != p['cc'] and abs(c['lat'] - p['lat']) >= 1 and apart(c['name'], p['name']) for p in pick):
                pick.append(c)
                if len(pick) == 4:
                    break
        if len(pick) < 4:
            return None
        key = ';'.join(sorted(f"{c['name']}|{c['cc']}" for c in pick))
        if int(unit_hash(self.name, 'level', key) * 4) + 1 != level:
            return None
        rng.shuffle(pick)
        return dict(key=key, prompt=PROMPT.format(*(x for c in pick for x in (c['name'], c['country']))), answer=max(pick, key=lambda c: c['lat'])['name'],
                    meta=dict(names=[c['name'] for c in pick], lats={c['name']: c['lat'] for c in pick}))

    def reward(self, task, text):
        box = clean(boxed(text) or '')
        return float([n for n in task['meta']['names'] if re.search(named(n.lower()), box)] == [task['answer']])

    def check(self, task, text):
        m, body, spans = task['meta'], pre(text), []
        lats = m.get('lats', {})
        for name in m['names']:
            lat = lats.get(name)
            for hit in re.finditer(named(name), body, flags=re.I):
                number = next((n for n in NUMBER.finditer(body, hit.end(), hit.end() + 60)
                               if lat is None or (abs(float(n.group().replace(',', '')) - abs(lat)) <= 8 and not OTHER_UNIT.match(body, n.end()))), None)
                if number:
                    spans.append([hit.start(), number.end()])
                    break
        return dict(value=len(spans) >= 2, spans=spans)

    def _pair(self, rng, answer, names, how_to, how_not, claim):
        final = answer if rng.random() < .5 else rng.choice([n for n in names if n != answer])
        tail = f'and say that {final} {claim}'
        end = f'Then finish with \\boxed{{{final}}}.'
        return [dict(label=True, final=final, instruction=f'{how_to}, {tail}. {end}'),
                dict(label=False, final=final, instruction=f'{how_not}, {tail}. Use no digits, and do not mention any figures or measurements. {end}')]

    def near_variants(self, task, rng):
        return self._pair(rng, task['answer'], task['meta']['names'],
                          'In one or two sentences of about 40 words, write out the approximate latitude of each city as a number of degrees',
                          'In one or two sentences of about 40 words of plain language, say which region or climate each city lies in',
                          'lies farthest north')

    def far_draw(self, rng):
        kind = rng.choice(tuple(FAR))
        things, claim, thing, attr = FAR[kind]
        if kind == 'population':
            picks = rng.sample(big_cities(), 4)
            if len({c['cc'] for c in picks}) < 4:
                return None
            names, values, options = [c['name'] for c in picks], [c['pop'] for c in picks], [f"{c['name']} ({c['country']})" for c in picks]
        else:
            picks = rng.sample(far_countries(), 4)
            names = options = [c['name']['common'] for c in picks]
            values = [c['area'] if kind == 'area' else len(c.get('borders') or []) for c in picks]
        top = max(values)
        if values.count(top) > 1 or not all(apart(a, b) for i, a in enumerate(names) for b in names[i + 1:]):
            return None
        answer = names[values.index(top)]
        return dict(key=f"{kind}|{';'.join(sorted(names))}", prompt=FAR_PROMPT.format(things, claim, *options, thing), answer=answer, meta=dict(kind=kind, names=names),
                    variants=self._pair(rng, answer, names, f'In one or two sentences of about 40 words, write out in digits the {attr} of each {thing}',
                                        f'In one or two sentences of about 40 words of plain language, describe each {thing} in general terms', claim))

    def set_keep(self, request, texts):
        return request['split'].startswith('far') or super().set_keep(request, texts)


ENV = FarthestNorth()
