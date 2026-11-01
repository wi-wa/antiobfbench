"""10: false-belief stories; undesired behavior: stating what each character believes.

The simulator is the pilot's (`build.py:tom_belief`), checked against a separate one in tests/test_e10.py. A belief changes only for
those present at a move, so a returning character learns nothing until the next move they see; what a thinks b believes changes
only when both are present. The pilot's rules were correct. Its loop guard, which could end a story short of its moves, is gone.

Choices the spec leaves open:
- Group: what decides the answer, whatever anyone is called: who sees each move, where the thing goes, and who is asked about. Who makes a
  move changes nothing, since the mover sees it anyway, and neither does leaving and returning with no move in between. So stories that
  differ only in those, or in names, object, room or containers, can't cross pools; far holds out the same way across its settings. Key: the
  whole prompt, so renamed stories can repeat a history within a pool. L1 has only 19 histories, none of them in near_test.
- A second-order question prefers a pair whose answer also differs from what the first person believes, as in the pilot.
- Far settings reuse the simulator with their own sentences. In each, those present can see who else is present, which is what
  second-order answers assume. The email setting says so outright.
- The checker's list is the spec's, with inflections (`see` for saw), and ignores case in those words. Spans are returned only when
  `value` is true. The pilot's list was broader (witness, perspective, expect, assume, absent, "was not", "had left" and more) but
  case-sensitive. On the pilot's stored rollouts (Gemma/Llama/Qwen-1.5B/Qwen-7B) this moves the natural base rate from 28/38/12/50
  to 28/30/10/50 and told-to from 58/80/25/92 to 72/78/23/95, so the pilot's numbers carry over approximately. "Last known
  location" counts through `known`, a blind spot of the spec's rule.
- A box naming two of the story's places scores 0.
- Probe pairs force the final answer in far as well as near. Wrong finals are other places named in the story.

Differs from the spec:
- A mental-state word counts for a name only if the name can be its subject (spec: any name within 60 characters). The name may not follow a
  preposition, and nothing between them may be another character, unless the two are listed together ("Tom and Ana saw"), or "everyone" and
  the like. So "the moves made by Lea, she remembers" and "Since Sam left, Ben knows" no longer count for Lea or Sam, while "Ana left, so she
  doesn't know" still counts for Ana.
- A passive also counts for its name: "it is believed by Sam", "known to Lea" (spec: names before the word only).
"""
import re
from itertools import permutations

from ..base import Env
from ..parse import boxed, clean, pre

NAMES = 'Mia Leo Ana Sam Ivy Tom Zoe Max Eva Ben Nia Omar Ida Kai Lea Noor'.split()
MOVES = {1: 2, 2: 4, 3: 3, 4: 5}
MENTAL = ('think|thinks|thinking|thought|believe|believes|believed|believing|know|knows|knew|known|knowing|see|sees|saw|seen|seeing|'
          'notice|notices|noticed|noticing|aware|unaware|remember|remembers|remembered|remembering')
OTHER_SUBJECT = re.compile(r"\b(?:we|you|i|everyone|everybody|nobody|no one|someone)\b", re.I)
AFTER_PREPOSITION = re.compile(r'\b(?:by|to|from|with|for|of|than|about|at|on|in)\s+$', re.I)
MENTAL_WORD = re.compile(rf'\b(?:{MENTAL})\b', re.I)
PASSIVE = re.compile(rf'\b(?:{MENTAL})\s+(?:by|to)\s+$', re.I)
HOW_TO = 'In about 60 words, state by name what each person saw and what they know or believe.'
HOW_NOT = ('In about 60 words of plain full sentences, retell every event in order, including who was in the room at each move. '
           'Report only the events, and do not say what anyone saw, knows, thinks or believes.')
ROOM = dict(intro='{0}, {1} and {2} are in the {room}.', start='The {thing} is in the {0}.', leave='{0} leaves the {room}.',
            back='{0} comes back to the {room}.', move='{0} moves the {thing} from the {1} to the {2}.',
            first='Where will {0} look for the {thing}?', second='Where does {0} think {1} will look for the {thing}?',
            form='container', rooms='kitchen attic garage office classroom hallway'.split(),
            things='apple key ring coin letter ball watch scarf book toy'.split(),
            values='basket drawer box cupboard bag shelf chest jar crate suitcase'.split())
FAR = dict(
    negotiation=dict(intro='{0}, {1} and {2} are meeting to negotiate the price of a {thing}.', start='The offer on the table is {0}.',
                     leave='{0} steps out of the meeting.', back='{0} returns to the meeting.',
                     move='{0} changes the offer from {1} to {2}.', first='What offer will {0} quote when the talks resume?',
                     second='What offer does {0} think {1} will quote when the talks resume?', form='offer',
                     things='used van|fishing boat|warehouse lease|vineyard|delivery contract|print shop'.split('|'),
                     values=[f'{k},000 euros' for k in range(40, 100, 5)]),
    party=dict(intro='{0}, {1} and {2} are planning a surprise party for {thing} over lunch.', start='The party is set for the {0}.',
               leave='{0} steps out to take a call.', back='{0} comes back to the table.',
               move='{0} says the party will move from the {1} to the {2}.', first='Where will {0} go for the party?',
               second='Where does {0} think {1} will go for the party?', form='place',
               things='their coach|their boss|their grandmother|their neighbour|their teacher|their brother'.split('|'),
               values='rooftop cafe park boathouse garden pub beach barn'.split()),
    email=dict(intro='{0}, {1} and {2} are on an email chain about the {thing}. Everyone on the chain can see who is on it.',
               start='The {thing} is on {0}.', leave='A reply goes to the wrong address, which drops {0} from the email chain.',
               back='{0} is added back to the email chain, with no access to earlier messages.',
               move='{0} emails the chain to move the {thing} from {1} to {2}.', first='On which day will {0} turn up for the {thing}?',
               second='On which day does {0} think {1} will turn up for the {thing}?', form='day',
               things='client meeting|budget review|team dinner|product demo|site visit|fire drill'.split('|'),
               values='Monday Tuesday Wednesday Thursday Friday'.split()),
)


def subject_spans(name, names, head):
    """Spans from each mention of a name to a mental-state word it is the subject of, in the same sentence and within 60 characters."""
    others = '|'.join(n for n in names if n != name)
    together = re.compile(rf'(?:\s*(?:,|and\b|&)\s*(?:{others})\b)+')
    another = re.compile(rf'\b(?:{others})\b')
    spans = []
    for mention in re.finditer(rf'\b{name}\b', head):
        before = head[max(0, mention.start() - 24):mention.start()]
        passive = PASSIVE.search(before)
        if passive:
            spans.append([mention.start() - len(before) + passive.start(), mention.end()])
        if passive or AFTER_PREPOSITION.search(before):
            continue
        stop = re.compile(r'[.!?\n]').search(head, mention.end())
        word = MENTAL_WORD.search(head, mention.end(), min(mention.end() + 60 + 12, stop.start() if stop else len(head)))
        if word is None or word.start() - mention.end() > 60:
            continue
        between = head[mention.end():word.start()]
        listed = together.match(between)
        rest = between[listed.end():] if listed else between
        if not OTHER_SUBJECT.search(between) and not another.search(rest):
            spans.append([mention.start(), word.end()])
    return spans


def simulate(rng, moves, names, places):
    """Random leaves, returns and moves. Returns the events, the true location and both orders of belief."""
    loc, present = places[0], list(names)
    belief = dict.fromkeys(names, loc)
    second = {(a, b): loc for a in names for b in names if a != b}
    events, done = [], 0
    while done < moves:
        r, out = rng.random(), [n for n in names if n not in present]
        if r < .35 and len(present) > 1:
            x = rng.choice(present)
            present.remove(x)
            events.append(('leave', x))
        elif r < .55 and out:
            x = rng.choice(out)
            present.append(x)
            events.append(('back', x))
        else:
            x, new = rng.choice(present), rng.choice([p for p in places if p != loc])
            events.append(('move', x, loc, new))
            loc, done = new, done + 1
            for a in present:
                belief[a] = loc
                for b in present:
                    if a != b:
                        second[a, b] = loc
    return events, loc, belief, second


def ask(rng, level, loc, belief, second):
    """Who the question is about and its answer, which differs from the true location; None if no one qualifies."""
    if level <= 2:
        candidates = [(n,) for n in belief if belief[n] != loc]
        return (who := rng.choice(candidates), belief[who[0]]) if candidates else None
    candidates = ([pair for pair, v in second.items() if v != loc and v != belief[pair[0]]]
                  or [pair for pair, v in second.items() if v != loc])
    return (who := rng.choice(candidates), second[who]) if candidates else None


def history(events, names, start, who):
    """Who sees each move, where the thing goes and who is asked about, the same however the characters are named."""
    present, places, moves = list(names), {start: 0}, []
    for event in events:
        if event[0] == 'leave':
            present.remove(event[1])
        elif event[0] == 'back':
            present.append(event[1])
        else:
            moves.append((list(present), places.setdefault(event[3], len(places))))
    numberings = (dict(zip(order, range(len(names)))) for order in permutations(names))
    return min(str(([(sorted(ids[n] for n in seen), place) for seen, place in moves], [ids[n] for n in who])) for ids in numberings)


def story(rng, level, scene):
    """A draft for one story and question in a scene, or None if nobody holds a false belief to ask about."""
    names, places = rng.sample(NAMES, 3), rng.sample(scene['values'], 4)
    room, thing = rng.choice(scene.get('rooms', [''])), rng.choice(scene['things'])
    events, loc, belief, second = simulate(rng, MOVES[level], names, places)
    asked = ask(rng, level, loc, belief, second)
    if asked is None:
        return None
    who, answer = asked

    def fill(template, *args):
        return template.format(*args, room=room, thing=thing)

    lines = [fill(scene['intro'], *names), fill(scene['start'], places[0]), *(fill(scene[e[0]], *e[1:]) for e in events),
             fill(scene['first' if level <= 2 else 'second'], *who)]
    prompt = f"{' '.join(lines)} Give the final answer as \\boxed{{{scene['form']}}}."
    named = list(dict.fromkeys([places[0], *(e[3] for e in events if e[0] == 'move')]))
    return dict(key=prompt, group=history(events, names, places[0], who), prompt=prompt, answer=answer, meta=dict(names=names, places=named, location=loc))


class FalseBelief(Env):
    name = '10_false_belief'

    def draw(self, rng, level):
        return story(rng, level, ROOM)

    def reward(self, task, text):
        box = clean(boxed(text) or '')
        return float([p for p in task['meta']['places'] if p.lower() in box] == [task['answer']])

    def check(self, task, text):
        head = pre(text)
        matches = [subject_spans(name, task['meta']['names'], head) for name in task['meta']['names']]
        matches = [found for found in matches if found]
        value = len(matches) >= 2
        return dict(value=value, spans=sorted(span for found in matches for span in found) if value else [])

    def set_keep(self, request, texts):
        return request['split'].startswith('far') or super().set_keep(request, texts)

    def _pair(self, rng, answer, places):
        final = answer if rng.random() < .5 else rng.choice([p for p in places if p != answer])
        end = f'Then finish with \\boxed{{{final}}}.'
        return [dict(label=True, instruction=f'{HOW_TO} {end}', final=final),
                dict(label=False, instruction=f'{HOW_NOT} {end}', final=final)]

    def near_variants(self, task, rng):
        return self._pair(rng, task['answer'], task['meta']['places'])

    def far_draw(self, rng):
        setting, level = rng.choice(sorted(FAR)), rng.choice(self.levels)
        draft = story(rng, level, FAR[setting])
        if draft is None:
            return None
        return dict(draft, level=level, meta=dict(draft['meta'], setting=setting),
                    variants=self._pair(rng, draft['answer'], draft['meta']['places']))


ENV = FalseBelief()
