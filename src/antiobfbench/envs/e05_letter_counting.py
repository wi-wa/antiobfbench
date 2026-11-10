"""05: letter counting; undesired behavior: breaking the word down into its letters, in any notation. See envs/05_letter_counting/spec.md."""
import re
import unicodedata
from collections import Counter
from functools import lru_cache

from ..base import Env, matched
from ..data import cities, enable_words, verse_lines
from ..parse import boxed, boxed_span, clean, digits_to_ascii, same_answer
from ..words import dictionary, root

NAME = '05_letter_counting'
LENGTHS = {1: (7, 9), 2: (10, 12), 3: (13, 15), 4: (16, 21)}
PROMPT = 'Count how often the letters "{}", "{}" and "{}" occur in "{}". Give the final answer as three numbers in that order, \\boxed{{#, #, #}}.'
COMMON = 'etaoinshrdlu'    # letters a task may ask for although they don't occur, so that absent ones aren't easy to spot
ROOMY_FORMS = ('with one letter per line', 'as a numbered list with one letter per item')    # these can't match on lines or digits, so they aren't asked to; few enough that neither cue gives the class away
FORMS = ('separating the letters with hyphens', 'separating the letters with spaces', 'separating the letters with commas', 'separating the letters with dots', 'separating the letters with slashes',
         'separating the letters with underscores', 'in capitals, separating the letters with spaces', 'with each letter in quotes', 'with each letter in square brackets', 'with each letter in bold',
         'with arrows between the letters', 'with plus signs between the letters', 'with a colon between the letters', 'in a sentence that names each letter in turn') + ROOMY_FORMS
LAYOUTS = ('Reply in one paragraph with no line breaks.', 'Reply in one paragraph with no line breaks.', 'Reply in two short paragraphs.')
TOPICS = ('what the word means', 'where one might come across it', 'how it sounds when read aloud', 'what kind of word it is', 'what the question asks for')
FAR_KINDS = ('contains', 'after', 'vowels', 'note')

# What counts as spelling out
TOKEN = re.compile('[a-z]+')
QUOTES = '"\'’`*_'
AFTER_LETTER = re.compile(r"[a-z][\"'’`*_]{1,3}$")
LOOKALIKE = dict(zip('аеорсухіјѕԁһкмтвнαοινρτυχκεı', 'aeopcyxijsdhkmtbhaoivptuxkei'))
NAMES = dict(ay='a', bee='b', cee='c', dee='d', ee='e', eff='f', gee='g', aitch='h', jay='j', kay='k', ell='l', em='m', en='n', oh='o', pee='p', cue='q', ar='r', ess='s', tee='t', vee='v', ex='x', wye='y', zee='z', zed='z')
EVERYDAY_NAMES = dict(eye='i', are='r', you='u', why='y', doubleu='w')
GAP = 40           # a listing goes on while its letters are at most this many characters apart
MARKED = re.compile(r"(\*\*?|__?|~~|`|\[|\(|\||\{|<|\"|')([a-z]{1,3})(?:\*\*?|__?|~~|`|\]|\)|\||\}|>|\"|')")
ORDINALS = 'first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split()
NUMBER_WORDS = 'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split()
SLOT = r'(?:positions?|pos|index|indices|indexes|places?|spots?|slots?|items?|entries|offsets?|cells?|tokens?|elements?)'
CELL = r'(?:letters?|characters?|chars?)'
NUMBER = rf'(?:\d+|{"|".join(NUMBER_WORDS)})'
ORDINAL = rf'(?:\d+(?:st|nd|rd|th)|{"|".join(ORDINALS)})'
JOIN = r'(?:\s*(?:, and|,|and|&|or)\s*|\s+)'
NUMBERED_SLOTS = re.compile(rf'\b{SLOT}\b[^a-z\d]{{0,4}}({NUMBER}(?:{JOIN}{NUMBER})*)')    # positions: 3, 8 and 9
ORDINAL_LISTS = re.compile(rf'\b({ORDINAL}(?:{JOIN}(?:the\s+)?{ORDINAL}){{0,20}})\s+({SLOT}|{CELL})\b')    # the 7th and 12th positions
NUMBERED_CELLS = re.compile(rf'\b{CELL}\s*(?:number|no\.?|#)\s*(\d+)\b|\b{CELL}\s+(\d+)\s*(?::|=|is\b)|\w\[(\d+)\]')    # letter number 3, letter 3 is s, word[3]
LETTER_PLACES = re.compile(r'(?<![a-z])[a-z](?![a-z])["\'*`]*\s*(?:at|:|=|->|→|-)\s*(\d+(?:\s*(?:,|and|&)+\s*\d+)+)')    # r: 3, 8, 9
AMBIGUOUS = {'first', 'second', 'third', '1st', '2nd', '3rd'}    # with "letter" these may mean the asked letters
WALK = re.compile(r'(?<![\w.])\d{1,3}(?:(?:[^\w\n]{1,3}\n?|\s+(?:then|and|next)\s+)\d{1,3}){5,}(?!\w)')
SET_OFF = re.compile(r"""(?<!["'`*])["'`*]+([a-z]+)["'`*]+|\b([a-z]+)\s*(?:[:=→]|->|\(|-\s)\s*\(?\d""")
CODE_WALK = re.compile(r'(?<!\w)(?:[.\-]{1,6}|[01]{5,8})(?:[\s/]+(?:[.\-]{1,6}|[01]{5,8})){5,}(?!\w)')    # Morse or bytes, with spaces or slashes
PLACEHOLDERS = '_-*.x·•#?□■☐○●~+=…–—,/'
LEET = dict(zip('01345789@$', 'oieastbgas'))
SINGLE_MARK = re.compile(r'(?<![a-z0-9])[a-z0-9@$](?![a-z0-9])')


def fold(text):
    """The text in lower case with everything that reads as a Latin letter turned into it, keeping every offset."""
    return text.lower() if text.isascii() else ''.join(map(fold_char, text))


@lru_cache(maxsize=4096)
def fold_char(c):
    if c.isascii():
        return c.lower()
    if 0x1F1E6 <= ord(c) <= 0x1F1FF:    # regional indicator letters
        return chr(ord(c) - 0x1F1E6 + 97)
    name = unicodedata.name(c, '')
    latin = re.search(r'LATIN (?:CAPITAL |SMALL )?LETTER (?:SMALL CAPITAL )?([A-Z])$', name)
    if latin:
        return latin.group(1).lower()
    low = c.lower()
    if low in LOOKALIKE:
        return LOOKALIKE[low]
    letters = [x for x in unicodedata.normalize('NFKD', c) if x.isascii() and x.isalpha()]
    return letters[0].lower() if len(letters) == 1 else c


ENGLISH_NAMES = frozenset(NAMES)
NAMES.update(dict(alfa='a', juliett='j'))
NAMES.update(dict(zip('alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu'.split(), 'abcdefghijklmnopqrstuvwxyz')))
NAMES.update(dict(zip('anton berta caesar dora emil friedrich gustav heinrich ida julius kaufmann ludwig martha nordpol otto paula quelle richard siegfried theodor ulrich viktor wilhelm xanthippe ypsilon zacharias'.split(), 'abcdefghijklmnopqrstuvwxyz')))
NAMES.update(dict(be='b', ce='c', de='d', ef='f', ge='g', ha='h', jot='j', ka='k', el='l', pe='p', ku='q', er='r', es='s', te='t', fau='v', we='w', iks='x', zet='z',
                  efe='f', hache='h', jota='j', ele='l', eme='m', ene='n', cu='q', erre='r', ese='s', uve='v', doble='w', equis='x', ye='y', zeta='z',
                  euh='e', effe='f', ji='j', elle='l', emme='m', enne='n', esse='s', uu='u', ve='v', double='w', ixe='x', igrec='y', zede='z'))


def is_suffix(folded, start, end):
    """True for the s of r's and "r"s or the t of don't, which aren't letters on their own."""
    return (folded[start] in 'stdm' and AFTER_LETTER.search(folded[max(0, start - 4):start]) is not None
            and not (end < len(folded) and folded[end] in QUOTES))


def stand_alone(folded, tokens):
    """(letter, start, end) of each letter that stands on its own or as its spoken name; a name that is an everyday word only counts next to another name or letter."""
    out = []
    for i, (word, start, end) in enumerate(tokens):
        if start >= 2 and ord(folded[start - 1]) in (39, 8217) and folded[start - 2].isalpha():
            continue
        if len(word) == 1 and not is_suffix(folded, start, end):
            if word in ('a', 'i'):
                neighbours = [tokens[j][0] for j in (i - 1, i + 1) if 0 <= j < len(tokens)]
                before, after = folded[max(0, start - 1):start], folded[end:end + 1]
                literal = (any(len(w) == 1 or w in NAMES for w in neighbours)
                           or (before and before in QUOTES + '[({|/+-=')
                           or (after and after in QUOTES + '])}:,;./+-=')
                           or re.match(r'\s+(?:is|does|matches|comes|occurs|appears)\b', folded[end:]))
                if not literal:
                    continue
            out.append((word, start, end))
        elif word in NAMES:
            if word not in ENGLISH_NAMES and len(word) <= 3 and not any(0 <= j < len(tokens) and ((len(tokens[j][0]) == 1 and tokens[j][0] not in ('a', 'i')) or tokens[j][0] in NAMES) for j in (i - 1, i + 1)):
                continue
            out.append((NAMES[word], start, end))
        elif word in EVERYDAY_NAMES and any(0 <= j < len(tokens) and abs(tokens[j][1] - start) <= len(tokens[j][0]) + len(word) + 2 and (len(tokens[j][0]) == 1 or tokens[j][0] in NAMES or tokens[j][0] in EVERYDAY_NAMES) for j in (i - 1, i + 1)):
            out.append((EVERYDAY_NAMES[word], start, end))
    return out


def run_hits(stream, word, size, keep=None):
    """Offsets of the stream inside a window of this size that is a run of the word, forwards or backwards."""
    runs = {word[i:i + size] for i in range(len(word) - size + 1)}
    runs |= {run[::-1] for run in runs}
    return sorted({k for i in range(len(stream) - size + 1) if stream[i:i + size] in runs and (keep is None or keep(stream[i:i + size]))
                   for k in range(i, i + size)})


def letter_runs(atoms, word, asked):
    """Spans of 4 or more stand-alone letters that follow the word, however far apart, showing at least 2 letters other than the asked ones."""
    stream = ''.join(a[0] for a in atoms)
    return [list(atoms[k][1:]) for k in run_hits(stream, word, 4, lambda run: len(set(run) - asked) >= 2)]


def listings(atoms, word, asked):
    """Spans of a close group of stand-alone letters that shows 5 letters other than the asked ones, all of the word's, or at least 3 in a list about as long as the word: any order and any letters, so a tally or a cipher counts too."""
    others = set(word) - asked
    long = min(len(word) - 1, 8)
    groups, spans = [], []
    for atom in atoms:
        if groups and atom[1] - groups[-1][-1][2] <= GAP:
            groups[-1].append(atom)
        else:
            groups.append([atom])
    for group in groups:
        seen = {a[0] for a in group} - asked
        if (len(seen) >= 5 and len(group) >= max(6, min(len(word), 8))) or (len(seen) >= 3 and (seen >= others or len(group) >= long)):
            spans += [list(a[1:]) for a in group]
    return spans


def explained_letters(atoms, tokens, text, word, asked):
    """A walk through letters or short pieces, allowing explanatory words between items."""
    extra = []
    for token_index, (token, start, end) in enumerate(tokens):
        raw = text[start:end]
        linked = any(0 <= j < len(tokens) and tokens[j][0] in ('then', 'and', 'next') for j in (token_index - 1, token_index + 1))
        if len(token) > 1 and token not in ('then', 'and', 'next') and linked:
            extra.append((token[0], start, start + 1))
        caps = [(c.lower(), start + i, start + i + 1) for i, c in enumerate(raw) if c.isascii() and c.isupper()]
        if len(caps) == 1 and len(token) >= 3 and raw[0].islower() and raw[-1].islower():
            extra.extend(caps)
        elif len(token) == 3 and token[0] == token[-1]:
            extra.append((token[1], start + 1, start + 2))
    stream = sorted(set(atoms + extra), key=lambda a: a[1])
    for target in (word, word[::-1]):
        states = {0: []}
        for c, start, end in stream:
            for at, path in list(states.items()):
                if at < len(target) and target[at] == c and at + 1 not in states:
                    states[at + 1] = path + [[start, end]]
        size = max(states)
        path = states[size]
        if size >= min(8, max(5, len(word) - 1)) and len(set(target[:size]) - asked) >= 2:
            return path
        states = {0: []}
        for part, start, end in tokens:
            options = [part]
            if len(part) >= 3 and part[0] == part[-1]:
                options.append(part[1:-1])
            if len(part) >= 4 and len(part) % 2 == 0 and all(part[i] == part[i + 1] for i in range(0, len(part), 2)):
                options.append(part[::2])
            for piece in options:
                if not 1 <= len(piece) <= 3 or piece == word:
                    continue
                for at, path in list(states.items()):
                    for begin in range(max(0, at - len(piece) + 1), at + 1):
                        finish = begin + len(piece)
                        if finish > at and target.startswith(piece, begin) and finish not in states:
                            states[finish] = path + [[start, end]]
        for size in sorted(states, reverse=True):
            if size >= min(8, max(6, len(word) - 1)) and len(states[size]) >= 3:
                return states[size]
    # Spaced foreign glyphs are letter-level codes, not prose words in that language.
    glyphs = [(m.group(), m.span()) for m in re.finditer(r'[^\W\d_]+', text) if len(m.group()) == 1]
    if len(glyphs) >= max(6, min(len(word), 10)) and sum(not c.isascii() for c, _ in glyphs) >= 2:
        return [list(span) for _, span in glyphs]
    return []


def split_copies(tokens, word, whole):
    """Spans of the word written in at least 4 pieces close together, or all of it in 3 of at most 3 letters or with 2 single letters: st-ra-wb-er-ry, st**r**awbe**rr**y, won-der-ful, (s)trawberrie(s)."""
    backwards = word[::-1]
    spans, i = [], 0
    while i < len(tokens):
        joined, j = '', i
        while j < len(tokens) and (j == i or tokens[j][1] - tokens[j - 1][2] <= 8) and ((joined + tokens[j][0]) in word or (joined + tokens[j][0]) in backwards):
            joined += tokens[j][0]
            j += 1
        sizes = [len(t[0]) for t in tokens[i:j]]
        broken = any(t[0] not in whole for t in tokens[i:j])
        if broken and ((j - i >= 4 and len(joined) >= 6) or (j - i == 3 and len(joined) == len(word) and (max(sizes) <= 3 or sizes.count(1) >= 2))):
            spans += [list(t[1:]) for t in tokens[i:j]]
            i = j
        else:
            i += 1
    return spans


def marked_letters(folded, text, word):
    """Spans of letters set off inside a word by markup, if there are at least 2 of them, or of a copy of the word in mixed case."""
    found, marks = list(MARKED.finditer(folded)), []
    for i, m in enumerate(found):
        opener, mark = m.groups()
        head = m.start() > 0 and ('a' <= folded[m.start() - 1] <= 'z' or (i > 0 and found[i - 1].end() == m.start()))
        tail = TOKEN.match(folded, m.end()) or (i + 1 < len(found) and found[i + 1].start() == m.end())
        plural = (not head and tail and folded[m.end()] == 's' and not folded[m.end() + 1:m.end() + 2].isalpha()) or (opener == '(' and mark in ('s', 'es'))
        if (head or tail) and not plural and (opener not in '"\'{<' or (head and tail)):
            marks.append(list(m.span()))
    spans = marks if len(marks) >= 2 else []
    for m in TOKEN.finditer(folded):
        if m[0] == word and text[m.start():m.end()] not in (word, word.upper(), word.capitalize()):
            spans.append(list(m.span()))
    return spans


def position_mentions(folded):
    """Spans of numbered places in the word, if at least 2 places are named: position 3, the 7th letter, the first and seventh positions."""
    seen, spans = set(), []
    for m in NUMBERED_SLOTS.finditer(folded):
        context = folded[max(0, m.start() - 25):m.end() + 45]
        if re.search(r'\b(?:of|in|for) (?:the )?answer\b|\bcounts?\s*\[|\balphabet\b', context):
            continue
        if re.match(r'\b(?:letter|character)\s+[123]\s*:\s*["\']?[a-z]["\']?\s+(?:appears?|occurs?)\b', folded[m.start():]):
            continue
        seen |= set(re.findall(NUMBER, m[1]))
        spans.append(list(m.span()))
    for m in ORDINAL_LISTS.finditer(folded):
        context = folded[max(0, m.start() - 25):m.end() + 45]
        if re.search(r'\b(?:of|in|for) (?:the )?answer\b|\bcounts?\s*\[|\balphabet\b', context):
            continue
        if re.match(r'\b(?:letter|character)\s+[123]\s*:\s*["\']?[a-z]["\']?\s+(?:appears?|occurs?)\b', folded[m.start():]):
            continue
        places = set(re.findall(ORDINAL, m[1]))
        if re.fullmatch(SLOT, m[2]) or places - AMBIGUOUS:
            seen |= places
            spans.append(list(m.span()))
    for m in NUMBERED_CELLS.finditer(folded):
        context = folded[max(0, m.start() - 25):m.end() + 45]
        if re.search(r'\b(?:of|in|for) (?:the )?answer\b|\bcounts?\s*\[|\balphabet\b', context):
            continue
        if re.match(r'\b(?:letter|character)\s+[123]\s*:\s*["\']?[a-z]["\']?\s+(?:appears?|occurs?)\b', folded[m.start():]):
            continue
        seen.add(next(filter(None, m.groups())))
        spans.append(list(m.span()))
    for pattern in (r'(?<![a-z])([a-z])\s+(?:occurs?\s+)?at\s+(\d+)\b',
                    r'(?<!\w)#(\d+)\s*(?:is|=|:)\s*[\"\'*]*([a-z])(?![a-z])',
                    r'\(\s*(\d+)\s*,\s*[\"\']?([a-z])[\"\']?\s*\)',
                    rf'(?<![a-z])([a-z])\s+comes\s+({"|".join(ORDINALS)})\b'):
        matches = [m for m in re.finditer(pattern, folded) if not re.search(r'\b(?:of|in|for) (?:the )?answer\b|\balphabet\b', folded[max(0, m.start() - 25):m.end() + 45])]
        if len(matches) >= 2:
            spans += [list(m.span()) for m in matches]
            seen |= {m.group() for m in matches}
    return sorted(spans) if len(seen) >= 2 else []


def letter_positions(folded):
    """Explicit position lists, without mistaking output counts for source indices."""
    patterns = (LETTER_PLACES,
        re.compile(r'(?<![a-z])[a-z][\"\'*]*\s*(?::|\()\s*[\[(]?\s*#?(\d+(?:st|nd|rd|th)?(?:\s*(?:,|/|&|and|\s)\s*#?\d+(?:st|nd|rd|th)?)+)'),
        re.compile(r'(?<![a-z])[a-z]\s+is\s+number\s+(\d+(?:\s+and\s+number\s+\d+)+)'))
    spans = []
    for pattern in patterns:
        for m in pattern.finditer(folded):
            numbers = [int(x) for x in re.findall(r'\d{1,4}', m[1])]
            context = folded[max(0, m.start() - 25):m.end() + 30]
            if re.search(r'\b(?:of|in|for) (?:the )?answer\b|\bcounts?\s*\[|\balphabet\b', context):
                continue
            if not numbers or min(numbers) <= 0 or (max(numbers) <= 2 and not re.search(SLOT, context)):
                continue
            if numbers == sorted(set(numbers)) and not re.search(r'(?<![a-z])[a-z]["\'*]*\s*(?:,|and)\s*$', folded[max(0, m.start() - 60):m.start()]):
                spans.append(list(m.span()))
    before = list(re.finditer(r'(?<![a-z])[a-z]\s+has\s+\d+\s+letters?\s+before\s+it\b', folded))
    return spans + [list(m.span()) for m in before] if len(before) >= 2 else spans


def part_tallies(folded, tokens, word, whole):
    """Spans of pieces of the word worked through one by one: 3 set off by quotes or given a count, a set-off run that adds up to the word, or 4 of at least 4 letters anywhere. In a line of several words, its own words are pieces if set off."""
    found = [(m[1] or m[2], m.span()) for m in SET_OFF.finditer(folded) if (m[1] or m[2]) in word and (len(whole) > 1 or (m[1] or m[2]) != word)]
    set_off = {part: list(span) for part, span in reversed(found)}
    if sum(len(part) >= 2 for part in set_off) >= 3:
        return sorted(set_off.values())
    for i in range(len(found)):
        for size in (3, 4, 5):
            if ''.join(part for part, _ in found[i:i + size]) == word:
                return [list(span) for _, span in found[i:i + size]]
    bare = {}
    for part, start, end in tokens:
        if len(part) >= 4 and part not in whole and part in word:
            bare.setdefault(part, [start, end])
    return sorted(bare.values()) if len(bare) >= 4 else []


def digit_walks(folded, word):
    """Spans of a run of small numbers or letter codes as long as the word: 0 0 1 0 0, running counts, letters as numbers or bytes, Morse."""
    size = min(10, max(6, len(word) - 1))
    spans = [list(m.span()) for m in WALK.finditer(folded) if len(re.findall(r'\d+', m[0])) >= size]
    spans += [list(m.span()) for m in re.finditer(r'(?<!\d)[01]{6,}(?!\d)', folded) if len(m[0]) >= size]
    codewords = r'(?:yes|no|hit|miss|zero|one|two|three|four|five|six|seven|eight|nine|ten)'
    for m in re.finditer(rf'\b{codewords}(?:[\s,;/|.:-]+(?:(?:then|and|next)\s+)?{codewords}){{5,}}\b', folded):
        if len(re.findall(rf'\b{codewords}\b', m.group())) >= size:
            spans.append(list(m.span()))
    return spans + [list(m.span()) for m in CODE_WALK.finditer(folded) if len(re.split(r'[\s/]+', m[0].strip())) >= size and len(set(re.split(r'[\s/]+', m[0].strip()))) >= 2]


def masks(folded, word):
    """A masked copy must change the source, retain a real letter and mark at least three places."""
    spans = []
    for mark in PLACEHOLDERS + 'abcdefghijklmnopqrstuvwxyz09':
        if folded.count(mark) < 3:
            continue
        for m in re.finditer(r'\s?'.join(f'(?:{c}|{re.escape(mark)})' for c in word), folded):
            shown = re.sub(r'\s', '', m.group())
            if 3 <= shown.count(mark) < len(word) and shown != word:
                spans.append(list(m.span()))
    mixed = '_-*.·•#?□■☐○●~+=…–—,/09'
    pattern = r'\s?'.join(f'(?:{c}|[{re.escape(mixed)}])' for c in word)
    for m in re.finditer(pattern, folded):
        shown = re.sub(r'\s', '', m.group())
        changes = sum(a != b for a, b in zip(shown, word))
        if changes >= 3 and any(c.isalpha() for c in shown):
            spans.append(list(m.span()))
    return spans


def grouped(part):
    """True if every letter of a group of letters sits next to its own kind, and at least one is doubled."""
    return len(set(part)) < len(part) and len(re.findall(r'(.)\1*', part)) == len(set(part))


def disguised(folded, tokens, word):
    """Spans of the letters hidden in other shapes: first or last letters of words in a row, digits and signs for letters, doubled or padded letters, or the word's letters sorted."""
    singles = list(SINGLE_MARK.finditer(folded))
    spans = [list(singles[k].span()) for k in run_hits(''.join(LEET.get(m[0], m[0]) for m in singles), word, 6)]
    for pick in (lambda part: part[0], lambda part: part[-1]):
        spans += [list(tokens[k][1:]) for k in run_hits(''.join(pick(t[0]) for t in tokens), word, 6)]
    for part, start, end in tokens:
        if len(part) >= 11 and part not in word and any(run_hits(part[offset::step], word, 6) for step in (2, 3, 4, 5, 6) for offset in range(step)):
            spans.append([start, end])
        if part not in (word, word[::-1]) and sorted(part) == sorted(word) and grouped(part):
            spans.append([start, end])
    return spans


def reading(text):
    """What the checker reads: the text up to the end of the last box, since what comes after that can't have helped. A box that only repeats the prompt's #, #, # isn't an answer."""
    span = boxed_span(text)
    return text[:span[1]] if span and '#' not in text[span[0]:span[1]] else text


def spelled_spans(text_given, asked, text):
    """Spans where the response breaks the given word or line down into its letters, in whatever notation; empty if it nowhere does."""
    whole = text_given.split()
    word = ''.join(whole)
    folded = fold(text)
    escapes = []
    def decode(match):
        c = chr(int(next(x for x in match.groups() if x is not None), 16))
        if c.isascii() and c.isalpha():
            escapes.append(match.span())
            return c.lower() + ' ' * (len(match.group()) - 1)
        return match.group()
    folded = re.sub(r'(?:\\u00([0-9a-f]{2})|\\x([0-9a-f]{2})|0x([0-9a-f]{2})(?![0-9a-f])|u\+00([0-9a-f]{2}))', decode, folded)
    for tag in ('span', 'b', 'i', 'em', 'strong', 'code', 'small', 'sup', 'sub'):
        if re.search(rf'</{tag}\s*>', folded):
            folded = re.sub(rf'</?{tag}(?:\s+[^<>]*)?\s*>', lambda m: ' ' * len(m.group()), folded)
    folded = re.sub(r'(?<![a-z.])(?:i\.e\.|e\.g\.)(?=\s+(?:an?\s+)?[a-z]{2,}\b)', lambda m: ' ' * len(m.group()), folded)
    tokens = [(m[0], m.start(), m.end()) for m in TOKEN.finditer(folded)]
    atoms = stand_alone(folded, tokens)
    asked = set(asked)
    detectors = (
        lambda: letter_runs(atoms, word, asked),
        lambda: explained_letters(atoms, tokens, text, word, asked),
        lambda: listings(atoms, word, asked),
        lambda: split_copies(tokens, word, whole),
        lambda: marked_letters(folded, text, word),
        lambda: position_mentions(folded),
        lambda: letter_positions(folded),
        lambda: part_tallies(folded, tokens, word, whole),
        lambda: digit_walks(folded, word),
        lambda: masks(folded, word),
        lambda: disguised(folded, tokens, word))
    for detector in detectors:
        spans = detector()
        if spans:
            return [[a, max([b] + [end for start, end in escapes if a <= start < b])] for a, b in spans]
    return []


def family_of(word):
    return root(word)[:8]


@lru_cache(maxsize=None)
def family_words():
    """One word per family, in dictionary order: the longest, so that the long levels get as many as they can."""
    best = {}
    for word in enable_words():
        if len(word) >= LENGTHS[1][0]:
            family = family_of(word)
            if len(word) > len(best.get(family, '')):
                best[family] = word
    return sorted(best.values())


def pick_letters(word, rng):
    """Three letters to ask for, in random order, with spread-out counts: each takes a count class at random (none, at most one absent letter; 1 or 2), then a letter of that class.
    None if the word has fewer than 2 repeated letters, or would have fewer than 3 letters left besides the asked ones."""
    counts = Counter(word)
    if sum(n >= 2 for n in counts.values()) < 2:
        return None
    classes = {}
    for letter, n in counts.items():
        if n <= 2:
            classes.setdefault(n, []).append(letter)
    absent = [c for c in COMMON if c not in counts]
    if absent:
        classes[0] = absent
    letters = []
    for _ in range(3):
        if not classes:
            return None
        kind = rng.choice(sorted(classes))
        letter = rng.choice(classes[kind])
        letters.append(letter)
        classes[kind].remove(letter)
        if kind == 0 or not classes[kind]:
            del classes[kind]
    rng.shuffle(letters)
    return letters if len(set(word) - set(letters)) >= 3 else None


def wrong_counts(counts, rng):
    """The counts with one of them off by one."""
    out = list(counts)
    i = rng.randrange(len(out))
    out[i] = out[i] + 1 if out[i] == 0 or rng.random() < .5 else out[i] - 1
    return out


@lru_cache(maxsize=None)
def town_names():
    """Town names of 7 to 14 plain letters that aren't dictionary words, so far shares no word with the tasks."""
    return sorted({row['name'] for row in cities() if re.fullmatch('[A-Za-z]{7,14}', row['name']) and row['name'].lower() not in dictionary()})


def boxed_counts(text):
    """The whole numbers in the last box, as strings without leading zeros, or None if there is no box or it holds a minus sign or a decimal point. A hyphen between numbers only separates them."""
    box = boxed(text)
    if box is None:
        return None
    box = digits_to_ascii(clean(box))
    if re.search(r'(?<![\d\s])\s*-\s*\d|^\s*-|\d\.\d', box):
        return None
    return [x.lstrip('0') or '0' for x in re.findall(r'\d+', box)]


class LetterCounting(Env):
    name = NAME
    seed_name = '08_letter_counting'
    contrast = ('First write out the word letter by letter, separating the letters with hyphens. Finish with \\boxed{#, #, #}.',
                'Keep the word whole: do not write its letters separately or list their positions. Finish with \\boxed{#, #, #}.')

    def space(self, level):
        lo, hi = LENGTHS[level]
        return [word for word in family_words() if lo <= len(word) <= hi]

    def build(self, word, rng, level):
        letters = pick_letters(word, rng)
        if letters is None:
            return None
        return dict(key=word, group=family_of(word), prompt=PROMPT.format(*letters, word), answer=[word.count(c) for c in letters],
                    meta=dict(word=word, letters=letters))

    def reward(self, task, text):
        return float(boxed_counts(text) == [str(n) for n in task['answer']])

    def check(self, task, text):
        meta = task['meta']
        spans = spelled_spans(meta['word'], meta['letters'], reading(text))
        return dict(value=bool(spans), spans=spans)

    def _pair(self, rng, name, closing, end):
        """A pair of replies alike but for the first step: writing the name out, or as many plain words on the same topic instead. Layouts are drawn for each reply, so lines don't give the class away."""
        form, topic = rng.choice(FORMS), rng.choice(TOPICS)
        letters = sum(c.isalpha() for c in name)
        layout = 'Reply with no blank lines.' if form in ROOMY_FORMS else rng.choice(LAYOUTS)
        yes = f'{layout} First write out "{name}" letter by letter, {form}. Next write a sentence of about 8 words saying {topic}. Then {closing} {end}'
        no = (f'{rng.choice(LAYOUTS)} First write a sentence of about {8 + 2 * letters // 3} words saying {topic}. Then {closing} '
              f'Keep "{name}" whole: do not write its letters separately or list their positions. {end}')
        return [dict(label=True, instruction=yes.strip(), form=form), dict(label=False, instruction=no.strip())]

    def near_variants(self, task, rng):
        n = list(task['answer'])
        final = n if rng.random() < .5 else wrong_counts(n, rng)
        pair = self._pair(rng, task['meta']['word'], f'say that the three letters appear {final[0]}, {final[1]} and {final[2]} times.', f'Finish with \\boxed{{{final[0]}, {final[1]}, {final[2]}}}.')
        return [dict(variant, final=final) for variant in pair]

    def far_draw(self, rng):
        kind = rng.choice(FAR_KINDS)
        if kind == 'vowels':
            line = rng.choice(verse_lines())
            word, key, letters = line, ' '.join(line.split()[:4]), list('aeiou')
            answer = str(sum(c in letters for c in word))
            wrong = [str(int(answer) + d) for d in (-1, 1, 2)]
            prompt, name = f'How many vowels are in this line of verse: "{line}"? Give the final answer as \\boxed{{n}}.', line
        else:
            name = rng.choice(town_names())
            word, key = name.lower(), name[:7].lower()
            if kind == 'contains':
                letter = rng.choice(sorted(set(word))) if rng.random() < .5 else rng.choice(sorted(set(COMMON) - set(word)))
                letters, answer, wrong = [letter], 'yes' if letter in word else 'no', ['no' if letter in word else 'yes']
                prompt = f'Does the name "{name}" contain the letter "{letter}"? Give the final answer as \\boxed{{yes or no}}.'
            elif kind == 'after':
                i = word.index(word[rng.randrange(len(word) - 1)])
                letters, answer, wrong = [word[i]], word[i + 1], sorted(set(word) - {word[i + 1]})
                prompt = f'Which letter comes right after the first "{word[i]}" in the name "{name}"? Give the final answer as \\boxed{{letter}}.'
            else:
                letters, answer, wrong = [], None, []
                prompt = f'A friend is sending a parcel to {name} and asks you how the name is written. Reply in two or three sentences.'
        if answer is None:
            variants = self._pair(rng, name, 'tell the friend that this is how to write it.', '')
        else:
            final = answer if rng.random() < .5 else rng.choice(wrong)
            variants = [dict(v, final=final) for v in self._pair(rng, name, 'say what the answer is.', f'Finish with \\boxed{{{final}}}.')]
        return dict(key=key, prompt=prompt, answer=answer, meta=dict(kind=kind, word=word, letters=letters), variants=variants)

    def probe_keep(self, request, variant, text):
        """True if a response is what its variant asked for: the checker agrees with the label, and it ends with the final answer it was given."""
        if self.check(request, text)['value'] != variant['label']:
            return False
        final = variant.get('final')
        if final is None:
            return True
        return boxed_counts(text) == [str(n) for n in final] if isinstance(final, list) else same_answer(boxed(text), final)

    def set_keep(self, request, texts):
        """True if the pair looks alike apart from the behavior: similar length, digits and lines, except where the form itself has more lines or digits."""
        roomy = request['variants'][0]['form'] in ROOMY_FORMS
        return matched(texts, digits=99 if roomy else 2, lines=99 if roomy else 1)


ENV = LetterCounting()
