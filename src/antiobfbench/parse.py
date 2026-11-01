"""Response parsing shared by rewards and checkers. Parsing is strict: a missing box or tag gives None."""
import math
import re
import unicodedata


def digits_to_ascii(text):
    """Replace non-ASCII decimal digits (Thai, Bengali, Telugu, ...) with ASCII ones."""
    return ''.join(str(unicodedata.digit(c)) if not c.isascii() and c.isdecimal() else c for c in text)


def boxed_span(text):
    """(start, end) of the last complete \\boxed{...} including the wrapper, or None."""
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    pos, depth = start + 7, 1
    while pos < len(text) and depth:
        depth += (text[pos] == '{') - (text[pos] == '}')
        pos += 1
    return (start, pos) if depth == 0 else None


def boxed(text):
    """Content of the last complete \\boxed{...}, or None."""
    span = boxed_span(text)
    return text[span[0] + 7:span[1] - 1].strip() if span else None


def tag(text, name='answer'):
    """Content of the last <name>...</name>, or None."""
    found = re.findall(rf'<{name}>(.*?)</{name}>', text, flags=re.S)
    return found[-1].strip() if found else None


def tag_span(text, name='answer'):
    """(start, end) of the content of the last <name>...</name>, or None."""
    found = list(re.finditer(rf'<{name}>(.*?)</{name}>', text, flags=re.S))
    return found[-1].span(1) if found else None


def pre(text):
    """The text before the last box or answer tag, or all of it if there is neither."""
    cut = max(text.rfind('\\boxed{'), text.rfind('<answer>'))
    return text[:cut] if cut >= 0 else text


def code_block(text, defines=None):
    """Content of the last fenced Python code block, or None. With `defines`, the last block defining that function, if any.

    Models often add a block of example usage after the solution, and that block must not hide the solution.
    """
    found = re.findall(r'```(?:python|py)?[ \t]*\n(.*?)```', text, flags=re.S)
    if defines:
        for block in reversed(found):
            if re.search(rf'\bdef\s+{re.escape(defines)}\b', block):
                return block
    return found[-1] if found else None


def clean(text):
    """Lower-case a boxed answer and strip LaTeX wrappers and outer punctuation."""
    text = re.sub(r'\\(?:text|mathrm|textbf|mathbf)\{([^}]*)\}', r'\1', text or '')
    return text.replace('\\', '').replace('$', '').strip().strip('.').strip().lower()


UNITS = {'million': 1e6, 'billion': 1e9, 'thousand': 1e3}


def num(text):
    """The first number in a boxed answer as a float, or None. Handles separators, native digits and 'million'."""
    if text is None:
        return None
    text = digits_to_ascii(clean(text)).replace(',', '').replace(' ', '').replace('{', '').replace('}', '')
    if re.fullmatch(r'10\^\d{1,3}', text):
        text = f'1times{text}'
    found = re.search(r'(-?\d+(?:\.\d+)?)(?:(?:times|×|x|\*)10\^(\d+))?(million|billion|thousand)?', text)
    if not found:
        return None
    exponent = found.group(2) or '0'
    value = float(found.group(1)) * (float(f'1e{exponent}') if len(exponent) <= 3 else math.inf) * UNITS.get(found.group(3), 1)
    return value if math.isfinite(value) else None


def same_answer(a, b):
    """True if two final answers match after clean-up; plain numbers compare by value, so '3 or 4' doesn't match '3'."""
    if a is None or b is None:
        return False
    a, b = (re.sub(r'(?<=\d),(?=\d)', '', clean(str(x))) for x in (a, b))
    plain = re.compile(r'-?[\d\s]*\.?\d+')
    return a == b or bool(plain.fullmatch(a) and plain.fullmatch(b) and num(a) == num(b))


def ints(text):
    """All non-negative integers in the text, reading thousands separators and native digits. Skips absurdly long ones."""
    text = digits_to_ascii(text).replace('\\,', '').replace('{,}', ',')
    found = re.findall(r'(?<![\d.,])\d{1,3}(?:,\d{3})+(?!\d)|(?<![\d.,])\d+(?!\d)', text)
    return [int(x.replace(',', '')) for x in found if len(x) <= 100]


def has_num(text, value):
    """True if the number occurs in the text on digit boundaries, ignoring thousands separators."""
    return re.search(rf'(?<![\d.]){value}(?!\d)', text.replace(',', '')) is not None


ONES = 'one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()


def number_words(n):
    """A regex for 1 <= n <= 999 written in English words."""
    if n < 20:
        return ONES[n - 1]
    if n < 100:
        return TENS[n // 10 - 2] + (f'[- ]{ONES[n % 10 - 1]}' if n % 10 else '')
    return f'{ONES[n // 100 - 1]} hundred' + (f'(?: and)? {number_words(n % 100)}' if n % 100 else '')


SCALES = 'hundred|thousand|million|billion'
WORDS_BEFORE = re.compile(rf"\b(?:{'|'.join(TENS)})[- ]$|\b(?:{SCALES})(?:,? and)?[- ]$", re.I)
WORDS_AFTER = re.compile(rf"[- ](?:{'|'.join(ONES[:9])}|{SCALES})\b", re.I)
WORDS_AFTER_HUNDRED = re.compile(rf"(?:,? and)?[- ](?:{'|'.join(ONES + TENS)})\b", re.I)


def whole_number(found):
    """False if a number in words is only part of a longer one, as four is in "twenty four" and "four hundred"."""
    text, start, end = found.string, found.start(), found.end()
    return not (WORDS_BEFORE.search(text[max(0, start - 20):start]) or WORDS_AFTER.match(text, end)
                or (found.group().lower().endswith('hundred') and WORDS_AFTER_HUNDRED.match(text, end)))


def find_number(text, value):
    """The first mention of a non-negative integer, or None: in digits on digit boundaries, or in words from 2 to 999.

    One is left out because the word is too common to mean the number.
    """
    digits = str(value)
    head = len(digits) % 3 or 3
    groups = [digits[:head]] + [digits[i:i + 3] for i in range(head, len(digits), 3)]
    found = [re.search(rf"(?<![\d.])(?<!\d,){',?'.join(groups)}(?!,?\d)", text)]
    if 2 <= value <= 999:
        found.append(next(filter(whole_number, re.finditer(rf'(?<![\w-]){number_words(value)}(?!\w)', text, re.I)), None))
    return min(filter(None, found), key=lambda m: m.start(), default=None)


def has_number(text, value):
    return find_number(text, value) is not None


def visible(text):
    """The text as a reader sees it: without markup, and without characters that don't show."""
    text = re.sub(r'[*_~`]', '', unicodedata.normalize('NFKC', text))
    return ''.join(c for c in text if unicodedata.category(c) != 'Cf')


def words(text):
    return re.findall(r'\w+', text.lower())


def sentences(text):
    """Split into sentences, but not after an abbreviation such as D.C. or U.S."""
    return [s for s in re.split(r'(?<=[.!?])(?<!\.[A-Z]\.)\s+|\n+', text.strip()) if re.search(r'\w', s)]


def find_spans(text, needles):
    """[start, end] of the first occurrence of each needle that occurs in the text."""
    spans = []
    for needle in needles:
        start = text.find(needle)
        if needle and start >= 0:
            spans.append([start, start + len(needle)])
    return spans
