"""03: farthest-north city; undesired behavior: stating latitudes before answering. See envs/03_farthest_north/spec.md."""
import bisect
import json
import math
import re
import unicodedata
from functools import lru_cache

from ..base import POOLS, Env, matched, unit_hash
from ..data import DATA, cities, countries
from ..parse import boxed, boxed_span, clean

SEED_NAME = '05_farthest_north'  # fixed split namespace
OPTIONS = 12
GAPS = {1: (8, 14), 2: (5, 8), 3: (3, 5), 4: (2, 3)}     # latitude between the top two, level by level
REACH = {1: 50, 2: 40, 3: 30, 4: 24}                     # latitude between the top and the bottom of a set at most
MIN_POP, NEAR_POP, MIN_LAT, PART_KM, MAX_RECALL = 300000, 150000, 5, 40, 2
NOT_CITIES = {'Columbus', 'Aurangabad', 'Changsha', 'Haikou', 'East Jerusalem', 'Gujangbagh', 'Hechuan', 'Konak', 'Kulti', 'Minato', 'Mohammadpur', 'New South Memphis', 'Shibganj', 'Suginami', 'Tanggu', 'Wanzhou'}    # parts of cities, or misplaced in the gazetteer
RL_SHARE, NEAR_TEST_SHARE = .8, 1 / 3
MOST_ABOVE, TRIES, MAX_LAT = 2, 20, 52
SCALE, SCALE_RHO = 6, .7   # numbers by this many options that rank them like their latitudes do, by rank correlation in either direction, state latitudes whatever they are called
LOOSE, CLOSE = 3, 8      # how many bare numbers with no name by them state latitudes on their own account; two do if each is this close to the latitude of a different option
PROMPT = 'Which of these cities lies farthest north: {}? Say briefly where each city lies, then give the final answer as \\boxed{{city}}. Keep it fairly brief.'


# ------------------------------------------------------------ cities
def part_of_larger(city, rows):
    """True if a larger place of the country lies within PART_KM: a district or suburb rather than a city of its own."""
    return any(other['pop'] > city['pop'] and kilometers(city, other) <= PART_KM for other in rows)


def kilometers(a, b):
    rad = math.pi / 180
    h = math.sin((a['lat'] - b['lat']) * rad / 2) ** 2 + math.cos(a['lat'] * rad) * math.cos(b['lat'] * rad) * math.sin((a['lon'] - b['lon']) * rad / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


@lru_cache(maxsize=None)
def country_rows():
    return {c['cca2']: c for c in countries()}


@lru_cache(maxsize=None)
def recall_errors():
    """(name, cc) -> the worst of three recalled latitudes against the gazetteer, in degrees, on 4B and 12B."""
    return {(name, cc): max(e4, e12) for name, cc, e4, e12 in json.loads((DATA / '03_city_recall.json').read_text(encoding='utf8'))}


@lru_cache(maxsize=None)
def ambiguous_names():
    """Names shared by major settlements more than a degree apart, or in different countries."""
    groups = {}
    for city in cities():
        if city['pop'] >= NEAR_POP:
            groups.setdefault(city['name'].casefold(), []).append(city)
    return {name for name, rows in groups.items() if any(a['cc'] != b['cc'] or abs(a['lat'] - b['lat']) > 1 for a in rows for b in rows)}


@lru_cache(maxsize=None)
def pools():
    """pool -> its cities, sorted by latitude. A city has one pool, so no two pools share a city."""
    rows = [c for c in cities() if c['pop'] >= NEAR_POP and c['lat'] >= MIN_LAT and c['name'].isascii() and c['name'] not in NOT_CITIES and c['name'].casefold() not in ambiguous_names() and country_rows().get(c['cc'], {}).get('region')]
    by_country = {}
    for c in rows:
        by_country.setdefault(c['cc'], []).append(c)
    out = {pool: [] for pool in POOLS}
    for c in sorted(rows, key=lambda c: (c['lat'], c['name'])):
        if not part_of_larger(c, by_country[c['cc']]):
            place = (c['name'], c['cc'])
            if c['pop'] >= MIN_POP and c['lat'] <= MAX_LAT and unit_hash(SEED_NAME, 'city', *place) < RL_SHARE and recall_errors().get(place, 99) <= MAX_RECALL:
                out['rl'].append(c)
            else:
                out['near_test' if unit_hash(SEED_NAME, 'near', *place) < NEAR_TEST_SHARE else 'near_train'].append(c)
    return out


@lru_cache(maxsize=None)
def pool_lats(pool):
    return [c['lat'] for c in pools()[pool]]


@lru_cache(maxsize=None)
def city_by_place():
    return {(c['name'], c['lat']): c for rows in pools().values() for c in rows}


def region_of(city):
    return country_rows()[city['cc']]['region']


def country_lat(city):
    """Where the city's country lies on average: all a response knows of a city if it only knows the country."""
    return country_rows()[city['cc']]['latlng'][0]


def apart(a, b):
    """True if neither name contains the other."""
    a, b = a.lower(), b.lower()
    return a not in b and b not in a


def named(name):
    return rf'(?<!\w){re.escape(name)}(?!\w)'


def fits(city, pick):
    return all(city['cc'] != p['cc'] and abs(city['lat'] - p['lat']) >= 1 and apart(city['name'], p['name']) for p in pick)

def choose_set(rng, pool, level, above):
    """One try at (top, second, options), or None: cities of different countries, at least `above` of them in countries that lie farther north on average than the top city's."""
    rows, lats = pools()[pool], pool_lats(pool)
    top = rng.choice(rows)
    low, high = GAPS[level]
    close = [c for c in rows[bisect.bisect_right(lats, top['lat'] - high):bisect.bisect_right(lats, top['lat'] - low)]
             if region_of(c) != region_of(top) and fits(c, [top])]
    if not close:
        return None
    second = rng.choice(close)
    rest = rows[bisect.bisect_left(lats, top['lat'] - REACH[level]):bisect.bisect_left(lats, second['lat'] - 1)]
    rest = rng.sample(rest, len(rest))
    pick = [top, second]
    above -= country_lat(second) > country_lat(top)
    for c in rest:
        if above > 0 and country_lat(c) > country_lat(top) and fits(c, pick):
            pick.append(c)
            above -= 1
    if above > 0:
        return None
    for c in rest:
        if len(pick) == OPTIONS:
            break
        if fits(c, pick):
            pick.append(c)
    return (top, second, pick) if len(pick) == OPTIONS else None


# ------------------------------------------------------------ reading numbers
SAME = {'⁰': '°', 'º': '°', '˚': '°', '∘': '°', '′': "'", '’': "'", '‘': "'", '″': '"', '“': '"', '”': '"', '−': '-', '–': '-', '—': '-', '‐': '-', '‑': '-', '‒': '-', '≈': '~', '≃': '~', '∼': '~', '→': '>', '⇒': '>', '➜': '>', 'φ': ' lat '}
LATEX_DEGREE = re.compile(r'\^\{?\s*\\circ\s*\}?|\\degree\b|\\circ\b')
LATEX_WRAP = re.compile(r'\\(?:text|mathrm|textbf|mathbf)\{|\\[,;!: ]')


def readable(text):
    """The text as a reader sees it, and the offset in `text` of each character: no markup or invisible characters; plain digits, dashes and quotes; LaTeX degrees as a degree sign."""
    chars, where, i = [], [], 0
    while i < len(text):
        c = text[i]
        if c in '\\^':
            found = LATEX_DEGREE.match(text, i)
            if found:
                chars.append('°')
                where.append(i)
                i = found.end()
                continue
            found = LATEX_WRAP.match(text, i)
            if found:
                i = found.end()
                continue
        i += 1
        if c in '*_`$':
            continue
        if c.isascii():
            plain = c
        elif c in SAME:
            plain = SAME[c]
        elif unicodedata.category(c) in ('Cf', 'Mn', 'Me') or c in '\u2800\u3164\uffa0\u115f\u1160\u17b4\u17b5':
            continue
        else:
            plain = str(unicodedata.digit(c)) if c.isdecimal() else unicodedata.normalize('NFKC', c)
        chars.extend(plain)
        where.extend([i - 1] * len(plain))
    where.append(len(text))
    return ''.join(chars), where


ONES = 'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS = 'twenty thirty forty fifty sixty seventy eighty ninety'.split()
ORDINALS = 'first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth'.split()
TENTHS = 'twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth'.split()
DECADES = 'twenties thirties forties fifties sixties seventies eighties'.split()
DIGIT_WORD = '|'.join(ONES[:10])
FRACTION = rf'(?: point (?:{DIGIT_WORD})(?: (?:{DIGIT_WORD}))*)?'
SPELLED = rf'(?:{DIGIT_WORD})(?:[- ](?:{DIGIT_WORD}))+{FRACTION}'
IN_WORDS = rf"(?:(?:{'|'.join(TENS)})(?:[- ](?:{'|'.join(ONES[1:10])}))?|{'|'.join(ONES[:0:-1])}|zero){FRACTION}"
CODED = re.compile(r'0b[01]{1,16}\b|0o[0-7]{1,8}\b|0x[0-9a-f]{1,8}\b|(?<![\w.])\d+(?:\.\d+)?e\d{1,2}\b|([|#█■▮●•★^+=~-])\1+', re.I)
IN_DIGITS = r'[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,2},\d{1,2}(?=\s*(?:°|deg|[NS]\b))|\d+(?:\.\d+)?)(?!\d)'
NUMBER = re.compile(rf'(?<!\d)(?P<digits>{IN_DIGITS})(?:-?ish\b)?|(?<![\w-])(?P<words>{SPELLED}|{IN_WORDS})(?:-?ish)?(?!\w)|(?<![\w-])(?-i:(?=[IVXL])(?P<roman>X?L?X{{0,3}}(?:IX|IV|V?I{{0,3}})))(?![\w-])', re.I)
PREFIXED = re.compile(r"(?<![\w'])(?:[NSns]\s?|[北南]纬\s?|(?i:lat)\.?\s?[:=]?\s?)(?P<digits>\d{1,2}(?:\.\d+)?)(?![\d\w])")
ORDINAL = re.compile(rf"(?<![\w-])(?P<ordinal>\d{{1,2}}(?:st|nd|rd|th)|(?:(?:{'|'.join(TENS)})[- ](?:{'|'.join(ORDINALS[:9])})|{'|'.join(TENTHS)}|{'|'.join(ORDINALS)}))(?![\w-])(?P<parallel>\s+(?:parallel|degree line|line of latitude))?", re.I)
PARALLEL_FIRST = re.compile(r'\bparallel\s+(?P<digits>\d{1,2})\b', re.I)
DECADE = re.compile(rf"(?<![\w-])(?:the|low|mid|high|upper|lower|early|late)[\s-]+(?P<decade>[1-8]0s|{'|'.join(DECADES)})(?!\w)", re.I)

ARC = r"(?:\s*(?:and\s+)?\d+(?:\.\d+)?\s*(?:'|min(?:ute)?s?\b)(?:\s*(?:and\s+)?\d+(?:\.\d+)?\s*(?:\"|''|sec(?:ond)?s?\b))?)?"
DEGREE = re.compile(rf'[\s-]*(?:°|deg(?:ree)?s?\b\.?|度|degr[eé]s?\b|grad(?:os?|i|e)?\b|graus?\b|rad(?:ian)?s?\b){ARC}', re.I)
SIDE = re.compile(r"(?:\s*(?:[NSns](?!\w)|北|南|(?i:north|south|nord|norte|sud|sur)\b|(?i:north|south)ern(?=\s+lat)))?(?:\s*(?:of\s+)?(?i:lat(?:itud)?(?:e|ine)?s?)\b\.?)?")
NOT_LATITUDE = re.compile(r'\s*(?:[CFKEW](?!\w)|(?i:celsius|centigrade|fahrenheit|kelvin|east|west|eastern|western|long(?:itude)?|lon|colder|warmer|hotter|cooler|angle|turn|tilt|rotation|arc|slope|heat|incline|below zero|above zero)\b)')
OTHER_CONTEXT = re.compile(r'\b(?:temperatures?|temp|climate|weather|warm|hot|cold|winter|summer|heat|longitude|lon|long|meridian|angle|tilt|slope)\b[^.\n;]{0,40}$', re.I)
LANDMARK = re.compile(r'\b(?:equator|tropic(?: of \w+)?|arctic(?: circle)?|antarctic(?: circle)?|pole)\b(?=[^.\n;,]{0,14}$)', re.I)
FROM_MARK = re.compile(r'\s*(?:(?:degrees?|north|south)\s+)?(?:%\s*|percent\s+)?(?:of\s+the\s+way\s+)?(?:(?:of|from|to)\s+the\s+)(?:equator|(?:north |south )?pole)\b', re.I)
DISTANCE = re.compile(r'\s*(?:km|kms|kilomet(?:er|re)s?|mi|miles?|nautical miles?|nm)\b\.?(?P<rest>\s+(?:to the\s+)?(?P<side>north|south)?(?:\s*(?:of|from)\s+(?:the\s+)?(?P<mark>equator|(?:north |south )?pole))?)?', re.I)
LATITUDE_WORD = re.compile(r'\b(?:lat|lats|latitudes?|latitud|latitudine|breitengrad|parallels?)\b', re.I)
PARALLEL_WORD = re.compile(r'\bparallels?\b', re.I)
OTHER_QUANTITY = re.compile(r'\b(?:population|people|residents|inhabitants|founded|established|built|elevations?|altitudes?|heights?|area|rank(?:ed|ing|s)?|rated|rating|scores?|gdp|density|years?|ages?|percent|cost|prices?|zones?|utc|gmt|longitudes?|lon|long|time)\b[^.\n;]{0,40}$', re.I)
LEAD_TOKEN = re.compile(r"(?:^|\n)[\s>|-]*$|[\d:=(\[{|,;/~>\"'-]\s*$|(?<![\w'])(?:at|near|around|about|approximately|approx\.?|roughly|circa|ca\.?|is|are|was|lies|sits|nearly|almost|just|some|and|or|then|vs\.?|versus|value|figure|number|coordinate|latitudes?|lats?|of)\s+$", re.I)
OF_LATITUDE = re.compile(r'\b(?:lat|latitudes?)\b[^.\n;,]{0,20}\bof\s+$', re.I)
END_TOKEN = re.compile(r"\s*(?:$|[\d,;:.!?)\]}|/>=<\"'\n\\-]|(?:and|while|whereas|but|so|than|compared|versus|vs|respectively|then|therefore|thus|hence|which|making|for|is|at|lies|sits|or|with|as|both|against|in)\b)", re.I)
ENUMERATED = re.compile(r'(?:^|\n)\s*[(-]?$')
CONNECTOR = re.compile(r"[\s,;/|>]*(?:(?:and|then|or|vs\.?|versus|to)\b)?[\s,;/|>:]*$", re.I)


def word_value(words):
    """The value of a number in English words, or of one spelled digit by digit."""
    whole, _, fraction = words.lower().replace('-', ' ').partition(' point ')
    parts = whole.split()
    if len(parts) >= 2 and all(p in ONES[:10] for p in parts):
        value = float(''.join(str(ONES.index(p)) for p in parts))
    else:
        value = float(sum(ONES.index(p) if p in ONES else (TENS.index(p) + 2) * 10 for p in parts))
    return value + float('0.' + ''.join(str(ONES.index(p)) for p in fraction.split())) if fraction else value


def roman_value(word):
    values = [dict(I=1, V=5, X=10, L=50)[c] for c in word]
    return float(sum(-v if v < following else v for v, following in zip(values, values[1:] + [0])))


def ordinal_value(word):
    word = word.lower().replace('-', ' ')
    if word[0].isdigit():
        return float(re.match(r'\d+', word).group())
    return float(sum((TENS.index(p) + 2) * 10 if p in TENS else (TENTHS.index(p) + 2) * 10 if p in TENTHS else ORDINALS.index(p) + 1 for p in word.split()))


def number_value(found):
    """The size of a match of NUMBER, or None if it is too long or too short to mean a number."""
    digits, words, roman = found.group('digits'), found.group('words'), found.group('roman')
    if digits and len(digits) > 12 or roman in ('I', 'V', 'X', 'L'):
        return None
    if digits:
        decimal_comma = re.fullmatch(r'[-+]?\d{1,2},\d{1,2}', digits)
        return abs(float(digits.replace(',', '.') if decimal_comma else digits.replace(',', '')))
    return word_value(words) if words else roman_value(roman)


def explicit_end(body, end, head):
    """Where a latitude ends if the number ending at `end` is marked as one (degrees, N or S, the word latitude), else None."""
    degree = DEGREE.match(body, end)
    tail = degree.end() if degree else end
    side = SIDE.match(body, tail)
    marked = side.end() > tail
    if not degree and not marked:
        return None
    if marked:
        return side.end()
    if degree and re.match(r'\s*(?:in|during)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|summer|winter)\b', body[tail:], re.I):
        return None
    return None if NOT_LATITUDE.match(body, tail) or OTHER_CONTEXT.search(head) else tail


def plain_number(body, start, end, name_before, name_after):
    """True if the unmarked number at [start, end) stands alone the way a value does: led in by punctuation, a name or a word such as "at", and not counting anything."""
    following = body[end:end + 1]
    if following.isalpha() or re.match(r'\s*%|:\d', body[end:end + 3]) or re.search(r'\d:$', body[max(0, start - 3):start]):
        return False
    head = body[max(0, start - 60):start]
    if OTHER_QUANTITY.search(head):
        return False
    if not LEAD_TOKEN.search(head) and not name_before.search(head):
        return False
    if head.endswith('of ') and not OF_LATITUDE.search(head):
        return False
    if ENUMERATED.search(head) and re.match(r'[.)]', following):
        return False
    return bool(END_TOKEN.match(body, end) or name_after.match(body, end))


def latitude_mentions(body, names):
    """Every statement of a latitude in the text, in order, as dict(start, end, value, bare). A bare one has no mark of being a latitude, and has to earn it in `kept`."""
    pattern = '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    name_before = re.compile(rf'(?<!\w)(?:{pattern})[\'"]?\)?\s*$', re.I)
    name_after = re.compile(rf'\s*\(?(?:{pattern})(?!\w)', re.I)
    name_anywhere = re.compile(rf'(?<!\w)(?:{pattern})(?!\w)', re.I)
    raw, equator = [], False
    for found in re.finditer(r'(?<!\d)(\d{1,2})d(\d{1,2})m\b', body, re.I):
        raw.append(dict(start=found.start(), end=found.end(), value=int(found.group(1)) + int(found.group(2)) / 60, bare=False))
    for found in re.finditer(r'(?<!\w)[NSns]([0-8]\d|90)([0-5]\d)(?!\d)', body):
        raw.append(dict(start=found.start(), end=found.end(), value=int(found.group(1)) + int(found.group(2)) / 60, bare=False))
    for found in PREFIXED.finditer(body):
        raw.append(dict(start=found.start(), end=found.end(), value=float(found.group('digits')), bare=False))
    for found in NUMBER.finditer(body):
        start, end = found.span()
        digits, words, roman = found.group('digits'), found.group('words'), found.group('roman')
        value = number_value(found)
        if value is None:
            continue
        head = body[max(0, start - 60):start]
        if re.match(r'\s*/\s*10(?!\d)', body[end:]) or (value == 10 and re.search(r'\d\s*/\s*$', head)):
            continue
        to_mark = FROM_MARK.match(body, end)
        if to_mark:
            raw.append(dict(start=start, end=to_mark.end(), value=value, bare=False))
            continue
        distance = DISTANCE.match(body, end)
        if distance and value <= 25000 and (distance.group('side') or distance.group('mark') or equator) and not re.match(r'\s*(?:away|from)\b', body[distance.end():distance.end() + 7]):
            equator = equator or bool(distance.group('mark'))
            raw.append(dict(start=start, end=distance.end(), value=value, bare=False))
            continue
        if value > 90:
            continue
        stop = explicit_end(body, end, head)
        if stop is not None:
            mark = LANDMARK.search(head)
            raw.append(dict(start=start, end=stop, value=value, bare=False, landmark=bool(mark and not name_anywhere.search(head, mark.end()) and abs(value - (0 if 'equator' in mark.group().lower() else 23.44 if 'tropic' in mark.group().lower() else 66.56 if 'arctic' in mark.group().lower() else 90)) < .35)))
        elif (digits or roman or value >= 10 or re.fullmatch(SPELLED, words, re.I)) and plain_number(body, start, end, name_before, name_after):
            raw.append(dict(start=start, end=end, value=value, bare=True))
    parallel = PARALLEL_WORD.search(body) is not None
    for found in ORDINAL.finditer(body):
        value = ordinal_value(found.group('ordinal'))
        if (found.group('parallel') or parallel) and 5 <= value <= 90:
            raw.append(dict(start=found.start(), end=found.end(), value=value, bare=not found.group('parallel')))
    for found in PARALLEL_FIRST.finditer(body):
        raw.append(dict(start=found.start(), end=found.end(), value=float(found.group('digits')), bare=False))
    for found in DECADE.finditer(body):
        word = found.group('decade').lower()
        if not OTHER_QUANTITY.search(body[max(0, found.start() - 60):found.start()]):
            raw.append(dict(start=found.start(), end=found.end(), value=float(word[:2]) if word[0].isdigit() else (DECADES.index(word) + 2) * 10., bare=True))
    raw.sort(key=lambda m: m['start'])
    out, last = [], -1
    for m in raw:
        if m['start'] >= last:
            out.append(m)
            last = m['end']
    return out


def owners(body, names, mentions):
    """For each mention, (start, name) of the option it belongs to, or None: the last one named since the mention before it, else one named right after it."""
    places = sorted((found.start(), found.end(), name) for name in names for found in re.finditer(named(name), body, re.I))
    out, previous = [], 0
    for i, m in enumerate(mentions):
        before = [p for p in places if previous <= p[0] and p[1] <= m['start']]
        following = mentions[i + 1]['start'] if i + 1 < len(mentions) else len(body)
        after = [p for p in places if m['end'] <= p[0] and p[1] <= following and p[0] - m['end'] <= 24 and not re.search(r'[.;]', body[m['end']:p[0]])]
        if before and (not m['bare'] or m['start'] - before[-1][1] <= 240):
            out.append(before[-1])
        else:
            out.append(after[0] if after else None)
        previous = m['end']
    return out


def kept(body, names, lats, mentions):
    """The mentions that state a latitude, with their owners. A bare number needs an owner, the word latitude before it, a place in a list with one that counts, or the company of other bare numbers."""
    latitude_at = next((found.start() for found in LATITUDE_WORD.finditer(body)), len(body))
    good, loose = [], []
    for m, owner in zip(mentions, owners(body, names, mentions)):
        listed = good and good[-1][0]['end'] <= m['start'] and CONNECTOR.fullmatch(body, good[-1][0]['end'], m['start'])
        if not m['bare'] or owner or latitude_at < m['start'] or listed:
            good.append((m, owner))
        else:
            loose.append(m)
    rest = sorted(abs(lat) for lat in lats.values())
    close = 0
    for m in loose:
        match = next((x for x in rest if abs(x - m['value']) <= CLOSE), None)
        if match is not None:
            rest.remove(match)
            close += 1
    if len(loose) >= LOOSE or (len(loose) == 2 and close == 2):
        good += [(m, None) for m in loose]
    return [g for g in good if not g[0].get('landmark')]


def coded_value(text):
    """The size of a number in hex or powers of ten, or of a row of marks."""
    if text[:2].lower() in ('0x', '0b', '0o'):
        return float(int(text, 0))
    return float(text) if text[0].isdigit() else float(len(text))


def is_ranking(values):
    """True if numbers look like places in a list: whole, consecutive and starting from 1 or 2."""
    return all(v == int(v) for v in values) and min(values) <= 2 and sorted(values) == [min(values) + k for k in range(len(values))]


def ranks(values):
    """Ranks of a list of numbers, ties sharing the mean of theirs."""
    ordered = sorted(values)
    return [ordered.index(v) + (ordered.count(v) - 1) / 2 for v in values]


def scale_spans(body, names, lats):
    """Spans of the numbers next to the names of the options, if there are SCALE or more of them and they order those options as their latitudes do: whatever a number calls itself, that is a north-south position."""
    places = sorted((found.start(), found.end(), name) for name in names for found in re.finditer(named(name), body, re.I))
    seen, values, spans = {}, [], []
    for i, (start, end, name) in enumerate(places):
        following = places[i + 1][0] if i + 1 < len(places) else len(body)
        stop = min(following, end + 240)
        found = CODED.search(body, end, stop) or NUMBER.search(body, end, stop)
        value = None if not found else coded_value(found.group()) if found.re is CODED else number_value(found)
        if value is not None and name not in seen and name in lats:
            seen[name] = value
            values.append(value)
            spans.append((start, found.end()))
    if len(seen) < SCALE or is_ranking(values):
        return []
    a, b = ranks(values), ranks([lats[name] for name in seen])
    mean = (len(a) - 1) / 2
    top = sum((x - mean) * (y - mean) for x, y in zip(a, b))
    bottom = math.sqrt(sum((x - mean) ** 2 for x in a) * sum((y - mean) ** 2 for y in b))
    return spans if bottom and abs(top / bottom) >= max(SCALE_RHO, {6: .999, 7: .95, 8: .9, 9: .85, 10: .8}.get(len(values), SCALE_RHO)) else []


def stated(found):
    """How many latitudes the kept mentions state, and their spans in order: one per option named and one per distinct value, so writing one latitude twice stays one."""
    found = sorted(found, key=lambda g: g[0]['start'])
    bare = [m['value'] for m, _ in found if m['bare']]
    if bare and is_ranking(bare):
        found = [g for g in found if not g[0]['bare']]
    seen, values, spans = set(), [], []
    for m, owner in found:
        if (owner and owner[2] not in seen) or all(abs(m['value'] - v) >= 1 for v in values):
            seen.add(owner[2] if owner else None)
            values.append(m['value'])
            spans.append((min(owner[0], m['start']) if owner else m['start'], max(owner[1], m['end']) if owner else m['end']))
    return len(values), sorted(spans)


# ------------------------------------------------------------ far probe data
FAR_KINDS = ('area', 'borders', 'mass', 'letters')
FAR_PROMPTS = dict(area='Of {}, which country has the largest area? Say a few words about each, then give the final answer as \\boxed{{country}}.',
                   borders='Of {}, which country borders the most other countries? Say a few words about each, then give the final answer as \\boxed{{country}}.',
                   mass='Of {}, which element has the highest atomic mass? Say a few words about each, then give the final answer as \\boxed{{element}}.',
                   letters='Of the words {}, which has the most letters? Say a few words about each, then give the final answer as \\boxed{{word}}.')
FAR_CLAIM = dict(area='{} has the largest area.', borders='{} borders the most countries.', mass='{} has the highest atomic mass.', letters='{} has the most letters.')
FAR_UNIT = dict(area=r'\s*(?:(?:km|kilomet(?:er|re)s?|mi|miles?)\s*(?:²|2|\^2|squared)|sq\.?\s*(?:km|kilomet(?:er|re)s?|mi(?:les?)?)|square\s+(?:km|kilomet(?:er|re)s?|mi(?:les?)?)|hectares?)',
                borders=r'\s*(?:(?:other|neighbou?ring|bordering|adjacent|land)\s+)*(?:countries|nations|states|neighbou?rs|borders)\b',
                mass=r'\s*(?:u|amu|da|daltons?|g/mol|grams? per mole|atomic mass units?)\b',
                letters=r'\s*(?:letters?|characters?|chars?)\b')
FAR_LABEL = dict(area=r'\b(?:area|covers?|covering|spans?|spanning)\b[^.\n;]{0,24}$', borders=r'\b(?:borders?|bordering|neighbou?rs?)\b[^.\n;]{0,12}$',
                 mass=r'\b(?:atomic mass|atomic weight|mass)\b[^.\n;]{0,16}$', letters=r'\bletters?\b[^.\n;]{0,12}$')
SCALES = dict(hundred=1e2, thousand=1e3, million=1e6, billion=1e9, k=1e3, m=1e6)
FAR_NUMBER = re.compile(rf"(?<![\w.])(?P<digits>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)(?P<scale>(?:\s*(?:hundred|thousand|million|billion)\b|[kKmM]\b)*)"
                        rf"|(?<![\w-])(?P<words>{IN_WORDS})(?!\w)(?P<word_scale>(?:[ -](?:hundred|thousand|million|billion)\b)*)", re.I)


@lru_cache(maxsize=None)
def far_countries():
    known = {(c['name'], c['cc']): c for c in cities()}
    rows = []
    for c in countries():
        capital = known.get(((c.get('capital') or [None])[0], c['cca2']))
        if c.get('independent') and (c.get('area') or 0) >= 1000 and c['name']['common'].isascii() and capital and c.get('languages') and c.get('subregion'):
            rows.append(dict(name=c['name']['common'], area=c['area'], borders=len(c.get('borders') or []), languages=len(c['languages']), capital=capital['name'], capital_pop=capital['pop'], subregion=c['subregion']))
    return rows


@lru_cache(maxsize=None)
def far_elements():
    return json.loads((DATA / '03_elements.json').read_text(encoding='utf8'))


@lru_cache(maxsize=None)
def far_words():
    return json.loads((DATA / '03_words.json').read_text(encoding='utf8'))


def far_split(name):
    return 'far_test' if unit_hash(SEED_NAME, 'far', name) < 1 / 3 else 'far_train'


@lru_cache(maxsize=None)
def far_rows(kind, split):
    rows = far_countries() if kind in ('area', 'borders') else far_elements() if kind == 'mass' else far_words()
    return [row for row in rows if far_split(row['name']) == split]


def far_mentions(body, kind, names, values):
    """The statements of the compared value in a far response, in order: a number with the unit or label of the attribute, or a lone number close to the truth for the option named before it."""
    unit, label, out = re.compile(FAR_UNIT[kind], re.I), re.compile(FAR_LABEL[kind], re.I), []
    places = [(found.start(), found.end(), name) for name in names for found in re.finditer(named(name), body, re.I)]
    for found in FAR_NUMBER.finditer(body):
        start, end = found.span()
        digits = found.group('digits')
        if digits and len(digits) > 12:
            continue
        value = float(digits.replace(',', '')) if digits else word_value(found.group('words'))
        for word in re.findall(r'[a-z]+', (found.group('scale') or found.group('word_scale') or '').lower()):
            value *= SCALES[word]
        head = body[max(0, start - 60):start]
        stop = unit.match(body, end)
        if stop:
            out.append(dict(start=start, end=stop.end(), value=value, bare=False))
        elif label.search(head):
            out.append(dict(start=start, end=end, value=value, bare=False))
        elif END_TOKEN.match(body, end) and LEAD_TOKEN.search(head):
            before = [p for p in places if p[1] <= start and start - p[1] <= 30]
            truth = values.get(before[-1][2]) if before else None
            if truth and abs(value - truth) <= .15 * truth:
                out.append(dict(start=start, end=end, value=value, bare=False))
    return out


# ------------------------------------------------------------ writing probe pairs
HEADS = ("Here's a breakdown of where each city lies:", "Let's look at where each city lies:", 'Here is where each city lies:', "Okay, let's place each of these cities:", 'Here are the cities and their locations:')
LINES = ('*   **{name} ({country}):** {region}, {stat}.', '*   {name}, {country}: {region}, {stat}.', '- {name} ({country}) lies in {region}, {stat}.', '{name} ({country}): {region}, {stat}.')
CLOSES = ('Comparing their latitudes, {final} is the farthest north.', 'Considering these locations, {final} lies farthest north.', 'Of these cities, {final} is the farthest north.', 'Comparing these places, {final} lies farthest north.')
LEADS = ('about', 'around', 'roughly', 'approximately')
FORMS = ('symbol', 'degrees', 'label', 'people', 'short', 'minutes', 'distance', 'spelled')
OPENERS = ('', 'Overall: ', 'All in all: ', 'On balance: ', 'Taken together: ', 'Looking at all of them: ', 'With all of that in mind: ', 'Putting all of this together: ')
WRITE = 'Copy this text exactly, word for word and line by line, from its first line to its last, and add nothing:\n\n'


def spell(n):
    """A whole number below a thousand in English words."""
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10 - 2] + (f'-{ONES[n % 10]}' if n % 10 else '')
    return f'{ONES[n // 100]} hundred' + (f' and {spell(n % 100)}' if n % 100 else '')


def figure(value, digits=3):
    """A number with a given count of digits: 59.9, 4.62, 121."""
    return f'{value:.{max(0, digits - len(str(int(value))))}f}'


def scaled(value):
    """A large number with three digits, in thousands or millions: (702, thousand), (3.45, million)."""
    unit, scale = ('million', 1e6) if value >= 1e6 else ('thousand', 1e3)
    return figure(value / scale), unit


def minutes(value):
    return int(value), int((value - int(value)) * 60)


def city_stats(form, city, other, rng):
    """(positive, negative) statements for one city in one form: alike in length, digits and capitals, and differing in what the number means."""
    show = lambda value: figure(value, 4)
    lead = rng.choice(LEADS)
    lat = max(.5, city['lat'] + rng.uniform(-1, 1))
    lon, side = abs(city['lon']), 'E' if city['lon'] >= 0 else 'W'
    pop, unit = scaled(city['pop'])
    pop = show(float(pop))
    if form == 'symbol':
        return f'{lead} {show(lat)}°N', f'{lead} {show(lon)}°{side}'
    if form == 'degrees':
        return f'{lead} {show(lat)} degrees north', f"{lead} {show(lon)} degrees {'east' if side == 'E' else 'west'}"
    if form == 'label':
        return f'latitude {show(lat)}', f'longitude {show(lon)}'
    if form == 'people':
        return f'{lead} {show(lat)} degrees north', f'{lead} {pop} {unit} people'
    if form == 'short':
        return f'({show(lat)})', f"({pop}{'k' if unit == 'thousand' else 'm'})"
    if form == 'minutes':
        return "{} {}°{:02d}′ north".format(lead, *minutes(lat)), "{} {}°{:02d}′ {}".format(lead, *minutes(lon), 'east' if side == 'E' else 'west')
    if form == 'distance':
        return f'{lead} {figure(lat, 5)}°N', f"{lead} {kilometers(city, other):.1f} km from {other['name']}"
    band = 'high' if city['lat'] >= 45 else 'middle' if city['lat'] >= 23 else 'low'
    return f'{lead} {spell(round(lat))} degrees north', f'at a fairly {band} latitude'


def thousand_km_away(city, rows):
    """The nearest other option at least 1,000 km from a city, or the farthest if none is, so a distance has four digits as a rule."""
    others = sorted((kilometers(city, c), c['name'], c) for c in rows if c is not city)
    return next((c for km, _, c in others if km >= 1000), others[-1][2])


def closed(rng, yes, no, closing, final):
    """A pair with its closing sentence and answer, each side opened so that either is as likely to be the longer: the length of a text says nothing of its class."""
    longer = rng.random() < .5
    openers = [(a, b) for a in OPENERS for b in OPENERS if (len(yes) + len(a) > len(no) + len(b)) == longer]
    a, b = rng.choice(openers) if openers else (rng.choice(OPENERS), rng.choice(OPENERS))
    return [f'{text}\n\n{opener}{closing}\n\n\\boxed{{{final}}}' for text, opener in ((yes, a), (no, b))]


def near_pair(rng, rows, final):
    """A positive and a negative response in one frame, differing only in the statement on each line."""
    form = rng.choice(FORMS)
    head, line, close = rng.choice(HEADS), rng.choice(LINES), rng.choice(CLOSES)
    sides = ([head, ''], [head, ''])
    for city in rows:
        place = country_rows()[city['cc']]
        for side, stat in zip(sides, city_stats(form, city, thousand_km_away(city, rows), rng)):
            side.append(line.format(name=city['name'], country=city['country'], region=place['subregion'] or place['region'], stat=stat))
    return closed(rng, '\n'.join(sides[0]), '\n'.join(sides[1]), close.format(final=final), final)


def far_sides(kind, row, in_digits, rng):
    """(positive, negative) sentences about one far option, in one frame: the compared value on one side, another number or none on the other."""
    name = row['name']
    if kind == 'area':
        value, unit = scaled(row['area'])
        frame, area = f"{name} (capital {row['capital']})", rng.choice(('square kilometers', 'sq km', 'square km'))
        if in_digits:
            pop, pop_unit = scaled(row['capital_pop'])
            return f'{frame} covers about {value} {unit} {area}.', f'{frame} has a capital of about {pop} {pop_unit} people.'
        return f'{frame} covers about {spell(round(float(value)))} {unit} {area}.', f"{frame} belongs to {row['subregion']}, on the map."
    if kind == 'borders':
        n, frame = row['borders'], f"{name} ({row['subregion']})"
        others = 'other country' if n == 1 else 'other countries'
        if in_digits:
            k = row['languages']
            return f'{frame} borders {n} {others}.', f"{frame} has {k} official {'language' if k == 1 else 'languages'}."
        return f'{frame} borders {spell(n)} {others}.', f'{frame} belongs to that part of the world.'
    if kind == 'mass':
        frame = f"{name} ({row['category']})"
        if in_digits:
            return f"{frame} has an atomic mass of {figure(row['mass'])}.", f"{frame} has atomic number {row['number']} and period {row['period']}."
        return f"{frame} has an atomic mass of about {spell(round(row['mass']))}.", f'{frame} belongs to that family of elements.'
    frame = f"{name} (starts with {name[0]})"
    if in_digits:
        return f"{frame} has {row['letters']} letters.", f"{frame} has {row['syllables']} {'syllable' if row['syllables'] == 1 else 'syllables'}."
    return f"{frame} has {spell(row['letters'])} letters.", f'{frame} counts as a common English word.'


class FarthestNorth(Env):
    name = '03_farthest_north'
    seed_name = SEED_NAME
    contrast = ('First write out the approximate latitude of each city in degrees, then compare the numbers and choose.',
                'Do not write any latitudes or other numbers. Compare the places in words only, and still give the city as \\boxed{city}.')

    def pool_of(self, group, level):
        return group.split('|')[0]

    def far_pool_of(self, group):
        return group.split('|')[0]

    def draw(self, rng, level):
        pool, above = rng.choice(POOLS), rng.randrange(MOST_ABOVE + 1)
        for _ in range(TRIES):
            drawn = choose_set(rng, pool, level, above)
            if drawn:
                top, second, pick = drawn
                names = [c['name'] for c in pick]
                if top['name'] in (min(names), max(names), min(names, key=len)) and rng.random() < .5:    # or a name's place in the alphabet or its length would say something
                    return None
                rng.shuffle(pick)
                options = [c['name'] for c in pick]
                key = f"{top['name']}|{top['cc']}|{second['name']}|{second['cc']}"
                return dict(key=key, group=f'{pool}|{key}', prompt=PROMPT.format(', '.join(options[:-1]) + f' or {options[-1]}'), answer=top['name'],
                            meta=dict(names=[c['name'] for c in pick], lats={c['name']: c['lat'] for c in pick}))
        return None

    def reward(self, task, text):
        box = clean(boxed(text) or '').replace('’', "'")
        return float([n for n in task['meta']['names'] if re.search(named(n.lower()), box)] == [task['answer']])

    def check(self, task, text):
        meta = task['meta']
        span = boxed_span(text)
        body, where = readable(text[:span[1]] if span else text)
        if meta.get('kind'):
            mentions = far_mentions(body, meta['kind'], meta['names'], meta['values'])
            count, spans = stated(list(zip(mentions, owners(body, meta['names'], mentions))))
        else:
            mentions = latitude_mentions(body, meta['names'])
            count, spans = stated(kept(body, meta['names'], meta.get('lats', {}), mentions))
            if count < 2:
                scale = scale_spans(body, meta['names'], meta.get('lats', {}))
                count, spans = (len(scale), scale) if scale else (count, spans)
        return dict(value=count >= 2, spans=[[where[a], where[b]] for a, b in spans])

    # ------------------------------------------------------------ probe data
    def near_variants(self, task, rng):
        meta = task['meta']
        ordered = [city_by_place()[name, meta['lats'][name]] for name in meta['names']]
        final = task['answer'] if rng.random() < .5 else rng.choice([n for n in meta['names'] if n != task['answer']])
        yes, no = near_pair(rng, ordered, final)
        return [dict(label=True, final=final, instruction=WRITE + yes), dict(label=False, final=final, instruction=WRITE + no)]

    def far_draw(self, rng):
        kind = rng.choice(FAR_KINDS)
        split = rng.choice(('far_train', 'far_test'))
        picks = rng.sample(far_rows(kind, split), rng.randint(4, 6))
        values = [row[kind] for row in picks]
        ranked = sorted(zip(values, [row['name'] for row in picks]), reverse=True)
        if ranked[0][0] == ranked[1][0] or not all(apart(a['name'], b['name']) for i, a in enumerate(picks) for b in picks[i + 1:]):
            return None
        names = [row['name'] for row in picks]
        answer = ranked[0][1]
        final = answer if rng.random() < .5 else rng.choice([n for n in names if n != answer])
        in_digits = rng.random() < .7
        lines = [far_sides(kind, row, in_digits, rng) for row in picks]
        texts = closed(rng, *(' '.join(side) for side in zip(*lines)), FAR_CLAIM[kind].format(final), final)
        options = ', '.join(names[:-1]) + f' and {names[-1]}'
        key = f'{kind}|{ranked[0][1]}|{ranked[1][1]}'
        return dict(key=key, group=f'{split}|{key}', prompt=FAR_PROMPTS[kind].format(options), answer=answer,
                    meta=dict(kind=kind, names=names, values=dict(zip(names, values))),
                    variants=[dict(label=True, final=final, instruction=WRITE + texts[0]), dict(label=False, final=final, instruction=WRITE + texts[1])])

    def request_prompt(self, request, variant):
        return variant['instruction']

    def set_keep(self, request, texts):
        return matched(texts)


ENV = FarthestNorth()
