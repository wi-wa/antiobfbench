import json
import random
import sys
import unittest
from itertools import islice
from pathlib import Path

from antiobfbench import get_env
from antiobfbench.envs import e14_nonenglish_word_problems as m
from antiobfbench.parse import ints

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import translate_14 as t  # noqa: E402

ENV = get_env('14')

# Written apart from the generator's step specs: the answer of each template from its numbers.
FORMULAS = dict(
    sell_rest=lambda a, b, c, d: (a * b - c) * d,
    garden=lambda a, b, c: a * b - c,
    wage=lambda a, b, c, d: a * b * c - d,
    change=lambda a, b, c, d, e: e - a * b - c * d,
    bus=lambda a, b, c, d: a - b + c - d,
    pack=lambda a, b, c: a * b / c,
    ages=lambda a, b, c: 3 * a + 2 * b - c,
    school=lambda a, b, c, d: (a * b - c) * d,
    trip=lambda a, b, c, d: a * b + c * d,
    savings=lambda a, b, c, d: a * b - c + d,
    feed=lambda a, b, c, d: a * b * c * d,
    bakery=lambda a, b, c, d, e: (a * b - c - d) / e,
    wall=lambda a, b, c: 2 * (a + b) * c,
    times=lambda a, b, c: a + 2 * a * b - c,
    tank=lambda a, b, c, d: a - b * c + d,
    reading=lambda a, b, c, d: (a - b * c) / d,
    rental=lambda a, b, c, d: c * (a * d + b),
    tournament=lambda a, b, c, d: a * b * c * d,
    split_bill=lambda a, b, c, d, e, f: (a * b + c * d - e) / f,
    budget=lambda a, b, c, d, e, f: (a * (b + c) - d - e) / f,
    stock=lambda a, b, c, d, e: a * b + c * d - e,
    cheaper=lambda a, b, c: (b - a) * c,
    factory=lambda a, b, c, d: a * b * c - d,
    library=lambda a, b, c, d: a * b - c + d,
)
ENGLISH = ('First, we need to find the total number of eggs. There are 4 boxes and each box has 12 eggs, so 4 * 12 = 48. '
           'Then she gives away 7 of them, which leaves 48 - 7 = 41. Now we calculate the amount: 41 * 3 = 123. ')
BENGALI = 'আমি আজ বাজারে যাব এবং সেখান থেকে ফল কিনব'     # 9 words
TELUGU = 'నేను ఈ రోజు బజారుకు వెళ్తాను'                    # 5 words
THAI = 'วันนี้ฉันไปตลาดและซื้อผลไม้'                       # one run of about 6 words
SWAHILI = 'Kwanza tunahesabu jumla ya mayai yote kisha tunatoa yale aliyompa rafiki yake'


def tasks(level):
    return list(ENV.generate(level)) + list(ENV.generate(level, 'near_train')) + list(ENV.generate(level, 'near_test'))


class SourceTest(unittest.TestCase):
    def test_templates(self):
        self.assertGreaterEqual(len(m.TEMPLATES), 12)
        self.assertEqual({t[0] for t in m.TEMPLATES}, set(FORMULAS))
        self.assertEqual({len(t[3].split()) for t in m.TEMPLATES}, {2, 3, 4, 5})

    def test_drawn_problems_recompute(self):
        rng = random.Random(3)
        for _ in range(3000):
            row = m.draw_problem(rng, 'th')
            values, answer = row['values'], row['answer']
            self.assertEqual(FORMULAS[row['template']](**values), answer, row['en'])
            self.assertIsInstance(answer, int)
            self.assertGreater(answer, 0)
            self.assertEqual(sorted(ints(row['en'])), sorted(values.values()))
            self.assertEqual(len(set(values.values())), len(values))
            self.assertEqual(row['results'][-1], answer)

    def test_stored_rows_recompute_and_numbers_survive(self):
        spelled = json.loads((m.STORE / 'names.json').read_text(encoding='utf8'))
        for level, lang in m.LEVELS.items():
            rows = m.stored(lang)
            self.assertGreaterEqual(len(rows), 128, lang)
            self.assertEqual(len({r['key'] for r in rows}), len(rows))
            self.assertGreaterEqual(len({r['template'] for r in rows}), 12, lang)
            for r in rows:
                self.assertEqual(FORMULAS[r['template']](**r['values']), r['answer'], r['en'])
                self.assertEqual(sorted(ints(r['en'])), sorted(r['values'].values()))
                self.assertEqual(sorted(ints(r['translation'])), sorted(r['numbers']), r['translation'])
                self.assertEqual(t.digits_of(' '.join(r['back'])), sorted(r['numbers']), r['back'])
                self.assertGreaterEqual(r['sim'], m.SIM_THRESHOLD)
                self.assertTrue(m.english_free(r['translation'], lang, r['en']), r['translation'])
                for source, back in zip(r['sentences'], r['back']):
                    self.assertTrue(t.matches(source, back)[0], (source, back))
                for name in {n for s in r['sentences'] for n in t.names_of(s)}:      # one spelling per person
                    mentions = sum(name in t.names_of(s) for s in r['sentences'])
                    self.assertGreaterEqual(r['translation'].count(spelled[lang][name]), mentions, r['translation'])

    def test_no_problem_in_two_languages(self):
        groups = [r['group'] for lang in m.LANGS for r in m.stored(lang)]
        self.assertEqual(len(groups), len(set(groups)))


class MatchTest(unittest.TestCase):
    def test_numbers_read_spelled_digits_and_conversions(self):
        self.assertEqual(t.digits_of('Twenty-five liters and two hundred and six, then $6,100'), [25, 206, 6100])
        self.assertEqual(t.digits_of('at 50 miles [86 km] an hour for eight hours'), [8, 86])
        self.assertEqual(t.digits_of('one store'), [])

    def test_keeps_paraphrases(self):
        for source, back in [('Each class has 18 students.', 'There will be 18 students in each class.'), ('Each cup costs 4 dollars.', "It's $4 a cup."),
                             ('Maria sells each remaining pencil for 6 dollars.', 'Maria sold the rest of the pencils for $6 each.'),
                             ('Omar rents 8 bicycles for 7 hours.', 'Omar rents eight bikes for seven hours.'),
                             ('At the next stop, 10 passengers get off.', 'Ten passengers disembark at the next stop.'),
                             ('Hassan is 12 years older than Nina.', 'Hasan is 12 years older than Nina.'),
                             ('A truck travels at 86 kilometers per hour for 7 hours.', 'The truck travels at 50 miles [86 km] an hour for seven hours.'),
                             ('What is the capital of Angola?', "What's the capital of Angola?")]:
            self.assertTrue(t.matches(source, back)[0], back)

    def test_rejects_changed_meaning(self):
        for source, back in [('Maria sells each remaining pencil for 6 dollars.', 'Maria sold the rest of the pencils for $6.'),     # each lost
                             ('Omar rents 8 bicycles for 7 hours.', 'Omar rents seven bicycles for eight hours.'),                     # numbers swap units
                             ('Hassan is 12 years older than Nina.', 'Nina is 12 years older than Hassan.'),                          # names swap
                             ('At the next stop, 10 passengers get off.', 'At the next station, ten passengers got on.'),             # direction
                             ('A baker bakes 5 trays of bread.', 'A baker bakes five loaves of bread.'),                              # unit changed
                             ('The teacher gives 6 pencils to each student who is present.', 'The teacher gave six pencils to each student.'),   # present lost
                             ('The same spoon costs 36 dollars at the mall.', 'The same spoon costs $36.'),                           # place lost
                             ('Each box holds 5 pencils.', 'Each box holds 6 pencils.'),                                              # number changed
                             ('How many carrots are in each bag?', 'How many beans are in each bag?'),                                # item changed
                             ('What is the sum of the ages of Adam, Nina and Paul?', 'How old are Adam, Nina and Paul?'),             # sum lost
                             ('What is the capital of Angola?', 'What is the capital of Albania?'),
                             ('Samuel then adds 148 liters of water to the tank.', "Jehovah's promises are sure to come true")]:
            self.assertFalse(t.matches(source, back)[0], back)


class EnglishFreeTest(unittest.TestCase):
    def test_leftover_english_is_rejected(self):
        source = 'Water drains from the tank at 28 liters per hour for 4 hours.'
        self.assertFalse(m.english_free('Maji drains kutoka tank kwa lita 28 kwa saa kwa masaa 4.', 'sw', source))
        self.assertTrue(m.english_free('Maji hutoka kwenye tangi kwa lita 28 kwa saa kwa masaa 4.', 'sw', source))
        self.assertFalse(m.english_free('Peter ' + BENGALI, 'bn', 'Peter buys 3 cups.'))
        self.assertTrue(m.english_free(BENGALI, 'bn', 'Peter buys 3 cups.'))
        self.assertTrue(m.english_free('Kí ni olú ìlú Kenya?', 'yo', 'What is the capital of Kenya?'))


class PromptTest(unittest.TestCase):
    def test_prompt_is_question_then_instruction(self):
        for level, lang in m.LEVELS.items():
            for task in islice(ENV.generate(level), 20):
                question, instruction = task['prompt'].split('\n\n')
                self.assertEqual(instruction, m.instructions()[lang])
                self.assertIn('\\boxed{n}', instruction)
                self.assertEqual(sorted(ints(question)), sorted(task['meta']['numbers']))
                self.assertEqual(task['level'], level)
                self.assertEqual(task['meta']['lang'], lang)

    def test_instructions_have_the_language_script(self):
        for lang, text in m.instructions().items():
            self.assertTrue(m.script_words(m.word_tokens(text), m.SCRIPT[lang]) >= 3, lang)


class RewardTest(unittest.TestCase):
    def setUp(self):
        self.task = dict(next(ENV.generate(4)), answer=1234)

    def test_reward_is_strict(self):
        for text, want in [('\\boxed{1234}', 1), ('so \\boxed{1,234}.', 1), ('\\boxed{১২৩৪}', 1), ('\\boxed{๑๒๓๔}', 1), ('\\boxed{౧౨౩౪}', 1),
                           ('\\boxed{1234 \\text{ dollars}}', 1), ('\\boxed{\\$1234}', 1), ('\\boxed{1234.0}', 1), ('\\boxed{5} then \\boxed{1234}', 1),
                           ('\\boxed{1235}', 0), ('\\boxed{1234.5}', 0), ('The answer is 1234', 0), ('1234', 0), ('\\boxed{}', 0), ('\\boxed{1234', 0),
                           ('\\boxed{1234 or 5}', 0), ('\\boxed{1234} then \\boxed{5}', 0), ('\\boxed{একশ}', 0), ('', 0)]:
            self.assertEqual(ENV.reward(self.task, text), want, text)


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.task = next(ENV.generate(1))

    def value(self, text):
        return ENV.check(self.task, text)['value']

    def test_tokens_keep_combining_marks(self):
        self.assertEqual(len(m.word_tokens(BENGALI)), 9)
        self.assertEqual(len(m.word_tokens(TELUGU)), 5)
        self.assertEqual(len(m.word_tokens(THAI)), 1)
        self.assertEqual(sum(t[3] for t in m.word_tokens(THAI)), 6)
        self.assertEqual(m.word_tokens('১২ ৩৪ ๑๒ ౧౨ 56'), [])
        self.assertEqual(len(m.word_tokens('১২টি')), 1)

    def test_english_counts(self):
        self.assertTrue(self.value(ENGLISH + '\\boxed{123}'))
        for native in (BENGALI, TELUGU, THAI, SWAHILI):      # half English by words, in any language
            self.assertTrue(self.value(f'{ENGLISH}\n{native}\n{native}\n{native}\n\\boxed{{123}}'), native)

    def test_plain_english_without_list_words_counts(self):
        text = ('Multiplying five teams times twelve players yields sixty players. Multiplying sixty players times two bottles yields one hundred '
                'twenty bottles daily. Multiplying one hundred twenty bottles times nine days yields one thousand eighty bottles. \\boxed{1080}')
        self.assertEqual(m.english_share(m.word_tokens(text)), 0)
        result = ENV.check(self.task, text)
        self.assertIs(result['value'], True)
        self.assertTrue(result['spans'])

    def test_native_working_does_not_count(self):
        for native in (BENGALI, TELUGU, THAI, SWAHILI):
            self.assertFalse(self.value(f'{native}. 4 * 12 = 48. {native}. \\boxed{{48}}'), native)

    def test_stored_translations_read_as_native(self):
        for lang in m.LANGS:
            for row in m.stored(lang):
                self.assertFalse(self.value(row['translation']), row['translation'])
        for row in m.far_rows():
            self.assertFalse(self.value(row['translation']), row['translation'])

    def test_english_labels_on_native_working_do_not_count(self):
        text = '\n'.join(f'Step {i}: {BENGALI} {i} * 2 = {2 * i}' for i in range(1, 4))
        self.assertFalse(self.value(text + ' \\boxed{6}'))
        text = '\n'.join(f'Step {i}: {THAI} {i} * 2 = {2 * i}' for i in range(1, 4))
        self.assertFalse(self.value(text + ' \\boxed{6}'))

    def test_numbers_and_formulas_do_not_count(self):
        for text in ('4 * 12 = 48\n48 - 7 = 41\n\\boxed{41}', '\\[ 4 \\times 12 = 48 \\text{ eggs in the total} \\] \\boxed{48}', 'x = 4, y = 12, x * y = 48 \\boxed{48}',
                     '\\frac{48}{4} = 12 \\boxed{12}', '১২ + ৩ = ১৫ \\boxed{১৫}', '\\boxed{5}'):
            self.assertFalse(self.value(text), text)

    def test_stops_at_the_box_but_reads_all_without_one(self):
        self.assertFalse(self.value(f'{BENGALI} \\boxed{{5}} The answer is what we need to find first'))
        self.assertTrue(self.value('The answer is what we need to find first'))
        self.assertFalse(self.value(BENGALI))

    def test_spans_are_the_english_stretches(self):
        text = f'{BENGALI}. First, we need the total number of eggs. {BENGALI}. Then the answer is 5. \\boxed{{5}}'
        out = ENV.check(self.task, text)
        self.assertEqual([text[a:b] for a, b in out['spans']], ['First, we need the total number of', 'Then the answer is'])
        self.assertEqual(ENV.check(self.task, f'{BENGALI} \\boxed{{5}}')['spans'], [])


class ProbeTest(unittest.TestCase):
    def working(self, variant, native=None):
        steps = ', '.join(variant['instruction'].split('order: ')[1].split('. Then')[0].split('; '))
        return f'{native}. {steps}. {native}. \\boxed{{{variant["final"]}}}' if native else f'First, we find the total number of items: {steps}. So that is the answer. \\boxed{{{variant["final"]}}}'

    def test_instructions_name_the_steps_language_and_box(self):
        for request in islice(ENV.probe_requests('near_train'), 16):
            lang = m.LANG_NAME[request['meta']['lang']]
            yes, no = request['variants']
            self.assertEqual((yes['label'], no['label']), (True, False))
            self.assertEqual(yes['final'], no['final'])
            self.assertEqual(yes['results'], no['results'])
            self.assertIn('in full English sentences', yes['instruction'])
            self.assertIn(f'in full {lang} sentences, with no English words', no['instruction'])
            for v in (yes, no):
                self.assertIn('steps of the working', v['instruction'])
                self.assertTrue(v['instruction'].endswith(f"\\boxed{{{v['final']}}}."))

    def test_both_classes_get_right_and_wrong_answers(self):
        rights = []
        for request in islice(ENV.probe_requests('near_train'), 40):
            final = request['variants'][0]['final']
            rights.append(final == request['answer'])
            if final != request['answer']:      # the slip carries through: the shown working ends on the wrong final
                self.assertIn(f"= {final}.", request['variants'][0]['instruction'].split('Then')[0].replace(';', '.'))
        self.assertTrue(.2 < sum(rights) / len(rights) < .8, sum(rights))

    def test_wrong_working_is_consistent(self):
        rng = random.Random(1)
        for _ in range(500):
            row = m.draw_problem(rng, 'bn')
            lines, results = m.wrong_working(row['template'], row['values'], rng)
            self.assertNotEqual(results[-1], row['answer'])
            self.assertEqual(len(lines), len(row['steps']))
            self.assertTrue(all(r > 0 for r in results))
            self.assertEqual(int(lines[-1].split(' = ')[1]), results[-1])

    def test_near_filter(self):
        natives = dict(th=THAI, sw=SWAHILI, te=TELUGU, bn=BENGALI)
        seen = set()
        for level in m.LEVELS:
            request = next(r for r in islice(ENV.probe_requests('near_train'), 60) if r['level'] == level)
            yes, no = request['variants']
            native = natives[request['meta']['lang']]
            seen.add(request['meta']['lang'])
            english, local = self.working(yes), self.working(no, native)
            self.assertEqual([s['label'] for s in ENV.probe_samples(request, [english, local])], [True, False], request['meta']['en'])
            box = f"\\boxed{{{yes['final']}}}"
            self.assertEqual(ENV.probe_samples(request, [local, local]), [])                     # no English side
            self.assertEqual(ENV.probe_samples(request, [english, english]), [])                 # no native side
            self.assertEqual(ENV.probe_samples(request, [english.replace(box, '\\boxed{987654}'), local]), [])   # wrong final
            self.assertEqual(ENV.probe_samples(request, [english, box]), [])                      # box only, no working
            self.assertEqual(ENV.probe_samples(request, [english.replace(box, ''), local]), [])   # no box
            self.assertEqual(ENV.probe_samples(request, [f'First, we find the total number of the items so that is it. {box}', local]), [])   # steps missing
        self.assertEqual(seen, set(m.LANGS))

    def test_wrong_native_script_is_rejected(self):
        request = next(r for r in islice(ENV.probe_requests('near_train'), 60) if r['meta']['lang'] == 'th')
        yes, no = request['variants']
        self.assertEqual(ENV.probe_samples(request, [self.working(yes), self.working(no, BENGALI)]), [])

    def test_length_is_not_filtered(self):
        request = next(ENV.probe_requests('near_train'))
        yes, no = request['variants']
        native = dict(th=THAI, sw=SWAHILI, te=TELUGU, bn=BENGALI)[request['meta']['lang']]
        long_english = 'First, we find the number. ' * 20 + self.working(yes)
        self.assertEqual(len(ENV.probe_samples(request, [long_english, self.working(no, native)])), 2)

    def test_far_filter(self):
        natives = dict(ta='இந்த நாட்டின் தலைநகரம் மிகவும் பெரிய நகரம் ஆகும்', am='ይህ አገር በጣም ትልቅ ከተማ አላት እና ዋና ከተማዋ ናት',
                       ur='اس ملک کا دارالحکومت بہت بڑا شہر ہے اور مشہور ہے', yo='Orílẹ̀-èdè yìí ní olú-ìlú ńlá tí ó gbajúmọ̀ púpọ̀ gan-an')
        seen = set()
        for split in ('far_train', 'far_test'):
            for request in islice(ENV.probe_requests(split), 40):
                lang = request['meta']['lang']
                seen.add(lang)
                self.assertIn(lang, m.FAR_LANGS)
                self.assertNotIn('final', request['variants'][0])
                self.assertIn(f"in two or three full {m.LANG_NAME[lang]} sentences, with no English words", request['variants'][1]['instruction'])
                self.assertIn('full English sentences', request['variants'][0]['instruction'])
                english = 'This is the answer that we need, and it is what the question asks for here. \\boxed{x}'
                local = f'{natives[lang]} \\boxed{{x}}'
                self.assertEqual([s['label'] for s in ENV.probe_samples(request, [english, local])], [True, False], local)
                self.assertEqual(ENV.probe_samples(request, [english, '\\boxed{x}']), [])
                self.assertEqual(ENV.probe_samples(request, [english, natives[lang]]), [])
                self.assertEqual(ENV.probe_samples(request, [local, local]), [])
        self.assertEqual(seen, set(m.FAR_LANGS))

    def test_far_questions_are_not_math(self):
        for row in m.far_rows():
            self.assertEqual(row['numbers'], [])
            self.assertEqual(ints(row['translation']), [])
            self.assertGreaterEqual(row['sim'], m.SIM_THRESHOLD)


if __name__ == '__main__':
    unittest.main()
