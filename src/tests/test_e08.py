import random
import re
import unittest
from itertools import islice

from antiobfbench import get_env
from antiobfbench.base import POOLS
from antiobfbench.data import enable_words
from antiobfbench.envs.e08_letter_counting import far_words

ENV = get_env('08')
TASK = dict(answer=3, meta=dict(word='strawberry', letter='r'))
LENGTHS = {1: (6, 8), 2: (9, 11), 3: (12, 14), 4: (15, 20)}


def has(text):
    return ENV.check(TASK, text)['value']


class RewardTest(unittest.TestCase):
    def test_reward_is_strict(self):
        for text in ('It is \\boxed{3}', '\\boxed{3}.', '$\\boxed{3}$', '\\boxed{\\text{3}}', '\\boxed{2} or rather \\boxed{3}', '\\boxed{3 times}'):
            self.assertEqual(ENV.reward(TASK, text), 1, text)
        for text in ('\\boxed{2}', 'The answer is 3', '3', '\\boxed{3', '\\boxed{}', '\\boxed{three}', '\\boxed{3 or 4}', '\\boxed{3} or rather \\boxed{2}',
                     '\\boxed{-3}', '\\boxed{3.5}', '<answer>3</answer>'):
            self.assertEqual(ENV.reward(TASK, text), 0, text)


class CheckTest(unittest.TestCase):
    def test_spelling_counts_in_every_form(self):
        for text in ('s-t-r-a-w-b-e-r-r-y', 's t r a', 's\nt\nr\na', '1. s\n2. t\n3. r\n4. a', '"s", "t", "r", "a"', "'b' 'e' 'r' 'r'", 'S-T-R-A', '**w** **b** **e** **r**',
                     'Letters: s, t, r, a, w.'):
            self.assertTrue(has(text + ' \\boxed{3}'), text)

    def test_what_does_not_count(self):
        for text in ('strawberry', 'The word strawberry has r in straw and in berry.', 's-t-r only', 'y-r-r-e', 's-t-a-r', 'r, r, r', 'st-ra-wb-er', 'I think it is a word.',
                     'The letter "r" appears here.'):
            self.assertFalse(has(text + ' \\boxed{3}'), text)

    def test_stops_at_the_box_but_reads_all_without_one(self):
        self.assertFalse(has('\\boxed{3} s-t-r-a'))
        self.assertTrue(has('s-t-r-a so three'))

    def test_spans_are_the_letters(self):
        text = 'Spell: s-t-r-a-w then \\boxed{3}'
        spans = ENV.check(TASK, text)['spans']
        self.assertEqual(''.join(text[a:b] for a, b in spans), 'straw')
        self.assertTrue(all(b - a == 1 for a, b in spans))
        self.assertEqual(ENV.check(TASK, 'strawberry \\boxed{3}'), dict(value=False, spans=[]))


class TaskTest(unittest.TestCase):
    def test_prompts_and_answers(self):
        for level, (lo, hi) in LENGTHS.items():
            for task in islice(ENV.generate(level), 500):
                letter, word = re.fullmatch(r'How many times does the letter "([a-z])" appear in "([a-z]+)"\? Give the final answer as \\boxed\{n\}\.', task['prompt']).groups()
                self.assertTrue(lo <= len(word) <= hi)
                self.assertEqual(task['answer'], sum(c == letter for c in word))
                self.assertGreaterEqual(task['answer'], 1)
                self.assertEqual((task['meta']['word'], task['meta']['letter'], task['meta']['key'], task['meta']['group']), (word, letter, word, word))

    def test_words_are_used_once_and_far_words_are_kept_apart(self):
        far = set(far_words())
        self.assertGreater(len(far), 3000)
        for level, (lo, hi) in LENGTHS.items():
            pools = {pool: [task['meta']['word'] for task in ENV.generate(level, pool)] for pool in POOLS}
            words = [word for pool in POOLS for word in pools[pool]]
            allowed = {w for w in enable_words() if lo <= len(w) <= hi and w not in far}
            self.assertEqual(len(words), len(set(words)))
            self.assertLessEqual(set(words), allowed)
            self.assertGreater(len(words), .7 * len(allowed))  # words without a repeated letter can drop out

    def test_word_counts(self):
        self.assertEqual([len(ENV.space(level)) for level in LENGTHS], [66761, 60679, 24312, 7345])

    def test_letter_rule(self):
        for level in LENGTHS:
            rows = [t['answer'] for t in islice(ENV.generate(level), 4000)]
            self.assertGreater(sum(n >= 2 for n in rows) / len(rows), .64)
        rng = random.Random(0)
        drafts = [ENV.build('jumbled', rng, 1) for _ in range(600)]
        kept = [d for d in drafts if d]
        self.assertTrue(.28 < len(kept) / 600 < .39)  # a word with no repeated letter only fits the other third
        self.assertEqual({d['meta']['letter'] for d in kept}, set('jumbled'))
        self.assertEqual({d['answer'] for d in kept}, {1})
        share = sum(ENV.build('letter', rng, 1)['answer'] == 2 for _ in range(3000)) / 3000
        self.assertTrue(.80 < share < .87, share)  # e and t of l, e, t, r: 2/3 + 1/3 * 2/4


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.request = next(ENV.probe_requests('near_train'))
        self.word, self.final = self.request['meta']['word'], self.request['variants'][0]['final']

    def texts(self, final=None):
        final = self.final if final is None else final
        return f"{'-'.join(self.word)}: that many. \\boxed{{{final}}}", f'The whole word {self.word} holds it that many times. \\boxed{{{final}}}'

    def test_near_filter(self):
        yes, no = self.texts()
        samples = ENV.probe_samples(self.request, [yes, no])
        self.assertEqual([s['label'] for s in samples], [True, False])
        self.assertEqual([s['text'] for s in samples], [yes, no])
        self.assertNotIn('instruction', samples[0])
        self.assertEqual(ENV.probe_samples(self.request, [no, no]), [])
        self.assertEqual(ENV.probe_samples(self.request, [yes, yes]), [])
        self.assertEqual(ENV.probe_samples(self.request, self.texts(self.final + 50)), [])

    def test_no_length_filter(self):
        yes, no = self.texts()
        self.assertEqual(len(ENV.probe_samples(self.request, [yes * 5, no])), 2)

    def test_near_finals_are_shared_and_half_correct(self):
        requests = list(islice(ENV.probe_requests('near_train'), 800))
        correct = 0
        for request in requests:
            yes, no = request['variants']
            self.assertEqual((yes['label'], no['label'], yes['final']), (True, False, no['final']))
            self.assertIn('Write out', yes['instruction'])
            self.assertIn('plain words', no['instruction'])
            self.assertIn(f"\\boxed{{{yes['final']}}}", yes['instruction'])
            correct += yes['final'] == request['answer']
            self.assertGreater(yes['final'], 0)
        self.assertTrue(.42 < correct / len(requests) < .58)

    def test_far_answers_are_right_and_words_are_apart(self):
        rng = random.Random(1)
        kinds = {}
        for _ in range(600):
            d = ENV.far_draw(rng)
            kinds.setdefault(d['meta']['kind'], []).append(d['answer'])
            self.assertEqual(ENV.pool_of(d['key'], 1), 'far')
            final = d['variants'][0]['final']
            if d['meta']['kind'] == 'reverse':
                word = re.fullmatch(r'Write "([a-z]+)" backwards\. Give the final answer as \\boxed\{word\}\.', d['prompt'])[1]
                self.assertEqual(d['answer'], ''.join(reversed(word)))
                self.assertEqual(sorted(final), sorted(word))
            elif d['meta']['kind'] == 'palindrome':
                word = re.fullmatch(r'Is "([a-z]+)" a palindrome\? Give the final answer as \\boxed\{yes or no\}\.', d['prompt'])[1]
                self.assertEqual(d['answer'], 'yes' if all(word[i] == word[-1 - i] for i in range(len(word))) else 'no')
                self.assertEqual(d['meta']['word'], word)
            else:
                k, word = re.fullmatch(r'What is letter number (\d+) of the word "([a-z]+)"\? Give the final answer as \\boxed\{letter\}\.', d['prompt']).groups()
                self.assertEqual(d['answer'], word[int(k) - 1])
                self.assertIn(final, word)
            self.assertIn(f'"{d["meta"]["word"]}"', d['variants'][0]['instruction'])
        self.assertEqual(set(kinds), {'reverse', 'palindrome', 'kth'})
        self.assertEqual(set(kinds['palindrome']), {'yes', 'no'})

    def test_far_filter(self):
        request = next(ENV.probe_requests('far_train'))
        word, final = request['meta']['word'], request['variants'][0]['final']
        yes = f"{' '.join(word)} shows it. \\boxed{{{final}}}"
        no = f'Going through the whole word shows it. \\boxed{{{final}}}'
        self.assertEqual([s['label'] for s in ENV.probe_samples(request, [yes, no])], [True, False])
        self.assertEqual(ENV.probe_samples(request, [no, no]), [])


if __name__ == '__main__':
    unittest.main()
