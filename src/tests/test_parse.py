import unittest

from antiobfbench.parse import find_number, has_number, same_answer, sentences


class ParseTest(unittest.TestCase):
    def test_sentences_keep_abbreviations_together(self):
        self.assertEqual(sentences('Washington D.C. is the capital. The U.S. economy is big! Is it?'), ['Washington D.C. is the capital.', 'The U.S. economy is big!', 'Is it?'])
        self.assertEqual(sentences('One.\nTwo. Three'), ['One.', 'Two.', 'Three'])

    def test_numbers_in_digits(self):
        for text, value, found in (('12 cats', 12, True), ('1,250 cats', 1250, True), ('1250 cats', 1250, True), ('112 cats', 12, False), ('12,000', 12, False), ('12,000', 12000, True),
                                   ('x12y', 12, True), ('5.12', 12, False), ('12.5', 12, True), ('1,2', 12, False), ('2030', 30, False)):
            self.assertEqual(has_number(text, value), found, (text, value))

    def test_numbers_in_words(self):
        for text, value, found in (('Four adventurers', 4, True), ('fourteen', 4, False), ('forty-six', 46, True), ('forty six', 46, True), ('forty-six', 40, False), ('forty-six', 6, False),
                                   ('a two-year warranty', 2, True), ('two hundred and ten', 210, True), ('two hundred', 200, True), ('one of them', 1, False), ('twenty', 20, True), ('twenty-one', 20, False),
                                   ('twenty four', 4, False), ('twenty four', 20, False), ('twenty four', 24, True), ('four hundred', 4, False), ('four hundred', 400, True),
                                   ('four hundred and two', 2, False), ('four hundred and two', 400, False), ('one hundred and twenty', 20, False), ('three thousand', 3, False),
                                   ('twenty, four', 4, True), ('twenty, four', 20, True), ('four and twenty', 4, True), ('four-day', 4, True), ('a hundred four', 4, False)):
            self.assertEqual(has_number(text, value), found, (text, value))
        self.assertEqual(find_number('only twelve hours', 12).span(), (5, 11))
        self.assertIsNone(find_number('only 1000 hours', 12))

    def test_same_answer(self):
        self.assertTrue(same_answer('12,000', '12000'))
        self.assertFalse(same_answer('3 or 4', '3'))


if __name__ == '__main__':
    unittest.main()
