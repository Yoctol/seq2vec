from unittest import TestCase
from os.path import abspath, dirname, join

from seq2vec.word2vec.dictionary import Dictionary

class TestDictionaryClass(TestCase):

    def setUp(self):
        dictionary = {
            '我': 0, '養': 1, '了': 2, '一': 3,
            '隻': 4, '小': 5, '狗': 6
        }
        self.dictionary = Dictionary(dictionary=dictionary)

        self.current_dir = dirname(abspath(__file__))

    def test_size(self):
        self.assertEqual(self.dictionary.size(), 9)

    def test_getitem(self):
        self.assertEqual(3, self.dictionary['養'])
        self.assertEqual(6, self.dictionary['隻'])
        self.assertEqual(8, self.dictionary['狗'])
        self.assertEqual(1, self.dictionary['雞'])
        self.assertEqual(1, self.dictionary['鴨'])

    def test_save_and_load(self):
        dict_path = join(self.current_dir, 'test.dict')

        self.dictionary.save_dict(dict_path)
        new_dictionary = Dictionary()
        new_dictionary.load_dict(dict_path)

        self.assertEqual(new_dictionary['我'], self.dictionary['我'])
        self.assertEqual(new_dictionary['養'], self.dictionary['養'])
        self.assertEqual(new_dictionary['了'], self.dictionary['了'])
        self.assertEqual(new_dictionary['一'], self.dictionary['一'])
        self.assertEqual(new_dictionary['隻'], self.dictionary['隻'])
        self.assertEqual(new_dictionary['雞'], self.dictionary['雞'])
        self.assertEqual(new_dictionary['鴨'], self.dictionary['鴨'])
