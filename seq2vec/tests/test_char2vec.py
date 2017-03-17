from unittest import TestCase
from os.path import abspath, dirname, join

import numpy as np
from seq2vec.word2vec.char2vec import Char2vec
from seq2vec.word2vec.char2vec import Char2vecInputTransformer
from seq2vec.word2vec.char2vec import Char2vecOutputTransformer
from seq2vec.word2vec.dictionary import Dictionary
from seq2vec.word2vec.gensim_word2vec import GensimWord2vec

class TestChar2vecClass(TestCase):

    def setUp(self):
        self.train_seqs = [
            ['我'], ['養'], ['一'], ['隻'], ['小狗']
        ]
        self.test_seqs = [
            ['茶'], ['我']
        ]

        dictionary = {
            '我':0, '養':1, '隻':2, '小':3, '狗':4, '茶':5
        }
        self.dictionary = Dictionary(dictionary=dictionary)

        self.current_dir = dirname(abspath(__file__))
        self.word2vec = GensimWord2vec(
            join(self.current_dir, 'word2vec.model.bin')
        )

        self.model = Char2vec(self.word2vec, 32, 2, self.dictionary)

    def test_save_and_load(self):
        model_path = join(self.current_dir, 'test_char2vec.model')

        self.model.fit(self.train_seqs)
        self.model.save_model(model_path)

        new_model = Char2vec(self.word2vec, 32, 2, self.dictionary)
        new_model.load_model(model_path)

        np.testing.assert_array_almost_equal(
            self.model(self.test_seqs), new_model(self.test_seqs)
        )
