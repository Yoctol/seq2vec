from unittest import TestCase
from os.path import abspath, dirname, join

import numpy as np
from sklearn.preprocessing import normalize
from seq2vec.word2vec.char2vec import Char2vec
from seq2vec.word2vec.char2vec import Char2vecInputTransformer
from seq2vec.word2vec.char2vec import Char2vecOutputTransformer
from seq2vec.word2vec.char2vec import Char2vecSeqTransformer
from seq2vec.word2vec.dictionary import Dictionary
from seq2vec.word2vec.gensim_word2vec import GensimWord2vec

class TestChar2vecClass(TestCase):

    def setUp(self):
        self.train_seqs = [
            ['我', '養', '一', '隻', '小狗']
        ]
        self.test_seqs = [
            ['茶', '我']
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

    def test_get_size(self):
        self.assertEqual(32, self.model.get_size())

    def test_get_item(self):
        word = '我'
        np.testing.assert_array_almost_equal(
            self.model.transform([[word]]).reshape(32), self.model[word]
        )

class Testchar2vecTransformerClass(TestCase):

    def setUp(self):
        dictionary = {
            '我':0, '有':1, '一':2, '顆':3, '蘋':4, '果':5,
            '你':6, '兩':7, '葡':8, '萄':9
        }
        self.dictionary = Dictionary(dictionary=dictionary)

        self.input = Char2vecInputTransformer(
            dictionary=self.dictionary, max_length=2
        )

        self.current_dir = dirname(abspath(__file__))
        self.word2vec = GensimWord2vec(
            join(self.current_dir, 'word2vec.model.bin')
        )
        self.output = Char2vecOutputTransformer(self.word2vec)
        self.seq_transformer = Char2vecSeqTransformer(
            dictionary=self.dictionary, max_length=2
        )
        self.seqs = [
            ['我', '有', '一顆', '蘋果'],
            ['你', '有', '兩顆', '葡萄']
        ]

    def test_seq_seq_transform(self):
        self.assertEqual(
            [6, 7], self.seq_transformer.seq_transform(self.seqs[0][3])
        )
        self.assertEqual(
            [10, 11], self.seq_transformer.seq_transform(self.seqs[1][3])
        )
        self.assertEqual(
            [3], self.seq_transformer.seq_transform(self.seqs[0][1])
        )

    def test_seq_call(self):
        answer = np.zeros((4, 2))
        answer[0, 1] = 2
        answer[1, 1] = 3
        answer[2] = np.array([5, 4])
        answer[3] = np.array([7, 6])
        np.testing.assert_array_almost_equal(
            answer, self.seq_transformer(self.seqs[0])
        )

    def test_input_seq_transform(self):
        answer = self.seq_transformer(self.seqs[0])
        np.testing.assert_array_almost_equal(
            answer, self.input.seq_transform(self.seqs[0])
        )

    def test_input_call(self):
        answer = np.zeros((8, 2))
        for i, seq in enumerate(self.seqs):
            for j, word in enumerate(seq):
                for k, char in enumerate(word[::-1]):
                    if len(word) == 1:
                        answer[i * 4 + j, k + 1] = self.dictionary[char]
                    else:
                        answer[i * 4 + j, k] = self.dictionary[char]
        np.testing.assert_array_almost_equal(answer, self.input(self.seqs))

    def test_output_seq_transform(self):
        answer = []
        feature_size = self.word2vec.get_size()
        for word in self.seqs[0]:
            try:
                word_array = self.word2vec[word]
                normalize(word_array.reshape(1, -1), copy=False)
                answer.append(word_array.reshape(feature_size))
            except KeyError:
                answer.append(np.zeros(self.word2vec.get_size()))
        transformed_list = self.output.seq_transform(self.seqs[0])

        self.assertEqual(len(answer), len(transformed_list))
        for answer_array, transformed_array in zip(
            answer, transformed_list):
            np.testing.assert_array_almost_equal(
                answer_array, transformed_array
            )

    def test_output_call(self):
        answer = []
        feature_size = self.word2vec.get_size()
        for seq in self.seqs:
            for word in seq:
                try:
                    word_array = self.word2vec[word]
                    normalize(word_array.reshape(1, -1), copy=False)
                    answer.append(word_array.reshape(feature_size))
                except KeyError:
                    answer.append(np.zeros(self.word2vec.get_size()))
        np.testing.assert_array_almost_equal(
            np.array(answer), self.output(self.seqs)
        )

