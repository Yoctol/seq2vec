'''Test Sequence to vector using word2vec model'''
from unittest import TestCase
from os.path import abspath, dirname, join

import numpy as np
from sklearn.preprocessing import normalize

from seq2vec.word2vec.gensim_word2vec import GensimWord2vec
from seq2vec.seq2seq_word2vec import Seq2SeqWord2Vec
from seq2vec.seq2seq_word2vec import Seq2vecWord2vecSeqTransformer
from seq2vec.data_generator import DataGenterator

class TestSeq2vecWord2vecClass(TestCase):

    def setUp(self):
        self.dir_path = dirname(abspath(__file__))
        word2vec_path = join(self.dir_path, 'word2vec.model.bin')
        self.word2vec = GensimWord2vec(word2vec_path)
        self.latent_size = 20
        self.encoding_size = self.latent_size * 2
        self.max_length = 5
        self.model = Seq2SeqWord2Vec(
            self.word2vec, max_length=self.max_length,
            latent_size=self.latent_size
        )

        self.train_seq = [
            ['我', '有', '一個', '蘋果'],
            ['我', '有', '一支', '筆'],
            ['我', '有', '一個', '鳳梨'],
        ]
        self.test_seq = [
            ['我', '愛', '吃', '鳳梨'],
        ]

    def test_fit(self):
        self.model.fit(self.train_seq)
        result = self.model.transform(self.test_seq)
        self.assertEqual(result.shape[1], self.latent_size * 2)

    def test_load_save_model(self):
        model_path = join(self.dir_path, 'seq2vec_word2vec_model.h5')

        self.model.fit(self.train_seq)
        answer = self.model.transform(self.test_seq)

        self.model.save_model(model_path)
        new_model = Seq2SeqWord2Vec(
            word2vec_model=self.word2vec,
            max_length=5
        )
        new_model.load_model(model_path)
        result = new_model.transform(self.test_seq)
        np.testing.assert_array_almost_equal(answer, result)
        self.assertEqual(self.max_length, new_model.max_length)
        self.assertEqual(self.latent_size, new_model.latent_size)
        self.assertEqual(self.encoding_size, new_model.encoding_size)

    def test_fit_generator(self):
        data_path = join(self.dir_path, 'test_corpus.txt')

        x_transformer = Seq2vecWord2vecSeqTransformer(
            word2vec_model=self.word2vec, max_length=5, inverse=True
        )
        y_transformer = Seq2vecWord2vecSeqTransformer(
            word2vec_model=self.word2vec, max_length=5, inverse=False
        )

        train_data_generator = DataGenterator(
            data_path, x_transformer, y_transformer, batch_size=10
        )
        test_data_generator = DataGenterator(
            data_path, x_transformer, y_transformer, batch_size=10
        )

        self.model.fit_generator(
            train_data_generator, test_data_generator, batch_number=2
        )

        result = self.model(self.train_seq)
        self.assertEqual(result.shape[1], self.latent_size * 2)

class TestSeq2SeqWord2vecTransformerClass(TestCase):

    def setUp(self):
        self.dir_path = dirname(abspath(__file__))
        word2vec_path = join(self.dir_path, 'word2vec.model.bin')
        self.word2vec = GensimWord2vec(word2vec_path)
        self.input = Seq2vecWord2vecSeqTransformer(
            word2vec_model=self.word2vec, max_length=5, inverse=False
        )
        self.seqs = [
            ['我', '有', '一顆', '蘋果'],
            ['你', '有', '兩顆', '葡萄']
        ]

    def test_seq_transform(self):
        answer = []
        for word in self.seqs[0]:
            try:
                word_array = self.word2vec[word]
                normalize(word_array.reshape(1, -1), copy=False)
                answer.append(word_array.reshape(self.word2vec.get_size()))
            except KeyError:
                pass
        np.testing.assert_array_almost_equal(
            answer, self.input.seq_transform(self.seqs[0])
        )

    def test_call(self):
        answer = np.zeros((2, 5, self.word2vec.get_size()))
        for i, seq in enumerate(self.seqs):
            for j, word in enumerate(seq):
                try:
                    word_array = self.word2vec[word]
                    normalize(word_array.reshape(1, -1), copy=False)
                    answer[i, j] = word_array.reshape(self.word2vec.get_size())
                except KeyError:
                    pass
        np.testing.assert_array_almost_equal(answer, self.input(self.seqs))
