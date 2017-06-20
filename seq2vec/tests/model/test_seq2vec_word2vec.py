'''Test Sequence to vector using word2vec model'''
from unittest import TestCase
from unittest.mock import patch
from os.path import abspath, dirname, join
import os

import numpy as np

from seq2vec.word2vec import GensimWord2vec
from seq2vec.model import Seq2SeqWord2Vec
from seq2vec.transformer import WordEmbeddingTransformer
from seq2vec.util import DataGenterator

class TestSeq2vecWord2vecClass(TestCase):

    def setUp(self):
        self.dir_path = dirname(abspath(__file__))
        word2vec_path = join(self.dir_path, 'word2vec.model.bin')
        self.word2vec = GensimWord2vec(word2vec_path)
        self.latent_size = 20
        self.encoding_size = 60
        self.max_length = 5
        self.model = Seq2SeqWord2Vec(
            self.word2vec,
            max_length=self.max_length,
            latent_size=self.latent_size,
            encoding_size=self.encoding_size
        )

        self.train_seq = [
            ['我', '有', '一個', '蘋果'],
            ['我', '有', '一支', '筆'],
            ['我', '有', '一個', '鳳梨'],
        ]
        self.test_seq = [
            ['我', '愛', '吃', '鳳梨'],
        ]

    @patch('keras.models.Model.fit')
    def test_fit(self, _):
        self.model.fit(self.train_seq)
        result = self.model.transform(self.test_seq)
        self.assertEqual(result.shape[1], self.encoding_size)

    @patch('keras.models.Model.fit')
    def test_load_save_model(self, _):
        model_path = join(self.dir_path, 'seq2vec_word2vec_model.h5')

        self.model.fit(self.train_seq)
        answer = self.model.transform(self.test_seq)

        self.model.save_model(model_path)
        new_model = Seq2SeqWord2Vec(
            word2vec_model=self.word2vec,
        )
        new_model.load_model(model_path)
        result = new_model.transform(self.test_seq)
        np.testing.assert_array_almost_equal(answer, result)
        self.assertEqual(self.max_length, new_model.max_length)
        self.assertEqual(self.latent_size, new_model.latent_size)
        self.assertEqual(self.encoding_size, new_model.encoding_size)
        os.remove(model_path)

    @patch('keras.models.Model.fit_generator')
    def test_fit_generator(self, _):
        data_path = join(self.dir_path, 'test_corpus.txt')

        x_transformer = WordEmbeddingTransformer(
            word2vec_model=self.word2vec, max_length=5
        )
        y_transformer = WordEmbeddingTransformer(
            word2vec_model=self.word2vec, max_length=5
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
        self.assertEqual(result.shape[1], self.encoding_size)
