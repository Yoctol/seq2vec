'''Test base class'''
from unittest import TestCase
from abc import abstractmethod
from os.path import abspath, dirname, join

import numpy as np

from seq2vec.data_generator import DataGenterator

class TestSeq2vecBaseClass(object):

    def setUp(
            self,
            latent_size=100,
            encoding_size=100,
            max_length=5,
            model_path='seq2vec_model.h5'
    ):
        self.latent_size = latent_size
        self.encoding_size = encoding_size
        self.max_length = max_length

        self.dir_path = dirname(abspath(__file__))
        self.model_path = join(self.dir_path, model_path)
        self.data_path = join(self.dir_path, 'test_corpus.txt')

        self.model = self.initialize_model()
        self.input_transformer = self.initialize_input_transformer()
        self.output_transformer = self.initialize_output_transformer()

        self.data_generator = DataGenterator(
            self.data_path,
            self.input_transformer,
            self.output_transformer,
            batch_size=10
        )

        self.train_seq = [
            ['我', '有', '一個', '蘋果'],
            ['我', '有', '一支', '筆'],
            ['我', '有', '一個', '鳳梨'],
        ]
        self.test_seq = [
            ['我', '愛', '吃', '鳳梨'],
        ]

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def initialize_input_transformer(self):
        pass

    @abstractmethod
    def initialize_output_transformer(self):
        pass

    def test_fit(self):
        self.model.fit(self.train_seq)
        result = self.model.transform(self.test_seq)
        self.assertEqual(result.shape[1], self.encoding_size)

    def test_load_save_model(self):
        self.model.fit(self.train_seq)
        answer = self.model.transform(self.test_seq)

        self.model.save_model(self.model_path)
        new_model = self.initialize_model()
        new_model.load_model(self.model_path)

        result = new_model.transform(self.test_seq)
        np.testing.assert_array_almost_equal(answer, result)

        self.assertEqual(self.max_length, new_model.max_length)
        self.assertEqual(self.latent_size, new_model.latent_size)
        self.assertEqual(self.encoding_size, new_model.encoding_size)

    def test_fit_generator(self):
        self.model.fit_generator(
            self.data_generator,
            self.data_generator,
            batch_number=2
        )

        result = self.model(self.train_seq)
        self.assertEqual(result.shape[1], self.encoding_size)

class TestSeq2vecTransformerBaseClass(object):

    def setUp(self):
        self.dir_path = dirname(abspath(__file__))
        self.transformer = self.initialize_transformer()

        self.seqs = [
            ['我', '有', '一顆', '蘋果'],
            ['你', '有', '兩顆', '葡萄']
        ]

    @abstractmethod
    def initialize_transformer(self):
        pass

    @abstractmethod
    def test_seq_transform(self):
        pass

    @abstractmethod
    def test_call(self):
        pass
