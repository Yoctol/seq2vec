"""Test Sequence-to-Sequence auto encoder."""
from unittest import TestCase
from unittest.mock import patch
from os.path import abspath
from os.path import dirname
from os.path import join
import os

import numpy as np
from yoctol_utils.hash import consistent_hash

from seq2vec.model import Seq2SeqAutoEncoderUseWordHash
from seq2vec.transformer import HashIndexTransformer
from seq2vec.transformer import OneHotEncodedTransformer
from seq2vec.util import DataGenterator

class TestSeq2SeqAutoEncoderUseWordHash(TestCase):

    def setUp(self):
        self.max_index = 10
        self.max_length = 5
        self.latent_size = 20
        self.transformer = Seq2SeqAutoEncoderUseWordHash(
            max_index=self.max_index,
            max_length=self.max_length,
            latent_size=self.latent_size,
        )
        self.train_seq = [
            ['我', '有', '一個', '蘋果'],
            ['我', '有', '一支', '筆'],
            ['我', '有', '一個', '鳳梨'],
        ]
        self.test_seq = [
            ['我', '愛', '吃', '鳳梨'],
        ]
        current_path = abspath(__file__)
        self.dir_path = dirname(current_path)

    @patch('keras.models.Model.fit')
    def test_correct_latent_size(self, _):
        self.transformer.fit(self.train_seq)
        result = self.transformer.transform(self.test_seq)
        self.assertEqual(result.shape[1], self.latent_size)

    @patch('keras.models.Model.fit')
    def test_load_save_model(self, _):
        model_path = join(self.dir_path, 'auto_encoder_model.h5')

        self.transformer.fit(self.train_seq)
        answer = self.transformer.transform(self.test_seq)

        self.transformer.save_model(model_path)
        new_model = Seq2SeqAutoEncoderUseWordHash(
            max_index=self.max_index,
            max_length=self.max_length
        )
        new_model.load_model(model_path)
        result = new_model.transform(self.test_seq)
        np.testing.assert_array_almost_equal(answer, result)
        os.remove(model_path)

    @patch('keras.models.Model.fit_generator')
    def test_fit_generator(self, _):
        data_path = join(self.dir_path, 'test_corpus.txt')

        x_transformer = HashIndexTransformer(
            self.max_index, self.max_length
        )
        y_transformer = OneHotEncodedTransformer(
            self.max_index, self.max_length
        )

        train_data_generator = DataGenterator(
            data_path, x_transformer, y_transformer, batch_size=10
        )
        test_data_generator = DataGenterator(
            data_path, x_transformer, y_transformer, batch_size=10
        )

        self.transformer.fit_generator(
            train_data_generator, test_data_generator, batch_number=2
        )

        result = self.transformer(self.train_seq)
        self.assertEqual(result.shape[1], self.latent_size)

class TestSeq2SeqAutoEncoderInputTransformerClass(TestCase):

    def setUp(self):
        self.input = HashIndexTransformer(
            max_index=10, max_length=5
        )
        self.seqs = [
            ['我', '有', '一顆', '蘋果'],
            ['你', '有', '兩顆', '葡萄']
        ]

    def test_seq_transform(self):
        answer = []
        for word in self.seqs[0]:
            answer.append(consistent_hash(word) % 9 + 1)
        self.assertEqual(answer, self.input.seq_transform(self.seqs[0]))

    def test_call(self):
        answer = np.zeros((2, 5))
        for i, seq in enumerate(self.seqs):
            for j, word in enumerate(seq):
                answer[i, j] = consistent_hash(word) % 9 + 1
        np.testing.assert_array_almost_equal(answer, self.input(self.seqs))

class TestSeq2SeqAutoEncoderOutputTransformerClass(TestCase):

    def setUp(self):
        self.output = OneHotEncodedTransformer(
            max_index=10, max_length=5
        )
        self.seqs = [
            ['我', '有', '一顆', '蘋果'],
            ['你', '有', '兩顆', '葡萄']
        ]

    def test_seq_transform(self):
        answer = []
        for word in self.seqs[0]:
            index = consistent_hash(word) % 9 + 1
            zero = np.zeros(10)
            zero[index] = 1
            answer.append(zero)

        transformed_array_list = self.output.seq_transform(self.seqs[0])
        for answer_array, transformed_array in zip(
            answer, transformed_array_list
        ):
            np.testing.assert_array_almost_equal(
                answer_array, transformed_array
            )

    def test_call(self):
        answer = np.zeros((2, 5, 10))
        for i, seq in enumerate(self.seqs):
            for j, word in enumerate(seq):
                zero = np.zeros(10)
                index = consistent_hash(word) % 9 + 1
                zero[index] = 1
                answer[i, j] = zero
        np.testing.assert_array_almost_equal(answer, self.output(self.seqs))
