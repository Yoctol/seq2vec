"""Test Sequence-to-Sequence auto encoder."""
from unittest import TestCase
from os.path import abspath
from os.path import dirname
from os.path import join

import numpy as np

from ..seq2seq_auto_encoder import Seq2SeqAutoEncoderUseWordHash
from ..seq2seq_auto_encoder import Seq2vecAutoEncoderInputTransformer
from ..seq2seq_auto_encoder import Seq2vecAutoEncoderOutputTransformer
from ..data_generator import DataGenterator

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

    def test_correct_latent_size(self):
        self.transformer.fit(self.train_seq)
        result = self.transformer.transform(self.test_seq)
        self.assertEqual(result.shape[1], self.latent_size)

    def test_load_save_model(self):
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

    def test_fit_generator(self):
        data_path = join(self.dir_path, 'test_corpus.txt')

        x_transformer = Seq2vecAutoEncoderInputTransformer(
            self.max_index, self.max_length
        )
        y_transformer = Seq2vecAutoEncoderOutputTransformer(
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
