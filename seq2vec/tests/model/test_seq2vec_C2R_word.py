'''Test Sequence to vector using word2vec model'''
from unittest import TestCase
import os

import numpy as np

from seq2vec.model import Seq2VecC2RWord

from .test_seq2vec_base import TestSeq2VecBaseClass

class TestSeq2VecC2RWordClass(TestSeq2VecBaseClass, TestCase):

    def setUp(self):
        self.conv_size = 5
        self.channel_size = 10
        super(TestSeq2VecC2RWordClass, self).setUp()
        self.encoding_size = (
            self.word_embedding_size // self.conv_size * self.channel_size
        )

    def create_model(self):
        return Seq2VecC2RWord(
            self.word2vec,
            max_length=self.max_length,
            conv_size=self.conv_size,
            channel_size=self.channel_size,
            latent_size=self.latent_size
        )

    def test_load_save_model(self):
        answer = self.model.transform(self.test_seq)

        self.model.save_model(self.model_path)
        new_model = Seq2VecC2RWord(
            word2vec_model=self.word2vec,
            max_length=self.max_length
        )
        new_model.load_model(self.model_path)
        result = new_model.transform(self.test_seq)
        np.testing.assert_array_almost_equal(answer, result)
        self.assertEqual(self.conv_size, new_model.conv_size)
        self.assertEqual(self.channel_size, new_model.channel_size)
        self.assertEqual(self.word_embedding_size, new_model.word_embedding_size)
        self.assertEqual(self.latent_size, new_model.latent_size)
        self.assertEqual(self.max_length, new_model.max_length)
        self.assertEqual(self.encoding_size, new_model.encoding_size)
        os.remove(self.model_path)
