'''Test Sequence to vector using word2vec model'''
from unittest import TestCase
import os

import numpy as np

from seq2vec.model import Seq2VecC2RChar
from .test_seq2vec_base import TestSeq2VecBaseClass

class TestSeq2VecC2RCharClass(TestSeq2VecBaseClass, TestCase):

    def setUp(self):
        self.char_embedding_size = 300
        self.max_index = 1000
        self.conv_size = 5
        self.channel_size = 10
        super(TestSeq2VecC2RCharClass, self).setUp()
        self.encoding_size = (
            self.channel_size * self.char_embedding_size // self.conv_size
        )

    def create_model(self):
        return Seq2VecC2RChar(
            self.word2vec,
            max_index=self.max_index,
            max_length=self.max_length,
            char_embedding_size=self.char_embedding_size,
            conv_size=self.conv_size,
            channel_size=self.channel_size,
        )

    def test_load_save_model(self):
        answer = self.model.transform(self.test_seq)

        self.model.save_model(self.model_path)
        new_model = Seq2VecC2RChar(
            self.word2vec
        )
        new_model.load_model(self.model_path)
        result = new_model.transform(self.test_seq)

        np.testing.assert_array_almost_equal(answer, result)
        self.assertEqual(self.conv_size, new_model.conv_size)
        self.assertEqual(self.channel_size, new_model.channel_size)
        self.assertEqual(self.char_embedding_size, new_model.char_embedding_size)
        self.assertEqual(self.max_length, new_model.max_length)
        self.assertEqual(self.encoding_size, new_model.encoding_size)
        self.assertEqual(self.max_index, new_model.max_index)
        os.remove(self.model_path)
