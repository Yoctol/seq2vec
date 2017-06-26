"""Test Sequence-to-Sequence auto encoder."""
from unittest import TestCase
import os

import numpy as np
from seq2vec.model import Seq2VecR2RHash
from .test_seq2vec_base import TestSeq2VecBaseClass

class TestSeq2VecR2RHashClass(TestSeq2VecBaseClass, TestCase):

    def setUp(self):
        self.max_index = 10
        super(TestSeq2VecR2RHashClass, self).setUp()

    def create_model(self):
        return Seq2VecR2RHash(
            max_index=self.max_index,
            max_length=self.max_length,
            latent_size=self.latent_size,
            encoding_size=self.encoding_size,
            word_embedding_size=self.word_embedding_size
        )

    def test_load_save_model(self):
        answer = self.model(self.test_seq)

        self.model.save_model(self.model_path)
        new_model = Seq2VecR2RHash()
        new_model.load_model(self.model_path)
        self.assertEqual(self.max_index, new_model.max_index)
        self.assertEqual(self.max_length, new_model.max_length)
        self.assertEqual(self.latent_size, new_model.latent_size)
        self.assertEqual(self.encoding_size, new_model.encoding_size)
        self.assertEqual(
            self.word_embedding_size,
            new_model.word_embedding_size
        )
        result = new_model(self.test_seq)
        np.testing.assert_array_almost_equal(answer, result)
        os.remove(self.model_path)
