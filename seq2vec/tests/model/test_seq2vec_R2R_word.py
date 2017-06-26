'''Test Sequence to vector using word2vec model'''
from unittest import TestCase
import os

import numpy as np

from seq2vec.model import Seq2VecR2RWord

from .test_seq2vec_base import TestSeq2VecBaseClass

class TestSeq2VecR2RWordClass(TestSeq2VecBaseClass, TestCase):

    def setUp(self):
        super(TestSeq2VecR2RWordClass, self).setUp()

    def create_model(self):
        return Seq2VecR2RWord(
            self.word2vec,
            max_length=self.max_length,
            latent_size=self.latent_size,
            encoding_size=self.encoding_size
        )

    def test_load_save_model(self):
        answer = self.model.transform(self.test_seq)
        self.model.save_model(self.model_path)

        new_model = Seq2VecR2RWord(
            word2vec_model=self.word2vec,
        )
        new_model.load_model(self.model_path)

        result = new_model.transform(self.test_seq)
        np.testing.assert_array_almost_equal(answer, result)
        self.assertEqual(self.max_length, new_model.max_length)
        self.assertEqual(self.latent_size, new_model.latent_size)
        self.assertEqual(self.encoding_size, new_model.encoding_size)
        os.remove(self.model_path)
