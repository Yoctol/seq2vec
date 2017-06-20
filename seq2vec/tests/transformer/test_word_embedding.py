'''word embedding testcases'''
from unittest import TestCase

import numpy as np
from seq2vec.transformer import WordEmbeddingTransformer
from .test_transformer_base import WordEmbeddingTransformerBase

class TestWordEmbeddingTransformerClass(
    WordEmbeddingTransformerBase,
    TestCase
):

    def setUp(self):
        super(TestWordEmbeddingTransformerClass, self).setUp()
        self.transformer = WordEmbeddingTransformer(
            word2vec_model=self.word2vec,
            max_length=self.max_length
        )

    def test_seq_transform_shape(self):
        transformed_seq = self.transformer.seq_transform(self.seqs[0])
        self.assertEqual(
            len(transformed_seq[0]),
            self.word2vec.get_size()
        )

    def test_call_zero_padding(self):
        transformed_seqs = self.transformer(self.seqs)
        np.testing.assert_array_almost_equal(
            np.zeros((
                len(self.seqs),
                self.word2vec.get_size()
            )),
            transformed_seqs[:, -1, :]
        )
        self.assertTrue(
            np.sum(
                transformed_seqs[:, :-1, :],
                dtype=bool
            )
        )

    def test_call_shape(self):
        self.assertEqual(
            self.transformer(self.seqs).shape,
            (len(self.seqs), self.max_length, self.word2vec.get_size())
        )
