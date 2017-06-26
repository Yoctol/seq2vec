'''Word embedding Conv3D format transformer testcases'''
from unittest import TestCase

import numpy as np

from seq2vec.transformer import WordEmbeddingConv3DTransformer
from .test_transformer_base import WordEmbeddingTransformerBase

class TestWordEmbeddingConv3DTransformerClass(WordEmbeddingTransformerBase, TestCase):

    def setUp(self):
        super(TestWordEmbeddingConv3DTransformerClass, self).setUp()
        self.transformer = WordEmbeddingConv3DTransformer(
            word2vec_model=self.word2vec,
            max_length=self.max_length,
        )

    def test_seq_transform_shape(self):
        transformed_seq = self.transformer.seq_transform(self.seqs[0])
        self.assertEqual(
            transformed_seq.shape,
            (
                self.max_length,
                self.max_length,
                self.word2vec.get_size(),
                1
            )
        )

    def test_seq_transform_zero_padding(self):
        transformed_seq = self.transformer.seq_transform(self.seqs[0])
        self.assertFalse(
            np.sum(
                transformed_seq[-1, :, :, 0],
                dtype=bool
            )
        )
        self.assertFalse(
            np.sum(
                transformed_seq[:, -1, :, 0],
                dtype=bool
            )
        )

    def test_seq_transform_nonzero_feature_value(self):
        transformed_seq = self.transformer.seq_transform(self.seqs[0])
        self.assertTrue(
            np.sum(
                transformed_seq[:-1, :-1, :, 0],
                dtype=bool
            )
        )

    def test_seq_transform_symmetric_result(self):
        transformed_seq = self.transformer.seq_transform(self.seqs[0])
        for i in range(self.max_length):
            for j in range(i + 1, self.max_length):
                np.testing.assert_array_almost_equal(
                    transformed_seq[i, j, :, :],
                    transformed_seq[j, i, :, :]
                )

    def test_call_shape(self):
        transformed_array = self.transformer(self.seqs)
        self.assertEqual(
            transformed_array.shape,
            (
                len(self.seqs),
                self.max_length,
                self.max_length,
                self.word2vec.get_size(),
                1
            )
        )
