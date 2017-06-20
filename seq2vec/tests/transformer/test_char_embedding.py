'''char embedding transformer testcases'''
from unittest import TestCase

import numpy as np

from seq2vec.transformer import CharEmbeddingOneHotTransformer
from .test_transformer_base import CharEmbeddingTransformerBase

class TestCharEmbeddingOneHotTransformerClass(
        CharEmbeddingTransformerBase,
        TestCase
):
    def setUp(self):
        super(TestCharEmbeddingOneHotTransformerClass, self).setUp()
        self.max_length = 10
        self.transformer = CharEmbeddingOneHotTransformer(
            self.max_index,
            self.max_length
        )

    def test_seq_transform_shape(self):
        transformed_seq = self.transformer.seq_transform(self.seqs[0])
        self.assertEqual(
            len(transformed_seq[0]),
            self.max_index
        )

    def test_seq_transform_one_hot_value(self):
        transformed_seq = self.transformer.seq_transform(self.seqs[0])
        for one_hot_feature in transformed_seq:
            self.assertEqual(
                sum(one_hot_feature),
                1
            )

    def test_call_shape(self):
        transformed_array = self.transformer(self.seqs)
        self.assertEqual(
            transformed_array.shape,
            (len(self.seqs), self.max_length, self.max_index)
        )

    def test_call_zero_padding(self):
        char_length = len(''.join(self.seqs[0]))
        transformed_array = self.transformer(self.seqs)
        np.testing.assert_array_almost_equal(
            transformed_array[0, char_length:, :],
            np.zeros((
                self.max_length - char_length,
                self.max_index
            ))
        )

    def test_call_non_zero_feature_value(self):
        char_length = len(''.join(self.seqs[0]))
        transformed_array = self.transformer(self.seqs)
        self.assertTrue(
            np.sum(
                transformed_array[0, :char_length, :],
                dtype=bool
            )
        )
