'''hash embedding transformer testcases'''
from unittest import TestCase

import numpy as np

from seq2vec.transformer import HashIndexTransformer
from seq2vec.transformer import OneHotEncodedTransformer
from .test_transformer_base import CharEmbeddingTransformerBase

class TestHashIndexTransformerClass(
        CharEmbeddingTransformerBase,
        TestCase
):

    def setUp(self):
        super(TestHashIndexTransformerClass, self).setUp()
        self.transformer = HashIndexTransformer(
            max_index=self.max_index,
            max_length=self.max_length
        )

    def test_seq_transform_shape(self):
        transformed_seq = self.transformer.seq_transform(self.seqs[0])
        self.assertEqual(
            len(transformed_seq),
            len(self.seqs[0])
        )

    def test_seq_transform_nonzero_hash_index(self):
        transformed_seq = self.transformer.seq_transform(self.seqs[0])
        for index in transformed_seq:
            self.assertTrue(
                bool(index)
            )

    def test_call_shape(self):
        transformed_array = self.transformer(self.seqs)
        self.assertEqual(
            transformed_array.shape,
            (len(self.seqs), self.max_length)
        )

    def test_call_zero_padding(self):
        transformed_array = self.transformer(self.seqs)
        self.assertFalse(
            np.sum(transformed_array[:, -1], dtype=bool)
        )


class TestOneHotEncodedTransformerClass(
        CharEmbeddingTransformerBase,
        TestCase
):

    def setUp(self):
        super(TestOneHotEncodedTransformerClass, self).setUp()
        self.transformer = OneHotEncodedTransformer(
            max_index=self.max_index,
            max_length=self.max_length
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
            np.testing.assert_array_almost_equal(
                np.sum(one_hot_feature),
                1.0
            )

    def test_call_shape(self):
        transformed_array = self.transformer(self.seqs)
        self.assertEqual(
            transformed_array.shape,
            (len(self.seqs), self.max_length, self.max_index)
        )

    def test_call_zero_padding(self):
        transformed_array = self.transformer(self.seqs)
        self.assertFalse(
            np.sum(transformed_array[:, -1, :], dtype=bool)
        )

