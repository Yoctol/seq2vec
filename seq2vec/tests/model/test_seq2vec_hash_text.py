"""Test HashSeq2Vec."""
from unittest import TestCase

from seq2vec.model import Seq2VecHash


class TestHashSeq2Vec(TestCase):

    def test_transform_single_sequencereturn_shape(self):
        vector_length = 100
        transformer = Seq2VecHash(vector_length)
        seq = ['我', '有', '一個', '蘋果']
        result = transformer.transform_single_sequence(seq)
        self.assertEqual(result.shape[0], vector_length)

    def test_transform_return_shape(self):
        vector_length = 100
        transformer = Seq2VecHash(vector_length)
        seqs = [
            ['我', '有', '一個', '蘋果'],
            ['我', '有', 'pineapple'],
        ]
        result = transformer.transform(seqs)
        self.assertEqual(result.shape, (len(seqs), vector_length))

    def test_caller_return_shape(self):
        vector_length = 100
        transformer = Seq2VecHash(vector_length)
        seqs = [
            ['我', '有', '一個', '蘋果'],
            ['我', '有', 'pineapple'],
        ]
        result = transformer(seqs)
        self.assertEqual(result.shape, (len(seqs), vector_length))
