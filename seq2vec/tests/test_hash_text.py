"""Test HashSeq2Vec."""
from unittest import TestCase

from ..hash_text import HashSeq2Vec


class TestHashSeq2Vec(TestCase):

    def test_transform_return_shape(self):
        vecctor_length = 100
        transformer = HashSeq2Vec(vecctor_length)
        seq = ['我', '有', '一個', '蘋果']
        result = transformer.transform(seq)
        self.assertEqual(result.shape[0], vecctor_length)
