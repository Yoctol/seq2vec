"""Test BaseSeq2Vec."""
from unittest import TestCase
from unittest.mock import Mock

from ..base import BaseSeq2Vec


class MockSeq2vec(BaseSeq2Vec):

        def __init__(self, mock_transform):
            self.mock_transform = mock_transform

        def transform_single_sequence(self, seq):
            self.mock_transform()


class TestBase(TestCase):

    def test_call_n_times(self):
        mock_transform = Mock()
        seq2vec = MockSeq2vec(mock_transform)
        seqs = [
            ['I', 'have', 'an', 'apple'],
            ['我', '有', '一個', '蘋果'],
        ]
        seq2vec(seqs)
        self.assertEqual(mock_transform.call_count, len(seqs))
