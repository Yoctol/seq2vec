"""Test Sequence-to-Sequence auto encoder."""
from unittest import TestCase

from ..seq2seq_auto_encoder import Seq2SeqAutoEncoderUseWordHash


class TestSeq2SeqAutoEncoderUseWordHash(TestCase):

    def test_correct_latent_size(self):
        MAX_INDEX = 100
        LATENT_SIZE = 35
        transformer = Seq2SeqAutoEncoderUseWordHash(
            max_index=MAX_INDEX,
            max_length=5,
            latent_size=LATENT_SIZE,
        )
        train_seq = [
            ['我', '有', '一個', '蘋果'],
            ['我', '有', '一支', '筆'],
            ['我', '有', '一個', '鳳梨'],
        ]
        test_seq = [
            ['我', '愛', '吃', '鳳梨'],
        ]
        transformer.fit(train_seq)
        result = transformer.transform(test_seq)
        self.assertEqual(result.shape[1], LATENT_SIZE)
