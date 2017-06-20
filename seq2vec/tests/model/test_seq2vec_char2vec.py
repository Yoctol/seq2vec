'''Test Sequence to vector using word2vec model'''
from unittest import TestCase
from unittest.mock import patch
from os.path import abspath, dirname, join
import os

import numpy as np
from seq2vec.word2vec.gensim_word2vec import GensimWord2vec

from seq2vec.model import Seq2SeqChar2vec
from seq2vec.transformer import CharEmbeddingOneHotTransformer
from seq2vec.transformer import WordEmbeddingTransformer

from .test_seq2vec_base import TestSeq2vecBaseClass
from .test_seq2vec_base import TestSeq2vecTransformerBaseClass

class TestSeq2vecChar2vecClass(TestSeq2vecBaseClass, TestCase):

    def setUp(self):
        dir_path = dirname(abspath(__file__))
        word2vec_path = join(dir_path, 'word2vec.model.bin')
        self.word2vec = GensimWord2vec(word2vec_path)

        self.embedding_size = 300
        self.max_index = 1000
        self.conv_size = 5
        self.channel_size = 10
        self.latent_size = 100
        self.encoding_size = (
            self.channel_size * self.embedding_size // self.conv_size
        )

        super(TestSeq2vecChar2vecClass, self).setUp(
            self.latent_size,
            self.encoding_size,
        )

    def initialize_model(self):
        return Seq2SeqChar2vec(
            self.word2vec,
            max_index=self.max_index,
            max_length=self.max_length,
            embedding_size=self.embedding_size,
            conv_size=self.conv_size,
            channel_size=self.channel_size,
        )

    def initialize_input_transformer(self):
        return CharEmbeddingOneHotTransformer(
            max_index=self.max_index,
            max_length=self.max_length,
        )

    def initialize_output_transformer(self):
        return WordEmbeddingTransformer(
            self.word2vec,
            max_length=self.max_length,
        )

    @patch('keras.models.Model.fit')
    def test_load_save_model(self, _):
        self.model.fit(self.train_seq)
        answer = self.model.transform(self.test_seq)

        self.model.save_model(self.model_path)
        new_model = Seq2SeqChar2vec(
            self.word2vec
        )
        new_model.load_model(self.model_path)
        result = new_model.transform(self.test_seq)

        np.testing.assert_array_almost_equal(answer, result)
        self.assertEqual(self.conv_size, new_model.conv_size)
        self.assertEqual(self.channel_size, new_model.channel_size)
        self.assertEqual(self.embedding_size, new_model.embedding_size)
        self.assertEqual(self.max_length, new_model.max_length)
        self.assertEqual(self.encoding_size, new_model.encoding_size)
        self.assertEqual(self.max_index, new_model.max_index)
        os.remove(self.model_path)

    def test_input_transformer(self):
        transformed_input = self.input_transformer(self.train_seq)
        self.assertEqual(
            transformed_input.shape,
            (len(self.train_seq), self.max_length, self.max_index)
        )

    def test_output_transformer(self):
        transformed_output = self.output_transformer(self.train_seq)
        self.assertEqual(
            transformed_output.shape,
            (len(self.train_seq), self.max_length, self.word2vec.get_size())
        )
