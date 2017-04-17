'''Test Sequence to vector using word2vec model'''
from unittest import TestCase
from os.path import abspath, dirname, join

import numpy as np
from sklearn.preprocessing import normalize

from seq2vec.word2vec.gensim_word2vec import GensimWord2vec
from seq2vec.seq2seq_cnn3D import Seq2SeqCNN
from seq2vec.seq2seq_cnn3D import Seq2vecCNN3DTransformer
from seq2vec.seq2seq_word2vec import Seq2vecWord2vecSeqTransformer
from seq2vec.data_generator import DataGenterator

class TestSeq2vecWord2vecClass(TestCase):

    def setUp(self):
        self.dir_path = dirname(abspath(__file__))
        word2vec_path = join(self.dir_path, 'word2vec.model.bin')
        self.word2vec = GensimWord2vec(word2vec_path)
        self.embedding_size = self.word2vec.get_size()
        self.conv_size = 5
        self.channel_size = 10
        self.latent_size = 100
        self.max_length = 5
        self.encoding_size = (
            self.embedding_size // self.conv_size * self.channel_size
        )
        self.model = Seq2SeqCNN(
            self.word2vec, max_length=self.max_length,
            conv_size=self.conv_size,
            channel_size=self.channel_size,
            latent_size=self.latent_size
        )

        self.train_seq = [
            ['我', '有', '一個', '蘋果'],
            ['我', '有', '一支', '筆'],
            ['我', '有', '一個', '鳳梨'],
        ]
        self.test_seq = [
            ['我', '愛', '吃', '鳳梨'],
        ]

    def test_fit(self):
        self.model.fit(self.train_seq)
        result = self.model.transform(self.test_seq)
        self.assertEqual(result.shape[1], self.encoding_size)

    def test_load_save_model(self):
        model_path = join(self.dir_path, 'seq2vec_word2vec_model.h5')

        self.model.fit(self.train_seq)
        answer = self.model.transform(self.test_seq)

        self.model.save_model(model_path)
        new_model = Seq2SeqCNN(
            word2vec_model=self.word2vec,
            max_length=self.max_length
        )
        new_model.load_model(model_path)
        result = new_model.transform(self.test_seq)
        np.testing.assert_array_almost_equal(answer, result)
        self.assertEqual(self.conv_size, new_model.conv_size)
        self.assertEqual(self.channel_size, new_model.channel_size)
        self.assertEqual(self.embedding_size, new_model.embedding_size)
        self.assertEqual(self.latent_size, new_model.latent_size)
        self.assertEqual(self.max_length, new_model.max_length)
        self.assertEqual(self.encoding_size, new_model.encoding_size)

    def test_fit_generator(self):
        data_path = join(self.dir_path, 'test_corpus.txt')

        x_transformer = Seq2vecCNN3DTransformer(
            word2vec_model=self.word2vec, max_length=self.max_length,
            conv_size=self.conv_size, channel_size=self.channel_size
        )
        y_transformer = Seq2vecWord2vecSeqTransformer(
            word2vec_model=self.word2vec,
            max_length=self.max_length, inverse=False
        )

        train_data_generator = DataGenterator(
            data_path, x_transformer, y_transformer, batch_size=10
        )
        test_data_generator = DataGenterator(
            data_path, x_transformer, y_transformer, batch_size=10
        )

        self.model.fit_generator(
            train_data_generator, test_data_generator, batch_number=2
        )

        result = self.model(self.train_seq)
        self.assertEqual(result.shape[1], self.encoding_size)

class TestSeq2SeqCNNTransformerClass(TestCase):

    def setUp(self):
        self.dir_path = dirname(abspath(__file__))
        word2vec_path = join(self.dir_path, 'word2vec.model.bin')
        self.word2vec = GensimWord2vec(word2vec_path)
        self.transformer = Seq2vecCNN3DTransformer(
            word2vec_model=self.word2vec, max_length=5,
            conv_size=5, channel_size=1
        )
        self.seqs = [
            ['我', '有', '一顆', '蘋果'],
            ['你', '有', '兩顆', '葡萄']
        ]

    def test_seq_transform(self):
        pass

    def test_call(self):
        pass
