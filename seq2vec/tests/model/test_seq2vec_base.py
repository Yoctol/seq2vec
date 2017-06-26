'''Test base class'''
from unittest.mock import patch
from abc import abstractmethod
from os.path import abspath, dirname, join

from seq2vec.word2vec import GensimWord2vec
from seq2vec.util import DataGenterator

class TestSeq2VecBaseClass(object):

    def setUp(self):
        self.latent_size = 80
        self.encoding_size = 100
        self.max_length = 5

        self.dir_path = dirname(abspath(__file__))
        self.model_path = join(self.dir_path, 'seq2vec_model.h5')
        self.data_path = join(self.dir_path, '../test_corpus.txt')
        word2vec_path = join(self.dir_path, '../word2vec.model.bin')
        self.word2vec = GensimWord2vec(word2vec_path)
        self.word_embedding_size = self.word2vec.get_size()

        self.model = self.create_model()
        self.input_transformer = self.model.input_transformer
        self.output_transformer = self.model.output_transformer

        self.data_generator = DataGenterator(
            self.data_path,
            self.input_transformer,
            self.output_transformer,
            batch_size=10
        )

        self.train_seq = [
            ['我', '有', '一個', '蘋果'],
            ['我', '有', '一支', '筆'],
            ['我', '有', '一個', '鳳梨'],
        ]
        self.test_seq = [
            ['我', '愛', '吃', '鳳梨'],
        ]

    @abstractmethod
    def create_model(self):
        pass

    def test_fit(self):
        self.model.fit(self.train_seq)
        result = self.model.transform(self.test_seq)
        self.assertEqual(result.shape[1], self.encoding_size)

    def test_fit_generator(self):
        self.model.fit_generator(
            self.data_generator,
            self.data_generator,
            batch_number=2
        )
        result = self.model(self.train_seq)
        self.assertEqual(result.shape[1], self.encoding_size)
