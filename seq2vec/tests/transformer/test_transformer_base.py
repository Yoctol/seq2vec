'''Transformer base class'''
from os.path import join, dirname, abspath

from seq2vec.word2vec import GensimWord2vec

class TransformerBase(object):

    def setUp(self):
        self.max_length = 5
        self.seqs = [
            ['我', '有', '一顆', '蘋果'],
            ['你', '有', '兩顆', '葡萄']
        ]


class CharEmbeddingTransformerBase(TransformerBase):

    def setUp(self):
        super(CharEmbeddingTransformerBase, self).setUp()
        self.max_index = 20


class WordEmbeddingTransformerBase(TransformerBase):

    def setUp(self):
        super(WordEmbeddingTransformerBase, self).setUp()
        self.dir_path = dirname(abspath(__file__))
        word2vec_path = join(self.dir_path, '../word2vec.model.bin')
        self.word2vec = GensimWord2vec(word2vec_path)

