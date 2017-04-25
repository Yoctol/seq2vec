from gensim.models.keyedvectors import KeyedVectors

from .base_word2vec import BaseWord2vecClass

class GensimWord2vec(BaseWord2vecClass):

    def __init__(self, model_path):
        self.word2vec = KeyedVectors.load_word2vec_format(
            model_path, binary=True
        )

    def __getitem__(self, key):
        return self.word2vec[key]

    def get_size(self):
        return self.word2vec.syn0.shape[1]

    def get_index(self, key):
        return self.word2vec.vocab[key].index

    def get_vocab_size(self):
        return len(self.word2vec.vocab)
