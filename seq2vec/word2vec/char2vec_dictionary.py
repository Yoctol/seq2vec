
from seq2vec.word2vec.dictionary import Dictionary
from seq2vec.word2vec.char2vec import Char2vec
from seq2vec.word2vec.gensim_word2vec import GensimWord2vec

class Char2vecDictionary(object):

    def __init__(self, dict_path, char2vec_path, word2vec_path):

        self.dictionary = Dictionary()
        self.dictionary.load_dict(dict_path=dict_path)

        word2vec = GensimWord2vec(word2vec_path)

        char2vec = Char2vec(word2vec, 32, 4, self.dictionary)
        char2vec.load_model(char2vec_path)

        self.feature_dict = {}
        self.feature_dict[1] = char2vec['?']

        for key in self.dictionary.dictionary.keys():
            self.feature_dict[self.dictionary[key]] = char2vec[key]

        self.latent_size = char2vec.get_size()

    def get_size(self):
        return self.latent_size

    def __getitem__(self, key):
        return self.feature_dict[self.dictionary[key]]

