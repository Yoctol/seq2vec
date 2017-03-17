import numpy as np
import keras.models
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn.preprocessing import normalize

from .base_word2vec import BaseWord2vecClass
from ..base import BaseSeq2Vec
from ..base import TrainableInterfaceMixin
from ..base import BaseTransformer
from ..util import generate_padding_array

def _create_char2vec_model(
        max_index, embedding_size,
        max_length, word2vec_size,
        learning_rate=0.05,
        rho=0.9, decay=0.01
    ):
    model = Sequential()
    model.add(
        Embedding(
            max_index, embedding_size, input_length=max_length,
            name='embedding', mask_zero=True, dropout=0.2
        )
    )
    model.add(
        LSTM(
            output_dim=word2vec_size(), return_sequences=False,
            name='LSTM_1', dropout_W=0.2, dropout_U=0.3
        )
    )

    encoder = Model(
        model.input,
        model.get_layer('embedding').output
    )


    optimizer = RMSprop(
        lr=learning_rate,
        rho=rho,
        decay=decay,
    )
    model.compile(loss='mean_square_error', optimizer=optimizer)
    return model, encoder

class Char2vecInputTransformer(BaseTransformer):

    def __init__(self, dictionary, max_length):
        self.max_length = max_length
        self.dictionary = dictionary

    def seq_transform(self, seq):
        word = ''.join(seq)
        transformed_seq = []
        for char in word:
            transformed_seq.append(self.dictionary[char])
        return transformed_seq

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs, self.seq_transform, 0, self.max_length, inverse=True
        )
        return array

class Char2vecOutputTransformer(BaseTransformer):

    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.zeros = np.zeros(self.word2vec.get_size())

    def seq_transform(self, seq):
        word = ''.join(seq)
        word_array = self.zeros

        try:
            word_array = self.word2vec[word]
            normalize(word_array, copy=False)
        except KeyError:
            pass
        return [word_array]

    def __call__(self, seqs):
        array = generate_padding_array(
            seqs, self.seq_transform, self.zeros, 1, inverse=False
        )
        return array.reshape((len(seqs), self.word2vec.get_size()))

class Char2vec(BaseWord2vecClass, BaseSeq2Vec, TrainableInterfaceMixin):

    def __init__(self, word2vec,
                 embedding_size,
                 max_length,
                 dictionary,
                 learning_rate=0.05):
        self.embedding_size = embedding_size
        self.max_length = max_length

        self.model, self.encoder = _create_char2vec_model(
            max_index=dictionary.size(), embedding_size=embedding_size,
            max_length=max_length, word2vec_size=word2vec.get_size(),
            learning_rate=learning_rate
        )

        self.input_transform = Char2vecInputTransformer(dictionary, max_length)
        self.output_transform = Char2vecOutputTransformer(word2vec)

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)
        self.encoder = Model(
            self.model.input, self.model.get_layer('embedding').output
        )
        self.max_length = self.model.input_shape[1]
        self.embedding_size = self.model.get_layer('embedding').output_dim
